r"""
Gas linepack extension.

Pipelines act as distributed storage: :math:`\text{linepack\_kg} = V_{pipe} \cdot \rho_{avg}` with
:math:`\rho = p \cdot M / (R \cdot T)`. Between timesteps,
:math:`\text{net\_pack\_kgs}(t) \cdot \Delta t = \text{linepack\_kg}(t) - \text{linepack\_kg}(t-1)` - positive = charging.
Each endpoint junction sees :math:`+0.5 \cdot \text{net\_pack\_kgs}` (outflow-positive convention).

Single-step solves pin ``net_pack_kgs = 0``; timeseries activates the temporal coupling.
"""

from __future__ import annotations

import math

from monee.model.branch import GasPipe
from monee.model.core import Var, set_initial_value
from monee.model.grid import GasGrid

from .core import NetworkAspect


class GasLinepack(NetworkAspect):
    """Linepack extension. Injects ``linepack_kg`` and ``net_pack_kgs`` on every
    GasPipe; auto-sizes initial/max from grid params. Per-pipe overrides via the
    ``overrides`` ``{branch_id: {"linepack_kg_initial": ..., "linepack_kg_max": ...}}``."""

    def __init__(self, overrides: dict[int, dict] | None = None) -> None:
        self._overrides: dict[int, dict] = overrides or {}
        self._pipe_volume: dict[int, float] = {}
        self._initial_lp: dict[int, float] = {}
        self._active_branches: set[int] = set()
        self._timeseries_active: bool = False

    @staticmethod
    def _density(grid: GasGrid, pressure_pa: float) -> float:
        """Return gas density [kg/m^3] via the ideal-gas EoS."""
        return pressure_pa * grid.molar_mass / (grid.universal_gas_constant * grid.t_k)

    @staticmethod
    def _nominal_pressure(grid: GasGrid) -> float:
        """Nominal ABSOLUTE operating pressure [Pa] (gauge + ambient)."""
        return grid.pressure_ref_pa * grid.nominal_pressure_pu + getattr(
            grid, "pressure_ambient_pa", 0.0
        )

    @staticmethod
    def _max_pressure(grid: GasGrid) -> float:
        """Maximum ABSOLUTE pressure [Pa] from the ``pressure_squared_pu_max`` bound."""
        return grid.pressure_ref_pa * math.sqrt(grid.pressure_squared_pu_max) + getattr(
            grid, "pressure_ambient_pa", 0.0
        )

    @staticmethod
    def _endpoint_ignored(branch, ignored_nodes: set) -> bool:
        """True if either endpoint node was dropped from the solve. Branch ids are
        3-tuples, so they never match the scalar node ids in ``ignored_nodes`` -
        check the endpoints instead (mirrors islanding/core.py)."""
        return (
            branch.from_node_id in ignored_nodes or branch.to_node_id in ignored_nodes
        )

    def prepare(self, network) -> None:
        self._pipe_volume = {}
        self._initial_lp = {}
        self._active_branches = set()
        self._timeseries_active = False

        for branch in network.branches:
            if not isinstance(branch.model, GasPipe):
                continue
            if not isinstance(branch.grid, GasGrid):
                continue

            bm = branch.model
            grid: GasGrid = branch.grid
            v_pipe = math.pi / 4 * bm.diameter_m**2 * bm.length_m

            rho_nominal = self._density(grid, self._nominal_pressure(grid))
            rho_max = self._density(grid, self._max_pressure(grid))
            auto_initial = v_pipe * rho_nominal
            auto_max = v_pipe * rho_max

            overrides = self._overrides.get(branch.id, {})
            lp_initial = overrides.get("linepack_kg_initial", auto_initial)
            lp_max = overrides.get("linepack_kg_max", auto_max)
            lp_max = max(lp_max, lp_initial * 1.05)
            bm.linepack_kg = Var(lp_initial, min=0, max=lp_max, name="linepack_kg")
            bm.net_pack_kgs = Var(
                0,
                min=-grid.max_mass_flow_kgs,
                max=grid.max_mass_flow_kgs,
                name="net_pack_kgs",
            )

            self._pipe_volume[branch.id] = v_pipe
            self._initial_lp[branch.id] = lp_initial
            self._active_branches.add(branch.id)

    def activate_timeseries(self, network, ignored_nodes: set, step_state=None) -> None:
        """Mark timeseries so equations() stops pinning net_pack_kgs=0; warm-start
        linepack_kg from step_state when available."""
        self._timeseries_active = True

        if step_state is None:
            return
        for branch in network.branches:
            if branch.id not in self._active_branches:
                continue
            if branch.ignored or self._endpoint_ignored(branch, ignored_nodes):
                continue
            prev_lp = step_state.get(branch.id, "linepack_kg")
            if prev_lp is not None:
                set_initial_value(branch.model.linepack_kg, prev_lp)

    def equations(self, network, ignored_nodes: set) -> list:
        r""":math:`\text{linepack\_kg} = V_{pipe} \cdot \text{gas\_density\_kg\_per\_m3}`; pin ``net_pack_kgs = 0`` in
        steady-state mode."""
        eqs = []
        for branch in network.branches:
            if branch.id not in self._active_branches:
                continue
            if branch.ignored or self._endpoint_ignored(branch, ignored_nodes):
                continue
            bm = branch.model
            eqs.append(
                bm.linepack_kg
                == self._pipe_volume[branch.id] * bm.gas_density_kg_per_m3
            )

            if not self._timeseries_active:
                eqs.append(bm.net_pack_kgs == 0)

        return eqs

    def inter_temporal_equations(
        self, network, ignored_nodes: set, temporal_state
    ) -> list:
        r""":math:`\text{net\_pack\_kgs}(t) \cdot \Delta t = \text{linepack\_kg}(t) - \text{linepack\_kg}(t-1)`."""
        eqs = []
        dt_s = temporal_state.dt_h * 3600.0
        for branch in network.branches:
            if branch.id not in self._active_branches:
                continue
            if branch.ignored or self._endpoint_ignored(branch, ignored_nodes):
                continue
            bm = branch.model
            prev_lp = temporal_state.get(branch.id, "linepack_kg")
            if prev_lp is None:
                prev_lp = self._initial_lp[branch.id]
            eqs.append(bm.net_pack_kgs * dt_s == bm.linepack_kg - prev_lp)
        return eqs

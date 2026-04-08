"""
Gas linepack extension for gas pipes.

Linepack models the inertia of the gas network: pipelines act as distributed
storage because the compressible gas column in each pipe can absorb or release
gas mass as pressure changes.  This enables limited short-term buffering of
supply/demand imbalances — the most important dynamic effect in quasi-static
gas-network timeseries simulations.

Physics
-------
The stored gas mass in a pipe at any instant is:

    linepack_kg(t)  =  V_pipe × ρ_avg(t)

where ``ρ_avg`` is the spatially-averaged gas density computed from the average
nodal pressure via the ideal-gas equation of state:

    ρ  =  p × M / (R × T)

The stored mass can change between timesteps.  The rate of change is the **net
packing flow** ``net_pack_kgs`` [kg/s]:

    net_pack_kgs(t) × Δt  =  linepack_kg(t) − linepack_kg(t−1)

When ``net_pack_kgs > 0`` the pipe is *charging* (absorbing gas from the
network); when negative it is *discharging* (releasing gas back).  The gas is
drawn equally from the two endpoint junctions, so each junction sees an
additional flow of ``−net_pack_kgs / 2`` in its mass balance.

Capacity calculation
--------------------
The extension auto-computes per-pipe capacities from the grid parameters:

* **Initial linepack** — ``V_pipe × ρ`` at nominal pressure
  (``pressure_ref × nominal_pressure_pu``).
* **Maximum linepack** — ``V_pipe × ρ`` at the maximum allowed pressure
  (``pressure_ref × sqrt(p_squared_pu_max)``).

Both can be overridden per pipe via the ``overrides`` argument.

Single-step vs timeseries
-------------------------
In a **single-step** (steady-state) solve the linepack variables are present
but ``net_pack_kgs`` is pinned to zero — no temporal dynamics, no effect on
the flow solution beyond defining ``linepack_kg`` from the solved pressures.

In a **timeseries or multi-period** solve the inter-temporal constraint links
successive linepack states and ``net_pack_kgs`` enters the junction mass
balances, so the solver accounts for linepack charging/discharging when
balancing supply and demand.

Usage::

    # All gas pipes, auto-computed capacity:
    net.add_extension(GasLinepack())

    # All gas pipes, with per-pipe overrides:
    net.add_extension(GasLinepack(overrides={
        pipe_id: dict(linepack_kg_initial=500, linepack_kg_max=2000),
    }))

Results are accessible via ``result.get_result_for_id(pipe_id, "linepack_kg")``
and ``result.get_result_for_id(pipe_id, "net_pack_kgs")`` in timeseries mode.
"""

from __future__ import annotations

import math

from monee.model.branch import GasPipe
from monee.model.core import Var
from monee.model.grid import GasGrid

from .core import NetworkAspect


class GasLinepack(NetworkAspect):
    """
    Linepack (stored gas mass) extension for gas pipes.

    By default applies to **all** :class:`~monee.model.branch.GasPipe` branches
    in the network.  Capacities (initial and maximum stored mass) are computed
    automatically from pipe geometry and gas-grid thermodynamic parameters.

    Two variables are injected onto each active pipe:

    * ``linepack_kg`` — stored gas mass [kg].
      Algebraically equal to ``V_pipe × gas_density`` (derived from nodal
      pressures), bounded by the auto-computed or overridden maximum.
    * ``net_pack_kgs`` — net packing flow rate [kg/s].  Zero in single-step
      mode; coupled to ``Δlinepack_kg / Δt`` in timeseries / multi-period
      solves, and contributes ``−net_pack_kgs / 2`` to each endpoint junction's
      mass balance.

    Capacity auto-calculation:

    * ``linepack_kg_initial`` = ``V_pipe × ρ(p_nominal)``
    * ``linepack_kg_max``     = ``V_pipe × ρ(p_max)``

    where ``p_nominal = pressure_ref × nominal_pressure_pu`` and
    ``p_max = pressure_ref × sqrt(p_squared_pu_max)``.

    Args:
        overrides: Optional ``{branch_id: dict}`` to override the auto-computed
            capacity for specific pipes.  Recognised keys are
            ``"linepack_kg_initial"`` and ``"linepack_kg_max"``.  Omitted keys
            fall back to the auto-computed values.
    """

    def __init__(self, overrides: dict[int, dict] | None = None) -> None:
        self._overrides: dict[int, dict] = overrides or {}
        self._pipe_volume: dict[int, float] = {}
        self._initial_lp: dict[int, float] = {}
        self._active_branches: set[int] = set()
        self._timeseries_active: bool = False

    # ------------------------------------------------------------------
    # Internal: capacity from grid thermodynamics
    # ------------------------------------------------------------------

    @staticmethod
    def _density(grid: GasGrid, pressure_pa: float) -> float:
        """Return gas density [kg/m³] via the ideal-gas EoS."""
        return pressure_pa * grid.molar_mass / (grid.universal_gas_constant * grid.t_k)

    @staticmethod
    def _nominal_pressure(grid: GasGrid) -> float:
        """Nominal operating pressure [Pa]."""
        return grid.pressure_ref * grid.nominal_pressure_pu

    @staticmethod
    def _max_pressure(grid: GasGrid) -> float:
        """Maximum pressure [Pa] derived from the ``p_squared_pu_max`` grid bound."""
        return grid.pressure_ref * math.sqrt(grid.p_squared_pu_max)

    # ------------------------------------------------------------------
    # Phase 0: inject Var placeholders before variable injection
    # ------------------------------------------------------------------

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

            # Auto-compute capacities from pipe geometry and grid properties.
            rho_nominal = self._density(grid, self._nominal_pressure(grid))
            rho_max = self._density(grid, self._max_pressure(grid))
            auto_initial = v_pipe * rho_nominal
            auto_max = v_pipe * rho_max

            # Apply per-pipe overrides if provided.
            overrides = self._overrides.get(branch.id, {})
            lp_initial = overrides.get("linepack_kg_initial", auto_initial)
            lp_max = overrides.get("linepack_kg_max", auto_max)
            # Guard: max must always be >= initial.
            lp_max = max(lp_max, lp_initial * 1.05)

            # State: stored gas mass in the pipe [kg].
            bm.linepack_kg = Var(lp_initial, min=0, max=lp_max, name="linepack_kg")
            # Rate: net mass flow rate into the pipe storage [kg/s].
            # Positive = pipe absorbing gas (charging), negative = releasing (discharging).
            bm.net_pack_kgs = Var(0, name="net_pack_kgs")

            self._pipe_volume[branch.id] = v_pipe
            self._initial_lp[branch.id] = lp_initial
            self._active_branches.add(branch.id)

    # ------------------------------------------------------------------
    # Phase 1a: timeseries / multi-period activation
    # ------------------------------------------------------------------

    def activate_timeseries(self, network, ignored_nodes: set, step_state=None) -> None:
        """
        Called before node equations are assembled in a timeseries / multi-period
        solve.  Sets the timeseries flag so that ``equations()`` does NOT pin
        ``net_pack_kgs == 0``, and warm-starts ``linepack_kg`` from the
        previous step's solved value.
        """
        self._timeseries_active = True

        if step_state is None:
            return
        for branch in network.branches:
            if branch.id not in self._active_branches:
                continue
            if branch.ignored or branch.id in ignored_nodes:
                continue
            prev_lp = step_state.get(branch.id, "linepack_kg")
            if prev_lp is not None:
                branch.model.linepack_kg.value = prev_lp

    # ------------------------------------------------------------------
    # Phase 1b: static equations (always active)
    # ------------------------------------------------------------------

    def equations(self, network, ignored_nodes: set) -> list:
        """
        Link ``linepack_kg`` to the average gas density from nodal pressures.

        In single-step (non-timeseries) mode also pins ``net_pack_kgs == 0``
        so the variable is fully constrained with no effect on the steady-state
        flow solution.
        """
        eqs = []
        for branch in network.branches:
            if branch.id not in self._active_branches:
                continue
            if branch.ignored or branch.id in ignored_nodes:
                continue
            bm = branch.model
            # Algebraic definition: stored mass = pipe volume × average density.
            # gas_density is set by NLWeymouthBranchFormulation from nodal pressures.
            eqs.append(bm.linepack_kg == self._pipe_volume[branch.id] * bm.gas_density)

            if not self._timeseries_active:
                # Steady-state: no packing flow.
                eqs.append(bm.net_pack_kgs == 0)

        return eqs

    # ------------------------------------------------------------------
    # Phase 1c: inter-temporal constraint (timeseries + multi-period)
    # ------------------------------------------------------------------

    def inter_temporal_equations(
        self, network, ignored_nodes: set, temporal_state
    ) -> list:
        """
        Mass-conservation update for linepack across timesteps / periods::

            net_pack_kgs(t) × Δt  =  linepack_kg(t) − linepack_kg(t−1)

        ``net_pack_kgs`` appears in the junction mass balance of both endpoint
        nodes (via :meth:`~monee.model.node.Junction.calc_signed_mass_flow`),
        so the nodal balances account for gas absorbed or released by the pipe.
        """
        eqs = []
        dt_s = temporal_state.dt_h * 3600.0
        for branch in network.branches:
            if branch.id not in self._active_branches:
                continue
            if branch.ignored or branch.id in ignored_nodes:
                continue
            bm = branch.model
            prev_lp = temporal_state.get(branch.id, "linepack_kg")
            if prev_lp is None:
                # First step: anchor to initial condition.
                prev_lp = self._initial_lp[branch.id]
            eqs.append(bm.net_pack_kgs * dt_s == bm.linepack_kg - prev_lp)
        return eqs

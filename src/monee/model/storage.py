"""
Storage child models for multi-energy timeseries simulation.

Each model exposes a tracked SoC variable persisted to ``StepState`` between
timesteps; ``inter_temporal_equations`` couples it to the previous step.

Load convention: positive = charging (consume from network), negative = discharging.

Dispatch (``p_mw`` / ``mass_flow_kgs``) is a plain float by default - fixed setpoint
in energy-flow solves. Call :meth:`make_controllable` (or
``OptimizationProblem.controllable_storages()``) to promote it to a Var.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .core import ChildModel, Var, is_plain_number, model

if TYPE_CHECKING:
    from monee.simulation.step_state import InterStepState


def _unity_regulation(regulation) -> bool:
    return is_plain_number(regulation) and regulation == 1


class _SocStorage(ChildModel):
    """Shared SoC-storage core: dispatch integration with an optional
    charge/discharge efficiency split. Subclasses parameterize the attribute
    names, Var names and the dt unit via class attributes."""

    _DISPATCH: str
    _SOC: str
    _SOC_INITIAL: str
    _CHARGE: str
    _DISCHARGE: str
    _BOUND_MIN: str
    _BOUND_MAX: str
    _DISPATCH_VAR_NAME: str
    _CHARGE_VAR_NAME: str
    _DISCHARGE_VAR_NAME: str
    _DT_FACTOR: float

    def _init_storage_state(
        self,
        soc_initial,
        soc_max,
        dispatch_max,
        dispatch_initial,
        efficiency_charge,
        efficiency_discharge,
        regulation,
    ) -> None:
        setattr(self, self._SOC_INITIAL, soc_initial)
        self.regulation = regulation
        self.efficiency_charge = efficiency_charge
        self.efficiency_discharge = efficiency_discharge
        setattr(self, self._DISPATCH, float(dispatch_initial))
        self._init_extra_state()
        setattr(self, self._SOC, Var(soc_initial, min=0, max=soc_max, name=self._SOC))
        self._lossy = (
            abs(efficiency_charge - 1.0) > 1e-12
            or abs(efficiency_discharge - 1.0) > 1e-12
        )
        # Bounds limit the raw dispatch; the nodal balance scales it by
        # regulation once, so the grid-side limit is dispatch_max * regulation.
        setattr(self, self._BOUND_MIN, -dispatch_max)
        setattr(self, self._BOUND_MAX, dispatch_max)

    def _init_extra_state(self) -> None:
        pass

    def make_controllable(self):
        """Promote the dispatch (and loss-split vars if lossy) into solver Vars."""
        current = getattr(self, self._DISPATCH)
        val = float(current) if isinstance(current, (int, float)) else 0.0
        setattr(
            self,
            self._DISPATCH,
            Var(
                val,
                min=getattr(self, self._BOUND_MIN),
                max=getattr(self, self._BOUND_MAX),
                name=self._DISPATCH_VAR_NAME,
            ),
        )
        if self._lossy:
            bound_max = getattr(self, self._BOUND_MAX)
            setattr(
                self,
                self._CHARGE,
                Var(0, min=0, max=bound_max, name=self._CHARGE_VAR_NAME),
            )
            setattr(
                self,
                self._DISCHARGE,
                Var(0, min=0, max=bound_max, name=self._DISCHARGE_VAR_NAME),
            )

    def equations(self, grid, node, **kwargs):
        # Lossy split only applies in optimisation mode (dispatch is a Var).
        dispatch = getattr(self, self._DISPATCH)
        if self._lossy and isinstance(dispatch, Var):
            return [
                dispatch == getattr(self, self._CHARGE) - getattr(self, self._DISCHARGE)
            ]
        return []

    def inter_temporal_equations(
        self, temporal_state: InterStepState, component_id, **kwargs
    ):
        prev = temporal_state.get(component_id, self._SOC)
        dt = temporal_state.dt_h * self._DT_FACTOR
        if prev is None:
            prev = getattr(self, self._SOC_INITIAL)
        soc = getattr(self, self._SOC)
        dispatch = getattr(self, self._DISPATCH)
        # The nodal balance sees regulation * dispatch; integrate the same
        # regulation-scaled quantity so the SoC conserves energy with the grid.
        reg = self.regulation
        unity = _unity_regulation(reg)
        if self._lossy:
            if isinstance(dispatch, (int, float)):
                # Plain energy flow: fixed dispatch - sign-based efficiency.
                p = float(dispatch)
                delta = (
                    dt * self.efficiency_charge * p
                    if p >= 0
                    else dt * p / self.efficiency_discharge
                )
                if not unity:
                    delta = delta * reg
                return [soc == prev + delta]
            charge = getattr(self, self._CHARGE)
            discharge = getattr(self, self._DISCHARGE)
            if unity:
                return [
                    soc
                    == prev
                    + dt * self.efficiency_charge * charge
                    - dt * discharge / self.efficiency_discharge,
                ]
            return [
                soc
                == prev
                + (
                    dt * self.efficiency_charge * charge
                    - dt * discharge / self.efficiency_discharge
                )
                * reg,
            ]
        if unity:
            return [soc == prev + dt * dispatch]
        return [soc == prev + dt * dispatch * reg]


@model
class ElectricStorage(_SocStorage):
    """
    Battery/electric storage attached to a power bus.

    SoC update: ``e_mwh(t) = e_mwh(t-1) + dt_h * p_mw(t)``. With efficiency
    losses, ``p_mw = p_charge_mw - p_discharge_mw`` (both :math:`\ge` 0) and SoC uses
    :math:`\eta_c \cdot p_{charge} - p_{discharge} / \eta_d`.
    """

    _DISPATCH = "p_mw"
    _SOC = "e_mwh"
    _SOC_INITIAL = "_e_mwh_initial"
    _CHARGE = "p_charge_mw"
    _DISCHARGE = "p_discharge_mw"
    _BOUND_MIN = "_p_min"
    _BOUND_MAX = "_p_max"
    _DISPATCH_VAR_NAME = "storage_p_mw"
    _CHARGE_VAR_NAME = "storage_p_charge_mw"
    _DISCHARGE_VAR_NAME = "storage_p_discharge_mw"
    _DT_FACTOR = 1.0

    def __init__(
        self,
        e_mwh_initial,
        e_mwh_max,
        p_max_mw,
        p_mw_initial=0.0,
        efficiency_charge=1.0,
        efficiency_discharge=1.0,
        regulation=1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._init_storage_state(
            e_mwh_initial,
            e_mwh_max,
            p_max_mw,
            p_mw_initial,
            efficiency_charge,
            efficiency_discharge,
            regulation,
        )

    def _init_extra_state(self) -> None:
        self.q_mvar = 0


@model
class GasStorage(_SocStorage):
    """
    Pressurised gas storage at a gas junction.

    SoC update: ``m_stored_kg(t) = m_stored_kg(t-1) + dt_s * mass_flow_kgs(t)``.
    Lossy: ``mass_flow_kgs = flow_charge_kgs - flow_discharge_kgs`` with
    ``η_c * charge - discharge / η_d`` in the SoC update.
    """

    _DISPATCH = "mass_flow_kgs"
    _SOC = "m_stored_kg"
    _SOC_INITIAL = "_m_stored_kg_initial"
    _CHARGE = "flow_charge_kgs"
    _DISCHARGE = "flow_discharge_kgs"
    _BOUND_MIN = "_flow_min"
    _BOUND_MAX = "_flow_max"
    _DISPATCH_VAR_NAME = "storage_mass_flow"
    _CHARGE_VAR_NAME = "storage_flow_charge_kgs"
    _DISCHARGE_VAR_NAME = "storage_flow_discharge_kgs"
    _DT_FACTOR = 3600.0

    def __init__(
        self,
        m_stored_kg_initial,
        m_stored_kg_max,
        flow_max_kgs,
        mass_flow_initial_kgs=0.0,
        efficiency_charge=1.0,
        efficiency_discharge=1.0,
        regulation=1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._init_storage_state(
            m_stored_kg_initial,
            m_stored_kg_max,
            flow_max_kgs,
            mass_flow_initial_kgs,
            efficiency_charge,
            efficiency_discharge,
            regulation,
        )


@model
class ThermalStorage(ChildModel):
    """
    Thermal storage (e.g. hot-water tank) at a water junction.

    SoC update: ``m_stored_kg(t) = m_stored_kg(t-1) - loss*dt_h*m_stored_kg(t-1) + dt_s*mass_flow_kgs(t)``.
    """

    def __init__(
        self,
        m_stored_kg_initial,
        m_stored_kg_max,
        flow_max_kgs,
        mass_flow_initial_kgs=0.0,
        loss_factor_per_h=0.0,
        regulation=1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._m_stored_kg_initial = m_stored_kg_initial
        self.regulation = regulation
        self.loss_factor_per_h = loss_factor_per_h
        self.mass_flow_kgs = float(mass_flow_initial_kgs)
        self.m_stored_kg = Var(
            m_stored_kg_initial,
            min=0,
            max=m_stored_kg_max,
            name="m_stored_kg",
        )
        # Raw-dispatch bounds; the nodal balance applies regulation once.
        self._flow_min = -flow_max_kgs
        self._flow_max = flow_max_kgs

    def make_controllable(self):
        """Promote ``mass_flow_kgs`` into a solver Var."""
        current = self.mass_flow_kgs
        val = float(current) if isinstance(current, (int, float)) else 0.0
        self.mass_flow_kgs = Var(
            val,
            min=self._flow_min,
            max=self._flow_max,
            name="thermal_storage_mass_flow",
        )

    def equations(self, grid, node, **kwargs):
        return []

    def inter_temporal_equations(
        self, temporal_state: InterStepState, component_id, **kwargs
    ):
        prev_m = temporal_state.get(component_id, "m_stored_kg")
        dt_s = temporal_state.dt_h * 3600.0
        dt_h = temporal_state.dt_h
        if prev_m is None:
            prev_m = self._m_stored_kg_initial
        loss = prev_m * self.loss_factor_per_h * dt_h
        # Integrate the regulation-scaled flow the nodal balance sees.
        reg = self.regulation
        if _unity_regulation(reg):
            return [self.m_stored_kg == prev_m - loss + dt_s * self.mass_flow_kgs]
        return [self.m_stored_kg == prev_m - loss + dt_s * self.mass_flow_kgs * reg]

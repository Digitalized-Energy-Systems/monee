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

from .core import ChildModel, Var, model

if TYPE_CHECKING:
    from monee.simulation.step_state import InterStepState


def _unity_regulation(regulation) -> bool:
    return (
        isinstance(regulation, (int, float))
        and not isinstance(regulation, bool)
        and regulation == 1
    )


@model
class ElectricStorage(ChildModel):
    """
    Battery/electric storage attached to a power bus.

    SoC update: ``e_mwh(t) = e_mwh(t-1) + dt_h * p_mw(t)``. With efficiency
    losses, ``p_mw = p_charge_mw - p_discharge_mw`` (both :math:`\ge` 0) and SoC uses
    :math:`\eta_c \cdot p_{charge} - p_{discharge} / \eta_d`.
    """

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
        self._e_mwh_initial = e_mwh_initial
        self.regulation = regulation
        self.efficiency_charge = efficiency_charge
        self.efficiency_discharge = efficiency_discharge
        self.p_mw = float(p_mw_initial)
        self.q_mvar = 0
        self.e_mwh = Var(
            e_mwh_initial,
            min=0,
            max=e_mwh_max,
            name="e_mwh",
        )
        self._lossy = (
            abs(efficiency_charge - 1.0) > 1e-12
            or abs(efficiency_discharge - 1.0) > 1e-12
        )
        # Bounds limit the raw dispatch; the nodal balance scales it by
        # regulation once, so the grid-side limit is p_max * regulation.
        self._p_min = -p_max_mw
        self._p_max = p_max_mw

    def make_controllable(self):
        """Promote ``p_mw`` (and loss-split vars if lossy) into solver Vars."""
        current = self.p_mw
        val = float(current) if isinstance(current, (int, float)) else 0.0
        self.p_mw = Var(val, min=self._p_min, max=self._p_max, name="storage_p_mw")
        if self._lossy:
            self.p_charge_mw = Var(
                0, min=0, max=self._p_max, name="storage_p_charge_mw"
            )
            self.p_discharge_mw = Var(
                0, min=0, max=self._p_max, name="storage_p_discharge_mw"
            )

    def equations(self, grid, node, **kwargs):
        # Lossy split only applies in optimisation mode (p_mw is a Var).
        if self._lossy and isinstance(self.p_mw, Var):
            return [self.p_mw == self.p_charge_mw - self.p_discharge_mw]
        return []

    def inter_temporal_equations(
        self, temporal_state: InterStepState, component_id, **kwargs
    ):
        prev_e = temporal_state.get(component_id, "e_mwh")
        dt_h = temporal_state.dt_h
        if prev_e is None:
            prev_e = self._e_mwh_initial
        # The nodal balance sees regulation * p_mw; integrate the same
        # regulation-scaled power so the SoC conserves energy with the grid.
        reg = self.regulation
        unity = _unity_regulation(reg)
        if self._lossy:
            if isinstance(self.p_mw, (int, float)):
                # Plain energy flow: fixed p_mw - sign-based efficiency.
                p = float(self.p_mw)
                delta = (
                    dt_h * self.efficiency_charge * p
                    if p >= 0
                    else dt_h * p / self.efficiency_discharge
                )
                if not unity:
                    delta = delta * reg
                return [self.e_mwh == prev_e + delta]
            if unity:
                return [
                    self.e_mwh
                    == prev_e
                    + dt_h * self.efficiency_charge * self.p_charge_mw
                    - dt_h * self.p_discharge_mw / self.efficiency_discharge,
                ]
            return [
                self.e_mwh
                == prev_e
                + (
                    dt_h * self.efficiency_charge * self.p_charge_mw
                    - dt_h * self.p_discharge_mw / self.efficiency_discharge
                )
                * reg,
            ]
        if unity:
            return [self.e_mwh == prev_e + dt_h * self.p_mw]
        return [self.e_mwh == prev_e + dt_h * self.p_mw * reg]


@model
class GasStorage(ChildModel):
    """
    Pressurised gas storage at a gas junction.

    SoC update: ``m_stored_kg(t) = m_stored_kg(t-1) + dt_s * mass_flow_kgs(t)``.
    Lossy: ``mass_flow_kgs = flow_charge_kgs - flow_discharge_kgs`` with
    ``η_c * charge - discharge / η_d`` in the SoC update.
    """

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
        self._m_stored_kg_initial = m_stored_kg_initial
        self.regulation = regulation
        self.efficiency_charge = efficiency_charge
        self.efficiency_discharge = efficiency_discharge
        self.mass_flow_kgs = float(mass_flow_initial_kgs)
        self.m_stored_kg = Var(
            m_stored_kg_initial,
            min=0,
            max=m_stored_kg_max,
            name="m_stored_kg",
        )
        self._lossy = (
            abs(efficiency_charge - 1.0) > 1e-12
            or abs(efficiency_discharge - 1.0) > 1e-12
        )
        # Raw-dispatch bounds; the nodal balance applies regulation once.
        self._flow_min = -flow_max_kgs
        self._flow_max = flow_max_kgs

    def make_controllable(self):
        """Promote ``mass_flow_kgs`` (and loss-split vars if lossy) into solver Vars."""
        current = self.mass_flow_kgs
        val = float(current) if isinstance(current, (int, float)) else 0.0
        self.mass_flow_kgs = Var(
            val, min=self._flow_min, max=self._flow_max, name="storage_mass_flow"
        )
        if self._lossy:
            self.flow_charge_kgs = Var(
                0, min=0, max=self._flow_max, name="storage_flow_charge_kgs"
            )
            self.flow_discharge_kgs = Var(
                0, min=0, max=self._flow_max, name="storage_flow_discharge_kgs"
            )

    def equations(self, grid, node, **kwargs):
        if self._lossy and isinstance(self.mass_flow_kgs, Var):
            return [
                self.mass_flow_kgs == self.flow_charge_kgs - self.flow_discharge_kgs
            ]
        return []

    def inter_temporal_equations(
        self, temporal_state: InterStepState, component_id, **kwargs
    ):
        prev_m = temporal_state.get(component_id, "m_stored_kg")
        dt_s = temporal_state.dt_h * 3600.0
        if prev_m is None:
            prev_m = self._m_stored_kg_initial
        # Integrate the regulation-scaled flow the nodal balance sees.
        reg = self.regulation
        unity = _unity_regulation(reg)
        if self._lossy:
            if isinstance(self.mass_flow_kgs, (int, float)):
                f = float(self.mass_flow_kgs)
                delta = (
                    dt_s * self.efficiency_charge * f
                    if f >= 0
                    else dt_s * f / self.efficiency_discharge
                )
                if not unity:
                    delta = delta * reg
                return [self.m_stored_kg == prev_m + delta]
            if unity:
                return [
                    self.m_stored_kg
                    == prev_m
                    + dt_s * self.efficiency_charge * self.flow_charge_kgs
                    - dt_s * self.flow_discharge_kgs / self.efficiency_discharge,
                ]
            return [
                self.m_stored_kg
                == prev_m
                + (
                    dt_s * self.efficiency_charge * self.flow_charge_kgs
                    - dt_s * self.flow_discharge_kgs / self.efficiency_discharge
                )
                * reg,
            ]
        if unity:
            return [self.m_stored_kg == prev_m + dt_s * self.mass_flow_kgs]
        return [self.m_stored_kg == prev_m + dt_s * self.mass_flow_kgs * reg]


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

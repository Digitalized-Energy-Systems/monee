"""
Storage child models for multi-energy timeseries simulation.

Each model exposes a ``tracked`` state-of-charge variable that is
automatically persisted to ``StepState`` between timesteps.  The
``inter_step_equations`` method returns constraints that couple the
current-step SoC to the previous-step value and the power/flow exchange.

Sign convention (load convention, consistent with the rest of monee):
  * Positive  →  consuming from the network  (charging)
  * Negative  →  injecting into the network  (discharging)

**Plain energy flow vs. optimisation**

By default, storage power (``p_mw`` / ``mass_flow``) is a plain Python float,
not a :class:`~monee.model.core.Var`.  This means:

* In a plain energy flow solve, the dispatch is fixed at its current value
  (default 0.0, or whatever was set via ``TimeseriesData``).  The solver only
  tracks the state-of-charge (``e_mwh`` / ``m_stored_kg``), which is still a
  ``tracked`` variable.  Dispatch can be prescribed step-by-step via
  ``TimeseriesData.add_child_series(bat_id, "p_mw", [...])`` .

* In an optimisation solve, call
  ``OptimizationProblem.controllable_storages()`` (or
  ``problem.controllable_storages_electric()`` / ``*_gas()`` / ``*_thermal()``)
  before running.  This converts ``p_mw`` (and the loss-split vars if relevant)
  into proper ``Var`` objects so the solver can optimise dispatch.

Example — specify dispatch externally in plain timeseries::

    td = TimeseriesData()
    td.add_child_series(bat_id, "p_mw", [1.0, -1.0, 0.5, -0.5])
    result = run_timeseries(net, td)

Example — optimise dispatch::

    problem = OptimizationProblem()
    problem.controllable_storages()   # enables optimisation for all storage
    result = run_multi_period(net, td, optimization_problem=problem)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .core import ChildModel, Var, model

if TYPE_CHECKING:
    from monee.simulation.step_state import InterStepState


@model
class ElectricStorage(ChildModel):
    """
    Battery / electric storage attached to a power bus.

    The active-power attribute ``p_mw`` follows the load convention:
    positive = charging (consuming from bus), negative = discharging
    (injecting into bus).  ``e_mwh`` is the state of charge in MWh.

    ``p_mw`` is stored as a plain float by default so it acts as a fixed
    dispatch setpoint in plain energy-flow solves.  Call
    ``OptimizationProblem.controllable_storages()`` to turn it into a
    :class:`~monee.model.core.Var` for optimisation.

    SoC update (no efficiency losses)::

        e_mwh(t) = e_mwh(t-1) + dt_h * p_mw(t)

    With efficiency losses the charge and discharge directions are modelled
    with separate non-negative variables ``p_charge_mw`` and
    ``p_discharge_mw`` (both ≥ 0), linked by ``p_mw = p_charge_mw -
    p_discharge_mw``::

        e_mwh(t) = e_mwh(t-1)
                   + dt_h * efficiency_charge   * p_charge_mw(t)
                   - dt_h * p_discharge_mw(t)   / efficiency_discharge

    In plain energy flow with a lossy model, the effective SoC change is
    computed directly from the sign of the fixed ``p_mw`` value — no
    ``p_charge_mw`` / ``p_discharge_mw`` variables are created.

    The initial SoC (*e_mwh_initial*) is used as the reference for the
    t=0 inter-step constraint, so the first period is fully anchored even
    when no prior timeseries step exists.

    Args:
        e_mwh_initial (float): Initial state of charge in MWh.
        e_mwh_max (float): Usable storage capacity in MWh.
        p_max_mw (float): Maximum charge/discharge power in MW.
        p_mw_initial (float): Initial / default dispatch in MW.  Used as the
            fixed setpoint in plain energy-flow solves and as the starting
            point for the solver in optimisation mode.  Defaults to 0.0.
        efficiency_charge (float): Round-trip efficiency on the charging
            side (0 < η ≤ 1).  Defaults to 1.0 (lossless).
        efficiency_discharge (float): Round-trip efficiency on the
            discharging side (0 < η ≤ 1).  Defaults to 1.0 (lossless).
        regulation (float): Scale factor applied to power limits (0–1).
            Useful for curtailment or de-rating studies.  Defaults to 1.
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
        # Fixed dispatch by default; converted to Var by controllable_storages()
        self.p_mw = float(p_mw_initial)
        self.q_mvar = 0
        self.e_mwh = Var(
            e_mwh_initial,
            min=0,
            max=e_mwh_max,
            name="e_mwh",
        )
        self._lossy = efficiency_charge != 1.0 or efficiency_discharge != 1.0
        # Store bounds for use by make_controllable() / controllable_storages().
        self._p_min = -p_max_mw * regulation
        self._p_max = p_max_mw * regulation

    def make_controllable(self):
        """Convert ``p_mw`` (and loss-split vars if lossy) into optimisation variables.

        Called automatically by :meth:`OptimizationProblem.controllable_storages`.
        May also be called manually before passing a network to an optimisation
        solver when no ``OptimizationProblem`` helper is used.
        """
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
        # Only add the lossy split constraint when p_mw is a solver variable
        # (i.e. optimisation mode).  In plain energy flow p_mw is a fixed
        # float and p_charge_mw / p_discharge_mw do not exist.
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
        if self._lossy:
            if isinstance(self.p_mw, (int, float)):
                # Plain energy flow: p_mw is fixed — apply sign-based efficiency.
                p = float(self.p_mw)
                delta = (
                    dt_h * self.efficiency_charge * p
                    if p >= 0
                    else dt_h * p / self.efficiency_discharge
                )
                return [self.e_mwh == prev_e + delta]
            # Optimisation mode: use the dedicated charge/discharge variables.
            return [
                self.e_mwh
                == prev_e
                + dt_h * self.efficiency_charge * self.p_charge_mw
                - dt_h * self.p_discharge_mw / self.efficiency_discharge,
            ]
        return [self.e_mwh == prev_e + dt_h * self.p_mw]


@model
class GasStorage(ChildModel):
    """
    Pressurised gas storage (e.g. tank or cavern) attached to a gas junction.

    The ``mass_flow`` attribute follows the load convention: positive =
    charging (gas is withdrawn from the junction and injected into storage),
    negative = discharging (gas is released from storage into the junction).
    ``m_stored_kg`` is the stored mass in kg.

    ``mass_flow`` is stored as a plain float by default so it acts as a
    fixed dispatch setpoint in plain energy-flow solves.  Call
    ``OptimizationProblem.controllable_storages()`` to turn it into a
    :class:`~monee.model.core.Var` for optimisation.

    SoC update (no efficiency losses)::

        m_stored_kg(t) = m_stored_kg(t-1) + dt_s * mass_flow(t)

    With efficiency losses the charge and discharge directions are modelled
    with separate non-negative variables ``flow_charge_kgs`` and
    ``flow_discharge_kgs`` (both ≥ 0), linked by
    ``mass_flow = flow_charge_kgs - flow_discharge_kgs``::

        m_stored_kg(t) = m_stored_kg(t-1)
                         + dt_s * efficiency_charge   * flow_charge_kgs(t)
                         - dt_s * flow_discharge_kgs(t) / efficiency_discharge

    The initial stored mass (*m_stored_kg_initial*) is used as the
    reference for the t=0 inter-step constraint.

    Args:
        m_stored_kg_initial (float): Initial stored mass in kg.
        m_stored_kg_max (float): Maximum storage capacity in kg.
        flow_max_kgs (float): Maximum charge/discharge flow rate in kg/s.
        mass_flow_initial (float): Initial / default dispatch in kg/s.  Used as
            the fixed setpoint in plain energy-flow solves and as the starting
            point for the solver in optimisation mode.  Defaults to 0.0.
        efficiency_charge (float): Injection efficiency (0 < η ≤ 1).
            Defaults to 1.0 (lossless).
        efficiency_discharge (float): Withdrawal efficiency (0 < η ≤ 1).
            Defaults to 1.0 (lossless).
        regulation (float): Scale factor applied to flow limits (0–1).
            Useful for curtailment or de-rating studies.  Defaults to 1.
    """

    def __init__(
        self,
        m_stored_kg_initial,
        m_stored_kg_max,
        flow_max_kgs,
        mass_flow_initial=0.0,
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
        # Fixed dispatch by default; converted to Var by controllable_storages()
        self.mass_flow = float(mass_flow_initial)
        self.m_stored_kg = Var(
            m_stored_kg_initial,
            min=0,
            max=m_stored_kg_max,
            name="m_stored_kg",
        )
        self._lossy = efficiency_charge != 1.0 or efficiency_discharge != 1.0
        # Store bounds for use by make_controllable() / controllable_storages().
        self._flow_min = -flow_max_kgs * regulation
        self._flow_max = flow_max_kgs * regulation

    def make_controllable(self):
        """Convert ``mass_flow`` (and loss-split vars if lossy) into optimisation variables."""
        current = self.mass_flow
        val = float(current) if isinstance(current, (int, float)) else 0.0
        self.mass_flow = Var(
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
        if self._lossy and isinstance(self.mass_flow, Var):
            return [self.mass_flow == self.flow_charge_kgs - self.flow_discharge_kgs]
        return []

    def inter_temporal_equations(
        self, temporal_state: InterStepState, component_id, **kwargs
    ):
        prev_m = temporal_state.get(component_id, "m_stored_kg")
        dt_s = temporal_state.dt_h * 3600.0
        if prev_m is None:
            prev_m = self._m_stored_kg_initial
        if self._lossy:
            if isinstance(self.mass_flow, (int, float)):
                f = float(self.mass_flow)
                delta = (
                    dt_s * self.efficiency_charge * f
                    if f >= 0
                    else dt_s * f / self.efficiency_discharge
                )
                return [self.m_stored_kg == prev_m + delta]
            return [
                self.m_stored_kg
                == prev_m
                + dt_s * self.efficiency_charge * self.flow_charge_kgs
                - dt_s * self.flow_discharge_kgs / self.efficiency_discharge,
            ]
        return [self.m_stored_kg == prev_m + dt_s * self.mass_flow]


@model
class ThermalStorage(ChildModel):
    """
    Thermal energy storage (e.g. hot-water tank) attached to a water junction.

    The ``mass_flow`` attribute follows the load convention: positive =
    charging (water is drawn from the junction supply into the tank),
    negative = discharging (water is released from the tank into the junction).
    ``m_stored_kg`` is the stored water mass in kg.

    ``mass_flow`` is stored as a plain float by default so it acts as a
    fixed dispatch setpoint in plain energy-flow solves.  Call
    ``OptimizationProblem.controllable_storages()`` to turn it into a
    :class:`~monee.model.core.Var` for optimisation.

    SoC update (with optional standing losses)::

        m_stored_kg(t) = m_stored_kg(t-1)
                         - loss_factor_per_h * dt_h * m_stored_kg(t-1)
                         + dt_s * mass_flow(t)

    The initial stored mass (*m_stored_kg_initial*) is used as the
    reference for the t=0 inter-step constraint.

    Args:
        m_stored_kg_initial (float): Initial stored mass in kg.
        m_stored_kg_max (float): Maximum storage capacity in kg.
        flow_max_kgs (float): Maximum charge/discharge flow rate in kg/s.
        mass_flow_initial (float): Initial / default dispatch in kg/s.  Used as
            the fixed setpoint in plain energy-flow solves and as the starting
            point for the solver in optimisation mode.  Defaults to 0.0.
        loss_factor_per_h (float): Fractional standing heat loss per hour
            (e.g. 0.01 = 1 % of stored mass per hour).  Defaults to 0.
        regulation (float): Scale factor applied to flow limits (0–1).
            Useful for curtailment or de-rating studies.  Defaults to 1.
    """

    def __init__(
        self,
        m_stored_kg_initial,
        m_stored_kg_max,
        flow_max_kgs,
        mass_flow_initial=0.0,
        loss_factor_per_h=0.0,
        regulation=1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._m_stored_kg_initial = m_stored_kg_initial
        self.regulation = regulation
        self.loss_factor_per_h = loss_factor_per_h
        # Fixed dispatch by default; converted to Var by controllable_storages()
        self.mass_flow = float(mass_flow_initial)
        self.m_stored_kg = Var(
            m_stored_kg_initial,
            min=0,
            max=m_stored_kg_max,
            name="m_stored_kg",
        )
        # Store bounds for use by make_controllable() / controllable_storages().
        self._flow_min = -flow_max_kgs * regulation
        self._flow_max = flow_max_kgs * regulation

    def make_controllable(self):
        """Convert ``mass_flow`` into an optimisation variable."""
        current = self.mass_flow
        val = float(current) if isinstance(current, (int, float)) else 0.0
        self.mass_flow = Var(
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
        return [self.m_stored_kg == prev_m - loss + dt_s * self.mass_flow]

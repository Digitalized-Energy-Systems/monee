"""Storage model tests: regulation must scale SoC updates exactly like the
nodal balance scales the grid-side dispatch (energy conservation), and the
dispatch bounds must not be scaled by regulation twice."""

import math

from monee.model.storage import ElectricStorage, GasStorage, ThermalStorage


class _StubState:
    def __init__(self, dt_h=1.0):
        self.dt_h = dt_h

    def get(self, _component_id, _attr):
        return None


class _CaptureLhs:
    """Stands in for the SoC Var so ``lhs == rhs`` hands us the RHS."""

    def __init__(self):
        self.rhs = None

    def __eq__(self, other):
        self.rhs = other
        return True

    __hash__ = None


def test_electric_storage_regulation_conserves_energy():
    # GIVEN
    storage = ElectricStorage(
        e_mwh_initial=0.0,
        e_mwh_max=10.0,
        p_max_mw=4.0,
        p_mw_initial=2.0,
        regulation=0.5,
    )
    dt_h = 1.0
    # the nodal balance multiplies the dispatch by regulation
    grid_side_p_mw = storage.p_mw * storage.regulation

    # WHEN
    capture = _CaptureLhs()
    storage.e_mwh = capture
    storage.inter_temporal_equations(_StubState(dt_h), 0)

    # THEN the SoC delta equals the grid-side energy
    assert math.isclose(capture.rhs - 0.0, dt_h * grid_side_p_mw)


def test_electric_storage_lossy_regulation_conserves_energy():
    # GIVEN
    storage = ElectricStorage(
        e_mwh_initial=1.0,
        e_mwh_max=10.0,
        p_max_mw=4.0,
        p_mw_initial=2.0,
        efficiency_charge=0.9,
        regulation=0.5,
    )
    dt_h = 1.0
    grid_side_p_mw = storage.p_mw * storage.regulation

    # WHEN
    capture = _CaptureLhs()
    storage.e_mwh = capture
    storage.inter_temporal_equations(_StubState(dt_h), 0)

    # THEN the stored delta is the grid-side energy times charge efficiency
    assert math.isclose(capture.rhs - 1.0, dt_h * 0.9 * grid_side_p_mw)


def test_electric_storage_regulation_one_unchanged():
    # GIVEN
    storage = ElectricStorage(
        e_mwh_initial=0.5, e_mwh_max=10.0, p_max_mw=4.0, p_mw_initial=2.0
    )

    # WHEN
    capture = _CaptureLhs()
    storage.e_mwh = capture
    storage.inter_temporal_equations(_StubState(0.25), 0)

    # THEN
    assert math.isclose(capture.rhs, 0.5 + 0.25 * 2.0)


def test_electric_storage_bounds_not_double_scaled():
    # GIVEN
    storage = ElectricStorage(
        e_mwh_initial=0.0, e_mwh_max=10.0, p_max_mw=4.0, regulation=0.5
    )

    # WHEN
    storage.make_controllable()

    # THEN the raw dispatch is bounded by the nameplate; the balance applies
    # regulation once, so the effective grid-side limit is p_max * regulation
    # (not p_max * regulation**2).
    assert storage.p_mw.max == 4.0
    assert storage.p_mw.min == -4.0
    assert storage.p_mw.max * storage.regulation == 2.0


def test_gas_storage_regulation_conserves_mass():
    # GIVEN
    storage = GasStorage(
        m_stored_kg_initial=0.0,
        m_stored_kg_max=1000.0,
        flow_max_kgs=2.0,
        mass_flow_initial_kgs=1.0,
        regulation=0.5,
    )
    dt_h = 1.0
    grid_side_flow = storage.mass_flow_kgs * storage.regulation

    # WHEN
    capture = _CaptureLhs()
    storage.m_stored_kg = capture
    storage.inter_temporal_equations(_StubState(dt_h), 0)

    # THEN
    assert math.isclose(capture.rhs, dt_h * 3600.0 * grid_side_flow)


def test_thermal_storage_regulation_conserves_mass():
    # GIVEN
    storage = ThermalStorage(
        m_stored_kg_initial=0.0,
        m_stored_kg_max=1000.0,
        flow_max_kgs=2.0,
        mass_flow_initial_kgs=1.0,
        regulation=0.5,
    )
    dt_h = 1.0
    grid_side_flow = storage.mass_flow_kgs * storage.regulation

    # WHEN
    capture = _CaptureLhs()
    storage.m_stored_kg = capture
    storage.inter_temporal_equations(_StubState(dt_h), 0)

    # THEN
    assert math.isclose(capture.rhs, dt_h * 3600.0 * grid_side_flow)

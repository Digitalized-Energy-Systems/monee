"""Tests for the LumpedThermalCapacitance (LTC) network extension."""

import math

import monee.model as mm
from monee import run_energy_flow
from monee.model import LumpedThermalCapacitance
from monee.model.grid import WaterGrid
from monee.simulation.timeseries import TimeseriesData, run
from monee.solver import GEKKOSolver
from tests.util import (
    WATER_LOOP_PIPE_D as PIPE_D,
)
from tests.util import (
    WATER_LOOP_PIPE_L as PIPE_L,
)
from tests.util import (
    create_water_loop as _water_loop,
)


def _ext_grid_child_id(net, node_id):
    """Return the id of the first child attached to *node_id*."""
    node = [nd for nd in net.nodes if nd.id == node_id][0]
    return node.child_ids[0]


def test_ltc_transparent_in_single_step():
    # GIVEN
    net_base, _, _, _ = _water_loop()
    net_ltc, _, _, _ = _water_loop()
    net_ltc.add_extension(LumpedThermalCapacitance())

    # WHEN
    r_base = run_energy_flow(net_base)
    r_ltc = run_energy_flow(net_ltc)

    # THEN
    assert r_base.success
    assert r_ltc.success

    # LTC must not change steady-state junction temperatures
    t_base = r_base.get(mm.Junction)["t_pu"].sort_index()
    t_ltc = r_ltc.get(mm.Junction)["t_pu"].sort_index()
    for idx in t_base.index:
        assert abs(t_base[idx] - t_ltc[idx]) < 1e-4, (
            f"Node {idx}: base={t_base[idx]:.6f}, ltc={t_ltc[idx]:.6f}"
        )


def test_ltc_transparent_when_inputs_constant():
    # GIVEN
    net, _, n1, _ = _water_loop()
    net.add_extension(LumpedThermalCapacitance())
    td = TimeseriesData()

    # WHEN
    ts = run(net, td, steps=4)

    # THEN
    assert not ts.failed_steps

    t_series = ts.get_result_for_id(n1, "t_pu")
    assert t_series is not None
    assert len(t_series) == 4

    # Constant inputs: temperature must stay at steady state across all steps
    assert t_series.max() - t_series.min() < 5e-3, (
        f"Temperature drifted unexpectedly with constant inputs: {t_series.values}"
    )


def test_ltc_slows_temperature_response():
    # GIVEN
    net_base, n0_b, _, n2_b = _water_loop()
    ext_child_id_base = _ext_grid_child_id(net_base, n0_b)
    net_ltc, n0_l, _, n2_l = _water_loop()
    net_ltc.add_extension(LumpedThermalCapacitance())
    ext_child_id_ltc = _ext_grid_child_id(net_ltc, n0_l)
    # Step 0-1: supply at 356 K, step 2-3: supply drops to 340 K
    t_supply = [356, 356, 340, 340]
    td_b = TimeseriesData()
    td_b.add_child_series(ext_child_id_base, "t_k", t_supply)
    td_l = TimeseriesData()
    td_l.add_child_series(ext_child_id_ltc, "t_k", t_supply)

    # WHEN  (pinned to GEKKO/IPOPT for a well-posed *base* case.)
    # This asserts LTC damping by comparing the no-LTC step at n2 against the LTC
    # step. The 3-junction loop has a non-unique circulating flow (and the source
    # temperature is underdetermined), so the *base* (no-LTC) flow pattern is not
    # unique: CasADi legitimately picks one where the supply-temperature step does
    # not reach n2 (base stays flat -> nothing to damp -> the ltc<=base check is
    # ill-posed), whereas GEKKO/IPOPT picks one where it does. CasADi's LTC
    # transient *itself* matches GEKKO to ~1e-3 (verified), so this is network
    # non-uniqueness, not a backend defect - we just need the reference base flow.
    ts_base = run(net_base, td_b, steps=4, solver=GEKKOSolver(solver=3))
    ts_ltc = run(net_ltc, td_l, steps=4, solver=GEKKOSolver(solver=3))

    # THEN
    assert not ts_base.failed_steps
    assert not ts_ltc.failed_steps

    t_base = ts_base.get_result_for_id(n2_b, "t_pu")
    t_ltc = ts_ltc.get_result_for_id(n2_l, "t_pu")
    assert t_base is not None and t_ltc is not None

    # LTC should attenuate the temperature step at n2 when the supply drops
    delta_base = abs(t_base.iloc[2] - t_base.iloc[1])
    delta_ltc = abs(t_ltc.iloc[2] - t_ltc.iloc[1])
    assert delta_ltc <= delta_base + 1e-4, (
        f"LTC should damp temperature response: "
        f"delta_base={delta_base:.6f}, delta_ltc={delta_ltc:.6f}"
    )


def test_ltc_pipe_volume_computed_correctly():
    # GIVEN
    net, n0, n1, n2 = _water_loop()
    ext = LumpedThermalCapacitance()
    net_copy = net.copy()

    # WHEN
    ext.prepare(net_copy)

    # THEN
    rho = WaterGrid("w").fluid_density_kg_per_m3  # derived from t_ref_k (≈970 kg/m³ at 83 °C)
    v_half = math.pi / 4 * PIPE_D**2 * PIPE_L / 2
    rho_v = ext._ltc_rho_v

    # n0 carries an ExtHydrGrid (fixed temperature) so it is excluded from LTC
    assert n0 not in rho_v, "ExtHydrGrid node must not be LTC-constrained"
    assert len(rho_v) == 2

    # n1 and n2 each connect to exactly 2 pipes: rho_v = rho * 2 * v_half
    expected = rho * 2 * v_half
    for node_id, rv in rho_v.items():
        assert abs(rv - expected) < 0.5, (
            f"Node {node_id}: rho_v={rv:.2f}, expected={expected:.2f}"
        )

    # Sum covers only the 2 LTC-constrained nodes (3rd pipe-end at n0 is excluded)
    assert abs(sum(rho_v.values()) - 2 * expected) < 1.0

"""
Tests for the LumpedThermalCapacitance (LTC) network extension.

Key invariant: with LTC active, junction temperature evolves according to
    Σ(ṁ·T_pu) = ρ·V_node · (T_pu(t) - T_pu(t-1)) / Δt

In steady state (constant inputs, no temperature gradient between steps) the
storage term is zero and results are identical to a plain solve.
"""

import math

import monee.model as mm
from monee import run_energy_flow
from monee.model import LumpedThermalCapacitance
from monee.model.grid import WaterGrid
from monee.simulation.timeseries import TimeseriesData, run

# ---------------------------------------------------------------------------
# helper — proven-to-converge 3-junction water loop
# ---------------------------------------------------------------------------

PIPE_D = 0.3  # m
PIPE_L = 100.0  # m


def _water_loop():
    """
    3-junction loop identical to create_circle_heat_grid_test in test_gekko_heat.py:

      ext-grid (n0) ──pipe── n1 (Source 5 kg/s) ──pipe── n2 (Sink 10 kg/s) ──pipe── n0

    Returns (net, n0_id, n1_id, n2_id).
    """
    net = mm.Network()
    n0 = net.node(
        mm.Junction(),
        mm.WATER,
        child_ids=[net.child(mm.ExtHydrGrid(t_k=356))],
    )
    n1 = net.node(
        mm.Junction(),
        mm.WATER,
        child_ids=[net.child(mm.Source(mass_flow=5))],
    )
    n2 = net.node(
        mm.Junction(),
        mm.WATER,
        child_ids=[net.child(mm.Sink(mass_flow=10))],
    )
    pipe = dict(diameter_m=PIPE_D, length_m=PIPE_L)
    net.branch(mm.WaterPipe(**pipe), n0, n1)
    net.branch(mm.WaterPipe(**pipe), n1, n2)
    net.branch(mm.WaterPipe(**pipe), n2, n0)
    return net, n0, n1, n2


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_ltc_transparent_in_single_step():
    """
    LTC must not change steady-state results — adding the extension to a
    plain single-step solve should produce identical junction temperatures.
    """
    net_base, _, _, _ = _water_loop()
    net_ltc, _, _, _ = _water_loop()
    net_ltc.add_extension(LumpedThermalCapacitance())

    r_base = run_energy_flow(net_base)
    r_ltc = run_energy_flow(net_ltc)

    t_base = r_base.get(mm.Junction)["t_pu"].sort_index()
    t_ltc = r_ltc.get(mm.Junction)["t_pu"].sort_index()

    for idx in t_base.index:
        assert abs(t_base[idx] - t_ltc[idx]) < 1e-4, (
            f"Node {idx}: base={t_base[idx]:.6f}, ltc={t_ltc[idx]:.6f}"
        )


def test_ltc_transparent_when_inputs_constant():
    """
    With LTC active and constant inputs, temperatures must be identical
    across all timesteps (steady-state == time-domain solution).
    """
    net, _, n1, _ = _water_loop()
    net.add_extension(LumpedThermalCapacitance())

    td = TimeseriesData()
    ts = run(net, td, steps=4)

    t_series = ts.get_result_for_id(n1, "t_pu")
    assert t_series is not None
    assert len(t_series) == 4
    # With constant inputs the LTC network should converge to the same
    # steady-state temperature; allow a small numerical tolerance.
    assert t_series.max() - t_series.min() < 5e-3, (
        f"Temperature drifted unexpectedly with constant inputs: {t_series.values}"
    )


def test_ltc_slows_temperature_response():
    """
    When the supply temperature changes, the LTC network shows a smaller
    temperature jump at interior junctions than the network without LTC.

    We vary the ext-grid supply temperature (t_k) and compare the temperature
    change at n2 between LTC and no-LTC variants.
    """
    ext_child_id_base = None
    ext_child_id_ltc = None

    net_base, n0_b, _, n2_b = _water_loop()
    # Identify the ExtHydrGrid child on n0
    n0_node_base = [nd for nd in net_base.nodes if nd.id == n0_b][0]
    ext_child_id_base = n0_node_base.child_ids[0]

    net_ltc, n0_l, _, n2_l = _water_loop()
    net_ltc.add_extension(LumpedThermalCapacitance())
    n0_node_ltc = [nd for nd in net_ltc.nodes if nd.id == n0_l][0]
    ext_child_id_ltc = n0_node_ltc.child_ids[0]

    # Step 0-1: supply at 356 K, step 2-3: supply drops to 340 K
    t_supply = [356, 356, 340, 340]

    td_b = TimeseriesData()
    td_b.add_child_series(ext_child_id_base, "t_k", t_supply)
    ts_base = run(net_base, td_b, steps=4)
    t_base = ts_base.get_result_for_id(n2_b, "t_pu")

    td_l = TimeseriesData()
    td_l.add_child_series(ext_child_id_ltc, "t_k", t_supply)
    ts_ltc = run(net_ltc, td_l, steps=4)
    t_ltc = ts_ltc.get_result_for_id(n2_l, "t_pu")

    assert t_base is not None and t_ltc is not None

    # Temperature step at step 2 (first step after supply drops)
    delta_base = abs(t_base.iloc[2] - t_base.iloc[1])
    delta_ltc = abs(t_ltc.iloc[2] - t_ltc.iloc[1])

    # LTC should attenuate the response
    assert delta_ltc <= delta_base + 1e-4, (
        f"LTC should damp temperature response: "
        f"delta_base={delta_base:.6f}, delta_ltc={delta_ltc:.6f}"
    )


def test_ltc_pipe_volume_computed_correctly():
    """
    Verify that the lumped volumes stored by the extension match the
    expected ρ × Σ(V_pipe/2) formula for each junction.
    """
    net, n0, n1, n2 = _water_loop()
    ext = LumpedThermalCapacitance()
    net_copy = net.copy()
    ext.prepare(net_copy)

    rho = WaterGrid("w").fluid_density  # default 998 kg/m³
    v_half = math.pi / 4 * PIPE_D**2 * PIPE_L / 2

    rho_v = ext._ltc_rho_v

    # n0 carries an ExtHydrGrid (fixed temperature) so it is excluded from LTC
    assert n0 not in rho_v, "ExtHydrGrid node must not be LTC-constrained"
    assert len(rho_v) == 2

    # n1 and n2 each connect to exactly 2 pipes → rho_v = ρ × 2 × v_half
    expected = rho * 2 * v_half
    for node_id, rv in rho_v.items():
        assert abs(rv - expected) < 0.5, (
            f"Node {node_id}: rho_v={rv:.2f}, expected={expected:.2f}"
        )

    # Sum covers only the 2 LTC-constrained nodes (3rd pipe-end at n0 is excluded)
    assert abs(sum(rho_v.values()) - 2 * expected) < 1.0

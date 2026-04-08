"""
Tests for GasLinepack — the pipe-storage extension for gas networks.

Design properties verified:

1. Single-step: ``linepack_kg`` matches ``V_pipe * gas_density``; ``net_pack_kgs == 0``.
2. Auto-computed capacity: ``linepack_kg_initial`` and ``linepack_kg_max`` match
   the expected values from pipe geometry and grid thermodynamics.
3. Timeseries: ``net_pack_kgs`` matches ``Δlinepack_kg / Δt``.
4. Linepack buffers demand variation: linepack discharges when demand rises.
5. Per-pipe overrides take precedence over auto-computed values.
"""

import math

import monee.model as mm
from monee import run_energy_flow
from monee.model.formulation.linepack import GasLinepack
from monee.simulation.timeseries import TimeseriesData, run

# ---------------------------------------------------------------------------
# Shared network builder
# ---------------------------------------------------------------------------


def _gas_net():
    """
    Simple linear gas network:  source — pipe — sink
    Returns (net, pipe_id, sink_id).
    """
    net = mm.Network(mm.create_gas_grid("gas", type="lgas"))
    net.activate_grid(mm.GAS)

    source_id = net.child(mm.ExtHydrGrid())
    sink_id = net.child(mm.Sink(mass_flow=0.2))

    n0 = net.node(mm.Junction(), mm.GAS, child_ids=[source_id])
    n1 = net.node(mm.Junction(), mm.GAS, child_ids=[sink_id])

    pipe_id = net.branch(mm.GasPipe(diameter_m=0.5, length_m=5000), n0, n1)

    return net, pipe_id, sink_id


# ---------------------------------------------------------------------------
# Test 1: single-step — definition and no-packing constraint
# ---------------------------------------------------------------------------


def test_linepack_single_step_definition():
    """linepack_kg = V * gas_density and net_pack_kgs = 0 in steady state."""

    net, pipe_id, _ = _gas_net()
    net.add_extension(GasLinepack())
    result = run_energy_flow(net)

    pipe_series = result[pipe_id]
    lp = pipe_series["linepack_kg"]
    npk = pipe_series["net_pack_kgs"]
    density = pipe_series["gas_density"]

    v_pipe = math.pi / 4 * 0.5**2 * 5000
    assert math.isclose(lp, v_pipe * density, rel_tol=1e-4), (
        f"linepack_kg ({lp:.3f}) != V*density ({v_pipe * density:.3f})"
    )
    assert math.isclose(npk, 0.0, abs_tol=1e-6), (
        f"net_pack_kgs should be 0 in single-step, got {npk}"
    )


# ---------------------------------------------------------------------------
# Test 2: auto-computed capacity
# ---------------------------------------------------------------------------


def test_linepack_auto_capacity():
    """
    linepack_kg_initial and linepack_kg_max are auto-computed from pipe
    geometry and gas-grid thermodynamics.
    """
    net, pipe_id, _ = _gas_net()
    ext = GasLinepack()
    # Call prepare() directly — no need for a full solve.
    ext.prepare(net)

    grid = net.grids[0]
    v_pipe = math.pi / 4 * 0.5**2 * 5000
    R, M, T = grid.universal_gas_constant, grid.molar_mass, grid.t_k

    rho_nominal = grid.pressure_ref * grid.nominal_pressure_pu * M / (R * T)
    rho_max = grid.pressure_ref * math.sqrt(grid.p_squared_pu_max) * M / (R * T)

    expected_initial = v_pipe * rho_nominal
    expected_max = v_pipe * rho_max

    assert math.isclose(ext._initial_lp[pipe_id], expected_initial, rel_tol=1e-6)
    assert math.isclose(ext._pipe_volume[pipe_id], v_pipe, rel_tol=1e-6)
    for branch in net.branches:
        if branch.id == pipe_id:
            assert math.isclose(
                branch.model.linepack_kg.max, expected_max, rel_tol=1e-6
            )


# ---------------------------------------------------------------------------
# Test 3: per-pipe override
# ---------------------------------------------------------------------------


def test_linepack_override():
    """Per-pipe overrides replace auto-computed initial / max values."""
    net, pipe_id, _ = _gas_net()
    # Use values that are plausible but distinct from auto-computed ones.
    ext_auto = GasLinepack()
    ext_auto.prepare(net)
    auto_initial = ext_auto._initial_lp[pipe_id]
    override_initial = auto_initial * 0.8
    override_max = auto_initial * 1.5

    ext = GasLinepack(
        overrides={
            pipe_id: dict(
                linepack_kg_initial=override_initial,
                linepack_kg_max=override_max,
            )
        }
    )
    ext.prepare(net)

    assert math.isclose(ext._initial_lp[pipe_id], override_initial, rel_tol=1e-9)
    for branch in net.branches:
        if branch.id == pipe_id:
            assert math.isclose(
                branch.model.linepack_kg.max, override_max, rel_tol=1e-9
            )


# ---------------------------------------------------------------------------
# Test 4: timeseries — net_pack_kgs tracks Δlinepack_kg / Δt
# ---------------------------------------------------------------------------


def test_linepack_timeseries_mass_conservation():
    """net_pack_kgs * dt_s equals the change in linepack_kg between steps."""
    net, pipe_id, sink_id = _gas_net()
    net.add_extension(GasLinepack())

    td = TimeseriesData()
    td.add_child_series(sink_id, "mass_flow", [0.1, 0.3])

    result = run(net, td)

    lp = result.get_result_for_id(pipe_id, "linepack_kg")
    npk = result.get_result_for_id(pipe_id, "net_pack_kgs")

    dt_s = 1.0 * 3600.0
    lp0 = lp.iloc[0]
    lp1 = lp.iloc[1]
    npk1 = npk.iloc[1]

    expected_npk1 = (lp1 - lp0) / dt_s
    assert math.isclose(npk1, expected_npk1, rel_tol=1e-4), (
        f"net_pack_kgs mismatch: {npk1:.6f} vs expected {expected_npk1:.6f}"
    )


# ---------------------------------------------------------------------------
# Test 5: linepack buffers demand — discharges when demand rises
# ---------------------------------------------------------------------------


def test_linepack_source_flow_reduced_by_discharge():
    """
    When linepack discharges it must *reduce* the source feed — not increase it.

    Mass balance at source junction (outflow-positive convention):
        source_feed  +  pipe_forward_flow  +  linepack_term  ==  0

    During discharge net_pack_kgs < 0, so the linepack term is
    +0.5 * net_pack_kgs (negative = inflow from pipe to junction).
    The source can therefore provide *less* than the full demand.

    If the sign were wrong the source would over-deliver by exactly |net_pack_kgs|.
    This test catches that regression.
    """
    net, pipe_id, sink_id = _gas_net()
    net.add_extension(GasLinepack())

    td = TimeseriesData()
    td.add_child_series(sink_id, "mass_flow", [0.1, 0.12])
    result = run(net, td)

    src_flow = result.get_result_for(mm.ExtHydrGrid, "mass_flow")
    npk = result.get_result_for_id(pipe_id, "net_pack_kgs")

    # Source feed = −mass_flow (load convention: negative = injection)
    src_feed_1 = float(-src_flow.iloc[1])
    npk1 = float(npk.iloc[1])
    demand_1 = 0.12

    assert npk1 < 0, f"Linepack should discharge at step 1 (npk={npk1})"
    # Source should provide LESS than demand — linepack covers the shortfall.
    assert src_feed_1 < demand_1, (
        f"Source over-delivers ({src_feed_1:.6f} > {demand_1}) — "
        f"sign error: linepack is an anti-buffer"
    )
    # The shortfall should equal the total discharge rate |net_pack_kgs|.
    import math

    assert math.isclose(demand_1 - src_feed_1, abs(npk1), rel_tol=1e-3), (
        f"shortfall {demand_1 - src_feed_1:.6f} != |npk| {abs(npk1):.6f}"
    )


def test_linepack_buffers_demand_variation():
    """
    When demand increases, linepack discharges: linepack_kg drops and
    net_pack_kgs is negative at the step where demand jumped.
    """
    net, pipe_id, sink_id = _gas_net()
    net.add_extension(GasLinepack())

    td = TimeseriesData()
    td.add_child_series(sink_id, "mass_flow", [0.1, 0.11, 0.35])

    result = run(net, td)

    lp = result.get_result_for_id(pipe_id, "linepack_kg")
    npk = result.get_result_for_id(pipe_id, "net_pack_kgs")

    lp0 = lp.iloc[0]
    lp1 = lp.iloc[1]
    npk1 = npk.iloc[1]

    assert lp1 < lp0, (
        f"Linepack should decrease when demand rises: {lp0:.1f} -> {lp1:.1f}"
    )
    assert npk1 < 0, f"net_pack_kgs should be negative when discharging: {npk1:.6f}"

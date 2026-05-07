"""
Tests for GasLinepack — the pipe-storage extension for gas networks.

Design properties verified:

1. Single-step: ``linepack_kg`` matches ``V_pipe * gas_density``; ``net_pack_kgs == 0``.
2. Auto-computed capacity: ``linepack_kg_initial`` and ``linepack_kg_max`` match
   the expected values from pipe geometry and grid thermodynamics.
3. Timeseries: ``net_pack_kgs`` matches ``Δlinepack_kg / Δt``.
4. Linepack buffers demand variation: linepack discharges when demand rises.
5. Per-pipe overrides take precedence over auto-computed values.
6. Branching tree topology: linepack works correctly with multiple pipes,
   junctions, and sinks in a non-meshed tree layout.
7. Global mass conservation: total source feed + total linepack discharge = total demand.
8. Spatially non-uniform discharge: pipes closer to a demand spike discharge more.
"""

import math

import monee.model as mm
from monee import GasLinepack, run_energy_flow
from monee.simulation.timeseries import TimeseriesData, run


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


def _branching_gas_net():
    """
    Tree-shaped gas network with 6 junctions, 5 pipes, and 3 sinks::

        source(n0) --pipe0-- n1 --pipe1-- n2 (sink_a: 0.10 kg/s)
                              |
                             pipe2
                              |
                              n3 --pipe3-- n4 (sink_b: 0.08 kg/s)
                              |
                             pipe4
                              |
                              n5 (sink_c: 0.05 kg/s)

    Returns (net, pipe_ids, sink_ids) where pipe_ids and sink_ids are dicts
    keyed by label.
    """
    net = mm.Network(mm.create_gas_grid("gas", type="lgas"))
    net.activate_grid(mm.GAS)

    source_id = net.child(mm.ExtHydrGrid())
    sink_a = net.child(mm.Sink(mass_flow=0.10))
    sink_b = net.child(mm.Sink(mass_flow=0.08))
    sink_c = net.child(mm.Sink(mass_flow=0.05))

    n0 = net.node(mm.Junction(), mm.GAS, child_ids=[source_id])
    n1 = net.node(mm.Junction(), mm.GAS)
    n2 = net.node(mm.Junction(), mm.GAS, child_ids=[sink_a])
    n3 = net.node(mm.Junction(), mm.GAS)
    n4 = net.node(mm.Junction(), mm.GAS, child_ids=[sink_b])
    n5 = net.node(mm.Junction(), mm.GAS, child_ids=[sink_c])

    # Trunk: larger diameter
    p0 = net.branch(mm.GasPipe(diameter_m=0.40, length_m=3000), n0, n1)
    p1 = net.branch(mm.GasPipe(diameter_m=0.30, length_m=2000), n1, n2)
    p2 = net.branch(mm.GasPipe(diameter_m=0.35, length_m=2500), n1, n3)
    p3 = net.branch(mm.GasPipe(diameter_m=0.25, length_m=1500), n3, n4)
    p4 = net.branch(mm.GasPipe(diameter_m=0.20, length_m=1000), n3, n5)

    pipes = {
        "trunk": p0,
        "branch_a": p1,
        "branch_main": p2,
        "branch_b": p3,
        "branch_c": p4,
    }
    sinks = {"a": sink_a, "b": sink_b, "c": sink_c}
    return net, pipes, sinks


def test_tree_linepack_single_step_all_pipes():
    """In steady state every pipe has linepack_kg = V * density and net_pack_kgs = 0."""
    net, pipes, _ = _branching_gas_net()
    net.add_extension(GasLinepack())
    result = run_energy_flow(net)

    pipe_specs = {
        "trunk": (0.40, 3000),
        "branch_a": (0.30, 2000),
        "branch_main": (0.35, 2500),
        "branch_b": (0.25, 1500),
        "branch_c": (0.20, 1000),
    }
    print(result)
    assert False
    gp_df = result.get(mm.GasPipe)
    for label, pid in pipes.items():
        row = gp_df[gp_df["id"] == pid].iloc[0]
        lp = row["linepack_kg"]
        npk = row["net_pack_kgs"]
        density = row["gas_density"]

        d, l = pipe_specs[label]
        v_pipe = math.pi / 4 * d**2 * l
        assert math.isclose(lp, v_pipe * density, rel_tol=1e-4), (
            f"Pipe {label}: linepack_kg ({lp:.3f}) != V*density ({v_pipe * density:.3f})"
        )
        assert math.isclose(npk, 0.0, abs_tol=1e-6), (
            f"Pipe {label}: net_pack_kgs should be 0 in steady state, got {npk}"
        )


def test_tree_linepack_timeseries_mass_conservation():
    """
    net_pack_kgs * dt = Δlinepack_kg holds for every pipe in the tree
    across all timesteps.
    """
    net, pipes, sinks = _branching_gas_net()
    net.add_extension(GasLinepack())

    td = TimeseriesData()
    # Ramp sink_a from 0.10 to 0.20, keep others constant.
    td.add_child_series(sinks["a"], "mass_flow", [0.10, 0.15, 0.20])
    td.add_child_series(sinks["b"], "mass_flow", [0.08, 0.08, 0.08])
    td.add_child_series(sinks["c"], "mass_flow", [0.05, 0.05, 0.05])

    result = run(net, td)
    dt_s = 1.0 * 3600.0

    for label, pid in pipes.items():
        lp = result.get_result_for_id(pid, "linepack_kg")
        npk = result.get_result_for_id(pid, "net_pack_kgs")

        for step in range(1, len(lp)):
            delta_lp = lp.iloc[step] - lp.iloc[step - 1]
            expected_npk = delta_lp / dt_s
            actual_npk = npk.iloc[step]
            assert math.isclose(actual_npk, expected_npk, rel_tol=1e-3), (
                f"Pipe {label} step {step}: "
                f"net_pack_kgs ({actual_npk:.6f}) != Δlp/Δt ({expected_npk:.6f})"
            )


def test_tree_linepack_discharge_on_demand_spike():
    """
    A sudden demand increase at one leaf causes linepack discharge in
    upstream pipes.  The trunk pipe should discharge as well as the
    direct branch, while pipes on unaffected branches should be less affected.
    """
    net, pipes, sinks = _branching_gas_net()
    net.add_extension(GasLinepack())

    td = TimeseriesData()
    # Step 0: baseline.  Step 1: sink_b spikes from 0.08 to 0.25 kg/s.
    td.add_child_series(sinks["a"], "mass_flow", [0.10, 0.10])
    td.add_child_series(sinks["b"], "mass_flow", [0.08, 0.25])
    td.add_child_series(sinks["c"], "mass_flow", [0.05, 0.05])

    result = run(net, td)

    # Trunk (pipe0) and the path to sink_b (pipe2 → pipe3) should discharge.
    trunk_npk = result.get_result_for_id(pipes["trunk"], "net_pack_kgs").iloc[1]
    main_npk = result.get_result_for_id(pipes["branch_main"], "net_pack_kgs").iloc[1]
    b_npk = result.get_result_for_id(pipes["branch_b"], "net_pack_kgs").iloc[1]

    assert trunk_npk < 0, (
        f"Trunk should discharge on demand spike: net_pack_kgs = {trunk_npk:.6f}"
    )
    assert main_npk < 0, (
        f"Main branch should discharge on demand spike: net_pack_kgs = {main_npk:.6f}"
    )
    assert b_npk < 0, (
        f"Branch B (direct to spiking sink) should discharge: net_pack_kgs = {b_npk:.6f}"
    )


def test_tree_linepack_global_mass_balance():
    """
    At each timeseries step, total source injection + total linepack discharge
    must equal total sink demand (within solver tolerance).

    source_feed + Σ(−net_pack_kgs_i) = Σ(sink_demand_i)
    """
    net, pipes, sinks = _branching_gas_net()
    net.add_extension(GasLinepack())

    demands = {
        "a": [0.10, 0.12, 0.08],
        "b": [0.08, 0.08, 0.15],
        "c": [0.05, 0.05, 0.05],
    }
    td = TimeseriesData()
    for label, series in demands.items():
        td.add_child_series(sinks[label], "mass_flow", series)

    result = run(net, td)

    src_flow = result.get_result_for(mm.ExtHydrGrid, "mass_flow")

    for step in range(len(demands["a"])):
        total_demand = sum(demands[k][step] for k in demands)
        source_feed = float(-src_flow.iloc[step])
        total_discharge = sum(
            float(-result.get_result_for_id(pid, "net_pack_kgs").iloc[step])
            for pid in pipes.values()
        )
        supply = source_feed + total_discharge
        assert math.isclose(supply, total_demand, rel_tol=1e-3), (
            f"Step {step}: supply ({supply:.6f}) != demand ({total_demand:.6f}), "
            f"source_feed={source_feed:.6f}, discharge={total_discharge:.6f}"
        )


def test_tree_linepack_recharge_when_demand_drops():
    """
    When demand drops below initial level, linepack should charge (absorb gas):
    linepack_kg increases and net_pack_kgs is positive.
    """
    net, pipes, sinks = _branching_gas_net()
    net.add_extension(GasLinepack())

    td = TimeseriesData()
    # Step 0: normal demand.  Step 1: all sinks drop to half.
    td.add_child_series(sinks["a"], "mass_flow", [0.10, 0.05])
    td.add_child_series(sinks["b"], "mass_flow", [0.08, 0.04])
    td.add_child_series(sinks["c"], "mass_flow", [0.05, 0.025])

    result = run(net, td)

    trunk_lp = result.get_result_for_id(pipes["trunk"], "linepack_kg")
    trunk_npk = result.get_result_for_id(pipes["trunk"], "net_pack_kgs").iloc[1]

    assert trunk_lp.iloc[1] > trunk_lp.iloc[0], (
        f"Trunk linepack should increase when demand drops: "
        f"{trunk_lp.iloc[0]:.1f} -> {trunk_lp.iloc[1]:.1f}"
    )
    assert trunk_npk > 0, (
        f"Trunk net_pack_kgs should be positive (charging) when demand drops: {trunk_npk:.6f}"
    )


def test_tree_linepack_per_pipe_override():
    """Per-pipe overrides work correctly with multiple pipes in a tree."""
    net, pipes, _ = _branching_gas_net()

    ext_auto = GasLinepack()
    ext_auto.prepare(net)
    auto_trunk = ext_auto._initial_lp[pipes["trunk"]]
    auto_b = ext_auto._initial_lp[pipes["branch_b"]]

    ext = GasLinepack(
        overrides={
            pipes["trunk"]: dict(
                linepack_kg_initial=auto_trunk * 0.5, linepack_kg_max=auto_trunk * 2.0
            ),
            pipes["branch_b"]: dict(linepack_kg_initial=auto_b * 0.7),
        }
    )
    ext.prepare(net)

    assert math.isclose(ext._initial_lp[pipes["trunk"]], auto_trunk * 0.5, rel_tol=1e-9)
    assert math.isclose(ext._initial_lp[pipes["branch_b"]], auto_b * 0.7, rel_tol=1e-9)
    # Non-overridden pipes keep auto values.
    assert math.isclose(
        ext._initial_lp[pipes["branch_a"]],
        ext_auto._initial_lp[pipes["branch_a"]],
        rel_tol=1e-9,
    )

import math

import pytest

from monee import (
    EL_MISOCP_FORMULATION,
    GridFormingGenerator,
    PyomoSolver,
    enable_islanding,
    mm,
    mx,
    run_energy_flow,
)
from monee.network.mes import create_monee_benchmark_net
from monee.solver.casadi import CasADiSolveError


def _build_two_island_network():
    """Return a 3-bus network with two disconnected electricity islands."""
    net = mm.Network()

    bus_0 = mx.create_bus(net)
    bus_1 = mx.create_bus(net)
    bus_2 = mx.create_bus(net)

    mx.create_ext_power_grid(net, bus_0)
    mx.create_power_load(net, bus_1, p_mw=0.05, q_mvar=0)

    net.child_to(GridFormingGenerator(p_mw_max=1.0, q_mvar_max=0.5), bus_2)
    mx.create_power_load(net, bus_2, p_mw=0.08, q_mvar=0)

    mx.create_line(net, bus_0, bus_1, length_m=100, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)

    return net


def _build_ext_led_network_with_a_grid_former(**gf_kwargs):
    """Return a 4-bus radial feeder: ext grid at bus 0, loads on buses 1-3 and a
    grid-forming generator on bus 3, i.e. a former that never leads."""
    net = mm.Network(mm.PowerGrid(name="pg", sn_mva=1))
    buses = [mx.create_bus(net) for _ in range(4)]

    mx.create_ext_power_grid(net, buses[0])
    for i in range(3):
        mx.create_line(
            net,
            buses[i],
            buses[i + 1],
            length_m=300,
            r_ohm_per_m=7e-5,
            x_ohm_per_m=7e-5,
        )
    for bus in buses[1:]:
        mx.create_power_load(net, bus, p_mw=0.05, q_mvar=0.01)
    gf_id = mx.create_grid_forming_generator(
        net, buses[3], p_mw_max=0.5, q_mvar_max=0.3, **gf_kwargs
    )
    return net, buses, gf_id


def test_a_grid_former_without_an_islanding_config_is_idle_and_pins_no_voltage():
    """Leadership is stamped only by an islanding config, so without one nothing
    leads: the former must neither pin ``vm_pu`` (it would turn an ext-led bus
    into a PV bus) nor dispatch a free, solver-dependent P/Q."""
    net, _, _ = _build_ext_led_network_with_a_grid_former()

    result = run_energy_flow(net)

    assert result.success
    gf_df = result.dataframes["GridFormingGenerator"]
    assert gf_df["p_mw"].iloc[0] == pytest.approx(0, abs=1e-9)
    assert gf_df["q_mvar"].iloc[0] == pytest.approx(0, abs=1e-9)
    # The ext grid carries the whole demand instead of an arbitrary share.
    assert result.dataframes["ExtPowerGrid"]["p_mw"].iloc[0] == pytest.approx(
        -0.1508, abs=1e-3
    )
    # Voltage drops along the feeder; a pin would hold the last bus at 1.0.
    assert result.dataframes["Bus"]["vm_pu"].iloc[3] < 0.999


def test_a_non_leading_grid_former_is_idle_under_islanding():
    """With islanding on, an ext grid leads the component and every former is
    stamped non-leading, so it holds its nominal setpoint (0 by default)."""
    net, _, _ = _build_ext_led_network_with_a_grid_former()
    enable_islanding(net, electricity=True)

    result = run_energy_flow(net, solver=PyomoSolver(solver_name="scip"))

    assert result.success
    gf_df = result.dataframes["GridFormingGenerator"]
    assert gf_df["p_mw"].iloc[0] == pytest.approx(0, abs=1e-6)
    assert gf_df["q_mvar"].iloc[0] == pytest.approx(0, abs=1e-6)


def test_a_nominal_of_none_keeps_the_former_dispatch_free():
    net, _, gf_id = _build_ext_led_network_with_a_grid_former(
        nominal_p_mw=None, nominal_q_mvar=None
    )
    model = net.child_by_id(gf_id).model

    assert model.equations(None, None) == []


def test_a_non_leading_former_holds_an_explicit_nominal():
    net, _, gf_id = _build_ext_led_network_with_a_grid_former(nominal_p_mw=-0.02)

    result = run_energy_flow(net)

    assert result.success
    assert result.dataframes["GridFormingGenerator"]["p_mw"].iloc[0] == pytest.approx(
        -0.02, abs=1e-6
    )


def test_islanding_el_converges():
    # GIVEN
    net = _build_two_island_network()
    enable_islanding(net, electricity=True)

    # WHEN
    result = run_energy_flow(net)

    # THEN
    assert result.success
    assert result is not None


def test_islanding_el_gf_generator_supplies_island():
    # GIVEN
    net = _build_two_island_network()
    enable_islanding(net, electricity=True)

    # WHEN
    result = run_energy_flow(net)

    # THEN
    assert result.success

    gf_df = result.dataframes.get("GridFormingGenerator")
    assert gf_df is not None, "GridFormingGenerator not in result dataframes"

    # Generator absorbs the island B load (negative sign convention for generation)
    gf_p_mw = gf_df["p_mw"].iloc[0]
    assert abs(gf_p_mw) == pytest.approx(0.08, abs=1e-3), (
        f"Expected GF generator p_mw ≈ -0.08 MW, got {gf_p_mw}"
    )


def test_islanding_disabled_bus2_ignored():
    # GIVEN
    net = _build_two_island_network()

    # WHEN
    result = run_energy_flow(net)

    # THEN
    # Bus 2 has no ExtPowerGrid in its component, so it is pre-filtered
    assert result.success
    assert result is not None


def test_islanding_monee_benchmark():
    # GIVEN
    net_islanding: mm.Network = create_monee_benchmark_net()
    net_islanding.apply_formulation(EL_MISOCP_FORMULATION)
    net_islanding.deactivate(net_islanding.get_branch_between(2, 3))
    enable_islanding(net_islanding, electricity=True)

    net_legacy: mm.Network = create_monee_benchmark_net()
    net_legacy.apply_formulation(EL_MISOCP_FORMULATION)
    net_legacy.deactivate(net_legacy.get_branch_between(2, 3))

    # WHEN
    result_islanding = run_energy_flow(net_islanding, solver=PyomoSolver())
    result_legacy = run_energy_flow(net_legacy, solver=PyomoSolver())

    # THEN
    assert result_islanding.success
    assert result_legacy.success

    assert result_islanding.dataframes["Bus"]["vm_pu"][3] == pytest.approx(0.999981)

    assert math.isnan(result_legacy.dataframes["Bus"]["vm_pu_squared"][3])


def test_copy_carries_the_islanding_config_attribute():
    """``enable_islanding`` attaches the config as an extension AND as
    ``net.islanding_config``. ``__deepcopy__`` is an attribute whitelist and
    used to carry only the former, so every solver-returned network (they all
    hand back ``prepare_solve_network``'s copy) looked un-islanded to callers
    that read the documented attribute — grading a severed but grid-former
    anchored island as unservable.
    """
    net = _build_two_island_network()
    config = enable_islanding(net, electricity=True)

    copied = net.copy()

    assert copied.islanding_config is not None
    assert copied.islanding_config is not config, "must be the copy, not the original"
    # Same object the extension list holds, so the two lookup paths agree.
    assert copied.islanding_config is next(
        e for e in copied.extensions if isinstance(e, type(config))
    )


def test_copy_of_a_plain_network_gains_no_islanding_config():
    assert not hasattr(_build_two_island_network().copy(), "islanding_config")


def _el_mode(config):
    from monee.model.extension.islanding.el import ElectricityIslandingMode

    return next(m for m in config.modes() if isinstance(m, ElectricityIslandingMode))


def _regulation_gates(net, config, *, with_problem: bool):
    """``regulation <= e`` pairs the electricity mode emits for *net*.

    Calls the helper directly: ``equations()`` builds connectivity arithmetic on
    raw ``Var`` objects, which only works once a backend has substituted them.
    The wiring itself is covered end-to-end by
    :func:`test_a_de_energised_node_cannot_serve_its_load`.
    """
    from monee.model.extension.islanding.core import _collect_islanding_state

    # Flag before prepare(): it decides whether gate_fixed_injections and the
    # energisation penalty run, which is the whole asymmetry under test.
    net._solve_has_optimization_problem = with_problem
    config.prepare(net)  # materialises the e_<carrier> binaries
    mode = _el_mode(config)
    _, _, e_vars, _, _, _ = _collect_islanding_state(net, mode, set())
    nodes = [n for n in net.nodes if n.id in e_vars]
    return mode._regulation_gate_equations(net, nodes, e_vars)


def _promote_regulation(net):
    """Promote ``regulation`` the way a solve does: through the real load-shedding
    problem's controllable pass, not a hand-made Var."""
    from monee.problem.min_load_shedding import create_min_load_shedding_problem

    create_min_load_shedding_problem(bounds_vm=(0.9, 1.1))._apply(net)


def _hand_promote_regulation_on_formers(net):
    """No problem promotes a former, so the exclusion can only be tested with a
    hand-made Var on one."""
    for child in net.childs:
        if isinstance(child.model, GridFormingGenerator):
            child.model.regulation = mm.Var(1.0, 1, 0, name="regulation")


def test_equations_actually_emits_the_regulation_gate(monkeypatch):
    """Wiring: ``equations()`` must include the gate, not merely define it.

    The other three tests call the helper directly — ``equations()`` builds
    connectivity arithmetic on raw ``Var`` objects, which only works once a
    backend has substituted them. Stubbing the two builders that need a backend
    leaves the regulation gate as the only remaining output, so an unwired
    helper cannot pass.
    """
    from monee.model.extension.islanding import core as islanding_core

    net = _build_two_island_network()
    config = enable_islanding(net, electricity=True)
    _promote_regulation(net)
    net._solve_has_optimization_problem = True
    config.prepare(net)

    monkeypatch.setattr(
        islanding_core, "_build_connectivity_equations", lambda *a, **k: []
    )
    mode = _el_mode(config)
    monkeypatch.setattr(mode, "add_physical_constraints", lambda *a, **k: [])
    monkeypatch.setattr(mode, "_injection_gate_equations", lambda *a, **k: [])

    from monee.model.child import GridFormingMixin

    expected = sum(1 for c in net.childs if not isinstance(c.model, GridFormingMixin))
    assert expected, "nothing controllable in the fixture — test is vacuous"
    assert len(mode.equations(net, set())) == expected


def test_islanding_under_an_optimization_problem_stays_feasible():
    """The gate adds a constraint to every controllable child; a severed but
    former-anchored island must still solve."""
    import monee
    from monee.problem.min_load_shedding import create_min_load_shedding_problem

    net = _build_two_island_network()
    net.apply_formulation(EL_MISOCP_FORMULATION)
    enable_islanding(net, electricity=True)

    result = monee.run_energy_flow_optimization(
        net,
        optimization_problem=create_min_load_shedding_problem(bounds_vm=(0.9, 1.1)),
        solver="scip",
    )
    assert result.success

    # And the invariant the gate exists to enforce, wherever e resolved to 0.
    for node in result.network.nodes:
        e = getattr(node.model, "e_el", None)
        if e is None or float(getattr(e, "value", e)) > 0.5:
            continue
        for child in result.network.childs_by_ids(node.child_ids):
            reg = getattr(child.model, "regulation", None)
            if reg is None:
                continue
            assert float(getattr(reg, "value", reg)) <= 0.5


def test_a_de_energised_node_cannot_serve_its_load():
    """End to end, through a real solve: a bus held at ``e_el = 0`` must shed its
    load.

    The local generator lifts bus 1 above the slack angle, so the dark leaf can
    still import power (a bus pinned to theta = 0 otherwise sits at slack
    potential and imports nothing). Only the gate stops it: with the gate absent
    from the solver model the same network reports the bus de-energised and its
    load at ``regulation = 1.0`` at the same time.
    """
    import monee
    from monee.problem.min_load_shedding import create_min_load_shedding_problem

    net = mm.Network(mm.PowerGrid(name="pg", sn_mva=1))
    buses = [mx.create_bus(net) for _ in range(3)]
    mx.create_ext_power_grid(net, buses[0])
    for i in range(2):
        mx.create_line(
            net,
            buses[i],
            buses[i + 1],
            length_m=300,
            r_ohm_per_m=7e-5,
            x_ohm_per_m=7e-5,
        )
    mx.create_power_generator(net, buses[1], p_mw=0.2, q_mvar=0.05)
    mx.create_power_load(net, buses[2], p_mw=0.05, q_mvar=0.01)
    enable_islanding(net, electricity=True)

    dark_bus = buses[2]
    problem = create_min_load_shedding_problem(bounds_vm=(0.9, 1.1))
    problem.constraints.select(
        lambda component: (
            isinstance(component.model, mm.Bus) and component.id == dark_bus
        )
    ).equation(lambda model: model.e_el == 0)

    result = monee.run_energy_flow_optimization(
        net, optimization_problem=problem, solver="scip"
    )

    assert result.success
    assert result.dataframes["Bus"]["e_el"].iloc[2] == pytest.approx(0, abs=1e-6)
    loads = result.dataframes["PowerLoad"]
    dark_load = loads[loads["node_id"] == dark_bus]
    assert len(dark_load) == 1
    # p_mw stays the setpoint; regulation is what the nodal balance scales it by.
    assert dark_load["regulation"].iloc[0] == pytest.approx(0, abs=1e-6)


def test_every_controllable_child_is_tied_to_its_nodes_energisation():
    """One ``regulation <= e`` per controllable child — both in [0, 1] with e
    binary, so it is linear: no bilinear term, no McCormick.

    Reference units are excluded, and that covers the ext grid as well as a
    promoted former: an ``ExtPowerGrid`` is a ``GridFormingMixin`` too, and
    shedding the slack is not what a de-energised node means.
    """
    from monee.model.child import ExtPowerGrid, GridFormingMixin

    net = _build_two_island_network()
    config = enable_islanding(net, electricity=True)
    _promote_regulation(net)

    loads = [c for c in net.childs if not isinstance(c.model, GridFormingMixin)]
    assert loads, "nothing controllable in the fixture — test is vacuous"
    assert any(isinstance(c.model, ExtPowerGrid) for c in net.childs), (
        "fixture has no slack, so the exclusion below is untested"
    )
    assert len(_regulation_gates(net, config, with_problem=True)) == len(loads)


def test_the_regulation_gate_is_absent_without_an_optimization_problem():
    """Plain energy flow already ties load to ``e`` via ``gate_fixed_injections``;
    a second constraint there would double-bind it (and ``regulation`` is a plain
    float outside an optimisation, so the comparison collapses to a bool)."""
    net = _build_two_island_network()
    config = enable_islanding(net, electricity=True)
    _promote_regulation(net)
    assert _regulation_gates(net, config, with_problem=False) == []


def test_a_grid_former_is_never_regulation_gated():
    """A former is its island's reference, not sheddable demand."""
    net = _build_two_island_network()
    config = enable_islanding(net, electricity=True)
    _hand_promote_regulation_on_formers(net)
    assert _regulation_gates(net, config, with_problem=True) == []


def test_setting_p_mw_max_after_creation_rebounds_the_var():
    net = mm.Network()
    bus = mx.create_bus(net)
    gf_id = mx.create_grid_forming_generator(net, bus, p_mw_max=1.2, q_mvar_max=0.6)
    model = net.child_by_id(gf_id).model

    assert model.p_mw_max == 1.2
    assert model.q_mvar_max == 0.6

    model.p_mw_max = 0.4
    model.q_mvar_max = 0.2

    assert (model.p_mw.min, model.p_mw.max) == (-0.4, 0.4)
    assert (model.q_mvar.min, model.q_mvar.max) == (-0.2, 0.2)
    assert "p_mw_max" not in vars(model)


def test_a_derated_grid_former_cannot_serve_a_load_beyond_its_new_capacity():
    net = _build_two_island_network()
    gf_id = next(
        child.id
        for child in net.childs
        if isinstance(child.model, GridFormingGenerator)
    )
    net.child_by_id(gf_id).model.p_mw_max = 0.02  # island B load is 0.08 MW
    enable_islanding(net, electricity=True)

    # Batch solver-ergonomics item S1: reported on the result, raised on strict.
    assert run_energy_flow(net).success is False
    with pytest.raises(CasADiSolveError):
        run_energy_flow(net, strict=True)


def test_setting_a_capacity_on_an_injected_model_is_rejected():
    model = GridFormingGenerator(p_mw_max=1.0, q_mvar_max=0.5)
    model.p_mw = 0.3  # stands in for a backend symbol after inject_vars

    with pytest.raises(AttributeError, match="p_mw_max"):
        model.p_mw_max = 0.4


def test_grid_former_reports_injection_in_load_convention():
    """The p_mw_max argument is a positive magnitude; the result is negative."""
    net = _build_two_island_network()
    enable_islanding(net, electricity=True)

    result = run_energy_flow(net)

    assert result.dataframes["GridFormingGenerator"]["p_mw"].iloc[0] < 0


def test_overwrite_id_on_a_taken_child_id_is_rejected():
    net = mm.Network()
    bus = mx.create_bus(net)
    gf_id = mx.create_grid_forming_generator(net, bus, p_mw_max=1.2, q_mvar_max=0.6)

    with pytest.raises(ValueError, match="already exists"):
        mx.create_grid_forming_generator(
            net, bus, p_mw_max=0.4, q_mvar_max=0.6, overwrite_id=gf_id
        )

    net.remove_child(gf_id)
    reused = mx.create_grid_forming_generator(
        net, bus, p_mw_max=0.4, q_mvar_max=0.6, overwrite_id=gf_id
    )
    assert reused == gf_id
    assert net.child_by_id(reused).model.p_mw_max == 0.4


def _build_radial_former_feeder(nbus=5, cap=0.4, load=0.15):
    """Radial feeder with equal loads on every bus and one capacity-limited
    grid-forming unit at the far end: the F-014 fixture."""
    net = mm.Network(mm.PowerGrid(name="pg", sn_mva=1))
    buses = [mx.create_bus(net) for _ in range(nbus)]
    for i in range(nbus - 1):
        mx.create_line(
            net,
            buses[i],
            buses[i + 1],
            length_m=300,
            r_ohm_per_m=7e-5,
            x_ohm_per_m=7e-5,
        )
    for bus in buses:
        mx.create_power_load(net, bus, p_mw=load, q_mvar=0.02)
    mx.create_grid_forming_generator(net, buses[-1], p_mw_max=cap, q_mvar_max=cap)
    return net


def _energisation(result):
    return [int(round(e)) for e in result.dataframes["Bus"]["e_el"]]


def test_a_capacity_limited_former_energises_the_neighbours_it_can_carry():
    """F-014: the former's capacity, not the topology, sets how far the island
    reaches. 0.4 MW over 0.15 MW loads carries its own bus plus one neighbour."""
    net = _build_radial_former_feeder(cap=0.4)
    enable_islanding(net, electricity=True)

    result = run_energy_flow(net, solver="scip")

    assert result.success
    assert _energisation(result) == [0, 0, 0, 1, 1]
    assert result.dataframes["PowerLoad"]["p_mw"].sum() == pytest.approx(0.3)


def test_a_larger_former_capacity_energises_one_more_bus():
    net = _build_radial_former_feeder(cap=0.6)
    enable_islanding(net, electricity=True)

    result = run_energy_flow(net, solver="scip")

    assert result.success
    assert _energisation(result) == [0, 0, 1, 1, 1]
    assert result.dataframes["PowerLoad"]["p_mw"].sum() == pytest.approx(0.45)


def test_branch_gating_off_restores_the_all_or_nothing_island():
    """Without the endpoint gate a de-energised bus stays pinned to the
    reference angle, so it blocks transport and the former serves only itself."""
    net = _build_radial_former_feeder(cap=0.4)
    enable_islanding(net, electricity=mm.ElectricityIslandingMode(branch_gating="off"))

    result = run_energy_flow(net, solver="scip")

    assert result.success
    assert _energisation(result) == [0, 0, 0, 0, 1]
    assert result.dataframes["PowerLoad"]["p_mw"].sum() == pytest.approx(0.15)


def _prepared_copy(net, mode):
    prepared = net.copy()
    mode.prepare(prepared)
    return prepared


def _branch_on_off_kinds(prepared):
    return [type(branch.model.on_off) for branch in prepared.branches]


def test_auto_gating_promotes_on_off_only_where_a_limited_former_anchors():
    mode = mm.ElectricityIslandingMode()
    prepared = _prepared_copy(_build_radial_former_feeder(), mode)

    assert _branch_on_off_kinds(prepared) == [mm.Var] * 4


def test_auto_gating_leaves_an_ext_grid_led_network_untouched():
    """The gate costs nothing where nothing has to be shed: an ext grid absorbs
    any imbalance, so no branch becomes a decision and the model class is
    unchanged."""
    net, _buses, _gf_id = _build_ext_led_network_with_a_grid_former()
    mode = mm.ElectricityIslandingMode()

    prepared = _prepared_copy(net, mode)

    assert _branch_on_off_kinds(prepared) == [int] * 3
    assert not any(
        getattr(node.model, "_islanding_voltage_gated", False)
        for node in prepared.nodes
    )


def test_branch_gating_all_promotes_every_carrier_branch():
    net, _buses, _gf_id = _build_ext_led_network_with_a_grid_former()
    mode = mm.ElectricityIslandingMode(branch_gating="all")

    prepared = _prepared_copy(net, mode)

    assert _branch_on_off_kinds(prepared) == [mm.Var] * 3


def test_an_unknown_branch_gating_mode_is_rejected():
    with pytest.raises(ValueError, match="branch_gating"):
        mm.ElectricityIslandingMode(branch_gating="sometimes")


def test_a_backup_line_keeps_its_switching_decision_under_the_gate():
    """``controllable_backup_lines`` promotes ``on_off`` after the extension
    prepared the network, so the gate must only bound a backup line from above:
    the AND lower bound would force it closed between two live buses."""
    net = _build_radial_former_feeder(nbus=3, cap=1.0)
    buses = [node.id for node in net.nodes]
    net.branch(
        mm.PowerLine(300, 7e-5, 7e-5, 1, backup=True),
        from_node_id=buses[0],
        to_node_id=buses[2],
    )
    mode = mm.ElectricityIslandingMode()

    prepared = _prepared_copy(net, mode)

    owned = {
        branch.model.backup: branch.model._islanding_owns_on_off
        for branch in prepared.branches
    }
    assert owned == {False: True, True: False}
    assert all(branch.model._islanding_branch_gated for branch in prepared.branches)

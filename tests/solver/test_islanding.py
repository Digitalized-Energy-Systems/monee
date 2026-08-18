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


def _promote_regulation(net, *, formers: bool):
    for child in net.childs:
        if isinstance(child.model, GridFormingGenerator) is not formers:
            continue
        if hasattr(child.model, "regulation"):
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
    _promote_regulation(net, formers=False)
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
    _promote_regulation(net, formers=False)

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
    _promote_regulation(net, formers=False)
    assert _regulation_gates(net, config, with_problem=False) == []


def test_a_grid_former_is_never_regulation_gated():
    """A former is its island's reference, not sheddable demand."""
    net = _build_two_island_network()
    config = enable_islanding(net, electricity=True)
    _promote_regulation(net, formers=True)
    assert _regulation_gates(net, config, with_problem=True) == []

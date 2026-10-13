import monee
from monee.model.core import Var
from monee.model.formulation import EL_MISOCP_FORMULATION
from monee.model.multi import (
    CHPControlNode,
    CHPHGControlNode,
    GasToHeatControlNode,
    GasToHeatHG,
    GasToPower,
    PowerToGas,
    PowerToHeatControlNode,
    PowerToHeatHG,
)
from monee.network import create_restoration_benchmark, create_urban_district_net
from monee.problem.min_load_shedding import create_min_load_shedding_problem
from monee.solver import PyomoSolver

# Coupling-point models: their regulation is a free decision variable when the
# problem is built with include_coupling_points=False (no objective cost), so
# cutting them is optimal dispatch (e.g. shutting a 65%-efficient P2G reduces
# net external import under the ext-grid zero-exchange slack), not shedding.
CP_TYPES = (
    CHPControlNode,
    CHPHGControlNode,
    GasToHeatControlNode,
    PowerToHeatControlNode,
    GasToPower,
    PowerToGas,
    PowerToHeatHG,
    GasToHeatHG,
)


def _all_regulations(network):
    """Collect regulation values of demand/generator components.

    Coupling points are excluded: with include_coupling_points=False their
    regulation is intentionally an unpenalised degree of freedom, so its value
    says nothing about load shedding.
    """
    regs = []
    for comp in network.all_components():
        m = comp.model
        if isinstance(m, CP_TYPES):
            continue
        if hasattr(m, "regulation"):
            reg = m.regulation
            val = reg.value if isinstance(reg, Var) else reg
            if isinstance(val, (int, float)):
                regs.append(val)
    return regs


def test_min_load_shedding_no_incident():
    # GIVEN
    net = create_urban_district_net()
    net.apply_formulation(EL_MISOCP_FORMULATION)
    problem = create_min_load_shedding_problem(
        bounds_vm=(0.5, 1.5),
        bounds_pressure=(0.5, 1.5),
        bounds_t=(0.5, 1.5),
        include_ext_grids=False,
        include_storages=False,
    )
    print(monee.run_energy_flow(net, solver=PyomoSolver()))

    # WHEN
    result = monee.run_energy_flow_optimization(
        net, optimization_problem=problem, solver=PyomoSolver()
    )
    print(result)

    # THEN
    assert result.success
    assert result is not None
    assert result.objective is not None

    # No incident, so all regulations should be ~1 (no shedding)
    regs = _all_regulations(result.network)
    assert len(regs) > 0, "Expected at least one component with regulation"
    for reg in regs:
        assert reg > 0.99, f"regulation {reg} is too low - unexpected shedding"


def test_min_load_shedding_with_ext_grids():
    # GIVEN
    net = create_urban_district_net()
    net.apply_formulation(EL_MISOCP_FORMULATION)
    problem = create_min_load_shedding_problem(
        bounds_vm=(0.5, 1.5),
        bounds_pressure=(0.5, 1.5),
        bounds_t=(0.5, 1.5),
        bounds_ext_el=(-100, 100),
        bounds_ext_gas=(-100, 100),
        bounds_ext_heat=(-100, 100),
        include_ext_grids=True,
        include_storages=False,
    )

    # WHEN
    result = monee.run_energy_flow_optimization(
        net, optimization_problem=problem, solver=PyomoSolver()
    )

    # THEN
    assert result.success
    assert result is not None
    assert result.objective is not None


def test_min_load_shedding_no_incident_large():
    # GIVEN
    net = create_restoration_benchmark(misocp=True)
    problem = create_min_load_shedding_problem(
        bounds_vm=(0.5, 1.5),
        bounds_pressure=(0.5, 1.5),
        bounds_t=(0.5, 1.5),
        include_ext_grids=True,
        include_storages=False,
    )
    print(monee.run_energy_flow(net, solver="scip"))

    # WHEN
    result = monee.run_energy_flow_optimization(
        net, optimization_problem=problem, solver="scip"
    )
    print(result)

    # THEN
    assert result.success
    assert result is not None
    assert result.objective is not None

    # No incident, so all regulations should be ~1 (no shedding)
    regs = _all_regulations(result.network)
    assert len(regs) > 0, "Expected at least one component with regulation"
    for reg in regs:
        assert reg > 0.99, f"regulation {reg} is too low - unexpected shedding"


def _closed_heat_loop(pin_temperature):
    """Closed loop with the reference on the return junction R (the documented
    plant-reference pattern): a 0.2 MW fixed-flow producer R -> S feeds a
    0.5 MW passive consumer S2 -> R, so 0.3 MW of heat is missing."""
    import monee.express as mx
    import monee.model as mm

    net = mm.Network()
    r, s, s2 = (mx.create_water_junction(net) for _ in range(3))
    mx.create_water_ext_grid(net, r, t_k=330, pin_temperature=pin_temperature)
    mx.create_heat_exchanger(net, r, s, q_mw=-0.2)
    mx.create_water_pipe(net, s, s2, diameter_m=0.15, length_m=100)
    mx.create_passive_heat_exchanger(net, s2, r, q_mw=0.5, diameter_m=0.15)
    return net, r


def _heat_shed(net, **problem_kwargs):
    from monee.problem import calc_general_resilience_performance

    result = monee.run_energy_flow_optimization(
        net,
        create_min_load_shedding_problem(
            check_vm=False, check_pressure=False, bounds_t=(0.5, 1.5), **problem_kwargs
        ),
        solver="scip",
    )
    assert result.success
    return calc_general_resilience_performance(result.network)[1]


def test_a_pinned_closed_loop_reference_is_an_unlimited_heat_source():
    """Documented backup-plant behaviour: the pinned reference re-heats the
    return at zero exchanged mass flow, so the kg/s cap does not limit it."""
    net, _ = _closed_heat_loop(pin_temperature=True)

    assert abs(_heat_shed(net, bounds_ext_heat=(-1e-3, 1e-3))) < 1e-3


def test_bounds_ext_heat_mw_caps_the_heat_of_a_pinned_reference():
    """Regression: load shedding had no heat cap in MW for the water slack (only
    the economic dispatch had ``bounds_ext_heat_mw``)."""
    net, _ = _closed_heat_loop(pin_temperature=True)

    assert abs(_heat_shed(net, bounds_ext_heat_mw=(0.0, 0.0)) - 0.3) < 0.01


def test_an_unpinned_reference_is_no_heat_source():
    """Regression: the heat balance of the reference node was dropped even with
    ``pin_temperature=False``, leaving the node an unlimited heat source/sink at
    a free temperature; the docs promise the temperature then follows the
    energy balance."""
    net, _ = _closed_heat_loop(pin_temperature=False)

    assert abs(_heat_shed(net) - 0.3) < 0.01


def test_only_a_temperature_pinned_reference_drops_its_heat_balance():
    from monee.solver.core import apply_child_overwrites, mark_heat_balance_slacks

    for pin in (True, False):
        net, r = _closed_heat_loop(pin_temperature=pin)
        apply_child_overwrites(net, net.nodes, set())

        mark_heat_balance_slacks(net, set())

        dropped = [
            n.id for n in net.nodes if getattr(n.model, "_drop_heat_balance", False)
        ]
        assert dropped == ([r] if pin else [])


def _el_feeder(regulation):
    import monee.express as mx

    net = mx.create_multi_energy_network()
    b0, b1 = mx.create_bus(net), mx.create_bus(net)
    mx.create_line(net, b0, b1, 300, 7e-5, 7e-5)
    mx.create_ext_power_grid(net, b0)
    load = mx.create_power_load(net, b1, p_mw=0.2, q_mvar=0.0)
    net.child_by_id(load).model.regulation = regulation
    return net


def test_replacing_a_part_load_regulation_warns():
    """Regression (usage pitfall): the shedding variable replaced a demand's
    input regulation (a documented part-load control) silently, so the case ran
    every hour at the design demand."""
    import pytest

    problem = create_min_load_shedding_problem(check_pressure=False, check_t=False)

    with pytest.warns(UserWarning, match="replaces the regulation of 1 demand"):
        monee.run_energy_flow_optimization(_el_feeder(0.5), problem, solver="scip")


def test_a_full_regulation_does_not_warn():
    import warnings

    problem = create_min_load_shedding_problem(check_pressure=False, check_t=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        monee.run_energy_flow_optimization(_el_feeder(1.0), problem, solver="scip")

    assert not [w for w in caught if "replaces the regulation" in str(w.message)]

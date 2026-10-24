"""Post-solve restoration of the relaxed (convex) Weymouth epigraph.

Under an economic objective nothing prices gas pressures, so the epigraph of
``gas_convex_miqcqp`` stays slack: mesh flows split arbitrarily and pressure
drops come out too large. ``restore_exact_weymouth`` replaces that state by the
exact Weymouth state of the solved injections.
"""

import pytest

import monee
import monee.express as mx
import monee.model as mm
from monee.model.grid import STANDARD_ATMOSPHERE_PA
from monee.problem import create_economic_dispatch_problem
from monee.simulation import run_multi_period
from monee.solver.pyo import _pyscipopt_available


def _gurobipy_available() -> bool:
    try:
        import gurobipy  # noqa: F401

        from monee.solver.gurobipy import GurobipySolver  # noqa: F401

        return True
    except Exception:
        return False


SOLVERS = [
    pytest.param(
        "scip",
        marks=pytest.mark.skipif(
            not _pyscipopt_available(), reason="pyscipopt missing"
        ),
    ),
    pytest.param(
        "gurobi",
        marks=pytest.mark.skipif(not _gurobipy_available(), reason="gurobipy missing"),
    ),
]


def _mesh(ambient_pa=0.0):
    grid = mm.create_gas_grid("gas", type="lgas")
    grid.pressure_ambient_pa = ambient_pa
    net = mm.Network(gas_model=grid)
    j = [mx.create_gas_junction(net) for _ in range(4)]
    mx.create_gas_ext_grid(net, j[0])
    for a, b, d, length in [
        (0, 1, 0.15, 400),
        (1, 2, 0.1, 800),
        (2, 3, 0.1, 600),
        (3, 0, 0.1, 1200),
        (1, 3, 0.08, 500),
    ]:
        mx.create_gas_pipe(net, j[a], j[b], diameter_m=d, length_m=length)
    return net, j


def _meshed_gas_net(ambient_pa=0.0):
    net, j = _mesh(ambient_pa)
    mx.create_gas_sink(net, j[2], mass_flow_kgs=0.3)
    mx.create_gas_sink(net, j[3], mass_flow_kgs=0.1)
    mx.create_gas_source(net, j[1], mass_flow_kgs=0.05, cost=50.0)
    return net


def _two_reference_net():
    # Equal-cost city gates: the relaxation splits the supply between them at
    # will, while exact Weymouth fixes the split through the pressures.
    net, j = _mesh()
    mx.create_gas_ext_grid(net, j[2])
    mx.create_gas_sink(net, j[1], mass_flow_kgs=0.3)
    mx.create_gas_sink(net, j[3], mass_flow_kgs=0.2)
    return net


def _dispatch(net, solver, formulation):
    result = monee.run_energy_flow_optimization(
        net,
        create_economic_dispatch_problem(gas_cost_default=30.0),
        solver=solver,
        formulation=formulation,
    )
    assert result.success
    return result


def _gas_state(result):
    pipes = result.get(mm.GasPipe)["mass_flow_kgs"].to_list()
    pressures = result.get(mm.Junction)["pressure_pu"].to_list()
    return pipes, pressures


@pytest.mark.parametrize("ambient_pa", [0.0, STANDARD_ATMOSPHERE_PA])
@pytest.mark.parametrize("solver", SOLVERS)
def test_convex_dispatch_returns_the_exact_gas_state(solver, ambient_pa):
    relaxed = _dispatch(_meshed_gas_net(ambient_pa), solver, "gas_convex_miqcqp")
    exact = _dispatch(_meshed_gas_net(ambient_pa), solver, "gas_nonconvex_miqcqp")

    flows, pressures = _gas_state(relaxed)
    exact_flows, exact_pressures = _gas_state(exact)
    assert flows == pytest.approx(exact_flows, abs=1e-4)
    assert pressures == pytest.approx(exact_pressures, abs=1e-5)
    assert relaxed.user_objective == pytest.approx(exact.user_objective, rel=1e-6)
    assert not [w for w in relaxed.warnings if w.category == "relaxation"]
    for branch in relaxed.network.branches:
        assert branch.formulation.relaxation_gap(branch.model) < 1e-9


def test_casadi_multi_period_dispatch_restores_the_exact_state_in_every_period():
    result = run_multi_period(
        _meshed_gas_net(),
        steps=2,
        optimization_problem=create_economic_dispatch_problem(gas_cost_default=30.0),
        solver="ipopt",
        formulation="gas_convex_miqcqp",
    )

    assert result.success
    for t in range(2):
        for branch in result.get_period_result(t).network.branches:
            assert branch.formulation.relaxation_gap(branch.model) < 1e-9
    assert not result.warnings


@pytest.mark.skipif(not _pyscipopt_available(), reason="pyscipopt missing")
def test_multi_period_reports_an_island_left_relaxed_in_every_period():
    result = run_multi_period(
        _two_reference_net(),
        steps=2,
        optimization_problem=create_economic_dispatch_problem(gas_cost_default=30.0),
        solver="scip",
        formulation="gas_convex_miqcqp",
    )

    assert result.success
    for t in range(2):
        entries = result.get_period_result(t).warnings
        assert [w.category for w in entries] == ["relaxation"]
        assert "gas_nonconvex_miqcqp" in entries[0].message


def test_gekko_dispatch_returns_the_exact_gas_state():
    """Regression: GEKKO keeps the default ``pressure_pu=1`` an int, which made
    the restored pressures an int array truncated to 0, so the island stayed
    relaxed without a word."""
    result = _dispatch(_meshed_gas_net(), "apopt", "gas_convex_miqcqp")

    for branch in result.network.branches:
        assert branch.formulation.relaxation_gap(branch.model) < 1e-9


@pytest.mark.parametrize("solver", SOLVERS)
def test_island_with_two_pressure_references_is_left_relaxed_and_reported(solver):
    result = _dispatch(_two_reference_net(), solver, "gas_convex_miqcqp")

    entries = [w for w in result.warnings if w.category == "relaxation"]
    assert len(entries) == 1
    assert "gas_nonconvex_miqcqp" in entries[0].message


@pytest.mark.skipif(not _pyscipopt_available(), reason="pyscipopt missing")
def test_multi_period_dispatch_restores_the_exact_state_in_every_period():
    single = _dispatch(_meshed_gas_net(), "scip", "gas_convex_miqcqp")
    result = run_multi_period(
        _meshed_gas_net(),
        steps=2,
        optimization_problem=create_economic_dispatch_problem(gas_cost_default=30.0),
        solver="scip",
        formulation="gas_convex_miqcqp",
    )

    assert result.success
    expected = single.get(mm.Junction, index="id")["pressure_pu"]
    pressures = result.get_result_for(mm.Junction, "pressure_pu")
    for t in range(2):
        period = pressures.iloc[t]
        assert period[expected.index].to_list() == pytest.approx(
            expected.to_list(), abs=1e-5
        )
        network = result.get_period_result(t).network
        for branch in network.branches:
            assert branch.formulation.relaxation_gap(branch.model) < 1e-9

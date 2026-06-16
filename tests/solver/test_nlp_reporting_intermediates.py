"""Validates that the AC-NLP current/loading reporting quantities are passive
intermediates (not materialised as decision variables), yet are still reported
correctly and still enforceable as a line-loading limit.

Background: ``i_from_ka`` / ``i_to_ka`` / ``loading_from_pu`` / ``loading_to_pu``
used to be free Vars pinned by an equality constraint (4 vars + 4 constraints per
branch). They are now ``Intermediate``s bound by ``IntermediateEq`` (mirroring the
MISOCP formulation), so they add nothing to the model unless a constraint
references them - in which case the defining expression inlines into it.
"""

import math

import pytest

import monee.model as mm
from monee import run_energy_flow, run_energy_flow_optimization
from monee.model import Network
from monee.model.child import ExtPowerGrid, PowerGenerator, PowerLoad
from monee.model.formulation import EL_NLP_FORMULATION
from monee.model.node import Bus
from monee.problem import create_economic_dispatch_problem
from monee.solver import GEKKOSolver


def _casadi_solver_cls():
    try:
        from monee.solver.casadi import CasADiSolver
    except ImportError:  # pragma: no cover - casadi optional
        return None
    return CasADiSolver


_BACKENDS = [("GEKKO", GEKKOSolver)]
if _casadi_solver_cls() is not None:
    _BACKENDS.append(("CasADi", _casadi_solver_cls()))


def _two_bus_net(max_i_ka=0.12, gen_mw=1.0):
    """ext-grid at b0, a 2 MW load + a dispatchable generator at b1, one line."""
    net = Network(mm.PowerGrid(name="p", sn_mva=1))
    b0 = net.node(
        Bus(base_kv=20),
        grid=mm.EL,
        child_ids=[net.child(ExtPowerGrid(p_mw=0, q_mvar=0, vm_pu=1, va_degree=0))],
    )
    lid = net.child(PowerLoad(p_mw=2.0, q_mvar=0.2))
    gid = net.child(PowerGenerator(p_mw=gen_mw, q_mvar=0))
    b1 = net.node(Bus(base_kv=20), grid=mm.EL, child_ids=[lid, gid])
    net.branch(
        mm.PowerLine(
            length_m=2000, r_ohm_per_m=1.5e-4, x_ohm_per_m=2e-4,
            parallel=1, max_i_ka=max_i_ka,
        ),
        b0,
        b1,
    )
    net.apply_formulation(EL_NLP_FORMULATION)
    return net, gid


def _branch_df(result):
    for key in ("PowerLine", "GenericPowerBranch"):
        if key in result.dataframes:
            return result.dataframes[key]
    raise AssertionError("no branch dataframe in result")


@pytest.mark.parametrize("name,solver_cls", _BACKENDS)
def test_current_and_loading_reported_and_match_formula(name, solver_cls):
    """Plain power flow: the reporting intermediates are present, finite, and
    equal the analytic current magnitude / loading from the solved p/q/vm."""
    net, _ = _two_bus_net()
    result = run_energy_flow(net, solver=solver_cls(), simulation=True)
    assert result.success

    bdf = _branch_df(result)
    row = bdf.iloc[0]
    p = float(row["p_from_mw"])
    q = float(row["q_from_mvar"])
    max_i = float(row["max_i_ka"])
    vm0 = float(result.dataframes["Bus"].iloc[0]["vm_pu"])

    eps = 1e-4  # CURRENT_SMOOTHING_EPS_MW
    i_expected = (p**2 + q**2 + eps**2) ** 0.5 / (vm0 * 20.0) / math.sqrt(3)

    assert math.isfinite(float(row["i_from_ka"]))
    assert math.isfinite(float(row["loading_from_pu"]))
    assert math.isclose(float(row["i_from_ka"]), i_expected, rel_tol=1e-4)
    assert math.isclose(
        float(row["loading_from_pu"]), float(row["i_from_ka"]) / max_i, rel_tol=1e-6
    )


@pytest.mark.skipif(
    _casadi_solver_cls() is None, reason="casadi backend not installed"
)
def test_reporting_quantities_are_not_decision_variables():
    """The current/loading reporting quantities must not enter the NLP as free
    variables (they are passive intermediates now)."""
    CasADiSolver = _casadi_solver_cls()
    net, _ = _two_bus_net()
    solver = CasADiSolver()
    solver.solve(net, simulation=True)
    reporting = [
        e for e in solver._reg
        if any(t in str(e["name"]) for t in ("i_from_ka", "i_to_ka", "loading"))
    ]
    assert reporting == [], f"reporting quantities materialised as vars: {reporting}"


@pytest.mark.parametrize("name,solver_cls", _BACKENDS)
def test_line_loading_limit_is_enforced(name, solver_cls):
    """A line-loading limit (economic dispatch, check_lp) must bind even though
    ``loading_*`` are intermediates - the limit constraint inlines the defining
    expression. A cap below the natural loading must drive the loading down to
    the cap by redispatching the local generator."""
    # Unconstrained: establish the natural loading.
    net, _ = _two_bus_net()
    free = run_energy_flow_optimization(
        net,
        create_economic_dispatch_problem(check_vm=False, check_lp=False),
        solver=solver_cls(),
    )
    natural = float(_branch_df(free).iloc[0]["loading_from_pu"])
    assert natural > 0.4  # the test net genuinely loads the line

    cap = 0.3
    assert cap < natural  # the cap must be binding
    net2, _ = _two_bus_net()
    limited = run_energy_flow_optimization(
        net2,
        create_economic_dispatch_problem(
            check_vm=False, check_lp=True, bounds_lp=(0, cap)
        ),
        solver=solver_cls(),
    )
    assert limited.success
    bdf = _branch_df(limited)
    assert float(bdf.iloc[0]["loading_from_pu"]) <= cap + 1e-3
    assert float(bdf.iloc[0]["loading_to_pu"]) <= cap + 1e-3

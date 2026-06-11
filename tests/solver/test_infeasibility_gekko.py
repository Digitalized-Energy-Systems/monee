"""
Tests for GEKKO/APMonitor (IPOPT) infeasibility diagnostic tools.
"""

import pytest

from monee.solver.infeasibility.apm import (
    GekkoInfeasibilityReport,
    GekkoSolveError,
    diagnose_gekko_infeasibility,
    parse_infeasibilities,
    sanitize_apm_name,
)

# Captured from a real APMonitor run (note the "INFEASBILE" typo is theirs).
_SAMPLE_INFEASIBILITIES = """\
************************************************
***** POSSIBLE INFEASBILE EQUATIONS ************
************************************************
____________________________________________________________________________
EQ Number   Lower        Residual     Upper        Infeas.     Name
         1   0.0000E+00  -3.2415E+00   0.0000E+00   3.2415E+00  ss.Eqn(1): 0 = (powergenerator_5_p_mw+node_4_q_mw)-(5)
 Variable   Lower        Value        Upper        $Value      Name
         1   0.0000E+00   1.0000E+00   1.0000E+00   0.0000E+00  ss.powergenerator_5_p_mw
         2   0.0000E+00   7.5846E-01   1.0000E+00   0.0000E+00  ss.node_4_q_mw
____________________________________________________________________________
EQ Number   Lower        Residual     Upper        Infeas.     Name
         2   0.0000E+00  -2.5846E-01   0.0000E+00   2.5846E-01  ss.Eqn(2): 0 = (powergenerator_5_p_mw-node_4_q_mw)-(0.5)
 Variable   Lower        Value        Upper        $Value      Name
         1   0.0000E+00   1.0000E+00   1.0000E+00   0.0000E+00  ss.powergenerator_5_p_mw
         2   0.0000E+00   7.5846E-01   1.0000E+00   0.0000E+00  ss.node_4_q_mw
************************************************
****** ACTIVE OBJECTIVE EQUATIONS **************
************************************************
Number           ID  Node    Horizon  Unscaled Res  Scaled Res   Scaling     Name
         1          4    1          1   7.5846E-01   7.5846E-01   1.0000E+00  ss.Eqn(4): 0 = v3
************************************************
************* ACTIVE EQUATIONS *****************
************************************************
Number           ID  Node    Horizon  Unscaled Res  Scaled Res   Scaling     Name
         1          1    1          1  -3.2415E+00  -3.2415E+00   1.0000E+00  ss.Eqn(1): 0 = (powergenerator_5_p_mw+node_4_q_mw)-(5)
"""

_NAME_MAP = {
    "powergenerator_5_p_mw": "powergenerator-5.p_mw",
    "node_4_q_mw": "node-4.q_mw",
}


def test_sanitize_apm_name():
    assert sanitize_apm_name("powergenerator-5.p_mw") == "powergenerator_5_p_mw"
    assert (
        sanitize_apm_name("genericpowerbranch-(0, 1, 0).p_from_mw")
        == "genericpowerbranch_0_1_0_p_from_mw"
    )


def test_parse_infeasibilities():
    equations = parse_infeasibilities(_SAMPLE_INFEASIBILITIES)
    # Only the two equations from the infeasible section, not the active ones.
    assert len(equations) == 2
    # Sorted by infeasibility magnitude (descending).
    assert equations[0].infeasibility == pytest.approx(3.2415)
    assert equations[0].residual == pytest.approx(-3.2415)
    assert equations[0].number == 1
    assert "powergenerator_5_p_mw" in equations[0].equation
    assert len(equations[0].variables) == 2
    var = equations[0].variables[0]
    assert var.name == "powergenerator_5_p_mw"
    assert var.value == pytest.approx(1.0)
    assert var.lower == pytest.approx(0.0)
    assert var.upper == pytest.approx(1.0)


def test_parse_infeasibilities_with_name_map():
    equations = parse_infeasibilities(_SAMPLE_INFEASIBILITIES, name_map=_NAME_MAP)
    assert "powergenerator-5.p_mw" in equations[0].equation
    assert "node-4.q_mw" in equations[0].equation
    assert equations[0].variables[0].display_name == "powergenerator-5.p_mw"


def test_report_summary():
    equations = parse_infeasibilities(_SAMPLE_INFEASIBILITIES, name_map=_NAME_MAP)
    report = GekkoInfeasibilityReport(
        solver_message="@error: Solution Not Found",
        infeasible_equations=equations,
    )
    summary = report.summary()
    assert "Solution Not Found" in summary
    assert "Possibly infeasible equations (2 total)" in summary
    assert "powergenerator-5.p_mw" in summary
    assert isinstance(repr(report), str)


def test_diagnose_failed_gekko_solve():
    """Diagnosis of a raw GEKKO model with conflicting equations."""
    from gekko import GEKKO

    m = GEKKO(remote=False)
    m.options.SOLVER = 3  # IPOPT
    m.options.IMODE = 3
    x = m.Var(1.0, lb=0, ub=1, name="powergenerator-5.p_mw")
    y = m.Var(1.0, lb=0, ub=1, name="node-4.q_mw")
    m.Equation(x + y == 5)  # impossible with x, y <= 1
    with pytest.raises(Exception):
        m.solve(disp=False)

    report = diagnose_gekko_infeasibility(
        m,
        name_map=_NAME_MAP,
        solver_message="@error: Solution Not Found",
    )
    assert report is not None
    assert len(report.infeasible_equations) >= 1
    summary = report.summary()
    assert "powergenerator-5.p_mw" in summary


def test_infeasible_monee_gekko_solve():
    """An infeasible monee problem solved with GEKKO/IPOPT raises a
    GekkoSolveError carrying the diagnostic report instead of the bare
    'Solution Not Found'."""
    import monee.model as mm
    from monee.model import Network
    from monee.model.child import ExtPowerGrid, PowerLoad
    from monee.model.node import Bus
    from monee.problem.core import Constraints, OptimizationProblem
    from monee.solver.gekko import GEKKOSolver

    _LINE = dict(length_m=100, r_ohm_per_m=1e-4, x_ohm_per_m=1e-4, parallel=1)
    net = Network(mm.PowerGrid(name="el", sn_mva=1))
    ext_id = net.child(ExtPowerGrid(p_mw=0, q_mvar=0, vm_pu=1.0, va_degree=0.0))
    b0 = net.node(Bus(base_kv=1), grid=mm.EL, child_ids=[ext_id])
    load_id = net.child(PowerLoad(p_mw=2.0, q_mvar=0.0))
    b1 = net.node(Bus(base_kv=1), grid=mm.EL, child_ids=[load_id])
    net.branch(mm.PowerLine(**_LINE), b0, b1)

    # Infeasible: ext-grid must cover the 2 MW load but is constrained
    # to [0.1, 0.5] MW.
    prob = OptimizationProblem()
    cons = Constraints()
    cons.select_types(ExtPowerGrid).equation(lambda m: m.p_mw >= 0.1).equation(
        lambda m: m.p_mw <= 0.5
    )
    prob.constraints = cons

    solver = GEKKOSolver(solver=3)  # IPOPT
    with pytest.raises(GekkoSolveError, match="Diagnostic report") as exc_info:
        solver.solve(net, optimization_problem=prob)
    assert isinstance(exc_info.value.report, GekkoInfeasibilityReport)

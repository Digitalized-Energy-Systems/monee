"""``SolverResult.backend_used`` / ``solver_used`` and the matching
``SolverInterface.backend_name`` / ``solver_name`` identity contract.

The dispatch-level checks construct solvers (cheap, no actual solve and no
solver licence) and assert the convention-resolved identity. One end-to-end
check runs the default solve and asserts the identity is stamped on the result.
"""

import pytest

import monee
from monee.solver.core import SolverResult
from monee.solver.dispatch import _casadi_available, resolve_solver

from ..util import create_water_loop


def test_default_resolves_to_ipopt_on_casadi_or_gekko():
    solver = resolve_solver()
    assert solver.solver_name == "ipopt"
    expected = "casadi" if _casadi_available() else "gekko"
    assert solver.backend_name == expected


def test_explicit_backends_report_their_identity():
    assert resolve_solver(backend="pyomo", solver="scip").backend_name == "pyomo"
    assert resolve_solver(backend="pyomo", solver="scip").solver_name == "scip"

    gurobipy = resolve_solver(backend="gurobipy")
    assert (gurobipy.backend_name, gurobipy.solver_name) == ("gurobipy", "gurobi")

    gekko = resolve_solver(backend="gekko", solver="apopt")
    assert (gekko.backend_name, gekko.solver_name) == ("gekko", "apopt")


def test_casadi_backend_identity():
    pytest.importorskip("casadi")
    solver = resolve_solver(solver="ipopt", backend="casadi")
    assert (solver.backend_name, solver.solver_name) == ("casadi", "ipopt")


def test_solver_result_identity_defaults_to_none():
    # Back-compat: the new fields are optional trailing fields.
    result = SolverResult(network=None, dataframes={}, objective=0.0, success=True)
    assert result.backend_used is None
    assert result.solver_used is None


def test_default_solve_stamps_the_result():
    net, *_ = create_water_loop(source_t_k=356)
    result = monee.solve(net)
    expected_backend = "casadi" if _casadi_available() else "gekko"
    assert result.backend_used == expected_backend
    assert result.solver_used == "ipopt"

"""The solver-injected ``simulation`` keyword reaches node and child models.

docs/source/concepts/formulations.rst documents ``simulation`` as one of the
keyword arguments the back-end injects into ``equations``; before this it only
reached branches, and child models were called with no keyword arguments at
all.
"""

import monee.express as mx
import monee.model as mm
import monee.solver as ms
from monee import run_energy_flow

SEEN: dict[str, list] = {"child": [], "node": []}


@mm.model
class ProbeLoad(mm.PowerLoad):
    def equations(self, grid, node_model, **kwargs):
        SEEN["child"].append(kwargs.get("simulation", "missing"))
        return super().equations(grid, node_model, **kwargs)


@mm.model
class ProbeBus(mm.Bus):
    def equations(self, grid, from_branch_models, to_branch_models, childs, **kwargs):
        SEEN["node"].append(kwargs.get("simulation", "missing"))
        return super().equations(
            grid, from_branch_models, to_branch_models, childs, **kwargs
        )


def _probe_network():
    net = mx.create_multi_energy_network()
    slack = mx.create_bus(net)
    mx.create_ext_power_grid(net, slack)
    bus = net.node(ProbeBus(base_kv=1), grid=mm.EL)
    mx.create_line(net, slack, bus, length_m=200, r_ohm_per_m=1e-4, x_ohm_per_m=1e-4)
    net.child_to(ProbeLoad(p_mw=0.5, q_mvar=0.0), bus)
    return net


def _seen(run):
    SEEN["child"].clear()
    SEEN["node"].clear()
    run()
    return set(SEEN["child"]), set(SEEN["node"])


def test_simulation_kwarg_reaches_node_and_child_on_casadi():
    # WHEN
    child, node = _seen(lambda: run_energy_flow(_probe_network()))

    # THEN
    assert child == {True}
    assert node == {True}


def test_simulation_kwarg_is_false_in_optimization_mode():
    # WHEN
    child, node = _seen(
        lambda: ms.CasADiSolver().solve(_probe_network(), simulation=False)
    )

    # THEN
    assert child == {False}
    assert node == {False}


def test_simulation_kwarg_reaches_node_and_child_on_pyomo():
    # WHEN
    child, node = _seen(
        lambda: ms.PyomoSolver().solve(
            _probe_network(), solver_name="scip", simulation=True
        )
    )

    # THEN
    assert child == {True}
    assert node == {True}


def test_simulation_kwarg_reaches_node_and_child_on_gekko():
    # WHEN
    child, node = _seen(
        lambda: ms.GEKKOSolver().solve(_probe_network(), simulation=True)
    )

    # THEN
    assert child == {True}
    assert node == {True}

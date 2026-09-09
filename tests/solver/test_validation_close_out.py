"""Close-out round: validation-layer corrections (batch reg-validate)."""

import pandas
import pytest

import monee as mn
import monee.express as mx
import monee.model as mm
from monee import run_energy_flow
from monee.problem.min_load_shedding import (
    _shed_headroom_validator,
    create_min_load_shedding_problem,
)
from monee.solver.core import (
    REGULATION_NOISE_FLOOR,
    SolverResult,
    uncovered_free_var_entry,
)


def _islanded_net():
    net = mx.create_multi_energy_network()
    buses = [mx.create_bus(net, base_kv=20) for _ in range(6)]
    for i in range(5):
        mx.create_line(
            net,
            buses[i],
            buses[i + 1],
            length_m=1000,
            r_ohm_per_m=1e-4,
            x_ohm_per_m=1e-4,
        )
    mx.create_ext_power_grid(net, buses[0])
    mx.create_grid_forming_generator(net, buses[3], p_mw_max=2.0, q_mvar_max=2.0)
    for bus, p_mw in ((buses[2], 1.3), (buses[4], 1.0), (buses[5], 1.0)):
        mx.create_power_load(net, bus, p_mw=p_mw, q_mvar=0.0)
    net.deactivate(net.branch_by_id((buses[0], buses[1], 0)))
    return net


def test_shed_headroom_silent_when_slack_is_in_another_island():
    net = _islanded_net()
    mn.enable_islanding(net, electricity=True)
    result = mn.solve(net, optimization_problem=create_min_load_shedding_problem())
    assert [w for w in result.warnings if w.category == "shed_headroom"] == []


def test_shed_headroom_still_fires_within_one_island():
    net = _islanded_net()
    net.activate(net.branch_by_id((0, 1, 0)))
    for child in net.childs:
        if isinstance(child.model, mm.PowerLoad):
            child.model.regulation = 0.5
    entries = _shed_headroom_validator(net, None)
    assert len(entries) == 1
    assert "in the same island" in entries[0].message


def test_shed_headroom_advice_never_names_the_solver_in_use():
    net = _islanded_net()
    net.activate(net.branch_by_id((0, 1, 0)))
    for child in net.childs:
        if isinstance(child.model, mm.PowerLoad):
            child.model.regulation = 0.5
    result = SolverResult(net, {}, 0.0, True, solver_used="scip")
    message = _shed_headroom_validator(net, result)[0].message
    assert "solver='scip'" not in message
    assert "solver='gurobi'" in message


def test_regulation_noise_is_clamped_in_result_frames_only():
    load = mm.PowerLoad(p_mw=1.0, q_mvar=0.0)
    load.regulation = -0.5 * REGULATION_NOISE_FLOOR
    frames = {
        "PowerLoad": pandas.DataFrame(
            {"regulation": [-0.5 * REGULATION_NOISE_FLOOR, 1.0 + 1e-9, 0.5]}
        )
    }
    result = SolverResult(None, frames, 0.0, True)
    assert list(result.dataframes["PowerLoad"]["regulation"]) == [0.0, 1.0, 0.5]
    assert load.regulation == -0.5 * REGULATION_NOISE_FLOOR


def test_regulation_beyond_the_noise_floor_is_left_alone():
    frames = {"PowerLoad": pandas.DataFrame({"regulation": [-0.01, 1.01]})}
    result = SolverResult(None, frames, 0.0, True)
    assert list(result.dataframes["PowerLoad"]["regulation"]) == [-0.01, 1.01]


@mm.model
class _FreeFlex(mm.ChildModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.p_mw = mm.Var(1.0, min=0.5, max=3.0, name="p_mw")
        self.q_mvar = 0.0

    def equations(self, grid, node_model, **kwargs):
        return []


def _free_var_net(with_flex=True):
    net = mx.create_multi_energy_network()
    b1 = mx.create_bus(net)
    b2 = mx.create_bus(net)
    mx.create_line(net, b1, b2, length_m=800, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)
    mx.create_ext_power_grid(net, b1)
    mx.create_power_load(net, b2, p_mw=1.0, q_mvar=0.0)
    if with_flex:
        mx.create_el_child(net, _FreeFlex(), node_id=b2)
    return net


def test_optimization_mode_flags_free_var_no_objective_covers():
    result = run_energy_flow(_free_var_net(), simulation=False)
    dof = [w for w in result.warnings if w.category == "dof"]
    assert len(dof) == 1
    assert "optimization mode without an objective" in dof[0].message
    assert dof[0].value == 1.0


def test_optimization_mode_stays_silent_on_a_square_model():
    result = run_energy_flow(_free_var_net(with_flex=False), simulation=False)
    assert [w for w in result.warnings if w.category == "dof"] == []


def test_uncovered_free_var_entry_silent_when_an_objective_exists():
    model = pytest.importorskip("monee.solver.casadi").CasModel()
    model.obj_terms.append(1)
    assert uncovered_free_var_entry([{"model": object(), "key": "x"}], model) is None

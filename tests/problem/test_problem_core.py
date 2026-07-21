"""Unit tests for controllable selection, bounds validation and the vm-bounds
hook of the problem layer."""

import pytest

import monee.express as mx
import monee.model as mm
from monee.model.core import Intermediate, Var
from monee.model.formulation import EL_MISOCP_FORMULATION
from monee.model.formulation.registry import attach_formulations
from monee.problem.core import REGULATION_ATTR, Constraints, OptimizationProblem
from monee.problem.economic_dispatch import create_economic_dispatch_problem
from monee.problem.min_load_shedding import create_min_load_shedding_problem


def _hx_pair_net():
    """Water net with one consuming and one generating bare HeatExchanger."""
    net = mx.create_multi_energy_network()
    j1 = mx.create_water_junction(net)
    j2 = mx.create_water_junction(net)
    j3 = mx.create_water_junction(net)
    # HeatExchanger stores q_mw_set = -q_mw: q_mw=-0.1 -> q_mw_set=+0.1 (load),
    # q_mw=+0.1 -> q_mw_set=-0.1 (generator).
    load_id = net.branch(
        mm.HeatExchanger(q_mw=-0.1), j1, j2, grid=mm.WATER, name="hx_load"
    )
    gen_id = net.branch(
        mm.HeatExchanger(q_mw=0.1), j2, j3, grid=mm.WATER, name="hx_gen"
    )
    return net, net.branch_by_id(load_id).model, net.branch_by_id(gen_id).model


def test_controllable_demands_selects_consuming_hx_only():
    net, hx_load, hx_gen = _hx_pair_net()
    problem = OptimizationProblem()
    problem.controllable_demands(REGULATION_ATTR)
    problem._apply(net)

    selected = set(problem._controllable_to_attr.keys())
    assert hx_load in selected
    assert hx_gen not in selected


def test_controllable_generators_selects_generating_hx_only():
    net, hx_load, hx_gen = _hx_pair_net()
    problem = OptimizationProblem()
    problem.controllable_generators(REGULATION_ATTR)
    problem._apply(net)

    selected = set(problem._controllable_to_attr.keys())
    assert hx_gen in selected
    assert hx_load not in selected


def test_bounds_requires_attributes():
    problem = OptimizationProblem()
    with pytest.raises(ValueError, match="non-empty list of attribute names"):
        problem.bounds((0.9, 1.1))
    with pytest.raises(ValueError, match="non-empty list of attribute names"):
        problem.bounds((0.9, 1.1), attributes=[])


def _small_power_net():
    net = mx.create_multi_energy_network()
    b0 = mx.create_bus(net, base_kv=20)
    b1 = mx.create_bus(net, base_kv=20)
    mx.create_ext_power_grid(net, b0, p_mw=0, q_mvar=0, vm_pu=1.0)
    mx.create_power_load(net, b1, p_mw=1, q_mvar=0.1)
    mx.create_line(net, b0, b1, length_m=100, r_ohm_per_m=3e-4, x_ohm_per_m=3e-4)
    return net


def _bus_models(network):
    return [n.model for n in network.nodes if type(n.model) is mm.Bus]


@pytest.mark.parametrize(
    "make_problem",
    [
        lambda: create_min_load_shedding_problem(bounds_vm=(0.9, 1.1)),
        lambda: create_economic_dispatch_problem(bounds_vm=(0.9, 1.1)),
    ],
    ids=["min_load_shedding", "economic_dispatch"],
)
def test_check_vm_bounds_squared_var_under_misocp(make_problem):
    net = _small_power_net()
    net.apply_formulation(EL_MISOCP_FORMULATION)
    network = net.copy()
    attach_formulations(network, None)

    problem = make_problem()
    problem._apply(network)

    buses = _bus_models(network)
    assert buses
    for bus in buses:
        # MISOCP: vm_pu is only a reporting Intermediate; the decision variable
        # vm_pu_squared must carry the squared bounds.
        assert isinstance(bus.vm_pu, Intermediate)
        assert type(bus.vm_pu_squared) is Var
        assert bus.vm_pu_squared.min == pytest.approx(0.81)
        assert bus.vm_pu_squared.max == pytest.approx(1.21)


def test_check_vm_bounds_vm_pu_under_nlp():
    net = _small_power_net()
    network = net.copy()
    attach_formulations(network, None)

    problem = create_min_load_shedding_problem(bounds_vm=(0.9, 1.1))
    problem._apply(network)

    buses = _bus_models(network)
    assert buses
    for bus in buses:
        assert type(bus.vm_pu) is Var
        assert bus.vm_pu.min == pytest.approx(0.9)
        assert bus.vm_pu.max == pytest.approx(1.1)


def test_economic_dispatch_rejects_nonzero_min_line_loading():
    with pytest.raises(ValueError, match="bounds_lp"):
        create_economic_dispatch_problem(bounds_lp=(0.1, 1.0))


def test_select_grids_returns_constraint():
    constraint = Constraints().select_grids((mm.PowerGrid,))
    assert constraint is not None


def test_controllable_all_returns_self():
    problem = OptimizationProblem()
    assert problem.controllable_all(["p_mw"]) is problem


def test_controllables_link_returns_callable():
    assert callable(OptimizationProblem().controllables_link())

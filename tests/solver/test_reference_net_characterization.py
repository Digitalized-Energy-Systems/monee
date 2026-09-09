"""Solver-visible characterization of the reference networks.

These cases exist so a future change to a default (a formulation swap, a start
point, a validation rule) cannot pass a green suite while breaking what users
actually run: the mildest N-1 contingency on the urban district net must still
converge on the default CasADi/IPOPT backend, the documented DHS ring must
still solve warning-free, and a small MISOCP case must still report a tight
cone residual. Each case is a single small solve.
"""

import warnings

import monee
import monee.express as mx
import monee.model as mm
from monee import run_energy_flow
from monee.model import Network
from monee.model.branch import GenericPowerBranch
from monee.model.child import ExtPowerGrid, PowerLoad
from monee.model.formulation import EL_MISOCP_FORMULATION
from monee.model.grid import PowerGrid
from monee.model.node import Bus
from monee.network import create_urban_district_net
from monee.problem import (
    calc_general_resilience_performance,
    create_min_load_shedding_problem,
)
from monee.solver import PyomoSolver

SHEDDING_BOUNDS = dict(
    bounds_vm=(0.95, 1.05),
    bounds_pressure=(0.8, 1.2),
    bounds_t=(0.9, 1.1),
    bounds_ext_el=(-1.0, 1.0),
    bounds_ext_gas=(-0.5, 0.5),
)


def _weighted_urban_district():
    net = create_urban_district_net()
    loads = [c for c in net.childs if isinstance(c.model, mm.PowerLoad)]
    max(loads, key=lambda c: c.model.p_mw).model._priority_weight = 1e5
    [b for b in net.branches if isinstance(b.model, mm.HeatExchangerLoad)][
        0
    ].model._priority_weight = 50.0
    return net


def test_leaf_bus_contingency_converges_on_the_default_backend():
    """The mildest urban-district contingency (line 2-4 out, so the leaf bus and
    its load are pruned) must solve on defaults, not raise CasADiSolveError."""
    net = _weighted_urban_district().copy()
    net.deactivate_by_id(mm.PowerLine, (2, 4, 0))
    problem = create_min_load_shedding_problem(
        weight_for_load=lambda model: getattr(model, "_priority_weight", None),
        **SHEDDING_BOUNDS,
    )

    result = monee.run_energy_flow_optimization(
        net, problem, exclude_unconnected_nodes=True
    )

    assert result.success
    power_shed, heat_shed, gas_shed = calc_general_resilience_performance(
        result.network
    )
    assert 3.9 < power_shed < 4.1
    assert abs(heat_shed) < 1e-3
    assert abs(gas_shed) < 1e-3


def test_documented_dhs_ring_solves_warning_free_on_the_default_backend():
    net = mm.Network()
    dhs = mx.dhs_structure(net, diameter_m=0.15, length_m=100)
    segment = dhs.ring(4, heat_exchanger_q_mw=0.01)
    dhs.attach_heat_plant(segment.supply.first, segment.return_.first, name="plant")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = run_energy_flow(net)

    assert result.success
    assert result.warnings == []
    assert not [w for w in caught if "Result validation" in str(w.message)]


def _two_bus_misocp_net(br_r_pu):
    net = Network(PowerGrid(name="power", sn_mva=1.0))
    slack = net.node(
        Bus(base_kv=1.0),
        child_ids=[net.child(ExtPowerGrid(p_mw=0, q_mvar=0, vm_pu=1.0, va_degree=0))],
        grid=mm.EL,
    )
    load = net.node(
        Bus(base_kv=1.0),
        child_ids=[net.child(PowerLoad(p_mw=0.9, q_mvar=0.3))],
        grid=mm.EL,
    )
    net.branch(
        GenericPowerBranch(
            tap=1,
            shift=0,
            br_r_pu=br_r_pu,
            br_x_pu=0.092,
            g_fr_pu=0.0,
            b_fr_pu=0.0,
            g_to_pu=0.0,
            b_to_pu=0.0,
        ),
        slack,
        load,
    )
    net.apply_formulation(EL_MISOCP_FORMULATION)
    return net


def test_small_misocp_case_reports_a_tight_cone_residual():
    """A resistive branch prices its lifted current, so the SOC cone comes out
    tight and no uncertified-relaxation warning is raised."""
    net = _two_bus_misocp_net(br_r_pu=0.017)
    result = PyomoSolver().solve(net)

    assert result.success
    branch = result.network.branches[0]
    formulation = branch.formulation
    residual = formulation.uncertified_relaxation_slack(branch, result.network)
    assert residual == 0.0

    bus = result.get(mm.Bus).sort_values("id").iloc[0]
    flow = result.get(mm.GenericPowerBranch).iloc[0]
    lhs = bus["vm_pu_squared"] * flow["current_pu_squared"]
    rhs = flow["p_series_from_mw"] ** 2 + flow["q_series_from_mvar"] ** 2
    gap = (lhs - rhs) / max(abs(lhs), 1e-6)
    assert abs(gap) <= formulation.RELAXATION_GAP_TOL

    assert not [w for w in result.warnings if w.category == "relaxation"]

import pytest
import simbench

import monee
from monee import PyomoSolver, run_energy_flow
from monee.io.from_pandapower import from_pandapower_net
from monee.model.formulation import MISOCP_NETWORK_FORMULATION
from monee.network import generate_supply_return_mes_based_on_power_net
from monee.problem.min_load_shedding import create_min_load_shedding_problem


@pytest.mark.pptest
def test_generate_mes():
    # GIVEN
    net = simbench.get_simbench_net("1-LV-rural3--1-no_sw")

    # WHEN
    mn = from_pandapower_net(net)
    print(run_energy_flow(mn))

    mes = generate_supply_return_mes_based_on_power_net(
        mn,
        coupling_density=0.5,
        centralized=False,
        couplings=("chp", "p2g", "p2h"),
        coupling_kwargs={"seed": 1},
    )
    mes.apply_formulation(MISOCP_NETWORK_FORMULATION)

    print(run_energy_flow(mes, solver=PyomoSolver(), solver_name="gurobi"))

    assert False


@pytest.mark.pptest
def test_generate_mes_min_load_shedding():
    # GIVEN
    net = simbench.get_simbench_net("1-LV-rural3--1-no_sw")
    mn = from_pandapower_net(net)
    # The McCormick-DHS formulation accounts for heat injection / extraction
    # via node-level ``q_w_heat`` only (branch-based HXs are invisible to its
    # nodal energy balance), so request node-based HeatLoad children here.
    mes = generate_supply_return_mes_based_on_power_net(
        mn,
        coupling_density=0.5,
        centralized=False,
        couplings=("chp", "p2g", "p2h"),
        coupling_kwargs={"seed": 1, "use_hg_variants": False},
        heat_kwargs={"node_based_heat_loads": False},
    )
    mes.apply_formulation(MISOCP_NETWORK_FORMULATION)

    problem = create_min_load_shedding_problem(
        bounds_el=(0.5, 1.5),
        bounds_gas=(0.5, 1.5),
        bounds_heat=(0.5, 1.5),
        ext_grid_el_bounds=(-100, 100),
        ext_grid_gas_bounds=(-100, 100),
        ext_grid_heat_bounds=(-100, 100),
        include_ext_grids=True,
        include_storages=False,
    )

    # WHEN
    result = monee.run_energy_flow_optimization(
        mes,
        optimization_problem=problem,
        solver=PyomoSolver(),
        solver_name="gurobi",
    )

    # THEN
    assert result is not None
    assert result.success, "min load shedding did not converge"
    assert result.objective is not None

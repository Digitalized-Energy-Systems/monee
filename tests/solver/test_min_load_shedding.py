import monee
from monee.model.core import Var
from monee.model.formulation import MISOCP_NETWORK_FORMULATION
from monee.network import create_restoration_benchmark, create_urban_district_net
from monee.problem.min_load_shedding import create_min_load_shedding_problem
from monee.solver import PyomoSolver


def _all_regulations(network):
    """Collect all regulation values from the solved network."""
    regs = []
    for comp in network.all_components():
        m = comp.model
        if hasattr(m, "regulation"):
            reg = m.regulation
            val = reg.value if isinstance(reg, Var) else reg
            if isinstance(val, (int, float)):
                regs.append(val)
    return regs


def test_min_load_shedding_no_incident():
    """Without any incident the problem should find full supply (no shedding)."""
    net = create_urban_district_net()
    net.apply_formulation(MISOCP_NETWORK_FORMULATION)
    problem = create_min_load_shedding_problem(
        bounds_el=(0.5, 1.5),
        bounds_gas=(0.5, 1.5),
        bounds_heat=(0.5, 1.5),
        include_ext_grids=False,
        include_storages=False,
    )
    print(monee.run_energy_flow(net, solver=PyomoSolver()))
    result = monee.run_energy_flow_optimization(
        net, optimization_problem=problem, solver=PyomoSolver()
    )

    print(result)
    assert result.success
    assert result is not None
    assert result.objective is not None
    # All regulations should be ~1 (no shedding needed)
    regs = _all_regulations(result.network)
    assert len(regs) > 0, "Expected at least one component with regulation"
    for reg in regs:
        assert reg > 0.99, f"regulation {reg} is too low — unexpected shedding"


def test_min_load_shedding_with_ext_grids():
    """With ext grid bounds the problem should still solve cleanly."""
    net = create_urban_district_net()
    net.apply_formulation(MISOCP_NETWORK_FORMULATION)

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
    result = monee.run_energy_flow_optimization(
        net, optimization_problem=problem, solver=PyomoSolver()
    )

    assert result is not None
    assert result.success
    assert result.objective is not None


def test_min_load_shedding_no_incident_large():
    """Without any incident the problem should find full supply (no shedding)."""
    net = create_restoration_benchmark(misocp=True)

    problem = create_min_load_shedding_problem(
        bounds_el=(0.5, 1.5),
        bounds_gas=(0.5, 1.5),
        bounds_heat=(0.5, 1.5),
        include_ext_grids=True,
        include_storages=False,
    )
    print(monee.run_energy_flow(net, solver=PyomoSolver(), solver_name="gurobi"))

    result = monee.run_energy_flow_optimization(
        net, optimization_problem=problem, solver=PyomoSolver(), solver_name="gurobi"
    )

    print(result)
    assert result.success
    assert result is not None
    assert result.objective is not None
    # All regulations should be ~1 (no shedding needed)
    regs = _all_regulations(result.network)
    assert len(regs) > 0, "Expected at least one component with regulation"
    for reg in regs:
        assert reg > 0.99, f"regulation {reg} is too low — unexpected shedding"

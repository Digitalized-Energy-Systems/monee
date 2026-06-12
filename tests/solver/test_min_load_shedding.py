import monee
from monee.model.core import Var
from monee.model.formulation import MISOCP_NETWORK_FORMULATION
from monee.model.multi import (
    CHPControlNode,
    CHPHGControlNode,
    GasToHeatControlNode,
    GasToHeatHG,
    GasToPower,
    PowerToGas,
    PowerToHeatControlNode,
    PowerToHeatHG,
)
from monee.network import create_restoration_benchmark, create_urban_district_net
from monee.problem.min_load_shedding import create_min_load_shedding_problem
from monee.solver import PyomoSolver

# Coupling-point models: their regulation is a free decision variable when the
# problem is built with include_coupling_points=False (no objective cost), so
# cutting them is optimal dispatch (e.g. shutting a 65%-efficient P2G reduces
# net external import under the ext-grid zero-exchange slack), not shedding.
CP_TYPES = (
    CHPControlNode,
    CHPHGControlNode,
    GasToHeatControlNode,
    PowerToHeatControlNode,
    GasToPower,
    PowerToGas,
    PowerToHeatHG,
    GasToHeatHG,
)


def _all_regulations(network):
    """Collect regulation values of demand/generator components.

    Coupling points are excluded: with include_coupling_points=False their
    regulation is intentionally an unpenalised degree of freedom, so its value
    says nothing about load shedding.
    """
    regs = []
    for comp in network.all_components():
        m = comp.model
        if isinstance(m, CP_TYPES):
            continue
        if hasattr(m, "regulation"):
            reg = m.regulation
            val = reg.value if isinstance(reg, Var) else reg
            if isinstance(val, (int, float)):
                regs.append(val)
    return regs


def test_min_load_shedding_no_incident():
    # GIVEN
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

    # WHEN
    result = monee.run_energy_flow_optimization(
        net, optimization_problem=problem, solver=PyomoSolver()
    )
    print(result)

    # THEN
    assert result.success
    assert result is not None
    assert result.objective is not None

    # No incident, so all regulations should be ~1 (no shedding)
    regs = _all_regulations(result.network)
    assert len(regs) > 0, "Expected at least one component with regulation"
    for reg in regs:
        assert reg > 0.99, f"regulation {reg} is too low - unexpected shedding"


def test_min_load_shedding_with_ext_grids():
    # GIVEN
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

    # WHEN
    result = monee.run_energy_flow_optimization(
        net, optimization_problem=problem, solver=PyomoSolver()
    )

    # THEN
    assert result.success
    assert result is not None
    assert result.objective is not None


def test_min_load_shedding_no_incident_large():
    # GIVEN
    net = create_restoration_benchmark(misocp=True)
    problem = create_min_load_shedding_problem(
        bounds_el=(0.5, 1.5),
        bounds_gas=(0.5, 1.5),
        bounds_heat=(0.5, 1.5),
        include_ext_grids=True,
        include_storages=False,
    )
    print(monee.run_energy_flow(net, solver="gurobi"))

    # WHEN
    result = monee.run_energy_flow_optimization(
        net, optimization_problem=problem, solver="gurobi"
    )
    print(result)

    # THEN
    assert result.success
    assert result is not None
    assert result.objective is not None

    # No incident, so all regulations should be ~1 (no shedding)
    regs = _all_regulations(result.network)
    assert len(regs) > 0, "Expected at least one component with regulation"
    for reg in regs:
        assert reg > 0.99, f"regulation {reg} is too low - unexpected shedding"

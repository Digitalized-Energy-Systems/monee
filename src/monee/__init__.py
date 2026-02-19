import monee.model as mm
import monee.solver as ms
import monee.problem as mp
import monee.express as mx
from monee.model import Network
from monee.simulation import (
    solve,
    TimeseriesData,
    run_timeseries,
    TimeseriesResult,
    StepHook,
)


def run_energy_flow(net: mm.Network, solver=None, **kwargs):
    """
    Performs a basic energy flow analysis on a network without applying optimization constraints.

    This function provides a straightforward assessment of energy flows within a network, making it useful for initial feasibility checks, diagnostics, or scenarios where optimization is not required. Use this function when you need quick insights into network behavior or as a baseline before introducing optimization-based analyses. It fits into workflows involving network validation, troubleshooting, or preliminary studies. Internally, the function delegates to `run_energy_flow_optimization` with the optimization problem set to `None`, utilizing the same solver infrastructure but bypassing optimization logic.

    Args:
        net (mm.Network): The network to analyze, represented as an `mm.Network` instance. The network must be fully defined, including all necessary nodes and parameters.
        solver (optional): The solver to use for the energy flow computation. If not specified, a default compatible solver is chosen. The solver must support the network's structure.
        **kwargs: Additional keyword arguments for solver configuration or analysis tuning. Refer to the solver's documentation for supported options.

    Returns:
        Any: The result of the energy flow analysis, typically including calculated flow values and status information. The exact structure depends on the solver and network configuration.

    Raises:
        ValueError: If the network is incomplete, invalid, or if incompatible parameters are provided.
        SolverError: If the solver fails to compute the energy flow or encounters an error during analysis.

    Examples:
        Run a basic energy flow analysis with a specified solver:
            result = run_energy_flow(my_network, solver='default_solver')

        Run analysis with additional solver options:
            result = run_energy_flow(my_network, max_iter=500, tol=1e-5)
    """
    return run_energy_flow_optimization(net, None, solver=solver, **kwargs)


def run_energy_flow_optimization(
    net: mm.Network, optimization_problem: mp.OptimizationProblem, solver=None, **kwargs
):
    """
    Executes an energy flow optimization on a given network using a specified optimization problem and solver.

    This function determines the optimal distribution of energy flows within a network, subject to constraints and objectives defined by the provided optimization problem. Use this function when you need to solve power grid management, load balancing, or energy distribution planning tasks. It is typically integrated into simulation, planning, or real-time control workflows for energy systems. The function delegates the optimization process to a solver, which processes the network and problem definition to compute the optimal solution.

    Args:
        net (mm.Network): The network to optimize. Must be a fully specified `mm.Network` object with all nodes, edges, and parameters defined.
        optimization_problem (mp.OptimizationProblem): The optimization problem instance, specifying constraints and objectives for the energy flow.
        solver (optional): The solver to use for optimization. If None, a default compatible solver is selected. Must support the problem's formulation.
        **kwargs: Additional keyword arguments for solver configuration or optimization tuning. Refer to the solver's documentation for supported options.

    Returns:
        Any: The optimization result, which may include optimal energy flows, solution status, and additional diagnostic information. The exact structure depends on the solver and problem definition.

    Raises:
        ValueError: If the network or optimization problem is invalid, incomplete, or incompatible.
        SolverError: If the solver fails to find a feasible solution or encounters an internal error.

    Examples:
        Optimize energy flow with a custom solver:
            result = run_energy_flow_optimization(my_network, my_problem, solver='cbc')

        Use the default solver with additional options:
            result = run_energy_flow_optimization(my_network, my_problem, max_iter=1000, tol=1e-6)
    """
    return solve(net, optimization_problem, solver, **kwargs)


def solve_load_shedding_problem(
    network: Network,
    bounds_vm: tuple,
    bounds_t: tuple,
    bounds_pressure: tuple,
    bounds_ext_el: tuple,
    bounds_ext_gas: tuple,
    use_ext_grid_bounds=False,
    use_ext_grid_objective=True,
    check_lp=True,
    check_vm=True,
    check_pressure=True,
    check_t=True,
    debug=False,
    **kwargs,
):
    """
    Solves a load shedding optimization problem for a network using specified operational bounds across subsystems.

    This function is designed for scenarios where minimizing load shedding is essential, such as during network contingencies or in energy management systems. Use it when you need to enforce operational limits on voltage, temperature, and pressure for electrical, thermal, and gas subsystems, as well as external grid interfaces. The function constructs a load shedding optimization problem using the provided bounds and delegates the solution process to the energy flow optimization routine. Enabling debug mode provides additional diagnostic output for troubleshooting or analysis.

    Args:
        network (Network): The network to optimize, representing the system's topology and parameters.
        bounds_vm (tuple): Voltage magnitude bounds (min, max) for the electrical subsystem. Must be a tuple of two numeric values.
        bounds_t (tuple): Temperature bounds (min, max) for the thermal subsystem. Must be a tuple of two numeric values.
        bounds_pressure (tuple): Pressure bounds (min, max) for the gas subsystem. Must be a tuple of two numeric values.
        bounds_ext_el (tuple): External grid voltage bounds (min, max) for the electrical subsystem. Must be a tuple of two numeric values.
        bounds_ext_gas (tuple): External grid pressure bounds (min, max) for the gas subsystem. Must be a tuple of two numeric values.
        debug (bool, optional): If True, enables verbose logging and diagnostics. Defaults to False.
        **kwargs: Additional keyword arguments for solver configuration or optimization tuning. Refer to the solver documentation for supported options.

    Returns:
        Any: The result of the load shedding optimization, typically including optimized energy flows, load shedding amounts, and status information. The structure of the result depends on the solver and problem formulation.

    Raises:
        ValueError: If the network or any bounds are missing, improperly defined, or incompatible.
        SolverError: If the solver fails to find a feasible solution or encounters an error during optimization.

    Examples:
        Solve a load shedding problem with specific operational bounds::

            result = solve_load_shedding_problem(
                my_network,
                bounds_vm=(0.95, 1.05),
                bounds_t=(60, 80),
                bounds_pressure=(30, 50),
                bounds_ext_el=(0.9, 1.1),
                bounds_ext_gas=(25, 45),
                debug=True
            )

        This executes the optimization with the specified bounds and returns the results.
    """
    optimization_problem = mp.create_load_shedding_optimization_problem(
        bounds_el=bounds_vm,
        bounds_heat=bounds_t,
        bounds_gas=bounds_pressure,
        ext_grid_el_bounds=bounds_ext_el,
        ext_grid_gas_bounds=bounds_ext_gas,
        use_ext_grid_bounds=use_ext_grid_bounds,
        use_ext_grid_objective=use_ext_grid_objective,
        check_lp=check_lp,
        check_vm=check_vm,
        check_pressure=check_pressure,
        check_t=check_t,
        debug=debug,
    )
    return run_energy_flow_optimization(
        network, optimization_problem=optimization_problem, **kwargs
    )

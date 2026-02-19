import monee.model as mm


def create_bus(
    network: mm.Network,
    base_kv=1,
    constraints=None,
    grid=mm.EL,
    overwrite_id=None,
    name=None,
    position=None,
):
    """
    Adds a bus node to the specified network with configurable voltage, constraints, grid type, and metadata.

    This function is used to define and insert a new bus into a network, serving as a connection point for electrical or other grid components such as generators, loads, or lines. Use it during network construction or expansion to customize bus properties like voltage level, operational constraints, grid type, and identification details. The function integrates the new bus directly into the network, supporting both electrical and non-electrical grids.

    Args:
        network (mm.Network): The network to which the bus will be added. Must be a valid `mm.Network` instance.
        base_kv (int, optional): Base voltage level of the bus in kilovolts. Defaults to 1.
        constraints (dict or object, optional): Operational constraints for the bus (e.g., voltage or current limits). Accepts a dictionary (e.g., `{'vmin': 0.95, 'vmax': 1.05}`) or a constraints object.
        grid (Any, optional): Grid type for the bus. Defaults to `mm.EL` (electrical grid). Accepts grid constants from the `mm` module (e.g., `mm.EL`, `mm.GAS`).
        overwrite_id (Any, optional): Custom identifier to override the default bus ID. Useful for maintaining consistent IDs.
        name (str, optional): Human-readable name for the bus for easier identification.
        position (tuple or str, optional): Geographical or logical position of the bus, typically as coordinates `(x, y)` or a descriptive string.

    Returns:
        mm.Bus: The created bus node, already integrated into the network and ready for further configuration or analysis.

    Raises:
        ValueError: If the network is invalid, not initialized, or if provided parameters are incompatible.

    Examples:
        Create a bus with a base voltage of 11 kV and a custom name::

            bus = create_bus(my_network, base_kv=11, name='Main Bus')

        Add a bus with voltage constraints and a specific position::

            bus = create_bus(
                my_network,
                constraints={'vmin': 0.95, 'vmax': 1.05},
                position=(10, 20)
            )

        Create a gas grid bus with a custom ID and name::

            bus = create_bus(
                my_network,
                grid=mm.GAS,
                overwrite_id='GAS_BUS_1',
                name='Gas Supply Node'
            )
    """
    return network.node(
        mm.Bus(base_kv=base_kv),
        constraints=constraints,
        grid=grid,
        overwrite_id=overwrite_id,
        name=name,
        position=position,
    )


def create_water_junction(
    network: mm.Network,
    grid=mm.WATER,
    constraints=None,
    overwrite_id=None,
    name=None,
    position=None,
):
    """
    No docstring provided.
    """
    return create_junction(network, grid, constraints, overwrite_id, name, position)


def create_gas_junction(
    network: mm.Network,
    grid=mm.GAS,
    constraints=None,
    overwrite_id=None,
    name=None,
    position=None,
):
    """
    Creates a gas junction node in the specified network, serving as a connection point for gas components and enabling flexible network expansion.

    This function is used to define nodes where gas pipelines, compressors, or other components connect within a gas network. Use it during network construction or modification to establish the topology and facilitate the flow and distribution of gas resources. The function allows you to specify the grid type (defaulting to `mm.GAS`), operational constraints, custom identifiers, names, and positions, and integrates the new node into the network structure for further configuration or analysis.

    Args:
        network (mm.Network): The network to which the gas junction will be added. Must be a valid `mm.Network` instance.
        grid (Any, optional): The grid type for the junction. Defaults to `mm.GAS`. Should be a grid constant (e.g., `mm.GAS`).
        constraints (dict or object, optional): Operational constraints for the junction, such as pressure or flow limits.
        overwrite_id (Any, optional): Custom identifier to override the default junction ID, useful for consistent referencing.
        name (str, optional): Human-readable name for the junction, aiding in identification and reporting.
        position (tuple or str, optional): Geographical or logical position of the junction within the network, typically as coordinates or a descriptive string.

    Returns:
        mm.Junction: The created gas junction node object, already integrated into the network and available for further configuration or analysis.

    Raises:
        ValueError: If the network is invalid, grid type is missing or incorrect, or if provided parameters are incompatible.

    Examples:
        Create a gas junction with a specific name and position::

            gas_junction = create_gas_junction(
                my_network,
                name='Main Gas Junction',
                position=(10, 20)
            )

        Create a gas junction with operational constraints and a custom ID::

            gas_junction = create_gas_junction(
                my_network,
                constraints={'max_flow': 1000},
                overwrite_id='GJ_1'
            )
    """
    return create_junction(network, grid, constraints, overwrite_id, name, position)


def create_junction(
    network: mm.Network,
    grid,
    constraints=None,
    overwrite_id=None,
    name=None,
    position=None,
):
    """
    Creates a junction node in the specified network for a given grid type, enabling flexible resource flow and network expansion.

    This function is used to define connection points for components such as pipes, compressors, or valves in resource grids like gas or water systems. Use it during network construction or modification to establish the topology and facilitate the flow and distribution of resources. The function allows you to specify the grid type, operational constraints, custom identifiers, names, and positions for the junction, and integrates the new node into the network structure for further configuration or analysis.

    Args:
        network (mm.Network): The network to which the junction will be added. Must be a valid `mm.Network` instance.
        grid: The grid type for the junction (e.g., `mm.GAS` for gas, `mm.WATER` for water). Determines the resource domain of the junction.
        constraints (dict or object, optional): Operational constraints for the junction, such as pressure or flow limits.
        overwrite_id (Any, optional): Custom identifier to override the default junction ID, useful for consistent referencing.
        name (str, optional): Human-readable name for the junction, aiding in identification and reporting.
        position (tuple or str, optional): Geographical or logical position of the junction within the network, typically as coordinates or a descriptive string.

    Returns:
        mm.Junction: The created junction node object, already integrated into the network and available for further configuration or analysis.

    Raises:
        ValueError: If the network is invalid, grid type is missing or incorrect, or if provided parameters are incompatible.

    Examples:
        Create a gas junction with a specific name and position::

            gas_junction = create_junction(
                my_network,
                grid=mm.GAS,
                name='Main Gas Junction',
                position=(10, 20)
            )

        Create a water junction with operational constraints and a custom ID::

            water_junction = create_junction(
                my_network,
                grid=mm.WATER,
                constraints={'max_flow': 500},
                overwrite_id='WJ_1'
            )
    """
    return network.node(
        mm.Junction(),
        constraints=constraints,
        grid=grid,
        overwrite_id=overwrite_id,
        name=name,
        position=position,
    )


def create_el_branch(
    network: mm.Network,
    from_node_id,
    to_node_id,
    model,
    constraints=None,
    grid=None,
    name=None,
):
    """
    Creates an electrical branch in the network, connecting two nodes with a specified electrical model and optional constraints.

    This function is used to define the pathways for electrical energy flow between nodes in a network, supporting tasks such as network construction, expansion, or reconfiguration. Use it when you need to represent transmission lines, feeders, or other electrical connections with specific electrical properties. The function integrates a branch object into the network using the provided model, and allows for additional customization through constraints, grid type, and naming for clarity and reporting.

    Args:
        network (mm.Network): The network to which the electrical branch will be added. Must be a valid `mm.Network` instance.
        from_node_id: Identifier of the starting node for the branch connection.
        to_node_id: Identifier of the ending node for the branch connection.
        model: Object or data structure defining the electrical characteristics of the branch (e.g., impedance, capacity). Must be compatible with the network's modeling framework.
        constraints (dict or object, optional): Operational constraints for the branch, such as current or voltage limits.
        grid (Any, optional): Grid type or configuration for the branch (e.g., `mm.EL` for electrical grid). Useful for multi-grid or sector-coupled networks.
        name (str, optional): Human-readable name for the branch, aiding in identification and reporting.

    Returns:
        Branch: The created electrical branch object, already integrated into the network and available for further configuration or analysis.

    Raises:
        ValueError: If the network is invalid, node identifiers are missing or incorrect, or if the model or other parameters are incompatible.

    Examples:
        Create an electrical branch between two nodes with a specific model::

            el_branch = create_el_branch(
                my_network,
                from_node_id=1,
                to_node_id=2,
                model=my_model,
                name='Main Line'
            )

        Add a branch with operational constraints and a custom grid type::

            el_branch = create_el_branch(
                my_network,
                from_node_id='BUS_A',
                to_node_id='BUS_B',
                model=custom_line_model,
                constraints={'max_current': 500},
                grid=mm.EL,
                name='Feeder 1'
            )
    """
    return network.branch(
        model,
        from_node_id=from_node_id,
        to_node_id=to_node_id,
        constraints=constraints,
        grid=grid,
        name=name,
    )


def create_line(
    network: mm.Network,
    from_node_id,
    to_node_id,
    length_m,
    r_ohm_per_m,
    x_ohm_per_m,
    parallel=1,
    constraints=None,
    grid=None,
    name=None,
    on_off=1,
):
    """
    No docstring provided.
    """
    return network.branch(
        mm.PowerLine(length_m, r_ohm_per_m, x_ohm_per_m, parallel, on_off=on_off),
        from_node_id=from_node_id,
        to_node_id=to_node_id,
        constraints=constraints,
        grid=grid,
        name=name,
        auto_node_creator=lambda: mm.Bus(1),
        auto_grid_key=mm.EL_KEY,
    )


def create_gas_pipe(
    network: mm.Network,
    from_node_id,
    to_node_id,
    diameter_m,
    length_m,
    temperature_ext_k=296.15,
    roughness=1e-05,
    on_off=1,
    constraints=None,
    grid=None,
    name=None,
):
    """
    Creates a gas pipe branch in the network, connecting two nodes with specified physical and operational parameters.

    This function is used to model gas pipelines within a network, enabling the simulation of gas flow between nodes. Use it during network construction, expansion, or modification to represent physical gas infrastructure. The function creates a gas pipe object with user-defined diameter, length, external temperature, roughness, and operational state, then integrates it as a branch between the specified nodes. Optional constraints, grid type, and a descriptive name can be provided for further customization. If the target nodes do not exist, gas junctions are automatically created as needed.

    Args:
        network (mm.Network): The network to which the gas pipe will be added. Must be a valid `mm.Network` instance.
        from_node_id: Identifier of the starting node for the gas pipe.
        to_node_id: Identifier of the ending node for the gas pipe.
        diameter_m (float): Inner diameter of the gas pipe in meters.
        length_m (float): Length of the gas pipe in meters.
        temperature_ext_k (float, optional): External temperature in Kelvin. Defaults to 296.15.
        roughness (float, optional): Pipe wall roughness in meters. Defaults to 1e-05.
        on_off (int, optional): Operational state of the pipe (1 for active, 0 for inactive). Defaults to 1.
        constraints (dict or object, optional): Operational constraints for the pipe, such as pressure or flow limits.
        grid (Any, optional): Grid type or configuration for the pipe (e.g., `mm.GAS`). Useful for multi-grid networks.
        name (str, optional): Human-readable name for the gas pipe, aiding in identification and reporting.

    Returns:
        mm.GasPipe: The created gas pipe object, already integrated into the network and connected between the specified nodes.

    Raises:
        ValueError: If the network is invalid, node identifiers are missing or incorrect, or if parameter values are out of valid ranges or incompatible.

    Examples:
        Add a gas pipe between two nodes with specific diameter and length::

            gas_pipe = create_gas_pipe(
                my_network,
                from_node_id=1,
                to_node_id=2,
                diameter_m=0.5,
                length_m=1000,
                name='Main Pipeline'
            )

        Add a gas pipe with custom roughness and operational constraints::

            gas_pipe = create_gas_pipe(
                my_network,
                from_node_id='J1',
                to_node_id='J2',
                diameter_m=0.3,
                length_m=500,
                roughness=2e-5,
                constraints={'max_flow': 2000}
            )
    """
    return network.branch(
        mm.GasPipe(diameter_m, length_m, temperature_ext_k, roughness, on_off=on_off),
        from_node_id=from_node_id,
        to_node_id=to_node_id,
        constraints=constraints,
        grid=grid,
        name=name,
        auto_node_creator=lambda: mm.Junction(),
        auto_grid_key=mm.GAS_KEY,
    )


def create_water_pipe(
    network: mm.Network,
    from_node_id,
    to_node_id,
    diameter_m,
    length_m,
    temperature_ext_k=296.15,
    roughness=0.001,
    lambda_insulation_w_per_k=0.025,
    insulation_thickness_m=0.2,
    on_off=1,
    constraints=None,
    grid=None,
    name=None,
):
    """
    No docstring provided.
    """
    return network.branch(
        mm.WaterPipe(
            diameter_m,
            length_m,
            temperature_ext_k=temperature_ext_k,
            roughness=roughness,
            lambda_insulation_w_per_k=lambda_insulation_w_per_k,
            insulation_thickness_m=insulation_thickness_m,
            on_off=on_off,
        ),
        from_node_id=from_node_id,
        to_node_id=to_node_id,
        constraints=constraints,
        grid=grid,
        name=name,
        auto_node_creator=lambda: mm.Junction(),
        auto_grid_key=mm.WATER_KEY,
    )


def create_el_child(
    network: mm.Network,
    model,
    node_id,
    constraints=None,
    overwrite_id=None,
    name=None,
    **kwargs,
):
    """
    Adds an electrical component as a child to a specified node in the network, supporting flexible configuration and automatic node creation.

    This function is used to attach electrical elements—such as loads, generators, or external grids—to a node within a network. Use it during network construction, expansion, or scenario modeling to represent new sources, sinks, or interconnections. The function ensures the component is properly connected to the specified node, applies any operational constraints, and allows for custom identification and naming. If the target node does not exist, an electrical bus is automatically created to facilitate integration.

    Args:
        network (mm.Network): The network to which the electrical component will be added. Must be a valid `mm.Network` instance.
        model: The model instance representing the electrical component (e.g., `mm.PowerLoad`, `mm.PowerGenerator`, `mm.ExtPowerGrid`). Must be compatible with the network's modeling framework.
        node_id: Identifier of the node where the component will be connected.
        constraints (dict or object, optional): Operational constraints for the component, such as power limits or regulatory requirements.
        overwrite_id (Any, optional): Custom identifier to override the default component ID, useful for consistent referencing.
        name (str, optional): Human-readable name for the component, aiding in identification and reporting.
        **kwargs: Additional keyword arguments for further customization (e.g., advanced modeling options).

    Returns:
        Any: The created electrical component object, already integrated into the network and attached to the specified node.

    Raises:
        ValueError: If the network is invalid, node_id is missing or incorrect, or if the model or other parameters are incompatible.

    Examples:
        Add a power load to a network node::

            power_load = create_el_child(
                my_network,
                model=mm.PowerLoad(p_mw=5, q_mvar=2),
                node_id=10,
                name='Load A'
            )

        Add a generator with operational constraints and a custom ID::

            generator = create_el_child(
                my_network,
                model=mm.PowerGenerator(p_mw=10),
                node_id='GEN_NODE',
                constraints={'max_output': 12},
                overwrite_id='GEN_1'
            )
    """
    return network.child_to(
        model,
        node_id=node_id,
        constraints=constraints,
        overwrite_id=overwrite_id,
        name=name,
        auto_node_creator=lambda: mm.Bus(1),
        auto_grid_key=mm.EL_KEY,
        **kwargs,
    )


def create_water_child(
    network: mm.Network,
    model,
    node_id,
    constraints=None,
    overwrite_id=None,
    name=None,
    **kwargs,
):
    """
    No docstring provided.
    """
    return network.child_to(
        model,
        node_id=node_id,
        constraints=constraints,
        overwrite_id=overwrite_id,
        name=name,
        auto_node_creator=mm.Junction,
        auto_grid_key=mm.WATER_KEY,
        **kwargs,
    )


def create_gas_child(
    network: mm.Network,
    model,
    node_id,
    constraints=None,
    overwrite_id=None,
    name=None,
    **kwargs,
):
    """
    Adds a gas component as a child to a specified node in the network, supporting flexible integration and automatic node creation.

    This function is used to attach gas-related elements—such as compressors, valves, or junctions—to a node within a network. Use it during network construction, expansion, or scenario modeling to represent new gas infrastructure or control devices. The function ensures the component is properly connected to the specified node, applies any operational constraints, and allows for custom identification and naming. If the target node does not exist, a gas junction is automatically created to facilitate integration.

    Args:
        network (mm.Network): The network to which the gas component will be added. Must be a valid `mm.Network` instance.
        model: The model instance representing the gas component (e.g., `mm.Compressor`, `mm.Valve`, or other gas-related models). Must be compatible with the network's modeling framework.
        node_id: Identifier of the node where the component will be connected.
        constraints (dict or object, optional): Operational constraints for the component, such as pressure or flow limits.
        overwrite_id (Any, optional): Custom identifier to override the default component ID, useful for consistent referencing.
        name (str, optional): Human-readable name for the component, aiding in identification and reporting.
        **kwargs: Additional keyword arguments for further customization (e.g., advanced modeling options).

    Returns:
        Any: The created gas component object, already integrated into the network and attached to the specified node.

    Raises:
        ValueError: If the network is invalid, node_id is missing or incorrect, or if the model or other parameters are incompatible.

    Examples:
        Add a compressor to a network node::

            compressor = create_gas_child(
                my_network,
                model=mm.Compressor(ratio=1.5),
                node_id=10,
                name='Compressor A'
            )

        Add a valve with operational constraints and a custom ID::

            valve = create_gas_child(
                my_network,
                model=mm.Valve(opening=0.8),
                node_id='NODE_5',
                constraints={'max_flow': 1000},
                overwrite_id='VALVE_1'
            )
    """
    return network.child_to(
        model,
        node_id=node_id,
        constraints=constraints,
        overwrite_id=overwrite_id,
        name=name,
        auto_node_creator=mm.Junction,
        auto_grid_key=mm.GAS_KEY,
        **kwargs,
    )


def create_power_load(
    network: mm.Network,
    node_id,
    p_mw,
    q_mvar,
    constraints=None,
    overwrite_id=None,
    name=None,
    **kwargs,
):
    """
    No docstring provided.
    """
    return create_el_child(
        network,
        mm.PowerLoad(p_mw, q_mvar, **kwargs),
        node_id=node_id,
        constraints=constraints,
        overwrite_id=overwrite_id,
        name=name,
    )


def create_power_generator(
    network: mm.Network,
    node_id,
    p_mw,
    q_mvar,
    constraints=None,
    overwrite_id=None,
    name=None,
    **kwargs,
):
    """
    No docstring provided.
    """
    return create_el_child(
        network,
        mm.PowerGenerator(p_mw, q_mvar, **kwargs),
        node_id=node_id,
        constraints=constraints,
        overwrite_id=overwrite_id,
        name=name,
    )


def create_ext_power_grid(
    network: mm.Network,
    node_id,
    p_mw=0,
    q_mvar=0,
    vm_pu=1,
    va_degree=0,
    constraints=None,
    overwrite_id=None,
    name=None,
    **kwargs,
):
    """
    Adds an external power grid to a specified node in the network, enabling simulation of power exchange with external sources.

    This function is used to represent the connection between your network and an external power supply, such as a transmission grid or utility interconnection. Use it when modeling scenarios involving grid import/export, contingency analysis, or integration of distributed energy resources. The function creates an external grid object with user-defined electrical parameters (active/reactive power, voltage magnitude, and angle) and attaches it to the chosen node. Additional customization is supported through operational constraints, custom identifiers, and metadata.

    Args:
        network (mm.Network): The network to which the external power grid will be added. Must be a valid `mm.Network` instance.
        node_id: Identifier of the node where the external grid will be connected.
        p_mw (float, optional): Initial value for the active power variable in megawatts. Defaults to 0.
        q_mvar (float, optional): Initial value for the reactive power variable in megavolt-amperes reactive. Defaults to 0.
        vm_pu (float, optional): Voltage magnitude at the external grid in per unit. Defaults to 1.
        va_degree (float, optional): Voltage angle at the external grid in degrees. Defaults to 0.
        constraints (dict or object, optional): Operational constraints for the external grid, such as power or voltage limits.
        overwrite_id (Any, optional): Custom identifier to override the default grid ID.
        name (str, optional): Human-readable name for the external power grid.
        **kwargs: Additional keyword arguments for further customization (e.g., advanced modeling options).

    Returns:
        mm.ExtPowerGrid: The created external power grid object, already integrated into the network and attached to the specified node.

    Raises:
        ValueError: If the network is invalid, node_id is missing or incorrect, or if parameter values are out of valid ranges or incompatible.

    Examples:
        Add an external power grid with custom power and voltage settings::

            ext_power_grid = create_ext_power_grid(
                my_network,
                node_id=5,
                p_mw=10,
                q_mvar=5,
                vm_pu=1.02,
                va_degree=5,
                name='External Grid A'
            )

        Add an external grid with operational constraints and a custom ID::

            ext_power_grid = create_ext_power_grid(
                my_network,
                node_id='EXT_NODE',
                p_mw=20,
                constraints={'max_power': 25},
                overwrite_id='EXT_GRID_1'
            )
    """
    return create_el_child(
        network,
        mm.ExtPowerGrid(p_mw, q_mvar, vm_pu, va_degree, **kwargs),
        node_id=node_id,
        constraints=constraints,
        overwrite_id=overwrite_id,
        name=name,
    )


def create_ext_hydr_grid(
    network: mm.Network,
    node_id,
    mass_flow=1,
    pressure_pu=1,
    t_k=356,
    grid_key=mm.GAS_KEY,
    constraints=None,
    overwrite_id=None,
    name=None,
    **kwargs,
):
    """
    Adds an external hydraulic grid to a specified node in the network with configurable flow, pressure, and operational parameters.

    This function is used to model the integration of external hydraulic sources into an energy network, supporting scenarios such as sector coupling, hydraulic fueling, or storage. Use it during network setup or expansion to represent points where hydraulic is supplied from outside the system. The function creates an external hydraulic grid object with user-defined mass flow, pressure, and temperature, applies any operational constraints, and connects it to the designated node in the network. Additional customization is available via keyword arguments for advanced modeling needs.

    Args:
        network (mm.Network): The network to which the external hydraulic grid will be added. Must be a valid `mm.Network` instance.
        node_id: Identifier of the node where the external hydraulic grid will be connected.
        mass_flow (float, optional): Mass flow rate of hydraulic in kilograms per second. Defaults to 1.
        pressure_pa (float, optional): hydraulic pressure in pascals. Defaults to 1,000,000.
        t_k (float, optional): hydraulic temperature in Kelvin. Defaults to 359.
        constraints (dict or object, optional): Operational constraints for the hydraulic grid, such as flow or pressure limits.
        overwrite_id (Any, optional): Custom identifier to override the default grid ID.
        name (str, optional): Human-readable name for the external hydraulic grid.
        **kwargs: Additional keyword arguments for further customization (e.g., advanced modeling or solver options).

    Returns:
        mm.ExtHydrGrid: The created external hydraulic grid object, integrated into the network and connected to the specified node.

    Raises:
        ValueError: If the network is invalid, node_id is missing or incorrect, or if parameter values are out of valid ranges or incompatible.

    Examples:
        Add an external hydraulic grid with custom mass flow and pressure::

            ext_hydr_grid = create_ext_hydr_grid(
                my_network,
                node_id=5,
                mass_flow=2,
                pressure_pa=1500000,
                name='External hydraulic Grid A'
            )

        Add an external hydraulic grid with operational constraints and a custom ID::

            ext_hydr_grid = create_ext_hydr_grid(
                my_network,
                node_id='EXT_H2_NODE',
                mass_flow=3.5,
                constraints={'max_flow': 5.0},
                overwrite_id='EXT_H2_GRID_1'
            )
    """
    return network.child_to(
        mm.ExtHydrGrid(mass_flow=mass_flow, pressure_pu=pressure_pu, t_k=t_k, **kwargs),
        node_id=node_id,
        constraints=constraints,
        overwrite_id=overwrite_id,
        name=name,
        auto_node_creator=mm.Junction,
        auto_grid_key=grid_key,
    )


def create_source(
    network: mm.Network,
    node_id,
    mass_flow=1,
    grid_key=mm.GAS_KEY,
    constraints=None,
    overwrite_id=None,
    name=None,
    **kwargs,
):
    """
    No docstring provided.
    """
    return network.child_to(
        mm.Source(mass_flow, **kwargs),
        node_id=node_id,
        constraints=constraints,
        overwrite_id=overwrite_id,
        name=name,
        auto_node_creator=mm.Junction,
        auto_grid_key=grid_key,
    )


def create_consume_hydr_grid(
    network: mm.Network,
    node_id,
    mass_flow=1,
    pressure_pa=1000000,
    t_k=293,
    constraints=None,
    overwrite_id=None,
    name=None,
    **kwargs,
):
    """
    Adds a hydraulic consumption grid to a specified node in the network with configurable flow, pressure, and operational parameters.

    This function is intended for modeling hydraulic demand points within an energy network, such as those required for fuel cell integration, hydraulic storage, or sector coupling applications. Use it during network setup or expansion to represent locations where hydraulic is consumed. The function creates a hydraulic consumption grid object with user-defined mass flow, pressure, and temperature, applies any operational constraints, and integrates it into the network at the designated node. Additional customization is supported via keyword arguments for advanced modeling needs.

    Args:
        network (mm.Network): The network to which the hydraulic consumption grid will be added. Must be a valid `mm.Network` instance.
        node_id: Identifier of the node where the hydraulic grid will be connected.
        mass_flow (float, optional): Mass flow rate of hydraulic in kilograms per second. Defaults to 1.
        pressure_pa (float, optional): hydraulic pressure in pascals. Defaults to 1,000,000.
        t_k (float, optional): hydraulic temperature in Kelvin. Defaults to 293.
        constraints (dict or object, optional): Operational constraints for the hydraulic grid, such as flow or pressure limits.
        overwrite_id (Any, optional): Custom identifier to override the default grid ID.
        name (str, optional): Human-readable name for the hydraulic consumption grid.
        **kwargs: Additional keyword arguments for further customization (e.g., advanced modeling or solver options).

    Returns:
        mm.ConsumeHydrGrid: The created hydraulic consumption grid object, integrated into the network and connected to the specified node.

    Raises:
        ValueError: If the network is invalid, node_id is missing or incorrect, or if parameter values are out of valid ranges or incompatible.

    Examples:
        Add a hydraulic consumption grid with custom mass flow and pressure::

            hydr_grid = create_consume_hydr_grid(
                my_network,
                node_id=5,
                mass_flow=2,
                pressure_pa=1500000,
                name='hydraulic Grid A'
            )

        Add a hydraulic grid with operational constraints and a custom ID::

            hydr_grid = create_consume_hydr_grid(
                my_network,
                node_id='H2_NODE',
                mass_flow=3.5,
                constraints={'max_flow': 5.0},
                overwrite_id='H2_GRID_1'
            )
    """
    return network.child_to(
        mm.ConsumeHydrGrid(
            mass_flow=mass_flow, pressure_pa=pressure_pa, t_k=t_k, **kwargs
        ),
        node_id=node_id,
        constraints=constraints,
        overwrite_id=overwrite_id,
        name=name,
    )


def create_sink(
    network: mm.Network,
    node_id,
    mass_flow=1,
    grid_key=mm.GAS_KEY,
    constraints=None,
    overwrite_id=None,
    name=None,
    **kwargs,
):
    """
    No docstring provided.
    """
    return network.child_to(
        mm.Sink(mass_flow=mass_flow, **kwargs),
        node_id=node_id,
        constraints=constraints,
        overwrite_id=overwrite_id,
        name=name,
        auto_node_creator=mm.Junction,
        auto_grid_key=grid_key,
    )


def create_heat_exchanger(
    network: mm.Network,
    from_node_id,
    to_node_id,
    q_mw,
    diameter_m=0.1,
    temperature_ext_k=293,
    constraints=None,
    grid=None,
    name=None,
):
    """
    No docstring provided.
    """
    return network.branch(
        mm.HeatExchangerLoad(
            q_mw=-q_mw, diameter_m=diameter_m, temperature_ext_k=temperature_ext_k
        )
        if q_mw > 0
        else mm.HeatExchangerGenerator(
            q_mw=-q_mw, diameter_m=diameter_m, temperature_ext_k=temperature_ext_k
        ),
        from_node_id=from_node_id,
        to_node_id=to_node_id,
        constraints=constraints,
        grid=grid,
        name=name,
        auto_node_creator=mm.Junction,
        auto_grid_key=mm.WATER_KEY,
    )


def create_p2g(
    network: mm.Network,
    from_node_id,
    to_node_id,
    efficiency,
    mass_flow_setpoint,
    consume_q_mvar_setpoint=0,
    regulation=1,
    constraints=None,
    grid=None,
    name=None,
):
    """
    No docstring provided.
    """
    return network.branch(
        mm.PowerToGas(
            efficiency=efficiency,
            mass_flow_setpoint=mass_flow_setpoint,
            consume_q_mvar_setpoint=consume_q_mvar_setpoint,
            regulation=regulation,
        ),
        from_node_id=from_node_id,
        to_node_id=to_node_id,
        constraints=constraints,
        grid=grid,
        name=name,
    )


def create_g2p(
    network: mm.Network,
    from_node_id,
    to_node_id,
    efficiency,
    p_mw_setpoint,
    q_mvar_setpoint=0,
    regulation=1,
    constraints=None,
    grid=None,
    name=None,
):
    """
    Adds a gas-to-power conversion branch to the network, connecting specified nodes and enabling gas-to-electricity conversion with defined operational parameters.

    This function is used to model the conversion of gas energy into electrical power within an energy network, such as in combined cycle plants or distributed generation scenarios. Use it during network setup or expansion to represent gas turbines or similar conversion equipment. The function creates a gas-to-power branch with user-defined efficiency, active and reactive power setpoints, and regulation factor, then connects it between the designated gas and power nodes. Optional constraints, grid type, and a descriptive name can be provided for further customization and clarity.

    Args:
        network (mm.Network): The network to which the gas-to-power conversion branch will be added. Must be a valid `mm.Network` instance.
        from_node_id: Identifier of the starting node (gas side) for the conversion branch.
        to_node_id: Identifier of the ending node (power side) for the conversion branch.
        efficiency (float): Conversion efficiency (0 < value ≤ 1), representing the ratio of electrical output to gas input.
        p_mw_setpoint (float): Active power setpoint in megawatts for the conversion branch.
        q_mvar_setpoint (float, optional): Reactive power setpoint in megavolt-amperes reactive. Defaults to 0.
        regulation (float, optional): Regulation factor affecting responsiveness to load changes. Defaults to 1.
        constraints (dict or object, optional): Operational constraints for the branch, such as output or efficiency limits.
        grid (Any, optional): Grid type or configuration for the branch (e.g., `mm.EL` for electrical grid). Useful for multi-grid or sector-coupled networks.
        name (str, optional): Human-readable name for the conversion branch, aiding in identification and reporting.

    Returns:
        mm.GasToPower: The created gas-to-power conversion branch object, already integrated into the network and connected between the specified nodes.

    Raises:
        ValueError: If the network is invalid, node identifiers are missing or incorrect, or if parameter values are out of valid ranges or incompatible.

    Examples:
        Add a gas-to-power conversion branch with specific power setpoints and efficiency::

            g2p_branch = create_g2p(
                my_network,
                from_node_id=1,
                to_node_id=2,
                efficiency=0.9,
                p_mw_setpoint=50,
                q_mvar_setpoint=10,
                name='Gas to Power Line A'
            )

        Add a branch with custom regulation and operational constraints::

            g2p_branch = create_g2p(
                my_network,
                from_node_id='GAS_NODE',
                to_node_id='EL_NODE',
                efficiency=0.85,
                p_mw_setpoint=30,
                regulation=0.95,
                constraints={'max_output': 40}
            )
    """
    return network.branch(
        mm.GasToPower(
            efficiency=efficiency,
            p_mw_setpoint=p_mw_setpoint,
            q_mvar_setpoint=q_mvar_setpoint,
            regulation=regulation,
        ),
        from_node_id=from_node_id,
        to_node_id=to_node_id,
        constraints=constraints,
        grid=grid,
        name=name,
    )


def create_chp(
    network: mm.Network,
    power_node_id,
    heat_node_id,
    heat_return_node_id,
    gas_node_id,
    diameter_m,
    efficiency_power,
    efficiency_heat,
    mass_flow_setpoint,
    regulation=1,
    constraints=None,
    remove_existing_branch=False,
):
    """
    Adds a Combined Heat and Power (CHP) unit to the network with specified connectivity, efficiency, and operational parameters.

    This function is intended for modeling cogeneration systems that simultaneously generate electricity and heat, enhancing overall energy efficiency in multi-energy networks. Use it during network construction or expansion to represent distributed energy resources that require explicit connections to power, heat, and gas nodes. The function creates a CHP unit with user-defined physical and operational characteristics, applies optional constraints and regulation factors, and integrates the unit into the network by connecting it to the specified nodes.

    Args:
        network (mm.Network): The network to which the CHP unit will be added. Must be a valid `mm.Network` instance.
        power_node_id: Identifier for the power node (electrical connection).
        heat_node_id: Identifier for the heat node (thermal output).
        heat_return_node_id: Identifier for the heat return node (heating circuit return).
        gas_node_id: Identifier for the gas node (fuel supply).
        diameter_m (float): Diameter of the CHP unit in meters, affecting capacity and efficiency.
        efficiency_power (float): Electrical efficiency (0 < value ≤ 1), representing the ratio of electrical output to fuel input.
        efficiency_heat (float): Thermal efficiency (0 < value ≤ 1), representing the ratio of heat output to fuel input.
        mass_flow_setpoint (float): Setpoint for the mass flow rate through the CHP unit.
        regulation (float, optional): Regulation factor for load responsiveness. Defaults to 1.
        constraints (dict or object, optional): Operational constraints (e.g., output limits, regulatory requirements).
        remove_existing_branch (bool, optional): Whether to remove the existing branch between the heat nodes

    Returns:
        mm.CHP: The created CHP unit object, integrated into the network and connected to the specified nodes.

    Raises:
        ValueError: If the network is invalid, any node identifier is missing or incorrect, or if parameter values are out of valid ranges or incompatible.

    Examples:
        Add a CHP unit with specified efficiencies and diameter::

            chp = create_chp(
                my_network,
                power_node_id=1,
                heat_node_id=2,
                heat_return_node_id=3,
                gas_node_id=4,
                diameter_m=0.5,
                efficiency_power=0.4,
                efficiency_heat=0.5,
                mass_flow_setpoint=10
            )

        Add a CHP unit with custom regulation and operational constraints::

            chp = create_chp(
                my_network,
                power_node_id='P1',
                heat_node_id='H1',
                heat_return_node_id='HR1',
                gas_node_id='G1',
                diameter_m=0.7,
                efficiency_power=0.42,
                efficiency_heat=0.48,
                mass_flow_setpoint=12,
                regulation=0.8,
                constraints={'max_output': 5.0}
            )
    """
    if remove_existing_branch:
        network.remove_branch_between(heat_node_id, heat_return_node_id)
    return network.compound(
        mm.CHP(
            diameter_m,
            efficiency_power,
            efficiency_heat,
            mass_flow_setpoint,
            q_mvar_setpoint=0,
            temperature_ext_k=293,
            regulation=regulation,
        ),
        constraints=constraints,
        power_node_id=power_node_id,
        heat_node_id=heat_node_id,
        heat_return_node_id=heat_return_node_id,
        gas_node_id=gas_node_id,
    )


def create_p2h(
    network: mm.Network,
    power_node_id,
    heat_node_id,
    heat_return_node_id,
    heat_energy_mw,
    diameter_m,
    efficiency,
    temperature_ext_k=293,
    q_mvar_setpoint=0,
    constraints=None,
):
    """
    No docstring provided.
    """
    return network.compound(
        mm.PowerToHeat(
            heat_energy_mw=heat_energy_mw,
            diameter_m=diameter_m,
            temperature_ext_k=temperature_ext_k,
            efficiency=efficiency,
            q_mvar_setpoint=q_mvar_setpoint,
        ),
        constraints=constraints,
        power_node_id=power_node_id,
        heat_node_id=heat_node_id,
        heat_return_node_id=heat_return_node_id,
    )


def create_g2h(
    network: mm.Network,
    gas_node_id,
    heat_node_id,
    heat_return_node_id,
    heat_energy_w,
    diameter_m,
    efficiency,
    temperature_ext_k=293,
    constraints=None,
):
    """
    Adds a gas-to-heat conversion unit to the network, connecting specified gas and heat nodes with defined operational parameters.

    This function is used to model the conversion of gas energy into heat within an energy network, such as in district heating systems or industrial processes requiring gas-fired heating. Use it during network setup or expansion to represent gas boilers or similar equipment. The function creates a gas-to-heat unit with user-defined heat output, efficiency, and physical characteristics, then connects it to the appropriate gas supply, heat delivery, and heat return nodes. Optional constraints and external temperature settings can be applied to tailor the unit's operation and ensure compliance with system requirements.

    Args:
        network (mm.Network): The network to which the gas-to-heat conversion unit will be added. Must be a valid `mm.Network` instance.
        gas_node_id: Identifier of the gas node supplying fuel to the conversion unit.
        heat_node_id: Identifier of the heat node receiving the converted heat energy.
        heat_return_node_id: Identifier of the heat return node for the heating circuit.
        heat_energy_w (float): Amount of heat energy produced by the unit, in watts.
        diameter_m (float): Diameter of the conversion unit in meters, influencing capacity and efficiency.
        efficiency (float): Conversion efficiency (0 < value ≤ 1), representing the ratio of heat output to gas input.
        temperature_ext_k (float, optional): External temperature in Kelvin, affecting conversion performance. Defaults to 293.
        constraints (dict or object, optional): Operational constraints for the unit, such as output or efficiency limits.

    Returns:
        mm.GasToHeat: The created gas-to-heat conversion unit object, already integrated into the network and connected to the specified nodes.

    Raises:
        ValueError: If the network is invalid, node identifiers are missing or incorrect, or if parameter values are out of valid ranges or incompatible.

    Examples:
        Add a gas-to-heat conversion unit with specific energy output and efficiency::

            g2h_unit = create_g2h(
                my_network,
                gas_node_id=1,
                heat_node_id=2,
                heat_return_node_id=3,
                heat_energy_w=5000,
                diameter_m=0.3,
                efficiency=0.85
            )

        Add a unit with custom external temperature and operational constraints::

            g2h_unit = create_g2h(
                my_network,
                gas_node_id='GAS1',
                heat_node_id='HEAT1',
                heat_return_node_id='HEAT_RET1',
                heat_energy_w=10000,
                diameter_m=0.5,
                efficiency=0.9,
                temperature_ext_k=310,
                constraints={'max_output': 12000}
            )
    """
    return network.compound(
        mm.GasToHeat(
            heat_energy_w=heat_energy_w,
            diameter_m=diameter_m,
            temperature_ext_k=temperature_ext_k,
            efficiency=efficiency,
        ),
        constraints=constraints,
        gas_node_id=gas_node_id,
        heat_node_id=heat_node_id,
        heat_return_node_id=heat_return_node_id,
    )


def create_trafo(
    network: mm.Network,
    from_node_id,
    to_node_id,
    vk_percent=12.2,
    vkr_percent=0.25,
    sn_trafo_mva=160,
    shift=0,
    constraints=None,
    grid=None,
    name=None,
):
    """
    Adds a two-winding transformer branch to the network.

    The from-node is treated as the low-voltage side and the to-node as the
    high-voltage side when computing per-unit impedance. If the referenced
    nodes do not yet exist, electrical buses are created automatically.

    Args:
        network (mm.Network): The network to which the transformer is added.
        from_node_id: Low-voltage bus identifier.
        to_node_id: High-voltage bus identifier.
        vk_percent (float): Short-circuit voltage in percent. Defaults to 12.2.
        vkr_percent (float): Real part of short-circuit voltage in percent. Defaults to 0.25.
        sn_trafo_mva (float): Rated apparent power in MVA. Defaults to 160.
        shift (float): Phase shift in radians. Defaults to 0.
        constraints: Operational constraints for the transformer.
        grid: Grid type. Defaults to the electrical grid.
        name (str): Human-readable name for the transformer.

    Returns:
        mm.Trafo: The created transformer object integrated into the network.
    """
    return network.branch(
        mm.Trafo(
            vk_percent=vk_percent,
            vkr_percent=vkr_percent,
            sn_trafo_mva=sn_trafo_mva,
            shift=shift,
        ),
        from_node_id=from_node_id,
        to_node_id=to_node_id,
        constraints=constraints,
        grid=grid,
        name=name,
        auto_node_creator=lambda: mm.Bus(1),
        auto_grid_key=mm.EL_KEY,
    )


def create_gas_ext_grid(
    network: mm.Network,
    node_id,
    mass_flow=1,
    pressure_pu=1,
    t_k=356,
    constraints=None,
    overwrite_id=None,
    name=None,
    **kwargs,
):
    """
    Adds an external hydraulic grid to a gas node, auto-creating a gas junction if needed.
    """
    return create_ext_hydr_grid(
        network,
        node_id,
        mass_flow=mass_flow,
        pressure_pu=pressure_pu,
        t_k=t_k,
        grid_key=mm.GAS_KEY,
        constraints=constraints,
        overwrite_id=overwrite_id,
        name=name,
        **kwargs,
    )


def create_water_ext_grid(
    network: mm.Network,
    node_id,
    mass_flow=1,
    pressure_pu=1,
    t_k=356,
    constraints=None,
    overwrite_id=None,
    name=None,
    **kwargs,
):
    """
    Adds an external hydraulic grid to a water node, auto-creating a water junction if needed.
    """
    return create_ext_hydr_grid(
        network,
        node_id,
        mass_flow=mass_flow,
        pressure_pu=pressure_pu,
        t_k=t_k,
        grid_key=mm.WATER_KEY,
        constraints=constraints,
        overwrite_id=overwrite_id,
        name=name,
        **kwargs,
    )


def create_gas_source(
    network: mm.Network,
    node_id,
    mass_flow=1,
    constraints=None,
    overwrite_id=None,
    name=None,
    **kwargs,
):
    """
    Adds a gas source (injection) to a gas node, auto-creating a junction if needed.
    """
    return create_source(
        network,
        node_id,
        mass_flow=mass_flow,
        grid_key=mm.GAS_KEY,
        constraints=constraints,
        overwrite_id=overwrite_id,
        name=name,
        **kwargs,
    )


def create_gas_sink(
    network: mm.Network,
    node_id,
    mass_flow=1,
    constraints=None,
    overwrite_id=None,
    name=None,
    **kwargs,
):
    """
    Adds a gas sink (consumption) to a gas node, auto-creating a junction if needed.
    """
    return create_sink(
        network,
        node_id,
        mass_flow=mass_flow,
        grid_key=mm.GAS_KEY,
        constraints=constraints,
        overwrite_id=overwrite_id,
        name=name,
        **kwargs,
    )


def create_water_source(
    network: mm.Network,
    node_id,
    mass_flow=1,
    constraints=None,
    overwrite_id=None,
    name=None,
    **kwargs,
):
    """
    Adds a water source (injection) to a water node, auto-creating a junction if needed.
    """
    return create_source(
        network,
        node_id,
        mass_flow=mass_flow,
        grid_key=mm.WATER_KEY,
        constraints=constraints,
        overwrite_id=overwrite_id,
        name=name,
        **kwargs,
    )


def create_water_sink(
    network: mm.Network,
    node_id,
    mass_flow=1,
    constraints=None,
    overwrite_id=None,
    name=None,
    **kwargs,
):
    """
    Adds a water sink (consumption) to a water node, auto-creating a junction if needed.
    """
    return create_sink(
        network,
        node_id,
        mass_flow=mass_flow,
        grid_key=mm.WATER_KEY,
        constraints=constraints,
        overwrite_id=overwrite_id,
        name=name,
        **kwargs,
    )


def create_multi_energy_network():
    """
    No docstring provided.
    """
    return mm.Network()

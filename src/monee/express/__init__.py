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
        network (mm.Network): The network to which the bus will be added.
        base_kv (float, optional): Base voltage level of the bus in kilovolts. Defaults to 1.
        constraints (list, optional): List of constraint callables for the bus.
        grid: Grid domain for the bus. Defaults to ``mm.EL`` (electrical grid).
        overwrite_id: Custom identifier to override the auto-assigned node ID.
        name (str, optional): Human-readable name for the bus.
        position (tuple, optional): Geographical position as ``(x, y)`` coordinates.

    Returns:
        int: The node ID of the created bus.  Pass this ID to branch and child
        creation functions (e.g. ``create_line``, ``create_power_load``).

    Examples::

        bus_0 = create_bus(net, base_kv=11, name='HV Bus')
        bus_1 = create_bus(net)
        create_line(net, bus_0, bus_1, length_m=500, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)
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
    Add a junction node to the water/heat grid.

    Args:
        network (mm.Network): Target network.
        grid: Grid domain. Defaults to ``mm.WATER``.
        constraints (list, optional): Constraint callables for the junction.
        overwrite_id: Custom identifier to override the auto-assigned node ID.
        name (str, optional): Human-readable name.
        position (tuple, optional): Geographical position as ``(x, y)``.

    Returns:
        int: The node ID of the created junction.
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
        network (mm.Network): Target network.
        grid: Grid domain. Defaults to ``mm.GAS``.
        constraints (list, optional): Constraint callables for the junction.
        overwrite_id: Custom identifier to override the auto-assigned node ID.
        name (str, optional): Human-readable name.
        position (tuple, optional): Geographical position as ``(x, y)``.

    Returns:
        int: The node ID of the created junction.
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
        network (mm.Network): Target network.
        grid: Grid domain (e.g. ``mm.GAS`` or ``mm.WATER``).
        constraints (list, optional): Constraint callables for the junction.
        overwrite_id: Custom identifier to override the auto-assigned node ID.
        name (str, optional): Human-readable name.
        position (tuple, optional): Geographical position as ``(x, y)``.

    Returns:
        int: The node ID of the created junction.
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
        network (mm.Network): Target network.
        from_node_id: Source bus node ID.
        to_node_id: Destination bus node ID.
        model: A :class:`mm.BranchModel` instance defining the branch physics.
        constraints (list, optional): Constraint callables for the branch.
        grid: Grid domain override. Usually inferred from the node grid.
        name (str, optional): Human-readable name.

    Returns:
        tuple: The branch ID ``(from_node_id, to_node_id, edge_key)``.
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
    Add a power line branch to the electrical grid.

    If either referenced node does not yet exist, an electrical bus is created
    automatically.

    Args:
        network (mm.Network): Target network.
        from_node_id: Source bus node ID.
        to_node_id: Destination bus node ID.
        length_m (float): Line length in metres.
        r_ohm_per_m (float): Resistance per metre in Ω/m.
        x_ohm_per_m (float): Reactance per metre in Ω/m.
        parallel (int, optional): Number of parallel conductors. Defaults to 1.
        constraints (list, optional): Constraint callables.
        grid: Grid domain override.
        name (str, optional): Human-readable name.
        on_off (int or Var): ``1`` = active, ``0`` = disconnected. Defaults to 1.

    Returns:
        tuple: The branch ID ``(from_node_id, to_node_id, edge_key)``.
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
        network (mm.Network): Target network.
        from_node_id: Source junction node ID.
        to_node_id: Destination junction node ID.
        diameter_m (float): Inner pipe diameter in metres.
        length_m (float): Pipe length in metres.
        temperature_ext_k (float, optional): Ambient temperature in Kelvin. Defaults to 296.15.
        roughness (float, optional): Pipe wall roughness in metres. Defaults to 1e-5.
        on_off (int, optional): ``1`` = active, ``0`` = disconnected. Defaults to 1.
        constraints (list, optional): Constraint callables.
        grid: Grid domain override.
        name (str, optional): Human-readable name.

    Returns:
        tuple: The branch ID ``(from_node_id, to_node_id, edge_key)``.
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
    Add a water/heat pipe branch to the hydraulic grid.

    If either referenced node does not yet exist, a water junction is created
    automatically.

    Args:
        network (mm.Network): Target network.
        from_node_id: Source junction node ID.
        to_node_id: Destination junction node ID.
        diameter_m (float): Inner pipe diameter in metres.
        length_m (float): Pipe length in metres.
        temperature_ext_k (float, optional): Ambient temperature in Kelvin. Defaults to 296.15.
        roughness (float, optional): Pipe wall roughness in metres. Defaults to 0.001.
        lambda_insulation_w_per_k (float, optional): Thermal conductivity of insulation in W/(m·K).
            Defaults to 0.025.
        insulation_thickness_m (float, optional): Insulation thickness in metres. Defaults to 0.2.
        on_off (int, optional): ``1`` = active, ``0`` = disconnected. Defaults to 1.
        constraints (list, optional): Constraint callables.
        grid: Grid domain override.
        name (str, optional): Human-readable name.

    Returns:
        tuple: The branch ID ``(from_node_id, to_node_id, edge_key)``.
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
        network (mm.Network): Target network.
        model: A :class:`mm.ChildModel` instance (e.g. ``mm.PowerLoad``, ``mm.ExtPowerGrid``).
        node_id: Bus node ID to attach the component to.  An electrical bus is
            created automatically if the node does not yet exist.
        constraints (list, optional): Constraint callables.
        overwrite_id: Custom identifier to override the auto-assigned child ID.
        name (str, optional): Human-readable name.
        **kwargs: Forwarded to :meth:`Network.child_to`.

    Returns:
        int: The child ID of the created component.
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
    Attach a hydraulic child component to a water/heat junction node.

    Args:
        network (mm.Network): Target network.
        model: A :class:`mm.ChildModel` instance.
        node_id: Junction node ID.  A water junction is created automatically
            if the node does not yet exist.
        constraints (list, optional): Constraint callables.
        overwrite_id: Custom identifier to override the auto-assigned child ID.
        name (str, optional): Human-readable name.
        **kwargs: Forwarded to :meth:`Network.child_to`.

    Returns:
        int: The child ID of the created component.
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
        network (mm.Network): Target network.
        model: A :class:`mm.ChildModel` instance.
        node_id: Junction node ID.  A gas junction is created automatically
            if the node does not yet exist.
        constraints (list, optional): Constraint callables.
        overwrite_id: Custom identifier to override the auto-assigned child ID.
        name (str, optional): Human-readable name.
        **kwargs: Forwarded to :meth:`Network.child_to`.

    Returns:
        int: The child ID of the created component.
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
    Attach a fixed-setpoint power load to an electrical bus.

    Sign convention: positive values represent *consumption* (load).

    Args:
        network (mm.Network): Target network.
        node_id: Bus node ID to attach the load to.
        p_mw (float): Active power demand in MW (positive = consumption).
        q_mvar (float): Reactive power demand in Mvar (positive = consumption).
        constraints (list, optional): Constraint callables.
        overwrite_id: Custom identifier.
        name (str, optional): Human-readable name.
        **kwargs: Forwarded to :class:`mm.PowerLoad`.

    Returns:
        int: The child ID of the created load.
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
    Attach a fixed-setpoint power generator to an electrical bus.

    Sign convention: supply positive magnitudes — the constructor negates them
    internally so the node balance sees an injection (negative = generation).

    Args:
        network (mm.Network): Target network.
        node_id: Bus node ID to attach the generator to.
        p_mw (float): Active power output in MW (positive = generation).
        q_mvar (float): Reactive power output in Mvar (positive = generation).
        constraints (list, optional): Constraint callables.
        overwrite_id: Custom identifier.
        name (str, optional): Human-readable name.
        **kwargs: Forwarded to :class:`mm.PowerGenerator`.

    Returns:
        int: The child ID of the created generator.
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
        network (mm.Network): Target network.
        node_id: Bus node ID to attach the slack source to.
        p_mw (float, optional): Initial active power exchange guess in MW. Defaults to 0.
        q_mvar (float, optional): Initial reactive power exchange guess in Mvar. Defaults to 0.
        vm_pu (float, optional): Voltage magnitude setpoint in per-unit. Defaults to 1.0.
        va_degree (float, optional): Voltage angle setpoint in degrees. Defaults to 0.0.
        constraints (list, optional): Constraint callables.
        overwrite_id: Custom identifier.
        name (str, optional): Human-readable name.
        **kwargs: Forwarded to :class:`mm.ExtPowerGrid`.

    Returns:
        int: The child ID of the created external grid.
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
        network (mm.Network): Target network.
        node_id: Junction node ID to attach the slack source to.
        mass_flow (float, optional): Initial mass-flow guess in kg/s.  The solver
            determines the final value; negative = injection (source). Defaults to 1.
        pressure_pu (float, optional): Junction pressure setpoint in per-unit. Defaults to 1.0.
        t_k (float, optional): Supply temperature setpoint in Kelvin. Defaults to 356.
        grid_key (str, optional): Carrier key — ``mm.GAS_KEY`` or ``mm.WATER_KEY``.
            Prefer the carrier-specific shortcuts :func:`create_gas_ext_grid` and
            :func:`create_water_ext_grid`. Defaults to ``mm.GAS_KEY``.
        constraints (list, optional): Constraint callables.
        overwrite_id: Custom identifier.
        name (str, optional): Human-readable name.
        **kwargs: Forwarded to :class:`mm.ExtHydrGrid`.

    Returns:
        int: The child ID of the created external hydraulic grid.

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
    Attach a fixed mass-flow source (injection) to a gas or water junction.

    Sign convention: supply a positive magnitude — the constructor negates it
    internally so the junction balance sees an injection (negative = generation).

    .. note::
        *grid_key* defaults to ``mm.GAS_KEY``.  For water/heat networks use
        ``grid_key=mm.WATER_KEY`` or the dedicated :func:`create_water_source`.

    Args:
        network (mm.Network): Target network.
        node_id: Junction node ID.
        mass_flow (float, optional): Mass flow rate to inject in kg/s (positive = injection).
            Defaults to 1.
        grid_key (str, optional): Carrier key. Defaults to ``mm.GAS_KEY``.
        constraints (list, optional): Constraint callables.
        overwrite_id: Custom identifier.
        name (str, optional): Human-readable name.
        **kwargs: Forwarded to :class:`mm.Source`.

    Returns:
        int: The child ID of the created source.
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
        network (mm.Network): Target network.
        node_id: Junction node ID.
        mass_flow (float, optional): Mass flow rate to consume in kg/s (positive = consumption).
            Defaults to 1.
        pressure_pa (float, optional): Junction pressure setpoint in Pa. Defaults to 1,000,000.
        t_k (float, optional): Return temperature in Kelvin. Defaults to 293.
        constraints (list, optional): Constraint callables.
        overwrite_id: Custom identifier.
        name (str, optional): Human-readable name.
        **kwargs: Forwarded to :class:`mm.ConsumeHydrGrid`.

    Returns:
        int: The child ID of the created consumption grid.
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
    Attach a fixed mass-flow sink (withdrawal) to a gas or water junction.

    Sign convention: positive values represent *consumption*.  Unlike
    :func:`create_source`, the value is stored as-is (not negated).

    .. note::
        *grid_key* defaults to ``mm.GAS_KEY``.  For water/heat networks use
        ``grid_key=mm.WATER_KEY`` or the dedicated :func:`create_water_sink`.

    Args:
        network (mm.Network): Target network.
        node_id: Junction node ID.
        mass_flow (float, optional): Mass flow rate to withdraw in kg/s (positive = consumption).
            Defaults to 1.
        grid_key (str, optional): Carrier key. Defaults to ``mm.GAS_KEY``.
        constraints (list, optional): Constraint callables.
        overwrite_id: Custom identifier.
        name (str, optional): Human-readable name.
        **kwargs: Forwarded to :class:`mm.Sink`.

    Returns:
        int: The child ID of the created sink.
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
    Add a heat exchanger branch to the water/heat grid.

    Sign convention (load convention — positive = consumption):

    * ``q_mw > 0`` → :class:`mm.HeatExchangerLoad` (heat *consumed* along the branch).
    * ``q_mw < 0`` → :class:`mm.HeatExchangerGenerator` (heat *injected* into the branch).

    Args:
        network (mm.Network): Target network.
        from_node_id: Upstream junction node ID.
        to_node_id: Downstream junction node ID.
        q_mw (float): Heat exchange power in MW.  Positive = consumption (load),
            negative = generation (injection).
        diameter_m (float, optional): Equivalent pipe diameter in metres. Defaults to 0.1.
        temperature_ext_k (float, optional): Ambient temperature in Kelvin. Defaults to 293.
        constraints (list, optional): Constraint callables.
        grid: Grid domain override.
        name (str, optional): Human-readable name.

    Returns:
        tuple: The branch ID ``(from_node_id, to_node_id, edge_key)``.
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
    Add a Power-to-Gas (P2G) coupling branch between an electrical bus and a gas junction.

    Converts electrical power to gas mass flow.  The branch spans two different
    carrier domains and is therefore treated as a control point in the solver.

    Args:
        network (mm.Network): Target network.
        from_node_id: Electrical bus node ID (power side).
        to_node_id: Gas junction node ID (gas side).
        efficiency (float): Conversion efficiency in ``(0, 1]`` — ratio of
            chemical energy output to electrical energy input.
        mass_flow_setpoint (float): Target gas mass flow in kg/s (positive = injection
            into the gas network, i.e. generation).
        consume_q_mvar_setpoint (float, optional): Reactive power consumed from the
            electrical side in Mvar. Defaults to 0.
        regulation (float, optional): Dispatch fraction in ``[0.0, 1.0]``.
            ``1.0`` = full setpoint, ``0.0`` = off. Defaults to 1.
        constraints (list, optional): Constraint callables.
        grid: Grid domain override.
        name (str, optional): Human-readable name.

    Returns:
        tuple: The branch ID ``(from_node_id, to_node_id, edge_key)``.
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
        network (mm.Network): Target network.
        from_node_id: Gas junction node ID (gas side).
        to_node_id: Electrical bus node ID (power side).
        efficiency (float): Conversion efficiency in ``(0, 1]`` — ratio of
            electrical output to gas energy input.
        p_mw_setpoint (float): Target active power output in MW (positive = generation
            onto the electrical bus, i.e. injection).
        q_mvar_setpoint (float, optional): Target reactive power in Mvar. Defaults to 0.
        regulation (float, optional): Dispatch fraction in ``[0.0, 1.0]``.
            ``1.0`` = full setpoint, ``0.0`` = off. Defaults to 1.
        constraints (list, optional): Constraint callables.
        grid: Grid domain override.
        name (str, optional): Human-readable name.

    Returns:
        tuple: The branch ID ``(from_node_id, to_node_id, edge_key)``.
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
        network (mm.Network): Target network.
        power_node_id: Electrical bus node ID (electrical output side).
        heat_node_id: Water junction node ID (heat supply side).
        heat_return_node_id: Water junction node ID (heat return side).
        gas_node_id: Gas junction node ID (fuel supply side).
        diameter_m (float): Internal pipe diameter of the water-side branch in metres.
        efficiency_power (float): Electrical efficiency in ``(0, 1]``.
        efficiency_heat (float): Thermal efficiency in ``(0, 1]``.
        mass_flow_setpoint (float): Target water-side mass flow in kg/s.
        regulation (float, optional): Dispatch fraction in ``[0.0, 1.0]``.
            ``1.0`` = full setpoint, ``0.0`` = off. Defaults to 1.
        constraints (list, optional): Constraint callables.
        remove_existing_branch (bool, optional): If ``True``, remove any existing branch
            between *heat_node_id* and *heat_return_node_id* before adding the CHP.
            Defaults to ``False``.

    Returns:
        int: The compound ID of the created CHP unit.
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
    heat_energy_w,
    diameter_m,
    efficiency,
    temperature_ext_k=293,
    q_mvar_setpoint=0,
    constraints=None,
):
    """
    Add a Power-to-Heat (P2H) compound to the network.

    Couples an electrical bus to a water/heat circuit.  The unit draws
    electrical power and delivers heat as a water mass flow.

    Args:
        network (mm.Network): Target network.
        power_node_id: Electrical bus node ID (power consumption side).
        heat_node_id: Water junction node ID (heat supply side).
        heat_return_node_id: Water junction node ID (heat return side).
        heat_energy_w (float): Heat output setpoint in **watts**.
        diameter_m (float): Inner diameter of the internal water-side branch in metres.
        efficiency (float): Electrical-to-heat conversion efficiency in ``(0, 1]``.
        temperature_ext_k (float, optional): Ambient temperature in Kelvin. Defaults to 293.
        q_mvar_setpoint (float, optional): Reactive power consumed from the electrical
            bus in Mvar. Defaults to 0.
        constraints (list, optional): Constraint callables.

    Returns:
        int: The compound ID of the created P2H unit.
    """
    return network.compound(
        mm.PowerToHeat(
            heat_energy_w=heat_energy_w,
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
        network (mm.Network): Target network.
        gas_node_id: Gas junction node ID (fuel supply side).
        heat_node_id: Water junction node ID (heat supply side).
        heat_return_node_id: Water junction node ID (heat return side).
        heat_energy_w (float): Heat output setpoint in **watts**.
        diameter_m (float): Inner diameter of the internal water-side branch in metres.
        efficiency (float): Gas-to-heat conversion efficiency in ``(0, 1]``.
        temperature_ext_k (float, optional): Ambient temperature in Kelvin. Defaults to 293.
        constraints (list, optional): Constraint callables.

    Returns:
        int: The compound ID of the created G2H unit.
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


def create_grid_forming_generator(
    network: mm.Network,
    node_id,
    p_mw_max: float,
    q_mvar_max: float,
    vm_pu: float = 1.0,
    constraints=None,
    overwrite_id=None,
    name=None,
):
    """
    Attach a grid-forming generator to an electricity bus, making it the slack
    node for its island.

    Unlike ``create_power_generator`` (fixed p/q setpoint), a grid-forming
    generator has *variable* active and reactive power so it can absorb any
    supply–demand imbalance within its island.  Its voltage magnitude is pinned
    to ``vm_pu``; the voltage angle is pinned to 0 by the islanding formulation.

    Requires ``enable_islanding(net, electricity=True)`` (or a custom
    ``ElectricityIslandingMode``) to be active on the network.

    Args:
        network (mm.Network): Target network.
        node_id: Bus identifier where the generator is attached.
        p_mw_max (float): Maximum active power injection/absorption in MW.
        q_mvar_max (float): Maximum reactive power injection/absorption in Mvar.
        vm_pu (float, optional): Voltage magnitude setpoint in per-unit. Defaults to 1.0.
        constraints: Operational constraints for the child component.
        overwrite_id: Custom identifier for the child, overriding the auto-assigned one.
        name (str, optional): Human-readable name.

    Returns:
        GridFormingGenerator: The created component, already attached to the network.

    Examples:
        Add an islanded generator that can supply up to 5 MW::

            gf_gen = mx.create_grid_forming_generator(
                net, bus_island, p_mw_max=5.0, q_mvar_max=2.0
            )
    """
    from monee.model.islanding import GridFormingGenerator

    return create_el_child(
        network,
        GridFormingGenerator(p_mw_max=p_mw_max, q_mvar_max=q_mvar_max, vm_pu=vm_pu),
        node_id=node_id,
        constraints=constraints,
        overwrite_id=overwrite_id,
        name=name,
    )


def create_grid_forming_source(
    network: mm.Network,
    node_id,
    pressure_pu: float = 1.0,
    t_k: float = 356.0,
    mass_flow_max: float = 1e6,
    grid_key=mm.GAS_KEY,
    constraints=None,
    overwrite_id=None,
    name=None,
):
    """
    Attach a grid-forming hydraulic source to a gas or water junction, making
    it the pressure reference for its island.

    The junction's pressure is pinned to ``pressure_pu`` and the source's
    ``mass_flow`` is a *variable* that absorbs the island's supply–demand
    imbalance, mirroring the role of ``ExtHydrGrid`` but usable on an isolated
    island.

    Requires ``enable_islanding(net, gas=True)`` or ``enable_islanding(net, water=True)``
    to be active on the network for the relevant carrier.

    Args:
        network (mm.Network): Target network.
        node_id: Junction identifier where the source is attached.
        pressure_pu (float, optional): Pressure setpoint in per-unit. Defaults to 1.0.
        t_k (float, optional): Temperature in Kelvin. Defaults to 356.0.
        mass_flow_max (float, optional): Maximum absolute mass flow in kg/s. Defaults to 1e6.
        grid_key: Grid key determining the carrier (``mm.GAS_KEY`` or ``mm.WATER_KEY``).
            Defaults to ``mm.GAS_KEY``.
        constraints: Operational constraints for the child component.
        overwrite_id: Custom identifier for the child, overriding the auto-assigned one.
        name (str, optional): Human-readable name.

    Returns:
        GridFormingSource: The created component, already attached to the network.

    Examples:
        Add an islanded gas source at 1.0 pu pressure::

            gf_src = mx.create_grid_forming_source(net, junction_island, pressure_pu=1.0)
    """
    from monee.model.islanding import GridFormingSource

    return network.child_to(
        GridFormingSource(
            pressure_pu=pressure_pu, t_k=t_k, mass_flow_max=mass_flow_max
        ),
        node_id=node_id,
        constraints=constraints,
        overwrite_id=overwrite_id,
        name=name,
        auto_node_creator=mm.Junction,
        auto_grid_key=grid_key,
    )


def create_gas_grid_forming_source(
    network: mm.Network,
    node_id,
    pressure_pu: float = 1.0,
    t_k: float = 356.0,
    mass_flow_max: float = 1e6,
    constraints=None,
    overwrite_id=None,
    name=None,
):
    """
    Attach a grid-forming source to a gas junction.  Shortcut for
    ``create_grid_forming_source(..., grid_key=mm.GAS_KEY)``.

    Args:
        network (mm.Network): Target network.
        node_id: Gas junction identifier.
        pressure_pu (float, optional): Pressure setpoint in per-unit. Defaults to 1.0.
        t_k (float, optional): Gas temperature in Kelvin. Defaults to 356.0.
        mass_flow_max (float, optional): Maximum mass flow in kg/s. Defaults to 1e6.
        constraints: Operational constraints for the child component.
        overwrite_id: Custom identifier for the child.
        name (str, optional): Human-readable name.

    Returns:
        GridFormingSource: The created component.
    """
    return create_grid_forming_source(
        network,
        node_id,
        pressure_pu=pressure_pu,
        t_k=t_k,
        mass_flow_max=mass_flow_max,
        grid_key=mm.GAS_KEY,
        constraints=constraints,
        overwrite_id=overwrite_id,
        name=name,
    )


def create_water_grid_forming_source(
    network: mm.Network,
    node_id,
    pressure_pu: float = 1.0,
    t_k: float = 356.0,
    mass_flow_max: float = 1e6,
    constraints=None,
    overwrite_id=None,
    name=None,
):
    """
    Attach a grid-forming source to a water/heat junction.  Shortcut for
    ``create_grid_forming_source(..., grid_key=mm.WATER_KEY)``.

    Args:
        network (mm.Network): Target network.
        node_id: Water junction identifier.
        pressure_pu (float, optional): Pressure setpoint in per-unit. Defaults to 1.0.
        t_k (float, optional): Water temperature in Kelvin. Defaults to 356.0.
        mass_flow_max (float, optional): Maximum mass flow in kg/s. Defaults to 1e6.
        constraints: Operational constraints for the child component.
        overwrite_id: Custom identifier for the child.
        name (str, optional): Human-readable name.

    Returns:
        GridFormingSource: The created component.
    """
    return create_grid_forming_source(
        network,
        node_id,
        pressure_pu=pressure_pu,
        t_k=t_k,
        mass_flow_max=mass_flow_max,
        grid_key=mm.WATER_KEY,
        constraints=constraints,
        overwrite_id=overwrite_id,
        name=name,
    )


def create_multi_energy_network():
    """
    Create an empty multi-energy :class:`mm.Network` container.

    This is a convenience alias for ``mm.Network()`` that makes the intent
    explicit in user code.  The returned network supports all three carriers
    (electricity, gas, water/heat) and can be populated with any combination
    of nodes, branches, children, and compounds.

    Returns:
        mm.Network: A new, empty network.

    Example::

        net = create_multi_energy_network()
        bus = create_bus(net)
        create_ext_power_grid(net, bus)
    """
    return mm.Network()

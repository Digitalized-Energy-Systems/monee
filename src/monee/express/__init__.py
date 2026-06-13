import monee.model as mm
from monee.model.extension.islanding import GridFormingGenerator, GridFormingSource


def create_bus(
    network: mm.Network,
    base_kv=1,
    constraints=None,
    grid=mm.EL,
    overwrite_id=None,
    name=None,
    position=None,
):
    """Add a Bus node. Returns the node id."""
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
    """Add a water/heat junction. Returns the node id."""
    return create_junction(network, grid, constraints, overwrite_id, name, position)


def create_gas_junction(
    network: mm.Network,
    grid=mm.GAS,
    constraints=None,
    overwrite_id=None,
    name=None,
    position=None,
):
    """Add a gas junction. Returns the node id."""
    return create_junction(network, grid, constraints, overwrite_id, name, position)


def create_junction(
    network: mm.Network,
    grid,
    constraints=None,
    overwrite_id=None,
    name=None,
    position=None,
):
    """Add a Junction node on *grid*. Returns the node id."""
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
    """Add an electrical branch using *model*. Returns the branch id tuple."""
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
    """Add a :class:`PowerLine` between two buses; missing buses are auto-created."""
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
    roughness_m=1e-05,
    on_off=1,
    constraints=None,
    grid=None,
    name=None,
):
    """Add a :class:`GasPipe`; missing junctions are auto-created."""
    return network.branch(
        mm.GasPipe(diameter_m, length_m, temperature_ext_k, roughness_m, on_off=on_off),
        from_node_id=from_node_id,
        to_node_id=to_node_id,
        constraints=constraints,
        grid=grid,
        name=name,
        auto_node_creator=lambda: mm.Junction(),
        auto_grid_key=mm.GAS_KEY,
    )


def create_compressor(
    network: mm.Network,
    from_node_id,
    to_node_id,
    compression_ratio=1.5,
    max_flow_kgs=10.0,
    grid=None,
    name=None,
):
    """Add a :class:`GasCompressor` (unidirectional, fixed ratio)."""
    return network.branch(
        mm.GasCompressor(compression_ratio, max_flow_kgs),
        from_node_id=from_node_id,
        to_node_id=to_node_id,
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
    roughness_m=0.001,
    lambda_insulation_w_per_m_k=0.025,
    insulation_thickness_m=0.2,
    on_off=1,
    unidirectional=False,
    constraints=None,
    grid=None,
    name=None,
):
    """Add a :class:`WaterPipe`. ``unidirectional=True`` pins direction=0
    (drops the binary on DH trunks with known flow direction)."""
    return network.branch(
        mm.WaterPipe(
            diameter_m,
            length_m,
            temperature_ext_k=temperature_ext_k,
            roughness_m=roughness_m,
            lambda_insulation_w_per_m_k=lambda_insulation_w_per_m_k,
            insulation_thickness_m=insulation_thickness_m,
            on_off=on_off,
            unidirectional=unidirectional,
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
    """Attach an electrical child *model* to ``node_id``; missing buses auto-created."""
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
    """Attach a hydraulic child *model* to a water/heat junction."""
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
    """Attach a gas child *model* to a junction."""
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
    """Attach a :class:`PowerLoad` (positive = consumption)."""
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
    """Attach a :class:`PowerGenerator`. Pass positive magnitudes; the
    constructor negates internally."""
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
    max_import_mw=None,
    max_export_mw=None,
    constraints=None,
    overwrite_id=None,
    name=None,
    **kwargs,
):
    """Attach an :class:`ExtPowerGrid` (slack bus, pinned vm_pu/va_degree)."""
    return create_el_child(
        network,
        mm.ExtPowerGrid(
            p_mw,
            q_mvar,
            vm_pu,
            va_degree,
            max_import_mw=max_import_mw,
            max_export_mw=max_export_mw,
            **kwargs,
        ),
        node_id=node_id,
        constraints=constraints,
        overwrite_id=overwrite_id,
        name=name,
    )


def create_ext_hydr_grid(
    network: mm.Network,
    node_id,
    mass_flow_kgs=1,
    pressure_pu=1,
    t_k=356,
    grid_key=mm.GAS_KEY,
    max_import_kgs=None,
    max_export_kgs=None,
    pin_temperature=True,
    constraints=None,
    overwrite_id=None,
    name=None,
    **kwargs,
):
    """Attach an :class:`ExtHydrGrid` (slack source). Prefer the carrier-specific
    :func:`create_gas_ext_grid` / :func:`create_water_ext_grid` shortcuts."""
    return network.child_to(
        mm.ExtHydrGrid(
            mass_flow_kgs=mass_flow_kgs,
            pressure_pu=pressure_pu,
            t_k=t_k,
            max_import_kgs=max_import_kgs,
            max_export_kgs=max_export_kgs,
            pin_temperature=pin_temperature,
            **kwargs,
        ),
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
    mass_flow_kgs=1,
    grid_key=mm.GAS_KEY,
    constraints=None,
    overwrite_id=None,
    name=None,
    **kwargs,
):
    """Attach a :class:`Source`. Pass positive magnitude; constructor negates internally."""
    return network.child_to(
        mm.Source(mass_flow_kgs, **kwargs),
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
    mass_flow_kgs=1,
    pressure_pu=1,
    t_k=293,
    grid_key=mm.WATER_KEY,
    constraints=None,
    overwrite_id=None,
    name=None,
    **kwargs,
):
    """Attach a :class:`ConsumeHydrGrid` (fixed-pressure consumption point).

    ``pressure_pu`` is per-unit (relative to the grid's reference pressure),
    matching :func:`create_ext_hydr_grid` and the model's own convention."""
    return network.child_to(
        mm.ConsumeHydrGrid(
            mass_flow_kgs=mass_flow_kgs, pressure_pu=pressure_pu, t_k=t_k, **kwargs
        ),
        node_id=node_id,
        constraints=constraints,
        overwrite_id=overwrite_id,
        name=name,
        auto_node_creator=mm.Junction,
        auto_grid_key=grid_key,
    )


def create_sink(
    network: mm.Network,
    node_id,
    mass_flow_kgs=1,
    grid_key=mm.GAS_KEY,
    constraints=None,
    overwrite_id=None,
    name=None,
    **kwargs,
):
    """Attach a :class:`Sink` (positive = consumption)."""
    return network.child_to(
        mm.Sink(mass_flow_kgs=mass_flow_kgs, **kwargs),
        node_id=node_id,
        constraints=constraints,
        overwrite_id=overwrite_id,
        name=name,
        auto_node_creator=mm.Junction,
        auto_grid_key=grid_key,
    )


def create_heat_generator(
    network: mm.Network,
    node_id,
    q_mw,
    constraints=None,
    overwrite_id=None,
    name=None,
    **kwargs,
):
    """Attach a :class:`HeatGenerator` to a water junction. Positive magnitude."""
    return create_water_child(
        network,
        mm.HeatGenerator(q_mw=q_mw, **kwargs),
        node_id=node_id,
        constraints=constraints,
        overwrite_id=overwrite_id,
        name=name,
    )


def create_heat_load(
    network: mm.Network,
    node_id,
    q_mw,
    constraints=None,
    overwrite_id=None,
    name=None,
    **kwargs,
):
    """Attach a :class:`HeatLoad` to a water junction (positive = consumption)."""
    return create_water_child(
        network,
        mm.HeatLoad(q_mw=q_mw, **kwargs),
        node_id=node_id,
        constraints=constraints,
        overwrite_id=overwrite_id,
        name=name,
    )


def create_heat_exchanger(
    network: mm.Network,
    from_node_id,
    to_node_id,
    q_mw,
    regulation=1,
    constraints=None,
    grid=None,
    name=None,
):
    """Add a heat-exchanger branch. ``q_mw > 0`` → :class:`HeatExchangerLoad`;
    ``q_mw < 0`` → :class:`HeatExchangerGenerator`."""
    return network.branch(
        mm.HeatExchangerLoad(q_mw=-q_mw, regulation=regulation)
        if q_mw > 0
        else mm.HeatExchangerGenerator(q_mw=-q_mw, regulation=regulation),
        from_node_id=from_node_id,
        to_node_id=to_node_id,
        constraints=constraints,
        grid=grid,
        name=name,
        auto_node_creator=mm.Junction,
        auto_grid_key=mm.WATER_KEY,
    )


def create_passive_heat_exchanger(
    network: mm.Network,
    from_node_id,
    to_node_id,
    q_mw,
    diameter_m,
    temperature_ext_k=293,
    constraints=None,
    grid=None,
    name=None,
):
    """Add a passive heat-exchanger branch: a fixed ``q_mw`` injected into or
    extracted from the free-flowing water stream, with the temperature change
    following from the actual mass flow (the surrounding hydraulics determine
    the flow). Use this for in-line consumers/sources on a distribution run;
    use :func:`create_heat_exchanger` for supply/return exchangers that drive
    their design mass flow. ``q_mw > 0`` → :class:`PassiveHeatExchangerLoad`;
    ``q_mw < 0`` → :class:`PassiveHeatExchangerGenerator`."""
    # Pass -q_mw to match create_heat_exchanger: both model bases store
    # q_mw_set = -q_mw, so a positive public q_mw must be negated here too for
    # the active and passive wrappers to agree on the load/generator sign.
    return network.branch(
        mm.PassiveHeatExchangerLoad(-q_mw, diameter_m, temperature_ext_k)
        if q_mw > 0
        else mm.PassiveHeatExchangerGenerator(-q_mw, diameter_m, temperature_ext_k),
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
    mass_flow_setpoint_kgs,
    consume_q_mvar_setpoint=0,
    regulation=1,
    constraints=None,
    grid=None,
    name=None,
):
    """Add a Power-to-Gas coupling branch (power → gas)."""
    return network.branch(
        mm.PowerToGas(
            efficiency=efficiency,
            mass_flow_setpoint_kgs=mass_flow_setpoint_kgs,
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
    """Add a Gas-to-Power coupling branch (gas → power)."""
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
    mass_flow_setpoint_kgs,
    regulation=1,
    constraints=None,
    remove_existing_branch=False,
):
    """Add a CHP compound (gas → power + heat via internal HX branch)."""
    if remove_existing_branch:
        network.remove_branch_between(heat_node_id, heat_return_node_id)
    return network.compound(
        mm.CHP(
            diameter_m,
            efficiency_power,
            efficiency_heat,
            mass_flow_setpoint_kgs,
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
    regulation=1,
    constraints=None,
):
    """Add a Power-to-Heat compound with an internal water HX branch."""
    return network.compound(
        mm.PowerToHeat(
            heat_energy_mw=heat_energy_mw,
            diameter_m=diameter_m,
            temperature_ext_k=temperature_ext_k,
            efficiency=efficiency,
            q_mvar_setpoint=q_mvar_setpoint,
            regulation=regulation,
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
    heat_energy_mw,
    diameter_m,
    efficiency,
    temperature_ext_k=293,
    constraints=None,
):
    """Add a Gas-to-Heat compound with an internal water HX branch."""
    return network.compound(
        mm.GasToHeat(
            heat_energy_mw=heat_energy_mw,
            diameter_m=diameter_m,
            temperature_ext_k=temperature_ext_k,
            efficiency=efficiency,
        ),
        constraints=constraints,
        gas_node_id=gas_node_id,
        heat_node_id=heat_node_id,
        heat_return_node_id=heat_return_node_id,
    )


def create_chp_hg(
    network: mm.Network,
    power_node_id,
    heat_node_id,
    gas_node_id,
    efficiency_power,
    efficiency_heat,
    mass_flow_setpoint_kgs,
    regulation=1,
    constraints=None,
):
    """CHP variant using a SubHG node-based heat injector (no heat_return / diameter)."""
    return network.compound(
        mm.CHPHG(
            efficiency_power=efficiency_power,
            efficiency_heat=efficiency_heat,
            mass_flow_setpoint_kgs=mass_flow_setpoint_kgs,
            q_mvar_setpoint=0,
            regulation=regulation,
        ),
        constraints=constraints,
        power_node_id=power_node_id,
        heat_node_id=heat_node_id,
        gas_node_id=gas_node_id,
    )


def create_p2h_hg(
    network: mm.Network,
    power_node_id,
    heat_node_id,
    heat_energy_mw,
    efficiency,
    q_mvar_setpoint=0,
    constraints=None,
):
    """Single-branch P2H using :class:`PowerToHeatHG` (heat injected at the to-node)."""
    return network.branch(
        mm.PowerToHeatHG(
            heat_energy_mw=heat_energy_mw,
            efficiency=efficiency,
            q_mvar_setpoint=q_mvar_setpoint,
        ),
        from_node_id=power_node_id,
        to_node_id=heat_node_id,
        constraints=constraints,
    )


def create_g2h_hg(
    network: mm.Network,
    gas_node_id,
    heat_node_id,
    heat_energy_mw,
    efficiency,
    constraints=None,
):
    """Single-branch G2H using :class:`GasToHeatHG` (heat injected at the to-node)."""
    return network.branch(
        mm.GasToHeatHG(
            heat_energy_mw=heat_energy_mw,
            efficiency=efficiency,
        ),
        from_node_id=gas_node_id,
        to_node_id=heat_node_id,
        constraints=constraints,
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
    """Add a two-winding :class:`Trafo`. from-node = LV side, to-node = HV side."""
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
    mass_flow_kgs=1,
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
        mass_flow_kgs=mass_flow_kgs,
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
    mass_flow_kgs=1,
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
        mass_flow_kgs=mass_flow_kgs,
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
    mass_flow_kgs=1,
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
        mass_flow_kgs=mass_flow_kgs,
        grid_key=mm.GAS_KEY,
        constraints=constraints,
        overwrite_id=overwrite_id,
        name=name,
        **kwargs,
    )


def create_gas_sink(
    network: mm.Network,
    node_id,
    mass_flow_kgs=1,
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
        mass_flow_kgs=mass_flow_kgs,
        grid_key=mm.GAS_KEY,
        constraints=constraints,
        overwrite_id=overwrite_id,
        name=name,
        **kwargs,
    )


def create_water_source(
    network: mm.Network,
    node_id,
    mass_flow_kgs=1,
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
        mass_flow_kgs=mass_flow_kgs,
        grid_key=mm.WATER_KEY,
        constraints=constraints,
        overwrite_id=overwrite_id,
        name=name,
        **kwargs,
    )


def create_water_sink(
    network: mm.Network,
    node_id,
    mass_flow_kgs=1,
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
        mass_flow_kgs=mass_flow_kgs,
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
    """Attach a :class:`GridFormingGenerator` (island slack bus). Requires
    ``enable_islanding(net, electricity=True)``."""
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
    mass_flow_max_kgs: float = 1e6,
    grid_key=mm.GAS_KEY,
    constraints=None,
    overwrite_id=None,
    name=None,
):
    """Attach a :class:`GridFormingSource` (island pressure reference). Requires
    ``enable_islanding`` for the relevant carrier."""
    return network.child_to(
        GridFormingSource(
            pressure_pu=pressure_pu, t_k=t_k, mass_flow_max_kgs=mass_flow_max_kgs
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
    mass_flow_max_kgs: float = 1e6,
    constraints=None,
    overwrite_id=None,
    name=None,
):
    """Gas-side shortcut for :func:`create_grid_forming_source`."""
    return create_grid_forming_source(
        network,
        node_id,
        pressure_pu=pressure_pu,
        t_k=t_k,
        mass_flow_max_kgs=mass_flow_max_kgs,
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
    mass_flow_max_kgs: float = 1e6,
    constraints=None,
    overwrite_id=None,
    name=None,
):
    """Water-side shortcut for :func:`create_grid_forming_source`."""
    return create_grid_forming_source(
        network,
        node_id,
        pressure_pu=pressure_pu,
        t_k=t_k,
        mass_flow_max_kgs=mass_flow_max_kgs,
        grid_key=mm.WATER_KEY,
        constraints=constraints,
        overwrite_id=overwrite_id,
        name=name,
    )


def create_multi_energy_network():
    """Empty multi-energy :class:`mm.Network` (convenience alias for ``mm.Network()``)."""
    return mm.Network()


from monee.express.structures import (  # noqa: E402
    DhsSegment,
    DhsStructure,
    ElStructure,
    GasStructure,
    Segment,
    StarSegment,
    WaterStructure,
    dhs_structure,
    el_structure,
    gas_structure,
    water_structure,
)

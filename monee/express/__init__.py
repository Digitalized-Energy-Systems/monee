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
    return create_junction(
        network,
        grid,
        constraints,
        overwrite_id,
        name,
        position,
    )


def create_gas_junction(
    network: mm.Network,
    grid=mm.GAS,
    constraints=None,
    overwrite_id=None,
    name=None,
    position=None,
):
    return create_junction(
        network,
        grid,
        constraints,
        overwrite_id,
        name,
        position,
    )


def create_junction(
    network: mm.Network,
    grid,
    constraints=None,
    overwrite_id=None,
    name=None,
    position=None,
):
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
):
    return network.branch(
        mm.PowerLine(length_m, r_ohm_per_m, x_ohm_per_m, parallel),
        from_node_id=from_node_id,
        to_node_id=to_node_id,
        constraints=constraints,
        grid=grid,
        name=name,
    )


def create_gas_pipe(
    network: mm.Network,
    from_node_id,
    to_node_id,
    diameter_m,
    length_m,
    temperature_ext_k=296.15,
    roughness=0.00001,
    constraints=None,
    grid=None,
    name=None,
):
    return network.branch(
        mm.GasPipe(diameter_m, length_m, temperature_ext_k, roughness),
        from_node_id=from_node_id,
        to_node_id=to_node_id,
        constraints=constraints,
        grid=grid,
        name=name,
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
    constraints=None,
    grid=None,
    name=None,
):
    return network.branch(
        mm.WaterPipe(
            diameter_m,
            length_m,
            temperature_ext_k=temperature_ext_k,
            roughness=roughness,
            lambda_insulation_w_per_k=lambda_insulation_w_per_k,
            insulation_thickness_m=insulation_thickness_m,
        ),
        from_node_id=from_node_id,
        to_node_id=to_node_id,
        constraints=constraints,
        grid=grid,
        name=name,
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
    return network.child_to(
        model,
        node_id=node_id,
        constraints=constraints,
        overwrite_id=overwrite_id,
        name=name,
        auto_node_type=mm.Junction,
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
    return network.child_to(
        model,
        node_id=node_id,
        constraints=constraints,
        overwrite_id=overwrite_id,
        name=name,
        auto_node_type=mm.Junction,
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
    p_mw=1,
    q_mvar=1,
    vm_pu=1,
    va_degree=0,
    constraints=None,
    overwrite_id=None,
    name=None,
    **kwargs,
):
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
    pressure_pa=1000000,
    t_k=300,
    constraints=None,
    overwrite_id=None,
    name=None,
    **kwargs,
):
    return network.child_to(
        mm.ExtHydrGrid(mass_flow=mass_flow, pressure_pa=pressure_pa, t_k=t_k, **kwargs),
        node_id=node_id,
        constraints=constraints,
        overwrite_id=overwrite_id,
        name=name,
    )


def create_source(
    network: mm.Network,
    node_id,
    mass_flow=1,
    constraints=None,
    overwrite_id=None,
    name=None,
    **kwargs,
):
    return network.child_to(
        mm.Source(mass_flow, **kwargs),
        node_id=node_id,
        constraints=constraints,
        overwrite_id=overwrite_id,
        name=name,
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
    constraints=None,
    overwrite_id=None,
    name=None,
    **kwargs,
):
    return network.child_to(
        mm.Sink(mass_flow=mass_flow, **kwargs),
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
    diameter_m=0.10,
    temperature_ext_k=293,
    constraints=None,
    grid=None,
    name=None,
):
    return network.branch(
        mm.HeatExchangerLoad(
            q_mw=-q_mw,
            diameter_m=diameter_m,
            temperature_ext_k=temperature_ext_k,
        )
        if q_mw < 0
        else mm.HeatExchanger(
            q_mw=q_mw,
            diameter_m=diameter_m,
            temperature_ext_k=temperature_ext_k,
        ),
        from_node_id=from_node_id,
        to_node_id=to_node_id,
        constraints=constraints,
        grid=grid,
        name=name,
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
    return network.branch(
        mm.PowerToGas(
            efficiency=efficiency,
            mass_flow_setpoint=mass_flow_setpoint,
            consume_q_mvar_setpoint=consume_q_mvar_setpoint,
            regulation=regulation
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
    return network.branch(
        mm.GasToPower(
            efficiency=efficiency,
            p_mw_setpoint=p_mw_setpoint,
            q_mvar_setpoint=q_mvar_setpoint,
            regulation=regulation
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
    constraints=None,
):
    return network.compound(
        mm.CHP(
            diameter_m,
            efficiency_power,
            efficiency_heat,
            mass_flow_setpoint,
            q_mvar_setpoint=0,
            temperature_ext_k=293,
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
    in_line_operation=False,
    constraints=None,
):
    return network.compound(
        mm.PowerToHeat(
            heat_energy_mw=heat_energy_mw,
            diameter_m=diameter_m,
            temperature_ext_k=temperature_ext_k,
            efficiency=efficiency,
            q_mvar_setpoint=q_mvar_setpoint,
            in_line_operation=in_line_operation,
        ),
        constraints=constraints,
        power_node_id=power_node_id,
        heat_node_id=heat_node_id,
        heat_return_node_id=heat_return_node_id,
    )


def create_multi_energy_network():
    return mm.Network(
        mm.create_power_grid("power"),
        mm.create_water_grid("water"),
        mm.create_gas_grid("gas"),
    )

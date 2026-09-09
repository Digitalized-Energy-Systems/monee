import monee.model as mm
from monee.model.extension.islanding import GridFormingGenerator, GridFormingSource


def create_bus(
    network: mm.Network,
    base_kv=1,
    *,
    constraints=None,
    grid=mm.EL,
    overwrite_id=None,
    name=None,
    position=None,
):
    """Add a Bus node. Returns the node id.

    ``base_kv`` defaults to 1 kV; set it to the bus's real voltage level so
    ohmic line parameters are converted to per unit correctly."""
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
    *,
    grid=mm.WATER,
    constraints=None,
    overwrite_id=None,
    name=None,
    position=None,
):
    """Add a water/heat junction. Returns the node id."""
    return create_junction(
        network,
        grid,
        constraints=constraints,
        overwrite_id=overwrite_id,
        name=name,
        position=position,
    )


def create_gas_junction(
    network: mm.Network,
    *,
    grid=mm.GAS,
    constraints=None,
    overwrite_id=None,
    name=None,
    position=None,
):
    """Add a gas junction. Returns the node id.

    Without an explicit ``grid`` the node lands on the network's default gas
    grid (lgas, 10 bar reference pressure, 300 K operating temperature)."""
    return create_junction(
        network,
        grid,
        constraints=constraints,
        overwrite_id=overwrite_id,
        name=name,
        position=position,
    )


def create_junction(
    network: mm.Network,
    grid,
    *,
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
    max_i_ka=None,
    max_s_mva=None,
):
    """Add a :class:`PowerLine` between two buses; missing buses are auto-created.
    ``max_i_ka`` / ``max_s_mva`` set the thermal rating (model defaults apply
    when omitted)."""
    rating_kwargs = {}
    if max_i_ka is not None:
        rating_kwargs["max_i_ka"] = max_i_ka
    if max_s_mva is not None:
        rating_kwargs["max_s_mva"] = max_s_mva
    return network.branch(
        mm.PowerLine(
            length_m,
            r_ohm_per_m,
            x_ohm_per_m,
            parallel,
            on_off=on_off,
            **rating_kwargs,
        ),
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
    constraints=None,
    grid=None,
    name=None,
):
    """Add a :class:`GasCompressor` (unidirectional, fixed ratio)."""
    return network.branch(
        mm.GasCompressor(compression_ratio, max_flow_kgs),
        from_node_id=from_node_id,
        to_node_id=to_node_id,
        constraints=constraints,
        grid=grid,
        name=name,
        auto_node_creator=lambda: mm.Junction(),
        auto_grid_key=mm.GAS_KEY,
    )


def create_water_pipe(  # NOSONAR
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
    cost=None,
    constraints=None,
    overwrite_id=None,
    name=None,
    **kwargs,
):
    """Attach a :class:`PowerGenerator`. Pass positive magnitudes; the
    constructor negates internally. ``cost`` (currency/MW) feeds the economic
    dispatch objective."""
    return create_el_child(
        network,
        mm.PowerGenerator(p_mw, q_mvar, cost=cost, **kwargs),
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
    regulate_vm=True,
    cost=None,
    constraints=None,
    overwrite_id=None,
    name=None,
    **kwargs,
):
    """Attach an :class:`ExtPowerGrid` (slack bus, pinned vm_pu/va_degree).

    Args:
        node_id: bus to attach to; a missing bus is auto-created.
        p_mw: initial guess for the free exchange Var, not a setpoint.
            Load convention: positive p_mw is export into the external grid,
            an import into the network is negative.
        q_mvar: initial guess for the free reactive exchange Var.
        vm_pu: voltage magnitude the slack pins the bus to (default 1).
        va_degree: pinned reference angle (default 0).
        max_import_mw: import cap; positive magnitude, bounds p_mw from
            below at -max_import_mw (load convention). Default None
            (unbounded).
        max_export_mw: export cap; positive magnitude, bounds p_mw from
            above. Default None (unbounded).
        regulate_vm: True (default, power-flow semantics) holds the bus
            voltage at vm_pu; False pins only the angle and leaves the
            magnitude an OPF decision variable in [vm_pu_min, vm_pu_max].
        cost: currency/MW price of the exchange in the economic dispatch
            objective, overriding ext_grid_cost_default for this grid.
    """
    return create_el_child(
        network,
        mm.ExtPowerGrid(
            p_mw,
            q_mvar,
            vm_pu,
            va_degree,
            max_import_mw=max_import_mw,
            max_export_mw=max_export_mw,
            regulate_vm=regulate_vm,
            cost=cost,
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
    t_k=None,
    grid_key=mm.GAS_KEY,
    max_import_kgs=None,
    max_export_kgs=None,
    pin_temperature=True,
    free_pressure_bounds=None,
    constraints=None,
    overwrite_id=None,
    name=None,
    **kwargs,
):
    """Attach an :class:`ExtHydrGrid` (slack source). Prefer the carrier-specific
    :func:`create_gas_ext_grid` / :func:`create_water_ext_grid` shortcuts.

    The slack exchanges whatever mass flow the balance needs; on a water loop
    that makes it an unlimited backup heat plant unless capped (see
    :class:`ExtHydrGrid` and the concepts/multi_energy page).

    Args:
        node_id: junction to attach to. A missing node is auto-created as a
            junction on ``grid_key`` (default ``mm.GAS_KEY``, i.e. gas).
        mass_flow_kgs: initial guess for the free exchange Var, not a
            setpoint. Load convention: negative mass_flow_kgs is injection
            into the network (import), positive is withdrawal (export).
        pressure_pu: pressure the slack pins the junction to, per unit of
            the grid's ``pressure_ref_pa`` (default gas grid: 1 pu = 10 bar).
        t_k: temperature of the injected stream in K. Default None resolves
            at solve time to the grid's own operating temperature (gas grid
            t_k, 300 K by default; water grids fall back to t_ref_k, 356 K).
            Before 2026-09 the default was a fixed 356 K for both carriers.
        grid_key: grid key used when the junction is auto-created.
        max_import_kgs: import cap; positive magnitude, bounds
            mass_flow_kgs from below at -max_import_kgs (load convention).
            Default None (unbounded). Re-read at every solve, so setting it
            on the returned child's model later takes effect on the next
            solve.
        max_export_kgs: export cap; positive magnitude, bounds
            mass_flow_kgs from above. Default None (unbounded). Re-read at
            every solve like max_import_kgs.
        pin_temperature: True (default) pins the junction temperature to
            t_k; False leaves it to the surrounding network.
        free_pressure_bounds: (lo, hi) tuple in pu; instead of pinning the
            pressure at pressure_pu, leave it a Var bounded to [lo, hi].
            Default None (pressure pinned).
    """
    return network.child_to(
        mm.ExtHydrGrid(
            mass_flow_kgs=mass_flow_kgs,
            pressure_pu=pressure_pu,
            t_k=t_k,
            max_import_kgs=max_import_kgs,
            max_export_kgs=max_export_kgs,
            pin_temperature=pin_temperature,
            free_pressure_bounds=free_pressure_bounds,
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
    t_k=None,
    constraints=None,
    overwrite_id=None,
    name=None,
    **kwargs,
):
    """Attach a :class:`Source` (fixed-setpoint injection).

    Args:
        node_id: junction to attach to. A missing node is auto-created as a
            junction on ``grid_key`` (default ``mm.GAS_KEY``, i.e. gas).
        mass_flow_kgs: injected mass flow. Pass a positive magnitude; the
            constructor negates internally (load convention).
        grid_key: grid key used when the junction is auto-created.
        t_k: temperature of the injected stream in K. Default None credits
            the injection at the junction's own mixed temperature.
    """
    return network.child_to(
        mm.Source(mass_flow_kgs, t_k=t_k, **kwargs),
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
    matching :func:`create_ext_hydr_grid` and the model's own convention.

    When ``node_id`` does not exist yet, the auto-created junction is a water
    junction (default ``grid_key=mm.WATER_KEY``)."""
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
    """Attach a :class:`Sink` (positive = consumption).

    When ``node_id`` does not exist yet, the auto-created junction is a gas
    junction (default ``grid_key=mm.GAS_KEY``)."""
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
    """Attach a :class:`HeatGenerator` to a water junction. Positive magnitude.

    Part load in one line: drive the stored ``q_mw_heat`` attribute; a nodal
    heat source has no mass flow of its own, so the exchangers' constant-flow
    vs variable-flow distinction does not apply."""
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
    """Attach a :class:`HeatLoad` to a water junction (positive = consumption).

    Part load in one line: drive the stored ``q_mw_heat`` attribute; a nodal
    heat load has no mass flow of its own, so the exchangers' constant-flow
    vs variable-flow distinction does not apply."""
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
    mass_flow_design_kgs=None,
    constraints=None,
    grid=None,
    name=None,
):
    """Add a heat-exchanger branch. ``q_mw > 0`` gives a
    :class:`HeatExchangerLoad` (consumption); ``q_mw < 0`` a
    :class:`HeatExchangerGenerator` (injection).

    This is the fixed-flow exchanger: a numeric ``q_mw`` pins the branch mass
    flow to the design flow ``|q_mw| * 1e6 / (c_p * T_delta_design_K)`` (a 30 K
    design spread by default), scaled by ``regulation``, so the branch prescribes the
    loop hydraulics instead of following them. Two of them on the same closed
    loop over-determine it. Use :func:`create_passive_heat_exchanger` for
    in-line consumers whose flow should follow the surrounding hydraulics.

    ``mass_flow_design_kgs`` prescribes the driven mass flow in kg/s directly
    instead of deriving it from ``q_mw`` at the design spread; it is stored on
    the model under the same name, so a timeseries can drive it too.

    Part load in one line: driving ``regulation`` scales duty and mass flow
    together (variable flow at the design temperature spread), while driving
    ``q_mw_set`` changes the duty at the constant design mass flow (the
    temperature spread moves instead of the flow)."""
    return network.branch(
        mm.HeatExchangerLoad(
            q_mw=-q_mw,
            mass_flow_design_kgs=mass_flow_design_kgs,
            regulation=regulation,
        )
        if q_mw > 0
        else mm.HeatExchangerGenerator(
            q_mw=-q_mw,
            mass_flow_design_kgs=mass_flow_design_kgs,
            regulation=regulation,
        ),
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
    their design mass flow. ``q_mw > 0`` gives a
    :class:`PassiveHeatExchangerLoad` (consumption); ``q_mw < 0`` a
    :class:`PassiveHeatExchangerGenerator` (injection).

    Part load in one line: drive ``q_mw_set`` or ``regulation`` (they multiply
    into the duty); the mass flow follows the surrounding hydraulics either
    way, so there is no constant-flow vs variable-flow choice here."""
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
    """Add a Power-to-Gas coupling branch (from-node power, to-node gas).

    ``mass_flow_setpoint_kgs`` is the positive produced gas flow; the branch
    draws the corresponding electrical power (load convention) at the
    from-node. ``regulation`` scales the setpoint (default 1)."""
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
    """Add a Gas-to-Power coupling branch (from-node gas, to-node power).

    ``p_mw_setpoint`` is the positive produced electrical power; the branch
    draws the corresponding gas mass flow at the from-node. ``regulation``
    scales the setpoint (default 1)."""
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


def _resolve_heat_sides(heat_node_id, heat_return_node_id, cold_node_id, hot_node_id):
    if heat_node_id is not None and cold_node_id is not None:
        raise ValueError("Pass either heat_node_id or cold_node_id, not both.")
    if heat_return_node_id is not None and hot_node_id is not None:
        raise ValueError("Pass either heat_return_node_id or hot_node_id, not both.")
    cold = heat_node_id if heat_node_id is not None else cold_node_id
    hot = heat_return_node_id if heat_return_node_id is not None else hot_node_id
    if cold is None or hot is None:
        raise ValueError(
            "The heat side needs both junctions: heat_node_id/cold_node_id (inlet) "
            "and heat_return_node_id/hot_node_id (heated outlet)."
        )
    return cold, hot


def _require_arguments(func_name, **values):
    missing = [name for name, value in values.items() if value is None]
    if missing:
        raise TypeError(f"{func_name}() is missing arguments: {', '.join(missing)}")


def create_chp(
    network: mm.Network,
    power_node_id,
    heat_node_id=None,
    heat_return_node_id=None,
    gas_node_id=None,
    diameter_m=None,
    efficiency_power=None,
    efficiency_heat=None,
    mass_flow_setpoint_kgs=None,
    regulation=1,
    constraints=None,
    remove_existing_branch=False,
    cold_node_id=None,
    hot_node_id=None,
    name=None,
):
    """Add a CHP compound (gas to power plus heat via an internal HX branch).

    Water enters the internal heat exchanger at ``heat_node_id`` and leaves it at
    ``heat_return_node_id``, gaining the thermal output on the way, so
    ``heat_return_node_id`` is the HOT outlet and ``heat_node_id`` the cold inlet.
    The names describe the junctions of a supply/return pair, not their
    temperatures; ``cold_node_id`` / ``hot_node_id`` are aliases that say which
    is which. The multi-energy concepts page has the full loop pattern.

    Args:
        power_node_id: bus receiving the electrical output (injected with
            the generator sign, i.e. negative p_mw under the load
            convention).
        heat_node_id: cold inlet junction of the internal heat exchanger
            (alias: cold_node_id; pass one of the two, required).
        heat_return_node_id: hot outlet junction of the internal heat
            exchanger (alias: hot_node_id; pass one of the two, required).
        gas_node_id: junction the unit draws gas from (required).
        diameter_m: internal HX branch diameter in m (required).
        efficiency_power: fraction of the gas energy converted to
            electricity (required).
        efficiency_heat: fraction of the gas energy converted to heat
            (required).
        mass_flow_setpoint_kgs: positive gas consumption setpoint in kg/s
            (required).
        regulation: scales the setpoint; 1 (default) is full dispatch, 0 off.
        remove_existing_branch: drop an existing branch between
            heat_node_id and heat_return_node_id before inserting the HX.
        cold_node_id: alias for heat_node_id.
        hot_node_id: alias for heat_return_node_id.
        name: compound name, resolvable by
            :meth:`TimeseriesData.add_compound_series_by_name`.
    """
    heat_node_id, heat_return_node_id = _resolve_heat_sides(
        heat_node_id, heat_return_node_id, cold_node_id, hot_node_id
    )
    _require_arguments(
        "create_chp",
        gas_node_id=gas_node_id,
        diameter_m=diameter_m,
        efficiency_power=efficiency_power,
        efficiency_heat=efficiency_heat,
        mass_flow_setpoint_kgs=mass_flow_setpoint_kgs,
    )
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
        name=name,
        power_node_id=power_node_id,
        heat_node_id=heat_node_id,
        heat_return_node_id=heat_return_node_id,
        gas_node_id=gas_node_id,
    )


def create_p2h(
    network: mm.Network,
    power_node_id,
    heat_node_id=None,
    heat_return_node_id=None,
    heat_energy_mw=None,
    diameter_m=None,
    efficiency=None,
    temperature_ext_k=293,
    q_mvar_setpoint=0,
    regulation=1,
    constraints=None,
    cold_node_id=None,
    hot_node_id=None,
    name=None,
):
    """Add a Power-to-Heat compound with an internal water HX branch.

    Water enters the internal heat exchanger at ``heat_node_id`` and leaves it at
    ``heat_return_node_id``, gaining the thermal output on the way, so
    ``heat_return_node_id`` is the HOT outlet and ``heat_node_id`` the cold inlet.
    The names describe the junctions of a supply/return pair, not their
    temperatures; ``cold_node_id`` / ``hot_node_id`` are aliases that say which
    is which. The multi-energy concepts page has the full loop pattern.

    Args:
        power_node_id: bus the unit draws ``heat_energy_mw / efficiency``
            from (positive consumption, load convention).
        heat_node_id: cold inlet junction of the internal heat exchanger
            (alias: cold_node_id; pass one of the two, required).
        heat_return_node_id: hot outlet junction of the internal heat
            exchanger (alias: hot_node_id; pass one of the two, required).
        heat_energy_mw: positive thermal output in MW (required).
        diameter_m: internal HX branch diameter in m (required).
        efficiency: electric-to-heat conversion efficiency (required).
        temperature_ext_k: ambient temperature of the internal HX branch in
            K (default 293).
        q_mvar_setpoint: reactive power drawn at the power node (default 0).
        regulation: scales the setpoint; 1 (default) is full dispatch, 0 off.
        cold_node_id: alias for heat_node_id.
        hot_node_id: alias for heat_return_node_id.
        name: compound name, resolvable by
            :meth:`TimeseriesData.add_compound_series_by_name`.
    """
    heat_node_id, heat_return_node_id = _resolve_heat_sides(
        heat_node_id, heat_return_node_id, cold_node_id, hot_node_id
    )
    _require_arguments(
        "create_p2h",
        heat_energy_mw=heat_energy_mw,
        diameter_m=diameter_m,
        efficiency=efficiency,
    )
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
        name=name,
        power_node_id=power_node_id,
        heat_node_id=heat_node_id,
        heat_return_node_id=heat_return_node_id,
    )


def create_g2h(
    network: mm.Network,
    gas_node_id,
    heat_node_id=None,
    heat_return_node_id=None,
    heat_energy_mw=None,
    diameter_m=None,
    efficiency=None,
    temperature_ext_k=293,
    regulation=1,
    constraints=None,
    cold_node_id=None,
    hot_node_id=None,
    name=None,
):
    """Add a Gas-to-Heat compound with an internal water HX branch.

    Water enters the internal heat exchanger at ``heat_node_id`` and leaves it at
    ``heat_return_node_id``, gaining the thermal output on the way, so
    ``heat_return_node_id`` is the HOT outlet and ``heat_node_id`` the cold inlet.
    The names describe the junctions of a supply/return pair, not their
    temperatures; ``cold_node_id`` / ``hot_node_id`` are aliases that say which
    is which. The multi-energy concepts page has the full loop pattern.

    Args:
        gas_node_id: junction the unit draws gas from (positive
            consumption, load convention).
        heat_node_id: cold inlet junction of the internal heat exchanger
            (alias: cold_node_id; pass one of the two, required).
        heat_return_node_id: hot outlet junction of the internal heat
            exchanger (alias: hot_node_id; pass one of the two, required).
        heat_energy_mw: positive thermal output in MW (required).
        diameter_m: internal HX branch diameter in m (required).
        efficiency: gas-to-heat conversion efficiency (required).
        temperature_ext_k: ambient temperature of the internal HX branch in
            K (default 293).
        regulation: scales the setpoint; 1 (default) is full dispatch, 0 off.
        cold_node_id: alias for heat_node_id.
        hot_node_id: alias for heat_return_node_id.
        name: compound name, resolvable by
            :meth:`TimeseriesData.add_compound_series_by_name`.
    """
    heat_node_id, heat_return_node_id = _resolve_heat_sides(
        heat_node_id, heat_return_node_id, cold_node_id, hot_node_id
    )
    _require_arguments(
        "create_g2h",
        heat_energy_mw=heat_energy_mw,
        diameter_m=diameter_m,
        efficiency=efficiency,
    )
    return network.compound(
        mm.GasToHeat(
            heat_energy_mw=heat_energy_mw,
            diameter_m=diameter_m,
            temperature_ext_k=temperature_ext_k,
            efficiency=efficiency,
            regulation=regulation,
        ),
        constraints=constraints,
        name=name,
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
    name=None,
):
    """CHP variant using a SubHG node-based heat injector (no heat_return /
    diameter). ``mass_flow_setpoint_kgs`` is the positive gas consumption in
    kg/s; power and heat outputs follow from the efficiencies. ``regulation``
    scales the setpoint (default 1). ``name`` makes the compound resolvable by
    :meth:`TimeseriesData.add_compound_series_by_name`."""
    return network.compound(
        mm.CHPHG(
            efficiency_power=efficiency_power,
            efficiency_heat=efficiency_heat,
            mass_flow_setpoint_kgs=mass_flow_setpoint_kgs,
            q_mvar_setpoint=0,
            regulation=regulation,
        ),
        constraints=constraints,
        name=name,
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
    regulation=1,
    constraints=None,
    name=None,
):
    """Single-branch P2H using :class:`PowerToHeatHG` (heat injected at the
    to-node). ``heat_energy_mw`` is the positive thermal output; the branch
    draws ``heat_energy_mw / efficiency`` at the power node (load
    convention). ``regulation`` scales the setpoint (default 1). ``name``
    makes the branch resolvable by
    :meth:`TimeseriesData.add_branch_series_by_name`."""
    return network.branch(
        mm.PowerToHeatHG(
            heat_energy_mw=heat_energy_mw,
            efficiency=efficiency,
            q_mvar_setpoint=q_mvar_setpoint,
            regulation=regulation,
        ),
        from_node_id=power_node_id,
        to_node_id=heat_node_id,
        constraints=constraints,
        name=name,
    )


def create_g2h_hg(
    network: mm.Network,
    gas_node_id,
    heat_node_id,
    heat_energy_mw,
    efficiency,
    regulation=1,
    constraints=None,
    name=None,
):
    """Single-branch G2H using :class:`GasToHeatHG` (heat injected at the
    to-node). ``heat_energy_mw`` is the positive thermal output; the branch
    draws the corresponding gas mass flow at the gas node (load convention).
    ``regulation`` scales the setpoint (default 1). ``name`` makes the branch
    resolvable by :meth:`TimeseriesData.add_branch_series_by_name`."""
    return network.branch(
        mm.GasToHeatHG(
            heat_energy_mw=heat_energy_mw,
            efficiency=efficiency,
            regulation=regulation,
        ),
        from_node_id=gas_node_id,
        to_node_id=heat_node_id,
        constraints=constraints,
        name=name,
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
    t_k=None,
    max_import_kgs=None,
    max_export_kgs=None,
    pin_temperature=True,
    free_pressure_bounds=None,
    **kwargs,
):
    """Attach an external gas grid (slack) to a gas node, auto-creating a gas
    junction if needed.

    The attached child's model class is :class:`ExtHydrGrid` (the shared
    hydraulic slack model, not a gas-specific class); pass that as
    ``model_type`` when querying results for it.

    Args:
        node_id: gas junction to attach to; a missing node is auto-created.
        mass_flow_kgs: initial guess for the free exchange Var, not a
            setpoint. Load convention: negative mass_flow_kgs is injection
            into the network (import), positive is withdrawal (export).
        pressure_pu: pressure the slack pins the junction to, per unit of
            the grid's ``pressure_ref_pa``. The default gas grid (lgas,
            300 K operating temperature) has a 10 bar reference, so the
            default 1 pu means 10 bar absolute.
        t_k: temperature of the injected stream in K. Default None resolves
            at solve time to the gas grid's own operating temperature
            (``grid.t_k``, 300 K for the default lgas grid). Before 2026-09
            the default was the 356 K heat-network value; pass ``t_k=356``
            to restore that.
        max_import_kgs: import cap; positive magnitude, bounds
            mass_flow_kgs from below at -max_import_kgs (load convention).
            Default None (unbounded). Re-read at every solve, so setting it
            on the returned child's model later takes effect on the next
            solve.
        max_export_kgs: export cap; positive magnitude, bounds
            mass_flow_kgs from above. Default None (unbounded). Re-read at
            every solve like max_import_kgs.
        pin_temperature: True (default) pins the junction temperature to
            t_k; False leaves it to the surrounding network.
        free_pressure_bounds: (lo, hi) tuple in pu; instead of pinning the
            pressure at pressure_pu, leave it a Var bounded to [lo, hi].
            Default None (pressure pinned).
    """
    return create_ext_hydr_grid(
        network,
        node_id,
        mass_flow_kgs=mass_flow_kgs,
        pressure_pu=pressure_pu,
        t_k=t_k,
        grid_key=mm.GAS_KEY,
        max_import_kgs=max_import_kgs,
        max_export_kgs=max_export_kgs,
        pin_temperature=pin_temperature,
        free_pressure_bounds=free_pressure_bounds,
        **kwargs,
    )


def create_water_ext_grid(
    network: mm.Network,
    node_id,
    mass_flow_kgs=1,
    pressure_pu=1,
    t_k=None,
    max_import_kgs=None,
    max_export_kgs=None,
    pin_temperature=True,
    free_pressure_bounds=None,
    **kwargs,
):
    """Attach an external water/heat grid (slack) to a water node,
    auto-creating a water junction if needed.

    On a heat network this slack behaves as an unlimited backup heat plant:
    it delivers any mass flow at its pinned feed temperature, so lost heat
    sources elsewhere are compensated silently instead of shedding heat load.
    Cap it with ``max_import_kgs`` / ``max_export_kgs``, or with
    ``bounds_ext_heat`` on the load-shedding problem, when that is not the
    intended physics (see the concepts/multi_energy page).

    Args:
        node_id: water junction to attach to; a missing node is auto-created.
        mass_flow_kgs: initial guess for the free exchange Var, not a
            setpoint. Load convention: negative mass_flow_kgs is injection
            into the network (import), positive is withdrawal (export).
        pressure_pu: pressure the slack pins the junction to, per unit of
            the grid's ``pressure_ref_pa`` (default water grid: 10 bar).
        t_k: feed temperature of the injected stream in K. Default None
            resolves at solve time to the water grid's reference temperature
            (``t_ref_k``, 356 K by default), matching the pre-2026-09 fixed
            356 K default.
        max_import_kgs: import cap; positive magnitude, bounds
            mass_flow_kgs from below at -max_import_kgs (load convention).
            Default None (unbounded). Re-read at every solve, so setting it
            on the returned child's model later takes effect on the next
            solve.
        max_export_kgs: export cap; positive magnitude, bounds
            mass_flow_kgs from above. Default None (unbounded). Re-read at
            every solve like max_import_kgs.
        pin_temperature: True (default) pins the junction temperature to
            t_k; False leaves it to the surrounding network.
        free_pressure_bounds: (lo, hi) tuple in pu; instead of pinning the
            pressure at pressure_pu, leave it a Var bounded to [lo, hi].
            Default None (pressure pinned).
    """
    return create_ext_hydr_grid(
        network,
        node_id,
        mass_flow_kgs=mass_flow_kgs,
        pressure_pu=pressure_pu,
        t_k=t_k,
        grid_key=mm.WATER_KEY,
        max_import_kgs=max_import_kgs,
        max_export_kgs=max_export_kgs,
        pin_temperature=pin_temperature,
        free_pressure_bounds=free_pressure_bounds,
        **kwargs,
    )


def create_gas_source(
    network: mm.Network, node_id, mass_flow_kgs=1, t_k=None, **kwargs
):
    """Attach a gas source (fixed injection) to a gas node, auto-creating a
    junction if needed. Pass a positive ``mass_flow_kgs`` magnitude; the sign
    is handled internally (load convention). ``t_k`` optionally sets the
    injected stream's temperature; see :func:`create_source`."""
    return create_source(
        network,
        node_id,
        mass_flow_kgs=mass_flow_kgs,
        grid_key=mm.GAS_KEY,
        t_k=t_k,
        **kwargs,
    )


def create_gas_sink(network: mm.Network, node_id, mass_flow_kgs=1, **kwargs):
    """Attach a gas sink (fixed consumption, positive ``mass_flow_kgs`` =
    consumption) to a gas node, auto-creating a junction if needed."""
    return create_sink(
        network, node_id, mass_flow_kgs=mass_flow_kgs, grid_key=mm.GAS_KEY, **kwargs
    )


def create_water_source(
    network: mm.Network, node_id, mass_flow_kgs=1, t_k=None, **kwargs
):
    """Attach a water source (fixed injection) to a water node, auto-creating a
    junction if needed. Pass a positive ``mass_flow_kgs`` magnitude; the sign
    is handled internally (load convention). ``t_k`` optionally sets the
    injected stream's temperature; see :func:`create_source`."""
    return create_source(
        network,
        node_id,
        mass_flow_kgs=mass_flow_kgs,
        grid_key=mm.WATER_KEY,
        t_k=t_k,
        **kwargs,
    )


def create_water_sink(network: mm.Network, node_id, mass_flow_kgs=1, **kwargs):
    """Attach a water sink (fixed consumption, positive ``mass_flow_kgs`` =
    consumption) to a water node, auto-creating a junction if needed."""
    return create_sink(
        network, node_id, mass_flow_kgs=mass_flow_kgs, grid_key=mm.WATER_KEY, **kwargs
    )


def create_grid_forming_generator(
    network: mm.Network,
    node_id,
    p_mw_max: float,
    q_mvar_max: float,
    vm_pu: float = 1.0,
    nominal_p_mw: float | None = 0.0,
    nominal_q_mvar: float | None = 0.0,
    constraints=None,
    overwrite_id=None,
    name=None,
):
    """Attach a :class:`GridFormingGenerator` (island slack bus). Requires
    ``enable_islanding(net, electricity=True)``.

    While the unit does not lead an island (an ext grid leads its component, or
    no islanding config is registered) it holds ``nominal_p_mw``/
    ``nominal_q_mvar``; pass ``None`` for either to leave that injection free.

    ``p_mw_max``/``q_mvar_max`` are positive capacities bounding the injection
    symmetrically. Results stay in load convention, so a unit injecting 2 MW
    reports ``p_mw = -2.0`` in ``result.dataframes['GridFormingGenerator']``.
    Both capacities remain writable on the model
    (``net.child_by_id(gid).model.p_mw_max = 0.4``) to rescale a unit between
    scenario solves.

    ``overwrite_id`` assigns an explicit, still-free child id; it does not
    replace an existing child and raises if the id is taken. To swap a unit
    out, call ``net.remove_child(gid)`` first, or set the capacities in place."""
    return create_el_child(
        network,
        GridFormingGenerator(
            p_mw_max=p_mw_max,
            q_mvar_max=q_mvar_max,
            vm_pu=vm_pu,
            nominal_p_mw=nominal_p_mw,
            nominal_q_mvar=nominal_q_mvar,
        ),
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
    **kwargs,
):
    """Gas-side shortcut for :func:`create_grid_forming_source`."""
    return create_grid_forming_source(
        network,
        node_id,
        pressure_pu=pressure_pu,
        t_k=t_k,
        mass_flow_max_kgs=mass_flow_max_kgs,
        grid_key=mm.GAS_KEY,
        **kwargs,
    )


def create_water_grid_forming_source(
    network: mm.Network,
    node_id,
    pressure_pu: float = 1.0,
    t_k: float = 356.0,
    mass_flow_max_kgs: float = 1e6,
    **kwargs,
):
    """Water-side shortcut for :func:`create_grid_forming_source`."""
    return create_grid_forming_source(
        network,
        node_id,
        pressure_pu=pressure_pu,
        t_k=t_k,
        mass_flow_max_kgs=mass_flow_max_kgs,
        grid_key=mm.WATER_KEY,
        **kwargs,
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

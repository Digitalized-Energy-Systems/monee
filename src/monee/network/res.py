import monee.express as mx
import monee.model as mm


def _line(net, from_id, to_id, length_m):
    """Add a 20 kV cable PowerLine (~10 MVA/line)."""
    r, x, max_i_ka = 3e-4, 3e-4, 0.30
    net.branch(
        mm.PowerLine(
            length_m=length_m,
            r_ohm_per_m=r,
            x_ohm_per_m=x,
            max_i_ka=max_i_ka,
            parallel=1,
        ),
        from_node_id=from_id,
        to_node_id=to_id,
    )


def _tie_line(net, from_id, to_id, length_m, name):
    """Add a normally-open 20 kV tie line (same cable as :func:`_line`)."""
    mx.create_line(
        net,
        from_id,
        to_id,
        length_m=length_m,
        r_ohm_per_m=3e-4,
        x_ohm_per_m=3e-4,
        parallel=1,
        on_off=0,
        max_i_ka=0.30,
        name=name,
    )


def create_urban_district_net() -> mm.Network:
    """
    Urban residential district: 20 kV power + medium-pressure gas + district heat.

    5 buses · 5 gas junctions · 4 heat junctions (3 supply + 1 return) · 3 CPs.
    Highest CP-to-node ratio of the three grids.
    Suitable for: testing CHP-centred resilience.

    The heat grid uses a supply-return two-pipe structure: the CHP injects
    heat on the return→supply side (r1→s1, ~102 kW) and a single
    ``HeatExchangerLoad`` consumer extracts on the supply→return side (s3→r1).
    """
    net = mx.create_multi_energy_network()

    b0 = mx.create_bus(net, base_kv=20, name="B0_gen")
    b1 = mx.create_bus(net, base_kv=20, name="B1_slack")
    b2 = mx.create_bus(net, base_kv=20, name="B2_load")
    b3 = mx.create_bus(net, base_kv=20, name="B3_load")
    b4 = mx.create_bus(net, base_kv=20, name="B4_load")

    mx.create_ext_power_grid(net, b1, p_mw=0, q_mvar=0, vm_pu=1.0)
    mx.create_power_generator(net, b0, p_mw=8, q_mvar=0)
    mx.create_power_load(net, b2, p_mw=3, q_mvar=0.5)
    mx.create_power_load(net, b3, p_mw=5, q_mvar=0.5)
    mx.create_power_load(net, b4, p_mw=4, q_mvar=0.5)

    _line(net, b0, b1, length_m=500)
    _line(net, b1, b2, length_m=400)
    _line(net, b2, b3, length_m=300)
    _line(net, b2, b4, length_m=350)

    g0 = mx.create_gas_junction(net, name="G0_ext")
    g1 = mx.create_gas_junction(net, name="G1")
    g2 = mx.create_gas_junction(net, name="G2")
    g3 = mx.create_gas_junction(net, name="G3_sink")
    g4 = mx.create_gas_junction(net, name="G4_sink")

    mx.create_ext_hydr_grid(net, g0)
    mx.create_sink(net, g3, mass_flow_kgs=0.04)  # direct gas consumer
    mx.create_sink(net, g4, mass_flow_kgs=0.02)  # P2G injection node + consumer

    mx.create_gas_pipe(net, g0, g1, diameter_m=0.15, length_m=400)
    mx.create_gas_pipe(net, g1, g2, diameter_m=0.12, length_m=300)
    mx.create_gas_pipe(net, g2, g3, diameter_m=0.10, length_m=250)
    mx.create_gas_pipe(net, g1, g4, diameter_m=0.10, length_m=200)

    # Supply side (hot, ~356 K): CP hot outlets + distribution + HE consumers.
    # CHP and P2H bridge return→supply, just as HEs bridge supply→return.
    s1 = mx.create_water_junction(net, name="s1")
    s2 = mx.create_water_junction(net, name="s2")
    s3 = mx.create_water_junction(net, name="s3")
    mx.create_water_pipe(net, s1, s2, diameter_m=0.10, length_m=100)
    mx.create_water_pipe(net, s2, s3, diameter_m=0.10, length_m=100)
    mx.create_ext_hydr_grid(net, s1)

    r1 = mx.create_water_junction(net, name="r1")
    mx.create_consume_hydr_grid(net, r1)

    mx.create_heat_exchanger(net, s3, r1, 0.2)

    # CHP: gas at G2 → power at B3, heat from r1→s1.
    # heat_w = 0.40 × 0.006 kg/s × 3.6 × 11.79011 kWh/kg × 1e6 ≈ 101 866 W
    # (default gas is lgas, HHV 11.79011 kWh/kg)
    mx.create_chp(
        net,
        power_node_id=b3,
        heat_node_id=r1,
        heat_return_node_id=s1,
        gas_node_id=g2,
        diameter_m=0.10,
        efficiency_power=0.40,
        efficiency_heat=0.40,
        mass_flow_setpoint_kgs=0.006,
        regulation=1,
    )
    # P2G: surplus power at B0 → hydrogen injection at G4
    mx.create_p2g(
        net,
        from_node_id=b0,
        to_node_id=g4,
        efficiency=0.70,
        mass_flow_setpoint_kgs=0.010,
        regulation=1,
    )
    # G2P: gas peaker at G3 → power backup at B2
    mx.create_g2p(
        net,
        from_node_id=g3,
        to_node_id=b2,
        efficiency=0.85,
        p_mw_setpoint=1.5,
        regulation=1,
    )
    return net


def _add_urban_district(net, idx: int, slack_bus, ext_gas_junction):
    """Add a single urban-district sub-grid attached to shared external feeders.

    Returns ``(district_slack_bus, district_gas_head)`` so the caller can wire
    inter-district trunks.
    """
    suffix = f"_d{idx}"

    b0 = mx.create_bus(net, base_kv=20, name=f"B0_gen{suffix}")
    b1 = mx.create_bus(net, base_kv=20, name=f"B1_local{suffix}")
    b2 = mx.create_bus(net, base_kv=20, name=f"B2_load{suffix}")
    b3 = mx.create_bus(net, base_kv=20, name=f"B3_load{suffix}")
    b4 = mx.create_bus(net, base_kv=20, name=f"B4_load{suffix}")

    mx.create_power_generator(net, b0, p_mw=4, q_mvar=0)
    mx.create_power_load(net, b2, p_mw=2, q_mvar=0.3)
    mx.create_power_load(net, b3, p_mw=3, q_mvar=0.3)
    mx.create_power_load(net, b4, p_mw=2, q_mvar=0.3)

    _line(net, slack_bus, b1, length_m=600)
    _line(net, b0, b1, length_m=400)
    _line(net, b1, b2, length_m=300)
    _line(net, b2, b3, length_m=250)
    _line(net, b2, b4, length_m=300)

    _tie_line(net, b3, b4, length_m=350, name=f"tie_b3_b4{suffix}")

    g1 = mx.create_gas_junction(net, name=f"G1{suffix}")
    g2 = mx.create_gas_junction(net, name=f"G2{suffix}")
    g3 = mx.create_gas_junction(net, name=f"G3_sink{suffix}")
    g4 = mx.create_gas_junction(net, name=f"G4_sink{suffix}")

    mx.create_sink(net, g3, mass_flow_kgs=0.03)
    mx.create_sink(net, g4, mass_flow_kgs=0.015)

    mx.create_gas_pipe(net, ext_gas_junction, g1, diameter_m=0.15, length_m=400)
    mx.create_gas_pipe(net, g1, g2, diameter_m=0.12, length_m=250)
    mx.create_gas_pipe(net, g2, g3, diameter_m=0.10, length_m=200)
    mx.create_gas_pipe(net, g1, g4, diameter_m=0.10, length_m=180)
    mx.create_gas_pipe(
        net,
        ext_gas_junction,
        g2,
        diameter_m=0.12,
        length_m=550,
        on_off=0,
        name=f"tie_g0_g2{suffix}",
    )

    s1 = mx.create_water_junction(net, name=f"s1{suffix}")
    s2 = mx.create_water_junction(net, name=f"s2{suffix}")
    s3 = mx.create_water_junction(net, name=f"s3{suffix}")
    mx.create_water_pipe(net, s1, s2, diameter_m=0.10, length_m=100)
    mx.create_water_pipe(net, s2, s3, diameter_m=0.10, length_m=100)
    mx.create_water_pipe(
        net,
        s1,
        s3,
        diameter_m=0.10,
        length_m=200,
        on_off=0,
        name=f"tie_s1_s3{suffix}",
    )
    mx.create_ext_hydr_grid(net, s1)

    r1 = mx.create_water_junction(net, name=f"r1{suffix}")
    mx.create_consume_hydr_grid(net, r1)
    mx.create_heat_exchanger(net, s3, r1, 0.15)

    r2 = mx.create_water_junction(net, name=f"r2{suffix}")
    mx.create_consume_hydr_grid(net, r2)
    mx.create_heat_exchanger(net, s2, r2, 0.08)

    mx.create_chp(
        net,
        power_node_id=b3,
        heat_node_id=r1,
        heat_return_node_id=s1,
        gas_node_id=g2,
        diameter_m=0.10,
        efficiency_power=0.40,
        efficiency_heat=0.40,
        mass_flow_setpoint_kgs=0.005,
        regulation=1,
    )
    mx.create_p2g(
        net,
        from_node_id=b0,
        to_node_id=g4,
        efficiency=0.70,
        mass_flow_setpoint_kgs=0.008,
        regulation=1,
    )
    mx.create_g2p(
        net,
        from_node_id=g3,
        to_node_id=b2,
        efficiency=0.85,
        p_mw_setpoint=1.0,
        regulation=1,
    )
    return b1, g1


def create_large_urban_mes_net(n_districts: int = 6) -> mm.Network:
    """
    Scaled multi-district urban MES for large-grid restoration evaluation.

    Replicates the urban-district pattern ``n_districts`` times under a
    shared HV slack and a shared external gas feeder, linking adjacent
    districts with normally-open MV power ties and live gas trunks.
    Heat is intentionally local-only (one return-side consumer pair per
    district), so the heat sector decomposes into one connected
    component per district while electricity and gas remain
    single-component.

    Sizing (per district): 5 power buses, 4 gas junctions, 4 water
    junctions, 2 heat consumers, 3 CPs (CHP/P2G/G2P), 4 internal ties.

    Defaults (``n_districts=6``) yield ~96 nodes / ~30 children.  Set
    ``n_districts=20`` for a ~320-node / ~100-child stress test that
    actually exercises gossip scaling and meaningful holon formation
    in the heat sector (one group per district).

    Args:
        n_districts: Number of urban districts to replicate.  Must be ≥1.
    """
    if n_districts < 1:
        raise ValueError("n_districts must be >= 1")

    net = mx.create_multi_energy_network()

    slack = mx.create_bus(net, base_kv=20, name="HV_slack")
    mx.create_ext_power_grid(net, slack, p_mw=0, q_mvar=0, vm_pu=1.0)

    ext_gas = mx.create_gas_junction(net, name="G_ext")
    mx.create_ext_hydr_grid(net, ext_gas)

    district_heads_power: list = []
    district_heads_gas: list = []
    for i in range(n_districts):
        bp, gh = _add_urban_district(net, i, slack, ext_gas)
        district_heads_power.append(bp)
        district_heads_gas.append(gh)

    for i in range(n_districts - 1):
        a, b = district_heads_power[i], district_heads_power[i + 1]
        _tie_line(net, a, b, length_m=2000, name=f"inter_district_power_{i}")
        ga, gb = district_heads_gas[i], district_heads_gas[i + 1]
        mx.create_gas_pipe(
            net,
            ga,
            gb,
            diameter_m=0.20,
            length_m=1500,
            name=f"inter_district_gas_{i}",
        )

    return net


if __name__ == "__main__":
    import monee

    net = create_urban_district_net()
    net.apply_formulation(monee.EL_MISOCP_FORMULATION)
    print(monee.run_energy_flow(net, solver="gurobi"))

import monee.express as mx
import monee.model as mm
from monee.model.grid import GasGrid
from monee.simulation.timeseries import TimeseriesData


def _line(net, from_id, to_id, length_m, kv_class="120"):
    """Add a PowerLine with parameters appropriate for the voltage class."""
    if kv_class == "20":
        r, x, max_i_ka = 3e-4, 3e-4, 0.30  # 20 kV cable   → ~10 MVA/line
    elif kv_class == "110":
        r, x, max_i_ka = 7e-5, 7e-5, 0.40  # 110 kV OHL    → ~76 MVA/line
    else:  # 120 kV
        r, x, max_i_ka = 3e-4, 3e-4, 0.30  # 120 kV cable  → ~62 MVA/line
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


def create_urban_district_net() -> mm.Network:
    """
    Urban residential district: 20 kV power + medium-pressure gas + district heat.

    5 buses · 5 gas junctions · 6 heat junctions · 4 CPs.
    Highest CP-to-node ratio of the three grids.
    Suitable for: testing CHP-centred resilience, P2H demand-side flexibility.

    Heat consumers are modelled as ``HeatExchangerLoad`` branches (fixed heat
    demand in kW), paired with a return-side ``Sink`` for mass balance.  CHP
    supplies 176 kW and P2H 20 kW; the corresponding heat exchangers enforce
    those demands explicitly at a nominal 25 K supply-to-return temperature drop.
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

    _line(net, b0, b1, length_m=500, kv_class="20")
    _line(net, b1, b2, length_m=400, kv_class="20")
    _line(net, b2, b3, length_m=300, kv_class="20")
    _line(net, b2, b4, length_m=350, kv_class="20")

    g0 = mx.create_gas_junction(net, name="G0_ext")
    g1 = mx.create_gas_junction(net, name="G1")
    g2 = mx.create_gas_junction(net, name="G2")
    g3 = mx.create_gas_junction(net, name="G3_sink")
    g4 = mx.create_gas_junction(net, name="G4_sink")

    mx.create_ext_hydr_grid(net, g0)
    mx.create_sink(net, g3, mass_flow=0.04)  # direct gas consumer
    mx.create_sink(net, g4, mass_flow=0.02)  # P2G injection node + consumer

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

    # CHP: gas at G2 → power at B3, heat injected between H1 and H2.
    # heat_w = 0.40 × 0.008 kg/s × 3.6 × 15.3 kWh/kg × 1e6 = 176 256 W
    mx.create_chp(
        net,
        power_node_id=b3,
        heat_node_id=r1,
        heat_return_node_id=s1,
        gas_node_id=g2,
        diameter_m=0.10,
        efficiency_power=0.40,
        efficiency_heat=0.40,
        mass_flow_setpoint=0.006,
        regulation=1,
    )
    # P2H: power at B4 → heat injected between H4 and H5 (20 kW).
    # el_mw = 20 000 W / (0.95 × 1e6) ≈ 0.021 MW consumed from B4.
    # mx.create_p2h(
    #     net,
    #     power_node_id=b4,
    #     heat_node_id=r1,
    #     heat_return_node_id=s2,
    #     heat_energy_w=10_000,
    #     diameter_m=0.10,
    #     efficiency=0.95,
    # )
    # P2G: surplus power at B0 → hydrogen injection at G4
    mx.create_p2g(
        net,
        from_node_id=b0,
        to_node_id=g4,
        efficiency=0.70,
        mass_flow_setpoint=0.010,
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


def create_urban_district_net_with_ties() -> mm.Network:
    """
    Urban district variant with normally-open tie switches and a second
    heat consumer, designed for meaningful restoration evaluation.

    Same primary topology as :func:`create_urban_district_net`, plus:

    * **Power tie** ``b3↔b4`` — alternative path between the two large loads
      so a failure on ``b2-b3`` or ``b2-b4`` is recoverable by closing it.
    * **Gas tie** ``g0↔g2`` — bypass around ``g1`` for failures on
      ``g0-g1`` or ``g1-g2``.
    * **Gas tie** ``g3↔g4`` — meshes the two sink junctions.
    * **Heat supply tie** ``s1↔s3`` — bypass around ``s2`` on the supply
      chain.
    * **Second heat consumer** at ``r2`` (fed by an HE off ``s2``) so the
      heat sector has more than one load and a single HE failure does not
      collapse all heat demand.

    All ties are created with ``on_off=0`` (open).  The ``GridReconfigurator``
    role detects them via ``obs.on_off == 0`` and closes them when a path
    search through them succeeds.
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

    _line(net, b0, b1, length_m=500, kv_class="20")
    _line(net, b1, b2, length_m=400, kv_class="20")
    _line(net, b2, b3, length_m=300, kv_class="20")
    _line(net, b2, b4, length_m=350, kv_class="20")

    mx.create_line(
        net,
        b3,
        b4,
        length_m=400,
        r_ohm_per_m=3e-4,
        x_ohm_per_m=3e-4,
        parallel=1,
        on_off=0,
        name="tie_b3_b4",
    )

    g0 = mx.create_gas_junction(net, name="G0_ext")
    g1 = mx.create_gas_junction(net, name="G1")
    g2 = mx.create_gas_junction(net, name="G2")
    g3 = mx.create_gas_junction(net, name="G3_sink")
    g4 = mx.create_gas_junction(net, name="G4_sink")

    mx.create_ext_hydr_grid(net, g0)
    mx.create_sink(net, g3, mass_flow=0.04)
    mx.create_sink(net, g4, mass_flow=0.02)

    mx.create_gas_pipe(net, g0, g1, diameter_m=0.15, length_m=400)
    mx.create_gas_pipe(net, g1, g2, diameter_m=0.12, length_m=300)
    mx.create_gas_pipe(net, g2, g3, diameter_m=0.10, length_m=250)
    mx.create_gas_pipe(net, g1, g4, diameter_m=0.10, length_m=200)

    mx.create_gas_pipe(
        net,
        g0,
        g2,
        diameter_m=0.12,
        length_m=600,
        on_off=0,
        name="tie_g0_g2",
    )
    mx.create_gas_pipe(
        net,
        g3,
        g4,
        diameter_m=0.10,
        length_m=350,
        on_off=0,
        name="tie_g3_g4",
    )

    s1 = mx.create_water_junction(net, name="s1")
    s2 = mx.create_water_junction(net, name="s2")
    s3 = mx.create_water_junction(net, name="s3")
    mx.create_water_pipe(net, s1, s2, diameter_m=0.10, length_m=100)
    mx.create_water_pipe(net, s2, s3, diameter_m=0.10, length_m=100)
    mx.create_water_pipe(
        net,
        s1,
        s3,
        diameter_m=0.10,
        length_m=200,
        on_off=0,
        name="tie_s1_s3",
    )
    mx.create_ext_hydr_grid(net, s1)

    r1 = mx.create_water_junction(net, name="r1")
    mx.create_consume_hydr_grid(net, r1)
    mx.create_heat_exchanger(net, s3, r1, 0.2)

    r2 = mx.create_water_junction(net, name="r2")
    mx.create_consume_hydr_grid(net, r2)
    mx.create_heat_exchanger(net, s2, r2, 0.1)

    mx.create_chp(
        net,
        power_node_id=b3,
        heat_node_id=r1,
        heat_return_node_id=s1,
        gas_node_id=g2,
        diameter_m=0.10,
        efficiency_power=0.40,
        efficiency_heat=0.40,
        mass_flow_setpoint=0.006,
        regulation=1,
    )
    mx.create_p2g(
        net,
        from_node_id=b0,
        to_node_id=g4,
        efficiency=0.70,
        mass_flow_setpoint=0.010,
        regulation=1,
    )
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

    _line(net, slack_bus, b1, length_m=600, kv_class="20")
    _line(net, b0, b1, length_m=400, kv_class="20")
    _line(net, b1, b2, length_m=300, kv_class="20")
    _line(net, b2, b3, length_m=250, kv_class="20")
    _line(net, b2, b4, length_m=300, kv_class="20")

    mx.create_line(
        net,
        b3,
        b4,
        length_m=350,
        r_ohm_per_m=3e-4,
        x_ohm_per_m=3e-4,
        parallel=1,
        on_off=0,
        name=f"tie_b3_b4{suffix}",
    )

    g1 = mx.create_gas_junction(net, name=f"G1{suffix}")
    g2 = mx.create_gas_junction(net, name=f"G2{suffix}")
    g3 = mx.create_gas_junction(net, name=f"G3_sink{suffix}")
    g4 = mx.create_gas_junction(net, name=f"G4_sink{suffix}")

    mx.create_sink(net, g3, mass_flow=0.03)
    mx.create_sink(net, g4, mass_flow=0.015)

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
        mass_flow_setpoint=0.005,
        regulation=1,
    )
    mx.create_p2g(
        net,
        from_node_id=b0,
        to_node_id=g4,
        efficiency=0.70,
        mass_flow_setpoint=0.008,
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
        mx.create_line(
            net,
            a,
            b,
            length_m=2000,
            r_ohm_per_m=3e-4,
            x_ohm_per_m=3e-4,
            parallel=1,
            on_off=0,
            name=f"inter_district_power_{i}",
        )
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


def create_industrial_hub_net() -> mm.Network:
    """
    Industrial energy hub: 110 kV meshed power + high-pressure gas, no district heat.

    8 buses (ring + 2 cross-ties) · 7 gas junctions (ring).
    5 CPs: 3× G2P + 2× P2G — strong gas-backup power capacity.
    Suitable for: sparse-heat resilience analysis, gas-turbine backup power testing.
    """
    net = mx.create_multi_energy_network()

    b0 = mx.create_bus(net, base_kv=110, name="B0_slack")
    b1 = mx.create_bus(net, base_kv=110, name="B1_gen")
    b2 = mx.create_bus(net, base_kv=110, name="B2_load")
    b3 = mx.create_bus(net, base_kv=110, name="B3_load")
    b4 = mx.create_bus(net, base_kv=110, name="B4_load")
    b5 = mx.create_bus(net, base_kv=110, name="B5_load")
    b6 = mx.create_bus(net, base_kv=110, name="B6_gen")
    b7 = mx.create_bus(net, base_kv=110, name="B7_load")

    mx.create_ext_power_grid(net, b0, p_mw=0, q_mvar=0, vm_pu=1.0, max_import_mw=0.0)
    mx.create_power_generator(net, b1, p_mw=50, q_mvar=0)
    mx.create_power_generator(net, b6, p_mw=30, q_mvar=0)
    mx.create_power_load(net, b2, p_mw=15, q_mvar=2)
    mx.create_power_load(net, b3, p_mw=20, q_mvar=2)
    mx.create_power_load(net, b4, p_mw=25, q_mvar=3)
    mx.create_power_load(net, b5, p_mw=10, q_mvar=1)
    mx.create_power_load(net, b7, p_mw=18, q_mvar=2)

    # Ring
    for fn, tn, le in [
        (b0, b1, 500),
        (b1, b2, 400),
        (b2, b3, 600),
        (b3, b4, 500),
        (b4, b5, 300),
        (b5, b6, 400),
        (b6, b7, 350),
        (b7, b0, 500),
    ]:
        _line(net, fn, tn, le, kv_class="110")
    # Cross-ties (meshing for redundancy)
    _line(net, b1, b5, length_m=800, kv_class="110")
    _line(net, b2, b6, length_m=700, kv_class="110")

    g0 = mx.create_gas_junction(net, name="G0_ext")
    g1 = mx.create_gas_junction(net, name="G1")
    g2 = mx.create_gas_junction(net, name="G2_industrial")
    g3 = mx.create_gas_junction(net, name="G3")
    g4 = mx.create_gas_junction(net, name="G4_p2g")
    g5 = mx.create_gas_junction(net, name="G5_sink")
    g6 = mx.create_gas_junction(net, name="G6_sink")

    mx.create_ext_hydr_grid(net, g0, max_import_kgs=1.1)
    mx.create_sink(net, g2, mass_flow=0.30)  # large industrial process load
    mx.create_sink(net, g5, mass_flow=0.20)
    mx.create_sink(net, g6, mass_flow=0.15)

    mx.create_gas_pipe(net, g0, g1, diameter_m=0.50, length_m=800)
    mx.create_gas_pipe(net, g1, g2, diameter_m=0.40, length_m=600)
    mx.create_gas_pipe(net, g2, g3, diameter_m=0.40, length_m=500)
    mx.create_gas_pipe(net, g3, g4, diameter_m=0.40, length_m=400)
    mx.create_gas_pipe(net, g1, g5, diameter_m=0.30, length_m=400)
    mx.create_gas_pipe(net, g3, g6, diameter_m=0.30, length_m=350)
    mx.create_gas_pipe(net, g0, g3, diameter_m=0.40, length_m=1200)  # ring closure

    # G2P: gas turbines at mid-load buses — backup / peak-shaving generation
    mx.create_g2p(
        net,
        from_node_id=g1,
        to_node_id=b2,
        efficiency=0.88,
        p_mw_setpoint=8,
        regulation=1,
    )
    mx.create_g2p(
        net,
        from_node_id=g2,
        to_node_id=b4,
        efficiency=0.88,
        p_mw_setpoint=12,
        regulation=1,
    )
    mx.create_g2p(
        net,
        from_node_id=g3,
        to_node_id=b6,
        efficiency=0.88,
        p_mw_setpoint=6,
        regulation=1,
    )
    # P2G: electrolysers at generator buses — store surplus as hydrogen
    mx.create_p2g(
        net,
        from_node_id=b1,
        to_node_id=g0,
        efficiency=0.70,
        mass_flow_setpoint=0.05,
        regulation=1,
    )
    mx.create_p2g(
        net,
        from_node_id=b6,
        to_node_id=g4,
        efficiency=0.70,
        mass_flow_setpoint=0.04,
        regulation=1,
    )

    return net


def create_regional_mes_net() -> mm.Network:
    """
    Regional integrated MES: 120 kV ring power, gas tree, district heating.

    8 buses · 8 gas junctions · 6 heat junctions (3 supply + 3 return) · 5 CPs.
    All coupling point types (CHP, G2P, P2G, P2H) — broadest carrier diversity.
    Single cross-tie in power ring for N-1 security.
    Suitable for: comprehensive CP criticality analysis, all-carrier failure scenarios.

    The heat grid uses a supply-return two-pipe structure: CHP and P2H inject
    heat on the return→supply side, HE consumers extract on the supply→return
    side.  CHP supplies ~176 kW and P2H 20 kW.
    """
    net = mx.create_multi_energy_network()

    b0 = mx.create_bus(net, base_kv=120, name="B0_slack")
    b1 = mx.create_bus(net, base_kv=120, name="B1_gen")
    b2 = mx.create_bus(net, base_kv=120, name="B2_load")
    b3 = mx.create_bus(net, base_kv=120, name="B3_load")
    b4 = mx.create_bus(net, base_kv=120, name="B4_load")
    b5 = mx.create_bus(net, base_kv=120, name="B5_load")
    b6 = mx.create_bus(net, base_kv=120, name="B6_load")
    b7 = mx.create_bus(net, base_kv=120, name="B7_load")

    mx.create_ext_power_grid(net, b0, p_mw=0, q_mvar=0, vm_pu=1.0, max_import_mw=46)
    mx.create_power_generator(net, b1, p_mw=30, q_mvar=0)
    mx.create_power_load(net, b2, p_mw=12, q_mvar=2)
    mx.create_power_load(net, b3, p_mw=15, q_mvar=2)
    mx.create_power_load(net, b4, p_mw=12, q_mvar=2)
    mx.create_power_load(net, b5, p_mw=10, q_mvar=1)
    mx.create_power_load(net, b6, p_mw=14, q_mvar=2)
    mx.create_power_load(net, b7, p_mw=18, q_mvar=3)

    # Ring
    for fn, tn, le in [
        (b0, b1, 300),
        (b1, b2, 400),
        (b2, b3, 500),
        (b3, b4, 350),
        (b4, b5, 300),
        (b5, b6, 450),
        (b6, b7, 400),
        (b7, b0, 400),
    ]:
        _line(net, fn, tn, le, kv_class="120")
    # Cross-tie (alternative path for N-1 security)
    _line(net, b1, b5, length_m=900, kv_class="120")

    g0 = mx.create_gas_junction(net, name="G0_ext")
    g1 = mx.create_gas_junction(net, name="G1")
    g2 = mx.create_gas_junction(net, name="G2")
    g3 = mx.create_gas_junction(net, name="G3")
    g4 = mx.create_gas_junction(net, name="G4_sink")
    g5 = mx.create_gas_junction(net, name="G5")
    g6 = mx.create_gas_junction(net, name="G6_sink")
    g7 = mx.create_gas_junction(net, name="G7_sink")

    mx.create_ext_hydr_grid(net, g0, max_import_kgs=0.42)
    mx.create_sink(net, g4, mass_flow=0.08)
    mx.create_sink(net, g6, mass_flow=0.06)
    mx.create_sink(net, g7, mass_flow=0.10)

    mx.create_gas_pipe(net, g0, g1, diameter_m=0.40, length_m=600)
    mx.create_gas_pipe(net, g1, g2, diameter_m=0.35, length_m=500)
    mx.create_gas_pipe(net, g2, g3, diameter_m=0.35, length_m=400)
    mx.create_gas_pipe(net, g3, g4, diameter_m=0.25, length_m=300)
    mx.create_gas_pipe(net, g1, g5, diameter_m=0.20, length_m=300)
    mx.create_gas_pipe(net, g2, g6, diameter_m=0.20, length_m=250)
    mx.create_gas_pipe(net, g3, g7, diameter_m=0.20, length_m=350)

    # Supply side (hot, ~356 K)
    s1 = mx.create_water_junction(net, name="s1")
    s2 = mx.create_water_junction(net, name="s2")
    s3 = mx.create_water_junction(net, name="s3")
    mx.create_water_pipe(net, s1, s2, diameter_m=0.12, length_m=200)
    mx.create_water_pipe(net, s2, s3, diameter_m=0.10, length_m=150)
    mx.create_ext_hydr_grid(net, s1, max_import_kgs=0.1)

    # Return side (cold, ~330 K)
    r1 = mx.create_water_junction(net, name="r1")
    r2 = mx.create_water_junction(net, name="r2")
    r3 = mx.create_water_junction(net, name="r3")
    mx.create_water_pipe(net, r1, r2, diameter_m=0.12, length_m=200)
    mx.create_water_pipe(net, r2, r3, diameter_m=0.10, length_m=150)
    mx.create_consume_hydr_grid(net, r1)

    # HEs bridge supply→return
    mx.create_heat_exchanger(net, s2, r2, 0.176)  # 176 kW CHP consumer
    mx.create_heat_exchanger(net, s3, r3, 0.020)  # 20 kW P2H consumer

    # CHP: gas at G1 → power at B2, heat from r1→s1.
    mx.create_chp(
        net,
        power_node_id=b2,
        heat_node_id=r1,
        heat_return_node_id=s1,
        gas_node_id=g1,
        diameter_m=0.12,
        efficiency_power=0.40,
        efficiency_heat=0.40,
        mass_flow_setpoint=0.008,
        regulation=1,
    )
    # G2P-1: gas turbine peaker at mid-ring load bus
    mx.create_g2p(
        net,
        from_node_id=g2,
        to_node_id=b3,
        efficiency=0.88,
        p_mw_setpoint=5,
        regulation=1,
    )
    # G2P-2: gas turbine at ring opposite side
    mx.create_g2p(
        net,
        from_node_id=g4,
        to_node_id=b6,
        efficiency=0.88,
        p_mw_setpoint=4,
        regulation=1,
    )
    # P2G: electrolyser at load bus (stores surplus power as gas)
    mx.create_p2g(
        net,
        from_node_id=b4,
        to_node_id=g5,
        efficiency=0.70,
        mass_flow_setpoint=0.03,
        regulation=1,
    )
    # P2H: electric booster at B5, heat from r2→s2 (20 kW).
    mx.create_p2h(
        net,
        power_node_id=b5,
        heat_node_id=r2,
        heat_return_node_id=s2,
        heat_energy_w=20_000,
        diameter_m=0.12,
        efficiency=0.95,
    )
    return net


def create_balanced_urban_mes_net() -> mm.Network:
    """
    Balanced urban MES: 5 buses · 5 gas junctions · 8 heat junctions
    (4 supply + 4 return) · 4 CPs.

    Carrier energy flows are scaled so that single-carrier failures cause
    comparable amounts of load shedding.

    Carrier direct-load targets (shed-able demand):
        Power  : B2 0.5 MW + B3 0.7 MW + B4 0.5 MW            = 1.7 MW
        Gas    : G3 0.015 kg/s + G4 0.010 kg/s                  ≈ 1.4 MW
        Heat   : HE1 0.550 MW + HE2 0.300 MW                    = 0.85 MW

    The heat grid uses a supply-return two-pipe structure: CHP and P2H inject
    heat on the return→supply side, HE consumers extract on the supply→return
    side.  CHP supplies ~572 kW (0.026 kg/s gas) and P2H supplies 300 kW.

    Coupling points:
        CHP  : G2 → B3 (0.57 MW power) + r1→s1 (0.57 MW heat)
        P2H  : B4 → r2→s2, 300 kW heat (0.316 MW electric consumption)
        P2G  : B0 → G4, 0.003 kg/s hydrogen
        G2P  : G3 → B2, 0.5 MW electric backup
    """
    net = mx.create_multi_energy_network()

    b0 = mx.create_bus(net, base_kv=20, name="B0_gen")
    b1 = mx.create_bus(net, base_kv=20, name="B1_slack")
    b2 = mx.create_bus(net, base_kv=20, name="B2_load")
    b3 = mx.create_bus(net, base_kv=20, name="B3_load")
    b4 = mx.create_bus(net, base_kv=20, name="B4_load")

    mx.create_ext_power_grid(net, b1, p_mw=0, q_mvar=0, vm_pu=1.0, max_import_mw=0.5)
    mx.create_power_generator(net, b0, p_mw=1.5, q_mvar=0)
    mx.create_power_load(net, b2, p_mw=0.5, q_mvar=0.1)
    mx.create_power_load(net, b3, p_mw=0.7, q_mvar=0.1)
    mx.create_power_load(net, b4, p_mw=0.5, q_mvar=0.1)

    _line(net, b0, b1, length_m=500, kv_class="20")
    _line(net, b1, b2, length_m=400, kv_class="20")
    _line(net, b2, b3, length_m=300, kv_class="20")
    _line(net, b2, b4, length_m=350, kv_class="20")

    g0 = mx.create_gas_junction(net, name="G0_ext")
    g1 = mx.create_gas_junction(net, name="G1")
    g2 = mx.create_gas_junction(net, name="G2")
    g3 = mx.create_gas_junction(net, name="G3_sink")
    g4 = mx.create_gas_junction(net, name="G4_sink")

    mx.create_ext_hydr_grid(net, g0, max_import_kgs=0.06)
    mx.create_sink(net, g3, mass_flow=0.015)  # ~0.83 MW direct gas consumer
    mx.create_sink(net, g4, mass_flow=0.010)  # ~0.55 MW P2G injection node + consumer

    mx.create_gas_pipe(net, g0, g1, diameter_m=0.15, length_m=400)
    mx.create_gas_pipe(net, g1, g2, diameter_m=0.12, length_m=300)
    mx.create_gas_pipe(net, g2, g3, diameter_m=0.10, length_m=250)
    mx.create_gas_pipe(net, g1, g4, diameter_m=0.10, length_m=200)

    # Supply side (hot, ~356 K)
    s1 = mx.create_water_junction(net, name="s1")
    s2 = mx.create_water_junction(net, name="s2")
    s3 = mx.create_water_junction(net, name="s3")
    s4 = mx.create_water_junction(net, name="s4")
    mx.create_water_pipe(net, s1, s2, diameter_m=0.20, length_m=100)
    mx.create_water_pipe(net, s2, s3, diameter_m=0.15, length_m=150)
    mx.create_water_pipe(net, s3, s4, diameter_m=0.15, length_m=100)
    mx.create_ext_hydr_grid(net, s1, max_import_kgs=0.1)

    # Return side (cold, ~330 K)
    r1 = mx.create_water_junction(net, name="r1")
    r2 = mx.create_water_junction(net, name="r2")
    r3 = mx.create_water_junction(net, name="r3")
    r4 = mx.create_water_junction(net, name="r4")
    mx.create_water_pipe(net, r1, r2, diameter_m=0.20, length_m=100)
    mx.create_water_pipe(net, r2, r3, diameter_m=0.15, length_m=150)
    mx.create_water_pipe(net, r3, r4, diameter_m=0.15, length_m=100)
    mx.create_consume_hydr_grid(net, r1)

    # HEs bridge supply→return
    mx.create_heat_exchanger(net, s3, r3, 0.550)  # 550 kW CHP consumer
    mx.create_heat_exchanger(net, s4, r4, 0.300)  # 300 kW P2H consumer

    # CHP: gas at G2 → power at B3, heat from r1→s1.
    # heat_w = 0.40 × 0.026 kg/s × 3.6 × 15.3 kWh/kg × 1e6 ≈ 572 kW
    mx.create_chp(
        net,
        power_node_id=b3,
        heat_node_id=r1,
        heat_return_node_id=s1,
        gas_node_id=g2,
        diameter_m=0.15,
        efficiency_power=0.40,
        efficiency_heat=0.40,
        mass_flow_setpoint=0.026,
        regulation=1,
    )
    # P2H: power at B4 → 300 kW heat from r2→s2.
    # el_mw = 300 000 / (0.95 × 1e6) ≈ 0.316 MW consumed from B4.
    mx.create_p2h(
        net,
        power_node_id=b4,
        heat_node_id=r2,
        heat_return_node_id=s2,
        heat_energy_w=300_000,
        diameter_m=0.15,
        efficiency=0.95,
    )
    # P2G: surplus power at B0 → hydrogen injection at G4
    mx.create_p2g(
        net,
        from_node_id=b0,
        to_node_id=g4,
        efficiency=0.70,
        mass_flow_setpoint=0.003,
        regulation=1,
    )
    # G2P: gas peaker at G3 → power backup at B2
    mx.create_g2p(
        net,
        from_node_id=g3,
        to_node_id=b2,
        efficiency=0.85,
        p_mw_setpoint=0.5,
        regulation=1,
    )
    return net


def create_balanced_urban_mes_timeseries(
    net: mm.Network, n_steps: int = 24, seed: int = 0
) -> TimeseriesData:
    """
    Generate realistic hourly timeseries for the balanced urban MES.

    Produces ``steps`` timesteps of synthetic winter-weekday demand profiles
    for every direct consumer in ``net``:

    * **Power loads** (``PowerLoad`` children) — residential/commercial
      pattern: low overnight, morning ramp, broad daytime plateau, evening
      peak, gradual decline.  Both ``p_mw`` and ``q_mvar`` are scaled
      proportionally to preserve constant power factor.

    * **Gas sinks** (``Sink`` children attached to a ``GasGrid`` node) —
      heating + cooking pattern: twin spikes at breakfast and dinner,
      lower midday, minimum at night.

    * **Heat exchanger loads** (``HeatExchangerLoad`` branches) — space
      heating pattern, anti-correlated with outdoor temperature: highest
      in the cold night hours, lowest in the warm afternoon.

    All profiles are expressed as fractions of each component's rated
    setpoint that is already stored in ``net``, so the function works
    correctly even if the network values are tuned after creation.

    When ``steps != 24`` the 24-point reference profiles are resampled to
    the requested length via piecewise-linear interpolation.

    Args:
        net:   A network returned by :func:`create_balanced_urban_mes_net`
               (or any MES with the same component types).
        steps: Number of simulation timesteps.  Defaults to 24 (one per hour).

    Returns:
        :class:`~monee.simulation.timeseries.TimeseriesData` ready to pass
        to :func:`~monee.simulation.timeseries.run`.
    """
    # Hour index:   0     1     2     3     4     5     6     7
    #               8     9    10    11    12    13    14    15
    #              16    17    18    19    20    21    22    23
    _POWER = [
        0.35,
        0.30,
        0.28,
        0.28,
        0.30,
        0.45,
        0.65,
        0.78,
        0.82,
        0.80,
        0.80,
        0.78,
        0.75,
        0.73,
        0.72,
        0.75,
        0.80,
        0.88,
        0.95,
        1.00,
        0.95,
        0.87,
        0.72,
        0.50,
    ]
    _GAS = [
        0.55,
        0.50,
        0.45,
        0.42,
        0.45,
        0.65,
        0.90,
        0.95,
        0.85,
        0.75,
        0.65,
        0.60,
        0.60,
        0.58,
        0.58,
        0.62,
        0.70,
        0.85,
        0.95,
        1.00,
        0.95,
        0.85,
        0.75,
        0.65,
    ]
    _HEAT = [
        0.92,
        0.98,
        1.00,
        1.00,
        0.98,
        0.95,
        0.88,
        0.78,
        0.68,
        0.60,
        0.57,
        0.55,
        0.57,
        0.60,
        0.62,
        0.65,
        0.70,
        0.78,
        0.85,
        0.88,
        0.90,
        0.90,
        0.88,
        0.92,
    ]

    def _resample(pattern, n):
        if n == len(pattern):
            return list(pattern)
        m = len(pattern)
        out = []
        for i in range(n):
            x = i / (n - 1) * (m - 1) if n > 1 else 0.0
            lo = int(x)
            hi = min(lo + 1, m - 1)
            t = x - lo
            out.append(pattern[lo] * (1.0 - t) + pattern[hi] * t)
        return out

    p_series = _resample(_POWER, n_steps)
    g_series = _resample(_GAS, n_steps)
    h_series = _resample(_HEAT, n_steps)

    td = TimeseriesData()

    for child in net.childs_by_type(mm.PowerLoad):
        p_rated = child.model.p_mw
        q_rated = child.model.q_mvar
        td.add_child_series(child.id, "p_mw", [p_rated * f for f in p_series])
        td.add_child_series(child.id, "q_mvar", [q_rated * f for f in p_series])

    # ── Gas sinks (direct consumers; excludes heat-return Sinks on WaterGrid) ─
    for child in net.childs_by_type(mm.Sink):
        if isinstance(child.grid, GasGrid):
            mf_rated = child.model.mass_flow
            td.add_child_series(child.id, "mass_flow", [mf_rated * f for f in g_series])

    # q_w_set is stored in Watts with sign convention: negative = consumer.
    for branch in net.branches_by_type(mm.HeatExchangerLoad):
        q_w_rated = branch.model.q_w_set  # e.g. -800 000 W for 800 kW consumer
        td.add_branch_series(branch.id, "q_w_set", [q_w_rated * f for f in h_series])

    return td


def create_resilient_urban_mes_net() -> mm.Network:
    """
    Resilient urban MES: 6 buses · 7 gas junctions · 12 heat junctions
    (6 supply + 6 return) · 6 CPs.

    Extends the balanced urban MES with a second gas source, a second CHP,
    a solar generator, and a third HE consumer.  The richer topology makes
    carrier dependence and redundancy directly observable.

    **Carrier dependence via coupling points**
        Each CHP converts gas → power AND heat simultaneously.  Losing
        G0_ext1 curtails CHP1 (B3 + heat s1).  P2H couples electricity → heat.

    **Resilience from redundancy**
        * Two gas sources (G0_ext1, G5_ext2) via G1–G5 inter-connection.
        * Two CHPs in independent sub-trees.
        * Solar at B5 is independent of gas and heat.
        * Three HE consumers on separate supply-return pairs.
        * G2P at G3→B2 provides gas-backed power.

    The heat grid uses a supply-return two-pipe structure with 6+6 junctions.
    CHP1, CHP2 and P2H inject heat on the return→supply side; three HE
    consumers extract on the supply→return side.
    """
    net = mx.create_multi_energy_network()

    b0 = mx.create_bus(net, base_kv=20, name="B0_wind")
    b1 = mx.create_bus(net, base_kv=20, name="B1_slack")
    b2 = mx.create_bus(net, base_kv=20, name="B2_load")
    b3 = mx.create_bus(net, base_kv=20, name="B3_load_chp1")
    b4 = mx.create_bus(net, base_kv=20, name="B4_load_p2h")
    b5 = mx.create_bus(net, base_kv=20, name="B5_load_chp2_solar")

    mx.create_ext_power_grid(net, b1, p_mw=0, q_mvar=0, vm_pu=1.0, max_import_mw=1.0)
    mx.create_power_generator(net, b0, p_mw=1.5, q_mvar=0)  # wind: independent of gas
    mx.create_power_generator(net, b5, p_mw=0.8, q_mvar=0)  # solar: independent of gas
    mx.create_power_load(net, b2, p_mw=0.5, q_mvar=0.1)
    mx.create_power_load(net, b3, p_mw=0.3, q_mvar=0.05)
    mx.create_power_load(net, b4, p_mw=0.5, q_mvar=0.1)
    mx.create_power_load(net, b5, p_mw=0.4, q_mvar=0.05)

    _line(net, b0, b1, length_m=500, kv_class="20")
    _line(net, b1, b2, length_m=400, kv_class="20")
    _line(net, b2, b3, length_m=300, kv_class="20")
    _line(net, b2, b4, length_m=350, kv_class="20")
    _line(net, b1, b5, length_m=450, kv_class="20")  # second branch from slack

    # G0_ext1: primary source; G5_ext2: secondary source.
    # G1–G5 inter-connection pipe allows either source to serve both CHPs.
    g0 = mx.create_gas_junction(net, name="G0_ext1")
    g1 = mx.create_gas_junction(net, name="G1")
    g2 = mx.create_gas_junction(net, name="G2_chp1")
    g3 = mx.create_gas_junction(net, name="G3_g2p")
    g4 = mx.create_gas_junction(net, name="G4_p2g")
    g5 = mx.create_gas_junction(net, name="G5_sec")
    g6 = mx.create_gas_junction(net, name="G6_chp2")

    mx.create_ext_hydr_grid(net, g0, max_import_kgs=0.06)  # primary gas source
    mx.create_gas_source(net, g5, mass_flow=0.04)  # secondary gas source
    mx.create_sink(net, g3, mass_flow=0.015)  # direct gas consumer
    mx.create_sink(net, g4, mass_flow=0.010)  # P2G node + consumer

    mx.create_gas_pipe(net, g0, g1, diameter_m=0.15, length_m=400)
    mx.create_gas_pipe(net, g1, g2, diameter_m=0.12, length_m=300)
    mx.create_gas_pipe(net, g2, g3, diameter_m=0.10, length_m=250)
    mx.create_gas_pipe(net, g1, g4, diameter_m=0.10, length_m=200)
    mx.create_gas_pipe(net, g1, g5, diameter_m=0.12, length_m=250)  # inter-connection
    mx.create_gas_pipe(net, g5, g6, diameter_m=0.10, length_m=150)

    # Supply side (hot, ~356 K)
    s1 = mx.create_water_junction(net, name="s1")
    s2 = mx.create_water_junction(net, name="s2")
    s3 = mx.create_water_junction(net, name="s3")
    s4 = mx.create_water_junction(net, name="s4")
    s5 = mx.create_water_junction(net, name="s5")
    s6 = mx.create_water_junction(net, name="s6")
    mx.create_water_pipe(net, s1, s2, diameter_m=0.25, length_m=100)
    mx.create_water_pipe(net, s2, s3, diameter_m=0.20, length_m=100)
    mx.create_water_pipe(net, s3, s4, diameter_m=0.20, length_m=100)
    mx.create_water_pipe(net, s4, s5, diameter_m=0.15, length_m=100)
    mx.create_water_pipe(net, s5, s6, diameter_m=0.15, length_m=100)
    mx.create_ext_hydr_grid(net, s1, max_import_kgs=0.1)

    # Return side (cold, ~330 K)
    r1 = mx.create_water_junction(net, name="r1")
    r2 = mx.create_water_junction(net, name="r2")
    r3 = mx.create_water_junction(net, name="r3")
    r4 = mx.create_water_junction(net, name="r4")
    r5 = mx.create_water_junction(net, name="r5")
    r6 = mx.create_water_junction(net, name="r6")
    mx.create_water_pipe(net, r1, r2, diameter_m=0.25, length_m=100)
    mx.create_water_pipe(net, r2, r3, diameter_m=0.20, length_m=100)
    mx.create_water_pipe(net, r3, r4, diameter_m=0.20, length_m=100)
    mx.create_water_pipe(net, r4, r5, diameter_m=0.15, length_m=100)
    mx.create_water_pipe(net, r5, r6, diameter_m=0.15, length_m=100)
    mx.create_consume_hydr_grid(net, r1)

    # HEs bridge supply→return
    mx.create_heat_exchanger(net, s4, r4, 0.550)  # 550 kW CHP1 consumer
    mx.create_heat_exchanger(net, s5, r5, 0.300)  # 300 kW P2H consumer
    mx.create_heat_exchanger(net, s6, r6, 0.300)  # 300 kW CHP2 consumer

    # CHP1: G2 → B3 (power), heat from r1→s1.
    # heat_w ≈ 0.40 × 0.026 × 55.08 × 1e6 = 573 kW
    mx.create_chp(
        net,
        power_node_id=b3,
        heat_node_id=r1,
        heat_return_node_id=s1,
        gas_node_id=g2,
        diameter_m=0.15,
        efficiency_power=0.40,
        efficiency_heat=0.40,
        mass_flow_setpoint=0.026,
        regulation=1,
    )
    # CHP2: G6 → B5 (power), heat from r2→s2.
    # heat_w ≈ 0.40 × 0.018 × 55.08 × 1e6 = 397 kW
    mx.create_chp(
        net,
        power_node_id=b5,
        heat_node_id=r2,
        heat_return_node_id=s2,
        gas_node_id=g6,
        diameter_m=0.12,
        efficiency_power=0.38,
        efficiency_heat=0.40,
        mass_flow_setpoint=0.018,
        regulation=1,
    )
    # P2H: B4 (power) → heat from r3→s3 (300 kW).
    # el_mw = 300 000 / (0.95 × 1e6) ≈ 0.316 MW consumed from B4.
    mx.create_p2h(
        net,
        power_node_id=b4,
        heat_node_id=r3,
        heat_return_node_id=s3,
        heat_energy_w=300_000,
        diameter_m=0.15,
        efficiency=0.95,
    )
    # P2G: surplus wind at B0 → gas injection at G4.
    mx.create_p2g(
        net,
        from_node_id=b0,
        to_node_id=g4,
        efficiency=0.70,
        mass_flow_setpoint=0.003,
        regulation=1,
    )
    # G2P: gas peaker at G3 → backup power at B2.
    mx.create_g2p(
        net,
        from_node_id=g3,
        to_node_id=b2,
        efficiency=0.85,
        p_mw_setpoint=0.5,
        regulation=1,
    )
    return net


if __name__ == "__main__":
    from monee import PyomoSolver, run_energy_flow
    from monee.model.formulation import MISOCP_NETWORK_FORMULATION

    net = create_urban_district_net()
    net.apply_formulation(MISOCP_NETWORK_FORMULATION)
    print(run_energy_flow(net, solver=PyomoSolver(), solver_name="gurobi"))

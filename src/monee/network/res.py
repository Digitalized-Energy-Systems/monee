import monee.express as mx
import monee.model as mm


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

    # ── Power (20 kV) ──────────────────────────────────────────────────────
    b0 = mx.create_bus(net, base_kv=20, name="B0_gen")
    b1 = mx.create_bus(net, base_kv=20, name="B1_slack")
    b2 = mx.create_bus(net, base_kv=20, name="B2_load")
    b3 = mx.create_bus(net, base_kv=20, name="B3_load")
    b4 = mx.create_bus(net, base_kv=20, name="B4_load")

    mx.create_ext_power_grid(net, b1, p_mw=0, q_mvar=0, vm_pu=1.0, max_import_mw=4.0)
    mx.create_power_generator(net, b0, p_mw=8, q_mvar=0)
    mx.create_power_load(net, b2, p_mw=3, q_mvar=0.5)
    mx.create_power_load(net, b3, p_mw=5, q_mvar=0.5)
    mx.create_power_load(net, b4, p_mw=4, q_mvar=0.5)

    _line(net, b0, b1, length_m=500, kv_class="20")
    _line(net, b1, b2, length_m=400, kv_class="20")
    _line(net, b2, b3, length_m=300, kv_class="20")
    _line(net, b2, b4, length_m=350, kv_class="20")

    # ── Gas (medium pressure, city distribution) ───────────────────────────
    g0 = mx.create_gas_junction(net, name="G0_ext")
    g1 = mx.create_gas_junction(net, name="G1")
    g2 = mx.create_gas_junction(net, name="G2")
    g3 = mx.create_gas_junction(net, name="G3_sink")
    g4 = mx.create_gas_junction(net, name="G4_sink")

    mx.create_ext_hydr_grid(net, g0, max_import_kgs=0.1)
    mx.create_sink(net, g3, mass_flow=0.04)  # direct gas consumer
    mx.create_sink(net, g4, mass_flow=0.02)  # P2G injection node + consumer

    mx.create_gas_pipe(net, g0, g1, diameter_m=0.15, length_m=400)
    mx.create_gas_pipe(net, g1, g2, diameter_m=0.12, length_m=300)
    mx.create_gas_pipe(net, g2, g3, diameter_m=0.10, length_m=250)
    mx.create_gas_pipe(net, g1, g4, diameter_m=0.10, length_m=200)

    # ── Heat (district heating, supply ~356 K ≈ 83 °C) ────────────────────
    # HeatExchangerLoad branches enforce fixed heat demand; Sinks handle mass
    # balance at the return nodes (sized for 25 K drop: Q/(cp·ΔT)):
    #   CHP consumer:  176 kW / (4186 J/kg·K × 25 K) ≈ 1.684 kg/s
    #   P2H consumer:   20 kW / (4186 J/kg·K × 25 K) ≈ 0.191 kg/s
    h0 = mx.create_water_junction(net, name="H0_ext")
    h1 = mx.create_water_junction(net, name="H1")
    h2 = mx.create_water_junction(net, name="H2")  # CHP heat_return_node
    h3 = mx.create_water_junction(net, name="H3_sink")
    h4 = mx.create_water_junction(net, name="H4")
    h5 = mx.create_water_junction(net, name="H5_sink")  # P2H heat_return_node

    mx.create_ext_hydr_grid(net, h0)
    mx.create_sink(net, h2, mass_flow=0.091)  # return-side mass withdrawal
    mx.create_sink(net, h3, mass_flow=1.684)  # return-side mass withdrawal
    mx.create_sink(net, h5, mass_flow=0.191)  # return-side mass withdrawal

    mx.create_water_pipe(net, h0, h1, diameter_m=0.15, length_m=150)
    # mx.create_water_pipe(net, h1, h2, diameter_m=0.15, length_m=200)
    mx.create_heat_exchanger(
        net, h2, h3, q_mw=0.176, diameter_m=0.15
    )  # 176 kW consumer
    mx.create_water_pipe(net, h1, h4, diameter_m=0.10, length_m=100)
    # mx.create_heat_exchanger(net, h4, h5, q_mw=0.020, diameter_m=0.10)  # 20 kW consumer

    # ── Coupling points ────────────────────────────────────────────────────
    # CHP: gas at G2 → power at B3, heat injected between H1 and H2.
    # heat_w = 0.40 × 0.008 kg/s × 3.6 × 15.3 kWh/kg × 1e6 = 176 256 W
    mx.create_chp(
        net,
        power_node_id=b3,
        heat_node_id=h1,
        heat_return_node_id=h2,
        gas_node_id=g2,
        diameter_m=0.10,
        efficiency_power=0.40,
        efficiency_heat=0.40,
        mass_flow_setpoint=0.008,
        regulation=1,
    )
    # P2H: power at B4 → heat injected between H4 and H5 (20 kW).
    # el_mw = 20 000 W / (0.95 × 1e6) ≈ 0.021 MW consumed from B4.
    mx.create_p2h(
        net,
        power_node_id=b4,
        heat_node_id=h4,
        heat_return_node_id=h5,
        heat_energy_w=20_000,
        diameter_m=0.10,
        efficiency=0.95,
    )
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


def create_industrial_hub_net() -> mm.Network:
    """
    Industrial energy hub: 110 kV meshed power + high-pressure gas, no district heat.

    8 buses (ring + 2 cross-ties) · 7 gas junctions (ring).
    5 CPs: 3× G2P + 2× P2G — strong gas-backup power capacity.
    Suitable for: sparse-heat resilience analysis, gas-turbine backup power testing.
    """
    net = mx.create_multi_energy_network()

    # ── Power (110 kV) ─────────────────────────────────────────────────────
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

    # ── Gas (high pressure) ────────────────────────────────────────────────
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

    # ── Coupling points ────────────────────────────────────────────────────
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
    Regional integrated MES: 120 kV ring power, gas tree, district heating tree.

    8 buses · 8 gas junctions · 5 heat junctions · 5 CPs.
    All coupling point types (CHP, G2P, P2G, P2H) — broadest carrier diversity.
    Single cross-tie in power ring for N-1 security.
    Suitable for: comprehensive CP criticality analysis, all-carrier failure scenarios.

    Heat consumers are modelled as ``HeatExchangerLoad`` branches (fixed heat
    demand in kW), paired with a return-side ``Sink`` for mass balance.  CHP
    supplies 176 kW and P2H 20 kW; the corresponding heat exchangers enforce
    those demands explicitly at a nominal 25 K temperature drop.
    """
    net = mx.create_multi_energy_network()

    # ── Power (120 kV, ring + 1 cross-tie) ────────────────────────────────
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

    # # ── Gas (medium-high pressure, tree) ──────────────────────────────────
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

    # ── Heat (district heating, tree) ─────────────────────────────────────
    # HeatExchangerLoad branches enforce fixed heat demand; Sinks handle mass
    # balance at the return nodes (sized for 25 K drop: Q/(cp·ΔT)):
    #   CHP consumer:  176 kW / (4186 × 25) ≈ 1.684 kg/s
    #   P2H consumer:   20 kW / (4186 × 25) ≈ 0.191 kg/s
    h0 = mx.create_water_junction(net, name="H0_ext")
    h1 = mx.create_water_junction(net, name="H1")
    h2 = mx.create_water_junction(net, name="H2")  # CHP heat_return_node
    h3 = mx.create_water_junction(net, name="H3_sink")
    h4 = mx.create_water_junction(net, name="H4_sink")  # P2H heat_return_node
    h5 = mx.create_water_junction(net, name="H5")

    mx.create_ext_hydr_grid(net, h0)
    mx.create_sink(net, h3, mass_flow=1.684)  # return-side mass withdrawal
    mx.create_sink(net, h5, mass_flow=0.191)  # return-side mass withdrawal

    mx.create_water_pipe(net, h0, h1, diameter_m=0.20, length_m=200)
    mx.create_water_pipe(net, h1, h2, diameter_m=0.20, length_m=200)
    mx.create_water_pipe(net, h3, h4, diameter_m=0.20, length_m=250)
    mx.create_heat_exchanger(
        net, h2, h3, q_mw=0.176, diameter_m=0.15
    )  # 176 kW consumer
    mx.create_heat_exchanger(net, h4, h5, q_mw=0.020, diameter_m=0.10)  # 20 kW consumer

    # ── Coupling points (all types) ────────────────────────────────────────
    # CHP: gas at G1 → power at B2, heat injected between H1 and H2.
    mx.create_chp(
        net,
        power_node_id=b2,
        heat_node_id=h1,
        heat_return_node_id=h2,
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
    # P2H: electric booster at B5, heat injected between H1 and H4 (20 kW).
    mx.create_p2h(
        net,
        power_node_id=b5,
        heat_node_id=h1,
        heat_return_node_id=h4,
        heat_energy_w=20_000,
        diameter_m=0.12,
        efficiency=0.95,
    )
    return net


if __name__ == "__main__":
    from monee import PyomoSolver, run_energy_flow
    from monee.model.formulation import MISOCP_NETWORK_FORMULATION

    net = create_regional_mes_net()
    net.apply_formulation(MISOCP_NETWORK_FORMULATION)
    print(run_energy_flow(net, solver=PyomoSolver()))

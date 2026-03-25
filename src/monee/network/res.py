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


def create_balanced_urban_mes_net() -> mm.Network:
    """
    Balanced urban MES: same 5-bus · 5-gas · 6-heat topology as
    ``create_urban_district_net``, but with carrier energy flows scaled to a
    similar magnitude so that single-carrier failures cause comparable amounts
    of load shedding.

    Carrier direct-load targets (shed-able demand):
        Power  : B2 0.5 MW + B3 0.7 MW + B4 0.5 MW            = 1.7 MW
        Gas    : G3 0.015 kg/s + G4 0.010 kg/s                  ≈ 1.4 MW
                 (using gas LHV 15.3 kWh/kg → 55.08 MJ/kg)
        Heat   : HE consumer H2–H3  q_mw = 0.550 MW             = 0.55 MW

    vs. urban district reference:
        Power ~12 MW · Gas direct ~1.4 MW · Heat direct ~0.2 MW.

    Heat generation is entirely internal (CHP + P2H); the ExtHydrGrid at H0
    acts as the cold-return reference (330 K ≈ 57 °C) for pressure and
    temperature, NOT as a heat source.  At rated output the heat network is
    thermally self-consistent with a 25 K supply-to-return temperature swing:

        CHP  (0.026 kg/s gas → 572 kW heat) heats 5.35 kg/s from 330 K to
        ~356 K; HE consumer (550 kW) cools it back to ~330 K.           [✓]
        P2H  (300 kW) heats 2.87 kg/s on a parallel branch from 330 K to
        ~355 K; the return Sink at H5 represents the consumer.          [✓]

    Coupling points (same types as urban district):
        CHP  : G2 → B3 (0.57 MW power) + H1→H2 (0.57 MW heat),
               gas mass-flow 0.026 kg/s
        P2H  : B4 → H4→H5, 300 kW heat (0.316 MW electric consumption)
        P2G  : B0 → G4, 0.003 kg/s hydrogen
        G2P  : G3 → B2, 0.5 MW electric backup

    Heat network mass-balance sinks (25 K supply-to-return ΔT):
        H2  : 0.10 kg/s  – small bypass consumer at CHP supply node
        H3  : 5.25 kg/s  – 550 kW HE consumer return  (550 000/(4186×25))
        H5  : 2.87 kg/s  – 300 kW P2H return           (300 000/(4186×25))
    """
    net = mx.create_multi_energy_network()

    # ── Power (20 kV) ──────────────────────────────────────────────────────
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

    # ── Gas (medium pressure, city distribution) ───────────────────────────
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

    # ── Heat (district heating) ────────────────────────────────────────────
    # H0 is the cold-return reference (330 K ≈ 57 °C).  The CHP heats the
    # circulating water from 330 K to ~356 K on the H1→H2 branch; the HE
    # consumer cools it back to ~330 K on H2→H3.  The P2H runs a parallel
    # branch H4→H5 and covers its own consumer via the H5 return Sink.
    #
    # Sinks represent the simplified return pipe (Q / (cp × ΔT), ΔT = 25 K):
    #   CHP bypass     :   25 kW → H2 sink  0.10 kg/s  (25 000/(4186×25) ≈ 0.24, rounded)
    #   HE consumer    :  550 kW → H3 sink  5.25 kg/s  (550 000/(4186×25))
    #   P2H            :  300 kW → H5 sink  2.87 kg/s  (300 000/(4186×25))
    h0 = mx.create_water_junction(net, name="H0_ext")
    h1 = mx.create_water_junction(net, name="H1")
    h2 = mx.create_water_junction(net, name="H2")  # CHP heat_return_node
    h3 = mx.create_water_junction(net, name="H3_sink")
    h4 = mx.create_water_junction(net, name="H4")
    h5 = mx.create_water_junction(net, name="H5_sink")  # P2H heat_return_node

    # Cold-return reference: 330 K ≈ 57 °C (NOT the heat source).
    mx.create_water_ext_grid(net, h0, t_k=330)
    mx.create_sink(net, h2, mass_flow=0.10)  # small bypass at CHP supply node
    mx.create_sink(net, h3, mass_flow=5.25)  # HE consumer return
    mx.create_sink(net, h5, mass_flow=2.87)  # P2H return

    mx.create_water_pipe(net, h0, h1, diameter_m=0.25, length_m=150)
    mx.create_heat_exchanger(
        net, h2, h3, q_mw=0.550, diameter_m=0.25
    )  # 550 kW consumer
    mx.create_water_pipe(net, h1, h4, diameter_m=0.15, length_m=100)

    # ── Coupling points ────────────────────────────────────────────────────
    # CHP: gas at G2 → power at B3, heat sourced between H1 and H2.
    # Heats supply water from ~330 K (H1) to ~356 K (H2).
    # heat_w = 0.40 × 0.026 kg/s × 3.6 × 15.3 kWh/kg × 1e6 ≈ 572 kW
    mx.create_chp(
        net,
        power_node_id=b3,
        heat_node_id=h1,
        heat_return_node_id=h2,
        gas_node_id=g2,
        diameter_m=0.15,
        efficiency_power=0.40,
        efficiency_heat=0.40,
        mass_flow_setpoint=0.026,
        regulation=1,
    )
    # P2H: power at B4 → 300 kW heat injected between H4 and H5.
    # el_mw = 300 000 / (0.95 × 1e6) ≈ 0.316 MW consumed from B4.
    mx.create_p2h(
        net,
        power_node_id=b4,
        heat_node_id=h4,
        heat_return_node_id=h5,
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
    # ── 24-point unit profiles (fraction of rated, winter weekday) ──────────
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

    # ── Power loads ─────────────────────────────────────────────────────────
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

    # ── District-heat exchanger loads ────────────────────────────────────────
    # q_w_set is stored in Watts with sign convention: negative = consumer.
    for branch in net.branches_by_type(mm.HeatExchangerLoad):
        q_w_rated = branch.model.q_w_set  # e.g. -800 000 W for 800 kW consumer
        td.add_branch_series(branch.id, "q_w_set", [q_w_rated * f for f in h_series])

    return td


def create_resilient_urban_mes_net() -> mm.Network:
    """
    Resilient urban MES: 6 buses · 7 gas junctions · 9 heat junctions · 6 CPs.

    Extends ``create_balanced_urban_mes_net`` with a second gas source, a second
    CHP, a solar generator, and a second district-heat consumer branch.  The
    richer topology makes two phenomena directly observable:

    **Carrier dependence via coupling points**
        Each CHP converts gas → power AND heat simultaneously.  Losing the
        primary gas source (G0_ext1) curtails CHP1 and simultaneously reduces
        power generation at B3 and heat supply at H1/H2.  If the secondary
        source (G5_ext2) also fails, CHP2 at B5/H6 is also lost, cascading
        through two carriers at once.  P2H couples electricity → heat: a
        power-side fault that isolates B4 also kills heat supply on H4/H5.

    **Resilience from redundancy**
        * Two gas sources (G0_ext1, G5_ext2): losing one leaves the other to
          serve both CHPs via the G1–G5 inter-connection pipe.
        * Two CHPs in independent sub-trees: CHP1 failure leaves CHP2 + solar
          + wind fully operational.
        * Solar generator at B5 is independent of gas and heat — it keeps the
          power grid alive even if both CHPs and G2P are off.
        * Two HE consumers (HE1 at H2→H3, HE2 at H7→H8) in separate heat
          branches: failure of the CHP1 branch does not affect HE2.
        * G2P at G3→B2 provides gas-backed power regardless of wind/solar.

    Topology summary::

        Power (20 kV):
            B0(wind 1.5 MW)─B1(slack ≤1 MW)─B2(0.5 MW, G2P output)
                                            ─B3(0.3 MW, CHP1 ~0.57 MW)
                                            ─B4(0.5 MW, P2H consumer)
            B1─B5(0.4 MW, CHP2 ~0.38 MW, solar 0.8 MW)

        Gas (medium pressure):
            G0_ext1─G1─G2(CHP1)─G3(G2P + sink 0.015 kg/s)
                      ─G4(P2G + sink 0.010 kg/s)
                      ─G5─G6(CHP2)
            G5_ext2─G5

        Heat (district heating, return ref 330 K):
            H0_ret─H1(CHP1 inlet)─[CHP1]─H2(hot ~356 K)─HE1(550 kW)─H3(sink)
                  ─H4(P2H inlet)─[P2H]─H5(sink 300 kW)
                  ─H6(CHP2 inlet)─[CHP2]─H7(hot ~356 K)─HE2(300 kW)─H8(sink)

    Heat mass-balance sinks (25 K ΔT, Q/(cp·ΔT)):
        H2 : 0.10 kg/s – CHP1 bypass
        H3 : 5.25 kg/s – HE1 consumer (550 kW)
        H5 : 2.87 kg/s – P2H consumer (300 kW)
        H7 : 0.95 kg/s – CHP2 bypass  (~100 kW)
        H8 : 2.87 kg/s – HE2 consumer (300 kW)
    """
    net = mx.create_multi_energy_network()

    # ── Power (20 kV) ──────────────────────────────────────────────────────
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

    # ── Gas (medium pressure) ──────────────────────────────────────────────
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

    # ── Heat (district heating, return reference 330 K) ───────────────────
    # Two independent supply branches share the same cold-return node (H0).
    # CHP1 branch: H0→H1→(CHP1)→H2→HE1→H3
    # P2H  branch: H1→H4→(P2H)→H5
    # CHP2 branch: H0→H6→(CHP2)→H7→HE2→H8
    #
    # Sink mass flows (Q / (cp × ΔT), ΔT = 25 K, cp = 4186 J/kg·K):
    #   H2 bypass :  10 kW → 0.10 kg/s   H3 (HE1) : 550 kW → 5.25 kg/s
    #   H5 (P2H)  : 300 kW → 2.87 kg/s
    #   H7 bypass : 100 kW → 0.95 kg/s   H8 (HE2) : 300 kW → 2.87 kg/s
    h0 = mx.create_water_junction(net, name="H0_ret")
    h1 = mx.create_water_junction(net, name="H1_chp1_in")
    h2 = mx.create_water_junction(net, name="H2_chp1_hot")
    h3 = mx.create_water_junction(net, name="H3_he1_ret")
    h4 = mx.create_water_junction(net, name="H4_p2h_in")
    h5 = mx.create_water_junction(net, name="H5_p2h_ret")
    h6 = mx.create_water_junction(net, name="H6_chp2_in")
    h7 = mx.create_water_junction(net, name="H7_chp2_hot")
    h8 = mx.create_water_junction(net, name="H8_he2_ret")

    mx.create_water_ext_grid(net, h0, t_k=330)  # cold-return pressure/temp reference
    mx.create_sink(net, h2, mass_flow=0.10)  # CHP1 bypass
    mx.create_sink(net, h3, mass_flow=5.25)  # HE1 consumer return (550 kW)
    mx.create_sink(net, h5, mass_flow=2.87)  # P2H consumer return (300 kW)
    mx.create_sink(net, h7, mass_flow=0.95)  # CHP2 bypass (~100 kW)
    mx.create_sink(net, h8, mass_flow=2.87)  # HE2 consumer return (300 kW)

    mx.create_water_pipe(net, h0, h1, diameter_m=0.25, length_m=150)
    mx.create_heat_exchanger(net, h2, h3, q_mw=0.550, diameter_m=0.25)  # HE1: 550 kW
    mx.create_water_pipe(net, h1, h4, diameter_m=0.15, length_m=100)
    mx.create_water_pipe(net, h0, h6, diameter_m=0.20, length_m=120)  # CHP2 branch
    mx.create_heat_exchanger(net, h7, h8, q_mw=0.300, diameter_m=0.15)  # HE2: 300 kW

    # ── Coupling points ────────────────────────────────────────────────────
    # CHP1: G2 → B3 (power) + H1/H2 (heat). Connects gas ↔ power ↔ heat.
    # heat_w ≈ 0.40 × 0.026 × 55.08 × 1e6 = 573 kW
    mx.create_chp(
        net,
        power_node_id=b3,
        heat_node_id=h1,
        heat_return_node_id=h2,
        gas_node_id=g2,
        diameter_m=0.15,
        efficiency_power=0.40,
        efficiency_heat=0.40,
        mass_flow_setpoint=0.026,
        regulation=1,
    )
    # CHP2: G6 → B5 (power) + H6/H7 (heat). Second independent CP.
    # heat_w ≈ 0.40 × 0.018 × 55.08 × 1e6 = 397 kW;  power ≈ 376 kW
    mx.create_chp(
        net,
        power_node_id=b5,
        heat_node_id=h6,
        heat_return_node_id=h7,
        gas_node_id=g6,
        diameter_m=0.12,
        efficiency_power=0.38,
        efficiency_heat=0.40,
        mass_flow_setpoint=0.018,
        regulation=1,
    )
    # P2H: B4 (power) → H4/H5 (heat). Couples electricity → heat.
    # el_mw = 300 000 / (0.95 × 1e6) ≈ 0.316 MW consumed from B4.
    mx.create_p2h(
        net,
        power_node_id=b4,
        heat_node_id=h4,
        heat_return_node_id=h5,
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

    net = create_resilient_urban_mes_net()
    net.apply_formulation(MISOCP_NETWORK_FORMULATION)
    print(run_energy_flow(net, solver=PyomoSolver()))

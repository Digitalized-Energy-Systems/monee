"""
Multi-energy benchmark grid for grid restoration studies.

Topology
--------

**Electricity** (~27 buses):
    Two 110 kV feeders (ext-grids) supply a meshed 20 kV distribution
    backbone via transformers.  Three open chains (industrial,
    commercial, residential) carry loads and distributed generation.
    A battery provides storage flexibility.

**Gas** (~30 junctions):
    Two high-pressure feeders supply a medium-pressure distribution
    network via compressors.  Three distribution chains serve gas sinks.
    A gas storage cavern adds flexibility.

**Thermal / District Heating** (~55 junctions -- 32 supply + 23 return):
    Two heat plants feed a supply-pipe tree.  Heat exchangers at
    consumer nodes bridge supply to return junctions.  Sinks on return
    nodes withdraw cooled water.  A thermal storage tank is present.

**Coupling points** (10 total -- all variants):
    - 3 x CHP  (gas -> electricity + heat)
    - 2 x P2H  (electricity -> heat)
    - 2 x G2H  (gas -> heat)
    - 1 x P2G  (electricity -> gas)
    - 2 x G2P  (gas -> electricity)

**Extensions**:
    - ``GasLinepack``              -- inherent gas storage in pipes
    - ``LumpedThermalCapacitance``  -- thermal inertia at junctions (LTC)

The grid is intended for multi-period optimisation, restoration
sequence planning, and resilience studies.

.. note::

   This benchmark is designed for the ``PyomoSolver`` with **ipopt**.
   Recommended solver options::

       from monee.solver.pyo import DEFAULT_SOLVER_OPTIONS
       DEFAULT_SOLVER_OPTIONS["max_iter"] = 5000
       DEFAULT_SOLVER_OPTIONS["tol"] = 1e-4
       DEFAULT_SOLVER_OPTIONS["acceptable_tol"] = 1e-3
       DEFAULT_SOLVER_OPTIONS["acceptable_iter"] = 10
"""

from __future__ import annotations

import monee.express as mx
import monee.model as mm
from monee.model import GasLinepack, LumpedThermalCapacitance

# -- electrical line catalogue --
_HV_LINE = dict(length_m=5000, r_ohm_per_m=3e-5, x_ohm_per_m=4e-5, parallel=1)
_MV_LINE = dict(length_m=200, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5, parallel=1)
_MV_LINE_LONG = dict(length_m=500, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5, parallel=1)

# -- gas pipe catalogue (short lengths for solver convergence) --
_HP_PIPE = dict(diameter_m=0.5, length_m=200)
_MP_PIPE = dict(diameter_m=0.3, length_m=100)
_MP_PIPE_SHORT = dict(diameter_m=0.3, length_m=80)

# -- water pipe catalogue --
_DH_PIPE = dict(diameter_m=0.15, length_m=100)
_DH_PIPE_LONG = dict(diameter_m=0.15, length_m=150)
_DH_TRUNK = dict(diameter_m=0.25, length_m=100)

# -- heat sizing helpers --
_CP = 4186.0  # water specific heat, J/(kg K)
_DT = 25.0  # supply-return temperature drop, K
_HHV_MJ = 55.08  # gas higher heating value, MJ/kg (l-gas)


def _mf(q_mw):
    """Convert heat load (MW) to mass flow (kg/s) for a 25 K drop."""
    return q_mw * 1e6 / (_CP * _DT)


def create_restoration_benchmark(
    *,
    linepack: bool = True,
    ltc: bool = True,
    misocp: bool = False,
) -> mm.Network:
    """
    Build and return the multi-energy restoration benchmark grid.

    Args:
        linepack: Attach :class:`GasLinepack` extension.  Default ``True``.
        ltc: Attach :class:`LumpedThermalCapacitance` extension.
            Default ``True``.
        misocp: Replace the default AC power-flow formulation with the
            MISOCP relaxation (suitable for ``PyomoSolver``).
            Default ``False``.

    Returns:
        mm.Network: The fully built network.
    """
    net = mx.create_multi_energy_network()

    # ==================================================================
    # 1.  ELECTRICITY GRID  (27 buses)
    # ==================================================================
    # Two 110 kV feeders
    hv1 = mx.create_bus(net, base_kv=110, name="HV_feeder_1")
    hv2 = mx.create_bus(net, base_kv=110, name="HV_feeder_2")
    mx.create_ext_power_grid(net, hv1, vm_pu=1.0, name="ExtGrid_HV1")
    mx.create_ext_power_grid(net, hv2, vm_pu=1.0, name="ExtGrid_HV2")

    # HV tie line
    mx.create_line(net, hv1, hv2, **_HV_LINE, name="HV_tie")

    # 20 kV backbone buses (4 substations)
    mv_sub = []
    for i in range(4):
        mv_sub.append(mx.create_bus(net, base_kv=20, name=f"MV_sub_{i}"))

    # Transformers: HV -> MV
    mx.create_trafo(net, hv1, mv_sub[0], sn_trafo_mva=63, name="Trafo_HV1_S0")
    mx.create_trafo(net, hv1, mv_sub[1], sn_trafo_mva=63, name="Trafo_HV1_S1")
    mx.create_trafo(net, hv2, mv_sub[2], sn_trafo_mva=63, name="Trafo_HV2_S2")
    mx.create_trafo(net, hv2, mv_sub[3], sn_trafo_mva=63, name="Trafo_HV2_S3")

    # Chain A: industrial (5 buses, mv_sub[0] -> mv_sub[1])
    ring_a = []
    for i in range(5):
        ring_a.append(mx.create_bus(net, base_kv=20, name=f"Ind_{i}"))
    mx.create_line(net, mv_sub[0], ring_a[0], **_MV_LINE, name="Ind_in")
    for i in range(4):
        mx.create_line(
            net, ring_a[i], ring_a[i + 1], **_MV_LINE, name=f"Ind_{i}_{i + 1}"
        )
    mx.create_line(net, ring_a[4], mv_sub[1], **_MV_LINE_LONG, name="Ind_out")

    # Chain B: commercial (5 buses, mv_sub[1] -> mv_sub[2])
    ring_b = []
    for i in range(5):
        ring_b.append(mx.create_bus(net, base_kv=20, name=f"Com_{i}"))
    mx.create_line(net, mv_sub[1], ring_b[0], **_MV_LINE, name="Com_in")
    for i in range(4):
        mx.create_line(
            net, ring_b[i], ring_b[i + 1], **_MV_LINE, name=f"Com_{i}_{i + 1}"
        )
    mx.create_line(net, ring_b[4], mv_sub[2], **_MV_LINE_LONG, name="Com_out")

    # Chain C: residential (5 buses, mv_sub[2] -> mv_sub[3])
    ring_c = []
    for i in range(5):
        ring_c.append(mx.create_bus(net, base_kv=20, name=f"Res_{i}"))
    mx.create_line(net, mv_sub[2], ring_c[0], **_MV_LINE, name="Res_in")
    for i in range(4):
        mx.create_line(
            net, ring_c[i], ring_c[i + 1], **_MV_LINE, name=f"Res_{i}_{i + 1}"
        )
    mx.create_line(net, ring_c[4], mv_sub[3], **_MV_LINE_LONG, name="Res_out")

    # MV tie (meshing substations)
    mx.create_line(net, mv_sub[0], mv_sub[3], **_MV_LINE_LONG, name="MV_tie_03")

    # Loads on chain A (industrial)
    for i, (p, q) in enumerate(
        [(0.50, 0.10), (0.30, 0.06), (0.40, 0.08), (0.25, 0.05), (0.35, 0.07)]
    ):
        mx.create_power_load(net, ring_a[i], p_mw=p, q_mvar=q, name=f"Load_Ind_{i}")
    mx.create_power_generator(net, ring_a[2], p_mw=0.3, q_mvar=0.0, name="DG_Ind_solar")

    # Loads on chain B (commercial)
    for i, (p, q) in enumerate(
        [(0.20, 0.04), (0.15, 0.03), (0.30, 0.06), (0.25, 0.05), (0.18, 0.04)]
    ):
        mx.create_power_load(net, ring_b[i], p_mw=p, q_mvar=q, name=f"Load_Com_{i}")
    mx.create_power_generator(
        net, ring_b[3], p_mw=0.15, q_mvar=0.0, name="DG_Com_solar"
    )

    # Loads on chain C (residential)
    for i, (p, q) in enumerate(
        [(0.10, 0.02), (0.08, 0.015), (0.12, 0.025), (0.09, 0.018), (0.07, 0.014)]
    ):
        mx.create_power_load(net, ring_c[i], p_mw=p, q_mvar=q, name=f"Load_Res_{i}")
    mx.create_power_generator(net, ring_c[4], p_mw=0.08, q_mvar=0.0, name="DG_Res_wind")

    # Battery storage
    net.child_to(
        mm.ElectricStorage(e_mwh_initial=2.0, e_mwh_max=5.0, p_max_mw=1.0),
        mv_sub[1],
        name="Battery_MV1",
    )

    # ==================================================================
    # 2.  GAS GRID  (30 junctions)
    # ==================================================================
    # Two HP feeders
    gf1 = mx.create_gas_junction(net, name="Gas_HP_feeder_1")
    gf2 = mx.create_gas_junction(net, name="Gas_HP_feeder_2")
    mx.create_ext_hydr_grid(net, gf1, grid_key=mm.GAS_KEY, name="GasExtGrid_1")
    mx.create_ext_hydr_grid(net, gf2, grid_key=mm.GAS_KEY, name="GasExtGrid_2")

    # HP trunk
    mx.create_gas_pipe(net, gf1, gf2, **_HP_PIPE, name="Gas_HP_trunk")

    # MP distribution: 3 branches x 8 junctions = 24 junctions
    gas_a = []
    gas_b = []
    gas_c = []
    for i in range(8):
        gas_a.append(mx.create_gas_junction(net, name=f"Gas_A{i}"))
    for i in range(8):
        gas_b.append(mx.create_gas_junction(net, name=f"Gas_B{i}"))
    for i in range(8):
        gas_c.append(mx.create_gas_junction(net, name=f"Gas_C{i}"))

    # Compressors HP -> first MP junction
    mx.create_compressor(
        net, gf1, gas_a[0], compression_ratio=1.4, max_flow_kgs=3.0, name="Comp_A"
    )
    mx.create_compressor(
        net, gf1, gas_b[0], compression_ratio=1.4, max_flow_kgs=3.0, name="Comp_B"
    )
    mx.create_compressor(
        net, gf2, gas_c[0], compression_ratio=1.4, max_flow_kgs=3.0, name="Comp_C"
    )

    # MP pipes -- linear chains (no cross-ties for solver stability)
    for branch, label in [(gas_a, "GasA"), (gas_b, "GasB"), (gas_c, "GasC")]:
        for i in range(7):
            mx.create_gas_pipe(
                net, branch[i], branch[i + 1], **_MP_PIPE, name=f"{label}_{i}_{i + 1}"
            )

    # 2 spur junctions
    gs0 = mx.create_gas_junction(net, name="Gas_spur_0")
    gs1 = mx.create_gas_junction(net, name="Gas_spur_1")
    mx.create_gas_pipe(net, gas_a[3], gs0, **_MP_PIPE_SHORT, name="GasSpur_A3")
    mx.create_gas_pipe(net, gas_b[4], gs1, **_MP_PIPE_SHORT, name="GasSpur_B4")

    # Gas sinks
    for node, mf in [
        (gas_a[1], 0.05),
        (gas_a[3], 0.06),
        (gas_a[5], 0.04),
        (gas_a[7], 0.03),
        (gas_b[1], 0.04),
        (gas_b[3], 0.05),
        (gas_b[5], 0.03),
        (gas_b[7], 0.02),
        (gas_c[1], 0.02),
        (gas_c[3], 0.03),
        (gas_c[5], 0.02),
        (gas_c[7], 0.01),
        (gs0, 0.03),
        (gs1, 0.02),
    ]:
        mx.create_gas_sink(net, node, mass_flow=mf)

    # Gas storage cavern
    net.child_to(
        mm.GasStorage(
            m_stored_kg_initial=500.0,
            m_stored_kg_max=1200.0,
            flow_max_kgs=0.5,
        ),
        gas_b[2],
        name="GasCavern_B2",
    )

    # ==================================================================
    # 3.  THERMAL / DISTRICT HEATING
    # ==================================================================
    # Supply chain: 32 junctions in two sub-chains connected by cross-tie
    hs = []
    for i in range(32):
        hs.append(mx.create_water_junction(net, name=f"Heat_S_{i}"))

    # Ext-grids (heat plants)
    mx.create_ext_hydr_grid(net, hs[0], grid_key=mm.WATER_KEY, name="HeatPlant_1")
    mx.create_ext_hydr_grid(net, hs[16], grid_key=mm.WATER_KEY, name="HeatPlant_2")

    # Supply trunk pipes
    for i in range(15):
        kw = _DH_TRUNK if i < 3 else _DH_PIPE
        mx.create_water_pipe(net, hs[i], hs[i + 1], **kw, name=f"HS_{i}_{i + 1}")
    for i in range(16, 31):
        kw = _DH_TRUNK if i < 19 else _DH_PIPE
        mx.create_water_pipe(net, hs[i], hs[i + 1], **kw, name=f"HS_{i}_{i + 1}")
    mx.create_water_pipe(net, hs[8], hs[24], **_DH_PIPE_LONG, name="HS_tie")

    # Heat loads: HX from supply -> return junctions with sinks
    _heat_loads = [
        (2, 0.020),
        (4, 0.030),
        (6, 0.025),
        (9, 0.015),
        (10, 0.028),
        (11, 0.012),
        (13, 0.018),
        (15, 0.014),
        (18, 0.024),
        (20, 0.035),
        (22, 0.020),
        (24, 0.012),
        (25, 0.016),
        (27, 0.026),
        (29, 0.016),
        (31, 0.012),
    ]
    for si, q_mw in _heat_loads:
        rj = mx.create_water_junction(net, name=f"Heat_R_{si}")
        mx.create_heat_exchanger(
            net, hs[si], rj, q_mw=q_mw, diameter_m=0.12, name=f"HX_load_{si}"
        )
        mx.create_water_sink(net, rj, mass_flow=_mf(q_mw))

    # Return junctions for coupling points
    hr_chp1 = mx.create_water_junction(net, name="Heat_R_CHP1")
    hr_chp2 = mx.create_water_junction(net, name="Heat_R_CHP2")
    hr_chp3 = mx.create_water_junction(net, name="Heat_R_CHP3")
    hr_p2h1 = mx.create_water_junction(net, name="Heat_R_P2H1")
    hr_p2h2 = mx.create_water_junction(net, name="Heat_R_P2H2")
    hr_g2h1 = mx.create_water_junction(net, name="Heat_R_G2H1")
    hr_g2h2 = mx.create_water_junction(net, name="Heat_R_G2H2")

    # Thermal storage
    net.child_to(
        mm.ThermalStorage(
            m_stored_kg_initial=2000.0,
            m_stored_kg_max=5000.0,
            flow_max_kgs=1.0,
        ),
        hs[1],
        name="ThermalTank_S1",
    )

    # ==================================================================
    # 4.  COUPLING POINTS  (10 -- all variants)
    # ==================================================================

    # CHP 1: industrial
    _chp1_mf = 0.008
    _chp1_q_w = _chp1_mf * _HHV_MJ * 1e6 * 0.40
    mx.create_chp(
        net,
        power_node_id=ring_a[1],
        heat_node_id=hs[3],
        heat_return_node_id=hr_chp1,
        gas_node_id=gas_a[2],
        diameter_m=0.10,
        efficiency_power=0.40,
        efficiency_heat=0.40,
        mass_flow_setpoint=_chp1_mf,
    )
    mx.create_water_sink(net, hr_chp1, mass_flow=_mf(_chp1_q_w / 1e6))

    # CHP 2: commercial
    _chp2_mf = 0.006
    _chp2_q_w = _chp2_mf * _HHV_MJ * 1e6 * 0.42
    mx.create_chp(
        net,
        power_node_id=ring_b[2],
        heat_node_id=hs[19],
        heat_return_node_id=hr_chp2,
        gas_node_id=gas_b[3],
        diameter_m=0.10,
        efficiency_power=0.38,
        efficiency_heat=0.42,
        mass_flow_setpoint=_chp2_mf,
    )
    mx.create_water_sink(net, hr_chp2, mass_flow=_mf(_chp2_q_w / 1e6))

    # CHP 3: residential
    _chp3_mf = 0.004
    _chp3_q_w = _chp3_mf * _HHV_MJ * 1e6 * 0.45
    mx.create_chp(
        net,
        power_node_id=ring_c[3],
        heat_node_id=hs[26],
        heat_return_node_id=hr_chp3,
        gas_node_id=gas_c[4],
        diameter_m=0.10,
        efficiency_power=0.35,
        efficiency_heat=0.45,
        mass_flow_setpoint=_chp3_mf,
    )
    mx.create_water_sink(net, hr_chp3, mass_flow=_mf(_chp3_q_w / 1e6))

    # P2H 1: industrial
    _p2h1_q_w = 20_000
    mx.create_p2h(
        net,
        power_node_id=ring_a[4],
        heat_node_id=hs[5],
        heat_return_node_id=hr_p2h1,
        heat_energy_w=_p2h1_q_w,
        diameter_m=0.10,
        efficiency=0.95,
    )
    mx.create_water_sink(net, hr_p2h1, mass_flow=_mf(_p2h1_q_w / 1e6))

    # P2H 2: commercial
    _p2h2_q_w = 15_000
    mx.create_p2h(
        net,
        power_node_id=ring_b[4],
        heat_node_id=hs[21],
        heat_return_node_id=hr_p2h2,
        heat_energy_w=_p2h2_q_w,
        diameter_m=0.10,
        efficiency=0.92,
    )
    mx.create_water_sink(net, hr_p2h2, mass_flow=_mf(_p2h2_q_w / 1e6))

    # G2H 1: industrial gas boiler
    _g2h1_q_w = 18_000
    mx.create_g2h(
        net,
        gas_node_id=gas_a[4],
        heat_node_id=hs[7],
        heat_return_node_id=hr_g2h1,
        heat_energy_w=_g2h1_q_w,
        diameter_m=0.10,
        efficiency=0.90,
    )
    mx.create_water_sink(net, hr_g2h1, mass_flow=_mf(_g2h1_q_w / 1e6))

    # G2H 2: residential gas boiler
    _g2h2_q_w = 12_000
    mx.create_g2h(
        net,
        gas_node_id=gas_c[6],
        heat_node_id=hs[28],
        heat_return_node_id=hr_g2h2,
        heat_energy_w=_g2h2_q_w,
        diameter_m=0.10,
        efficiency=0.88,
    )
    mx.create_water_sink(net, hr_g2h2, mass_flow=_mf(_g2h2_q_w / 1e6))

    # # P2G: electrolyser
    # mx.create_p2g(
    #     net, from_node_id=ring_a[4], to_node_id=gas_a[7],
    #     efficiency=0.65, mass_flow_setpoint=0.005,
    # )

    # # G2P 1: gas turbine (industrial)
    # mx.create_g2p(
    #     net, from_node_id=gas_b[6], to_node_id=ring_b[4],
    #     efficiency=0.42, p_mw_setpoint=0.3,
    # )

    # # G2P 2: peaker (residential)
    # mx.create_g2p(
    #     net, from_node_id=gas_c[2], to_node_id=ring_c[1],
    #     efficiency=0.38, p_mw_setpoint=0.15,
    # )

    # ==================================================================
    # 5.  EXTENSIONS
    # ==================================================================
    if linepack:
        net.add_extension(GasLinepack())
    if ltc:
        net.add_extension(LumpedThermalCapacitance())

    if misocp:
        from monee.model.formulation import MISOCP_NETWORK_FORMULATION

        net.apply_formulation(MISOCP_NETWORK_FORMULATION)

    return net


if __name__ == "__main__":
    res = create_restoration_benchmark(linepack=False, ltc=True, misocp=True)
    from monee.solver import PyomoSolver

    print(PyomoSolver().solve(res))

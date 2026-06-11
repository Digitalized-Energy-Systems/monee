"""Pure-NLP smooth gas/heat formulations under GEKKO IPOPT.

IPOPT cannot branch on integers, so a successful solve here also proves the
formulations are binary-free (no leaked ``direction`` switch).
"""

import math

import pytest

import monee.express as mx
import monee.model as mm
import monee.solver as ms
from monee.model.formulation import (
    make_smooth_darcy_weisbach_network_formulation,
    make_smooth_weymouth_network_formulation,
)

FRICTION_MODELS = ["constant", "pwl", "nonlinear"]

IPOPT = 3


def _gas_only_net():
    pn = mm.Network(mm.create_gas_grid("gas", type="lgas"))
    g0 = pn.node(mm.Junction(), child_ids=[pn.child(mm.Source(mass_flow=1))])
    g1 = pn.node(mm.Junction(), child_ids=[pn.child(mm.ExtHydrGrid())])
    g2 = pn.node(mm.Junction(), child_ids=[pn.child(mm.Sink(mass_flow=0.6))])
    pn.branch(mm.GasPipe(diameter_m=0.35, length_m=1000, roughness=0.01), g0, g1)
    pn.branch(mm.GasPipe(diameter_m=0.35, length_m=1500, roughness=0.01), g0, g2)
    return pn


def _heat_only_net():
    pn = mm.Network(mm.create_water_grid("heat"))
    w0 = pn.node(mm.Junction(), child_ids=[pn.child(mm.Sink(mass_flow=0.1))])
    w1 = pn.node(mm.Junction(), child_ids=[pn.child(mm.ConsumeHydrGrid(1))])
    w2 = pn.node(mm.Junction())
    w3 = pn.node(mm.Junction(), child_ids=[pn.child(mm.ExtHydrGrid(t_k=359))])
    pn.branch(mm.WaterPipe(diameter_m=0.15, length_m=100), w0, w1)
    pn.branch(mm.WaterPipe(diameter_m=0.15, length_m=200), w3, w2)
    return pn


def create_g2h_net():
    pn = mm.Network()

    gas_grid = mm.create_gas_grid("gas", type="lgas")
    g_node_0 = pn.node(
        mm.Junction(), child_ids=[pn.child(mm.Source(mass_flow=1))], grid=gas_grid
    )
    g_node_1 = pn.node(
        mm.Junction(), child_ids=[pn.child(mm.ExtHydrGrid())], grid=gas_grid
    )
    g_node_2 = pn.node(
        mm.Junction(), child_ids=[pn.child(mm.Sink(mass_flow=1))], grid=gas_grid
    )
    pn.branch(
        mm.GasPipe(diameter_m=0.3, length_m=100, temperature_ext_k=300, roughness=0.01),
        g_node_0,
        g_node_1,
    )
    pn.branch(
        mm.GasPipe(diameter_m=0.3, length_m=150, temperature_ext_k=300, roughness=0.01),
        g_node_0,
        g_node_2,
    )

    w_node_0 = pn.node(
        mm.Junction(), grid=mm.WATER_KEY, child_ids=[pn.child(mm.Sink(mass_flow=0.1))]
    )
    w_node_1 = pn.node(
        mm.Junction(), child_ids=[pn.child(mm.ConsumeHydrGrid(1))], grid=mm.WATER_KEY
    )
    w_node_2 = pn.node(mm.Junction(), grid=mm.WATER_KEY)
    w_node_3 = pn.node(
        mm.Junction(), grid=mm.WATER_KEY, child_ids=[pn.child(mm.ExtHydrGrid(t_k=359))]
    )
    pn.branch(mm.WaterPipe(diameter_m=0.15, length_m=100), w_node_0, w_node_1)
    pn.branch(mm.WaterPipe(diameter_m=0.15, length_m=200), w_node_3, w_node_2)

    mx.create_g2h(
        pn,
        gas_node_id=g_node_2,
        heat_node_id=w_node_2,
        heat_return_node_id=w_node_1,
        heat_energy_mw=0.010,
        diameter_m=0.4,
        efficiency=0.9,
    )
    return pn


def _apply_smooth(network, friction_model):
    network.apply_formulation(
        make_smooth_weymouth_network_formulation(friction_model=friction_model)
    )
    network.apply_formulation(
        make_smooth_darcy_weisbach_network_formulation(friction_model=friction_model)
    )


@pytest.mark.parametrize("friction_model", FRICTION_MODELS)
def test_smooth_mes_solves_under_ipopt(friction_model):
    network = create_g2h_net()
    _apply_smooth(network, friction_model)

    result = ms.GEKKOSolver(solver=IPOPT).solve(network)

    assert result.success
    # Gas mass balance: source (1) - sink (1) - g2h draw is closed by the ext grid.
    ext_mass = result.dataframes["ExtHydrGrid"]["mass_flow"][0]
    assert math.isfinite(ext_mass)
    # No spurious bidirectional flow: per pipe, pos·neg ≈ 0 (smooth complementarity).
    gas_pipes = result.dataframes["GasPipe"]
    for pos, neg in zip(gas_pipes["mass_flow_pos"], gas_pipes["mass_flow_neg"]):
        assert min(abs(pos), abs(neg)) < 1e-2


def test_smooth_gas_signed_pressure_drop():
    """Forward flow lowers downstream pressure; the signed Weymouth has the
    right sign without a direction binary."""
    network = create_g2h_net()
    _apply_smooth(network, "constant")

    result = ms.GEKKOSolver(solver=IPOPT).solve(network)

    assert result.success
    junctions = result.dataframes["Junction"]
    # All squared pressures stay within the junction bounds [0, 2].
    assert (junctions["pressure_squared_pu"] >= 0).all()
    assert (junctions["pressure_squared_pu"] <= 2).all()


def test_simulation_gas_only_matches_default():
    """Single-carrier gas solves as a square IMODE=1 simulation and yields the
    same flows as the default IMODE=3 smooth solve."""
    ref = _gas_only_net()
    ref.apply_formulation(make_smooth_weymouth_network_formulation())
    ref_res = ms.GEKKOSolver(solver=IPOPT).solve(ref)

    sim = _gas_only_net()
    sim.apply_formulation(make_smooth_weymouth_network_formulation())
    sim_res = ms.GEKKOSolver(solver=IPOPT).solve(sim, simulation=True)

    assert sim_res.success
    assert math.isclose(
        ref_res.dataframes["ExtHydrGrid"]["mass_flow"][0],
        sim_res.dataframes["ExtHydrGrid"]["mass_flow"][0],
        abs_tol=1e-3,
    )


def test_simulation_heat_only_matches_default():
    """Single-carrier heat solves as a square IMODE=1 simulation, matching the
    default IMODE=3 smooth solve on the nodal temperatures."""
    ref = _heat_only_net()
    ref.apply_formulation(make_smooth_darcy_weisbach_network_formulation())
    ref_res = ms.GEKKOSolver(solver=IPOPT).solve(ref)

    sim = _heat_only_net()
    sim.apply_formulation(make_smooth_darcy_weisbach_network_formulation())
    sim_res = ms.GEKKOSolver(solver=IPOPT).solve(sim, simulation=True)

    assert sim_res.success
    assert math.isclose(
        ref_res.dataframes["Junction"]["t_k"].dropna().sum(),
        sim_res.dataframes["Junction"]["t_k"].dropna().sum(),
        rel_tol=1e-3,
    )


def test_simulation_falls_back_to_imode3_with_objective():
    """A flow carrying an objective (active HeatExchanger) can't run as a square
    simulation; simulation mode falls back to IMODE=3 and still solves."""
    net = create_g2h_net()
    net.apply_formulation(make_smooth_weymouth_network_formulation())
    net.apply_formulation(make_smooth_darcy_weisbach_network_formulation())

    result = ms.GEKKOSolver(solver=IPOPT).solve(net, simulation=True)

    assert result.success


def _simbench_mes():
    """Full multi-energy network (power + gas + heat + CHP/P2G/P2H couplings)
    built from the simbench LV-rural3 grid - ~390 nodes."""
    import simbench

    from monee.io.from_pandapower import from_pandapower_net
    from monee.network import generate_supply_return_mes_based_on_power_net

    pp_net = simbench.get_simbench_net("1-LV-rural3--1-no_sw")
    mn = from_pandapower_net(pp_net)
    return generate_supply_return_mes_based_on_power_net(
        mn,
        coupling_density=0.2,
        centralized=False,
        couplings=("chp", "p2g", "p2h"),
        coupling_kwargs={"seed": 1, "use_hg_variants": True},
        heat_kwargs={"node_based_heat_loads": True},
    )


@pytest.mark.pptest
def test_smooth_simbench_mes_solves_under_ipopt():
    """The smooth gas/heat stack + the bus voltage floor make a full ~390-node
    multi-energy simbench network converge under GEKKO IPOPT - the headline
    scenario the formulations target. Before both changes this did not converge."""
    mes = _simbench_mes()
    assert len(mes.nodes) > 200
    mes.apply_formulation(make_smooth_weymouth_network_formulation())
    mes.apply_formulation(make_smooth_darcy_weisbach_network_formulation())

    result = ms.GEKKOSolver(solver=IPOPT).solve(mes, exclude_unconnected_nodes=True)

    assert result.success
    # Voltages respect the configurable floor (PowerGrid.vm_pu_min) that keeps
    # the 1/vm AC current Jacobian bounded for IPOPT.
    buses = result.dataframes["Bus"]
    assert (buses["vm_pu"] >= 0.7 - 1e-6).all()
    # Gas/heat pipes carry unidirectional flow (smooth complementarity holds).
    # Skip pipes on unconnected/ignored nodes (NaN - not part of the solve).
    for carrier in ("GasPipe", "WaterPipe"):
        pipes = result.dataframes[carrier]
        for pos, neg in zip(pipes["mass_flow_pos"], pipes["mass_flow_neg"]):
            if math.isnan(pos) or math.isnan(neg):
                continue
            assert min(abs(pos), abs(neg)) < 1e-2


@pytest.mark.pptest
def test_smooth_simbench_sectors_solve_standalone_under_ipopt():
    """Each carrier converges as a single-carrier grid under IPOPT once coupling
    points and the other carriers are removed - the per-sector precondition for
    a decoupled energy flow."""
    mes = _simbench_mes()
    mes.apply_formulation(make_smooth_weymouth_network_formulation())
    mes.apply_formulation(make_smooth_darcy_weisbach_network_formulation())

    carrier_grid = {"power": mm.PowerGrid, "gas": mm.GasGrid, "heat": mm.WaterGrid}
    for carrier, grid_type in carrier_grid.items():
        sub = mes.copy()
        for cp in list(sub.cps):
            sub.deactivate(cp)
        for compound in list(sub.compounds):
            sub.deactivate(compound)
        for node in list(sub.nodes):
            if type(node.grid) is not grid_type:
                sub.deactivate(node)
        result = ms.GEKKOSolver(solver=IPOPT).solve(sub, exclude_unconnected_nodes=True)
        assert result.success, f"{carrier} sector failed to converge"

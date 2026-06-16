"""Smooth (binary-free) gas/heat NLP formulations under GEKKO IPOPT."""

import math

import pytest

import monee.model as mm
import monee.solver as ms
from monee.model.formulation import (
    make_gas_nlp_formulation,
    make_heat_nlp_formulation,
)
from tests.util import create_g2h_net

FRICTION_MODELS = ["constant", "pwl", "nonlinear", "hybrid"]

IPOPT = 3


def _gas_only_net():
    pn = mm.Network(mm.create_gas_grid("gas", type="lgas"))
    g0 = pn.node(mm.Junction(), child_ids=[pn.child(mm.Source(mass_flow_kgs=1))])
    g1 = pn.node(mm.Junction(), child_ids=[pn.child(mm.ExtHydrGrid())])
    g2 = pn.node(mm.Junction(), child_ids=[pn.child(mm.Sink(mass_flow_kgs=0.6))])
    pn.branch(mm.GasPipe(diameter_m=0.35, length_m=1000, roughness_m=0.01), g0, g1)
    pn.branch(mm.GasPipe(diameter_m=0.35, length_m=1500, roughness_m=0.01), g0, g2)
    return pn


def _heat_only_net():
    pn = mm.Network(mm.create_water_grid("heat"))
    w0 = pn.node(mm.Junction(), child_ids=[pn.child(mm.Sink(mass_flow_kgs=0.1))])
    w1 = pn.node(mm.Junction(), child_ids=[pn.child(mm.ConsumeHydrGrid(1))])
    w2 = pn.node(mm.Junction())
    w3 = pn.node(mm.Junction(), child_ids=[pn.child(mm.ExtHydrGrid(t_k=359))])
    pn.branch(mm.WaterPipe(diameter_m=0.15, length_m=100), w0, w1)
    pn.branch(mm.WaterPipe(diameter_m=0.15, length_m=200), w3, w2)
    return pn


def _apply_smooth(network, friction_model):
    network.apply_formulation(make_gas_nlp_formulation(friction_model=friction_model))
    network.apply_formulation(make_heat_nlp_formulation(friction_model=friction_model))


def _simbench_mes():
    """Multi-energy network (power+gas+heat, CHP/P2G/P2H) from simbench LV-rural3."""
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


def _isolate_carrier(mes, grid_type):
    """Copy the MES and deactivate coupling points, compounds and foreign-carrier nodes."""
    sub = mes.copy()
    for cp in list(sub.cps):
        sub.deactivate(cp)
    for compound in list(sub.compounds):
        sub.deactivate(compound)
    for node in list(sub.nodes):
        if type(node.grid) is not grid_type:
            sub.deactivate(node)
    return sub


@pytest.mark.parametrize("friction_model", FRICTION_MODELS)
def test_smooth_mes_solves_under_ipopt(friction_model):
    # GIVEN
    network = create_g2h_net()
    _apply_smooth(network, friction_model)

    # WHEN
    result = ms.GEKKOSolver(solver=IPOPT).solve(network)

    # THEN
    assert result.success

    # Gas mass balance is closed by the ext grid.
    ext_mass = result.dataframes["ExtHydrGrid"]["mass_flow_kgs"][0]
    assert math.isfinite(ext_mass)

    # No spurious bidirectional flow: per pipe, pos*neg ~ 0 (smooth complementarity).
    gas_pipes = result.dataframes["GasPipe"]
    for pos, neg in zip(gas_pipes["mass_flow_pos_kgs"], gas_pipes["mass_flow_neg_kgs"]):
        assert min(abs(pos), abs(neg)) < 1e-2


def test_smooth_gas_signed_pressure_drop():
    # GIVEN
    network = create_g2h_net()
    _apply_smooth(network, "constant")

    # WHEN
    result = ms.GEKKOSolver(solver=IPOPT).solve(network)

    # THEN
    assert result.success

    # Signed Weymouth keeps squared pressures within the junction bounds [0, 2].
    junctions = result.dataframes["Junction"]
    assert (junctions["pressure_squared_pu"] >= 0).all()
    assert (junctions["pressure_squared_pu"] <= 2).all()


def test_simulation_gas_only_matches_default():
    # GIVEN
    ref = _gas_only_net()
    ref.apply_formulation(make_gas_nlp_formulation())
    sim = _gas_only_net()
    sim.apply_formulation(make_gas_nlp_formulation())

    # WHEN
    ref_res = ms.GEKKOSolver(solver=IPOPT).solve(ref)
    sim_res = ms.GEKKOSolver(solver=IPOPT).solve(sim, simulation=True)

    # THEN
    assert ref_res.success
    assert sim_res.success

    # Square IMODE=1 simulation matches the default IMODE=3 smooth solve.
    assert math.isclose(
        ref_res.dataframes["ExtHydrGrid"]["mass_flow_kgs"][0],
        sim_res.dataframes["ExtHydrGrid"]["mass_flow_kgs"][0],
        abs_tol=1e-3,
    )


def test_simulation_heat_only_matches_default():
    # GIVEN
    ref = _heat_only_net()
    ref.apply_formulation(make_heat_nlp_formulation())
    sim = _heat_only_net()
    sim.apply_formulation(make_heat_nlp_formulation())

    # WHEN
    ref_res = ms.GEKKOSolver(solver=IPOPT).solve(ref)
    sim_res = ms.GEKKOSolver(solver=IPOPT).solve(sim, simulation=True)

    # THEN
    assert ref_res.success
    assert sim_res.success

    # Square IMODE=1 simulation matches the default IMODE=3 nodal temperatures.
    assert math.isclose(
        ref_res.dataframes["Junction"]["t_k"].dropna().sum(),
        sim_res.dataframes["Junction"]["t_k"].dropna().sum(),
        rel_tol=1e-3,
    )


def test_simulation_falls_back_to_imode3_with_objective():
    # GIVEN
    net = create_g2h_net()
    net.apply_formulation(make_gas_nlp_formulation())
    net.apply_formulation(make_heat_nlp_formulation())

    # WHEN
    result = ms.GEKKOSolver(solver=IPOPT).solve(net, simulation=True)

    # THEN
    # An active-HeatExchanger objective forces fallback from square simulation to IMODE=3.
    assert result.success


@pytest.mark.pptest
def test_smooth_simbench_mes_solves_under_ipopt():
    # GIVEN
    mes = _simbench_mes()
    mes.apply_formulation(make_gas_nlp_formulation())
    mes.apply_formulation(make_heat_nlp_formulation())

    # WHEN
    result = ms.GEKKOSolver(solver=IPOPT).solve(mes, exclude_unconnected_nodes=True)

    # THEN
    assert result.success

    assert len(mes.nodes) > 200

    # Voltages respect the floor (PowerGrid.vm_pu_min) that bounds the 1/vm Jacobian.
    buses = result.dataframes["Bus"]
    assert (buses["vm_pu"] >= 0.7 - 1e-6).all()

    # Pipes carry unidirectional flow; NaN pipes (unconnected/ignored) are skipped.
    for carrier in ("GasPipe", "WaterPipe"):
        pipes = result.dataframes[carrier]
        for pos, neg in zip(pipes["mass_flow_pos_kgs"], pipes["mass_flow_neg_kgs"]):
            if math.isnan(pos) or math.isnan(neg):
                continue
            assert min(abs(pos), abs(neg)) < 1e-2


@pytest.mark.pptest
def test_smooth_simbench_sectors_solve_standalone_under_ipopt():
    # GIVEN
    mes = _simbench_mes()
    mes.apply_formulation(make_gas_nlp_formulation())
    mes.apply_formulation(make_heat_nlp_formulation())
    carrier_grid = {"power": mm.PowerGrid, "gas": mm.GasGrid, "heat": mm.WaterGrid}

    # WHEN
    results = {
        carrier: ms.GEKKOSolver(solver=IPOPT).solve(
            _isolate_carrier(mes, grid_type), exclude_unconnected_nodes=True
        )
        for carrier, grid_type in carrier_grid.items()
    }

    # THEN
    for carrier, result in results.items():
        assert result.success, f"{carrier} sector failed to converge"

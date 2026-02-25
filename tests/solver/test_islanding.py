import math

import pytest

from monee import PyomoSolver, enable_islanding, mm, mx, run_energy_flow
from monee.model.formulation import MISOCP_NETWORK_FORMULATION
from monee.model.islanding import GridFormingGenerator
from monee.network.mes import create_monee_benchmark_net


def _build_two_island_network():
    """Return a 3-bus network with two disconnected electricity islands."""
    net = mm.Network()

    bus_0 = mx.create_bus(net)  # island A — reference
    bus_1 = mx.create_bus(net)  # island A — load
    bus_2 = mx.create_bus(net)  # island B — isolated

    mx.create_ext_power_grid(net, bus_0)
    mx.create_power_load(net, bus_1, p_mw=0.05, q_mvar=0)

    # Island B: grid-forming generator absorbs its own load
    net.child_to(GridFormingGenerator(p_mw_max=1.0, q_mvar_max=0.5), bus_2)
    mx.create_power_load(net, bus_2, p_mw=0.08, q_mvar=0)

    # Only line within island A; bus 2 is completely disconnected.
    mx.create_line(net, bus_0, bus_1, length_m=100, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)

    return net


def test_islanding_el_converges():
    """With islanding enabled, both islands must solve without error."""
    net = _build_two_island_network()
    enable_islanding(net, electricity=True)

    result = run_energy_flow(net)

    assert result is not None


def test_islanding_el_gf_generator_supplies_island():
    """The GridFormingGenerator on bus 2 should supply ≈ 0.08 MW to its island."""
    net = _build_two_island_network()
    enable_islanding(net, electricity=True)

    result = run_energy_flow(net)
    print(result)
    gf_df = result.dataframes.get("GridFormingGenerator")
    assert gf_df is not None, "GridFormingGenerator not in result dataframes"

    # The generator must absorb the island B load (negative sign convention for generation)
    gf_p_mw = gf_df["p_mw"].iloc[0]
    assert abs(gf_p_mw) == pytest.approx(0.08, abs=1e-3), (
        f"Expected GF generator p_mw ≈ -0.08 MW, got {gf_p_mw}"
    )


def test_islanding_disabled_bus2_ignored():
    """Without islanding, bus 2 (no ExtPowerGrid in its component) is pre-filtered."""
    net = _build_two_island_network()
    # No enable_islanding → legacy behavior

    result = run_energy_flow(net)

    # Bus 2 is disconnected from any ExtPowerGrid, so it should be ignored.
    # The solver should still converge (bus 2 simply not in the solve).
    assert result is not None


def test_islanding_monee_benchmark():
    net_multi: mm.Network = create_monee_benchmark_net()
    net_multi.apply_formulation(MISOCP_NETWORK_FORMULATION)

    branch_tbd = net_multi.get_branch_between(2, 3)
    net_multi.deactivate(branch_tbd)
    enable_islanding(net_multi, electricity=True)

    result = run_energy_flow(net_multi, solver=PyomoSolver())
    print(result)

    assert result.dataframes["Bus"]["vm_pu"][3] == pytest.approx(0.999981)

    net_multi: mm.Network = create_monee_benchmark_net()
    net_multi.apply_formulation(MISOCP_NETWORK_FORMULATION)

    branch_tbd = net_multi.get_branch_between(2, 3)
    net_multi.deactivate(branch_tbd)

    result = run_energy_flow(net_multi, solver=PyomoSolver())
    print(result)

    assert math.isnan(result.dataframes["Bus"]["vm_pu_squared"][3])

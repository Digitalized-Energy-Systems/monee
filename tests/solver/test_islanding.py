import math

import pytest

from monee import (
    EL_MISOCP_FORMULATION,
    GridFormingGenerator,
    PyomoSolver,
    enable_islanding,
    mm,
    mx,
    run_energy_flow,
)
from monee.network.mes import create_monee_benchmark_net


def _build_two_island_network():
    """Return a 3-bus network with two disconnected electricity islands."""
    net = mm.Network()

    bus_0 = mx.create_bus(net)
    bus_1 = mx.create_bus(net)
    bus_2 = mx.create_bus(net)

    mx.create_ext_power_grid(net, bus_0)
    mx.create_power_load(net, bus_1, p_mw=0.05, q_mvar=0)

    net.child_to(GridFormingGenerator(p_mw_max=1.0, q_mvar_max=0.5), bus_2)
    mx.create_power_load(net, bus_2, p_mw=0.08, q_mvar=0)

    mx.create_line(net, bus_0, bus_1, length_m=100, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)

    return net


def test_islanding_el_converges():
    # GIVEN
    net = _build_two_island_network()
    enable_islanding(net, electricity=True)

    # WHEN
    result = run_energy_flow(net)

    # THEN
    assert result.success
    assert result is not None


def test_islanding_el_gf_generator_supplies_island():
    # GIVEN
    net = _build_two_island_network()
    enable_islanding(net, electricity=True)

    # WHEN
    result = run_energy_flow(net)

    # THEN
    assert result.success

    gf_df = result.dataframes.get("GridFormingGenerator")
    assert gf_df is not None, "GridFormingGenerator not in result dataframes"

    # Generator absorbs the island B load (negative sign convention for generation)
    gf_p_mw = gf_df["p_mw"].iloc[0]
    assert abs(gf_p_mw) == pytest.approx(0.08, abs=1e-3), (
        f"Expected GF generator p_mw ≈ -0.08 MW, got {gf_p_mw}"
    )


def test_islanding_disabled_bus2_ignored():
    # GIVEN
    net = _build_two_island_network()

    # WHEN
    result = run_energy_flow(net)

    # THEN
    # Bus 2 has no ExtPowerGrid in its component, so it is pre-filtered
    assert result.success
    assert result is not None


def test_islanding_monee_benchmark():
    # GIVEN
    net_islanding: mm.Network = create_monee_benchmark_net()
    net_islanding.apply_formulation(EL_MISOCP_FORMULATION)
    net_islanding.deactivate(net_islanding.get_branch_between(2, 3))
    enable_islanding(net_islanding, electricity=True)

    net_legacy: mm.Network = create_monee_benchmark_net()
    net_legacy.apply_formulation(EL_MISOCP_FORMULATION)
    net_legacy.deactivate(net_legacy.get_branch_between(2, 3))

    # WHEN
    result_islanding = run_energy_flow(net_islanding, solver=PyomoSolver())
    result_legacy = run_energy_flow(net_legacy, solver=PyomoSolver())

    # THEN
    assert result_islanding.success
    assert result_legacy.success

    assert result_islanding.dataframes["Bus"]["vm_pu"][3] == pytest.approx(0.999981)

    assert math.isnan(result_legacy.dataframes["Bus"]["vm_pu_squared"][3])

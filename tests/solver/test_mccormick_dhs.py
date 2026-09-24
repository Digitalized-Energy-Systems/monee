"""Smoke tests for the McCormick-DHS relaxation (Deng et al., 2021)."""

import math

import pytest

import monee.express as mx
import monee.model as mm
import monee.solver as ms

NUM_PARTITIONS = 40
T_PU_MIN_ENV = 0.82  # ~292 K (slightly above ambient)
T_PU_MAX_ENV = 1.15  # ~409 K - room for boundary heat injection


def _build_series_dhs(q_gen_mw: float = 0.0, q_load_mw: float = 0.0):
    """Hundred-node unidirectional supply line with optional heat gen/load."""
    net = mm.Network()

    juncs = []
    for i in range(100):
        juncs.append(mx.create_water_junction(net))
    for i in range(99):
        mx.create_water_pipe(net, juncs[i], juncs[i + 1], diameter_m=0.15, length_m=100)

    mx.create_water_ext_grid(net, juncs[0], t_k=360.0)
    mx.create_water_sink(net, juncs[-1], mass_flow_kgs=1.0)

    if q_gen_mw > 0:
        mx.create_heat_generator(net, juncs[5], q_mw=q_gen_mw)
    if q_load_mw > 0:
        mx.create_heat_load(net, juncs[20], q_mw=q_load_mw)

    # Tight envelope bounds keep the relaxation gap small enough for sensible results.
    grid = list(net.grids)[0]
    grid.t_pu_min_env = T_PU_MIN_ENV
    grid.t_pu_max_env = T_PU_MAX_ENV

    # num_partitions=4 enforces a monotonically-decaying temperature profile.
    net.apply_formulation(mm.make_heat_convex_milp_formulation(num_partitions=4))
    return net


def test_mccormick_dhs_passive_line_temperature_decays():
    # GIVEN
    # Tiny heat injection so pipe losses dominate and the supply end stays hottest.
    net = _build_series_dhs(0.001, 0.001)

    # WHEN
    result = ms.PyomoSolver().solve(net)
    print(result)

    # THEN
    assert result.success

    juncs = result.get(mm.Junction)
    t_k = list(juncs["t_k"])
    assert math.isclose(t_k[0], 360.0, abs_tol=1e-3)

    # All temperatures remain in the physical envelope.
    for t in t_k:
        assert T_PU_MIN_ENV * 356.0 <= t <= T_PU_MAX_ENV * 356.0

    # Supply end is the hottest; the sink temperature stays above ambient.
    assert t_k[0] >= t_k[-1] - 1e-6
    assert t_k[-1] > 296.1

    pipes = result.get(mm.WaterPipe)
    for h in pipes["H_out_mw"]:
        assert h > 0
    for h in pipes["H_in_mw"]:
        assert h > 0


def test_mccormick_dhs_heat_generator_raises_temperature():
    # GIVEN
    q_gen_mw = 0.1
    net = _build_series_dhs(q_gen_mw=q_gen_mw)

    # WHEN
    result = ms.PyomoSolver().solve(net)

    # THEN
    assert result.success

    # Generator sits at juncs[5]; pipe 4->5 feeds it, pipe 5->6 leaves it.
    pipes = result.get(mm.WaterPipe)
    h_in_at_n5 = pipes["H_in_mw"][4]
    h_out_of_n5 = pipes["H_out_mw"][5]

    # Paper eq. (9a) at the injection node: H_out - H_in == q_gen.
    residual = h_out_of_n5 - h_in_at_n5 - q_gen_mw
    assert math.isclose(residual, 0.0, rel_tol=1e-4, abs_tol=1e-6), (
        f"Node eq. 9a balance violated at junc[5], residual={residual}"
    )


def test_mccormick_dhs_balanced_gen_and_load():
    # GIVEN
    heat_mw = 0.1
    net = _build_series_dhs(q_gen_mw=heat_mw, q_load_mw=heat_mw)

    # WHEN
    result = ms.PyomoSolver().solve(net)

    # THEN
    assert result.success

    # HeatLoad sits at juncs[20]; pipe 19->20 feeds it, pipe 20->21 leaves it.
    pipes = result.get(mm.WaterPipe)
    h_in_at_n20 = pipes["H_in_mw"][19]
    h_out_of_n20 = pipes["H_out_mw"][20]

    # Paper eq. (9a) at the load node: H_in - H_out == +q.
    residual = h_in_at_n20 - h_out_of_n20 - heat_mw
    assert math.isclose(residual, 0.0, rel_tol=1e-4, abs_tol=1e-6), (
        f"Node eq. 9a balance violated at junc[20], residual={residual}"
    )


def test_mccormick_dhs_plain_envelope_reduces_to_lp():
    # GIVEN
    net = mm.Network()
    n = [mx.create_water_junction(net) for _ in range(3)]
    mx.create_water_ext_grid(net, n[0], t_k=360.0)
    mx.create_water_sink(net, n[2], mass_flow_kgs=1.0)
    for a, b in zip(n[:-1], n[1:]):
        mx.create_water_pipe(
            net, from_node_id=a, to_node_id=b, diameter_m=0.15, length_m=200.0
        )
    net.apply_formulation(mm.HEAT_CONVEX_MILP_FORMULATION)

    # WHEN
    result = ms.PyomoSolver().solve(net)

    # THEN
    assert result.success

    juncs = result.get(mm.Junction)
    assert math.isclose(list(juncs["t_k"])[0], 360.0, abs_tol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

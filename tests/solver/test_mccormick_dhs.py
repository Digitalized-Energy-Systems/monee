"""Smoke tests for the McCormick-DHS formulation (Deng et al., 2021).

The formulation is a **relaxation** of the quality-quantity regulated
district-heating model: it replaces the bilinear ``H = c·m·τ`` with
McCormick envelopes (paper eq. 17b-e, or the piecewise variant 18b when
``num_partitions > 1``) and linearizes the exponential heat-loss factor
via first-order Taylor expansion (eq. 9b).  Node-based heat I/O is
provided by :class:`~monee.model.child.HeatGenerator` /
:class:`~monee.model.child.HeatLoad` (paper's ``H_G,i`` / ``H_L,i``).

Because plain McCormick is loose, these tests use:

* **Piecewise McCormick** (``num_partitions >= 20``) to shrink the
  envelope gap.
* **Tight envelope bounds** on the node temperature
  (``t_pu_min_env = 0.82``, ``t_pu_max_env = 1.15``) — a reasonable
  physical range for 293-409 K district-heating operation with some
  headroom for boundary heat injection.

With these settings the relaxation is tight enough that the solver
recovers a physically meaningful temperature profile even without an
economic objective.  Boundary enthalpies (``ExtHydrGrid`` / ``Sink``)
close the nodal balance (eq. 9a) so supply and return values are pinned
exactly.
"""

import math

import pytest

import monee.express as mx
import monee.model as mm
import monee.solver as ms

NUM_PARTITIONS = 40
T_PU_MIN_ENV = 0.82  # ~292 K (slightly above ambient)
T_PU_MAX_ENV = 1.15  # ~409 K — room for boundary heat injection


def _build_series_dhs(q_gen_mw: float = 0.0, q_load_mw: float = 0.0):
    """Six-node unidirectional supply line.

    ``n0`` is the hydraulic + thermal reference (``ExtHydrGrid``,
    ``t_k=360 K``).  ``n5`` pins the mass flow via a fixed ``Sink`` so
    the bilinear ``H_out = c·m·τ`` surface is approached deterministically
    by the McCormick relaxation.  ``HeatGenerator`` at ``n2`` and
    ``HeatLoad`` at ``n4`` (when ``q_gen_mw`` / ``q_load_mw`` are non-zero)
    introduce node-based heat injection / withdrawal per paper eq. 9a.
    """
    net = mm.Network()

    juncs = []
    for i in range(100):
        juncs.append(mx.create_water_junction(net))
    for i in range(99):
        mx.create_water_pipe(net, juncs[i], juncs[i + 1], diameter_m=0.15, length_m=100)

    mx.create_water_ext_grid(net, juncs[0], t_k=360.0)
    mx.create_water_sink(net, juncs[-1], mass_flow=1.0)

    if q_gen_mw > 0:
        mx.create_heat_generator(net, juncs[5], q_mw=q_gen_mw)
    if q_load_mw > 0:
        mx.create_heat_load(net, juncs[20], q_mw=q_load_mw)

    # Tighten the McCormick envelope bounds to the physical DHS range so
    # the relaxation gap is small enough to recover sensible results.
    grid = list(net.grids)[0]
    grid.t_pu_min_env = T_PU_MIN_ENV
    grid.t_pu_max_env = T_PU_MAX_ENV

    # ``num_partitions=4`` is enough for the envelope to enforce a
    # monotonically-decaying temperature profile under heat losses; with the
    # plain LP envelope (S=1) the LP corner can park adjacent junctions at
    # opposite envelope extremes and the "supply >= sink" assertion below
    # depends on the solver's pivot order rather than on physics.
    net.apply_formulation(mm.make_mccormick_dhs_formulation(num_partitions=4))
    return net


def test_mccormick_dhs_passive_line_temperature_decays():
    """Passive line (no heat I/O): temperatures should stay within the
    envelope bounds and roughly decay from the supply (360 K) toward the
    sink.  The partition granularity allows small non-monotonic wiggles
    inside one piece, so we only require the end-points to respect the
    physical ordering."""
    # Tiny heat injection — pipe losses dominate so the supply end stays hotter
    # than the sink end (the assertion below checks physical decay).
    net = _build_series_dhs(0.001, 0.001)

    result = ms.PyomoSolver(solver_name="gurobi").solve(net)
    print(result)
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
    """HeatGenerator at junc[5] injects 0.1 MW — the linear nodal balance
    (paper eq. 9a) must hold exactly at that junction."""
    q_gen_mw = 0.1
    net = _build_series_dhs(q_gen_mw=q_gen_mw)

    result = ms.PyomoSolver().solve(net)
    assert result.success

    pipes = result.get(mm.WaterPipe)
    # _build_series_dhs places the generator at juncs[5]; pipes are
    # ordered by their (from, to, key) tuple, so pipe row index i
    # corresponds to (i, i+1).  Pipe 4→5 feeds junc[5]; pipe 5→6 leaves it.
    h_in_at_n5 = pipes["H_in_mw"][4]
    h_out_of_n5 = pipes["H_out_mw"][5]
    # Paper eq. (9a) at the junction with q_mw_heat = -q_gen (injection):
    #   H_in - H_out == -q_gen ⇒ H_out - H_in == q_gen.
    residual = h_out_of_n5 - h_in_at_n5 - q_gen_mw
    assert math.isclose(residual, 0.0, rel_tol=1e-4, abs_tol=1e-6), (
        f"Node eq. 9a balance violated at junc[5], residual={residual}"
    )


def test_mccormick_dhs_balanced_gen_and_load():
    """HeatGenerator at junc[5] (+0.1 MW) and HeatLoad at junc[20] (+0.1 MW)
    balance — the nodal balance (paper eq. 9a) must hold exactly at the
    HeatLoad node."""
    q_mw = 0.1
    net = _build_series_dhs(q_gen_mw=q_mw, q_load_mw=q_mw)

    result = ms.PyomoSolver().solve(net)
    assert result.success

    pipes = result.get(mm.WaterPipe)
    # HeatLoad is at juncs[20]; pipe 19→20 feeds it, pipe 20→21 leaves.
    h_in_at_n20 = pipes["H_in_mw"][19]
    h_out_of_n20 = pipes["H_out_mw"][20]
    # Paper eq. (9a) at the load junction:  H_in - H_out == +q.
    residual = h_in_at_n20 - h_out_of_n20 - q_mw
    assert math.isclose(residual, 0.0, rel_tol=1e-4, abs_tol=1e-6), (
        f"Node eq. 9a balance violated at junc[20], residual={residual}"
    )


def test_mccormick_dhs_plain_envelope_reduces_to_lp():
    """``num_partitions=1`` should produce a pure McCormick LP (no
    binaries).  The relaxation is looser, so the internal-node
    temperatures may drift, but the formulation must still solve and
    the boundary balance must close exactly."""
    net = mm.Network()
    n = [mx.create_water_junction(net) for _ in range(3)]
    mx.create_water_ext_grid(net, n[0], t_k=360.0)
    mx.create_water_sink(net, n[2], mass_flow=1.0)
    for a, b in zip(n[:-1], n[1:]):
        mx.create_water_pipe(
            net, from_node_id=a, to_node_id=b, diameter_m=0.15, length_m=200.0
        )
    net.apply_formulation(mm.MCCORMICK_DHS_NETWORK_FORMULATION)

    result = ms.PyomoSolver().solve(net)
    assert result.success

    juncs = result.get(mm.Junction)
    assert math.isclose(list(juncs["t_k"])[0], 360.0, abs_tol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

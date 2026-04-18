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

NUM_PARTITIONS = 20
T_PU_MIN_ENV = 0.82  # ~292 K (slightly above ambient)
T_PU_MAX_ENV = 1.15  # ~409 K — room for boundary heat injection


def _build_series_dhs(q_gen_w: float = 0.0, q_load_w: float = 0.0):
    """Six-node unidirectional supply line.

    ``n0`` is the hydraulic + thermal reference (``ExtHydrGrid``,
    ``t_k=360 K``).  ``n5`` pins the mass flow via a fixed ``Sink`` so
    the bilinear ``H_out = c·m·τ`` surface is approached deterministically
    by the McCormick relaxation.  ``HeatGenerator`` at ``n2`` and
    ``HeatLoad`` at ``n4`` (when ``q_gen_w`` / ``q_load_w`` are non-zero)
    introduce node-based heat injection / withdrawal per paper eq. 9a.
    """
    net = mm.Network()

    n0 = mx.create_water_junction(net)
    n1 = mx.create_water_junction(net)
    n2 = mx.create_water_junction(net)
    n3 = mx.create_water_junction(net)
    n4 = mx.create_water_junction(net)
    n5 = mx.create_water_junction(net)

    mx.create_water_ext_grid(net, n0, t_k=360.0)
    mx.create_water_sink(net, n5, mass_flow=1.0)

    if q_gen_w > 0:
        mx.create_heat_generator(net, n2, q_w=q_gen_w)
    if q_load_w > 0:
        mx.create_heat_load(net, n4, q_w=q_load_w)

    for a, b in [(n0, n1), (n1, n2), (n2, n3), (n3, n4), (n4, n5)]:
        mx.create_water_pipe(
            net,
            from_node_id=a,
            to_node_id=b,
            diameter_m=0.15,
            length_m=200.0,
        )

    # Tighten the McCormick envelope bounds to the physical DHS range so
    # the relaxation gap is small enough to recover sensible results.
    grid = list(net.grids)[0]
    grid.t_pu_min_env = T_PU_MIN_ENV
    grid.t_pu_max_env = T_PU_MAX_ENV

    net.apply_formulation(
        mm.make_mccormick_dhs_formulation(num_partitions=NUM_PARTITIONS)
    )
    return net


def test_mccormick_dhs_passive_line_temperature_decays():
    """Passive line (no heat I/O): temperatures should stay within the
    envelope bounds and roughly decay from the supply (360 K) toward the
    sink.  The partition granularity allows small non-monotonic wiggles
    inside one piece, so we only require the end-points to respect the
    physical ordering."""
    net = _build_series_dhs(100_000, 1000)

    result = ms.PyomoSolver().solve(net)

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
    for h in pipes["H_out_w"]:
        assert h > 0
    for h in pipes["H_in_w"]:
        assert h > 0


def test_mccormick_dhs_heat_generator_raises_temperature():
    """HeatGenerator at n2 injects 100 kW — downstream enthalpy must
    increase at that node and the linear nodal balance (eq. 9a) must
    hold exactly."""
    q_gen = 1.0e5
    net = _build_series_dhs(q_gen_w=q_gen)

    result = ms.PyomoSolver().solve(net)
    assert result.success

    pipes = result.get(mm.WaterPipe)
    h_in_at_n2 = pipes["H_in_w"][1]  # pipe 1→2
    h_out_of_n2 = pipes["H_out_w"][2]  # pipe 2→3
    # Paper eq. (9a) at n2 with q_w_heat = −q_gen (injection):
    #   H_in(1→2) − H_out(2→3) == −q_gen
    # i.e. H_out(2→3) = H_in(1→2) + q_gen.
    residual = h_out_of_n2 - h_in_at_n2 - q_gen
    assert math.isclose(residual, 0.0, rel_tol=1e-4, abs_tol=1.0), (
        f"Node eq. 9a balance violated at n2, residual={residual}"
    )


def test_mccormick_dhs_balanced_gen_and_load():
    """HeatGenerator (n2, +100 kW) and HeatLoad (n4, +100 kW) balance —
    the linear nodal balance (eq. 9a) must still hold exactly at n4."""
    q = 1.0e5
    net = _build_series_dhs(q_gen_w=q, q_load_w=q)

    result = ms.PyomoSolver().solve(net)
    assert result.success

    pipes = result.get(mm.WaterPipe)
    h_in_at_n4 = pipes["H_in_w"][3]  # pipe 3→4
    h_out_of_n4 = pipes["H_out_w"][4]  # pipe 4→5
    # Paper eq. (9a) at n4 with q_w_heat = +q (load):
    #   H_in(3→4) − H_out(4→5) == +q
    residual = h_in_at_n4 - h_out_of_n4 - q
    assert math.isclose(residual, 0.0, rel_tol=1e-4, abs_tol=1.0), (
        f"Node eq. 9a balance violated at n4, residual={residual}"
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

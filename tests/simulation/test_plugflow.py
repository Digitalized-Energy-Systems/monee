"""Tests for the PlugFlowTransport (node method) network extension."""

import math

import numpy as np
import pytest

import monee.express as mx
import monee.model as mm
from monee import run_energy_flow
from monee.model import LumpedThermalCapacitance, PlugFlowTransport
from monee.model.phys.nonlinear.hf import SPECIFIC_HEAT_CAP_WATER
from monee.model.phys.nonlinear.wf import pipe_insulation_ua
from monee.simulation.timeseries import TimeseriesData, run

pytest.importorskip("casadi")

D_M, L_M, FLOW_KGS, DT_H = 0.3, 1000.0, 10.0, 0.25


def _line(extension=None, *far_childs):
    """Supply (ext grid, series temperature) -> 1 km pipe -> 10 kg/s sink."""
    net = mm.Network()
    supply = net.child(mm.ExtHydrGrid(t_k=356))
    n0 = net.node(mm.Junction(), mm.WATER, child_ids=[supply])
    far = [net.child(c) for c in (mm.Sink(mass_flow_kgs=FLOW_KGS), *far_childs)]
    n1 = net.node(mm.Junction(), mm.WATER, child_ids=far)
    pipe = net.branch(mm.WaterPipe(diameter_m=D_M, length_m=L_M), n0, n1)
    if extension is not None:
        net.add_extension(extension)
    return net, supply, n1, pipe


def _outlet_k(net, pipe, t_in_k):
    """Steady outlet temperature of the pipe for an inlet temperature."""
    t_ext = net.branch_by_id(pipe).model.temperature_ext_k
    ua_c = pipe_insulation_ua(net.branch_by_id(pipe).model) / SPECIFIC_HEAT_CAP_WATER
    return t_ext + (t_in_k - t_ext) / (1 + ua_c / FLOW_KGS)


def _run(extension, t_supply):
    net, supply, n1, pipe = _line(extension)
    td = TimeseriesData()
    td.add_child_series(supply, "t_k", t_supply)
    ts = run(net, td, steps=len(t_supply), dt_h=DT_H, solver="ipopt")
    assert not ts.failed_steps
    return net, pipe, ts.get_result_for_id(n1, "t_pu") * net.node_by_id(n1).grid.t_ref_k


def test_plug_flow_transparent_in_single_step():
    plain, _, n1, _ = _line()
    plug, _, _, _ = _line(PlugFlowTransport())

    t_plain = run_energy_flow(plain).get(mm.Junction)["t_pu"]
    t_plug = run_energy_flow(plug).get(mm.Junction)["t_pu"]

    assert abs(t_plain[n1] - t_plug[n1]) < 1e-9


def test_plug_flow_keeps_the_steady_state_under_constant_inputs():
    net, pipe, t_out = _run(PlugFlowTransport(), [356.0] * 6)

    assert (t_out - _outlet_k(net, pipe, 356.0)).abs().max() < 1e-4


def test_supply_step_arrives_after_the_transport_delay_without_diffusion():
    # GIVEN the supply drops 16 K at step 2: the front reaches the outlet after
    # the transport delay M / m = 7.6 steps, i.e. 62 % into step 9
    t_supply = [356.0] * 2 + [340.0] * 12

    # WHEN
    net, pipe, t_out = _run(PlugFlowTransport(), t_supply)

    # THEN the old water arrives until step 8, step 9 is the 62/38 mean and
    # the new water arrives from step 10 on, each cooled by its residence time
    old, new = _outlet_k(net, pipe, 356.0), _outlet_k(net, pipe, 340.0)
    rho = net.branch_by_id(pipe).grid.fluid_density_kg_per_m3
    delay_steps = rho * math.pi / 4 * D_M**2 * L_M / FLOW_KGS / (DT_H * 3600)
    arrival = 2 + delay_steps
    assert math.floor(arrival) == 9
    share_old = arrival - math.floor(arrival)
    assert (t_out.iloc[:9] - old).abs().max() < 1e-3
    assert t_out.iloc[9] == pytest.approx(
        share_old * old + (1 - share_old) * new, abs=1e-3
    )
    assert (t_out.iloc[10:] - new).abs().max() < 1e-3


def test_front_keeps_its_shape_across_a_junction():
    # GIVEN a second pipe behind a junction that draws 4 of the 10 kg/s: the
    # front reaches the junction 62 % into step 9 and the far end, 6.35 steps
    # later, 97 % into step 15
    net, supply, n1, pipe_a = _line(PlugFlowTransport())
    net.child_by_id(net.node_by_id(n1).child_ids[0]).model.mass_flow_kgs = 4.0
    n2 = net.node(
        mm.Junction(), mm.WATER, child_ids=[net.child(mm.Sink(mass_flow_kgs=6.0))]
    )
    pipe_b = net.branch(mm.WaterPipe(diameter_m=D_M, length_m=L_M / 2), n1, n2)
    td = TimeseriesData()
    td.add_child_series(supply, "t_k", [356.0] * 2 + [340.0] * 18)

    # WHEN
    ts = run(net, td, steps=20, dt_h=DT_H, solver="ipopt")

    # THEN the far end sees one mixed step, not a front smeared over two
    assert not ts.failed_steps
    t_ref_k = net.node_by_id(n2).grid.t_ref_k
    t_far = ts.get_result_for_id(n2, "t_pu") * t_ref_k
    model_b = net.branch_by_id(pipe_b).model
    ua_c = pipe_insulation_ua(model_b) / SPECIFIC_HEAT_CAP_WATER
    t_ext = model_b.temperature_ext_k

    def far_k(t_in_k):
        t_mid = _outlet_k(net, pipe_a, t_in_k)
        return t_ext + (t_mid - t_ext) / (1 + ua_c / 6.0)

    rho = net.branch_by_id(pipe_a).grid.fluid_density_kg_per_m3
    mass_a = rho * math.pi / 4 * D_M**2 * L_M
    arrival = 2 + (mass_a / FLOW_KGS + mass_a / 2 / 6.0) / (DT_H * 3600)
    share_old = arrival - math.floor(arrival)
    assert math.floor(arrival) == 15
    assert (t_far.iloc[:15] - far_k(356.0)).abs().max() < 1e-3
    assert t_far.iloc[15] == pytest.approx(
        share_old * far_k(356.0) + (1 - share_old) * far_k(340.0), abs=2e-3
    )
    assert (t_far.iloc[16:] - far_k(340.0)).abs().max() < 1e-3


def test_parcel_count_stays_bounded_and_keeps_the_front():
    # GIVEN 100 parcels of slowly rising entry temperature (a 1-min twin
    # adds one per step, none close enough to merge) with a 16 K front
    from monee.model.extension.plugflow import _MAX_PARCELS, _merge

    seq = [
        (1.0, 1.0 + 1e-3 * k + (0.05 if k >= 60 else 0.0), 60.0 * k) for k in range(100)
    ]

    # WHEN
    merged = _merge(seq, max_kg=1.0, now_t=lambda parcel: parcel[1])

    # THEN the count is capped, mass and heat are kept, the front survives
    assert len(merged) == _MAX_PARCELS
    assert sum(m for m, _, _ in merged) == pytest.approx(100.0)
    assert sum(m * t for m, t, _ in merged) == pytest.approx(
        sum(m * t for m, t, _ in seq)
    )
    jumps = [b[1] - a[1] for a, b in zip(merged, merged[1:])]
    assert max(jumps) > 0.05


def test_lumped_capacitance_smears_the_same_front():
    # the tank the LTC lumps at the outlet starts cooling at once
    t_supply = [356.0] * 2 + [340.0] * 12
    net, pipe, t_out = _run(LumpedThermalCapacitance(), t_supply)

    assert t_out.iloc[3] < _outlet_k(net, pipe, 356.0) - 1.0


def _first_water_outlet_k(net, pipe, exit_age_s, x_kg, steps):
    """Closed-form step means of the outlet temperature while the 356 K water
    of the steady first step drains, ``x_kg`` per step; the water ``c`` kg from
    the outlet leaves at the age ``exit_age_s(c)``."""
    bm = net.branch_by_id(pipe).model
    ua_c = pipe_insulation_ua(bm) / SPECIFIC_HEAT_CAP_WATER
    t_ext = bm.temperature_ext_k
    mass = _pipe_mass(net, pipe)
    c = [np.linspace(k * x_kg, (k + 1) * x_kg, 2001) for k in range(steps)]
    return np.array(
        [
            (t_ext + (356.0 - t_ext) / (1 + ua_c * exit_age_s(ck) / mass)).mean()
            for ck in c
        ]
    )


def _pipe_mass(net, pipe):
    rho = net.branch_by_id(pipe).grid.fluid_density_kg_per_m3
    return rho * math.pi / 4 * D_M**2 * L_M


def test_outlet_ramps_while_the_first_water_drains_after_a_flow_drop():
    # GIVEN 10 kg/s in the steady first step, then 2.5 kg/s: the water c kg
    # from the outlet entered (M - c) / 10 s ago and leaves c / 2.5 s later
    net, _, n1, pipe = _line(PlugFlowTransport())
    sink = net.node_by_id(n1).child_ids[0]
    td = TimeseriesData()
    td.add_child_series(sink, "mass_flow_kgs", [FLOW_KGS] + [FLOW_KGS / 4] * 30)

    # WHEN
    ts = run(net, td, steps=31, dt_h=DT_H, solver="ipopt")

    # THEN the outlet follows the closed form instead of holding one value
    assert not ts.failed_steps
    mass = _pipe_mass(net, pipe)
    exact = _first_water_outlet_k(
        net,
        pipe,
        lambda c: (mass - c) / FLOW_KGS + c / (FLOW_KGS / 4),
        FLOW_KGS / 4 * DT_H * 3600,
        30,
    )
    t_ref_k = net.node_by_id(n1).grid.t_ref_k
    t_out = ts.get_result_for_id(n1, "t_pu").to_numpy()[1:] * t_ref_k
    assert exact[0] - exact[-1] > 1.0
    assert np.abs(t_out - exact).max() < 0.05


def test_plug_flow_follows_a_flow_reversal():
    # GIVEN a 340 K source at the far end that outgrows the sink at step 2: the
    # flow reverses, and the hot water that entered at the supply end c kg
    # from it c / 10 s ago leaves there first, c / 10 s later
    net, supply, n1, pipe = _line(
        PlugFlowTransport(), mm.Source(mass_flow_kgs=1e-3, t_k=340)
    )
    source = net.node_by_id(n1).child_ids[1]
    td = TimeseriesData()
    # the series sets the stored (load-convention) attribute: negative injects
    td.add_child_series(source, "mass_flow_kgs", [-1e-3, -1e-3] + [-2 * FLOW_KGS] * 7)

    # WHEN
    ts = run(net, td, steps=9, dt_h=DT_H, solver="ipopt")

    # THEN
    assert not ts.failed_steps
    flow = ts.get_result_for_id(pipe, "mass_flow_kgs")
    t_out_k = ts.get_result_for_id(pipe, "t_out_pu") * net.node_by_id(n1).grid.t_ref_k
    assert flow.iloc[1] < 0 < flow.iloc[2]  # mass_flow_kgs is positive to -> from
    exact = _first_water_outlet_k(
        net,
        pipe,
        lambda c: c / (FLOW_KGS - 1e-3) + c / FLOW_KGS,
        FLOW_KGS * DT_H * 3600,
        7,
    )
    assert exact[0] - exact[-1] > 0.5  # a steady pipe would deliver ~340 K
    assert np.abs(t_out_k.to_numpy()[2:] - exact).max() < 0.05


def _two_pipes(extension, heat):
    """Supply -> pipe -> junction with ``heat`` -> pipe -> 10 kg/s sink."""
    net = mm.Network()
    supply = net.child(mm.ExtHydrGrid(t_k=356))
    n0 = net.node(mm.Junction(), mm.WATER, child_ids=[supply])
    n1 = net.node(mm.Junction(), mm.WATER, child_ids=[net.child(heat)])
    n2 = net.node(
        mm.Junction(), mm.WATER, child_ids=[net.child(mm.Sink(mass_flow_kgs=FLOW_KGS))]
    )
    for a, b in ((n0, n1), (n1, n2)):
        net.branch(mm.WaterPipe(diameter_m=D_M, length_m=L_M), a, b)
    if extension is not None:
        net.add_extension(extension)
    return net, supply, n2


@pytest.mark.parametrize("heat", [mm.HeatLoad, mm.HeatGenerator])
def test_heat_at_a_single_feed_junction_enters_the_next_pipe(heat):
    # GIVEN 2 MW taken or added at a junction fed by one pipe
    steady, _, n2 = _two_pipes(None, heat(q_mw=2.0))
    t_ref_k = steady.node_by_id(n2).grid.t_ref_k
    t_steady = run_energy_flow(steady).get(mm.Junction)["t_pu"][n2] * t_ref_k
    net, supply, n2 = _two_pipes(PlugFlowTransport(), heat(q_mw=2.0))
    td = TimeseriesData()
    td.add_child_series(supply, "t_k", [356.0] * 20)

    # WHEN
    ts = run(net, td, steps=20, dt_h=DT_H, solver="ipopt")

    # THEN once both pipes are flushed the far end is at the steady temperature
    assert not ts.failed_steps
    t_far = ts.get_result_for_id(n2, "t_pu") * t_ref_k
    assert t_far.iloc[-1] == pytest.approx(t_steady, abs=1e-3)


def test_pinned_plant_temperature_enters_the_supply_pipe_of_a_closed_loop():
    # GIVEN a closed loop whose plant pins 356 K at one node at ~zero net flow
    def loop(extension):
        net = mm.Network(water_model=mm.create_water_grid("heat", t_ref_k=356.0))
        plant, consumer, ret = (mx.create_water_junction(net) for _ in range(3))
        supply = mx.create_water_ext_grid(net, plant, t_k=356.0, pressure_pu=1.0)
        mx.create_water_pipe(net, plant, consumer, diameter_m=0.15, length_m=1000)
        mx.create_heat_exchanger(net, consumer, ret, q_mw=1.0)
        mx.create_water_pipe(net, ret, plant, diameter_m=0.15, length_m=1000)
        if extension is not None:
            net.add_extension(extension)
        return net, supply, consumer

    steady, _, consumer = loop(None)
    t_steady = run_energy_flow(steady).get(mm.Junction, index="id")["t_pu"][consumer]
    net, supply, consumer = loop(PlugFlowTransport())
    td = TimeseriesData()
    td.add_child_series(supply, "t_k", [356.0] * 12)

    # WHEN
    ts = run(net, td, steps=12, dt_h=DT_H, solver="ipopt")

    # THEN the consumer keeps the steady supply temperature
    assert not ts.failed_steps
    t_consumer = ts.get_result_for_id(consumer, "t_pu")
    assert (t_consumer - t_steady).abs().max() < 1e-5


def test_plug_flow_and_lumped_capacitance_are_exclusive():
    net, supply, _, _ = _line(PlugFlowTransport())
    net.add_extension(LumpedThermalCapacitance())
    td = TimeseriesData()
    td.add_child_series(supply, "t_k", [356.0, 356.0])

    with pytest.raises(ValueError, match="attach only one"):
        run(net, td, steps=2, dt_h=DT_H, solver="ipopt")

"""Transformer impedance, fixed tap and line charging against pandapower's runpp."""

import math

import pytest

import monee
import monee.express as mx
import monee.model as mm
from tests.solver.test_gurobipy import requires_gurobi

pytestmark = pytest.mark.pptest

NLP = {}
MISOCP = {"solver": "gurobi", "formulation": "el_misocp", "simulation": False}
EXACT = {"solver": "gurobi", "formulation": "el_nonconvex_miqcqp", "simulation": False}
FORMULATIONS = [
    pytest.param(NLP, id="nlp"),
    pytest.param(MISOCP, id="misocp", marks=requires_gurobi),
    pytest.param(EXACT, id="exact", marks=requires_gurobi),
]


def _pp_trafo_vm(tap_side=None, tap_pos=0):
    pp = pytest.importorskip("pandapower")
    net = pp.create_empty_network()
    hv, lv = pp.create_bus(net, 110), pp.create_bus(net, 20)
    pp.create_ext_grid(net, hv, vm_pu=1.0)
    tap = {}
    if tap_side is not None:
        # The tap of pandapower's "25 MVA 110/20 kV" standard type, on either side.
        tap = dict(
            tap_side=tap_side,
            tap_neutral=0,
            tap_min=-9,
            tap_max=9,
            tap_step_percent=1.5,
            tap_pos=tap_pos,
            tap_changer_type="Ratio",
        )
    pp.create_transformer_from_parameters(
        net, hv, lv, 25, 110, 20, 0.16, 12.0, 0, 0, **tap
    )
    pp.create_load(net, lv, 15.0, 5.0)
    pp.runpp(net)
    return float(net.res_bus.vm_pu[lv])


def _monee_trafo_vm(lv_first, tap=1.0, **solve):
    net = mm.Network(el_model=mm.create_power_grid("power", sn_mva=10))
    hv = mx.create_bus(net, base_kv=110.0)
    lv = mx.create_bus(net, base_kv=20.0)
    mx.create_ext_power_grid(net, hv, vm_pu=1.0)
    ends = (lv, hv) if lv_first else (hv, lv)
    mx.create_trafo(
        net, *ends, vk_percent=12.0, vkr_percent=0.16, sn_trafo_mva=25.0, tap=tap
    )
    mx.create_power_load(net, lv, p_mw=15.0, q_mvar=5.0)
    result = monee.run_energy_flow(net, **solve)
    assert result.success
    return float(result.get(mm.Bus).set_index("id")["vm_pu"][lv])


@pytest.mark.parametrize("lv_first", [True, False], ids=["lv_to_hv", "hv_to_lv"])
def test_trafo_impedance_matches_pandapower_in_both_orientations(lv_first):
    assert _pp_trafo_vm() == pytest.approx(0.97159, abs=1e-5)
    assert _monee_trafo_vm(lv_first) == pytest.approx(_pp_trafo_vm(), abs=1e-5)


@pytest.mark.parametrize("solve", FORMULATIONS[:2])
@pytest.mark.parametrize("tap_side", ["hv", "lv"])
def test_fixed_tap_matches_pandapower_tap_pos(tap_side, solve):
    # tap acts at the from-node, so a tap on side s is wired with the from-node on s.
    vm = _monee_trafo_vm(lv_first=tap_side == "lv", tap=1 - 2 * 0.015, **solve)

    assert vm == pytest.approx(_pp_trafo_vm(tap_side, -2), abs=1e-5)
    assert abs(vm - _pp_trafo_vm()) > 0.02


def _cigre_mv():
    pn = pytest.importorskip("pandapower.networks")
    pp = pytest.importorskip("pandapower")
    ref = pn.create_cigre_network_mv(with_der="pv_wind")
    pp.runpp(ref)

    net = mm.Network(el_model=mm.create_power_grid("power", sn_mva=1.0))
    bus = {b: mx.create_bus(net, base_kv=float(r.vn_kv)) for b, r in ref.bus.iterrows()}
    mx.create_ext_power_grid(net, bus[0], vm_pu=float(ref.ext_grid.vm_pu.iloc[0]))
    for t in ref.trafo.itertuples():
        mx.create_trafo(
            net,
            bus[t.lv_bus],
            bus[t.hv_bus],
            vk_percent=t.vk_percent,
            vkr_percent=t.vkr_percent,
            sn_trafo_mva=t.sn_mva,
            max_i_ka=t.sn_mva / (math.sqrt(3) * t.vn_lv_kv),
        )
    # pandapower keeps a line behind an open line switch energised from its other
    # end; a zero load stops monee pruning that dangling end bus.
    open_end = {
        sw.element: sw.bus
        for sw in ref.switch.itertuples()
        if sw.et == "l" and not sw.closed
    }
    line = {}
    for ln in ref.line.itertuples():
        ends = [bus[ln.from_bus], bus[ln.to_bus]]
        if ln.Index in open_end:
            stub = mx.create_bus(net, base_kv=float(ref.bus.vn_kv[open_end[ln.Index]]))
            mx.create_power_load(net, stub, p_mw=0.0, q_mvar=0.0)
            ends[0 if open_end[ln.Index] == ln.from_bus else 1] = stub
        line[ln.Index] = mx.create_line(
            net,
            *ends,
            length_m=ln.length_km * 1e3,
            r_ohm_per_m=ln.r_ohm_per_km / 1e3,
            x_ohm_per_m=ln.x_ohm_per_km / 1e3,
            c_nf_per_m=ln.c_nf_per_km / 1e3,
            max_i_ka=ln.max_i_ka,
        )
    for ld in ref.load.itertuples():
        mx.create_power_load(net, bus[ld.bus], p_mw=ld.p_mw, q_mvar=ld.q_mvar)
    for sg in ref.sgen.itertuples():
        mx.create_power_generator(net, bus[sg.bus], p_mw=sg.p_mw, q_mvar=sg.q_mvar)
    return ref, net, bus, line, set(open_end)


@pytest.mark.parametrize("solve", FORMULATIONS)
def test_cigre_mv_with_line_charging_matches_pandapower(solve):
    ref, net, bus, line, open_ended = _cigre_mv()

    result = monee.run_energy_flow(net, copy=True, **solve)

    assert result.success
    vm = result.get(mm.Bus).set_index("id")["vm_pu"]
    for b in ref.bus.index:
        assert vm[bus[b]] == pytest.approx(ref.res_bus.vm_pu[b], abs=1e-4)
    lines = result.get(mm.PowerLine).set_index("id")
    for i, bid in line.items():
        if solve is MISOCP and i in open_ended:
            # The relaxation inflates the current of an open-ended charged cable to
            # lift its voltage and charging output; the exact formulation has no gap.
            continue
        loading = 100 * max(lines.loading_from_pu[bid], lines.loading_to_pu[bid])
        assert loading == pytest.approx(ref.res_line.loading_percent[i], abs=0.05)

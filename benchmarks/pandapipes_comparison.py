"""monee (multi-energy NLP) vs pandapipes on gas, heat and coupled el+gas flow.

The multi-energy analogue of ``pandapower_comparison.py``. The AC benchmark
hands the identical electrical model to both engines through the neutral
MATPOWER ``mpc`` exchange, but gas and heat have no neutral exchange format:
monee solves a smooth Weymouth (gas) or Darcy-Weisbach plus temperature (heat)
NLP, while pandapipes solves its own hydraulic/thermal Newton system. So
like-for-like networks are built independently in each tool from one spec, and
the comparison is agreement within tolerance rather than machine equality.

pandapipes does steady-state flow only (no OMEF/OPF), so this is a power-flow
comparison with no optimization arm. monee runs its default in-process
CasADi/IPOPT backend, the most directly comparable option.

Scope (PF only):
    GAS   gas hydraulics                 pandapipes pipeflow   vs monee GAS NLP
    HEAT  water hydraulics + thermal     pandapipes pipeflow   vs monee HEAT NLP
    MES   el+gas (P2G/G2P) + el+gas+heat (CHP)  multinet       vs monee MES NLP

Every network is built before the timer starts; only the solve call is timed.
Each case reports a cross-tool agreement metric (junction pressure in bar, or
junction temperature in K for heat, or bus voltage in pu for MES) plus
wall-clock solve time. The MES case is where monee's simultaneous-NLP coupling
pays off: it matches pandapipes' iterative ``multinet`` to machine precision on
voltage and solves the coupled problem faster.

Outputs: ``results/pandapipes_comparison.csv`` and a Plotly figure
(``results/pandapipes_comparison.html`` plus ``.png`` when kaleido is available).

Run:        python benchmarks/pandapipes_comparison.py
Plot only:  python benchmarks/pandapipes_comparison.py --plot-only

Requires the in-process pandapipes stack (pandapower 3.3.3 / pandapipes 0.14.0 /
simbench 1.6.1); see the project ``testpp`` extra.
"""

from __future__ import annotations

import contextlib
import io
import math
import os
import statistics
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import monee.express as mx
import monee.model as mm
from monee import run_energy_flow
from monee.model.formulation import (
    EL_NLP_FORMULATION,
    make_gas_nlp_formulation,
    make_heat_nlp_formulation,
)
from monee.model.grid import STANDARD_ATMOSPHERE_PA

import pandapower as pp
import pandapipes as ppi
from pandapipes.multinet.control.controller.multinet_control import (
    G2PControlMultiEnergy,
    P2GControlMultiEnergy,
)
from pandapipes.multinet.control.run_control_multinet import run_control
from pandapipes.multinet.create_multinet import add_net_to_multinet, create_empty_multinet

HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, "results")

PANDAPIPES = "pandapipes"
MONEE = "monee · CasADi"

# Shared reference conditions (both tools build from these).
P_REF_BAR = 10.0
GAS_T_K = 300.0
WATER_T_K = 356.0
ROUGHNESS_M = 1e-4
K_MM = ROUGHNESS_M * 1000.0
T_AMBIENT_K = 283.15
# monee WaterPipe insulation defaults (model/branch.py): the heat loss is
# UA = 2*pi*lambda*L / ln(r_out/r_in). pandapipes parametrises the same loss as
# alpha_w_per_m2k * (pi*D*L), so the matching coefficient is 2*lambda / (D*ln(...)).
_LAMBDA_INS = 0.025
_INS_THICK = 0.12


def alpha_match(diameter_m):
    """pandapipes ``alpha_w_per_m2k`` that reproduces monee's insulation heat loss."""
    r_in = diameter_m / 2.0
    r_out = r_in + _INS_THICK
    return 2.0 * _LAMBDA_INS / (diameter_m * math.log(r_out / r_in))


@contextlib.contextmanager
def _silent():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        yield


def _time(fn, repeats: int = 3):
    """Median wall-clock of fn over repeats runs, returning (last result, seconds)."""
    times, res = [], None
    for _ in range(repeats):
        with _silent():
            t0 = time.perf_counter()
            res = fn()
            times.append(time.perf_counter() - t0)
    return res, statistics.median(times)


def _agree(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = min(len(a), len(b))
    return float(np.nanmax(np.abs(a[:n] - b[:n])))


# Shared spec: a star of n junctions, ext grid at the hub (junction 0) and every
# other junction a sink fed by its own spoke pipe. Each pipe carries the same
# single-sink flow, so the pressure regime stays fixed as n scales. A series line
# would pile the cumulative downstream flow onto the trunk and blow the drop up
# super-linearly, diverging both engines.
def line_spec(n, diameter_m, length_m, sink_kgs):
    return dict(
        n=n,
        diameter_m=diameter_m,
        length_m=length_m,
        sink_kgs=sink_kgs,
        pipes=[(0, i) for i in range(1, n)],
    )


# monee builders
def build_monee_gas(spec):
    # pandapipes treats junction p_bar as gauge (absolute = p_bar + 1 atm). Match
    # that here so both engines model the same absolute pressure. Otherwise monee's
    # 10 bar is read as absolute against pandapipes' 11.013 bar, a ~9% gas-drop bias.
    gas = mm.create_gas_grid("gas", type="lgas")
    gas.pressure_ambient_pa = STANDARD_ATMOSPHERE_PA
    net = mm.Network()
    net.set_default_grid(mm.GAS_KEY, gas)
    net.activate_grid(mm.GAS)
    nodes = []
    for i in range(spec["n"]):
        if i == 0:
            child = [net.child(mm.ExtHydrGrid())]
        else:
            child = [net.child(mm.Sink(mass_flow_kgs=spec["sink_kgs"]))]
        nodes.append(net.node(mm.Junction(), mm.GAS, child_ids=child))
    for a, b in spec["pipes"]:
        net.branch(
            mm.GasPipe(diameter_m=spec["diameter_m"], length_m=spec["length_m"],
                       roughness_m=ROUGHNESS_M),
            nodes[a], nodes[b],
        )
    # "hybrid" = 64/Re plus fully-rough, matching pandapipes' default nikuradse
    # law so friction agrees too. A constant friction factor under-predicts by ~4%.
    net.apply_formulation(make_gas_nlp_formulation(friction_model="hybrid"))
    return net


def build_monee_heat(spec):
    net = mm.Network(mm.create_water_grid("water"))
    net.activate_grid(mm.WATER)
    nodes = []
    for i in range(spec["n"]):
        if i == 0:
            child = [net.child(mm.ExtHydrGrid(t_k=WATER_T_K))]
        else:
            child = [net.child(mm.Sink(mass_flow_kgs=spec["sink_kgs"]))]
        nodes.append(net.node(mm.Junction(), mm.WATER, child_ids=child))
    for a, b in spec["pipes"]:
        net.branch(
            mm.WaterPipe(diameter_m=spec["diameter_m"], length_m=spec["length_m"],
                         temperature_ext_k=283.15, roughness_m=ROUGHNESS_M),
            nodes[a], nodes[b],
        )
    net.apply_formulation(make_heat_nlp_formulation(friction_model="hybrid"))
    return net


def _monee_pressure_bar(res, ref_pa=1_000_000.0):
    return res.dataframes["Junction"]["pressure_pu"].to_numpy(float) * ref_pa / 1e5


# pandapipes builders
def build_ppipes_line(spec, fluid, t_k, thermal=False):
    net = ppi.create_empty_network(fluid=fluid)
    j = [ppi.create_junction(net, pn_bar=P_REF_BAR, tfluid_k=t_k) for _ in range(spec["n"])]
    ppi.create_ext_grid(net, junction=j[0], p_bar=P_REF_BAR, t_k=t_k, type="pt")
    # Match monee's insulation heat loss for the thermal (water) comparison; the
    # isothermal gas comparison runs adiabatic (alpha = 0).
    alpha = alpha_match(spec["diameter_m"]) if thermal else 0.0
    for a, b in spec["pipes"]:
        ppi.create_pipe_from_parameters(
            net, from_junction=j[a], to_junction=j[b],
            length_km=spec["length_m"] / 1000.0, diameter_m=spec["diameter_m"],
            k_mm=K_MM, alpha_w_per_m2k=alpha, text_k=T_AMBIENT_K, sections=5,
        )
    for i in range(1, spec["n"]):
        ppi.create_sink(net, junction=j[i], mdot_kg_per_s=spec["sink_kgs"])
    return net


# Suites
def _row(group, case, n_junc, p_pp, p_mn, time_pp, time_mn,
         temp_pp=None, temp_mn=None, vm_pp=None, vm_mn=None):
    drop = max(P_REF_BAR - float(np.nanmin(p_pp)), 1e-9)
    row = dict(
        group=group, case=case, n_junc=n_junc,
        t_pandapipes_ms=round(time_pp * 1000, 2),
        t_monee_ms=round(time_mn * 1000, 1),
        slow_monee=round(time_mn / time_pp, 1) if time_pp else float("nan"),
        p_err_bar=_agree(p_pp, p_mn),
        p_reldiff_pct=round(100 * _agree(p_pp, p_mn) / drop, 1),
        t_err_k=_agree(temp_pp, temp_mn) if temp_pp is not None else float("nan"),
        vm_err_pu=_agree(vm_pp, vm_mn) if vm_pp is not None else float("nan"),
    )
    return row


def run_gas_suite():
    print("\n--- Gas hydraulics: pandapipes pipeflow vs monee Weymouth NLP ---")
    rows = []
    cases = [
        ("gas line-5 D0.3", line_spec(5, 0.3, 8000, 0.4)),
        ("gas line-10 D0.3", line_spec(10, 0.3, 6000, 0.4)),
        ("gas line-20 D0.35", line_spec(20, 0.35, 5000, 0.5)),
        ("gas line-40 D0.4", line_spec(40, 0.4, 4000, 0.6)),
    ]
    for name, spec in cases:
        pnet = build_ppipes_line(spec, "lgas", GAS_T_K)
        _, t_pp = _time(lambda: ppi.pipeflow(pnet))
        p_pp = pnet.res_junction.p_bar.to_numpy(float)

        mnet = build_monee_gas(spec)
        rmn, t_mn = _time(lambda: run_energy_flow(build_monee_gas(spec)), repeats=1)
        p_mn = _monee_pressure_bar(rmn)

        rows.append(_row("GAS", name, spec["n"], p_pp, p_mn, t_pp, t_mn))
        print(f"  {name:18s} pp {t_pp*1000:6.2f}ms  monee {t_mn*1000:7.1f}ms  "
              f"p_err {rows[-1]['p_err_bar']:.4f} bar ({rows[-1]['p_reldiff_pct']}%)")
    return rows


def run_heat_suite():
    # monee's heat NLP always solves coupled hydraulics and temperature transport,
    # so the fair pandapipes comparison runs its thermal solve too
    # (mode="sequential"). Pipe heat loss is matched to monee's insulation model.
    print("\n--- Water hydraulics + thermal: pandapipes vs monee Darcy + temperature NLP ---")
    rows = []
    cases = [
        ("heat line-5 D0.15", line_spec(5, 0.15, 500, 1.0)),
        ("heat line-10 D0.15", line_spec(10, 0.15, 400, 1.0)),
        ("heat line-20 D0.2", line_spec(20, 0.2, 300, 1.5)),
        ("heat line-40 D0.25", line_spec(40, 0.25, 250, 2.0)),
    ]
    for name, spec in cases:
        pnet = build_ppipes_line(spec, "water", WATER_T_K, thermal=True)
        _, t_pp = _time(lambda: ppi.pipeflow(pnet, mode="sequential"))
        p_pp = pnet.res_junction.p_bar.to_numpy(float)
        temp_pp = pnet.res_junction.t_k.to_numpy(float)

        rmn, t_mn = _time(lambda: run_energy_flow(build_monee_heat(spec)), repeats=1)
        p_mn = _monee_pressure_bar(rmn)
        temp_mn = rmn.dataframes["Junction"]["t_k"].to_numpy(float)

        rows.append(_row("HEAT", name, spec["n"], p_pp, p_mn, t_pp, t_mn, temp_pp, temp_mn))
        print(f"  {name:18s} pp {t_pp*1000:6.2f}ms  monee {t_mn*1000:7.1f}ms  "
              f"p {rows[-1]['p_reldiff_pct']:5.1f}%  T_err {rows[-1]['t_err_k']:.3f} K")
    return rows


class CHPControlMultiEnergy(G2PControlMultiEnergy):
    """Three-sector CHP controller for pandapipes' multinet (none ships with it).

    Gas-driven like G2P: reads a gas sink and writes the electrical output to an
    sgen (``eff_power``) and the heat output to a heat_exchanger in a third
    (water) net (``eff_heat``; ``qext_w < 0`` adds heat to the fluid).
    """

    def __init__(self, multinet, idx_power, idx_gas, idx_heat, eff_power, eff_heat,
                 name_heat_net="heat", **kwargs):
        super().__init__(multinet, idx_power, idx_gas, eff_power,
                         name_power_net="power", name_gas_net="gas", **kwargs)
        self.eff_heat = eff_heat
        self.idx_heat = idx_heat
        self.name_net_heat = name_heat_net

    def get_all_net_names(self):
        return [self.name_net_gas, self.name_net_power, self.name_net_heat]

    def control_step(self, multinet):
        gas = float(multinet["nets"][self.name_net_gas].sink.at[self.elm_idx_gas, "mdot_kg_per_s"])
        conv = self.conversion_factor_kgps_to_mw()
        self.power_gen = gas * conv * self.efficiency
        multinet["nets"][self.name_net_power][self.elm_type_power].at[
            self.elm_idx_power, "p_mw"] = self.power_gen
        multinet["nets"][self.name_net_heat].heat_exchanger.at[
            self.idx_heat, "qext_w"] = -gas * conv * self.eff_heat * 1e6
        self.applied = True


# Coupled electricity + gas (P2G / G2P) and electricity + gas + heat (CHP).
# pandapipes couples a pandapower net and pandapipes gas/water nets through
# multinet controllers (P2G/G2P ship; the CHP one is provided above). monee
# couples every carrier in one NLP. To make each pair the same determined
# problem, the operating point is aligned (e.g. monee's P2G gas setpoint equals
# the gas pandapipes produced) and the gas fluids are aligned (both lgas) so the
# HHV, and hence the coupling, matches.
_MES_EFF, _MES_BASE_MW, _MES_SINK_KGS = 0.7, 0.5, 0.3
_MES_GAS_D, _MES_GAS_L = 0.25, 3000.0


def build_ppipes_mes(p_el_mw):
    pn = pp.create_empty_network()
    b0 = pp.create_bus(pn, vn_kv=20.0)
    b1 = pp.create_bus(pn, vn_kv=20.0)
    pp.create_ext_grid(pn, b0, vm_pu=1.0)
    pp.create_line_from_parameters(pn, b0, b1, length_km=2.0, r_ohm_per_km=0.3,
                                   x_ohm_per_km=0.2, c_nf_per_km=0.0, max_i_ka=1.0)
    pp.create_load(pn, b1, p_mw=_MES_BASE_MW)
    lid = pp.create_load(pn, b1, p_mw=p_el_mw)  # the P2G electrical draw
    gn = ppi.create_empty_network(fluid="lgas")
    j0 = ppi.create_junction(gn, pn_bar=P_REF_BAR, tfluid_k=GAS_T_K)
    j1 = ppi.create_junction(gn, pn_bar=P_REF_BAR, tfluid_k=GAS_T_K)
    ppi.create_ext_grid(gn, junction=j0, p_bar=P_REF_BAR, t_k=GAS_T_K, type="pt")
    ppi.create_pipe_from_parameters(gn, from_junction=j0, to_junction=j1,
                                    length_km=_MES_GAS_L / 1000.0, diameter_m=_MES_GAS_D, k_mm=K_MM)
    ppi.create_sink(gn, junction=j1, mdot_kg_per_s=_MES_SINK_KGS)
    src = ppi.create_source(gn, junction=j1, mdot_kg_per_s=0.0)  # filled by P2G
    mnet = create_empty_multinet("mes")
    add_net_to_multinet(mnet, pn, "power")
    add_net_to_multinet(mnet, gn, "gas")
    P2GControlMultiEnergy(mnet, lid, src, efficiency=_MES_EFF,
                          name_power_net="power", name_gas_net="gas")
    return mnet, pn, gn, src


def build_monee_mes(g_produced_kgs):
    net = mm.Network(mm.PowerGrid(name="power", sn_mva=1))
    gas = mm.create_gas_grid("gas", type="lgas")
    gas.pressure_ambient_pa = STANDARD_ATMOSPHERE_PA  # gauge pressures (match pandapipes)
    b0 = net.node(mm.Bus(base_kv=20), mm.EL,
                  child_ids=[net.child(mm.ExtPowerGrid(p_mw=0, q_mvar=0, vm_pu=1, va_degree=0))])
    b1 = net.node(mm.Bus(base_kv=20), mm.EL,
                  child_ids=[net.child(mm.PowerLoad(p_mw=_MES_BASE_MW, q_mvar=0))])
    net.branch(mm.PowerLine(length_m=2000, r_ohm_per_m=0.0003, x_ohm_per_m=0.0002, parallel=1), b0, b1)
    gj0 = net.node(mm.Junction(), child_ids=[net.child(mm.ExtHydrGrid())], grid=gas)
    gj1 = net.node(mm.Junction(), child_ids=[net.child(mm.Sink(mass_flow_kgs=_MES_SINK_KGS))], grid=gas)
    net.branch(mm.GasPipe(diameter_m=_MES_GAS_D, length_m=_MES_GAS_L, roughness_m=ROUGHNESS_M), gj0, gj1)
    mx.create_p2g(net, from_node_id=b1, to_node_id=gj1, efficiency=_MES_EFF,
                  mass_flow_setpoint_kgs=g_produced_kgs, regulation=1)
    net.apply_formulation(EL_NLP_FORMULATION)
    net.apply_formulation(make_gas_nlp_formulation(friction_model="hybrid"))
    return net


# G2P: the reverse coupling (gas turbine, gas in -> power out). pandapipes' G2P
# is gas-driven (a fixed gas sink produces power = mdot*HHV*eff on an sgen);
# monee's is power-driven (p_mw_setpoint draws the matching gas). So the monee
# setpoint is the power pandapipes produced, and both consume the same gas.
_G2P_EFF, _G2P_BASE_LOAD_MW, _G2P_BASE_SINK_KGS = 0.4, 2.0, 0.05


def build_ppipes_mes_g2p(g2p_gas_kgs):
    pn = pp.create_empty_network()
    b0 = pp.create_bus(pn, vn_kv=20.0)
    b1 = pp.create_bus(pn, vn_kv=20.0)
    pp.create_ext_grid(pn, b0, vm_pu=1.0)
    pp.create_line_from_parameters(pn, b0, b1, length_km=2.0, r_ohm_per_km=0.3,
                                   x_ohm_per_km=0.2, c_nf_per_km=0.0, max_i_ka=1.0)
    pp.create_load(pn, b1, p_mw=_G2P_BASE_LOAD_MW)
    sg = pp.create_sgen(pn, b1, p_mw=0.0)  # the G2P power output
    gn = ppi.create_empty_network(fluid="lgas")
    j0 = ppi.create_junction(gn, pn_bar=P_REF_BAR, tfluid_k=GAS_T_K)
    j1 = ppi.create_junction(gn, pn_bar=P_REF_BAR, tfluid_k=GAS_T_K)
    ppi.create_ext_grid(gn, junction=j0, p_bar=P_REF_BAR, t_k=GAS_T_K, type="pt")
    ppi.create_pipe_from_parameters(gn, from_junction=j0, to_junction=j1,
                                    length_km=_MES_GAS_L / 1000.0, diameter_m=_MES_GAS_D, k_mm=K_MM)
    ppi.create_sink(gn, junction=j1, mdot_kg_per_s=_G2P_BASE_SINK_KGS)  # base gas demand
    g2p_sink = ppi.create_sink(gn, junction=j1, mdot_kg_per_s=g2p_gas_kgs)  # G2P consumption
    mnet = create_empty_multinet("mes")
    add_net_to_multinet(mnet, pn, "power")
    add_net_to_multinet(mnet, gn, "gas")
    G2PControlMultiEnergy(mnet, sg, g2p_sink, efficiency=_G2P_EFF, name_power_net="power",
                          name_gas_net="gas", element_type_power="sgen")
    return mnet, pn, gn, sg


def build_monee_mes_g2p(p_produced_mw):
    net = mm.Network(mm.PowerGrid(name="power", sn_mva=1))
    gas = mm.create_gas_grid("gas", type="lgas")
    gas.pressure_ambient_pa = STANDARD_ATMOSPHERE_PA  # gauge pressures (match pandapipes)
    b0 = net.node(mm.Bus(base_kv=20), mm.EL,
                  child_ids=[net.child(mm.ExtPowerGrid(p_mw=0, q_mvar=0, vm_pu=1, va_degree=0))])
    b1 = net.node(mm.Bus(base_kv=20), mm.EL,
                  child_ids=[net.child(mm.PowerLoad(p_mw=_G2P_BASE_LOAD_MW, q_mvar=0))])
    net.branch(mm.PowerLine(length_m=2000, r_ohm_per_m=0.0003, x_ohm_per_m=0.0002, parallel=1), b0, b1)
    gj0 = net.node(mm.Junction(), child_ids=[net.child(mm.ExtHydrGrid())], grid=gas)
    gj1 = net.node(mm.Junction(), child_ids=[net.child(mm.Sink(mass_flow_kgs=_G2P_BASE_SINK_KGS))], grid=gas)
    net.branch(mm.GasPipe(diameter_m=_MES_GAS_D, length_m=_MES_GAS_L, roughness_m=ROUGHNESS_M), gj0, gj1)
    mx.create_g2p(net, from_node_id=gj1, to_node_id=b1, efficiency=_G2P_EFF,
                  p_mw_setpoint=p_produced_mw, regulation=1)
    net.apply_formulation(EL_NLP_FORMULATION)
    net.apply_formulation(make_gas_nlp_formulation(friction_model="hybrid"))
    return net


# CHP: gas -> power + heat (all three sectors). Both tools are gas-driven (a
# fixed gas input produces both outputs at eff_power / eff_heat), and with the
# gas fluids aligned (lgas) the P_el and Q_heat match by construction. The heat
# raises the water temperature; the two DH thermal models differ (like the HEAT
# suite), so the cross-tool agreement metrics here are the bus voltage (power)
# and gas pressure, both still machine-precise.
_CHP_EFF_P, _CHP_EFF_H, _CHP_BASE_LOAD_MW = 0.4, 0.45, 2.0
_CHP_MDOT_W, _CHP_T_IN = 3.0, 330.0


def build_ppipes_mes_chp(gas_kgs, water_flow_kgs, t_in_k):
    # water_flow_kgs and t_in_k come from monee's solved CHP so the heat exchanger
    # sees the same mass flow and inlet temperature. Then the HE temperature rise
    # (= Q_heat / (m*cp)) is a like-for-like cross-tool metric.
    pn = pp.create_empty_network()
    b0 = pp.create_bus(pn, vn_kv=20.0)
    b1 = pp.create_bus(pn, vn_kv=20.0)
    pp.create_ext_grid(pn, b0, vm_pu=1.0)
    pp.create_line_from_parameters(pn, b0, b1, length_km=2.0, r_ohm_per_km=0.3,
                                   x_ohm_per_km=0.2, c_nf_per_km=0.0, max_i_ka=1.0)
    pp.create_load(pn, b1, p_mw=_CHP_BASE_LOAD_MW)
    sg = pp.create_sgen(pn, b1, p_mw=0.0)  # CHP electrical output
    gn = ppi.create_empty_network(fluid="lgas")
    gj0 = ppi.create_junction(gn, pn_bar=P_REF_BAR, tfluid_k=GAS_T_K)
    gj1 = ppi.create_junction(gn, pn_bar=P_REF_BAR, tfluid_k=GAS_T_K)
    ppi.create_ext_grid(gn, junction=gj0, p_bar=P_REF_BAR, t_k=GAS_T_K, type="pt")
    ppi.create_pipe_from_parameters(gn, from_junction=gj0, to_junction=gj1,
                                    length_km=_MES_GAS_L / 1000.0, diameter_m=_MES_GAS_D, k_mm=K_MM)
    cg = ppi.create_sink(gn, junction=gj1, mdot_kg_per_s=gas_kgs)  # CHP gas consumption
    hn = ppi.create_empty_network(fluid="water")
    h0 = ppi.create_junction(hn, pn_bar=5, tfluid_k=t_in_k)
    h1 = ppi.create_junction(hn, pn_bar=5, tfluid_k=t_in_k)
    h2 = ppi.create_junction(hn, pn_bar=5, tfluid_k=t_in_k)
    ppi.create_ext_grid(hn, junction=h0, p_bar=5, t_k=t_in_k, type="pt")
    ppi.create_pipe_from_parameters(hn, from_junction=h0, to_junction=h1, length_km=0.05,
                                    diameter_m=0.15, k_mm=K_MM, alpha_w_per_m2k=0.0)
    ch = ppi.create_heat_exchanger(hn, h1, h2, qext_w=-1.0, inner_diameter_mm=150.0)  # CHP heat output
    ppi.create_sink(hn, junction=h2, mdot_kg_per_s=water_flow_kgs)
    ppi.set_user_pf_options(hn, mode="sequential")  # run the heat net's thermal solve
    mnet = create_empty_multinet("mes")
    add_net_to_multinet(mnet, pn, "power")
    add_net_to_multinet(mnet, gn, "gas")
    add_net_to_multinet(mnet, hn, "heat")
    CHPControlMultiEnergy(mnet, sg, cg, ch, _CHP_EFF_P, _CHP_EFF_H, element_type_power="sgen")
    return mnet, pn, gn, hn, h1, h2


def build_monee_mes_chp(gas_kgs):
    net = mm.Network()
    gas = mm.create_gas_grid("gas", type="lgas")
    gas.pressure_ambient_pa = STANDARD_ATMOSPHERE_PA  # gauge pressures (match pandapipes)
    power = mm.create_power_grid("power")
    b0 = net.node(mm.Bus(base_kv=20), grid=power,
                  child_ids=[net.child(mm.ExtPowerGrid(p_mw=0, q_mvar=0, vm_pu=1, va_degree=0))])
    b1 = net.node(mm.Bus(base_kv=20), grid=power,
                  child_ids=[net.child(mm.PowerLoad(p_mw=_CHP_BASE_LOAD_MW, q_mvar=0))])
    net.branch(mm.PowerLine(length_m=2000, r_ohm_per_m=0.0003, x_ohm_per_m=0.0002, parallel=1), b0, b1)
    gj0 = net.node(mm.Junction(), grid=gas, child_ids=[net.child(mm.ExtHydrGrid())])
    gj1 = net.node(mm.Junction(), grid=gas, child_ids=[net.child(mm.Sink(mass_flow_kgs=0.0))])
    net.branch(mm.GasPipe(diameter_m=_MES_GAS_D, length_m=_MES_GAS_L, roughness_m=ROUGHNESS_M), gj0, gj1)
    w_ret = net.node(mm.Junction(), grid=mm.WATER_KEY, child_ids=[net.child(mm.ExtHydrGrid(t_k=_CHP_T_IN))])
    w_sup = net.node(mm.Junction(), grid=mm.WATER_KEY)
    w_c = net.node(mm.Junction(), grid=mm.WATER_KEY, child_ids=[net.child(mm.ConsumeHydrGrid(_CHP_MDOT_W))])
    net.branch(mm.WaterPipe(diameter_m=0.15, length_m=100), w_sup, w_c)
    mx.create_chp(net, power_node_id=b1, heat_node_id=w_ret, heat_return_node_id=w_sup, gas_node_id=gj1,
                  mass_flow_setpoint_kgs=gas_kgs, diameter_m=0.15,
                  efficiency_power=_CHP_EFF_P, efficiency_heat=_CHP_EFF_H, regulation=1)
    net.apply_formulation(EL_NLP_FORMULATION)
    net.apply_formulation(make_gas_nlp_formulation(friction_model="hybrid"))
    net.apply_formulation(make_heat_nlp_formulation(friction_model="hybrid"))
    return net


def run_mes_suite():
    print("\n--- Coupled MES (P2G / G2P / CHP): pandapipes multinet vs monee NLP ---")
    rows = []
    # P2G: power -> gas
    for name, p_el in [("mes P2G 0.5MW", 0.5), ("mes P2G 1.0MW", 1.0), ("mes P2G 2.0MW", 2.0)]:
        mnet, pn, gn, src = build_ppipes_mes(p_el)
        _, t_pp = _time(lambda: run_control(mnet), repeats=1)
        vm_pp = pn.res_bus.vm_pu.to_numpy(float)
        p_pp = gn.res_junction.p_bar.to_numpy(float)
        g_produced = float(gn.source.mdot_kg_per_s[src])

        rmn, t_mn = _time(lambda: run_energy_flow(build_monee_mes(g_produced)), repeats=1)
        vm_mn = rmn.dataframes["Bus"]["vm_pu"].to_numpy(float)
        p_mn = rmn.dataframes["Junction"]["pressure_pu"].to_numpy(float) * P_REF_BAR

        rows.append(_row("MES", name, 2, p_pp, p_mn, t_pp, t_mn, vm_pp=vm_pp, vm_mn=vm_mn))
        print(f"  {name:16s} pp {t_pp*1000:6.2f}ms  monee {t_mn*1000:7.1f}ms  "
              f"vm_err {rows[-1]['vm_err_pu']:.2e} pu  gas_p {rows[-1]['p_reldiff_pct']:5.1f}%")

    # G2P: gas -> power
    for name, g2p_gas in [("mes G2P 0.05kgs", 0.05), ("mes G2P 0.10kgs", 0.10), ("mes G2P 0.15kgs", 0.15)]:
        mnet, pn, gn, sg = build_ppipes_mes_g2p(g2p_gas)
        _, t_pp = _time(lambda: run_control(mnet), repeats=1)
        vm_pp = pn.res_bus.vm_pu.to_numpy(float)
        p_pp = gn.res_junction.p_bar.to_numpy(float)
        p_produced = float(pn.res_sgen.p_mw[sg])

        rmn, t_mn = _time(lambda: run_energy_flow(build_monee_mes_g2p(p_produced)), repeats=1)
        vm_mn = rmn.dataframes["Bus"]["vm_pu"].to_numpy(float)
        p_mn = rmn.dataframes["Junction"]["pressure_pu"].to_numpy(float) * P_REF_BAR

        rows.append(_row("MES", name, 2, p_pp, p_mn, t_pp, t_mn, vm_pp=vm_pp, vm_mn=vm_mn))
        print(f"  {name:16s} pp {t_pp*1000:6.2f}ms  monee {t_mn*1000:7.1f}ms  "
              f"vm_err {rows[-1]['vm_err_pu']:.2e} pu  gas_p {rows[-1]['p_reldiff_pct']:5.1f}%")

    # CHP: gas -> power + heat (all three sectors). monee is solved first; its CHP
    # heat-exchanger water flow and inlet temperature are fed to pandapipes so the
    # HE temperature rise (= Q_heat / (m*cp)) is a like-for-like metric. The
    # absolute supply temperature is not; see the README on the CHP heat models.
    for name, gas_kgs in [("mes CHP 0.05kgs", 0.05), ("mes CHP 0.10kgs", 0.10)]:
        rmn, t_mn = _time(lambda: run_energy_flow(build_monee_mes_chp(gas_kgs)), repeats=1)
        vm_mn = rmn.dataframes["Bus"]["vm_pu"].to_numpy(float)
        # gas junctions are the first two Junction rows (created before the water ones)
        p_mn = rmn.dataframes["Junction"]["pressure_pu"].to_numpy(float)[:2] * P_REF_BAR
        she = rmn.dataframes["SubHE"].iloc[0]
        mdot_w = abs(float(she["mass_flow_kgs"]))
        t_ref = 356.0
        t_in_m = float(she["t_in_pu"]) * t_ref
        dT_m = (float(she["t_out_pu"]) - float(she["t_in_pu"])) * t_ref

        mnet, pn, gn, hn, h1, h2 = build_ppipes_mes_chp(gas_kgs, mdot_w, t_in_m)
        _, t_pp = _time(lambda: run_control(mnet), repeats=1)
        vm_pp = pn.res_bus.vm_pu.to_numpy(float)
        p_pp = gn.res_junction.p_bar.to_numpy(float)
        dT_p = float(hn.res_junction.t_k[h2]) - float(hn.res_junction.t_k[h1])

        row = _row("MES", name, 2, p_pp, p_mn, t_pp, t_mn, vm_pp=vm_pp, vm_mn=vm_mn)
        row["t_err_k"] = abs(dT_m - dT_p)  # CHP heat-exchanger temperature-rise agreement (K)
        rows.append(row)
        print(f"  {name:16s} pp {t_pp*1000:6.2f}ms  monee {t_mn*1000:7.1f}ms  "
              f"vm_err {row['vm_err_pu']:.2e} pu  HE_dT_err {row['t_err_k']:.2f} K")
    return rows


CSV_PATH = os.path.join(RESULTS, "pandapipes_comparison.csv")
HTML_PATH = os.path.join(RESULTS, "pandapipes_comparison.html")
PNG_PATH = os.path.join(RESULTS, "pandapipes_comparison.png")
SVG_PATH = os.path.join(RESULTS, "pandapipes_comparison.svg")


# Shared publication palette / typography (kept consistent across all three
# benchmark figures so the suite reads as one set of plots).
C_PANDAPIPES = "#d62728"   # reference engine (red)
C_MONEE = "#2ca02c"        # monee (green)
C_PRESSURE = "#1f77b4"     # pressure-agreement metric (blue)
C_TEMP = "#ff7f0e"         # temperature-agreement metric (orange)
# The figures are transparent and embedded in docs under both a light and a dark
# theme, and site CSS cannot reach inside an <img>-embedded SVG. So all text uses
# one theme-neutral mid-grey that keeps at least 4:1 contrast on both white and
# dark backgrounds; only the data bars carry saturated colour.
TEXT = "#737373"
C_BANNER = TEXT
GRID = "rgba(128,128,128,0.22)"
AXIS_LINE = "rgba(128,128,128,0.5)"
FONT_FAMILY = "Inter, Segoe UI, Helvetica, Arial"
# Sized to read at the width the figure is embedded at (no zoom needed).
AXIS_TITLE_SIZE = 22
TICK_SIZE = 20
LABEL_SIZE = 19
BANNER_SIZE = 23


def _style_axes(fig):
    """Uniform axis cosmetics applied to every subplot."""
    fig.update_xaxes(
        showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False,
        showline=True, linecolor=AXIS_LINE, linewidth=1, ticks="outside",
        ticklen=4, tickcolor=AXIS_LINE, tickfont={"size": TICK_SIZE, "color": TEXT},
        title_font={"size": AXIS_TITLE_SIZE, "color": TEXT},
    )
    fig.update_yaxes(
        showgrid=False, zeroline=False, showline=False,
        tickfont={"size": TICK_SIZE, "color": TEXT}, automargin=True,
    )


def make_plot(df, out_html, out_png, out_svg=None):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    color = {PANDAPIPES: C_PANDAPIPES, MONEE: C_MONEE}
    groups = [
        ("GAS", "Gas hydraulics: pandapipes pipeflow vs monee Weymouth NLP"),
        ("HEAT", "Water hydraulics + thermal: pandapipes pipeflow vs monee Darcy NLP"),
        ("MES", "Coupled MES (P2G + G2P + CHP): pandapipes multinet vs monee NLP"),
    ]
    groups = [g for g in groups if (df.group == g[0]).any()]

    # Row heights proportional to each group's case count (figure height
    # proportional to the total) so every bar is the SAME on-screen thickness,
    # whatever number of cases a group holds.
    counts = [int((df.group == g[0]).sum()) for g in groups]
    total_cases = max(sum(counts), 1)
    n_groups = len(groups)

    # Three metric columns: solve time | pressure agreement | temperature
    # agreement. The temperature column is only populated where heat participates
    # (HEAT suite + the CHP MES cases); other rows are left blank by design.
    fig = make_subplots(
        rows=n_groups, cols=3,
        column_widths=[0.46, 0.27, 0.27],
        row_heights=[c / total_cases for c in counts],
        shared_yaxes=True,
        horizontal_spacing=0.06, vertical_spacing=0.13,
    )

    for r, (g, _t) in enumerate(groups, start=1):
        sub = df[df.group == g].iloc[::-1]
        cases = sub.case.tolist()
        last_row = r == n_groups

        # col 1: solve time (grouped, log)
        for col, backend in [("t_pandapipes_ms", PANDAPIPES), ("t_monee_ms", MONEE)]:
            fig.add_trace(
                go.Bar(y=cases, x=sub[col], name=backend, orientation="h",
                       marker_color=color[backend], marker_line_width=0,
                       legendgroup=backend, showlegend=(r == 1), cliponaxis=False,
                       text=[f"{v:.1f}" for v in sub[col]], textposition="outside",
                       textfont={"size": LABEL_SIZE, "color": TEXT},
                       hovertemplate=f"{backend}: %{{x:.2f}} ms<extra></extra>"),
                row=r, col=1,
            )
        tvals = sub[["t_pandapipes_ms", "t_monee_ms"]].to_numpy(float)
        fig.update_xaxes(
            type="log", row=r, col=1,
            title_text="solve time (ms, log)" if last_row else None,
            range=[np.log10(np.nanmin(tvals) * 0.5),
                   np.log10(np.nanmax(tvals) * 3.4)],
        )

        # col 2: pressure agreement (% of pipe drop)
        p = sub.p_reldiff_pct.to_numpy(float)
        fig.add_trace(
            go.Bar(y=cases, x=p, orientation="h", showlegend=False,
                   marker_color=C_PRESSURE, marker_line_width=0, cliponaxis=False,
                   text=[f"{v:.1f}%" for v in p], textposition="outside",
                   textfont={"size": LABEL_SIZE, "color": TEXT},
                   hovertemplate="pressure diff %{x:.2f}% of drop<extra></extra>"),
            row=r, col=2,
        )
        fig.update_xaxes(
            row=r, col=2,
            title_text="pressure diff (% of drop)" if last_row else None,
            range=[0, max(float(np.nanmax(p)) * 1.5, 0.1)],
        )

        # col 3: temperature agreement (K), heat-only, blank elsewhere
        t = sub.t_err_k.to_numpy(float)
        finite = np.isfinite(t)
        fig.add_trace(
            go.Bar(y=cases,
                   x=[v if f else None for v, f in zip(t, finite)],
                   orientation="h", showlegend=False,
                   marker_color=C_TEMP, marker_line_width=0, cliponaxis=False,
                   text=[f"{v:.3g}" if f else "" for v, f in zip(t, finite)],
                   textposition="outside",
                   textfont={"size": LABEL_SIZE, "color": TEXT},
                   hovertemplate="ΔT %{x:.4g} K<extra></extra>"),
            row=r, col=3,
        )
        if finite.any():
            fig.update_xaxes(
                row=r, col=3,
                title_text="temperature diff (K)" if last_row else None,
                range=[0, float(np.nanmax(t[finite])) * 1.5],
            )
        else:
            # GAS is isothermal, so no temperature is modelled. Keep the cell but
            # label it so the blank reads as intentional, not missing data.
            fig.update_xaxes(row=r, col=3, range=[0, 1], showticklabels=False,
                             title_text="temperature diff (K)" if last_row else None)
            fig.add_annotation(text="isothermal, no ΔT", row=r, col=3,
                               xref="x domain", yref="y domain", x=0.5, y=0.5,
                               showarrow=False, font={"size": 20, "color": TEXT},
                               xanchor="center")

        # group banner over the left column
        fig.add_annotation(text=f"<b>{_t}</b>", row=r, col=1,
                           xref="x domain", yref="y domain", x=0, y=1.0,
                           yshift=20, showarrow=False, font={"size": BANNER_SIZE,
                           "color": C_BANNER}, xanchor="left")

    _style_axes(fig)
    fig.update_layout(
        title={"text": "<b>monee (multi-energy NLP) vs pandapipes</b><br>"
                       f"<span style='font-size:23px;color:{TEXT}'>gas, heat &amp; "
                       "coupled el+gas+heat flow: solve time and cross-tool "
                       "agreement</span>",
               "x": 0.5, "xanchor": "center", "y": 0.978, "yanchor": "top",
               "font": {"size": 31, "color": TEXT}},
        barmode="group", bargap=0.32, bargroupgap=0.12, template="plotly_white",
        height=int(80 * total_cases + 110 * n_groups + 170), width=1280,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.012,
                "xanchor": "right", "x": 1.0, "font": {"size": 21, "color": TEXT},
                "bgcolor": "rgba(0,0,0,0)"},
        margin={"l": 135, "r": 60, "t": 215, "b": 70},
        font={"family": FONT_FAMILY, "size": 21, "color": TEXT},
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        uniformtext={"mode": "hide", "minsize": 11},
    )
    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"Wrote {out_html}")
    for path, scale in [(out_png, 2), (out_svg, 1)]:
        if not path:
            continue
        try:
            fig.write_image(path, scale=scale)
            print(f"Wrote {path}")
        except Exception as exc:
            print(f"(static export skipped for {os.path.basename(path)}: {exc})")


def regenerate_plot():
    if not os.path.exists(CSV_PATH):
        raise SystemExit(f"No results CSV at {CSV_PATH}; run the benchmark once first.")
    df = pd.read_csv(CSV_PATH)
    print(f"Re-plotting from {CSV_PATH} ({len(df)} cases)")
    make_plot(df, HTML_PATH, PNG_PATH, SVG_PATH)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    # warm-up (import / JIT out of the timings)
    with _silent():
        try:
            ppi.pipeflow(build_ppipes_line(line_spec(3, 0.3, 1000, 0.2), "lgas", GAS_T_K))
            run_energy_flow(build_monee_gas(line_spec(3, 0.3, 1000, 0.2)))
            mnet, _pn, gn, src = build_ppipes_mes(0.5)
            run_control(mnet)
            run_energy_flow(build_monee_mes(float(gn.source.mdot_kg_per_s[src])))
        except Exception:
            pass

    rows = run_gas_suite() + run_heat_suite() + run_mes_suite()
    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)
    print("\n=== monee vs pandapipes (gas / heat / coupled el+gas) ===\n")
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(df.to_string(index=False))
    mes = df[df.group == "MES"]
    print(f"\nmax gas/heat pressure gap: "
          f"{df[df.group != 'MES'].p_reldiff_pct.max():.1f}% of pipe drop")
    if not mes.empty:
        print(f"coupled el+gas: bus voltage agrees to {mes.vm_err_pu.max():.2e} pu, "
              f"gas pressure to {mes.p_err_bar.max():.2e} bar")
    print(f"Wrote {CSV_PATH}")
    # Plot shows all three suites: solve time, pressure agreement, and, where
    # heat participates (HEAT plus the CHP MES cases), temperature agreement. The
    # coupled MES voltage match is reported in the CSV/console above.
    make_plot(df, HTML_PATH, PNG_PATH, SVG_PATH)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip solves; only regenerate the plot from the CSV.")
    args = parser.parse_args()
    if args.plot_only:
        regenerate_plot()
    else:
        main()

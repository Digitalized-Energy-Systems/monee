"""monee (electric, smooth-NLP) vs pandapower's native AC solvers.

Companion to ``backend_comparison.py``: same electric grids, restricted to the
NLP (AC) electricity cases, benchmarked against pandapower's dedicated solvers,
Newton-Raphson power flow (``pp.runpp``) and the PYPOWER AC-OPF (``pp.runopp``).
The question is how monee's general multi-energy NLP formulation compares, on
pure AC electricity, to a tool built only for power systems.

monee runs through its in-process CasADi/IPOPT backend (the default for IPOPT
requests), which is the most directly comparable option to pandapower's
in-process solvers. GEKKO is omitted because CasADi solves the same NLP
in-process and is faster here; see ``backend_comparison.py`` for that shoot-out.

As in ``backend_comparison.py``, only the solve call is timed
(``run_energy_flow`` / ``run_energy_flow_optimization`` / ``pp.runpp`` /
``pp.runopp``); every network is built before the timer starts. Each case also
reports a cross-tool correctness metric, since the solvers must agree on the
physics or optimum. pandapower is expected to win on speed, solving a dedicated
power-flow Jacobian rather than a general NLP; the point is to quantify the gap.

Grids are reproducible pandapower built-ins (no simbench download):
    PF      cigre_mv, cigre_mv+DER, cigre_lv   (pp.runpp   vs monee AC NLP)
    OPF     feeder_40, feeder_80               (pp.runopp  vs monee econ dispatch)
    OPF-LL  feeder_40, feeder_80 (line limit)  (pp.runopp  vs monee econ dispatch)

Each case reports the per-engine solve time plus the two cross-tool agreement
metrics the plot shows: the bus-voltage signature error ``|Δvm|`` (pu) and the
slack active-power error ``|ΔP|`` (MW). Together they show that monee reproduces
pandapower's voltage and power solution rather than just running near it.

Outputs: ``results/pandapower_comparison.csv`` and a Plotly figure
(``results/pandapower_comparison.html`` + ``.png`` / ``.svg`` via kaleido).

Run:            python benchmarks/pandapower_comparison.py
Plot only:      python benchmarks/pandapower_comparison.py --plot-only
"""

from __future__ import annotations

import os
import warnings

warnings.filterwarnings("ignore")

# PYPOWER (pandapower's AC-OPF engine) uses ``sparse.H``, removed in SciPy>=1.11.
# Restore it before any runopp call so the OPF benchmark works on modern SciPy.
import scipy.sparse as _sp

for _cls in (_sp.csc_matrix, _sp.csr_matrix, _sp.coo_matrix):
    if not hasattr(_cls, "H"):
        _cls.H = property(lambda self: self.conj().transpose())

import tempfile
import uuid

import numpy as np
import pandapower as pp
import pandapower.networks as ppn
import pandas as pd
import scipy.io

# Reuse the shared builders/timing from the sibling benchmark (same directory is
# on sys.path when this script is run directly).
from backend_comparison import (
    RESULTS,
    _feeder,
    _mnet,
    _time,
    to_mpc,
)

from monee import run_energy_flow, run_energy_flow_optimization
from monee.io.matpower import _mpc_from_mat, build_matpower_opf
from monee.solver.casadi import CasADiSolver

_SQRT_3 = float(np.sqrt(3))

PANDAPOWER = "pandapower"
CASADI = "monee · CasADi"


def _matpower_opf(pp_net, max_loading=None, max_i_ka=None):
    tmp = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.mat")
    to_mpc(pp_net, init="flat", filename=tmp)
    mpc = _mpc_from_mat(scipy.io.loadmat(tmp))
    os.remove(tmp)
    if max_i_ka is not None:
        bus_kv = {int(row[0]): row[9] for row in mpc["bus"]}
        for row in mpc["branch"]:
            row[5] = max_i_ka * _SQRT_3 * bus_kv[int(row[0])]
    return build_matpower_opf(mpc, max_loading=max_loading, limit_basis="current")


def _vm_stats(arr):
    """Voltage signature [min, mean, max]. The matpower bridge adds a few
    auxiliary buses, so a per-bus diff would not line up across tools."""
    arr = np.asarray(arr, dtype=float)
    return np.array([np.nanmin(arr), np.nanmean(arr), np.nanmax(arr)])


def _agree(a, b):
    return float(np.nanmax(np.abs(np.asarray(a) - np.asarray(b))))


def _rel(a, b):
    return abs(a - b) / max(abs(b), 1.0)


def _slack_p_pp(net):
    """Total slack (ext-grid) active power [MW], pandapower sign-agnostic."""
    return abs(float(net.res_ext_grid.p_mw.sum()))


def _slack_p_mn(res):
    """Total slack (ext-grid) active power [MW] from a monee result. monee uses
    the load sign convention (slack import is negative), so we compare magnitudes."""
    return abs(float(res.dataframes["ExtPowerGrid"]["p_mw"].sum()))


def _pf_row(
    group, case, n_bus, t_pp, t_cas, vm_pp, vm_cas, p_pp, p_cas, cost_rel_err, ok
):
    """One comparison row: timings plus the two cross-tool agreement metrics the
    plot shows, bus-voltage |Δvm| (pu) and slack-power |ΔP| (MW)."""
    return dict(
        group=group,
        case=case,
        n_bus=n_bus,
        t_pandapower_ms=round(t_pp * 1000, 1),
        t_casadi_ms=round(t_cas * 1000, 1),
        slow_casadi=round(t_cas / t_pp, 2) if t_pp else float("nan"),
        vm_err_pu=_agree(vm_cas, vm_pp),
        p_err_mw=abs(p_cas - p_pp),
        cost_rel_err=cost_rel_err,
        ok=ok,
    )


def run_pf_suite():
    """AC power flow: pp.runpp vs monee CasADi/IPOPT."""
    print("\n--- Power flow: pandapower runpp vs monee NLP ---")
    rows = []
    # CIGRE MV reference feeder, its DER variant, and the larger CIGRE LV grid.
    # All distribution networks, which is where monee's PQ+slack NLP applies.
    cases = [
        ("cigre_mv", ppn.create_cigre_network_mv),
        ("cigre_mv + DER", lambda: ppn.create_cigre_network_mv(with_der="all")),
        ("cigre_lv", ppn.create_cigre_network_lv),
    ]
    for name, loader in cases:
        pp_net = loader()
        _, t_pp = _time(lambda: pp.runpp(pp_net))
        vm_pp = _vm_stats(pp_net.res_bus.vm_pu.to_numpy())
        p_pp = _slack_p_pp(pp_net)
        n_bus = int(len(pp_net.res_bus))

        # Same grid into monee's AC NLP via the matpower bridge.
        from monee.model.formulation import EL_NLP_FORMULATION

        def _mn():
            m = _mnet(loader)
            m.apply_formulation(EL_NLP_FORMULATION)
            return m

        nc = _mn()  # build the network outside the timed region
        rc, t_cas = _time(
            lambda: run_energy_flow(nc, solver=CasADiSolver(), simulation=True)
        )
        vm_cas = _vm_stats(rc.dataframes["Bus"]["vm_pu"].to_numpy())
        p_cas = _slack_p_mn(rc)

        rows.append(
            _pf_row(
                "PF",
                f"PF · {name}",
                n_bus,
                t_pp,
                t_cas,
                vm_pp,
                vm_cas,
                p_pp,
                p_cas,
                float("nan"),
                bool(rc.success),
            )
        )
        print(
            f"  {rows[-1]['case']:18s} pp {t_pp * 1000:6.1f}ms  "
            f"CasADi {t_cas * 1000:6.1f}ms  dVm {rows[-1]['vm_err_pu']:.1e}  "
            f"dP {rows[-1]['p_err_mw']:.1e} MW"
        )
    return rows


def run_opf_suite():
    """AC OPF: pp.runopp vs monee, both solving the same MATPOWER OPF case."""
    print("\n--- Optimal power flow: pandapower runopp vs monee (MATPOWER OPF) ---")
    rows = []
    for n in (40, 80):
        pp_net = _feeder(n)
        _, t_pp = _time(lambda: pp.runopp(pp_net))
        cost_pp = float(pp_net.res_cost)
        vm_pp = _vm_stats(pp_net.res_bus.vm_pu.to_numpy())
        p_pp = _slack_p_pp(pp_net)
        n_bus = int(len(pp_net.res_bus))

        mc, pcb = _matpower_opf(_feeder(n))
        rc, t_cas = _time(
            lambda: run_energy_flow_optimization(mc, pcb, solver=CasADiSolver())
        )
        vm_cas = _vm_stats(rc.dataframes["Bus"]["vm_pu"].to_numpy())
        p_cas = _slack_p_mn(rc)

        rows.append(
            _pf_row(
                "OPF",
                f"OPF · feeder_{n}",
                n_bus,
                t_pp,
                t_cas,
                vm_pp,
                vm_cas,
                p_pp,
                p_cas,
                _rel(float(rc.objective), cost_pp),
                bool(rc.success),
            )
        )
        print(
            f"  {rows[-1]['case']:18s} pp {t_pp * 1000:6.1f}ms  "
            f"CasADi {t_cas * 1000:6.1f}ms  dVm {rows[-1]['vm_err_pu']:.1e}  "
            f"dP {rows[-1]['p_err_mw']:.1e} MW  cost pp={cost_pp:.2f} monee={float(rc.objective):.2f}"
        )
    return rows


# A feeder whose unconstrained least-cost dispatch overloads its lines (cheap
# slack import plus expensive local generators on every lateral), so a
# line-loading limit binds and forces costlier local redispatch. ``max_i_ka``
# is sized so the unconstrained optimum loads the lines to roughly 150%.
_LL_MAX_I_KA = 0.035
_LL_MAX_LOADING = 1.0  # 100% line-loading cap (binds)


def _congested_feeder(n_bus, max_i_ka, lat_len=6):
    net = pp.create_empty_network(sn_mva=1.0)
    b0 = pp.create_bus(net, vn_kv=20.0, min_vm_pu=0.9, max_vm_pu=1.1)
    pp.create_ext_grid(
        net,
        b0,
        vm_pu=1.0,
        va_degree=0.0,
        min_p_mw=-1e3,
        max_p_mw=1e3,
        min_q_mvar=-1e3,
        max_q_mvar=1e3,
    )
    made, lat = 1, 0
    while made < n_bus:
        prev = b0
        for _ in range(min(lat_len, n_bus - made)):
            bb = pp.create_bus(net, vn_kv=20.0, min_vm_pu=0.9, max_vm_pu=1.1)
            pp.create_line_from_parameters(
                net,
                prev,
                bb,
                length_km=0.8,
                r_ohm_per_km=0.15,
                x_ohm_per_km=0.2,
                c_nf_per_km=0.0,
                max_i_ka=max_i_ka,
            )
            pp.create_load(net, bb, p_mw=0.3, q_mvar=0.06)
            prev = bb
            made += 1
        gi = pp.create_gen(
            net,
            prev,
            p_mw=0.4,
            min_p_mw=0.0,
            max_p_mw=2.5,
            min_q_mvar=-3,
            max_q_mvar=3,
            vm_pu=1.0,
            controllable=True,
        )
        pp.create_poly_cost(net, gi, "gen", cp1_eur_per_mw=40.0 + 5.0 * lat)
        lat += 1
    pp.create_poly_cost(
        net, 0, "ext_grid", cp1_eur_per_mw=10.0
    )  # cheap import overloads lines
    return net


def run_opf_limited_suite():
    """Line-limited AC OPF: pandapower runopp (max_loading_percent) vs monee on
    the same MATPOWER OPF case with a line-loading-limit constraint. The cap
    binds, so monee's current/loading intermediates re-enter the model (inlined
    into the limit). pandapower always models line limits; here monee does too,
    and the limit is active on both, which keeps the comparison fair."""
    print(
        "\n--- Line-limited OPF: pandapower runopp vs monee dispatch (limit binds) ---"
    )
    rows = []
    for n in (40, 80):
        pp_net = _congested_feeder(n, _LL_MAX_I_KA)
        pp_net.line["max_loading_percent"] = _LL_MAX_LOADING * 100.0
        _, t_pp = _time(lambda: pp.runopp(pp_net))
        cost_pp = float(pp_net.res_cost)
        vm_pp = _vm_stats(pp_net.res_bus.vm_pu.to_numpy())
        p_pp = _slack_p_pp(pp_net)
        n_bus = int(len(pp_net.res_bus))

        mc, pcb = _matpower_opf(
            _congested_feeder(n, _LL_MAX_I_KA),
            max_loading=_LL_MAX_LOADING,
            max_i_ka=_LL_MAX_I_KA,
        )
        rc, t_cas = _time(
            lambda: run_energy_flow_optimization(mc, pcb, solver=CasADiSolver())
        )
        vm_cas = _vm_stats(rc.dataframes["Bus"]["vm_pu"].to_numpy())
        p_cas = _slack_p_mn(rc)

        rows.append(
            _pf_row(
                "OPF-LL",
                f"OPF-LL · feeder_{n}",
                n_bus,
                t_pp,
                t_cas,
                vm_pp,
                vm_cas,
                p_pp,
                p_cas,
                _rel(float(rc.objective), cost_pp),
                bool(rc.success),
            )
        )
        print(
            f"  {rows[-1]['case']:18s} pp {t_pp * 1000:6.1f}ms  "
            f"CasADi {t_cas * 1000:6.1f}ms  dVm {rows[-1]['vm_err_pu']:.1e}  "
            f"dP {rows[-1]['p_err_mw']:.1e} MW  cost pp={cost_pp:.1f} monee={float(rc.objective):.1f}"
        )
    return rows


# Plot, same visual language as backend_comparison.py.
CSV_PATH = os.path.join(RESULTS, "pandapower_comparison.csv")
HTML_PATH = os.path.join(RESULTS, "pandapower_comparison.html")
PNG_PATH = os.path.join(RESULTS, "pandapower_comparison.png")
SVG_PATH = os.path.join(RESULTS, "pandapower_comparison.svg")

# Shared publication palette (consistent with the pandapipes figure).
C_PANDAPOWER = "#d62728"  # dedicated reference (red)
C_MONEE = "#2ca02c"  # monee (green)
C_VM = "#1f77b4"  # voltage-agreement metric (blue)
C_P = "#ff7f0e"  # power-agreement metric (orange)
# The figure is transparent and embeds in both light and dark docs themes, so
# all text uses one mid-grey with at least 4:1 contrast on either background and
# saturated colour stays on the data bars only.
TEXT = "#737373"
GRID = "rgba(128,128,128,0.22)"
AXIS_LINE = "rgba(128,128,128,0.5)"


def _log_range(vals, floor=1e-12, lo_pad=0.3, hi_pad=25.0):
    """[log10 lo, log10 hi] range for a validation column, clipped to ``floor``
    since errors can be near zero, with headroom for the outside labels."""
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    v = np.clip(v, floor, None)
    if v.size == 0:
        return [np.log10(floor), 0.0]
    return [np.log10(v.min() * lo_pad), np.log10(v.max() * hi_pad)]


def make_plot(
    df: pd.DataFrame, out_html: str, out_png: str, out_svg: str | None = None
):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    color = {PANDAPOWER: C_PANDAPOWER, CASADI: C_MONEE}
    groups = [
        ("PF", "Power flow (AC): pandapower runpp vs monee CasADi"),
        ("OPF", "Optimal power flow, no line limit: pandapower runopp vs monee CasADi"),
        (
            "OPF-LL",
            "Optimal power flow, line-loading limit binds: pandapower runopp vs monee CasADi",
        ),
    ]
    groups = [g for g in groups if (df.group == g[0]).any()]

    # Row heights track each group's case count, and figure height tracks the
    # total, so every bar has the same on-screen thickness regardless of how
    # many cases a group has.
    counts = [int((df.group == g[0]).sum()) for g in groups]
    total_cases = max(sum(counts), 1)
    n_groups = len(groups)

    # Three columns: solve time, voltage agreement, power agreement. The two
    # agreement columns replace the old speedup bar; the story is that monee
    # reproduces pandapower's solution, not how much slower it is.
    fig = make_subplots(
        rows=n_groups,
        cols=3,
        column_widths=[0.46, 0.27, 0.27],
        row_heights=[c / total_cases for c in counts],
        shared_yaxes=True,  # case labels render once, on the far left only
        horizontal_spacing=0.06,
        vertical_spacing=0.12,
    )

    for r, (g, _title) in enumerate(groups, start=1):
        sub = df[df.group == g].iloc[::-1]
        cases = sub.case.tolist()
        last_row = r == n_groups

        # col 1: solve time (grouped, log)
        for col, backend in [("t_pandapower_ms", PANDAPOWER), ("t_casadi_ms", CASADI)]:
            fig.add_trace(
                go.Bar(
                    y=cases,
                    x=sub[col],
                    name=backend,
                    orientation="h",
                    marker_color=color[backend],
                    marker_line_width=0,
                    legendgroup=backend,
                    showlegend=(r == 1),
                    cliponaxis=False,
                    text=[f"{v:.0f}" for v in sub[col]],
                    textposition="outside",
                    textfont={"size": 19, "color": TEXT},
                    hovertemplate=f"{backend}: %{{x:.1f}} ms<extra></extra>",
                ),
                row=r,
                col=1,
            )
        tvals = sub[["t_pandapower_ms", "t_casadi_ms"]].to_numpy(dtype=float)
        fig.update_xaxes(
            type="log",
            row=r,
            col=1,
            title_text="solve time (ms, log)" if last_row else None,
            range=[np.log10(np.nanmin(tvals) * 0.5), np.log10(np.nanmax(tvals) * 3.4)],
        )

        # col 2: bus-voltage agreement |Δvm| (pu, log)
        vm = np.clip(sub.vm_err_pu.to_numpy(float), 1e-12, None)
        fig.add_trace(
            go.Bar(
                y=cases,
                x=vm,
                orientation="h",
                showlegend=False,
                marker_color=C_VM,
                marker_line_width=0,
                cliponaxis=False,
                text=[f"{v:.1e}" for v in vm],
                textposition="outside",
                textfont={"size": 19, "color": TEXT},
                hovertemplate="|Δvm| %{x:.2e} pu vs pandapower<extra></extra>",
            ),
            row=r,
            col=2,
        )
        fig.update_xaxes(
            type="log",
            row=r,
            col=2,
            title_text="|Δvm| vs pandapower (pu, log)" if last_row else None,
            range=_log_range(sub.vm_err_pu),
        )

        # col 3: slack-power agreement |ΔP| (MW, log)
        pw = np.clip(sub.p_err_mw.to_numpy(float), 1e-12, None)
        fig.add_trace(
            go.Bar(
                y=cases,
                x=pw,
                orientation="h",
                showlegend=False,
                marker_color=C_P,
                marker_line_width=0,
                cliponaxis=False,
                text=[f"{v:.1e}" for v in pw],
                textposition="outside",
                textfont={"size": 19, "color": TEXT},
                hovertemplate="|ΔP| %{x:.2e} MW vs pandapower<extra></extra>",
            ),
            row=r,
            col=3,
        )
        fig.update_xaxes(
            type="log",
            row=r,
            col=3,
            title_text="|ΔP_slack| vs pandapower (MW, log)" if last_row else None,
            range=_log_range(sub.p_err_mw),
        )

        # group banner over the left column, anchored to the subplot domain
        fig.add_annotation(
            text=f"<b>{_title}</b>",
            row=r,
            col=1,
            xref="x domain",
            yref="y domain",
            x=0,
            y=1.0,
            yshift=22,
            showarrow=False,
            font={"size": 23, "color": TEXT},
            xanchor="left",
        )

    # uniform axis cosmetics on every subplot
    fig.update_xaxes(
        showgrid=True,
        gridcolor=GRID,
        gridwidth=1,
        zeroline=False,
        showline=True,
        linecolor=AXIS_LINE,
        linewidth=1,
        ticks="outside",
        ticklen=4,
        tickcolor=AXIS_LINE,
        tickfont={"size": 20, "color": TEXT},
        title_font={"size": 22, "color": TEXT},
    )
    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        showline=False,
        tickfont={"size": 20, "color": TEXT},
        automargin=True,
    )

    fig.update_layout(
        title={
            "text": "<b>monee (electric, NLP) vs pandapower native AC solvers</b><br>"
            f"<span style='font-size:23px;color:{TEXT}'>identical grids via the "
            "MATPOWER exchange: solve time and solution agreement "
            "(voltage &amp; power)</span>",
            "x": 0.5,
            "xanchor": "center",
            "y": 0.978,
            "yanchor": "top",
            "font": {"size": 31, "color": TEXT},
        },
        barmode="group",
        bargap=0.3,
        bargroupgap=0.08,
        template="plotly_white",
        height=int(80 * total_cases + 110 * n_groups + 170),
        width=1280,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.012,
            "xanchor": "right",
            "x": 1.0,
            "font": {"size": 21, "color": TEXT},
            "bgcolor": "rgba(0,0,0,0)",
        },
        margin={"l": 235, "r": 60, "t": 215, "b": 70},
        font={"family": "Inter, Segoe UI, Helvetica, Arial", "size": 21, "color": TEXT},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
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
        raise SystemExit(
            f"No results CSV at {CSV_PATH}; run the benchmark once "
            "(without --plot-only) to generate it first."
        )
    df = pd.read_csv(CSV_PATH)
    print(f"Re-plotting from {CSV_PATH} ({len(df)} cases)")
    make_plot(df, HTML_PATH, PNG_PATH, SVG_PATH)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    # warm-up to keep import, licence, and JIT costs out of the timings
    import contextlib
    import io as _io

    with (
        contextlib.redirect_stdout(_io.StringIO()),
        contextlib.redirect_stderr(_io.StringIO()),
    ):
        try:
            pp.runpp(ppn.create_cigre_network_mv())
        except Exception:
            pass
        try:
            m = _mnet(ppn.create_cigre_network_mv)
            run_energy_flow(m, solver=CasADiSolver(), simulation=True)
        except Exception:
            pass

    rows = run_pf_suite() + run_opf_suite() + run_opf_limited_suite()
    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)

    print("\n=== monee vs pandapower (electric, NLP) ===\n")
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(df.to_string(index=False))
    print(
        f"\nmax voltage agreement error (vs pandapower): "
        f"{df['vm_err_pu'].to_numpy().max():.2e} pu"
    )
    print(
        f"max slack-power agreement error (vs pandapower): "
        f"{df['p_err_mw'].to_numpy().max():.2e} MW"
    )
    print(f"Wrote {CSV_PATH}")

    make_plot(df, HTML_PATH, PNG_PATH, SVG_PATH)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip all solves; only regenerate the plot from the existing CSV.",
    )
    args = parser.parse_args()
    if args.plot_only:
        regenerate_plot()
    else:
        main()

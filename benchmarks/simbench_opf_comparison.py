"""monee (electric, smooth-NLP) vs pandapower's AC-OPF across the whole SimBench
benchmark set.

Companion to ``pandapower_comparison.py``. Where that script compares the two
tools on a handful of hand-built feeders, this one runs the *same* AC optimal
power flow on **every** SimBench grid, to see how the comparison holds up across
a realistic spread of European distribution and transmission networks.

Grid selection (``simbench.collect_all_simbench_codes``):
    * every SimBench grid code, no exceptions, restricted to
    * the *no-switch* variant (``-no_sw``), the solvable single-busbar model, and
    * the most future load/generation scenario (``-2-``, year ~2034),
which leaves the 40 codes ending in ``-2-no_sw`` (LV/MV/HV/EHV feeders, the
mixed HVMV and MVLV combinations, and the full aggregated ``-all`` grids up to
~10k buses).

The OPF is one economic dispatch applied uniformly to every grid: curtailable,
reactive-capable distributed generators (``sgen``, cheap) backed by the external
grid slack (expensive), bus voltages bounded to [0.9, 1.1] pu, line/transformer
loading limits relaxed so the problem is a pure dispatch-plus-voltage OPF rather
than a base-case-infeasible one. pandapower solves it with ``pp.runopp`` (the
PYPOWER MIPS interior-point OPF); monee solves the identical case, imported
through the MATPOWER exchange, with its CasADi/IPOPT NLP backend.

Many grids do not solve on one tool or the other -- PYPOWER's OPF in particular
is fragile on meshed MV feeders -- which is the point: the three reported
dimensions are

    (1) average solve time   (seconds, over grids both tools solved)
    (2) success rate          (share of all 40 grids each tool solved)
    (3) objective delta       (|Δ cost| relative, over grids both tools solved)

Only the solve call is timed (``pp.runopp`` / ``run_energy_flow_optimization``);
every network is built and prepared before the timer starts. A per-solve wall
clock (``--timeout``) caps the largest grids; a solve that exceeds it counts as
a failure for that tool.

Outputs: ``results/simbench_opf_comparison.csv`` and a Plotly figure
(``results/simbench_opf_comparison.html`` + ``.png`` / ``.svg`` via kaleido).

Run:            python benchmarks/simbench_opf_comparison.py
Quick subset:   python benchmarks/simbench_opf_comparison.py --max-bus 1000
Plot only:      python benchmarks/simbench_opf_comparison.py --plot-only
"""

from __future__ import annotations

import ctypes
import os
import threading
import warnings

warnings.filterwarnings("ignore")

# PYPOWER (pandapower's AC-OPF engine) uses ``sparse.H``, removed in SciPy>=1.11.
# Restore it before any runopp call so the OPF benchmark works on modern SciPy.
import scipy.sparse as _sp

for _cls in (_sp.csc_matrix, _sp.csr_matrix, _sp.coo_matrix):
    if not hasattr(_cls, "H"):
        _cls.H = property(lambda self: self.conj().transpose())

import contextlib
import io
import tempfile
import time
import uuid

import numpy as np
import pandapower as pp
import pandas as pd
import scipy.io

# Reuse the shared results dir and the MATPOWER exporter resolved in the sibling
# benchmark (the benchmarks/ directory is on sys.path when run directly).
from backend_comparison import RESULTS, to_mpc

from monee import run_energy_flow_optimization
from monee.io.from_pandapower import _strip_transformer_vector_group_shifts
from monee.io.matpower import _mpc_from_mat, build_matpower_opf
from monee.solver.casadi import CasADiSolver

PANDAPOWER = "pandapower"
CASADI = "monee · CasADi"

# OPF setup, applied identically to every grid.
_VM_MIN, _VM_MAX = 0.9, 1.1  # bus voltage band (pu)
_SGEN_COST = 8.0  # €/MW for distributed generation (cheap, dispatched first)
_SLACK_COST = 40.0  # €/MW for external-grid import (expensive backstop)
_BIG = 1e4  # slack P/Q bounds (MW / MVAr), effectively unbounded
_LOOSE_LOADING = 1e4  # line/trafo loading cap (%), relaxed out of the way


class _Timeout(Exception):
    pass


@contextlib.contextmanager
def _time_limit(seconds):
    """Wall-clock cap for a single solve, identical on every platform.

    A watcher thread injects ``_Timeout`` into the solving (main) thread via
    ``PyThreadState_SetAsyncExc`` once *seconds* elapse; the caller records the
    overrun as a failure. Like a signal handler the exception is delivered at the
    next Python bytecode boundary, so a solve sitting in one long C call is capped
    when it next returns to Python. No leaked worker (the solve runs in the main
    thread); the watcher exits as soon as the block completes. ``seconds <= 0``
    disables the cap. Unlike the old SIGALRM/setitimer version this needs no
    Unix-only signal API, so it works the same on Windows, macOS and Linux.
    """
    if not seconds or seconds <= 0:
        yield
        return

    target_id = threading.get_ident()
    done = threading.Event()

    def _set_async_exc(exc):
        # exc=None clears any pending async exception on the target thread.
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(target_id), exc)

    def _watch():
        if done.wait(seconds):
            return  # block finished within the cap; do not fire
        _set_async_exc(ctypes.py_object(_Timeout))

    watcher = threading.Thread(target=_watch, daemon=True)
    watcher.start()
    try:
        yield
    finally:
        done.set()
        watcher.join()
        # If the watcher fired just as the block finished (lost race), drop the
        # not-yet-delivered _Timeout so it can't surface in unrelated later code.
        _set_async_exc(None)


@contextlib.contextmanager
def _silent():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        yield


def simbench_codes():
    """All SimBench grid codes, no-switch, most-future scenario (``-2-no_sw``)."""
    import simbench as sb

    return sorted(c for c in sb.collect_all_simbench_codes() if c.endswith("-2-no_sw"))


def _prepare_opf(net):
    """Turn a raw SimBench net into an economic-dispatch OPF, in place.

    Every distributed generator (``sgen``) becomes a curtailable, reactive-capable
    decision variable at a low marginal price; the external grid is the expensive
    slack. Bus voltages are bounded to [0.9, 1.1] pu and line/transformer loading
    limits are relaxed, so the OPF optimises dispatch and voltage rather than
    fighting a base case that already exceeds its thermal ratings.
    """
    net.bus["min_vm_pu"] = _VM_MIN
    net.bus["max_vm_pu"] = _VM_MAX
    net.line["max_loading_percent"] = _LOOSE_LOADING
    if len(net.trafo):
        net.trafo["max_loading_percent"] = _LOOSE_LOADING
    if len(net.trafo3w):
        net.trafo3w["max_loading_percent"] = _LOOSE_LOADING
    # Zero transformer vector-group phase shifts (Dyn5-style 150 degree shifts are
    # ubiquitous in SimBench MV/LV grids). A lone shift across a transformer makes
    # the flat-start AC-OPF collapse onto a spurious low-voltage root that *both*
    # solvers fail on -- PYPOWER's MIPS reports OPFNotConverged and monee's IPOPT
    # reports infeasible. Removing them here means pandapower runopp and the
    # to_mpc -> monee export both solve the identical shift-free problem, which is
    # what makes the cross-tool comparison fair (and lets either tool converge at
    # all on the distribution grids).
    if len(net.trafo):
        net.trafo["shift_degree"] = 0.0
    if len(net.trafo3w):
        for _col in ("shift_mv_degree", "shift_lv_degree"):
            if _col in net.trafo3w.columns:
                net.trafo3w[_col] = 0.0
    # The external grid is the dispatchable slack. pandapower 3.x requires a
    # proper boolean ``controllable`` column on every generator element (it does
    # ``~net.ext_grid.controllable`` while exporting), so set the whole column,
    # not just per-row, to avoid a NaN/float slipping through.
    net.ext_grid["controllable"] = True
    net.ext_grid["min_p_mw"] = -_BIG
    net.ext_grid["max_p_mw"] = _BIG
    net.ext_grid["min_q_mvar"] = -_BIG
    net.ext_grid["max_q_mvar"] = _BIG
    net.sgen["controllable"] = True
    # Any conventional generators are dispatchable too, priced between the cheap
    # distributed generation and the expensive slack.
    if len(net.gen):
        net.gen["controllable"] = True
        for i in net.gen.index:
            pp.create_poly_cost(
                net, i, "gen", cp1_eur_per_mw=0.5 * (_SGEN_COST + _SLACK_COST)
            )
    for i in net.sgen.index:
        p = abs(float(net.sgen.at[i, "p_mw"]))
        cap = max(p, 1e-3)
        net.sgen.at[i, "min_p_mw"] = 0.0
        net.sgen.at[i, "max_p_mw"] = cap
        net.sgen.at[i, "min_q_mvar"] = -cap
        net.sgen.at[i, "max_q_mvar"] = cap
        pp.create_poly_cost(net, i, "sgen", cp1_eur_per_mw=_SGEN_COST)
    pp.create_poly_cost(net, 0, "ext_grid", cp1_eur_per_mw=_SLACK_COST)
    return net


def _solve_pandapower(net, timeout):
    """Time ``pp.runopp`` on a prepared net. Returns (ok, seconds, cost)."""
    try:
        with _silent(), _time_limit(timeout):
            t0 = time.perf_counter()
            pp.runopp(net)
            dt = time.perf_counter() - t0
        return True, dt, float(net.res_cost)
    except (_Timeout, Exception):
        return False, float("nan"), float("nan")


def _build_monee_opf(net):
    """Export a prepared net through the MATPOWER bridge into monee's AC OPF.

    Built outside the timed region: only the solve is timed, matching the
    pandapower side and the sibling benchmarks.
    """
    tmp = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.mat")
    to_mpc(net, init="flat", filename=tmp)
    try:
        mpc = _mpc_from_mat(scipy.io.loadmat(tmp))
        # Line limits are relaxed on the pandapower side too, so the monee OPF is
        # the matching dispatch-plus-voltage problem (no line-loading constraint).
        network, problem = build_matpower_opf(mpc, max_loading=None)
        # Normalise transformer vector-group phase shifts that to_mpc exports
        # (most SimBench MV/LV grids carry Dyn5-style 150 degree trafo shifts). A
        # lone shift across one branch collapses the flat-start AC NLP onto a
        # spurious low-voltage root that IPOPT then reports as infeasible; the
        # sibling benchmarks strip these on the monee side too (see
        # from_pandapower_net / backend_comparison._mnet).
        _strip_transformer_vector_group_shifts(network)
        return network, problem
    finally:
        os.remove(tmp)


def _solve_monee(net, timeout):
    """Time monee's CasADi/IPOPT OPF on a prepared net. Returns (ok, s, obj)."""
    try:
        mc, prob = _build_monee_opf(net)
    except Exception:
        return False, float("nan"), float("nan")
    try:
        with _silent(), _time_limit(timeout):
            t0 = time.perf_counter()
            res = run_energy_flow_optimization(mc, prob, solver=CasADiSolver())
            dt = time.perf_counter() - t0
        if not bool(res.success):
            return False, dt, float("nan")
        return True, dt, float(res.objective)
    except (_Timeout, Exception):
        return False, float("nan"), float("nan")


def run_suite(max_bus=None, timeout=240.0):
    """Run the OPF on every selected SimBench grid, both tools. Returns rows."""
    import simbench as sb

    codes = simbench_codes()
    print(
        f"\n--- SimBench AC-OPF: pandapower runopp vs monee CasADi ({len(codes)} grids) ---"
    )
    rows = []
    for k, code in enumerate(codes, 1):
        try:
            net = sb.get_simbench_net(code)
        except Exception as exc:
            print(f"  [{k:2d}/{len(codes)}] {code:30s} load failed: {exc}")
            continue
        n_bus = int(len(net.bus))
        if max_bus is not None and n_bus > max_bus:
            print(
                f"  [{k:2d}/{len(codes)}] {code:30s} skipped (n_bus {n_bus} > {max_bus})"
            )
            continue

        # Prepare independently for each tool (runopp mutates the net in place).
        pp_ok, t_pp, cost_pp = _solve_pandapower(_prepare_opf(net), timeout)
        mn_ok, t_mn, obj_mn = _solve_monee(
            _prepare_opf(sb.get_simbench_net(code)), timeout
        )

        both = pp_ok and mn_ok
        obj_rel = (
            abs(obj_mn - cost_pp) / max(abs(cost_pp), 1.0) if both else float("nan")
        )
        rows.append(
            dict(
                code=code,
                grid=code[2:-9],  # strip the constant "1-" prefix and "-2-no_sw"
                n_bus=n_bus,
                pp_ok=pp_ok,
                mn_ok=mn_ok,
                t_pandapower_s=round(t_pp, 4) if pp_ok else float("nan"),
                t_casadi_s=round(t_mn, 4) if mn_ok else float("nan"),
                cost_pandapower=cost_pp if pp_ok else float("nan"),
                obj_casadi=obj_mn if mn_ok else float("nan"),
                obj_rel_err=obj_rel,
            )
        )
        r = rows[-1]
        print(
            f"  [{k:2d}/{len(codes)}] {r['grid']:24s} n_bus {n_bus:5d}  "
            f"pp {'OK ' if pp_ok else 'FAIL'} {t_pp:7.2f}s  "
            f"monee {'OK ' if mn_ok else 'FAIL'} {t_mn:7.2f}s  "
            + (f"|d_obj| {obj_rel:.2e}" if both else "")
        )
    return rows


# Plot, same visual language as the sibling pandapower_comparison.py figure.
CSV_PATH = os.path.join(RESULTS, "simbench_opf_comparison.csv")
HTML_PATH = os.path.join(RESULTS, "simbench_opf_comparison.html")
PNG_PATH = os.path.join(RESULTS, "simbench_opf_comparison.png")
SVG_PATH = os.path.join(RESULTS, "simbench_opf_comparison.svg")

C_PANDAPOWER = "#d62728"  # dedicated reference (red)
C_MONEE = "#2ca02c"  # monee (green)
C_DELTA = "#ff7f0e"  # objective-agreement metric (orange)
TEXT = "#737373"
GRID = "rgba(128,128,128,0.22)"
AXIS_LINE = "rgba(128,128,128,0.5)"


_DEFAULT_TITLE = (
    "<b>monee (electric, NLP) vs pandapower AC-OPF across all SimBench grids</b><br>"
    "<span style='font-size:20px;color:{text}'>40 grids "
    "(no-switch, future scenario 2) via the MATPOWER exchange: "
    "solve time, success rate, and objective agreement</span>"
)


def make_plot(df: pd.DataFrame, out_html, out_png, out_svg=None, title_text=None):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    n_total = int(len(df))
    both = df[df.pp_ok & df.mn_ok]
    n_both = int(len(both))

    # (1) average solve time over the grids both tools solved (fair head-to-head)
    avg_t_pp = float(both.t_pandapower_s.mean()) if n_both else float("nan")
    avg_t_mn = float(both.t_casadi_s.mean()) if n_both else float("nan")
    # (2) success rate over all selected grids
    rate_pp = 100.0 * float(df.pp_ok.mean()) if n_total else 0.0
    rate_mn = 100.0 * float(df.mn_ok.mean()) if n_total else 0.0

    fig = make_subplots(
        rows=2,
        cols=2,
        row_heights=[0.32, 0.68],
        specs=[
            [{}, {}],
            [{"colspan": 2}, None],
        ],
        vertical_spacing=0.16,
        horizontal_spacing=0.14,
        subplot_titles=(
            f"<b>Average solve time</b>  (both solved, n={n_both})",
            f"<b>Success rate</b>  (all grids, n={n_total})",
            f"<b>Objective delta</b>  |Δ cost| relative, per grid (both solved, n={n_both})",
        ),
    )

    # (1) average solve time
    fig.add_trace(
        go.Bar(
            x=[PANDAPOWER, CASADI],
            y=[avg_t_pp, avg_t_mn],
            marker_color=[C_PANDAPOWER, C_MONEE],
            marker_line_width=0,
            text=[f"{avg_t_pp:.2f}s", f"{avg_t_mn:.2f}s"],
            textposition="outside",
            textfont={"size": 19, "color": TEXT},
            showlegend=False,
            cliponaxis=False,
            hovertemplate="%{x}: %{y:.3f} s<extra></extra>",
        ),
        row=1,
        col=1,
    )
    tmax = np.nanmax([avg_t_pp, avg_t_mn, 1e-9])
    fig.update_yaxes(title_text="seconds", range=[0, tmax * 1.25], row=1, col=1)

    # (2) success rate
    fig.add_trace(
        go.Bar(
            x=[PANDAPOWER, CASADI],
            y=[rate_pp, rate_mn],
            marker_color=[C_PANDAPOWER, C_MONEE],
            marker_line_width=0,
            text=[f"{rate_pp:.0f}%", f"{rate_mn:.0f}%"],
            textposition="outside",
            textfont={"size": 19, "color": TEXT},
            showlegend=False,
            cliponaxis=False,
            hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.update_yaxes(title_text="% of grids solved", range=[0, 112], row=1, col=2)

    # (3) per-grid objective delta over the both-solved set, sorted worst-first
    b = both.sort_values("obj_rel_err")
    labels = b.grid.tolist()
    vals = np.clip(b.obj_rel_err.to_numpy(float), 1e-12, None)
    fig.add_trace(
        go.Bar(
            y=labels,
            x=vals,
            orientation="h",
            marker_color=C_DELTA,
            marker_line_width=0,
            showlegend=False,
            cliponaxis=False,
            text=[f"{v:.1e}" for v in vals],
            textposition="outside",
            textfont={"size": 15, "color": TEXT},
            hovertemplate="%{y}: |Δobj| %{x:.2e}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    if len(vals):
        fig.update_xaxes(
            type="log",
            title_text="|Δ objective| relative to pandapower (log)",
            range=[np.log10(vals.min() * 0.3), np.log10(vals.max() * 25)],
            row=2,
            col=1,
        )

    fig.update_xaxes(
        showgrid=True,
        gridcolor=GRID,
        zeroline=False,
        showline=True,
        linecolor=AXIS_LINE,
        ticks="outside",
        ticklen=4,
        tickcolor=AXIS_LINE,
        tickfont={"size": 17, "color": TEXT},
        title_font={"size": 19, "color": TEXT},
    )
    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        showline=False,
        tickfont={"size": 17, "color": TEXT},
        automargin=True,
        title_font={"size": 19, "color": TEXT},
    )
    for ann in fig.layout.annotations:
        ann.font.size = 21
        ann.font.color = TEXT

    fig.update_layout(
        title={
            "text": (title_text or _DEFAULT_TITLE).format(text=TEXT),
            "x": 0.5,
            "xanchor": "center",
            "y": 0.975,
            "yanchor": "top",
            "font": {"size": 27, "color": TEXT},
        },
        template="plotly_white",
        height=max(620, 70 + 22 * max(n_both, 1) + 260),
        width=1280,
        bargap=0.45,
        margin={"l": 220, "r": 70, "t": 150, "b": 70},
        font={"family": "Inter, Segoe UI, Helvetica, Arial", "size": 19, "color": TEXT},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        uniformtext={"mode": "hide", "minsize": 9},
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
    print(f"Re-plotting from {CSV_PATH} ({len(df)} grids)")
    make_plot(df, HTML_PATH, PNG_PATH, SVG_PATH)


def main(max_bus=None, timeout=240.0):
    os.makedirs(RESULTS, exist_ok=True)
    import simbench as sb

    # Warm up each solver once on a small grid to keep import / licence / JIT
    # costs out of the timings.
    with _silent():
        try:
            warm = _prepare_opf(sb.get_simbench_net("1-LV-rural1--2-no_sw"))
            pp.runopp(warm)
        except Exception:
            pass
        try:
            mc, prob = _build_monee_opf(
                _prepare_opf(sb.get_simbench_net("1-LV-rural1--2-no_sw"))
            )
            run_energy_flow_optimization(mc, prob, solver=CasADiSolver())
        except Exception:
            pass

    rows = run_suite(max_bus=max_bus, timeout=timeout)
    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)

    print("\n=== monee vs pandapower AC-OPF on SimBench ===\n")
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(
            df[
                [
                    "grid",
                    "n_bus",
                    "pp_ok",
                    "t_pandapower_s",
                    "mn_ok",
                    "t_casadi_s",
                    "obj_rel_err",
                ]
            ].to_string(index=False)
        )
    both = df[df.pp_ok & df.mn_ok]
    print(
        f"\npandapower solved {int(df.pp_ok.sum())}/{len(df)}  "
        f"({100 * df.pp_ok.mean():.0f}%),  "
        f"monee solved {int(df.mn_ok.sum())}/{len(df)}  "
        f"({100 * df.mn_ok.mean():.0f}%)"
    )
    if len(both):
        print(
            f"on the {len(both)} both solved: "
            f"avg time pp {both.t_pandapower_s.mean():.2f}s "
            f"monee {both.t_casadi_s.mean():.2f}s,  "
            f"max |d_obj| rel {both.obj_rel_err.max():.2e}"
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
    parser.add_argument(
        "--max-bus",
        type=int,
        default=None,
        help="Skip grids larger than this many buses (quick subset run).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=240.0,
        help="Per-solve wall-clock cap (s); an overrun counts as a failure. "
        "0 disables.",
    )
    args = parser.parse_args()
    if args.plot_only:
        regenerate_plot()
    else:
        main(max_bus=args.max_bus, timeout=args.timeout)

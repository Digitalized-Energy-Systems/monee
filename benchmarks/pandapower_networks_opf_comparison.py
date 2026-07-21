"""monee (electric, smooth-NLP) vs pandapower's AC-OPF across the networks that
ship *inside* pandapower.

Sibling of ``simbench_opf_comparison.py``. That script runs the head-to-head
AC optimal power flow on every SimBench grid; this one runs the *same*
comparison, the *same* way (``pp.runopp`` vs monee's CasADi/IPOPT NLP through the
MATPOWER exchange), on every network bundled in ``pandapower.networks`` - the
MATPOWER/PYPOWER ``caseNN`` library plus the CIGRE, Kerber, oberrhein, GB,
iceland and example test feeders.

OPF setup, per network:
    * If the bundled network *already carries OPF data* (a ``poly_cost`` /
      ``pwl_cost`` table - true of the ``caseNN`` MATPOWER cases, GB, iceland,
      illinois, ...), it is kept verbatim: its generator costs, PMIN/PMAX/QMIN/
      QMAX limits, bus VMIN/VMAX band and branch RATE_A line ratings are exactly
      the case as published. The monee side then *enforces those same line
      ratings* (``max_loading=1`` against the exported RATE_A), so both tools
      solve the identical constrained OPF.
    * Otherwise (the CIGRE/Kerber/oberrhein/example feeders, and the
      power-flow-only ``case4gs`` / ``case11_iwamoto``) the *same* economic
      dispatch as the SimBench benchmark is applied via ``_prepare_opf``:
      curtailable reactive-capable ``sgen`` (cheap) backed by the external grid
      slack (expensive), bus voltages in [0.9, 1.1] pu, and line/trafo loading
      relaxed out of the way - so monee drops line limits (``max_loading=None``)
      to match, exactly as in the SimBench run.

The three reported dimensions are the same as the SimBench benchmark:

    (1) average solve time   (seconds, over networks both tools solved)
    (2) success rate          (share of all networks each tool solved)
    (3) objective delta       (|Δ cost| relative, over networks both solved)

Only the solve call is timed; every network is built and prepared before the
timer starts. A per-solve wall clock (``--timeout``) caps the largest cases.

Outputs: ``results/pandapower_networks_opf_comparison.csv`` and a Plotly figure
(``results/pandapower_networks_opf_comparison.html`` + ``.png`` / ``.svg``).

Run:            python benchmarks/pandapower_networks_opf_comparison.py
Quick subset:   python benchmarks/pandapower_networks_opf_comparison.py --max-bus 500
Plot only:      python benchmarks/pandapower_networks_opf_comparison.py --plot-only
"""

from __future__ import annotations

import inspect
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
import time
import uuid

import pandas as pd

# Reuse the shared results dir, the MATPOWER exporter, and every piece of the
# SimBench OPF benchmark that is network-source agnostic: the economic-dispatch
# preparation, the pandapower solve+time helper, the timeout/silence context
# managers, and the plotting routine (with a benchmark-specific title).
from backend_comparison import RESULTS, to_mpc
from simbench_opf_comparison import (
    _prepare_opf,
    _silent,
    _solve_pandapower,
    _time_limit,
    _Timeout,
    make_plot,
)

from monee import run_energy_flow_optimization
from monee.io.from_pandapower import strip_transformer_vector_group_shifts
from monee.io.matpower import build_matpower_opf, read_mpc
from monee.solver.casadi import CasADiSolver

PANDAPOWER = "pandapower"
CASADI = "monee · CasADi"


def _extra_network_builders(ppn):
    """Documented network *variants* that take a non-default argument and so are
    not reachable by the zero-argument discovery below.

    The CIGRE MV medium-voltage feeder ships in three published flavours via
    ``with_der``: no distributed generation (the zero-arg default, already
    discovered), ``"pv_wind"`` (9 DER sgens) and ``"all"`` (13 DER sgens). The
    two DER variants are distinct grids on the CIGRE networks page, so add them.
    """
    return [
        (
            "create_cigre_network_mv(with_der=pv_wind)",
            lambda: ppn.create_cigre_network_mv(with_der="pv_wind"),
        ),
        (
            "create_cigre_network_mv(with_der=all)",
            lambda: ppn.create_cigre_network_mv(with_der="all"),
        ),
    ]


def pandapower_network_builders():
    """Every network bundled in ``pandapower.networks``.

    Discovers all zero-argument builders (the full power-system test-case
    library - ``caseNN``, GB, iceland, illinois - plus the CIGRE/Kerber/
    oberrhein/example feeders) and adds the documented parameterized variants
    from :func:`_extra_network_builders` (the CIGRE MV DER grids). The run loop
    builds each and discards anything that is not a real network (helper
    utilities, ``create_empty_network``, stray re-exports).
    """
    import pandapower.networks as ppn

    builders = {}
    for name in dir(ppn):
        if name.startswith("_"):
            continue
        fn = getattr(ppn, name)
        if not callable(fn):
            continue
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            continue
        required = [
            p
            for p in sig.parameters.values()
            if p.default is inspect.Parameter.empty
            and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        if required:
            continue
        builders[name] = fn
    for name, fn in _extra_network_builders(ppn):
        builders.setdefault(name, fn)
    return sorted(builders.items(), key=lambda nf: nf[0])


def _has_opf_data(net):
    """A network carries OPF data when it already defines generator costs."""
    return len(net.poly_cost) > 0 or len(net.pwl_cost) > 0


def _prepared(net, has_opf):
    """Keep a network that already has OPF data; otherwise apply the SimBench
    economic-dispatch setup (in place)."""
    return net if has_opf else _prepare_opf(net)


def _build_monee_opf(net, max_loading):
    """Export a prepared net through the MATPOWER bridge into monee's AC OPF.

    ``max_loading`` mirrors how line limits are treated on the pandapower side:
    ``1`` enforces the bundled RATE_A ratings (networks that keep their own OPF
    data), ``None`` drops line limits (networks given the relaxed SimBench
    dispatch setup). Built outside the timed region.
    """
    tmp = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.mat")
    to_mpc(net, init="flat", filename=tmp)
    try:
        mpc = read_mpc(tmp)
        network, problem = build_matpower_opf(mpc, max_loading=max_loading)
        # Normalise transformer vector-group phase shifts (see the SimBench
        # benchmark / from_pandapower) so the flat-start AC NLP does not collapse.
        strip_transformer_vector_group_shifts(network)
        return network, problem
    finally:
        os.remove(tmp)


def _solve_monee(net, timeout, max_loading):
    """Time monee's CasADi/IPOPT OPF on a prepared net. Returns (ok, s, obj)."""
    try:
        mc, prob = _build_monee_opf(net, max_loading)
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
    """Run the OPF on every bundled pandapower network, both tools. Returns rows."""
    builders = pandapower_network_builders()
    print(
        f"\n--- pandapower networks AC-OPF: pandapower runopp vs monee CasADi "
        f"({len(builders)} candidates) ---"
    )
    rows = []
    for k, (name, fn) in enumerate(builders, 1):
        try:
            net = fn()
        except Exception as exc:
            print(f"  [{k:2d}/{len(builders)}] {name:34s} build failed: {exc}")
            continue
        if not hasattr(net, "bus") or len(net.bus) == 0:
            continue  # not a real network (helper / empty re-export)
        n_bus = int(len(net.bus))
        if max_bus is not None and n_bus > max_bus:
            print(
                f"  [{k:2d}/{len(builders)}] {name:34s} skipped (n_bus {n_bus} > {max_bus})"
            )
            continue

        has_opf = _has_opf_data(net)
        # Keep the bundled line ratings on the monee side (max_loading=1) when the
        # network keeps its own OPF data; drop them (None) for the relaxed
        # SimBench dispatch setup, exactly as the SimBench benchmark does.
        max_loading = 1.0 if has_opf else None

        # Prepare independently for each tool (runopp mutates the net in place).
        pp_ok, t_pp, cost_pp = _solve_pandapower(_prepared(net, has_opf), timeout)
        mn_ok, t_mn, obj_mn = _solve_monee(
            _prepared(fn(), has_opf), timeout, max_loading
        )

        both = pp_ok and mn_ok
        obj_rel = (
            abs(obj_mn - cost_pp) / max(abs(cost_pp), 1.0) if both else float("nan")
        )
        rows.append(
            dict(
                code=name,
                grid=name,
                n_bus=n_bus,
                has_opf_data=has_opf,
                pp_ok=pp_ok,
                mn_ok=mn_ok,
                t_pandapower_s=round(t_pp, 4) if pp_ok else float("nan"),
                t_casadi_s=round(t_mn, 4) if mn_ok else float("nan"),
                cost_pandapower=cost_pp if pp_ok else float("nan"),
                obj_casadi=obj_mn if mn_ok else float("nan"),
                obj_rel_err=obj_rel,
            )
        )
        print(
            f"  [{k:2d}/{len(builders)}] {name:34s} n_bus {n_bus:5d}  "
            f"{'opf ' if has_opf else 'disp'}  "
            f"pp {'OK ' if pp_ok else 'FAIL'} {t_pp:7.2f}s  "
            f"monee {'OK ' if mn_ok else 'FAIL'} {t_mn:7.2f}s  "
            + (f"|d_obj| {obj_rel:.2e}" if both else "")
        )
    return rows


CSV_PATH = os.path.join(RESULTS, "pandapower_networks_opf_comparison.csv")
HTML_PATH = os.path.join(RESULTS, "pandapower_networks_opf_comparison.html")
PNG_PATH = os.path.join(RESULTS, "pandapower_networks_opf_comparison.png")
SVG_PATH = os.path.join(RESULTS, "pandapower_networks_opf_comparison.svg")


def _title(n_total):
    # The make_plot helper fills the ``{text}`` colour placeholder; everything
    # else is resolved here (``{{text}}`` stays literal through the f-string).
    return (
        "<b>monee (electric, NLP) vs pandapower AC-OPF across pandapower's "
        "bundled networks</b><br>"
        f"<span style='font-size:20px;color:{{text}}'>{n_total} networks "
        "(MATPOWER cases keep their bundled OPF data and line limits; the other "
        "feeders get the SimBench dispatch+voltage setup) via the MATPOWER "
        "exchange: solve time, success rate, and objective agreement</span>"
    )


def regenerate_plot():
    if not os.path.exists(CSV_PATH):
        raise SystemExit(
            f"No results CSV at {CSV_PATH}; run the benchmark once "
            "(without --plot-only) to generate it first."
        )
    df = pd.read_csv(CSV_PATH)
    print(f"Re-plotting from {CSV_PATH} ({len(df)} networks)")
    make_plot(df, HTML_PATH, PNG_PATH, SVG_PATH, title_text=_title(len(df)))


def main(max_bus=None, timeout=240.0):
    os.makedirs(RESULTS, exist_ok=True)
    import pandapower as pp
    import pandapower.networks as ppn

    # Warm up each solver once (import / licence / JIT costs out of the timings),
    # covering both paths: a bundled-OPF case and a relaxed-dispatch feeder.
    with _silent():
        try:
            pp.runopp(ppn.case9())
        except Exception:
            pass
        try:
            mc, prob = _build_monee_opf(ppn.case9(), 1.0)
            run_energy_flow_optimization(mc, prob, solver=CasADiSolver())
        except Exception:
            pass

    rows = run_suite(max_bus=max_bus, timeout=timeout)
    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)

    print("\n=== monee vs pandapower AC-OPF on pandapower's bundled networks ===\n")
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(
            df[
                [
                    "grid",
                    "n_bus",
                    "has_opf_data",
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

    make_plot(df, HTML_PATH, PNG_PATH, SVG_PATH, title_text=_title(len(df)))


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
        help="Skip networks larger than this many buses (quick subset run).",
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

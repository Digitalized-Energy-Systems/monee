"""Storage capabilities in a multi-energy grid (MES).

Companion to ``tests/network/test_mes.py::test_generate_mes_storage_capabilities_timeseries``.
Builds the canonical MES used by the min-load-shedding test and runs it
twice over a short timeseries:

* once with :class:`~monee.model.formulation.linepack.GasLinepack` and
  :class:`~monee.model.formulation.ltc.LumpedThermalCapacitance` attached, and
* once without either extension as the baseline.

The same simbench load profile drives the underlying power loads (now
correctly bound by aggregated load name); a synthetic per-step swing on
the heat loads and gas sinks gives the storage extensions a varying
signal to react to.  The script then plots the aggregate differences
side-by-side and saves the figure next to this file.

The simbench rural network is supply-rich relative to its demand, so
``ext_grid`` bounds wide enough for the optimiser leave only a small
absolute storage response — but the response is unambiguous: aggregate
``|net_pack_kgs|`` is strictly zero without the extension (steady-state
constraint) and non-zero with it (inter-temporal mass balance).  The
plots zoom into that regime; for a high-utilisation system the same
metrics scale up correspondingly.

Requires: simbench, pandapower, matplotlib, gurobi (commercial solver).

Run::

    python examples/mes_storage_capabilities.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import simbench

import monee.model as mm
from monee import run_timeseries
from monee.io.from_pandapower import from_pandapower_net
from monee.io.from_simbench import obtain_simbench_profile_by_pp_net
from monee.model import GasLinepack, LumpedThermalCapacitance
from monee.model.core import value as mvalue
from monee.model.formulation import (
    MISOCP_NETWORK_FORMULATION,
    make_nl_weymouth_pwl_network_formulation,
)
from monee.network import generate_supply_return_mes_based_on_power_net
from monee.problem.min_load_shedding import create_min_load_shedding_problem
from monee.simulation.timeseries import TimeseriesData

DEMAND_FACTORS = [
    0.5,
    1.5,
    0.8,
    0.5,
    1.5,
    1.8,
    1.8,
    1.8,
    1.8,
    1.8,
    1.8,
    1.8,
    0.8,
    0.8,
    0.8,
    0.8,
    0.8,
]
STEPS = len(DEMAND_FACTORS)
# Same shape as the test (low / peak / recovery), slightly amplified to
# make the storage response easier to read off the plot.


def build_scenario(*, with_storage: bool):
    """Construct the canonical MES + the per-step demand timeseries.

    The build is identical to the one in the test, so both runs share the
    same component ids: that's what makes the with/without comparison
    meaningful at the per-component level.
    """
    pp_net = simbench.get_simbench_net("1-LV-rural3--1-no_sw")
    full_el_td = obtain_simbench_profile_by_pp_net(pp_net)
    mn = from_pandapower_net(pp_net)
    mes = generate_supply_return_mes_based_on_power_net(
        mn,
        coupling_density=0.5,
        centralized=False,
        couplings=("chp", "p2g", "p2h"),
        coupling_kwargs={"seed": 1, "use_hg_variants": True},
        heat_kwargs={"node_based_heat_loads": True},
    )
    mes.apply_formulation(MISOCP_NETWORK_FORMULATION)
    # mes.apply_formulation(make_mccormick_dhs_formulation(num_partitions=20))
    mes.apply_formulation(make_nl_weymouth_pwl_network_formulation())

    if with_storage:
        mes.add_extension(GasLinepack())
        mes.add_extension(LumpedThermalCapacitance(first_step_steady_state=True))

    td = TimeseriesData()
    for name, attrs in full_el_td.child_name_data.items():
        for attr, series in attrs.items():
            td.add_child_series_by_name(name, attr, list(series[:STEPS]))
    for c in mes.childs:
        if isinstance(c.model, mm.HeatLoad):
            base = float(mvalue(c.model.q_mw_heat))
            if base == 0:
                continue
            td.add_child_series(c.id, "q_mw_heat", [base * f for f in DEMAND_FACTORS])
        elif isinstance(c.model, mm.Sink):
            base = float(mvalue(c.model.mass_flow))
            if base == 0:
                continue
            td.add_child_series(c.id, "mass_flow", [base * f for f in DEMAND_FACTORS])
    return mes, td


def solve_scenario(*, with_storage: bool):
    mes, td = build_scenario(with_storage=with_storage)
    problem = create_min_load_shedding_problem(
        bounds_el=(0.9, 1.5),
        bounds_gas=(0.9, 1.5),
        bounds_heat=(0.7, 1.3),
        ext_grid_el_bounds=(-5, 5),
        ext_grid_gas_bounds=(-5, 5),
        ext_grid_heat_bounds=(-100, 100),
        include_ext_grids=True,
    )
    ts = run_timeseries(
        mes,
        timeseries_data=td,
        steps=STEPS,
        optimization_problem=problem,
        solver="gurobi",
    )
    return mes, ts


def water_junctions(mes):
    return [
        n
        for n in mes.nodes
        if isinstance(n.model, mm.Junction) and isinstance(n.grid, mm.WaterGrid)
    ]


def gas_pipes(mes):
    return [b for b in mes.branches if isinstance(b.model, mm.GasPipe)]


def mean_water_t_pu(mes, ts):
    """Per-step mean junction temperature across all water junctions [pu]."""
    juncs = water_junctions(mes)
    out = []
    for s in range(STEPS):
        vals = [
            float(ts.get_result_for_id(n.id, "t_pu").iloc[s])
            for n in juncs
            if ts.get_result_for_id(n.id, "t_pu").iloc[s] is not None
        ]
        out.append(sum(vals) / len(vals) if vals else float("nan"))
    return out


def t_pu_max_gap(mes, ts_with, ts_wo):
    """Per-step maximum |Δt_pu| across junctions (with − without) [pu]."""
    juncs = water_junctions(mes)
    out = []
    for s in range(STEPS):
        gaps = []
        for n in juncs:
            a = ts_with.get_result_for_id(n.id, "t_pu").iloc[s]
            b = ts_wo.get_result_for_id(n.id, "t_pu").iloc[s]
            if a is None or b is None:
                continue
            gaps.append(abs(float(a) - float(b)))
        out.append(max(gaps) if gaps else float("nan"))
    return out


def aggregate_pack_activity(mes, ts):
    """Per-step Σ |net_pack_kgs| across all gas pipes [kg/s].

    Strictly zero by construction in the no-extension run (the
    extension-injected variable doesn't exist) and a direct measure of
    inter-temporal storage activity in the with-extension run.
    """
    pipes = gas_pipes(mes)
    out = []
    for s in range(STEPS):
        total = 0.0
        for p in pipes:
            v = ts.get_result_for_id(p.id, "net_pack_kgs").iloc[s]
            if v is not None:
                total += abs(float(v))
        out.append(total)
    return out


def linepack_relative_deltas(mes, ts):
    """Per-pipe (linepack(t) − linepack(0)) / linepack(0) over time [ppm].

    Returns the trajectory of the top-5 pipes by swing magnitude so the
    plot stays readable.
    """
    pipes = gas_pipes(mes)
    rows = []
    for p in pipes:
        lp = ts.get_result_for_id(p.id, "linepack_kg")
        if lp.iloc[0] is None or float(lp.iloc[0]) == 0:
            continue
        lp0 = float(lp.iloc[0])
        deltas_ppm = [(float(v) - lp0) / lp0 * 1e6 for v in lp.values]
        swing = max(deltas_ppm) - min(deltas_ppm)
        rows.append((swing, p.id, deltas_ppm))
    rows.sort(reverse=True, key=lambda r: r[0])
    return rows[:5]


# Dissertation-grade matplotlib defaults — applied once at module import.
# Serif typography, mathtext via Computer Modern, hairline grid, no top/right
# spines, vector-friendly DPI for both PDF and PNG outputs.
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": [
            "Computer Modern Roman",
            "DejaVu Serif",
            "Times New Roman",
            "Times",
        ],
        "mathtext.fontset": "cm",
        "font.size": 10,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.titlepad": 8,
        "axes.labelsize": 10,
        "axes.labelpad": 4,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.linestyle": "-",
        "grid.linewidth": 0.4,
        "grid.alpha": 0.35,
        "lines.linewidth": 1.6,
        "lines.markersize": 4,
        "legend.fontsize": 8.5,
        "legend.frameon": True,
        "legend.fancybox": False,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.6",
        "legend.borderpad": 0.4,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.0,
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "pdf.fonttype": 42,  # embed TrueType, not Type-3 — required by most
        "ps.fonttype": 42,  # journal / dissertation submission systems
    }
)


# Paul Tol "vibrant" palette — colourblind-safe, prints well in greyscale.
C_WITH = "#0077BB"  # blue  — with-extension series
C_WITHOUT = "#CC3311"  # red   — without-extension series
C_DEMAND = "#EE7733"  # orange — demand modulation
C_LP = "#009988"  # teal  — linepack / Δt_pu overlay


def make_plot(
    tpu_with,
    tpu_wo,
    tpu_gap,
    pack_with,
    pack_wo,
    lp_traces,
    out_path: Path,
):
    """Render the 2×2 dissertation-grade summary figure.

    Layout::

        ┌────────────────────────┬────────────────────────┐
        │ (a) Demand modulation  │ (b) Linepack activity  │
        ├────────────────────────┼────────────────────────┤
        │ (c) Junction temperat. │ (d) Per-pipe linepack  │
        └────────────────────────┴────────────────────────┘

    Each subplot is independently labelled (a)–(d) per the dissertation
    convention; the overall figure caption belongs in the surrounding LaTeX
    document, not in the suptitle.

    The figure is saved to *out_path* and to a sibling ``.pdf`` for vector
    embedding in print.
    """
    steps = list(range(STEPS))

    # 6.5 in column width is standard for single-column dissertation figures
    # in 11 pt LaTeX classes; the 0.78 aspect ratio keeps a 2×2 grid balanced.
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.4))
    (ax_a, ax_b), (ax_c, ax_d) = axes

    # ── (a) Demand modulation profile ──────────────────────────────────────
    ax_a.step(steps, DEMAND_FACTORS, where="post", lw=1.4, color=C_DEMAND)
    ax_a.fill_between(
        steps,
        0,
        DEMAND_FACTORS,
        step="post",
        color=C_DEMAND,
        alpha=0.18,
        linewidth=0,
    )
    ax_a.set_ylabel(r"Demand factor $\phi_t$  [$\times$ nom.]")
    ax_a.set_title("(a) Synthetic gas/heat demand modulation", loc="left")
    ax_a.set_ylim(0, max(DEMAND_FACTORS) + 0.25)
    ax_a.set_xlim(-0.5, STEPS - 0.5)
    ax_a.set_xticks(steps[:: max(1, STEPS // 8)])

    # ── (b) Aggregated gas-linepack activity ───────────────────────────────
    # Σ |net_pack_kgs| in mg/s — strictly zero without the extension.
    pack_with_mg = [v * 1e6 for v in pack_with]
    pack_wo_mg = [v * 1e6 for v in pack_wo]
    bar_w = 0.36
    ax_b.bar(
        [s - bar_w / 2 for s in steps],
        pack_with_mg,
        width=bar_w,
        color=C_WITH,
        edgecolor="white",
        linewidth=0.4,
        label="with linepack",
    )
    ax_b.bar(
        [s + bar_w / 2 for s in steps],
        pack_wo_mg,
        width=bar_w,
        color=C_WITHOUT,
        edgecolor="white",
        linewidth=0.4,
        label=r"w/o linepack ($\equiv 0$)",
        hatch="///",
    )
    ax_b.set_ylabel(r"$\sum_p |\dot{m}_{\mathrm{pack},p}|$  [mg s$^{-1}$]")
    ax_b.set_title("(b) Inter-temporal gas-storage activity", loc="left")
    ax_b.legend(loc="upper right")
    ax_b.set_xlim(-0.5, STEPS - 0.5)
    ax_b.set_xticks(steps[:: max(1, STEPS // 8)])

    # ── (c) Mean water-junction temperature ─────────────────────────────────
    ax_c.plot(
        steps,
        tpu_with,
        marker="o",
        lw=1.6,
        color=C_WITH,
        label="with LTC",
        markeredgecolor="white",
        markeredgewidth=0.5,
    )
    ax_c.plot(
        steps,
        tpu_wo,
        marker="s",
        lw=1.4,
        color=C_WITHOUT,
        ls="--",
        label="without LTC",
        markeredgecolor="white",
        markeredgewidth=0.5,
    )
    ax_c.set_ylabel(r"$\bar{T}_{\mathrm{water}}$  [pu]")
    ax_c.set_xlabel(r"Timestep $t$")
    ax_c.set_title("(c) Mean water-junction temperature", loc="left")
    ax_c.legend(loc="best")
    ax_c.set_xlim(-0.5, STEPS - 0.5)
    ax_c.set_xticks(steps[:: max(1, STEPS // 8)])

    # Secondary axis: per-step max |Δt_pu| across junctions.
    ax_c2 = ax_c.twinx()
    ax_c2.bar(
        steps,
        tpu_gap,
        alpha=0.22,
        color=C_LP,
        width=0.55,
        linewidth=0,
        zorder=0,
    )
    ax_c2.set_ylabel(
        r"$\max_n |\Delta T_n|$  [pu]",
        color=C_LP,
        fontsize=9,
        labelpad=2,
    )
    ax_c2.tick_params(axis="y", labelcolor=C_LP, labelsize=8)
    ax_c2.spines["top"].set_visible(False)
    ax_c2.grid(False)

    # ── (d) Per-pipe linepack swing trajectories (top 5) ────────────────────
    if lp_traces:
        # Use a smooth muted-blue/teal sequential ramp so the family of
        # trajectories reads as variations of the same physical quantity.
        cmap = plt.get_cmap("viridis")
        n = len(lp_traces)
        for k, (swing, pid, deltas) in enumerate(lp_traces):
            ax_d.plot(
                steps,
                deltas,
                marker="o",
                markersize=3,
                lw=1.2,
                alpha=0.95,
                color=cmap(0.15 + 0.65 * k / max(1, n - 1)),
                label=f"pipe {pid}",
                markeredgecolor="white",
                markeredgewidth=0.4,
            )
        ax_d.axhline(0, color="0.5", lw=0.6, ls=":")
        ax_d.set_ylabel(r"$\Delta\,\mathrm{linepack}/\mathrm{lp}(0)$  [ppm]")
        ax_d.set_xlabel(r"Timestep $t$")
        ax_d.set_title("(d) Per-pipe linepack swing (top 5)", loc="left")
        ax_d.set_xlim(-0.5, STEPS - 0.5)
        ax_d.set_xticks(steps[:: max(1, STEPS // 8)])
        ax_d.legend(loc="best", ncol=1, handlelength=1.6)
    else:
        ax_d.set_title("(d) Per-pipe linepack swing", loc="left")
        ax_d.text(
            0.5,
            0.5,
            "no linepack variation recovered",
            transform=ax_d.transAxes,
            ha="center",
            va="center",
            fontsize=9,
            style="italic",
            color="0.4",
        )
        ax_d.set_xticks([])
        ax_d.set_yticks([])

    fig.tight_layout(pad=0.6, w_pad=1.2, h_pad=1.4)

    # Vector + raster siblings: PDF for LaTeX embedding, PNG for previews.
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    print("Solving MES with GasLinepack + LumpedThermalCapacitance ...")
    mes_with, ts_with = solve_scenario(with_storage=True)
    print("Solving MES baseline (no storage extensions) ...")
    mes_wo, ts_wo = solve_scenario(with_storage=False)

    tpu_with = mean_water_t_pu(mes_with, ts_with)
    tpu_wo = mean_water_t_pu(mes_wo, ts_wo)
    tpu_gap = t_pu_max_gap(mes_with, ts_with, ts_wo)
    pack_with = aggregate_pack_activity(mes_with, ts_with)
    pack_wo = aggregate_pack_activity(mes_wo, ts_wo)
    lp_traces = linepack_relative_deltas(mes_with, ts_with)

    out_path = Path(__file__).resolve().parent / "mes_storage_capabilities.png"
    make_plot(
        tpu_with=tpu_with,
        tpu_wo=tpu_wo,
        tpu_gap=tpu_gap,
        pack_with=pack_with,
        pack_wo=pack_wo,
        lp_traces=lp_traces,
        out_path=out_path,
    )

    print()
    print(f"Saved figure → {out_path}")
    print()
    print("Per-step comparison")
    print("-" * 88)
    header = (
        f"{'step':>4} | {'Σ|npk| with [mg/s]':>18} | {'Σ|npk| no':>10} | "
        f"{'t_pu (with)':>12} | {'t_pu (no)':>10} | {'max|Δt_pu|':>11}"
    )
    print(header)
    print("-" * len(header))
    for s in range(STEPS):
        print(
            f"{s:>4} | {1e6 * pack_with[s]:>18.5f} | "
            f"{1e6 * pack_wo[s]:>10.5f} | "
            f"{tpu_with[s]:>12.5f} | {tpu_wo[s]:>10.5f} | {tpu_gap[s]:>11.6f}"
        )

    if lp_traces:
        print()
        print("Top 5 pipes by linepack swing (with extension):")
        for swing, pid, deltas in lp_traces:
            print(
                f"  pipe {pid!s:>16} : swing = {swing:>8.3f} ppm  "
                f"trajectory (ppm) = {[round(d, 3) for d in deltas]}"
            )


if __name__ == "__main__":
    main()

"""Linepack and LTC on a trivial uncoupled gas + heat grid.

A minimal companion to ``examples/mes_storage_capabilities.py``: instead of a
full simbench-derived multi-energy system, this builds the smallest network
that still exercises both inter-temporal storage extensions, and runs it twice
over a short timeseries:

* once with :class:`~monee.model.formulation.linepack.GasLinepack` and
  :class:`~monee.model.formulation.ltc.LumpedThermalCapacitance` attached, and
* once without either extension as the baseline.

The network has two *uncoupled* carriers in a single :class:`~monee.model.Network`:

* a long gas line ``ext-grid -> 20 pipes in series -> sink`` whose sink demand
  is modulated step by step, so :class:`GasLinepack` charges and discharges the
  pipe inventory along the whole run, and
* a short heat line ``ext-grid -> pipe -> sink`` whose supply temperature is
  stepped down and back up, so :class:`LumpedThermalCapacitance` damps the
  temperature response at the consumer junction.

Both grids stay deliberately simple, so the absolute storage response is small,
but the comparison is unambiguous: aggregate ``|net_pack_kgs|`` is strictly
zero without the extension (steady-state constraint) and non-zero with it
(inter-temporal mass balance), and the consumer temperature steps sharply
without LTC but rolls off smoothly with it.

Unlike the MES example this runs a plain energy-flow timeseries (no
optimisation problem, no commercial solver): the bundled CasADi/IPOPT backend
handles the combined nonlinear gas + heat flow.

Requires: plotly. kaleido is optional (static PNG export); the interactive
HTML is always written.

Run::

    python examples/linepack_ltc_capabilities.py
"""

from __future__ import annotations

from pathlib import Path

import plotly.colors as pc
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import monee.model as mm
from monee import GasLinepack, LumpedThermalCapacitance, run_timeseries
from monee.simulation.timeseries import TimeseriesData

# Per-step gas-sink demand (× nominal) and heat-supply temperature [K].  Both
# share the same step axis: a low / peak / recovery demand swing for the gas
# side, and a stepped supply-temperature drop for the heat side.
# The gas side is a long line of GAS_N_PIPES segments in series (ext-grid at the
# head, sink at the far end), so linepack accumulates over a multi-kilometre run
# and every segment shows a distinct swing.
GAS_N_PIPES = 20
GAS_PIPE_LENGTH_M = 400
GAS_PIPE_DIAMETER_M = 0.15

GAS_BASE_KGS = 0.20
DEMAND_FACTORS = [0.5, 0.5, 1.0, 1.8, 2.2, 2.2, 1.4, 0.8, 0.5, 0.5, 1.2, 1.2]
HEAT_SUPPLY_T_K = [
    356.0,
    356.0,
    356.0,
    356.0,
    338.0,
    338.0,
    338.0,
    338.0,
    350.0,
    350.0,
    356.0,
    356.0,
]
STEPS = len(DEMAND_FACTORS)

# Cap the number of per-pipe trajectories drawn in panel (d) so the in-plot
# legend stays readable on a long line.
LP_TRACE_TOP_N = 5

_BAR_LINE_COLOR = "#2A2A2A"
_BAR_LINE_WIDTH = 1.6


def build_scenario(*, with_storage: bool):
    """Construct the trivial uncoupled gas + heat net and its demand timeseries.

    The build is deterministic, so both the with- and without-storage runs
    share component ids: that is what makes the per-component comparison
    meaningful.
    """
    net = mm.Network()

    # --- gas: ext-grid -> [GAS_N_PIPES pipes in series] -> sink ---
    gas_grid = mm.create_gas_grid("gas", type="lgas")
    g_source = net.child(mm.ExtHydrGrid())
    g_sink = net.child(mm.Sink(mass_flow_kgs=GAS_BASE_KGS * DEMAND_FACTORS[0]))
    head = net.node(mm.Junction(), grid=gas_grid, child_ids=[g_source])
    prev = head
    for i in range(GAS_N_PIPES):
        is_last = i == GAS_N_PIPES - 1
        node = net.node(
            mm.Junction(),
            grid=gas_grid,
            child_ids=[g_sink] if is_last else None,
        )
        net.branch(
            mm.GasPipe(diameter_m=GAS_PIPE_DIAMETER_M, length_m=GAS_PIPE_LENGTH_M),
            prev,
            node,
        )
        prev = node

    # --- heat: ext-grid (supply) -> pipe -> sink (a short line) ---
    w_supply = net.child(mm.ExtHydrGrid(t_k=HEAT_SUPPLY_T_K[0]))
    wn0 = net.node(mm.Junction(), mm.WATER, child_ids=[w_supply])
    wn1 = net.node(
        mm.Junction(), mm.WATER, child_ids=[net.child(mm.Sink(mass_flow_kgs=2))]
    )
    net.branch(mm.WaterPipe(diameter_m=0.15, length_m=300), wn0, wn1)

    if with_storage:
        net.add_extension(GasLinepack())
        net.add_extension(LumpedThermalCapacitance(first_step_steady_state=True))

    td = TimeseriesData()
    td.add_child_series(
        g_sink, "mass_flow_kgs", [GAS_BASE_KGS * f for f in DEMAND_FACTORS]
    )
    td.add_child_series(w_supply, "t_k", HEAT_SUPPLY_T_K)
    return net, td


def solve_scenario(*, with_storage: bool):
    net, td = build_scenario(with_storage=with_storage)
    ts = run_timeseries(
        net,
        timeseries_data=td,
        steps=STEPS,
        solver="ipopt",
    )
    return net, ts


def water_junctions(net):
    return [
        n
        for n in net.nodes
        if isinstance(n.model, mm.Junction) and isinstance(n.grid, mm.WaterGrid)
    ]


def gas_pipes(net):
    return [b for b in net.branches if isinstance(b.model, mm.GasPipe)]


def mean_water_t_pu(net, ts):
    """Per-step mean junction temperature across all water junctions [pu]."""
    juncs = water_junctions(net)
    out = []
    for s in range(STEPS):
        vals = [
            float(ts.get_result_for_id(n.id, "t_pu").iloc[s])
            for n in juncs
            if ts.get_result_for_id(n.id, "t_pu").iloc[s] is not None
        ]
        out.append(sum(vals) / len(vals) if vals else float("nan"))
    return out


def t_pu_max_gap(net, ts_with, ts_wo):
    """Per-step maximum |Δt_pu| across junctions (with − without) [pu]."""
    juncs = water_junctions(net)
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


def aggregate_pack_activity(net, ts):
    """Per-step Σ |net_pack_kgs| across all gas pipes [kg/s].

    Strictly zero by construction in the no-extension run (the
    extension-injected variable does not exist) and a direct measure of
    inter-temporal storage activity in the with-extension run.
    """
    pipes = gas_pipes(net)
    out = []
    for s in range(STEPS):
        total = 0.0
        for p in pipes:
            v = ts.get_result_for_id(p.id, "net_pack_kgs").iloc[s]
            if v is not None:
                total += abs(float(v))
        out.append(total)
    return out


def linepack_deltas(net, ts):
    """Per-pipe inventory deviation linepack(t) − linepack(0) over time [kg].

    Returns the trajectory of every pipe, sorted by swing magnitude.
    """
    pipes = gas_pipes(net)
    rows = []
    for p in pipes:
        lp = ts.get_result_for_id(p.id, "linepack_kg")
        if lp.iloc[0] is None or float(lp.iloc[0]) == 0:
            continue
        lp0 = float(lp.iloc[0])
        deltas_kg = [float(v) - lp0 for v in lp.values]
        swing = max(deltas_kg) - min(deltas_kg)
        rows.append((swing, p.id, deltas_kg))
    rows.sort(reverse=True, key=lambda r: r[0])
    return rows


# Libertinus-first serif stack, falling back to whatever serif the renderer has.
FONT_FAMILY = "Libertinus Serif, Libertinus, Times New Roman, serif"

# Paul Tol "vibrant" palette — colourblind-safe, prints well in greyscale.
C_WITH = "#0077BB"  # blue  — with-extension series
C_WITHOUT = "#CC3311"  # red   — without-extension series
C_DEMAND = "#EE7733"  # orange — demand modulation
C_SUPPLY = "#33BBEE"  # cyan  — supply-temperature modulation
C_LP = "#009988"  # teal  — Δt_pu overlay

# Publication sizing for a one-column dissertation figure: large fonts, thick
# lines, in-plot legends.
SUBPLOT_TITLE_SIZE = 27
AXIS_TITLE_SIZE = 25
TICK_SIZE = 23
LEGEND_SIZE = 21
LINE_WIDTH = 3.6
MARKER_SIZE = 9
BAR_GAP = 0.32  # shared by panels (b) and (c) so the bar widths match


def _inset_legend(x, y, xanchor="left", yanchor="top"):
    """A compact legend boxed inside a subplot at the given anchored corner."""
    return dict(
        x=x,
        y=y,
        xanchor=xanchor,
        yanchor=yanchor,
        font=dict(size=LEGEND_SIZE),
        bgcolor="rgba(255,255,255,0.78)",
        bordercolor="rgba(110,110,110,0.6)",
        borderwidth=1,
    )


def make_plot(
    tpu_with,
    tpu_wo,
    tpu_gap,
    pack_with,
    pack_wo,
    lp_traces,
    out_path: Path,
):
    """Render the 2×2 summary figure with Plotly and save HTML (+ PNG if able)."""
    steps = list(range(STEPS))

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "(a) Gas demand and heat-supply modulation",
            "(b) Inter-temporal gas-storage activity",
            "(c) Mean water-junction temperature",
            "(d) Per-pipe linepack swing",
        ),
        specs=[
            [{"secondary_y": True}, {}],
            [{"secondary_y": True}, {}],
        ],
        horizontal_spacing=0.18,
        vertical_spacing=0.16,
    )
    fig.update_annotations(font_size=SUBPLOT_TITLE_SIZE)  # subplot titles only

    # ── (a) gas demand mass flow (step+fill) + supply temperature twin axis ──
    gas_demand_kgs = [GAS_BASE_KGS * f for f in DEMAND_FACTORS]
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=gas_demand_kgs,
            mode="lines",
            line=dict(shape="hv", color=C_DEMAND, width=LINE_WIDTH),
            fill="tozeroy",
            fillcolor="rgba(238,119,51,0.18)",
            name=r"$\dot{m}_{\mathrm{gas}}$",
            legend="legend",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=HEAT_SUPPLY_T_K,
            mode="lines+markers",
            line=dict(shape="hv", color=C_SUPPLY, width=LINE_WIDTH, dash="dot"),
            marker=dict(size=MARKER_SIZE),
            name=r"$T_{\mathrm{supply}}$",
            legend="legend",
        ),
        row=1,
        col=1,
        secondary_y=True,
    )
    fig.update_yaxes(
        title_text=r"$\Large \dot{m}_{\mathrm{gas}}\;\;[\mathrm{kg\,s^{-1}}]$",
        title_font_size=AXIS_TITLE_SIZE,
        row=1,
        col=1,
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text=r"$\Large T_{\mathrm{supply}}\;\;[\mathrm{K}]$",
        title_font_size=AXIS_TITLE_SIZE,
        row=1,
        col=1,
        secondary_y=True,
        color=C_SUPPLY,
        showgrid=False,
    )

    # ── (b) Σ|net_pack_kgs| in kg/s — strictly zero without the extension ───
    fig.add_trace(
        go.Bar(
            x=steps,
            y=pack_with,
            marker=dict(
                color=C_WITH,
                line=dict(color=_BAR_LINE_COLOR, width=_BAR_LINE_WIDTH),
            ),
            name="with linepack",
            legend="legend2",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            x=steps,
            y=pack_wo,
            marker=dict(
                color=C_WITHOUT,
                line=dict(color=_BAR_LINE_COLOR, width=_BAR_LINE_WIDTH),
            ),
            marker_pattern_shape="/",
            name="w/o linepack (≡ 0)",
            legend="legend2",
        ),
        row=1,
        col=2,
    )
    fig.update_yaxes(
        title_text=r"$\Large \sum_p\,|\dot{m}_{\mathrm{pack}}|\;\;[\mathrm{kg\,s^{-1}}]$",
        title_font_size=AXIS_TITLE_SIZE,
        row=1,
        col=2,
    )

    # ── (c) mean water-junction temperature + max|Δt_pu| twin-axis bars ─────
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=tpu_with,
            mode="lines+markers",
            line=dict(color=C_WITH, width=LINE_WIDTH),
            marker=dict(symbol="circle", size=MARKER_SIZE),
            name="with LTC",
            legend="legend3",
        ),
        row=2,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=tpu_wo,
            mode="lines+markers",
            line=dict(color=C_WITHOUT, width=LINE_WIDTH, dash="dash"),
            marker=dict(symbol="square", size=MARKER_SIZE),
            name="without LTC",
            legend="legend3",
        ),
        row=2,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(
            x=steps,
            y=tpu_gap,
            marker=dict(
                color=C_LP,
                line=dict(color=_BAR_LINE_COLOR, width=_BAR_LINE_WIDTH),
            ),
            opacity=0.30,
            name=r"$\max_n |\Delta t_{\mathrm{pu}}|$",
            legend="legend3",
        ),
        row=2,
        col=1,
        secondary_y=True,
    )
    fig.update_yaxes(
        title_text=r"$\Large \bar{T}_{\mathrm{water}}\;\;[\mathrm{pu}]$",
        title_font_size=AXIS_TITLE_SIZE,
        row=2,
        col=1,
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text=r"$\Large \max_n |\Delta t_{\mathrm{pu}}|\;\;[\mathrm{pu}]$",
        title_font_size=AXIS_TITLE_SIZE,
        row=2,
        col=1,
        secondary_y=True,
        color=C_LP,
        showgrid=False,
    )
    fig.update_xaxes(
        title_text="timestep t", title_font_size=AXIS_TITLE_SIZE, row=2, col=1
    )

    # ── (d) per-pipe linepack swing trajectories [kg] ───────────────────────
    lp_traces = lp_traces[:LP_TRACE_TOP_N]
    if lp_traces:
        n = len(lp_traces)
        colors = pc.sample_colorscale(
            "Viridis", [0.15 + 0.65 * k / max(1, n - 1) for k in range(n)]
        )
        for (swing, pid, deltas), color in zip(lp_traces, colors):
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=deltas,
                    mode="lines+markers",
                    line=dict(color=color, width=LINE_WIDTH),
                    marker=dict(size=MARKER_SIZE - 2),
                    name=f"pipe {pid}",
                    legend="legend4",
                ),
                row=2,
                col=2,
            )
        fig.add_hline(row=2, col=2, y=0, line=dict(color="#888", width=1.2, dash="dot"))
        fig.update_yaxes(
            title_text=r"$\Large \Delta\,\mathrm{linepack}\;\;[\mathrm{kg}]$",
            title_font_size=AXIS_TITLE_SIZE,
            row=2,
            col=2,
        )
    else:
        fig.add_annotation(
            row=2,
            col=2,
            text="no linepack variation recovered",
            showarrow=False,
            font=dict(size=14, color="#888"),
        )
    fig.update_xaxes(
        title_text="timestep t", title_font_size=AXIS_TITLE_SIZE, row=2, col=2
    )

    # Per-subplot legends, boxed inside the top-left corner of each panel.
    fig.update_xaxes(tickfont_size=TICK_SIZE)
    fig.update_yaxes(tickfont_size=TICK_SIZE)
    fig.update_layout(
        font=dict(family=FONT_FAMILY, size=TICK_SIZE),
        barmode="overlay",
        bargap=BAR_GAP,
        template="plotly_white",
        showlegend=True,
        legend=_inset_legend(0.013, 0.79, "left", "middle"),
        legend2=_inset_legend(0.99, 0.985, "right", "top"),
        legend3=_inset_legend(0.013, 0.30, "left", "middle"),
        legend4=_inset_legend(0.99, 0.013, "right", "bottom"),
        width=1100,
        height=820,
        margin=dict(l=80, r=0, t=60, b=70),
    )

    html_path = out_path.with_suffix(".html")
    fig.write_html(html_path)
    saved = [html_path]
    try:
        fig.write_image(out_path, scale=2)
        fig.write_image(out_path.with_suffix(".pdf"))
        saved.append(out_path)
    except Exception as exc:  # kaleido missing or no renderer
        print(f"(skipped PNG export: {exc})")
    return saved


def main() -> None:
    print("Solving trivial gas+heat net with GasLinepack + LTC ...")
    net_with, ts_with = solve_scenario(with_storage=True)
    print("Solving baseline (no storage extensions) ...")
    net_wo, ts_wo = solve_scenario(with_storage=False)

    tpu_with = mean_water_t_pu(net_with, ts_with)
    tpu_wo = mean_water_t_pu(net_wo, ts_wo)
    tpu_gap = t_pu_max_gap(net_with, ts_with, ts_wo)
    pack_with = aggregate_pack_activity(net_with, ts_with)
    pack_wo = aggregate_pack_activity(net_wo, ts_wo)
    lp_traces = linepack_deltas(net_with, ts_with)

    out_path = Path(__file__).resolve().parent / "linepack_ltc_capabilities.png"
    saved = make_plot(
        tpu_with=tpu_with,
        tpu_wo=tpu_wo,
        tpu_gap=tpu_gap,
        pack_with=pack_with,
        pack_wo=pack_wo,
        lp_traces=lp_traces,
        out_path=out_path,
    )

    print()
    for p in saved:
        print(f"Saved -> {p}")
    print()
    print("Per-step comparison")
    header = (
        f"{'step':>4} | {'sum|npk| with[kg/s]':>19} | {'sum|npk| no':>11} | "
        f"{'t_pu (with)':>12} | {'t_pu (no)':>10} | {'max|dt_pu|':>11}"
    )
    print(header)
    print("-" * len(header))
    for s in range(STEPS):
        print(
            f"{s:>4} | {pack_with[s]:>18.8f} | "
            f"{pack_wo[s]:>10.5f} | "
            f"{tpu_with[s]:>12.5f} | {tpu_wo[s]:>10.5f} | {tpu_gap[s]:>11.6f}"
        )

    if lp_traces:
        print()
        print(f"Top {LP_TRACE_TOP_N} pipes by linepack swing (with extension):")
        for swing, pid, deltas in lp_traces[:LP_TRACE_TOP_N]:
            print(
                f"  pipe {pid!s:>16} : swing = {swing:>8.4f} kg  "
                f"trajectory (kg) = {[round(d, 4) for d in deltas]}"
            )


if __name__ == "__main__":
    main()

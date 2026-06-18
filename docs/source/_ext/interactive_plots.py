"""Build-time generators for the interactive (Plotly) documentation figures.

These run at Sphinx build time (where CasADi is available, e.g. on Read the
Docs' Linux builders): each ``build_*`` function recomputes a result and emits a
self-contained, responsive Plotly HTML file (plus a static PNG fallback for the
PDF build). No solver runs in the browser. The figures are embedded via an
``<iframe>`` so their bundled Plotly.js never clashes with the page's MathJax.

Light/dark mode: figures are authored with light-mode colours; an injected
``post_script`` reads the parent furo theme (``data-theme`` attribute, falling
back to ``prefers-color-scheme``) and recolours the text, grids and subplot
titles on load and whenever the theme is toggled. So the one font colour works
in both modes - it is swapped at runtime, not baked in.

Add a generator by writing a ``build_*`` function (use ``_base_layout`` +
``_write``) and registering it in ``GENERATORS``; ``setup(app)`` wires them to
the ``builder-inited`` event.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import monee.express as mx
import monee.model as mm
import monee.problem as mp
from monee.model import GasLinepack, LumpedThermalCapacitance
from monee.model.child import ExtPowerGrid
from monee.problem.core import Objectives, OptimizationProblem
from monee.simulation import TimeseriesData, run_multi_period, run_timeseries

# Shared palette. Saturated hues read on both light and dark backgrounds; the
# neutral text/grid colours are swapped at runtime by the theme post_script.
C_PRICE = "#8d6e63"
C_CHG = "#2c7bb6"
C_DIS = "#d7191c"
C_SOC = "#1a9641"
C_LOAD = "#f4a261"
C_HEAT = "#d7191c"
C_GAS = "#2c7bb6"
C_LP = "#2c7bb6"
C_ACCENT = "#1a9641"
TEXT = "#171717"  # light-mode default; post_script overrides for dark mode
FONT_FAMILY = "Inter, Segoe UI, Helvetica, Arial"

# Okabe-Ito colourblind-safe qualitative palette (distinguishable under deutan/
# protan/tritan vision and in greyscale). The benchmark figures draw from these
# so no comparison rests on a red-vs-green contrast alone.
CB_BLUE = "#0072B2"
CB_ORANGE = "#E69F00"
CB_GREEN = "#009E73"
CB_VERMILION = "#D55E00"
CB_SKY = "#56B4E9"
CB_PURPLE = "#CC79A7"
CB_YELLOW = "#F0E442"

# Every benchmark bar gets a thin dark outline (crisp on both light and dark
# pages); head-to-head series additionally carry a hatch pattern so the contrast
# never rests on hue alone.
BAR_LINE = "#222222"
BAR_LINE_WIDTH = 1.1


def _bar_marker(color, pattern=None):
    """Bar ``marker`` dict: solid *color* fill, crisp dark outline, optional hatch.

    *pattern* is a Plotly pattern shape (``"/"``, ``"\\\\"``, ``"x"``, ``"."`` ...);
    a falsy value leaves the bar solid. The hatch is drawn in the outline colour
    at low opacity so it adds texture without muddying the fill hue."""
    marker = {
        "color": color,
        "line": {"color": BAR_LINE, "width": BAR_LINE_WIDTH},
    }
    if pattern:
        marker["pattern"] = {
            "shape": pattern,
            "fgcolor": BAR_LINE,
            "fgopacity": 0.4,
            "size": 7,
            "solidity": 0.32,
        }
    return marker


# Repo root, for reading the committed benchmark result CSVs at build time.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _bench_csv(*parts):
    """Absolute path to a file under ``<repo>/benchmarks/``."""
    return os.path.join(_REPO_ROOT, "benchmarks", *parts)


# Injected after each figure: recolour text/grids/titles to match the furo theme.
# ``{plot_id}`` is substituted by Plotly's write_html with the figure div id.
_THEME_POST_SCRIPT = """
var gd = document.getElementById('{plot_id}');
var _styleEl = document.createElement('style');
document.head.appendChild(_styleEl);
function _apply(dark) {
    var fc = dark ? '#ffffff' : '#333333';
    var grid = dark ? 'rgba(200,200,200,0.22)' : 'rgba(80,80,80,0.12)';
    var zl = dark ? 'rgba(200,200,200,0.42)' : 'rgba(80,80,80,0.25)';
    // Bulletproof text recolour: override the fill of EVERY svg text element with
    // !important, which beats Plotly's inline fill regardless of whether it is a
    // tick label, legend entry, title, annotation or bar value label.
    _styleEl.textContent = 'text { fill: ' + fc + ' !important; }';
    // Grid / zero lines are svg strokes (not text) - update those via relayout.
    var up = {};
    Object.keys(gd.layout).forEach(function (k) {
        if (k.indexOf('xaxis') === 0 || k.indexOf('yaxis') === 0) {
            up[k + '.gridcolor'] = grid;
            up[k + '.zerolinecolor'] = zl;
        }
    });
    try { Plotly.relayout(gd, up); } catch (e) {}
}
function _detect() {
    try {
        var b = window.parent.document.body;
        var t = b && b.dataset ? b.dataset.theme : null;  // furo: light|dark|auto
        if (t === 'dark') return true;
        if (t === 'light') return false;
    } catch (e) {}
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
}
// Authoritative path: the parent furo page postMessages its theme (works even
// when window.parent reads are blocked, e.g. file://). Self-detect as fallback.
window.addEventListener('message', function (ev) {
    if (ev && ev.data && ev.data.type === 'monee-theme') _apply(ev.data.theme === 'dark');
});
_apply(_detect());
try {
    new MutationObserver(function () { _apply(_detect()); }).observe(
        window.parent.document.body,
        {attributes: true, attributeFilter: ['data-theme']});
} catch (e) {}
if (window.matchMedia) {
    try {
        window.matchMedia('(prefers-color-scheme: dark)')
            .addEventListener('change', function () { _apply(_detect()); });
    } catch (e) {}
}
"""


def _base_layout(fig, title, height=640):
    """Apply the shared look: transparent background (so the figure floats on the
    furo page), light-mode default colours, no legend, tidy margins."""
    fig.update_layout(
        title={
            "text": f"<b>{title}</b>",
            "x": 0.5,
            "xanchor": "center",
            "y": 0.97,
            "yanchor": "top",
            "font": {"size": 20, "color": TEXT},
        },
        template="plotly_white",
        autosize=True,
        height=height,
        bargap=0.25,
        showlegend=False,
        font={"family": FONT_FAMILY, "size": 14, "color": TEXT},
        margin={"l": 60, "r": 25, "t": 90, "b": 55},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(80,80,80,0.12)", zeroline=False)
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(80,80,80,0.12)",
        zeroline=True,
        zerolinecolor="rgba(80,80,80,0.25)",
    )
    return fig


def _write(fig, out_path, height_px=640):
    """Write the responsive interactive HTML (width 100%, so it fills its iframe
    with no horizontal scroll) plus best-effort static PNG and PDF copies for the
    PDF build."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.write_html(
        out_path,
        include_plotlyjs="cdn",
        full_html=True,
        default_width="100%",
        default_height=f"{height_px}px",
        config={"displayModeBar": False, "responsive": True},
        post_script=_THEME_POST_SCRIPT,
    )
    # Static copies for the LaTeX/PDF builder; never fail the build on them.
    for ext in ("png", "pdf"):
        try:
            fig.write_image(
                out_path[:-5] + "." + ext, width=900, height=height_px, scale=2
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"[interactive_plots] {ext.upper()} export skipped for {out_path}: {exc}"
            )
    return out_path


def _solve_storage(capacity_mwh, p_max_mw, load, price, dt_h=1.0):
    """Solve the price-arbitrage dispatch for one battery capacity.

    Returns (dispatch, soc, import_mw, bill) where bill is the energy purchase
    cost ``Σ price · import · dt`` (import = -ext_grid.p_mw, since slack import is
    a negative ``p_mw`` under the load sign convention).
    """
    net = mx.create_multi_energy_network()
    b0 = mx.create_bus(net)
    b1 = mx.create_bus(net)
    ext_id = mx.create_ext_power_grid(net, b0)
    mx.create_line(net, b0, b1, length_m=100, r_ohm_per_m=1e-4, x_ohm_per_m=1e-4)
    mx.create_power_load(net, b1, p_mw=0.0, q_mvar=0.0, name="load")
    bat = mm.ElectricStorage(
        e_mwh_initial=capacity_mwh / 2,
        e_mwh_max=capacity_mwh,
        p_max_mw=p_max_mw,
    )
    bat_id = mx.create_el_child(net, bat, node_id=b1, name="bat")

    td = TimeseriesData()
    td.add_child_series_by_name("load", "p_mw", load)
    td.add_objective_data(ext_id, "price", price)

    prob = OptimizationProblem()
    prob.controllable_storages()
    objectives = Objectives()
    # Minimise the purchase bill: price * import, with import = -p_mw.
    objectives.select(lambda m: isinstance(m, ExtPowerGrid)).calculate(
        lambda models: sum(getattr(m, "price", 1) * (-m.p_mw) for m in models)
    )
    prob.objectives = objectives

    result = run_multi_period(
        net,
        td,
        optimization_problem=prob,
        dt_h=dt_h,
        terminal_state={(bat_id, "e_mwh"): capacity_mwh / 2},
    )
    disp = [float(x) for x in result.get_result_for_id(bat_id, "p_mw").values]
    soc = [float(x) for x in result.get_result_for_id(bat_id, "e_mwh").values]
    imp = [-float(x) for x in result.get_result_for_id(ext_id, "p_mw").values]
    bill = sum(p * i * dt_h for p, i in zip(price, imp))
    return disp, soc, imp, bill


def _storage_dispatch_figure():
    """Optimised battery dispatch under a peaky price: the solver charges in the
    cheap off-peak hours and discharges into the expensive peak. Interactive
    (hover for exact values, zoom, pan); a single solve, no parameter sweep."""
    load = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
    price = [10, 12, 45, 60, 40, 15]
    capacity = 6.0
    hours = list(range(len(load)))

    disp, soc, imp, bill = _solve_storage(capacity, capacity / 2.0, load, price)
    dispatch_colors = [C_CHG if v >= 0 else C_DIS for v in disp]

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.09,
        subplot_titles=(
            "Electricity price",
            "Battery dispatch  (+ charge / - discharge)",
            "State of charge",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=hours,
            y=price,
            mode="lines",
            line_shape="hv",
            line={"color": C_PRICE, "width": 3},
            name="price",
            fill="tozeroy",
            fillcolor="rgba(141,110,99,0.12)",
            hovertemplate="hour %{x}: %{y} /MWh<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=hours,
            y=disp,
            marker_color=dispatch_colors,
            name="dispatch",
            hovertemplate="hour %{x}: %{y:.2f} MW<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=hours,
            y=soc,
            mode="lines+markers",
            line={"color": C_SOC, "width": 3},
            marker={"size": 8},
            name="SoC",
            fill="tozeroy",
            fillcolor="rgba(26,150,65,0.12)",
            hovertemplate="hour %{x}: %{y:.2f} MWh<extra></extra>",
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[hours[0], hours[-1]],
            y=[capacity, capacity],
            mode="lines",
            line={"color": C_DIS, "width": 1.5, "dash": "dot"},
            name="capacity",
            hoverinfo="skip",
        ),
        row=3,
        col=1,
    )

    _base_layout(fig, f"Optimised battery dispatch: {capacity:g} MWh price arbitrage")
    fig.update_xaxes(title_text="hour", row=3, col=1)
    fig.update_yaxes(title_text="/MWh", row=1, col=1)
    fig.update_yaxes(title_text="MW", row=2, col=1)
    fig.update_yaxes(title_text="MWh", row=3, col=1)

    return fig


def build_storage_dispatch(out_path):
    """Solve the optimised dispatch and write the interactive HTML to *out_path*."""
    return _write(_storage_dispatch_figure(), out_path, height_px=640)


def build_storage_prescribed(out_path):
    """Prescribed battery dispatch in a plain timeseries: SoC integrates the fixed
    charge/discharge schedule. Mirrors the storage.rst prescribed-dispatch block."""
    dispatch = [1.0, 0.5, -1.0, -1.5, 0.0, 0.5]
    net = mx.create_multi_energy_network()
    bus0 = mx.create_bus(net)
    bus1 = mx.create_bus(net)
    mx.create_ext_power_grid(net, bus0)
    mx.create_line(net, bus0, bus1, length_m=500, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)
    mx.create_power_load(net, bus1, p_mw=0.8, q_mvar=0.0)
    storage = mm.ElectricStorage(e_mwh_initial=5.0, e_mwh_max=10.0, p_max_mw=2.0)
    bat_id = mx.create_el_child(net, storage, node_id=bus1, name="battery")
    td = TimeseriesData()
    td.add_child_series(bat_id, "p_mw", dispatch)
    result = run_timeseries(net, td)
    soc = [float(x) for x in result.get_result_for_id(bat_id, "e_mwh").values]
    steps = list(range(len(dispatch)))
    colors = [C_CHG if v >= 0 else C_DIS for v in dispatch]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=(
            "Prescribed dispatch  (+ charge / - discharge)",
            "State of charge",
        ),
    )
    fig.add_trace(
        go.Bar(
            x=steps,
            y=dispatch,
            marker_color=colors,
            name="dispatch",
            hovertemplate="hour %{x}: %{y:.2f} MW<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=soc,
            mode="lines+markers",
            line={"color": C_SOC, "width": 3},
            marker={"size": 8},
            name="SoC",
            fill="tozeroy",
            fillcolor="rgba(26,150,65,0.12)",
            hovertemplate="hour %{x}: %{y:.2f} MWh<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[steps[0], steps[-1]],
            y=[10.0, 10.0],
            mode="lines",
            line={"color": C_DIS, "width": 1.5, "dash": "dot"},
            name="capacity",
            hovertemplate="capacity 10 MWh<extra></extra>",
        ),
        row=2,
        col=1,
    )
    _base_layout(fig, "Electric storage: prescribed dispatch", height=480)
    fig.update_xaxes(title_text="hour", row=2, col=1)
    fig.update_yaxes(title_text="MW", row=1, col=1)
    fig.update_yaxes(title_text="MWh", range=[0, 11], row=2, col=1)
    return _write(fig, out_path, height_px=480)


def build_storage_gas(out_path):
    """Prescribed gas-storage charge/discharge cycle over 8 hours. Mirrors the
    storage.rst gas-storage block."""
    dispatch = [0.05, 0.05, 0.0, 0.0, -0.05, -0.05, 0.0, 0.0]
    net_g = mx.create_multi_energy_network()
    j0 = mx.create_gas_junction(net_g)
    j1 = mx.create_gas_junction(net_g)
    mx.create_gas_ext_grid(net_g, j0)
    mx.create_gas_pipe(net_g, j0, j1, diameter_m=0.3, length_m=5000)
    mx.create_gas_sink(net_g, j1, mass_flow_kgs=0.05)
    tank = mm.GasStorage(
        m_stored_kg_initial=1000.0, m_stored_kg_max=5000.0, flow_max_kgs=0.2
    )
    tank_id = mx.create_gas_child(net_g, tank, node_id=j1, name="tank")
    td_g = TimeseriesData()
    td_g.add_child_series(tank_id, "mass_flow_kgs", dispatch)
    result_g = run_timeseries(net_g, td_g)
    stored = [
        float(x) for x in result_g.get_result_for_id(tank_id, "m_stored_kg").values
    ]
    steps = list(range(len(dispatch)))
    colors = [
        C_CHG if v > 0 else (C_DIS if v < 0 else "rgba(150,150,150,0.6)")
        for v in dispatch
    ]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=(
            "Prescribed gas storage dispatch  (+ charge / - discharge)",
            "Stored mass",
        ),
    )
    fig.add_trace(
        go.Bar(
            x=steps,
            y=dispatch,
            marker_color=colors,
            name="net flow",
            hovertemplate="hour %{x}: %{y:.3f} kg/s<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=stored,
            mode="lines+markers",
            line={"color": C_SOC, "width": 3},
            marker={"size": 8},
            name="stored",
            fill="tozeroy",
            fillcolor="rgba(26,150,65,0.12)",
            hovertemplate="hour %{x}: %{y:,.0f} kg<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[steps[0], steps[-1]],
            y=[1000.0, 1000.0],
            mode="lines",
            line={"color": "rgba(128,128,128,0.7)", "width": 1.5, "dash": "dash"},
            name="initial",
            hovertemplate="initial 1,000 kg<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[steps[0], steps[-1]],
            y=[5000.0, 5000.0],
            mode="lines",
            line={"color": C_DIS, "width": 1.5, "dash": "dot"},
            name="capacity",
            hovertemplate="capacity 5,000 kg<extra></extra>",
        ),
        row=2,
        col=1,
    )
    _base_layout(fig, "Gas storage: charge / discharge cycle", height=480)
    fig.update_xaxes(title_text="hour", row=2, col=1)
    fig.update_yaxes(title_text="kg/s", row=1, col=1)
    fig.update_yaxes(title_text="kg", row=2, col=1)
    return _write(fig, out_path, height_px=480)


def build_concepts_multi_period(out_path):
    """Multi-period battery dispatch (concepts/multi_period.rst): controllable
    storage with a cyclic terminal SoC over a 6-period horizon."""
    LOAD = [0.4, 0.5, 1.4, 1.8, 1.5, 0.4]

    net = mx.create_multi_energy_network()
    bus0 = mx.create_bus(net)
    bus1 = mx.create_bus(net)
    mx.create_ext_power_grid(net, bus0)
    mx.create_line(net, bus0, bus1, length_m=500, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)
    mx.create_power_load(net, bus1, p_mw=0.0, q_mvar=0.0, name="load")

    storage = mm.ElectricStorage(e_mwh_initial=2.0, e_mwh_max=4.0, p_max_mw=1.0)
    bat = mx.create_el_child(net, storage, node_id=bus1, name="battery")

    td = TimeseriesData()
    td.add_child_series_by_name("load", "p_mw", LOAD)

    prob = OptimizationProblem()
    prob.controllable_storages()
    result = run_multi_period(
        net,
        td,
        optimization_problem=prob,
        dt_h=1.0,
        terminal_state={(bat, "e_mwh"): 2.0},
    )

    soc = [float(x) for x in result.get_result_for_id(bat, "e_mwh").values]
    disp = [float(x) for x in result.get_result_for_id(bat, "p_mw").values]
    steps = list(range(len(LOAD)))
    dispatch_colors = [C_CHG if v >= 0 else C_DIS for v in disp]

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.09,
        subplot_titles=(
            "Consumer demand",
            "Optimised dispatch  (+ charge / - discharge)",
            "State of charge",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=LOAD,
            mode="lines",
            line_shape="hv",
            line={"color": C_LOAD, "width": 3},
            name="load",
            fill="tozeroy",
            fillcolor="rgba(244,162,97,0.15)",
            hovertemplate="period %{x}: %{y:.2f} MW<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=steps,
            y=disp,
            marker_color=dispatch_colors,
            name="dispatch",
            hovertemplate="period %{x}: %{y:.2f} MW<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=soc,
            mode="lines+markers",
            line={"color": C_SOC, "width": 3},
            marker={"size": 8},
            name="SoC",
            fill="tozeroy",
            fillcolor="rgba(26,150,65,0.12)",
            hovertemplate="period %{x}: %{y:.2f} MWh<extra></extra>",
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[steps[0], steps[-1]],
            y=[4.0, 4.0],
            mode="lines",
            line={"color": "grey", "width": 1.5, "dash": "dash"},
            name="capacity",
            hovertemplate="capacity 4 MWh<extra></extra>",
        ),
        row=3,
        col=1,
    )
    _base_layout(fig, "Multi-period battery dispatch", height=640)
    fig.update_xaxes(title_text="period", row=3, col=1)
    fig.update_yaxes(title_text="Load [MW]", row=1, col=1)
    fig.update_yaxes(title_text="Battery [MW]", row=2, col=1)
    fig.update_yaxes(title_text="SoC [MWh]", range=[0, 4.5], row=3, col=1)
    return _write(fig, out_path, height_px=640)


def build_temporal_extensions_1(out_path):
    """LTC thermal inertia: same network solved with and without
    LumpedThermalCapacitance; junction temperatures for a supply step-change."""
    supply_temp = [1.0, 1.0, 1.0, 1.0, 0.8, 0.8, 0.8, 0.8]

    def build_net(with_ltc):
        net = mx.create_multi_energy_network()
        j_supply = mx.create_water_junction(net)
        j_mid = mx.create_water_junction(net)
        j_load = mx.create_water_junction(net)
        mx.create_ext_hydr_grid(net, j_supply)
        mx.create_water_sink(net, j_load, mass_flow_kgs=0.5)
        mx.create_water_pipe(net, j_supply, j_mid, diameter_m=0.3, length_m=500)
        mx.create_water_pipe(net, j_mid, j_load, diameter_m=0.2, length_m=300)
        if with_ltc:
            net.add_extension(LumpedThermalCapacitance())
        return net, j_supply, j_mid, j_load

    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.06,
        subplot_titles=("Without LTC", "With LTC"),
    )
    series_spec = [
        ("supply (j0)", "supply", C_GAS),
        ("mid (j1)", "mid", C_HEAT),
        ("load (j2)", "load", C_ACCENT),
    ]
    for col, with_ltc in zip((1, 2), (False, True)):
        net, j_supply, j_mid, j_load = build_net(with_ltc)
        td = TimeseriesData()
        td.add_node_series(j_supply, "t_pu", supply_temp)
        result = run_timeseries(net, td)
        t_supply_s = result.get_result_for_id(j_supply, "t_pu")
        t_mid_s = result.get_result_for_id(j_mid, "t_pu")
        t_load_s = result.get_result_for_id(j_load, "t_pu")
        steps = list(range(len(t_supply_s)))
        node_values = {
            "supply": [float(v) for v in t_supply_s.values],
            "mid": [float(v) for v in t_mid_s.values],
            "load": [float(v) for v in t_load_s.values],
        }
        for label, key, color in series_spec:
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=node_values[key],
                    mode="lines",
                    line_shape="hv",
                    line={"color": color, "width": 2.5},
                    name=label,
                    hovertemplate=label + "<br>step %{x} h: %{y:.3f} pu<extra></extra>",
                ),
                row=1,
                col=col,
            )
        fig.add_vline(
            x=3.5,
            line={"color": "rgba(128,128,128,0.5)", "width": 1, "dash": "dash"},
            row=1,
            col=col,
        )
    _base_layout(fig, "Thermal inertia: supply step-change at t = 4", height=480)
    fig.update_xaxes(title_text="Timestep  [h]", row=1, col=1)
    fig.update_xaxes(title_text="Timestep  [h]", row=1, col=2)
    fig.update_yaxes(title_text="Temperature  [pu]", range=[0.75, 1.05], row=1, col=1)
    fig.update_yaxes(range=[0.75, 1.05], row=1, col=2)
    return _write(fig, out_path, height_px=480)


def build_temporal_extensions_2(out_path):
    """Gas linepack buffering: same network solved with and without GasLinepack.
    Consumer demand, source feed rate, and linepack stored-mass deviation."""
    DEMAND = [0.15, 0.16, 0.17, 0.14, 0.15, 0.10, 0.12, 0.04]  # kg/s

    def build_and_run(with_linepack):
        net = mx.create_multi_energy_network()
        j0 = mx.create_gas_junction(net)
        j1 = mx.create_gas_junction(net)
        j2 = mx.create_gas_junction(net)
        mx.create_gas_ext_grid(net, j0)
        mx.create_gas_sink(net, j2, mass_flow_kgs=0.20, name="consumer")
        pipe_id = mx.create_gas_pipe(net, j0, j1, diameter_m=0.6, length_m=40_000)
        mx.create_gas_pipe(net, j1, j2, diameter_m=0.3, length_m=8_000)
        if with_linepack:
            net.add_extension(GasLinepack())
        td = TimeseriesData()
        td.add_child_series_by_name("consumer", "mass_flow_kgs", DEMAND)
        result = run_timeseries(net, td)
        return result, pipe_id, j0

    result_lp, pipe_id, j0 = build_and_run(with_linepack=True)
    result_nolp, _, _ = build_and_run(with_linepack=False)
    src_lp = result_lp.get_result_for_id(j0, "mass_flow_kgs")
    src_nolp = result_nolp.get_result_for_id(j0, "mass_flow_kgs")
    lp_kg = result_lp.get_result_for_id(pipe_id, "linepack_kg")
    lp0 = float(lp_kg.values[0])
    steps = list(range(len(DEMAND)))
    feed_lp = [-float(v) for v in src_lp.values]
    feed_nolp = [-float(v) for v in src_nolp.values]
    delta = [float(v) - lp0 for v in lp_kg.values]

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.09,
        subplot_titles=(
            "Consumer demand",
            "Source feed rate",
            "Linepack: stored mass deviation from initial",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=DEMAND,
            mode="lines",
            line_shape="hv",
            line={"color": C_LOAD, "width": 3},
            name="demand",
            fill="tozeroy",
            fillcolor="rgba(244,162,97,0.15)",
            hovertemplate="hour %{x}: %{y:.2f} kg/s<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=feed_nolp,
            mode="lines",
            line_shape="hv",
            line={"color": C_DIS, "width": 2, "dash": "dash"},
            name="without linepack",
            hovertemplate="without linepack<br>hour %{x}: %{y:.3f} kg/s<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=feed_lp,
            mode="lines",
            line_shape="hv",
            line={"color": C_LP, "width": 2},
            name="with linepack",
            hovertemplate="with linepack<br>hour %{x}: %{y:.3f} kg/s<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=DEMAND,
            mode="lines",
            line_shape="hv",
            line={"color": C_LOAD, "width": 1, "dash": "dot"},
            name="demand (ref.)",
            opacity=0.6,
            hovertemplate="demand (ref.)<br>hour %{x}: %{y:.2f} kg/s<extra></extra>",
        ),
        row=2,
        col=1,
    )
    pos = [v if v >= 0 else 0.0 for v in delta]
    neg = [v if v < 0 else 0.0 for v in delta]
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=pos,
            mode="lines",
            line_shape="hv",
            line={"color": C_ACCENT, "width": 0},
            name="charging",
            fill="tozeroy",
            fillcolor="rgba(26,150,65,0.20)",
            hoverinfo="skip",
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=neg,
            mode="lines",
            line_shape="hv",
            line={"color": C_DIS, "width": 0},
            name="discharging",
            fill="tozeroy",
            fillcolor="rgba(215,25,28,0.20)",
            hoverinfo="skip",
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=delta,
            mode="lines",
            line_shape="hv",
            line={"color": C_ACCENT, "width": 2},
            name="linepack",
            hovertemplate="hour %{x}: %{y:,.0f} kg "
            + f"(initial {lp0:,.0f} kg)<extra></extra>",
        ),
        row=3,
        col=1,
    )
    fig.add_hline(
        y=0,
        line={"color": "rgba(128,128,128,0.5)", "width": 1, "dash": "dash"},
        row=3,
        col=1,
    )
    _base_layout(fig, "Gas linepack buffers source from demand variation", height=640)
    fig.update_xaxes(title_text="Hour", row=3, col=1)
    fig.update_yaxes(title_text="Flow  [kg/s]", row=1, col=1)
    fig.update_yaxes(title_text="Feed rate  [kg/s]", row=2, col=1)
    fig.update_yaxes(title_text="Δ stored mass  [kg]", row=3, col=1)
    return _write(fig, out_path, height_px=640)


def build_concepts_timeseries(out_path):
    """Six-step load profile: demand, external grid flow and bus voltage track the
    varying demand. Mirrors the concepts/timeseries.rst quick-start figure."""
    LOAD_PROFILE = [0.4, 0.8, 1.2, 1.0, 0.6, 0.3]
    STEPS = list(range(len(LOAD_PROFILE)))

    net = mx.create_multi_energy_network()
    bus0 = mx.create_bus(net)
    bus1 = mx.create_bus(net)
    mx.create_ext_power_grid(net, bus0)
    mx.create_line(net, bus0, bus1, length_m=500, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)
    mx.create_power_load(net, bus1, p_mw=1.0, q_mvar=0.0, name="demand")

    td = TimeseriesData()
    td.add_child_series_by_name("demand", "p_mw", LOAD_PROFILE)
    result = run_timeseries(net, td)

    vm_df = result.get_result_for(mm.Bus, "vm_pu")
    ext_df = result.get_result_for(mm.ExtPowerGrid, "p_mw")
    vm1 = [float(x) for x in vm_df.iloc[:, 1].values]
    grid_p = [float(x) for x in ext_df.iloc[:, 0].values]

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.09,
        subplot_titles=("Demand profile", "External grid flow", "Bus 1 voltage"),
    )
    fig.add_trace(
        go.Scatter(
            x=STEPS,
            y=LOAD_PROFILE,
            mode="lines",
            line_shape="hv",
            line={"color": C_LOAD, "width": 3},
            name="load",
            fill="tozeroy",
            fillcolor="rgba(244,162,97,0.18)",
            hovertemplate="step %{x}: %{y:.2f} MW<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=STEPS,
            y=grid_p,
            marker_color=C_GAS,
            marker_opacity=0.8,
            name="grid import",
            hovertemplate="step %{x}: %{y:.3f} MW<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=STEPS,
            y=vm1,
            mode="lines+markers",
            line={"color": C_ACCENT, "width": 3},
            marker={"size": 8},
            name="voltage",
            hovertemplate="step %{x}: %{y:.4f} pu<extra></extra>",
        ),
        row=3,
        col=1,
    )
    _base_layout(fig, "Timeseries simulation: quick start", height=640)
    fig.update_xaxes(title_text="step", tickmode="array", tickvals=STEPS, row=3, col=1)
    fig.update_yaxes(title_text="Load  [MW]", row=1, col=1)
    fig.update_yaxes(title_text="Grid import  [MW]", row=2, col=1)
    fig.update_yaxes(title_text="Voltage  [pu]", row=3, col=1)
    return _write(fig, out_path, height_px=640)


def build_howto_multi_period_1(out_path):
    """Battery optimal dispatch over a 6-hour horizon (how-to/multi_period.rst):
    controllable storage shifts charge to off-peak to serve the midday peak."""
    LOAD = [0.4, 0.5, 1.4, 1.8, 1.5, 0.4]
    net = mx.create_multi_energy_network()
    bus0 = mx.create_bus(net)
    bus1 = mx.create_bus(net)
    mx.create_ext_power_grid(net, bus0)
    mx.create_line(net, bus0, bus1, length_m=500, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)
    mx.create_power_load(net, bus1, p_mw=0.0, q_mvar=0.0, name="load")
    storage = mm.ElectricStorage(e_mwh_initial=2.0, e_mwh_max=4.0, p_max_mw=1.0)
    bat = mx.create_el_child(net, storage, node_id=bus1, name="battery")

    td = TimeseriesData()
    td.add_child_series_by_name("load", "p_mw", LOAD)
    prob = OptimizationProblem()
    prob.controllable_storages()
    result = run_multi_period(
        net,
        td,
        optimization_problem=prob,
        dt_h=1.0,
        terminal_state={(bat, "e_mwh"): 2.0},
    )

    soc = [float(x) for x in result.get_result_for_id(bat, "e_mwh").values]
    disp = [float(x) for x in result.get_result_for_id(bat, "p_mw").values]
    hours = list(range(len(LOAD)))
    bar_colors = [C_CHG if v >= 0 else C_DIS for v in disp]

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.09,
        subplot_titles=(
            "Consumer demand",
            "Optimised dispatch  (+ charge / - discharge)",
            "State of charge",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=hours,
            y=LOAD,
            mode="lines",
            line_shape="hv",
            line={"color": C_LOAD, "width": 3},
            name="load",
            fill="tozeroy",
            fillcolor="rgba(244,162,97,0.15)",
            hovertemplate="hour %{x}: %{y:.2f} MW<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=hours,
            y=disp,
            marker_color=bar_colors,
            name="dispatch",
            hovertemplate="hour %{x}: %{y:.2f} MW<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=hours,
            y=soc,
            mode="lines+markers",
            line={"color": C_SOC, "width": 3},
            marker={"size": 8},
            name="SoC",
            fill="tozeroy",
            fillcolor="rgba(26,150,65,0.12)",
            hovertemplate="hour %{x}: %{y:.2f} MWh<extra></extra>",
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[hours[0], hours[-1]],
            y=[4.0, 4.0],
            mode="lines",
            line={"color": C_DIS, "width": 1.5, "dash": "dot"},
            name="capacity",
            hovertemplate="capacity 4 MWh<extra></extra>",
        ),
        row=3,
        col=1,
    )
    _base_layout(fig, "Battery optimal dispatch over 6-hour horizon", height=640)
    fig.update_xaxes(title_text="hour", row=3, col=1)
    fig.update_yaxes(title_text="MW", row=1, col=1)
    fig.update_yaxes(title_text="MW", row=2, col=1)
    fig.update_yaxes(title_text="MWh", range=[0, 4.5], row=3, col=1)
    return _write(fig, out_path, height_px=640)


def build_howto_multi_period_chp(out_path):
    """CHP multi-period dispatch: regulation tracks combined electrical and heat
    demand (how-to/multi_period.rst; controllable_cps, queries CHPControlNode)."""
    EL_PROF = [0.8, 1.0, 1.4, 1.8, 1.6, 1.2]
    HEAT_PROF = [0.4, 0.5, 0.7, 0.9, 0.8, 0.6]

    net_mes = mx.create_multi_energy_network()
    bus_slack = mx.create_bus(net_mes)
    bus_load = mx.create_bus(net_mes)
    mx.create_ext_power_grid(net_mes, bus_slack)
    mx.create_line(
        net_mes, bus_slack, bus_load, length_m=200, r_ohm_per_m=1e-4, x_ohm_per_m=1e-4
    )
    mx.create_power_load(net_mes, bus_load, p_mw=0.0, q_mvar=0.0, name="el_load")
    j_gas = mx.create_gas_junction(net_mes)
    j_supply = mx.create_water_junction(net_mes)
    j_return = mx.create_water_junction(net_mes)
    mx.create_gas_ext_grid(net_mes, j_gas)
    mx.create_ext_hydr_grid(net_mes, j_supply)
    mx.create_water_sink(net_mes, j_return, mass_flow_kgs=0.0, name="heat_load")
    mx.create_chp(
        net_mes,
        power_node_id=bus_load,
        gas_node_id=j_gas,
        heat_node_id=j_supply,
        heat_return_node_id=j_return,
        diameter_m=0.1,
        efficiency_power=0.35,
        efficiency_heat=0.45,
        mass_flow_setpoint_kgs=0.1,
    )

    td_mes = TimeseriesData()
    td_mes.add_child_series_by_name("el_load", "p_mw", EL_PROF)
    td_mes.add_child_series_by_name("heat_load", "mass_flow_kgs", HEAT_PROF)
    prob = mp.OptimizationProblem()
    prob.controllable_cps(["regulation"])
    result_mes = run_multi_period(net_mes, td_mes, dt_h=1.0, optimization_problem=prob)
    chp_reg = [
        float(x)
        for x in result_mes.get_result_for(mm.CHPControlNode, "regulation")
        .iloc[:, 0]
        .values
    ]
    steps = list(range(len(EL_PROF)))

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.09,
        subplot_titles=("Electrical demand", "Heat demand", "CHP dispatch"),
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=EL_PROF,
            mode="lines",
            line_shape="hv",
            line={"color": C_GAS, "width": 3},
            name="electric",
            fill="tozeroy",
            fillcolor="rgba(44,123,182,0.15)",
            hovertemplate="period %{x}: %{y:.2f} MW<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=HEAT_PROF,
            mode="lines",
            line_shape="hv",
            line={"color": C_HEAT, "width": 3},
            name="heat",
            fill="tozeroy",
            fillcolor="rgba(215,25,28,0.15)",
            hovertemplate="period %{x}: %{y:.2f} kg/s<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=chp_reg,
            mode="lines+markers",
            line={"color": C_ACCENT, "width": 3},
            marker={"size": 8},
            name="regulation",
            fill="tozeroy",
            fillcolor="rgba(26,150,65,0.12)",
            hovertemplate="period %{x}: %{y:.3f}<extra></extra>",
        ),
        row=3,
        col=1,
    )
    _base_layout(fig, "CHP multi-period dispatch", height=640)
    fig.update_xaxes(title_text="period", row=3, col=1)
    fig.update_yaxes(title_text="MW", row=1, col=1)
    fig.update_yaxes(title_text="kg/s", row=2, col=1)
    fig.update_yaxes(title_text="regulation [-]", range=[0, 1.1], row=3, col=1)
    return _write(fig, out_path, height_px=640)


def build_howto_multi_period_linepack(out_path):
    """Gas linepack buffers the demand peak (how-to/multi_period.rst): stored mass
    rises at low demand and drains during the peak."""
    DEMAND = [0.3, 0.3, 0.6, 0.9, 0.8, 0.4]
    net_lp = mx.create_multi_energy_network()
    j0 = mx.create_gas_junction(net_lp)
    j1 = mx.create_gas_junction(net_lp)
    j2 = mx.create_gas_junction(net_lp)
    mx.create_gas_ext_grid(net_lp, j0)
    mx.create_gas_sink(net_lp, j2, mass_flow_kgs=0.3, name="consumer")
    pipe_id = mx.create_gas_pipe(net_lp, j0, j1, diameter_m=0.5, length_m=50_000)
    mx.create_gas_pipe(net_lp, j1, j2, diameter_m=0.3, length_m=10_000)
    net_lp.add_extension(GasLinepack())
    td_lp = TimeseriesData()
    td_lp.add_child_series_by_name("consumer", "mass_flow_kgs", DEMAND)
    result = run_multi_period(net_lp, td_lp, dt_h=1.0)
    lp_vals = [
        float(x) for x in result.get_result_for_id(pipe_id, "linepack_kg").values
    ]
    lp0 = lp_vals[0]
    steps = list(range(len(DEMAND)))

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=("Pipeline stored mass", "Consumer demand"),
    )
    fig.add_trace(
        go.Scatter(
            x=[steps[0], steps[-1]],
            y=[lp0, lp0],
            mode="lines",
            line={"color": "rgba(128,128,128,0.7)", "width": 1.5, "dash": "dash"},
            name="initial",
            hovertemplate=f"initial {lp0:,.0f} kg<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=lp_vals,
            mode="lines+markers",
            line={"color": C_LP, "width": 3},
            marker={"size": 8},
            name="linepack",
            fill="tonexty",
            fillcolor="rgba(44,123,182,0.18)",
            hovertemplate="hour %{x}: %{y:,.0f} kg<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=DEMAND,
            mode="lines",
            line_shape="hv",
            line={"color": C_LOAD, "width": 3},
            name="demand",
            fill="tozeroy",
            fillcolor="rgba(244,162,97,0.15)",
            hovertemplate="hour %{x}: %{y:.2f} kg/s<extra></extra>",
        ),
        row=2,
        col=1,
    )
    _base_layout(fig, "Gas linepack buffers demand peak", height=480)
    fig.update_xaxes(title_text="hour", row=2, col=1)
    fig.update_yaxes(title_text="kg", row=1, col=1)
    fig.update_yaxes(title_text="kg/s", row=2, col=1)
    return _write(fig, out_path, height_px=480)


def build_tutorial_timeseries(out_path):
    """Solar feeder day-ahead simulation (tutorials/02): load/PV power, external
    grid import/export and the residential bus voltage over eight 3-hour slots."""
    load_profile = [0.10, 0.10, 0.15, 0.20, 0.25, 0.35, 0.40, 0.25]
    pv_profile = [0.00, 0.00, -0.10, -0.30, -0.45, -0.30, -0.10, 0.00]
    hours = [0, 3, 6, 9, 12, 15, 18, 21]

    net = mx.create_multi_energy_network()
    bus_grid = mx.create_bus(net)
    bus_home = mx.create_bus(net)
    mx.create_line(net, bus_grid, bus_home, 500, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)
    mx.create_ext_power_grid(net, bus_grid)
    load_id = mx.create_power_load(net, bus_home, p_mw=0.30, q_mvar=0.0)
    pv_id = mx.create_power_load(net, bus_home, p_mw=0.0, q_mvar=0.0)

    td = TimeseriesData()
    td.add_child_series(load_id, "p_mw", load_profile)
    td.add_child_series(pv_id, "p_mw", pv_profile)
    ts_result = run_timeseries(net, td)

    vm_df = ts_result.get_result_for(mm.Bus, "vm_pu")
    ext_df = ts_result.get_result_for(mm.ExtPowerGrid, "p_mw")
    vm_home = [float(v) for v in vm_df.iloc[:, 1].values]
    grid_imp = [float(v) for v in ext_df.iloc[:, 0].values]
    pv_output = [-p for p in pv_profile]

    c_load, c_pv, c_vm, c_imp, c_exp = (
        "#f4a261",
        "#ffe066",
        "#2c7bb6",
        "#d7191c",
        "#1a9641",
    )
    imp_colors = [c_imp if v >= 0 else c_exp for v in grid_imp]

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.09,
        subplot_titles=(
            "Load and PV generation",
            "External grid import / export",
            "Residential bus voltage",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=hours,
            y=load_profile,
            mode="lines",
            line_shape="hv",
            line={"color": c_load, "width": 2},
            name="Load",
            fill="tozeroy",
            fillcolor="rgba(244,162,97,0.15)",
            hovertemplate="%{x:02d}:00 - load %{y:.2f} MW<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=hours,
            y=pv_output,
            mode="lines",
            line_shape="hv",
            line={"color": c_pv, "width": 2},
            name="PV output",
            fill="tozeroy",
            fillcolor="rgba(255,224,102,0.15)",
            hovertemplate="%{x:02d}:00 - PV %{y:.2f} MW<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=hours,
            y=grid_imp,
            marker_color=imp_colors,
            width=2.4,
            name="grid flow",
            hovertemplate="%{x:02d}:00 - %{y:.3f} MW<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=hours,
            y=vm_home,
            mode="lines+markers",
            line={"color": c_vm, "width": 2},
            marker={"size": 8},
            name="voltage",
            hovertemplate="%{x:02d}:00 - %{y:.4f} pu<extra></extra>",
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[hours[0], hours[-1]],
            y=[0.97, 0.97],
            mode="lines",
            line={"color": "grey", "width": 1.5, "dash": "dash"},
            name="undervoltage limit (0.97 pu)",
            hoverinfo="skip",
        ),
        row=3,
        col=1,
    )
    _base_layout(fig, "Solar feeder: day-ahead simulation", height=640)
    ticktext = [f"{h:02d}:00" for h in hours]
    fig.update_xaxes(tickvals=hours, ticktext=ticktext, row=1, col=1)
    fig.update_xaxes(tickvals=hours, ticktext=ticktext, row=2, col=1)
    fig.update_xaxes(
        tickvals=hours, ticktext=ticktext, title_text="Hour of day", row=3, col=1
    )
    fig.update_yaxes(title_text="Power [MW]", row=1, col=1)
    fig.update_yaxes(title_text="Grid flow [MW] (+ import / - export)", row=2, col=1)
    fig.update_yaxes(title_text="Voltage [pu]", row=3, col=1)
    return _write(fig, out_path, height_px=640)


def build_benchmark_backend(out_path):
    """Backend performance comparison (benchmarks/backend_selection.md): two
    head-to-head solver-backend shoot-outs across representative cases. Reads the
    committed CSV and reproduces backend_comparison.py's make_plot interactively."""
    df = pd.read_csv(_bench_csv("results", "backend_comparison.csv"))

    color = {
        "GEKKO": CB_PURPLE,
        "CasADi": CB_GREEN,
        "pyomo-gurobi": CB_BLUE,
        "gurobipy": CB_ORANGE,
    }
    # Hatch the reference backend in each pairing (GEKKO in group A, pyomo-gurobi
    # in group B); the native engine stays solid. Pattern carries the contrast so
    # the comparison reads in greyscale and under colour blindness.
    hatch = {
        "GEKKO": "/",
        "CasADi": "",
        "pyomo-gurobi": "/",
        "gurobipy": "",
    }
    GRID = "rgba(80,80,80,0.12)"
    AXIS_LINE = "rgba(128,128,128,0.5)"
    groups = [
        ("A", "Group A: GEKKO vs CasADi (smooth NLP)"),
        ("B", "Group B: Pyomo/Gurobi vs native gurobipy ((MI)QCQP)"),
    ]
    counts = [int((df.group == g[0]).sum()) for g in groups]
    total_cases = max(sum(counts), 1)

    fig = make_subplots(
        rows=2,
        cols=2,
        column_widths=[0.62, 0.38],
        row_heights=[c / total_cases for c in counts],
        shared_yaxes=True,
        horizontal_spacing=0.06,
        vertical_spacing=0.12,
    )

    for r, (g, _title) in enumerate(groups, start=1):
        sub = df[df.group == g].iloc[::-1]
        cases = sub.case.tolist()
        last_row = r == len(groups)
        ba = sub.backend_a.iloc[0]
        bb = sub.backend_b.iloc[0]
        fig.add_trace(
            go.Bar(
                y=cases,
                x=sub.time_a_ms,
                name=ba,
                orientation="h",
                marker=_bar_marker(color.get(ba, "#888"), pattern=hatch.get(ba)),
                legendgroup=ba,
                showlegend=True,
                cliponaxis=False,
                text=[f"{v:.0f}" for v in sub.time_a_ms],
                textposition="outside",
                textfont={"size": 12, "color": TEXT},
                hovertemplate=f"{ba}: %{{x:.1f}} ms<extra></extra>",
            ),
            row=r,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                y=cases,
                x=sub.time_b_ms,
                name=bb,
                orientation="h",
                marker=_bar_marker(color.get(bb, "#888"), pattern=hatch.get(bb)),
                legendgroup=bb,
                showlegend=True,
                cliponaxis=False,
                text=[f"{v:.0f}" for v in sub.time_b_ms],
                textposition="outside",
                textfont={"size": 12, "color": TEXT},
                hovertemplate=f"{bb}: %{{x:.1f}} ms<extra></extra>",
            ),
            row=r,
            col=1,
        )
        spd = sub.speedup
        fig.add_trace(
            go.Bar(
                y=cases,
                x=spd,
                orientation="h",
                showlegend=False,
                cliponaxis=False,
                marker=_bar_marker(
                    [color.get(bb, "#888") if v >= 1 else CB_VERMILION for v in spd]
                ),
                text=[f"×{v:.1f}" for v in spd],
                textposition="outside",
                textfont={"size": 12, "color": TEXT},
                hovertemplate="speedup ×%{x:.2f}<extra></extra>",
            ),
            row=r,
            col=2,
        )
        tvals = sub[["time_a_ms", "time_b_ms"]].to_numpy(dtype=float)
        fig.update_xaxes(
            type="log",
            nticks=6,
            row=r,
            col=1,
            title_text="solve time (ms, log)" if last_row else None,
            range=[np.log10(np.nanmin(tvals) * 0.5), np.log10(np.nanmax(tvals) * 3.4)],
        )
        fig.update_xaxes(
            row=r,
            col=2,
            title_text="× faster (native vs reference)" if last_row else None,
            range=[0, float(np.nanmax(spd)) * 1.32],
        )
        fig.add_vline(x=1.0, line_dash="dot", line_color=AXIS_LINE, row=r, col=2)
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
            font={"size": 14, "color": TEXT},
            xanchor="left",
        )

    fig.update_xaxes(
        showgrid=True,
        gridcolor=GRID,
        gridwidth=1,
        zeroline=False,
        showline=False,
        ticks="",
        tickfont={"size": 13, "color": TEXT},
        title_font={"size": 14, "color": TEXT},
    )
    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        showline=True,
        linecolor=AXIS_LINE,
        linewidth=1,
        tickfont={"size": 13, "color": TEXT},
        automargin=True,
    )

    height_px = int(70 * total_cases + 110 * len(groups) + 170) / 2
    fig.update_layout(
        barmode="group",
        bargap=0.25,
        bargroupgap=0.1,
        template="plotly_white",
        autosize=True,
        height=height_px,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.03,
            "xanchor": "right",
            "x": 1.0,
            "font": {"size": 13, "color": TEXT},
            "bgcolor": "rgba(0,0,0,0)",
        },
        margin={"l": 60, "r": 60, "t": 50, "b": 70},
        font={"family": FONT_FAMILY, "size": 13, "color": TEXT},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        uniformtext={"mode": "hide", "minsize": 8},
    )
    return _write(fig, out_path, height_px=height_px)


def build_benchmark_pandapower(out_path):
    """monee (electric, smooth-NLP) vs pandapower native AC solvers, rendered from
    the committed benchmark CSV. Solve time, voltage agreement and slack-power
    agreement across AC power flow, OPF and line-limited OPF."""
    C_PANDAPOWER = CB_VERMILION
    C_MONEE = CB_GREEN
    C_VM = CB_BLUE
    C_P = CB_ORANGE
    PP_TEXT = TEXT
    PP_GRID = "rgba(80,80,80,0.12)"
    PP_AXIS_LINE = "rgba(128,128,128,0.5)"
    PANDAPOWER = "pandapower"
    CASADI = "monee · CasADi"

    df = pd.read_csv(_bench_csv("results", "pandapower_comparison.csv"))

    def _log_range(vals, floor=1e-12, lo_pad=0.3, hi_pad=25.0):
        v = np.asarray(vals, float)
        v = v[np.isfinite(v)]
        v = np.clip(v, floor, None)
        if v.size == 0:
            return [np.log10(floor), 0.0]
        return [np.log10(v.min() * lo_pad), np.log10(v.max() * hi_pad)]

    color = {PANDAPOWER: C_PANDAPOWER, CASADI: C_MONEE}
    # Hatch the reference engine so the paired solve-time bars never rely on hue.
    hatch = {PANDAPOWER: "/", CASADI: ""}
    groups = [
        ("PF", "Power flow (AC): pandapower runpp vs monee CasADi"),
        ("OPF", "Optimal power flow, no line limit: pandapower runopp vs monee CasADi"),
        (
            "OPF-LL",
            "Optimal power flow, line-loading limit binds: pandapower runopp vs monee CasADi",
        ),
    ]
    groups = [g for g in groups if (df.group == g[0]).any()]
    counts = [int((df.group == g[0]).sum()) for g in groups]
    total_cases = max(sum(counts), 1)
    n_groups = len(groups)

    fig = make_subplots(
        rows=n_groups,
        cols=3,
        column_widths=[0.46, 0.27, 0.27],
        row_heights=[c / total_cases for c in counts],
        shared_yaxes=True,
        horizontal_spacing=0.06,
        vertical_spacing=0.12,
    )

    for r, (g, _title) in enumerate(groups, start=1):
        sub = df[df.group == g].iloc[::-1]
        cases = sub.case.tolist()
        last_row = r == n_groups
        for col, backend in [("t_pandapower_ms", PANDAPOWER), ("t_casadi_ms", CASADI)]:
            fig.add_trace(
                go.Bar(
                    y=cases,
                    x=sub[col],
                    name=backend,
                    orientation="h",
                    marker=_bar_marker(color[backend], pattern=hatch[backend]),
                    legendgroup=backend,
                    showlegend=(r == 1),
                    cliponaxis=False,
                    text=[f"{v:.0f}" for v in sub[col]],
                    textposition="outside",
                    textfont={"size": 12, "color": PP_TEXT},
                    hovertemplate=f"{backend}: %{{x:.1f}} ms<extra></extra>",
                ),
                row=r,
                col=1,
            )
        tvals = sub[["t_pandapower_ms", "t_casadi_ms"]].to_numpy(dtype=float)
        fig.update_xaxes(
            type="log",
            nticks=6,
            row=r,
            col=1,
            title_text="solve time (ms, log)" if last_row else None,
            range=[np.log10(np.nanmin(tvals) * 0.5), np.log10(np.nanmax(tvals) * 3.4)],
        )
        vm = np.clip(sub.vm_err_pu.to_numpy(float), 1e-12, None)
        fig.add_trace(
            go.Bar(
                y=cases,
                x=vm,
                orientation="h",
                showlegend=False,
                marker=_bar_marker(C_VM),
                cliponaxis=False,
                text=[f"{v:.1e}" for v in vm],
                textposition="outside",
                textfont={"size": 12, "color": PP_TEXT},
                hovertemplate="|Δvm| %{x:.2e} pu vs pandapower<extra></extra>",
            ),
            row=r,
            col=2,
        )
        fig.update_xaxes(
            type="log",
            nticks=4,
            row=r,
            col=2,
            title_text="|Δvm| (pu, log)" if last_row else None,
            range=_log_range(sub.vm_err_pu),
        )
        pw = np.clip(sub.p_err_mw.to_numpy(float), 1e-12, None)
        fig.add_trace(
            go.Bar(
                y=cases,
                x=pw,
                orientation="h",
                showlegend=False,
                marker=_bar_marker(C_P),
                cliponaxis=False,
                text=[f"{v:.1e}" for v in pw],
                textposition="outside",
                textfont={"size": 12, "color": PP_TEXT},
                hovertemplate="|ΔP| %{x:.2e} MW vs pandapower<extra></extra>",
            ),
            row=r,
            col=3,
        )
        fig.update_xaxes(
            type="log",
            nticks=4,
            row=r,
            col=3,
            title_text="|ΔP| (MW, log)" if last_row else None,
            range=_log_range(sub.p_err_mw),
        )
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
            font={"size": 14, "color": PP_TEXT},
            xanchor="left",
        )

    fig.update_xaxes(
        showgrid=True,
        gridcolor=PP_GRID,
        gridwidth=1,
        zeroline=False,
        showline=False,
        ticks="",
        tickfont={"size": 13, "color": PP_TEXT},
        title_font={"size": 14, "color": PP_TEXT},
    )
    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        showline=True,
        linecolor=PP_AXIS_LINE,
        linewidth=1,
        tickfont={"size": 13, "color": PP_TEXT},
        automargin=True,
    )
    height_px = int(80 * total_cases + 110 * n_groups + 170) / 2
    fig.update_layout(
        barmode="group",
        bargap=0.3,
        bargroupgap=0.08,
        template="plotly_white",
        autosize=True,
        height=height_px,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.028,
            "xanchor": "right",
            "x": 1.0,
            "font": {"size": 13, "color": PP_TEXT},
            "bgcolor": "rgba(0,0,0,0)",
        },
        margin={"l": 155, "r": 60, "t": 50, "b": 70},
        font={
            "family": "Inter, Segoe UI, Helvetica, Arial",
            "size": 13,
            "color": PP_TEXT,
        },
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        uniformtext={"mode": "hide", "minsize": 8},
    )
    return _write(fig, out_path, height_px=height_px)


def build_benchmark_pandapipes(out_path):
    """monee (multi-energy NLP) vs pandapipes, rendered from the committed CSV:
    solve time, pressure and (where heat participates) temperature agreement for
    gas, heat and coupled MES flow."""
    df = pd.read_csv(_bench_csv("results", "pandapipes_comparison.csv"))

    PANDAPIPES = "pandapipes"
    MONEE = "monee · CasADi"
    C_PANDAPIPES = CB_VERMILION
    C_MONEE = CB_GREEN
    C_PRESSURE = CB_BLUE
    C_TEMP = CB_ORANGE
    C_BANNER = TEXT
    GRID = "rgba(80,80,80,0.12)"
    AXIS_LINE = "rgba(128,128,128,0.5)"
    AXIS_TITLE_SIZE = 14
    TICK_SIZE = 13
    LABEL_SIZE = 12
    BANNER_SIZE = 14

    color = {PANDAPIPES: C_PANDAPIPES, MONEE: C_MONEE}
    # Hatch the reference engine so the two solve-time bars stay distinct without
    # relying on the vermillion-vs-green hue contrast.
    hatch = {PANDAPIPES: "/", MONEE: ""}
    groups = [
        ("GAS", "Gas hydraulics: pandapipes pipeflow vs monee Weymouth NLP"),
        ("HEAT", "Water hydraulics + thermal: pandapipes pipeflow vs monee Darcy NLP"),
        ("MES", "Coupled MES (P2G + G2P + CHP): pandapipes multinet vs monee NLP"),
    ]
    groups = [g for g in groups if (df.group == g[0]).any()]
    counts = [int((df.group == g[0]).sum()) for g in groups]
    total_cases = max(sum(counts), 1)
    n_groups = len(groups)

    fig = make_subplots(
        rows=n_groups,
        cols=3,
        column_widths=[0.46, 0.27, 0.27],
        row_heights=[c / total_cases for c in counts],
        shared_yaxes=True,
        horizontal_spacing=0.06,
        vertical_spacing=0.08,
    )

    for r, (g, _t) in enumerate(groups, start=1):
        sub = df[df.group == g].iloc[::-1]
        cases = sub.case.tolist()
        last_row = r == n_groups
        for col, backend in [("t_pandapipes_ms", PANDAPIPES), ("t_monee_ms", MONEE)]:
            fig.add_trace(
                go.Bar(
                    y=cases,
                    x=sub[col],
                    name=backend,
                    orientation="h",
                    marker=_bar_marker(color[backend], pattern=hatch[backend]),
                    legendgroup=backend,
                    showlegend=(r == 1),
                    cliponaxis=False,
                    text=[f"{v:.1f}" for v in sub[col]],
                    textposition="outside",
                    textfont={"size": LABEL_SIZE, "color": TEXT},
                    hovertemplate=f"{backend}: %{{x:.2f}} ms<extra></extra>",
                ),
                row=r,
                col=1,
            )
        tvals = sub[["t_pandapipes_ms", "t_monee_ms"]].to_numpy(float)
        fig.update_xaxes(
            type="log",
            nticks=6,
            row=r,
            col=1,
            title_text="solve time (ms, log)" if last_row else None,
            range=[np.log10(np.nanmin(tvals) * 0.5), np.log10(np.nanmax(tvals) * 3.4)],
        )
        p = sub.p_reldiff_pct.to_numpy(float)
        fig.add_trace(
            go.Bar(
                y=cases,
                x=p,
                orientation="h",
                showlegend=False,
                marker=_bar_marker(C_PRESSURE),
                cliponaxis=False,
                text=[f"{v:.1f}%" for v in p],
                textposition="outside",
                textfont={"size": LABEL_SIZE, "color": TEXT},
                hovertemplate="pressure diff %{x:.2f}% of drop<extra></extra>",
            ),
            row=r,
            col=2,
        )
        fig.update_xaxes(
            row=r,
            col=2,
            title_text="pressure diff (% of drop)" if last_row else None,
            range=[0, max(float(np.nanmax(p)) * 1.5, 0.1)],
        )
        t = sub.t_err_k.to_numpy(float)
        finite = np.isfinite(t)
        fig.add_trace(
            go.Bar(
                y=cases,
                x=[v if f else None for v, f in zip(t, finite)],
                orientation="h",
                showlegend=False,
                marker=_bar_marker(C_TEMP),
                cliponaxis=False,
                text=[f"{v:.3g}" if f else "" for v, f in zip(t, finite)],
                textposition="outside",
                textfont={"size": LABEL_SIZE, "color": TEXT},
                hovertemplate="ΔT %{x:.4g} K<extra></extra>",
            ),
            row=r,
            col=3,
        )
        if finite.any():
            fig.update_xaxes(
                row=r,
                col=3,
                title_text="temperature diff (K)" if last_row else None,
                range=[0, float(np.nanmax(t[finite])) * 1.5],
            )
        else:
            fig.update_xaxes(
                row=r,
                col=3,
                range=[0, 1],
                showticklabels=False,
                title_text="temperature diff (K)" if last_row else None,
            )
            fig.add_annotation(
                text="isothermal, no ΔT",
                row=r,
                col=3,
                xref="x domain",
                yref="y domain",
                x=0.5,
                y=0.5,
                showarrow=False,
                font={"size": 13, "color": TEXT},
                xanchor="center",
            )
        fig.add_annotation(
            text=f"<b>{_t}</b>",
            row=r,
            col=1,
            xref="x domain",
            yref="y domain",
            x=0,
            y=1.0,
            yshift=16,
            showarrow=False,
            font={"size": BANNER_SIZE, "color": C_BANNER},
            xanchor="left",
        )

    fig.update_xaxes(
        showgrid=True,
        gridcolor=GRID,
        gridwidth=1,
        zeroline=False,
        showline=False,
        ticks="",
        tickfont={"size": TICK_SIZE, "color": TEXT},
        title_font={"size": AXIS_TITLE_SIZE, "color": TEXT},
    )
    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        showline=True,
        linecolor=AXIS_LINE,
        linewidth=1,
        tickfont={"size": TICK_SIZE, "color": TEXT},
        automargin=True,
    )

    height = int(80 * total_cases + 110 * n_groups + 170) / 2
    fig.update_layout(
        barmode="group",
        bargap=0.32,
        bargroupgap=0.12,
        template="plotly_white",
        autosize=True,
        height=height,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.012,
            "xanchor": "right",
            "x": 1.0,
            "font": {"size": 13, "color": TEXT},
            "bgcolor": "rgba(0,0,0,0)",
        },
        margin={"l": 155, "r": 40, "t": 50, "b": 60},
        font={"family": FONT_FAMILY, "size": 13, "color": TEXT},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        uniformtext={"mode": "hide", "minsize": 8},
    )
    return _write(fig, out_path, height_px=height)


# Registry: (output filename under _static/interactive/, builder function).
GENERATORS = [
    ("benchmark_backend.html", build_benchmark_backend),
    ("benchmark_pandapower.html", build_benchmark_pandapower),
    ("benchmark_pandapipes.html", build_benchmark_pandapipes),
    ("storage_dispatch.html", build_storage_dispatch),
    ("storage_prescribed.html", build_storage_prescribed),
    ("storage_gas.html", build_storage_gas),
    ("concepts_multi_period.html", build_concepts_multi_period),
    ("concepts_timeseries.html", build_concepts_timeseries),
    ("temporal_extensions_1.html", build_temporal_extensions_1),
    ("temporal_extensions_2.html", build_temporal_extensions_2),
    ("howto_multi_period_1.html", build_howto_multi_period_1),
    ("howto_multi_period_chp.html", build_howto_multi_period_chp),
    ("howto_multi_period_linepack.html", build_howto_multi_period_linepack),
    ("tutorial_timeseries.html", build_tutorial_timeseries),
]


def _generate_all(app):
    static_dir = os.path.join(app.srcdir, "_static", "interactive")
    for filename, builder in GENERATORS:
        out_path = os.path.join(static_dir, filename)
        try:
            builder(out_path)
            app.info(f"[interactive_plots] wrote {out_path}") if hasattr(
                app, "info"
            ) else None
        except Exception as exc:  # noqa: BLE001 - never fail the whole build on one figure
            print(f"[interactive_plots] WARNING: failed to build {filename}: {exc}")


def setup(app):
    app.connect("builder-inited", _generate_all)
    # Parent-page script that pushes the furo theme into the plot iframes.
    app.add_js_file("iframe_theme.js")
    return {"parallel_read_safe": True, "parallel_write_safe": True}

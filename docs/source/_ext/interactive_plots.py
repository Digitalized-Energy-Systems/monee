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

import functools
import importlib.util
import os
import platform
import sys
import warnings

import casadi
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

C_BLACK = "rgba(0,0,0,0)"
C_GRIDCOLOR = "rgba(80,80,80,0.12)"


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
        plot_bgcolor=C_BLACK,
        paper_bgcolor=C_BLACK,
    )
    fig.update_xaxes(showgrid=True, gridcolor=C_GRIDCOLOR, zeroline=False)
    fig.update_yaxes(
        showgrid=True,
        gridcolor=C_BLACK,
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
    disp = [
        float(x)
        for x in result.get_result_for_id(bat_id, "p_mw", mm.ElectricStorage).values
    ]
    soc = [
        float(x)
        for x in result.get_result_for_id(bat_id, "e_mwh", mm.ElectricStorage).values
    ]
    imp = [
        -float(x)
        for x in result.get_result_for_id(ext_id, "p_mw", mm.ExtPowerGrid).values
    ]
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

    disp, soc, _, __ = _solve_storage(capacity, capacity / 2.0, load, price)
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
    soc = [
        float(x)
        for x in result.get_result_for_id(bat_id, "e_mwh", mm.ElectricStorage).values
    ]
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
        float(x)
        for x in result_g.get_result_for_id(
            tank_id, "m_stored_kg", mm.GasStorage
        ).values
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

    soc = [
        float(x)
        for x in result.get_result_for_id(bat, "e_mwh", mm.ElectricStorage).values
    ]
    disp = [
        float(x)
        for x in result.get_result_for_id(bat, "p_mw", mm.ElectricStorage).values
    ]
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
    supply_temp = [356.0, 356.0, 356.0, 356.0, 340.0, 340.0, 340.0, 340.0]

    def build_net(with_ltc):
        net = mx.create_multi_energy_network()
        j_supply = mx.create_water_junction(net)
        j_mid = mx.create_water_junction(net)
        j_load = mx.create_water_junction(net)
        ext = mx.create_ext_hydr_grid(net, j_supply)
        mx.create_water_sink(net, j_load, mass_flow_kgs=5.0)
        mx.create_water_pipe(net, j_supply, j_mid, diameter_m=0.3, length_m=500)
        mx.create_water_pipe(net, j_mid, j_load, diameter_m=0.2, length_m=300)
        if with_ltc:
            net.add_extension(LumpedThermalCapacitance())
        return net, ext, j_supply, j_mid, j_load

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
        net, ext, j_supply, j_mid, j_load = build_net(with_ltc)
        td = TimeseriesData()
        td.add_child_series(ext, "t_k", supply_temp)
        result = run_timeseries(net, td)
        t_supply_s = result.get_result_for_id(j_supply, "t_pu", mm.Junction)
        t_mid_s = result.get_result_for_id(j_mid, "t_pu", mm.Junction)
        t_load_s = result.get_result_for_id(j_load, "t_pu", mm.Junction)
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
    fig.update_yaxes(title_text="Temperature  [pu]", range=[0.94, 1.01], row=1, col=1)
    fig.update_yaxes(range=[0.94, 1.01], row=1, col=2)
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
    src_lp = result_lp.get_result_for_id(j0, "mass_flow_kgs", mm.Junction)
    src_nolp = result_nolp.get_result_for_id(j0, "mass_flow_kgs", mm.Junction)
    lp_kg = result_lp.get_result_for_id(pipe_id, "linepack_kg", mm.GasPipe)
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

    soc = [
        float(x)
        for x in result.get_result_for_id(bat, "e_mwh", mm.ElectricStorage).values
    ]
    disp = [
        float(x)
        for x in result.get_result_for_id(bat, "p_mw", mm.ElectricStorage).values
    ]
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
        float(x)
        for x in result.get_result_for_id(pipe_id, "linepack_kg", mm.GasPipe).values
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
        gridcolor=C_GRIDCOLOR,
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
            "bgcolor": C_BLACK,
        },
        margin={"l": 60, "r": 60, "t": 50, "b": 70},
        font={"family": FONT_FAMILY, "size": 13, "color": TEXT},
        plot_bgcolor=C_BLACK,
        paper_bgcolor=C_BLACK,
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
    PP_AXIS_LINE = "rgba(128,128,128,0.5)"
    PANDAPOWER = "pandapower"
    CASADI = "monee via CasADi/IPOPT"

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
        ("PF", "Power flow (AC): pandapower runpp vs monee via CasADi/IPOPT"),
        (
            "OPF",
            "Optimal power flow, no line limit: pandapower runopp vs monee via CasADi/IPOPT",
        ),
        (
            "OPF-LL",
            "Optimal power flow, line-loading limit binds: pandapower runopp vs monee via CasADi/IPOPT",
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
        gridcolor=C_GRIDCOLOR,
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
            "bgcolor": C_BLACK,
        },
        margin={"l": 155, "r": 60, "t": 50, "b": 70},
        font={
            "family": "Inter, Segoe UI, Helvetica, Arial",
            "size": 13,
            "color": PP_TEXT,
        },
        plot_bgcolor=C_BLACK,
        paper_bgcolor=C_BLACK,
        uniformtext={"mode": "hide", "minsize": 8},
    )
    return _write(fig, out_path, height_px=height_px)


def build_benchmark_pandapipes(out_path):
    """monee (multi-energy NLP) vs pandapipes, rendered from the committed CSV:
    solve time, pressure and (where heat participates) temperature agreement for
    gas, heat and coupled MES flow."""
    df = pd.read_csv(_bench_csv("results", "pandapipes_comparison.csv"))

    PANDAPIPES = "pandapipes"
    MONEE = "monee via CasADi/IPOPT"
    C_PANDAPIPES = CB_VERMILION
    C_MONEE = CB_GREEN
    C_PRESSURE = CB_BLUE
    C_TEMP = CB_ORANGE
    C_BANNER = TEXT
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
        gridcolor=C_GRIDCOLOR,
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
            "bgcolor": C_BLACK,
        },
        margin={"l": 155, "r": 40, "t": 50, "b": 60},
        font={"family": FONT_FAMILY, "size": 13, "color": TEXT},
        plot_bgcolor=C_BLACK,
        paper_bgcolor=C_BLACK,
        uniformtext={"mode": "hide", "minsize": 8},
    )
    return _write(fig, out_path, height_px=height)


# Registry: (output filename under _static/interactive/, builder function).
C_CHP = CB_VERMILION
C_E_BOILER = CB_ORANGE
C_ELECTROLYSER = CB_GREEN
C_PLANT = CB_PURPLE
C_BATTERY = CB_PURPLE
C_IMPORT = CB_BLUE
C_PV = CB_YELLOW
# Reference lines are svg strokes the theme script does not recolour, so they
# take a grey that reads on the light and the dark page.
C_REFERENCE = "#8a8a8a"
C_DEMAND = "#9e9e9e"
C_BIOGAS = CB_SKY


@functools.lru_cache(maxsize=1)
def _mes_dispatch_run():
    """Solve examples/mes_economic_dispatch.py once for both of its figures, so
    the tutorial plots whatever the example currently computes."""
    path = os.path.join(_REPO_ROOT, "examples", "mes_economic_dispatch.py")
    spec = importlib.util.spec_from_file_location("mes_economic_dispatch", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hours, baselines, checks = module.run()
    if checks.failed:
        raise RuntimeError(f"mes_economic_dispatch checks failed: {checks.failed}")
    return module, hours, baselines


def _balance_bars(fig, hours_x, supply, use, row, shown):
    """Supply stacked above zero, use stacked below: in a balanced hour the two
    stacks mirror each other up to the losses."""
    for sign, series in ((1, supply), (-1, use)):
        for name, values, color in series:
            fig.add_trace(
                go.Bar(
                    x=hours_x,
                    y=[sign * v for v in values],
                    name=name,
                    legendgroup=name,
                    showlegend=name not in shown,
                    marker=_bar_marker(color),
                    customdata=values,
                    hovertemplate=f"{name}: %{{customdata:.3f}} MW<extra></extra>",
                ),
                row=row,
                col=1,
            )
            shown.add(name)


def build_tutorial_mes_dispatch(out_path):
    """Hourly price with the analytic breakevens, and the supply and use of
    power, gas and heat by unit (tutorials/04_mes_economic_dispatch.rst)."""
    ex, hours, _ = _mes_dispatch_run()
    x = list(range(len(hours)))
    k = hours[0]["k"]
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(
            "Electricity price and breakeven prices",
            "Power: supply above zero, use below",
            "Gas: supply above zero, use below",
            "Heat: supply above zero, use below",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=[h["price"] for h in hours],
            mode="lines+markers",
            line_shape="hvh",
            line={"color": C_PRICE, "width": 3},
            marker={"size": 8},
            name="price",
            showlegend=False,
            hovertemplate="hour %{x}: %{y:.1f} per MWh<extra></extra>",
        ),
        row=1,
        col=1,
    )
    for label, value, color, position in (
        ("CHP runs above", ex.CHP_ABOVE, C_CHP, "top left"),
        ("e-boiler runs below", ex.E_BOILER_BELOW, C_E_BOILER, "bottom left"),
        (
            "electrolyser runs below",
            ex.ELECTROLYSER_BELOW,
            C_ELECTROLYSER,
            "bottom left",
        ),
    ):
        fig.add_hline(
            y=value,
            line={"color": color, "width": 2, "dash": "dash"},
            annotation_text=f"{label} {value:.2f}",
            annotation={"xshift": 12},
            annotation_position=position,
            annotation_font={"color": color, "size": 12},
            row=1,
            col=1,
        )
    shown = set()
    _balance_bars(
        fig,
        x,
        supply=[
            ("grid", [h["import_mw"] for h in hours], C_IMPORT),
            ("PV", [h["pv_mw"] for h in hours], C_PV),
            ("CHP", [h["chp_el_mw"] for h in hours], C_CHP),
        ],
        use=[
            ("town demand", [ex.LOAD_MW for _ in hours], C_DEMAND),
            ("e-boiler", [h["e_boiler_el_mw"] for h in hours], C_E_BOILER),
            ("electrolyser", [h["electrolyser_el_mw"] for h in hours], C_ELECTROLYSER),
        ],
        row=2,
        shown=shown,
    )
    _balance_bars(
        fig,
        x,
        supply=[
            ("grid", [k * h["gas_import_kgs"] for h in hours], C_IMPORT),
            ("biogas", [k * h["biogas_kgs"] for h in hours], C_BIOGAS),
            (
                "electrolyser",
                [k * h["electrolyser_gas_kgs"] for h in hours],
                C_ELECTROLYSER,
            ),
        ],
        use=[
            ("town demand", [k * ex.GAS_DEMAND_KGS for _ in hours], C_DEMAND),
            ("CHP", [k * h["chp_fuel_kgs"] for h in hours], C_CHP),
        ],
        row=3,
        shown=shown,
    )
    _balance_bars(
        fig,
        x,
        supply=[
            ("boiler plant", [h["plant_heat_mw"] for h in hours], C_PLANT),
            ("CHP", [h["chp_heat_mw"] for h in hours], C_CHP),
            ("e-boiler", [h["e_boiler_heat_mw"] for h in hours], C_E_BOILER),
        ],
        use=[("town demand", [h["hx_duty_mw"] for h in hours], C_DEMAND)],
        row=4,
        shown=shown,
    )
    _base_layout(fig, "A coupled day: price and dispatch", height=1060)
    fig.update_layout(
        barmode="relative",
        showlegend=True,
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.06,
            "xanchor": "center",
            "x": 0.5,
            "font": {"size": 13, "color": TEXT},
            "bgcolor": C_BLACK,
        },
        margin={"l": 60, "r": 25, "t": 90, "b": 110},
    )
    fig.update_xaxes(title_text="hour", dtick=1, row=4, col=1)
    fig.update_yaxes(title_text="per MWh", rangemode="tozero", row=1, col=1)
    for row in (2, 3, 4):
        fig.update_yaxes(title_text="MW", row=row, col=1)
    return _write(fig, out_path, height_px=1060)


def build_tutorial_mes_checks(out_path):
    """Validation of the coupled day: each unit's operating point against its
    hand computed margin, the solver's saving against the saving the margins
    predict, and the supply temperature against its setpoint
    (tutorials/04_mes_economic_dispatch.rst)."""
    ex, hours, baselines = _mes_dispatch_run()
    optimum = sum(h["objective"] for h in hours)
    x = list(range(len(hours)))
    fig = make_subplots(
        rows=2,
        cols=2,
        column_widths=[0.6, 0.4],
        row_heights=[0.55, 0.45],
        horizontal_spacing=0.12,
        vertical_spacing=0.2,
        specs=[[{}, {}], [{"colspan": 2}, None]],
        subplot_titles=(
            "Operating point against margin",
            "Saving against the baselines",
            "Supply and return temperature",
        ),
    )
    for key, label, color in (
        ("chp", "CHP", C_CHP),
        ("e_boiler", "e-boiler", C_E_BOILER),
        ("electrolyser", "electrolyser", C_ELECTROLYSER),
    ):
        margins = [ex.unit_margins(h["price"])[key] for h in hours]
        fig.add_trace(
            go.Scatter(
                x=margins,
                y=[h[key] for h in hours],
                mode="markers",
                marker={
                    "size": 13,
                    "color": color,
                    "line": {"color": BAR_LINE, "width": 1},
                },
                name=label,
                customdata=[h["price"] for h in hours],
                hovertemplate=(
                    f"{label}<br>price %{{customdata:.1f}}: margin %{{x:.2f}} per hour, "
                    "operating point %{y:.3f}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
    fig.add_vline(
        x=0, line={"color": C_REFERENCE, "width": 1.5, "dash": "dash"}, row=1, col=1
    )
    expected = ex.expected_savings()
    names = list(baselines)
    fig.add_trace(
        go.Bar(
            x=names,
            y=[baselines[n] - optimum for n in names],
            name="solver saving",
            marker=_bar_marker(CB_BLUE),
            text=[f"{baselines[n] - optimum:.1f}" for n in names],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="solver saving: %{y:.2f}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            x=names,
            y=[expected[n] for n in names],
            name="sum of unit margins",
            marker=_bar_marker(CB_SKY, pattern="/"),
            text=[f"{expected[n]:.1f}" for n in names],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="from the margins: %{y:.2f}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    for name, values, color, dash in (
        (
            "hottest supply junction",
            [max(h["supply_t_k"]) for h in hours],
            C_HEAT,
            None,
        ),
        (
            "coolest supply junction",
            [min(h["supply_t_k"]) for h in hours],
            C_HEAT,
            "dot",
        ),
        ("return to the heat plant", [h["return_t_k"] for h in hours], CB_BLUE, None),
    ):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=values,
                mode="lines+markers",
                line={"color": color, "width": 2.5, "dash": dash},
                marker={"size": 7},
                name=name,
                hovertemplate=f"{name}, hour %{{x}}: %{{y:.2f}} K<extra></extra>",
            ),
            row=2,
            col=1,
        )
    # After the traces: plotly drops shapes on a subplot that is still empty.
    lo_k, hi_k = ex.SUPPLY_T_BAND_K
    fig.add_hrect(
        y0=lo_k,
        y1=hi_k,
        fillcolor=C_ACCENT,
        opacity=0.18,
        line_width=0,
        annotation={
            "text": f"supply setpoint band {lo_k:.0f} to {hi_k:.0f} K",
            "yshift": 14,
            "xshift": 12,
            "font": {"color": C_ACCENT, "size": 12},
        },
        annotation_position="top left",
        row=2,
        col=1,
    )
    _base_layout(fig, "Checking the dispatch", height=860)
    fig.update_layout(
        barmode="group",
        showlegend=True,
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.1,
            "xanchor": "center",
            "x": 0.5,
            "font": {"size": 13, "color": TEXT},
            "bgcolor": C_BLACK,
        },
        margin={"l": 60, "r": 25, "t": 90, "b": 130},
    )
    fig.update_xaxes(title_text="margin at full output, per hour", row=1, col=1)
    fig.update_yaxes(title_text="operating point", range=[-0.1, 1.15], row=1, col=1)
    fig.update_yaxes(title_text="saving over the day", row=1, col=2)
    fig.update_xaxes(title_text="hour", dtick=1, row=2, col=1)
    fig.update_yaxes(title_text="K", range=[320, 366], row=2, col=1)
    return _write(fig, out_path, height_px=860)


@functools.lru_cache(maxsize=1)
def _mes_storage_run():
    """Solve examples/mes_storage_and_ramps.py once for both of its figures."""
    path = os.path.join(_REPO_ROOT, "examples", "mes_storage_and_ramps.py")
    spec = importlib.util.spec_from_file_location("mes_storage_and_ramps", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        days, plates, checks = module.run()
    if checks.failed:
        raise RuntimeError(f"mes_storage_and_ramps checks failed: {checks.failed}")
    return module, days, plates


def _legend_key(fig, name, row, col, marker=None, line=None):
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers" if marker else "lines",
            marker=marker,
            line=line,
            name=name,
            hoverinfo="skip",
        ),
        row=row,
        col=col,
    )


def build_tutorial_mes_storage_dispatch(out_path):
    """Price, the operating point of each coupling unit jointly, hour by hour
    and without ramp limits, and the battery's charge, discharge and state of
    charge (tutorials/mes_storage_and_ramps.rst)."""
    ex, days, _ = _mes_storage_run()
    free, hourly, _, joint = (days[name].hours for name in ex.DAYS)
    x = list(range(len(joint)))
    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.24, 0.14, 0.14, 0.14, 0.34],
        subplot_titles=(
            "Electricity price and breakeven prices",
            "CHP operating point",
            "e-boiler operating point",
            "electrolyser operating point",
            "Battery: charge above zero, discharge below, energy stored",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=[h["price"] for h in joint],
            mode="lines+markers",
            line_shape="hvh",
            line={"color": C_PRICE, "width": 3},
            marker={"size": 8},
            showlegend=False,
            hovertemplate="hour %{x}: %{y:.1f} per MWh<extra></extra>",
        ),
        row=1,
        col=1,
    )
    for label, value, color, position in (
        ("CHP runs above", ex.ed.CHP_ABOVE, C_CHP, "top left"),
        ("e-boiler runs below", ex.ed.E_BOILER_BELOW, C_E_BOILER, "bottom left"),
        (
            "electrolyser runs below",
            ex.ed.ELECTROLYSER_BELOW,
            C_ELECTROLYSER,
            "bottom left",
        ),
    ):
        fig.add_hline(
            y=value,
            line={"color": color, "width": 2, "dash": "dash"},
            annotation_text=f"{label} {value:.2f}",
            annotation_position=position,
            annotation_font={"color": color, "size": 12},
            row=1,
            col=1,
        )
    for row, (key, label, color) in enumerate(
        (
            ("chp", "CHP", C_CHP),
            ("e_boiler", "e-boiler", C_E_BOILER),
            ("electrolyser", "electrolyser", C_ELECTROLYSER),
        ),
        start=2,
    ):
        for hours, name, line in (
            (free, "no ramp limits", {"color": C_REFERENCE, "width": 2, "dash": "dot"}),
            (hourly, "hour by hour", {"color": color, "width": 2.5, "dash": "dash"}),
            (joint, "joint", {"color": color, "width": 3.5}),
        ):
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=[h[key] for h in hours],
                    mode="lines+markers" if name == "joint" else "lines",
                    line_shape="hvh",
                    line=line,
                    marker={"size": 7},
                    showlegend=False,
                    hovertemplate=f"{label}, {name}, hour %{{x}}: %{{y:.2f}}<extra></extra>",
                ),
                row=row,
                col=1,
            )
        fig.update_yaxes(range=[-0.12, 1.15], tickvals=[0, 0.5, 1], row=row, col=1)
    for name, dash in (
        ("joint: solid, in the unit's colour", None),
        ("hour by hour: dashed, in the unit's colour", "dash"),
        ("no ramp limits: grey dotted", "dot"),
    ):
        _legend_key(
            fig,
            name,
            row=2,
            col=1,
            line={"color": C_REFERENCE, "width": 3, "dash": dash},
        )
    fig.add_trace(
        go.Bar(
            x=x,
            y=[h["charge_mw"] - h["discharge_mw"] for h in joint],
            width=0.6,
            marker=_bar_marker(C_BATTERY),
            opacity=0.6,
            name="battery power",
            hovertemplate="hour %{x}: %{y:+.3f} MW<extra></extra>",
        ),
        row=5,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[t - 0.5 for t in range(len(joint) + 1)],
            y=[ex.BATTERY_START_MWH] + [h["stored_mwh"] for h in joint],
            mode="lines+markers",
            line={"color": C_BATTERY, "width": 3},
            marker={
                "size": 10,
                "symbol": "diamond",
                "line": {"color": BAR_LINE, "width": 1},
            },
            name="energy stored",
            hovertemplate="%{y:.3f} MWh stored<extra></extra>",
        ),
        row=5,
        col=1,
    )
    # After the traces: plotly drops shapes on a subplot that is still empty.
    fig.add_hline(
        y=ex.BATTERY_MWH,
        line={"color": C_REFERENCE, "width": 1.5, "dash": "dot"},
        annotation_text=f"capacity {ex.BATTERY_MWH:g} MWh",
        annotation_position="top left",
        annotation_font={"size": 12},
        row=5,
        col=1,
    )
    _base_layout(fig, "A coupled day with a battery and ramp limits", height=1250)
    fig.update_layout(
        showlegend=True,
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.06,
            "xanchor": "center",
            "x": 0.5,
            "font": {"size": 13, "color": TEXT},
            "bgcolor": C_BLACK,
        },
        margin={"l": 60, "r": 25, "t": 90, "b": 100},
    )
    fig.update_xaxes(range=[-0.6, len(joint) - 0.4])
    fig.update_xaxes(title_text="hour", dtick=1, row=5, col=1)
    fig.update_yaxes(title_text="per MWh", rangemode="tozero", row=1, col=1)
    fig.update_yaxes(title_text="MW, MWh", range=[-1.2, 2.45], row=5, col=1)
    return _write(fig, out_path, height_px=1250)


def build_tutorial_mes_storage_checks(out_path):
    """Validation of the joint day: ramp steps and state of charge against the
    copper plate, the battery's hourly saving against its cash flow, the
    solver's effects against the copper plate's, and the supply temperature
    (tutorials/mes_storage_and_ramps.rst)."""
    ex, days, plates = _mes_storage_run()
    _, _, joint, battery = (days[name].hours for name in ex.DAYS)
    plate = plates["battery"]
    x = list(range(len(battery)))
    ring = {"size": 17, "color": C_BLACK}
    fig = make_subplots(
        rows=3,
        cols=2,
        horizontal_spacing=0.12,
        vertical_spacing=0.14,
        row_heights=[0.37, 0.37, 0.26],
        specs=[[{}, {}], [{}, {}], [{"colspan": 2}, None]],
        subplot_titles=(
            f"Change of operating point, limit {ex.RAMP_LIMIT:g} per hour",
            "Energy stored at each hour boundary",
            "Battery's hourly saving and cash flow",
            "Solver against copper plate",
            "Supply and return temperature",
        ),
    )
    for offset, (key, label, color) in zip(
        (-0.22, 0.0, 0.22),
        (
            ("chp", "CHP", C_CHP),
            ("e_boiler", "e-boiler", C_E_BOILER),
            ("electrolyser", "electrolyser", C_ELECTROLYSER),
        ),
    ):
        steps = x[1:]
        fig.add_trace(
            go.Scatter(
                x=[t + offset for t in steps],
                y=[plate[key][t] - plate[key][t - 1] for t in steps],
                mode="markers",
                marker={**ring, "line": {"color": color, "width": 2}},
                showlegend=False,
                customdata=steps,
                hovertemplate=f"{label}, copper plate, hour %{{customdata}}: %{{y:+.2f}}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[t + offset for t in steps],
                y=[battery[t][key] - battery[t - 1][key] for t in steps],
                mode="markers",
                marker={
                    "size": 9,
                    "color": color,
                    "line": {"color": BAR_LINE, "width": 1},
                },
                name=label,
                customdata=steps,
                hovertemplate=f"{label}, hour %{{customdata}}: %{{y:+.3f}}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    _legend_key(
        fig,
        "copper plate: ring in the unit's colour",
        row=1,
        col=1,
        marker={**ring, "size": 15, "line": {"color": C_REFERENCE, "width": 2}},
    )
    ends = [t - 0.5 for t in range(len(battery) + 1)]
    fig.add_trace(
        go.Scatter(
            x=ends,
            y=[ex.BATTERY_START_MWH] + plate["stored_mwh"],
            mode="markers",
            marker={**ring, "line": {"color": C_BATTERY, "width": 2}},
            showlegend=False,
            hovertemplate="copper plate: %{y:.3f} MWh<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=ends,
            y=[ex.BATTERY_START_MWH] + [h["stored_mwh"] for h in battery],
            mode="lines+markers",
            line={"color": C_BATTERY, "width": 2.5},
            marker={
                "size": 9,
                "color": C_BATTERY,
                "line": {"color": BAR_LINE, "width": 1},
            },
            name="energy stored",
            hovertemplate="solver: %{y:.3f} MWh<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            x=x,
            y=[
                ex.ed.hourly_cost(c) - ex.ed.hourly_cost(d)
                for c, d in zip(joint, battery)
            ],
            marker=_bar_marker(CB_BLUE),
            name="hourly cost saved",
            hovertemplate="hour %{x}: %{y:+.3f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=[h["price"] * (h["discharge_mw"] - h["charge_mw"]) for h in battery],
            mode="markers",
            marker={
                "size": 10,
                "symbol": "diamond",
                "color": CB_SKY,
                "line": {"color": BAR_LINE, "width": 1},
            },
            name="battery cash flow",
            hovertemplate="hour %{x}: %{y:+.2f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    effects = ex.effects(days, plates)
    names = [name.replace(" of ", " of<br>", 1) for name in effects]
    for i, (label, pattern, color) in enumerate(
        (("solver effect", None, CB_BLUE), ("copper plate effect", "/", CB_SKY))
    ):
        values = [pair[i] for pair in effects.values()]
        fig.add_trace(
            go.Bar(
                x=names,
                y=values,
                name=label,
                marker=_bar_marker(color, pattern=pattern),
                text=[f"{v:.2f}" for v in values],
                textposition="outside",
                cliponaxis=False,
                hovertemplate=f"{label}: %{{y:.3f}}<extra></extra>",
            ),
            row=2,
            col=2,
        )
    for name, values, color, dash in (
        (
            "hottest supply junction",
            [max(h["supply_t_k"]) for h in battery],
            C_HEAT,
            None,
        ),
        (
            "coolest supply junction",
            [min(h["supply_t_k"]) for h in battery],
            C_HEAT,
            "dot",
        ),
        ("return to the heat plant", [h["return_t_k"] for h in battery], CB_BLUE, None),
    ):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=values,
                mode="lines+markers",
                line={"color": color, "width": 2.5, "dash": dash},
                marker={"size": 7},
                name=name,
                hovertemplate=f"{name}, hour %{{x}}: %{{y:.2f}} K<extra></extra>",
            ),
            row=3,
            col=1,
        )
    # After the traces: plotly drops shapes on a subplot that is still empty.
    fig.add_hrect(
        y0=-ex.RAMP_LIMIT,
        y1=ex.RAMP_LIMIT,
        fillcolor=C_ACCENT,
        opacity=0.14,
        line_width=0,
        layer="below",
        row=1,
        col=1,
    )
    fig.add_hrect(
        y0=0,
        y1=ex.BATTERY_MWH,
        fillcolor=C_ACCENT,
        opacity=0.12,
        line_width=0,
        layer="below",
        row=1,
        col=2,
    )
    lo_k, hi_k = ex.ed.SUPPLY_T_BAND_K
    fig.add_hrect(
        y0=lo_k,
        y1=hi_k,
        fillcolor=C_ACCENT,
        opacity=0.18,
        line_width=0,
        layer="below",
        annotation={
            "text": f"supply setpoint band {lo_k:.0f} to {hi_k:.0f} K",
            "yshift": 14,
            "font": {"color": C_ACCENT, "size": 12},
        },
        annotation_position="top left",
        row=3,
        col=1,
    )
    _base_layout(fig, "Checking the joint dispatch", height=1180)
    fig.update_layout(
        barmode="group",
        showlegend=True,
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.07,
            "xanchor": "center",
            "x": 0.5,
            "font": {"size": 13, "color": TEXT},
            "bgcolor": C_BLACK,
        },
        margin={"l": 60, "r": 25, "t": 90, "b": 130},
    )
    fig.update_xaxes(title_text="hour", dtick=1, row=1, col=1)
    fig.update_yaxes(range=[-0.75, 0.75], tickvals=[-0.5, 0, 0.5], row=1, col=1)
    fig.update_xaxes(
        title_text="hour", dtick=1, range=[-0.9, len(battery) - 0.1], row=1, col=2
    )
    fig.update_yaxes(title_text="MWh", range=[-0.2, 2.3], row=1, col=2)
    fig.update_xaxes(title_text="hour", dtick=1, row=2, col=1)
    fig.update_yaxes(title_text="per hour", row=2, col=1)
    fig.update_yaxes(title_text="over the day", range=[0, 150], row=2, col=2)
    fig.update_xaxes(title_text="hour", dtick=1, row=3, col=1)
    fig.update_yaxes(title_text="K", range=[320, 366], row=3, col=1)
    return _write(fig, out_path, height_px=1180)


@functools.lru_cache(maxsize=1)
def _mes_limits_run():
    """Solve examples/mes_network_limits.py once for both of its figures."""
    path = os.path.join(_REPO_ROOT, "examples", "mes_network_limits.py")
    spec = importlib.util.spec_from_file_location("mes_network_limits", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        days, checks = module.run()
    if checks.failed:
        raise RuntimeError(f"mes_network_limits checks failed: {checks.failed}")
    return module, days


def build_tutorial_mes_limits_dispatch(out_path):
    """Price, the load on the cable and the gas pipe to the plant site against
    their limits, and each unit's operating point with both limits and without
    (tutorials/mes_network_limits.rst)."""
    ex, days = _mes_limits_run()
    both, free = days["both limits"].hours, days["neither"].hours
    k = both[0]["k"]
    x = list(range(len(both)))
    fig = make_subplots(
        rows=6,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.045,
        row_heights=[0.2, 0.19, 0.19, 0.14, 0.14, 0.14],
        subplot_titles=(
            "Electricity price and breakeven prices",
            "Cable to the plant bus: apparent power",
            "Gas pipe to the plant junction: gas arriving",
            "CHP operating point",
            "e-boiler operating point",
            "electrolyser operating point",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=[h["price"] for h in both],
            mode="lines+markers",
            line_shape="hvh",
            line={"color": C_PRICE, "width": 3},
            marker={"size": 8},
            showlegend=False,
            hovertemplate="hour %{x}: %{y:.1f} per MWh<extra></extra>",
        ),
        row=1,
        col=1,
    )
    for label, value, color, position in (
        ("CHP runs above", ex.ed.CHP_ABOVE, C_CHP, "top left"),
        ("e-boiler runs below", ex.ed.E_BOILER_BELOW, C_E_BOILER, "bottom left"),
        (
            "electrolyser runs below",
            ex.ed.ELECTROLYSER_BELOW,
            C_ELECTROLYSER,
            "bottom left",
        ),
    ):
        fig.add_hline(
            y=value,
            line={"color": color, "width": 2, "dash": "dash"},
            annotation_text=f"{label} {value:.2f}",
            annotation_position=position,
            annotation_font={"color": color, "size": 12},
            row=1,
            col=1,
        )
    for row, key, scale, unit in (
        (2, "cable_mva", 1.0, "MVA"),
        (3, "pipe_kgs", k, "MW"),
    ):
        for hours, label, marker in (
            (free, "without limits", _bar_marker(C_REFERENCE, pattern="/")),
            (both, "with both limits", _bar_marker(CB_BLUE)),
        ):
            fig.add_trace(
                go.Bar(
                    x=x,
                    y=[scale * h[key] for h in hours],
                    name=label,
                    legendgroup=label,
                    showlegend=row == 2,
                    marker=marker,
                    hovertemplate=f"{label}, hour %{{x}}: %{{y:.3f}} {unit}<extra></extra>",
                ),
                row=row,
                col=1,
            )
    for row, (key, label, color) in enumerate(
        (
            ("chp", "CHP", C_CHP),
            ("e_boiler", "e-boiler", C_E_BOILER),
            ("electrolyser", "electrolyser", C_ELECTROLYSER),
        ),
        start=4,
    ):
        for hours, name, line in (
            (free, "without limits", {"color": C_REFERENCE, "width": 2, "dash": "dot"}),
            (both, "with both limits", {"color": color, "width": 3.5}),
        ):
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=[h[key] for h in hours],
                    mode="lines+markers" if name == "with both limits" else "lines",
                    line_shape="hvh",
                    line=line,
                    marker={"size": 7},
                    showlegend=False,
                    hovertemplate=f"{label}, {name}, hour %{{x}}: %{{y:.3f}}<extra></extra>",
                ),
                row=row,
                col=1,
            )
        fig.update_yaxes(range=[-0.12, 1.15], tickvals=[0, 0.5, 1], row=row, col=1)
    for name, dash in (
        ("operating point with both limits: solid, in the unit's colour", None),
        ("operating point without limits: grey dotted", "dot"),
    ):
        _legend_key(
            fig,
            name,
            row=4,
            col=1,
            line={"color": C_REFERENCE, "width": 3, "dash": dash},
        )
    # After the traces: plotly drops shapes on a subplot that is still empty.
    limited = days["both limits"]
    for row, value, text, position in (
        (2, limited.cable_mva, f"rating {limited.cable_mva:g} MVA", "top left"),
        (
            3,
            limited.pipe_capacity_kgs * k,
            f"capacity at the floor {limited.pipe_capacity_kgs * k:.2f} MW",
            "top",
        ),
    ):
        fig.add_hline(
            y=value,
            line={"color": C_REFERENCE, "width": 2, "dash": "dash"},
            annotation_text=text,
            annotation_position=position,
            annotation_font={"size": 12},
            row=row,
            col=1,
        )
    _base_layout(fig, "Where the network binds", height=1310)
    fig.update_layout(
        barmode="group",
        showlegend=True,
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.05,
            "xanchor": "center",
            "x": 0.5,
            "font": {"size": 13, "color": TEXT},
            "bgcolor": C_BLACK,
        },
        margin={"l": 60, "r": 25, "t": 90, "b": 110},
    )
    fig.update_xaxes(range=[-0.6, len(both) - 0.4])
    fig.update_xaxes(title_text="hour", dtick=1, row=6, col=1)
    fig.update_yaxes(title_text="per MWh", rangemode="tozero", row=1, col=1)
    fig.update_yaxes(title_text="MVA", range=[0, 2.4], row=2, col=1)
    fig.update_yaxes(title_text="MW", range=[-1.2, 3.1], row=3, col=1)
    return _write(fig, out_path, height_px=1310)


def build_tutorial_mes_limits_checks(out_path):
    """Validation of the limited day: the value of one more MW through each
    bottleneck against the marginal unit's margin, the congestion cost against
    the margins lost, the gas pipe on its Weymouth curve, every operating point
    against the hand dispatch, and the supply temperature
    (tutorials/mes_network_limits.rst)."""
    ex, days = _mes_limits_run()
    limited = days["both limits"]
    both, free = limited.hours, days["neither"].hours
    x = list(range(len(both)))
    ring = {"size": 17, "color": C_BLACK}
    fig = make_subplots(
        rows=3,
        cols=2,
        horizontal_spacing=0.12,
        vertical_spacing=0.14,
        row_heights=[0.37, 0.37, 0.26],
        specs=[[{}, {}], [{}, {}], [{"colspan": 2}, None]],
        subplot_titles=(
            "Value of one more MW through the bottleneck",
            "Congestion cost per hour",
            "Gas pipe: pressure at the plant junction against flow",
            "Operating point: solver against hand calculation",
            "Supply and return temperature",
        ),
    )
    values = ex.marginal_values(days)
    for limit, unit, color in (
        ("cable", "electrolyser", C_ELECTROLYSER),
        ("gas pipe", "CHP", C_CHP),
    ):
        hours = [t for t, (name, _, _) in values.items() if name == limit]
        fig.add_trace(
            go.Scatter(
                x=hours,
                y=[values[t][2] for t in hours],
                mode="markers",
                marker={**ring, "line": {"color": color, "width": 2}},
                showlegend=False,
                hovertemplate=f"{unit}'s margin per MW, hour %{{x}}: %{{y:.2f}}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=hours,
                y=[values[t][1] for t in hours],
                mode="markers",
                marker={
                    "size": 10,
                    "color": color,
                    "line": {"color": BAR_LINE, "width": 1},
                },
                name=f"{limit}: {unit} at the margin",
                hovertemplate=f"one more MW through the {limit}, hour %{{x}}: %{{y:.3f}}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    _legend_key(
        fig,
        "margin per MW, by hand: ring in the unit's colour",
        row=1,
        col=1,
        marker={**ring, "size": 15, "line": {"color": C_REFERENCE, "width": 2}},
    )
    congestion = ex.congestion(days, "both limits")
    fig.add_trace(
        go.Bar(
            x=x,
            y=[cost for cost, _ in congestion],
            name="congestion cost",
            marker=_bar_marker(CB_BLUE),
            hovertemplate="hour %{x}: %{y:.4f}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=[
                lost + h["price"] * (h["line_loss_mw"] - f["line_loss_mw"])
                for (_, lost), h, f in zip(congestion, both, free)
            ],
            mode="markers",
            marker={
                "size": 11,
                "symbol": "diamond",
                "color": CB_SKY,
                "line": {"color": BAR_LINE, "width": 1},
            },
            name="margins lost and line losses",
            hovertemplate="hour %{x}: %{y:.4f}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    net, _, site = ex.build_network()
    k_main = ex.weymouth_k(net, site.main_pipe)
    demand = ex.ed.GAS_DEMAND_KGS
    flows = np.linspace(-0.03, 0.068, 120)
    fig.add_trace(
        go.Scatter(
            x=flows,
            y=[
                1
                - k_main * (m + demand) * abs(m + demand)
                - limited.k_pipe * m * abs(m)
                for m in flows
            ],
            mode="lines",
            line={"color": C_GAS, "width": 2.5},
            name="Weymouth, by hand",
            hoverinfo="skip",
        ),
        row=2,
        col=1,
    )
    thin = [
        (label, h)
        for label, day in days.items()
        if day.k_pipe == limited.k_pipe
        for h in day.hours
    ]
    fig.add_trace(
        go.Scatter(
            x=[h["pipe_kgs"] for _, h in thin],
            y=[h["psq"][-1] for _, h in thin],
            mode="markers",
            marker={"size": 9, "color": C_GAS, "line": {"color": BAR_LINE, "width": 1}},
            name="solver, days with the 30 mm pipe",
            customdata=[label for label, _ in thin],
            hovertemplate="%{customdata}: %{x:.5f} kg/s, %{y:.4f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    full = max(h["pipe_kgs"] for h in free)
    fig.add_trace(
        go.Scatter(
            x=[full],
            y=[1 - k_main * (full + demand) ** 2 - limited.k_pipe * full**2],
            mode="markers",
            marker={**ring, "line": {"color": C_REFERENCE, "width": 2}},
            name="full CHP without the floor",
            hovertemplate="%{x:.4f} kg/s would leave %{y:.3f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    # Decreasing sizes keep the units visible where their points coincide.
    for key, label, color, size in (
        ("chp", "CHP", C_CHP, 17),
        ("e_boiler", "e-boiler", C_E_BOILER, 12),
        ("electrolyser", "electrolyser", C_ELECTROLYSER, 7),
    ):
        points = [
            (ex.expected_dispatch(day, h["price"], h["k"])[key], h[key], name)
            for name, day in days.items()
            for h in day.hours
        ]
        fig.add_trace(
            go.Scatter(
                x=[p[0] for p in points],
                y=[p[1] for p in points],
                mode="markers",
                marker={
                    "size": size,
                    "color": color,
                    "line": {"color": BAR_LINE, "width": 1},
                },
                name=label,
                customdata=[p[2] for p in points],
                hovertemplate=f"{label}, %{{customdata}}: by hand %{{x:.4f}}, solver %{{y:.4f}}<extra></extra>",
            ),
            row=2,
            col=2,
        )
    for name, values_k, color, dash in (
        ("hottest supply junction", [max(h["supply_t_k"]) for h in both], C_HEAT, None),
        (
            "coolest supply junction",
            [min(h["supply_t_k"]) for h in both],
            C_HEAT,
            "dot",
        ),
        ("return to the heat plant", [h["return_t_k"] for h in both], CB_BLUE, None),
    ):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=values_k,
                mode="lines+markers",
                line={"color": color, "width": 2.5, "dash": dash},
                marker={"size": 7},
                name=name,
                hovertemplate=f"{name}, hour %{{x}}: %{{y:.2f}} K<extra></extra>",
            ),
            row=3,
            col=1,
        )
    # After the traces: plotly drops shapes on a subplot that is still empty.
    for floor, dash in ((limited.floor, "dash"), (days[ex.FLOOR_RELAXED].floor, "dot")):
        fig.add_hline(
            y=floor,
            line={"color": C_REFERENCE, "width": 1.5, "dash": dash},
            annotation_text=f"floor {floor:g}",
            annotation_position="bottom left" if dash == "dot" else "top left",
            annotation_font={"size": 12},
            row=2,
            col=1,
        )
    fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        line={"color": C_REFERENCE, "width": 1.5, "dash": "dash"},
        layer="below",
        row=2,
        col=2,
    )
    lo_k, hi_k = ex.ed.SUPPLY_T_BAND_K
    fig.add_hrect(
        y0=lo_k,
        y1=hi_k,
        fillcolor=C_ACCENT,
        opacity=0.18,
        line_width=0,
        layer="below",
        annotation={
            "text": f"supply setpoint band {lo_k:.0f} to {hi_k:.0f} K",
            "yshift": 14,
            "font": {"color": C_ACCENT, "size": 12},
        },
        annotation_position="top left",
        row=3,
        col=1,
    )
    _base_layout(fig, "Checking the constrained dispatch", height=1180)
    fig.update_layout(
        showlegend=True,
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.07,
            "xanchor": "center",
            "x": 0.5,
            "font": {"size": 13, "color": TEXT},
            "bgcolor": C_BLACK,
        },
        margin={"l": 60, "r": 25, "t": 90, "b": 150},
    )
    fig.update_xaxes(title_text="hour", dtick=1, range=[-0.5, 7.5], row=1, col=1)
    fig.update_yaxes(title_text="per MW", range=[0, 30], row=1, col=1)
    fig.update_xaxes(title_text="hour", dtick=1, row=1, col=2)
    fig.update_yaxes(title_text="per hour", row=1, col=2)
    fig.update_xaxes(title_text="flow to the plant junction, kg/s", row=2, col=1)
    fig.update_yaxes(title_text="squared pressure, pu", range=[0.4, 1.12], row=2, col=1)
    fig.update_xaxes(title_text="by hand", range=[-0.08, 1.08], row=2, col=2)
    fig.update_yaxes(title_text="solver", range=[-0.08, 1.08], row=2, col=2)
    fig.update_xaxes(title_text="hour", dtick=1, row=3, col=1)
    fig.update_yaxes(title_text="K", range=[320, 366], row=3, col=1)
    return _write(fig, out_path, height_px=1180)


C_MISSED = "#d7191c"


@functools.lru_cache(maxsize=1)
def _mes_local_optima_solve():
    """Solve examples/mes_local_optima.py once for both of its figures, with
    the failures of the checks that do not depend on IPOPT. Those of IPOPT's
    results only warn: the sweep figure shows what IPOPT does on the build
    machine, whatever that is."""
    path = os.path.join(_REPO_ROOT, "examples", "mes_local_optima.py")
    spec = importlib.util.spec_from_file_location("mes_local_optima", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results, _ = module.run()
    solid, ipopt = module.ed.Checks(), module.ed.Checks()
    module.check_scip_and_apopt(solid, results)
    module.check_ipopt_results(ipopt, results)
    for label in ipopt.failed:
        print(
            f"[interactive_plots] WARNING: mes_local_optima, IPOPT on this build: {label}"
        )
    return module, results, tuple(solid.failed)


def _mes_local_optima_run():
    module, results, failed = _mes_local_optima_solve()
    if failed:
        raise RuntimeError(f"mes_local_optima checks failed: {list(failed)}")
    return module, results


def build_tutorial_mes_local_optima_day(out_path):
    """The network limits day by APOPT and by SCIP: the cost of each hour with
    the relaxation's bound, the CHP's operating point, the CHP exchanger's
    balance with APOPT's stops, and hour 0's cost along the CHP's operating
    point, and the loading of the cable and the gas pipe to the plant site
    (tutorials/mes_local_optima.rst)."""
    ex, results = _mes_local_optima_run()
    apopt, scip, bound = (results.day[k] for k in ("APOPT", "SCIP", "relaxation"))
    x = list(scip)
    fig = make_subplots(
        rows=4,
        cols=2,
        horizontal_spacing=0.12,
        vertical_spacing=0.07,
        row_heights=[0.29, 0.16, 0.2, 0.35],
        specs=[[{"colspan": 2}, None], [{"colspan": 2}, None], [{}, {}], [{}, {}]],
        subplot_titles=(
            "Cost of each hour",
            "CHP operating point",
            f"Cable loading, % of its {ex.DAY_LIMITS['cable_mva']:g} MVA rating",
            "Gas pipe loading, % of its capacity at the floor",
            "CHP exchanger: where its balance holds",
            "Hour 0 with the CHP pinned, by SCIP",
        ),
    )
    width = 0.38
    excess = {t: apopt[t].cost - scip[t].objective for t in ex.CHP_HOURS}
    for values, label, marker, offset, text in (
        (
            [apopt[t].cost for t in x],
            "APOPT",
            _bar_marker(C_REFERENCE, pattern="/"),
            -width,
            [f"+{excess[t]:.2f}" if t in excess else "" for t in x],
        ),
        (
            [scip[t].objective for t in x],
            "SCIP on the exact formulation",
            _bar_marker(CB_BLUE),
            0.0,
            None,
        ),
    ):
        fig.add_trace(
            go.Bar(
                x=x,
                y=values,
                width=width,
                offset=offset,
                name=f"{label}, day {sum(values):.2f}",
                legendgroup=label,
                marker=marker,
                text=text,
                textposition="outside",
                constraintext="none",
                cliponaxis=False,
                hovertemplate=f"{label}, hour %{{x}}: %{{y:.2f}}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=[t + width / 2 for t in x],
            y=[bound[t].objective for t in x],
            mode="markers",
            marker={
                "size": 11,
                "symbol": "diamond",
                "color": CB_SKY,
                "line": {"color": BAR_LINE, "width": 1},
            },
            name=f"relaxation's bound, day {sum(h.objective for h in bound.values()):.2f}",
            customdata=x,
            hovertemplate="bound, hour %{customdata}: %{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    for hours, name, line, mode in (
        (
            apopt,
            "CHP by APOPT",
            {"color": C_REFERENCE, "width": 2.5, "dash": "dot"},
            "lines",
        ),
        (scip, "CHP by SCIP", {"color": C_CHP, "width": 3.5}, "lines+markers"),
    ):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=[hours[t].dispatch["chp"] for t in x],
                mode=mode,
                line_shape="hvh",
                line=line,
                marker={"size": 7},
                name=name,
                hovertemplate=f"{name}, hour %{{x}}: %{{y:.3f}}<extra></extra>",
            ),
            row=2,
            col=1,
        )
    net, _, site = ex.nl.build_network(**ex.DAY_LIMITS)
    capacity_kgs = ex.nl.pipe_capacity_kgs(net, site, ex.DAY_LIMITS["floor"])
    loadings = (
        (1, lambda h: 100 * h.quantities["cable_mva"] / ex.DAY_LIMITS["cable_mva"]),
        (2, lambda h: 100 * abs(h.quantities["pipe_kgs"]) / capacity_kgs),
    )
    for col, loading in loadings:
        for hours, label, marker, offset in (
            (apopt, "APOPT", _bar_marker(C_REFERENCE, pattern="/"), -width),
            (scip, "SCIP on the exact formulation", _bar_marker(CB_BLUE), 0.0),
        ):
            fig.add_trace(
                go.Bar(
                    x=x,
                    y=[loading(hours[t]) for t in x],
                    width=width,
                    offset=offset,
                    marker=marker,
                    legendgroup=label,
                    showlegend=False,
                    customdata=[
                        (
                            hours[t].quantities["cable_mva"],
                            hours[t].quantities["pipe_kgs"],
                            "to the plant"
                            if hours[t].quantities["pipe_kgs"] >= 0
                            else "to the town",
                        )
                        for t in x
                    ],
                    hovertemplate=(
                        f"{label}, hour %{{x}}: %{{y:.0f}} %<br>cable "
                        "%{customdata[0]:.3f} MVA, pipe %{customdata[1]:.4f} kg/s "
                        "%{customdata[2]}<extra></extra>"
                    ),
                ),
                row=3,
                col=col,
            )
    on_k = scip[ex.CHP_HOURS[0]].exchangers["chp"].spread_k
    fig.add_trace(
        go.Scatter(
            x=[0, 0, None, 0, 2],
            y=[0, 100, None, on_k, on_k],
            mode="lines",
            line={"color": C_CHP, "width": 3},
            name="where the CHP exchanger's balance holds",
            hoverinfo="skip",
        ),
        row=4,
        col=1,
    )
    for hours, name, color in (
        (apopt, "APOPT's stops", C_REFERENCE),
        (scip, "SCIP's optimum", CB_BLUE),
    ):
        states = [hours[t].exchangers["chp"] for t in ex.CHP_HOURS]
        fig.add_trace(
            go.Scatter(
                x=[s.heat_mw for s in states],
                y=[s.spread_k for s in states],
                mode="markers",
                marker={
                    "size": 12,
                    "color": color,
                    "line": {"color": BAR_LINE, "width": 1},
                },
                name=name,
                customdata=[
                    (t, s.flow_kgs, s.node_k) for t, s in zip(ex.CHP_HOURS, states)
                ],
                hovertemplate=(
                    f"{name}, hour %{{customdata[0]}}: %{{x:.3f}} MW, spread "
                    "%{y:.2f} K, flow %{customdata[1]:.3g} kg/s, node "
                    "%{customdata[2]:.2f} K<extra></extra>"
                ),
            ),
            row=4,
            col=1,
        )
    points = ex.chp_cost_curve(results)
    fig.add_trace(
        go.Scatter(
            x=[p[0] for p in points],
            y=[p[1] for p in points],
            mode="lines+markers",
            line={"color": CB_BLUE, "width": 2.5},
            marker={
                "size": 9,
                "color": CB_BLUE,
                "line": {"color": BAR_LINE, "width": 1},
            },
            name="hour 0 by SCIP with the CHP pinned",
            hovertemplate="CHP pinned at %{x:.3f}: %{y:.2f}<extra></extra>",
        ),
        row=4,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=[apopt[0].dispatch["chp"]],
            y=[apopt[0].cost],
            mode="markers",
            marker={
                "size": 19,
                "color": C_BLACK,
                "line": {"color": C_REFERENCE, "width": 2.5},
            },
            showlegend=False,
            hovertemplate="APOPT, hour 0: CHP %{x:.3f}, %{y:.2f}<extra></extra>",
        ),
        row=4,
        col=2,
    )
    # After the traces: plotly drops shapes on a subplot that is still empty.
    cap = scip[ex.CHP_HOURS[0]].expected["chp"]
    fig.add_hline(
        y=cap,
        line={"color": C_REFERENCE, "width": 2, "dash": "dash"},
        layer="below",
        annotation_text=f"the gas pipe allows {cap:.3f}",
        annotation_position="top",
        annotation_font={"size": 12},
        row=2,
        col=1,
    )
    for col in (1, 2):
        fig.add_hline(
            y=100,
            line={"color": C_REFERENCE, "width": 2, "dash": "dash"},
            layer="below",
            row=3,
            col=col,
        )
    stop_k = max(apopt[t].exchangers["chp"].spread_k for t in ex.CHP_HOURS)
    margin = ex.ed.unit_margins(scip[0].price)["chp"]
    for col, annotation in (
        (
            1,
            {
                "x": 0,
                "y": stop_k,
                "text": f"APOPT stops here: feeding heat in<br>needs a {on_k:.0f} K spread first",
                "ax": 40,
                "ay": -70,
                "xanchor": "left",
            },
        ),
        (
            1,
            {
                "x": 0,
                "y": 0.97,
                "yref": "y domain",
                "text": "unit off: any spread",
                "showarrow": False,
                "xanchor": "left",
                "xshift": 8,
            },
        ),
        (
            1,
            {
                "x": 0.4,
                "y": on_k,
                "text": f"unit on: {on_k:.0f} K spread",
                "showarrow": False,
                "xanchor": "left",
                "yshift": -14,
            },
        ),
        (2, {"x": points[0][0], "y": points[0][1], "text": "APOPT", "ax": 45, "ay": 0}),
        (
            2,
            {
                "x": points[-1][0],
                "y": points[-1][1],
                "text": "optimum",
                "ax": -55,
                "ay": 20,
            },
        ),
        (
            2,
            {
                "x": 0.65,
                "y": 0.85,
                "xref": "x domain",
                "yref": "y domain",
                "text": f"falls about {margin:.0f} per unit,<br>the CHP's margin at {scip[0].price:g}",
                "showarrow": False,
            },
        ),
    ):
        fig.add_annotation(
            **{"arrowcolor": C_REFERENCE, "font": {"size": 12}, **annotation},
            row=4,
            col=col,
        )
    _base_layout(fig, "Where APOPT stops", height=1550)
    fig.update_layout(
        showlegend=True,
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.06,
            "xanchor": "center",
            "x": 0.5,
            "font": {"size": 13, "color": TEXT},
            "bgcolor": C_BLACK,
        },
        margin={"l": 60, "r": 25, "t": 90, "b": 120},
    )
    fig.update_xaxes(dtick=1, range=[-0.6, len(x) - 0.4], row=1, col=1)
    fig.update_yaxes(
        title_text="cost of the hour",
        range=[0, 1.15 * max(h.cost for h in apopt.values())],
        row=1,
        col=1,
    )
    fig.update_xaxes(
        title_text="hour", dtick=1, range=[-0.6, len(x) - 0.4], row=2, col=1
    )
    fig.update_yaxes(range=[-0.12, 1.15], tickvals=[0, 0.5, 1], row=2, col=1)
    for col in (1, 2):
        fig.update_xaxes(
            title_text="hour",
            dtick=1,
            range=[-0.6, len(x) - 0.4],
            showgrid=False,
            row=3,
            col=col,
        )
        fig.update_yaxes(
            range=[0, 118],
            tickvals=[0, 50, 100],
            ticktext=["0 %", "50 %", "100 %"],
            row=3,
            col=col,
        )
    fig.update_xaxes(title_text="heat fed in, MW", range=[-0.1, 1.35], row=4, col=1)
    fig.update_yaxes(title_text="spread, K", range=[25, 40], row=4, col=1)
    fig.update_xaxes(
        title_text="CHP operating point", range=[-0.08, 0.86], row=4, col=2
    )
    costs = [p[1] for p in points]
    fig.update_yaxes(
        title_text="cost of hour 0",
        range=[min(costs) - 12, max(costs) + 12],
        row=4,
        col=2,
    )
    return _write(fig, out_path, height_px=1550)


def _ipopt_outcome(ex, hour, optimum):
    if hour.failed:
        return "failed"
    if hour.deviation <= 1e-3 and abs(hour.objective - optimum.objective) <= 1e-3:
        return (
            "at the optimum after the retry"
            if hour.retried
            else "at the optimum, first attempt"
        )
    if hour.status == "Solve_Succeeded":
        return "off the optimum, Solve_Succeeded"
    return "off the optimum, another status"


def build_tutorial_mes_local_optima_sweep(out_path):
    """Hours 4 and 5 with the 34 mm pipe across the cable ratings: IPOPT's
    result from monee's default start and option presets against SCIP's
    optimum and the optimum with the e-boiler pinned off, how IPOPT ended,
    and the time per hourly solve. How IPOPT ends depends on the CasADi build,
    so the figure shows what the build machine computes
    (tutorials/mes_local_optima.rst)."""
    ex, results = _mes_local_optima_run()
    sweep = results.sweep
    outcomes = {
        "at the optimum, first attempt": _bar_marker(CB_SKY),
        "at the optimum after the retry": _bar_marker(CB_SKY, pattern="/"),
        "off the optimum, Solve_Succeeded": _bar_marker(C_MISSED),
        "off the optimum, another status": _bar_marker(C_MISSED, pattern="x"),
        "failed": _bar_marker(C_REFERENCE, pattern="."),
    }
    fig = make_subplots(
        rows=3,
        cols=2,
        horizontal_spacing=0.14,
        vertical_spacing=0.1,
        row_heights=[0.3, 0.3, 0.4],
        specs=[[{"colspan": 2}, None], [{"colspan": 2}, None], [{}, {}]],
        subplot_titles=tuple(
            f"Hour {t}, price {ex.ed.EL_PRICE[t]:g}: cost of the hour"
            for t in ex.SWEEP_HOURS
        )
        + (
            f"IPOPT's outcome, CasADi {casadi.__version__} on {platform.system()}",
            "Seconds per hourly solve, same build",
        ),
    )
    offset, width = 0.021, 0.038
    counts = {t: {name: [] for name in outcomes} for t in ex.SWEEP_HOURS}
    for row, t in enumerate(ex.SWEEP_HOURS, start=1):
        keys = [key for key in sweep["SCIP"] if key[1] == t]
        optima = [sweep["SCIP"][key] for key in keys]
        fig.add_trace(
            go.Bar(
                x=[key[0] - offset for key in keys],
                y=[h.objective for h in optima],
                width=width,
                marker=_bar_marker(CB_BLUE),
                name="SCIP, the global optimum",
                legendgroup="SCIP",
                showlegend=row == 1,
                customdata=[
                    (key[0], h.dispatch["e_boiler"], h.dispatch["electrolyser"])
                    for key, h in zip(keys, optima)
                ],
                hovertemplate=(
                    "SCIP, %{customdata[0]:.2f} MVA: %{y:.2f}<br>e-boiler "
                    "%{customdata[1]:.3f}, electrolyser %{customdata[2]:.3f}"
                    "<extra></extra>"
                ),
            ),
            row=row,
            col=1,
        )
        for key, optimum in zip(keys, optima):
            hour = sweep["IPOPT"][key]
            outcome = _ipopt_outcome(ex, hour, optimum)
            counts[t][outcome].append(f"{key[0]:.2f}")
            x = key[0] + offset
            if hour.failed:
                fig.add_annotation(
                    x=x,
                    y=optimum.objective / 2,
                    text="IPOPT failed",
                    textangle=-90,
                    showarrow=False,
                    font={"size": 11, "color": TEXT},
                    row=row,
                    col=1,
                )
                continue
            excess = hour.cost - optimum.objective
            fig.add_trace(
                go.Bar(
                    x=[x],
                    y=[hour.cost],
                    width=width,
                    marker=outcomes[outcome],
                    legendgroup=outcome,
                    showlegend=False,
                    text=[f"+{excess:.2f}" if outcome.startswith("off") else ""],
                    textposition="outside",
                    textfont={"size": 12, "color": TEXT},
                    cliponaxis=False,
                    customdata=[
                        (
                            key[0],
                            hour.status,
                            "after the retry"
                            if hour.retried
                            else "at the first attempt",
                            hour.dispatch["e_boiler"],
                            hour.dispatch["electrolyser"],
                            excess,
                        )
                    ],
                    hovertemplate=(
                        "IPOPT, %{customdata[0]:.2f} MVA: %{customdata[1]} "
                        "%{customdata[2]}<br>cost %{y:.2f}, %{customdata[5]:+.2f} "
                        "against SCIP<br>e-boiler %{customdata[3]:.3f}, "
                        "electrolyser %{customdata[4]:.3f}<extra></extra>"
                    ),
                ),
                row=row,
                col=1,
            )
    hours_x = [f"hour {t}" for t in ex.SWEEP_HOURS]
    for name, marker in outcomes.items():
        fig.add_trace(
            go.Bar(
                x=hours_x,
                y=[len(counts[t][name]) for t in ex.SWEEP_HOURS],
                name=f"IPOPT {name}",
                legendgroup=name,
                marker=marker,
                customdata=[
                    ", ".join(counts[t][name]) or "none" for t in ex.SWEEP_HOURS
                ],
                hovertemplate=f"{name}: %{{y}}, MVA %{{customdata}}<extra></extra>",
            ),
            row=3,
            col=1,
        )
    seconds = ex.seconds_per_solve(sweep)
    for name, value, marker in (
        ("IPOPT", seconds["IPOPT"], _bar_marker(CB_SKY)),
        ("SCIP<br>exact", seconds["SCIP"], _bar_marker(CB_BLUE)),
        (
            "SCIP<br>relaxation",
            seconds["relaxation"],
            _bar_marker(CB_BLUE, pattern="/"),
        ),
    ):
        fig.add_trace(
            go.Bar(
                x=[name],
                y=[value],
                marker=marker,
                showlegend=False,
                text=[f"{value:.2f} s"],
                textposition="outside",
                cliponaxis=False,
                hovertemplate="%{y:.3f} s per hourly solve<extra></extra>",
            ),
            row=3,
            col=2,
        )
    _base_layout(fig, "IPOPT near the network limits", height=1150)
    fig.update_layout(
        barmode="stack",
        showlegend=True,
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.07,
            "xanchor": "center",
            "x": 0.5,
            "font": {"size": 13, "color": TEXT},
            "bgcolor": C_BLACK,
        },
        margin={"l": 60, "r": 25, "t": 90, "b": 150},
    )
    ratings = ex.SWEEP_CABLE_MVA
    for row, t in enumerate(ex.SWEEP_HOURS, start=1):
        highest = max(
            sweep[label][key].cost
            for label in ("SCIP", "IPOPT")
            for key in sweep["SCIP"]
            if key[1] == t and not sweep[label][key].failed
        )
        fig.update_xaxes(
            tickvals=list(ratings),
            tickformat=".2f",
            range=[min(ratings) - 0.05, max(ratings) + 0.05],
            row=row,
            col=1,
        )
        fig.update_yaxes(title_text="cost", range=[0, 1.12 * highest], row=row, col=1)
    fig.update_xaxes(title_text="cable rating, MVA", row=2, col=1)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(
        title_text="cable ratings", dtick=1, range=[0, len(ratings) + 0.5], row=3, col=1
    )
    fig.update_xaxes(tickangle=0, row=3, col=2)
    fig.update_yaxes(title_text="seconds", rangemode="tozero", row=3, col=2)
    return _write(fig, out_path, height_px=1150)


def build_tutorial_mes_local_optima_nlp(out_path):
    """The network limits day's dispatches by SCIP and by APOPT on IPOPT's
    NLP formulation: the cost of each hour there against each solver's own,
    and how far cost and squared pressure move between the formulations
    (tutorials/mes_local_optima.rst)."""
    ex, results = _mes_local_optima_run()
    day = results.day
    x = list(day["SCIP"])
    solvers = (
        ("APOPT", _bar_marker(C_REFERENCE, pattern="/"), -1),
        ("SCIP", _bar_marker(CB_BLUE), 0),
    )
    fig = make_subplots(
        rows=2,
        cols=2,
        horizontal_spacing=0.12,
        vertical_spacing=0.16,
        row_heights=[0.55, 0.45],
        specs=[[{"colspan": 2}, None], [{}, {}]],
        subplot_titles=(
            "Cost of each hour on IPOPT's formulation",
            "IPOPT's cost minus the solver's own",
            "Largest difference in squared pressure",
        ),
    )
    width = 0.38
    nlp = {solver: day[f"{solver} on NLP"] for solver, _, _ in solvers}
    excess = {t: nlp["APOPT"][t].cost - nlp["SCIP"][t].cost for t in x}
    for solver, marker, side in solvers:
        own, on_nlp = day[solver], nlp[solver]
        fig.add_trace(
            go.Bar(
                x=x,
                y=[on_nlp[t].cost for t in x],
                width=width,
                offset=side * width,
                marker=marker,
                name=f"{solver}'s dispatch, day {sum(h.cost for h in on_nlp.values()):.2f}",
                legendgroup=solver,
                text=[
                    f"+{excess[t]:.2f}"
                    if solver == "APOPT" and excess[t] > 0.01
                    else ""
                    for t in x
                ],
                textposition="outside",
                constraintext="none",
                cliponaxis=False,
                hovertemplate=(
                    f"{solver}'s dispatch, hour %{{x}}: %{{y:.4f}} on IPOPT's "
                    "formulation<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[t + (side + 0.5) * width for t in x],
                y=[own[t].cost for t in x],
                mode="markers",
                marker={
                    "symbol": "line-ew",
                    "size": 26,
                    "line": {"color": TEXT, "width": 3},
                },
                name="each solver's cost on its own formulation",
                legendgroup="own",
                showlegend=solver == "SCIP",
                customdata=[solver] * len(x),
                hovertemplate=(
                    "%{customdata} on its own formulation, hour %{x:.0f}: "
                    "%{y:.4f}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
        for col, values, fmt in (
            (1, [on_nlp[t].cost - own[t].cost for t in x], ".5f"),
            (
                2,
                [
                    max(
                        abs(a - b)
                        for a, b in zip(
                            on_nlp[t].quantities["psq"], own[t].quantities["psq"]
                        )
                    )
                    for t in x
                ],
                ".3g",
            ),
        ):
            fig.add_trace(
                go.Bar(
                    x=x,
                    y=values,
                    width=width,
                    offset=side * width,
                    marker=marker,
                    legendgroup=solver,
                    showlegend=False,
                    text=[f"{v:.3f}" if col == 2 and v > 0.01 else "" for v in values],
                    textposition="outside",
                    textfont={"size": 12, "color": TEXT},
                    constraintext="none",
                    cliponaxis=False,
                    hovertemplate=(
                        f"{solver}, hour %{{x}}: %{{y:{fmt}}}<extra></extra>"
                    ),
                ),
                row=2,
                col=col,
            )
    fig.add_hline(
        y=1e-3,
        line={"color": C_REFERENCE, "width": 2, "dash": "dash"},
        annotation_text="tolerance of the check",
        annotation_position="top left",
        annotation_font={"size": 12},
        row=2,
        col=1,
    )
    _base_layout(fig, "Both dispatches on IPOPT's formulation", height=900)
    fig.update_layout(
        showlegend=True,
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.1,
            "xanchor": "center",
            "x": 0.5,
            "font": {"size": 13, "color": TEXT},
            "bgcolor": C_BLACK,
        },
        margin={"l": 70, "r": 25, "t": 90, "b": 110},
    )
    fig.update_xaxes(
        dtick=1, range=[-0.6, len(x) - 0.4], showgrid=False, title_text="hour"
    )
    fig.update_yaxes(
        title_text="cost of the hour",
        range=[0, 1.15 * max(h.cost for h in nlp["APOPT"].values())],
        row=1,
        col=1,
    )
    lowest = min(
        0.0, *(nlp[s][t].cost - day[s][t].cost for s, _, _ in solvers for t in x)
    )
    fig.update_yaxes(range=[1.2 * lowest, 1.25e-3], tickformat=".4f", row=2, col=1)
    fig.update_yaxes(rangemode="tozero", row=2, col=2)
    return _write(fig, out_path, height_px=900)


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
    ("tutorial_mes_dispatch.html", build_tutorial_mes_dispatch),
    ("tutorial_mes_checks.html", build_tutorial_mes_checks),
    ("tutorial_mes_storage_dispatch.html", build_tutorial_mes_storage_dispatch),
    ("tutorial_mes_storage_checks.html", build_tutorial_mes_storage_checks),
    ("tutorial_mes_limits_dispatch.html", build_tutorial_mes_limits_dispatch),
    ("tutorial_mes_limits_checks.html", build_tutorial_mes_limits_checks),
    ("tutorial_mes_local_optima_day.html", build_tutorial_mes_local_optima_day),
    ("tutorial_mes_local_optima_sweep.html", build_tutorial_mes_local_optima_sweep),
    ("tutorial_mes_local_optima_nlp.html", build_tutorial_mes_local_optima_nlp),
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

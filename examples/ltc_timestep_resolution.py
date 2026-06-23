"""LTC under different timestep resolutions (panel (c) only).

A focused companion to ``examples/linepack_ltc_capabilities.py``: it keeps only
the heat side of that network (a short line ``ext-grid -> pipe -> sink`` with
:class:`~monee.model.formulation.ltc.LumpedThermalCapacitance` attached) and
asks a single question:

    how does the LumpedThermalCapacitance transient depend on the simulation
    timestep Δt?

The same physical boundary schedule (a supply-temperature step down and back
up) is solved over the same wall-clock horizon at several Δt values. Because
LTC discretises the lumped-capacitance ODE with backward Euler
(``rho_v · (T(t) - T(t-1))/Δt = net_heat``), a fine Δt resolves the smooth
exponential roll-off, while a coarse Δt under-resolves it into a few large
jumps.

All curves are drawn on a common physical time axis with a zero-order hold
(step shape), so a coarse run literally holds ("duplicates") its value on the
sub-steps it never computed - which is exactly how a coarse result looks when
overlaid on a finer grid. As Δt shrinks the staircase converges to the
continuous transient.

Requires: plotly + pandas. kaleido is optional (static PNG export); the
interactive HTML is always written.

Run::

    python examples/ltc_timestep_resolution.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import monee.model as mm
from monee import LumpedThermalCapacitance, run_timeseries
from monee.simulation.timeseries import TimeseriesData

# Shared heat-line geometry (identical to the heat side of the capabilities
# example) and the boundary schedule, expressed in continuous time.
PIPE_DIAMETER_M = 0.15
PIPE_LENGTH_M = 300
SINK_MASS_FLOW_KGS = 2.0

T_TOTAL_H = 10.0  # wall-clock horizon shared by every run
T_SUPPLY_HIGH_K = 356.0
T_SUPPLY_LOW_K = 338.0
STEP_DOWN_H = 1.0  # supply drops here ...
STEP_UP_H = 5.0  # ... and recovers here

# Timestep resolutions to compare [h]; the finest doubles as the reference.
DT_HOURS = [2.0, 1.0, 0.5, 0.1]

BASE_TS = pd.Timestamp("2024-01-01")


def supply_t_k(t_h: float) -> float:
    """Continuous-time supply temperature: high, step down, step back up."""
    if t_h < STEP_DOWN_H or t_h >= STEP_UP_H:
        return T_SUPPLY_HIGH_K
    return T_SUPPLY_LOW_K


def build_heat_net():
    """Heat line ``ext-grid (supply) -> pipe -> sink`` with LTC attached.

    Returns ``(net, supply_child_id, consumer_node_id)``.
    """
    net = mm.Network()
    w_supply = net.child(mm.ExtHydrGrid(t_k=T_SUPPLY_HIGH_K))
    n_supply = net.node(mm.Junction(), mm.WATER, child_ids=[w_supply])
    n_consumer = net.node(
        mm.Junction(),
        mm.WATER,
        child_ids=[net.child(mm.Sink(mass_flow_kgs=SINK_MASS_FLOW_KGS))],
    )
    net.branch(
        mm.WaterPipe(diameter_m=PIPE_DIAMETER_M, length_m=PIPE_LENGTH_M),
        n_supply,
        n_consumer,
    )
    net.add_extension(LumpedThermalCapacitance(first_step_steady_state=True))
    return net, w_supply, n_consumer


def solve_for_dt(dt_h: float):
    """Solve the LTC transient at timestep *dt_h*; return (times_h, t_pu)."""
    net, w_supply, consumer = build_heat_net()
    n_steps = int(round(T_TOTAL_H / dt_h)) + 1
    times_h = [i * dt_h for i in range(n_steps)]
    datetime_index = pd.DatetimeIndex(
        [BASE_TS + pd.Timedelta(hours=t) for t in times_h]
    )

    td = TimeseriesData()
    td.add_child_series(w_supply, "t_k", [supply_t_k(t) for t in times_h])

    ts = run_timeseries(
        net,
        timeseries_data=td,
        steps=n_steps,
        datetime_index=datetime_index,
        solver="ipopt",
    )
    t_pu = [float(v) for v in ts.get_result_for_id(consumer, "t_pu")]
    return times_h, t_pu


# Libertinus-first serif stack, falling back to whatever serif the renderer has.
FONT_FAMILY = "Libertinus Serif, Libertinus, Times New Roman, serif"

C_SUPPLY = "#33BBEE"  # cyan — supply-temperature boundary

# Publication sizing for a one-column dissertation figure.
TITLE_SIZE = 25
AXIS_TITLE_SIZE = 23
TICK_SIZE = 20
LEGEND_SIZE = 19
LINE_WIDTH = 3.6
MARKER_SIZE = 9


def make_plot(curves, out_path: Path):
    """Render the single temperature panel for every Δt resolution.

    ``curves`` is a list of ``(dt_h, times_h, t_pu)`` ordered coarse -> fine.
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Boundary schedule on the secondary axis, sampled fine for a clean step.
    fine = [i * 0.02 for i in range(int(T_TOTAL_H / 0.02) + 1)]
    fig.add_trace(
        go.Scatter(
            x=fine,
            y=[supply_t_k(t) for t in fine],
            mode="lines",
            line=dict(shape="hv", color=C_SUPPLY, width=2.0, dash="dot"),
            name=r"$T_{\mathrm{supply}}$",
        ),
        secondary_y=True,
    )

    n = len(curves)
    colors = pc.sample_colorscale(
        "Viridis", [0.12 + 0.7 * k / max(1, n - 1) for k in range(n)]
    )
    for (dt_h, times_h, t_pu), color in zip(curves, colors):
        is_fine = dt_h == min(c[0] for c in curves)
        n_pts = len(times_h)
        fig.add_trace(
            go.Scatter(
                x=times_h,
                y=t_pu,
                mode="lines" if n_pts > 25 else "lines+markers",
                line=dict(
                    shape="hv",
                    color=color,
                    width=LINE_WIDTH + (1.0 if is_fine else 0.0),
                ),
                marker=dict(size=MARKER_SIZE),
                name=rf"$\Delta t = {dt_h:g}\,\mathrm{{h}}$",
            ),
            secondary_y=False,
        )

    fig.update_yaxes(
        title_text=r"$\Large T_{\mathrm{consumer}}\;\;[\mathrm{pu}]$",
        title_font_size=AXIS_TITLE_SIZE,
        tickfont_size=TICK_SIZE,
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text=r"$\Large T_{\mathrm{supply}}\;\;[\mathrm{K}]$",
        title_font_size=AXIS_TITLE_SIZE,
        tickfont_size=TICK_SIZE,
        secondary_y=True,
        color=C_SUPPLY,
        showgrid=False,
    )
    fig.update_xaxes(
        title_text=r"$\Large \text{time } t\;\;[\mathrm{h}]$",
        title_font_size=AXIS_TITLE_SIZE,
        tickfont_size=TICK_SIZE,
    )

    fig.update_layout(
        title=dict(
            text="LTC consumer-temperature transient vs timestep Δt",
            font=dict(size=TITLE_SIZE),
            x=0.5,
            xanchor="center",
        ),
        font=dict(family=FONT_FAMILY, size=TICK_SIZE),
        template="plotly_white",
        legend=dict(
            x=0.30,
            y=0.97,
            xanchor="center",
            yanchor="top",
            font=dict(size=LEGEND_SIZE),
            bgcolor="rgba(255,255,255,0.80)",
            bordercolor="rgba(110,110,110,0.6)",
            borderwidth=1,
        ),
        width=950,
        height=620,
        margin=dict(l=90, r=30, t=60, b=70),
    )

    html_path = out_path.with_suffix(".html")
    fig.write_html(html_path)
    saved = [html_path]
    try:
        fig.write_image(out_path, scale=2)
        saved.append(out_path)
    except Exception as exc:  # kaleido missing or no renderer
        print(f"(skipped PNG export: {exc})")
    return saved


def main() -> None:
    curves = []
    for dt_h in sorted(DT_HOURS, reverse=True):  # coarse -> fine
        print(f"Solving LTC transient at dt = {dt_h:g} h ...")
        times_h, t_pu = solve_for_dt(dt_h)
        curves.append((dt_h, times_h, t_pu))

    out_path = Path(__file__).resolve().parent / "ltc_timestep_resolution.png"
    saved = make_plot(curves, out_path)

    print()
    for p in saved:
        print(f"Saved -> {p}")
    print()
    print("Settled consumer t_pu (last step) per resolution:")
    for dt_h, _times, t_pu in curves:
        print(
            f"  dt = {dt_h:>4g} h : steps = {len(t_pu):>4d}  t_pu_end = {t_pu[-1]:.5f}"
        )


if __name__ == "__main__":
    main()

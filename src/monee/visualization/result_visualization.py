"""Annotated interactive graph visualization for :class:`~monee.solver.core.SolverResult`.

Entry point: :func:`plot_result`.
"""

import math

import networkx as nx
import plotly.graph_objects as go

from monee.solver.core import SolverResult

# ---------------------------------------------------------------------------
# Theme  –  clean light mode
# ---------------------------------------------------------------------------
_BG = "#ffffff"  # pure white canvas
_PANEL = "#f6f8fa"  # hover tooltip background
_BORDER = "#d0d7de"  # subtle border / separator
_FONT_COLOR = "#1f2328"  # near-black primary text
_DIM_COLOR = "#656d76"  # secondary / label text
_FONT = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"

# Traffic-light palette – readable on white
_TL_GREEN = "#22c55e"  # emerald
_TL_YELLOW = "#eab308"  # amber
_TL_RED = "#ef4444"  # red
_TL_GRAY = "#94a3b8"  # slate

# Per-grid accent colours for node borders
_ACCENT: dict[str, str] = {
    "power": "#2563eb",  # blue      – electricity
    "water": "#dc2626",  # red       – heat / water
    "gas": "#0891b2",  # cyan      – gas
    "cp": "#9333ea",  # purple    – control point
}

# Node shapes follow the existing visualization.py conventions
_GRID_SYMBOL: dict[str, str] = {
    "power": "square",
    "water": "pentagon",
    "gas": "triangle-up",
    "cp": "diamond",
}
_GRID_LABEL: dict[str, str] = {
    "power": "Electricity",
    "water": "Heat / Water",
    "gas": "Gas",
    "cp": "Control Point",
}

# Columns to hide from hover text
_META_COLS: frozenset[str] = frozenset({"active", "independent", "ignored"})
_ID_COLS: frozenset[str] = frozenset({"id", "node_id"})
_SKIP: frozenset[str] = _META_COLS | _ID_COLS | frozenset({"_type"})


# ---------------------------------------------------------------------------
# Value formatting
# ---------------------------------------------------------------------------


def _fmt(v) -> str:
    """Format a result value concisely for display."""
    if v is None:
        return "—"
    try:
        f = float(v)
        if math.isnan(f):
            return "—"
        return f"{f:.4g}"
    except (TypeError, ValueError):
        return str(v)


# ---------------------------------------------------------------------------
# Traffic-light helpers
# ---------------------------------------------------------------------------


def _bus_color(vm_pu) -> str:
    try:
        v = float(vm_pu)
    except (TypeError, ValueError):
        return _TL_GRAY
    if math.isnan(v):
        return _TL_GRAY
    if 0.95 <= v <= 1.05:
        return _TL_GREEN
    if 0.90 <= v <= 1.10:
        return _TL_YELLOW
    return _TL_RED


def _line_color(loading_pct) -> str:
    try:
        v = float(loading_pct)
    except (TypeError, ValueError):
        return _TL_GRAY
    if math.isnan(v):
        return _TL_GRAY
    if v < 70:
        return _TL_GREEN
    if v < 90:
        return _TL_YELLOW
    return _TL_RED


# ---------------------------------------------------------------------------
# Grid-type detection (mirrors existing visualization.py)
# ---------------------------------------------------------------------------


def _grid_type(grid) -> str:
    g = str(type(grid))
    if "Power" in g:
        return "power"
    if "Water" in g:
        return "water"
    if "Gas" in g:
        return "gas"
    return "cp"


# ---------------------------------------------------------------------------
# Build result lookup maps
# ---------------------------------------------------------------------------


def _node_result_map(result: SolverResult) -> dict:
    """node_id → result-row dict for all node types (Bus, Junction, …)."""
    m: dict = {}
    for type_name, df in result.dataframes.items():
        if df.empty or "id" not in df.columns:
            continue
        if "node_id" in df.columns:
            continue  # child — skip
        if isinstance(df["id"].iloc[0], tuple):
            continue  # branch — skip
        for _, row in df.iterrows():
            m[row["id"]] = {"_type": type_name, **row.to_dict()}
    return m


def _branch_result_map(result: SolverResult) -> dict:
    """branch_id (from, to, key) → result-row dict for all branch types.

    Both orderings of the endpoint pair are registered so that undirected
    MultiGraph edge iteration (which may reverse the stored direction) always
    finds the correct row.
    """
    m: dict = {}
    for type_name, df in result.dataframes.items():
        if df.empty or "id" not in df.columns:
            continue
        if not isinstance(df["id"].iloc[0], tuple):
            continue
        for _, row in df.iterrows():
            entry = {"_type": type_name, **row.to_dict()}
            bid = row["id"]
            m[bid] = entry
            # reversed direction alias so graph.edges() order never misses
            m[(bid[1], bid[0], bid[2])] = entry
    return m


def _child_by_node_map(result: SolverResult) -> dict:
    """node_id → list of child result-row dicts attached to that node."""
    m: dict = {}
    for type_name, df in result.dataframes.items():
        if df.empty or "node_id" not in df.columns:
            continue
        for _, row in df.iterrows():
            m.setdefault(row["node_id"], []).append(
                {"_type": type_name, **row.to_dict()}
            )
    return m


# ---------------------------------------------------------------------------
# Hover text builders
# ---------------------------------------------------------------------------


def _sep(label: str = "") -> str:
    if not label:
        return f"<span style='color:{_BORDER}'>{'─' * 26}</span>"
    return f"<span style='color:{_DIM_COLOR};font-size:10px'>{label.upper()}</span>"


def _node_hover(row: dict, children: list[dict], node_name: str | None) -> str:
    type_name = row.get("_type", "Node")
    node_id = row.get("id", "?")
    if node_name:
        header = (
            f"<b>{node_name}</b>"
            f"  <span style='color:{_DIM_COLOR}'>{type_name} #{node_id}</span>"
        )
    else:
        header = f"<b>{type_name} #{node_id}</b>"

    lines = [header, _sep()]
    for k, v in row.items():
        if k in _SKIP:
            continue
        lines.append(
            f"<span style='color:{_DIM_COLOR}'>{k}</span>&nbsp;&nbsp;<b>{_fmt(v)}</b>"
        )
    if children:
        lines.append("<br>" + _sep("attached"))
        for c in children:
            ctype = c.get("_type", "?")
            vals = "  ".join(
                f"<span style='color:{_DIM_COLOR}'>{k}</span>&nbsp;{_fmt(v)}"
                for k, v in c.items()
                if k not in _SKIP
            )
            lines.append(f"<i>[{ctype}]</i>&nbsp;&nbsp;{vals}")
    return "<br>".join(lines)


def _branch_hover(row: dict, from_id, to_id, branch_name: str | None) -> str:
    type_name = row.get("_type", "Branch")
    if branch_name:
        header = (
            f"<b>{branch_name}</b>  <span style='color:{_DIM_COLOR}'>{type_name}</span>"
        )
    else:
        header = f"<b>{type_name}</b>"

    lines = [
        header,
        f"<span style='color:{_DIM_COLOR}'>{from_id} → {to_id}</span>",
        _sep(),
    ]
    for k, v in row.items():
        if k in _SKIP:
            continue
        lines.append(
            f"<span style='color:{_DIM_COLOR}'>{k}</span>&nbsp;&nbsp;<b>{_fmt(v)}</b>"
        )
    return "<br>".join(lines)


# ---------------------------------------------------------------------------
# Key-metric label + traffic-light color
# ---------------------------------------------------------------------------


def _node_label_and_color(row: dict) -> tuple[str, str]:
    t = row.get("_type", "")
    if t == "Bus":
        vm = row.get("vm_pu")
        if vm is not None:
            try:
                return f"{float(vm):.3f} pu", _bus_color(vm)
            except (TypeError, ValueError):
                pass
    if "pressure_pu" in row:
        p = row.get("pressure_pu")
        if p is not None:
            try:
                return f"{float(p):.3f} pu", _TL_GRAY
            except (TypeError, ValueError):
                pass
    return "", _TL_GRAY


def _branch_label_and_color(row: dict, is_cp: bool = False) -> tuple[str, str]:
    """Return (short inline label, colour) for a branch result row.

    Single-grid branches use the traffic-light palette; coupling branches
    fall back to the CP accent colour.
    """
    # single-grid electrical loading
    for col in ("loading_percent", "loading_from_percent"):
        v = row.get(col)
        if v is not None:
            try:
                return f"{float(v):.0f}%", _line_color(v)
            except (TypeError, ValueError):
                pass

    # single-grid hydraulic mass flow
    for col in ("mass_flow", "mass_flow_pos"):
        v = row.get(col)
        if v is not None:
            try:
                f = float(v)
                if not math.isnan(f):
                    return f"{f:.3g} kg/s", _TL_GREEN
            except (TypeError, ValueError):
                pass

    # multi-grid: electrical power
    cp_color = _ACCENT["cp"]
    for col in ("el_mw", "p_mw", "p_from_mw", "p_to_mw"):
        v = row.get(col)
        if v is not None:
            try:
                f = float(v)
                if not math.isnan(f):
                    return f"{f:.3g} MW", cp_color
            except (TypeError, ValueError):
                pass

    # multi-grid: gas / hydraulic flow
    for col in ("gas_kgps", "from_mass_flow", "to_mass_flow"):
        v = row.get(col)
        if v is not None:
            try:
                f = float(v)
                if not math.isnan(f):
                    return f"{f:.3g} kg/s", cp_color
            except (TypeError, ValueError):
                pass

    # multi-grid: heat
    for col in ("heat_w", "q_w"):
        v = row.get(col)
        if v is not None:
            try:
                f = float(v)
                if not math.isnan(f):
                    return f"{f:.3g} W", cp_color
            except (TypeError, ValueError):
                pass

    return "", cp_color if is_cp else _TL_GRAY


# ---------------------------------------------------------------------------
# Graph layout  –  spread out nodes for readability
# ---------------------------------------------------------------------------


def _compute_layout(graph: nx.Graph, network, use_monee_positions: bool) -> dict:
    if use_monee_positions:
        return {
            nid: (
                graph.nodes[nid]["internal_node"].position[0],
                graph.nodes[nid]["internal_node"].position[1],
            )
            for nid in graph.nodes
        }

    pos = None
    for prog, args in [
        ("fdp", "-Goverlap=false -Gmode=ipsep -Gsep=200"),
    ]:
        try:
            import networkx.drawing.nx_agraph as nxd

            pos = nxd.pygraphviz_layout(graph, prog=prog, args=args)
            break
        except Exception:
            continue

    return pos


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------


def plot_result(
    result: SolverResult,
    title: str | None = None,
    show_children: bool = True,
    use_monee_positions: bool = False,
    write_to: str | None = None,
) -> go.Figure:
    """Plot a :class:`~monee.solver.core.SolverResult` as an annotated
    interactive network graph.

    **Node coloring** (traffic-light):

    * Electrical buses: green when ``vm_pu ∈ [0.95, 1.05]``, yellow for
      ``[0.90, 0.95)`` or ``(1.05, 1.10]``, red otherwise.
    * Gas / water junctions: neutral gray with the current ``pressure_pu``
      as an inline label.

    **Branch coloring** (traffic-light):

    * Power lines / transformers: green ``< 70 %``, yellow ``70–90 %``,
      red ``≥ 90 %`` loading.
    * Hydraulic pipes: green, labeled with mass flow (kg/s).
    * Multi-grid (CP) branches: dotted.

    Hover over any node or branch to see the full result table for that
    component.  Children (loads, generators, ext-grids, …) are listed in
    their parent node's hover text when *show_children* is ``True``.

    Args:
        result: The :class:`~monee.solver.core.SolverResult` to visualise.
        title: Figure title.  Defaults to ``"Network Result"``.
        show_children: Include child components in parent-node hover text.
        use_monee_positions: Use stored ``node.position`` coordinates.
        write_to: Optional path to export the figure (PDF / PNG / SVG).

    Returns:
        A :class:`plotly.graph_objects.Figure`.
    """
    network = result.network
    graph: nx.Graph = network._network_internal

    node_map = _node_result_map(result)
    branch_map = _branch_result_map(result)
    child_map = _child_by_node_map(result) if show_children else {}
    pos = _compute_layout(graph, network, use_monee_positions)

    # -----------------------------------------------------------------------
    # Node data – collected per grid type
    # -----------------------------------------------------------------------
    grid_data: dict[str, dict] = {
        g: {"x": [], "y": [], "tl_colors": [], "hover": [], "labels": []}
        for g in ("power", "water", "gas", "cp")
    }

    for node_id in graph.nodes:
        int_node = graph.nodes[node_id]["internal_node"]
        gtype = "cp" if not int_node.independent else _grid_type(int_node.grid)

        x, y = pos[node_id]
        row = node_map.get(node_id, {})
        children = child_map.get(node_id, [])
        nname = getattr(int_node, "name", None)

        label, tl_color = _node_label_and_color(row) if row else ("", _TL_GRAY)
        hover = _node_hover(row, children, nname) if row else f"node {node_id}"

        d = grid_data[gtype]
        d["x"].append(x)
        d["y"].append(y)
        d["tl_colors"].append(tl_color)
        d["hover"].append(hover)
        d["labels"].append(label)

    # Build traces: glow behind nodes, then the actual markers on top
    glow_traces = []
    marker_traces = []

    for gtype, d in grid_data.items():
        if not d["x"]:
            continue

        # Soft glow – wide semi-transparent shape renders beneath the marker
        glow_traces.append(
            go.Scatter(
                x=d["x"],
                y=d["y"],
                mode="markers",
                hoverinfo="skip",
                showlegend=False,
                marker=dict(
                    symbol=_GRID_SYMBOL[gtype],
                    size=42,
                    color=d["tl_colors"],
                    opacity=0.12,
                    line=dict(width=0),
                ),
            )
        )

        # Main node markers
        marker_traces.append(
            go.Scatter(
                x=d["x"],
                y=d["y"],
                mode="markers+text",
                textposition="top center",
                text=d["labels"],
                textfont=dict(family=_FONT, size=11, color=_DIM_COLOR),
                hovertext=d["hover"],
                hoverinfo="text",
                name=_GRID_LABEL[gtype],
                marker=dict(
                    symbol=_GRID_SYMBOL[gtype],
                    size=24,
                    color=d["tl_colors"],
                    opacity=0.88,
                    line=dict(width=3, color=_ACCENT[gtype]),
                ),
            )
        )

    # -----------------------------------------------------------------------
    # Branch traces
    # Lines are grouped by (color, is_cp) – one Scatter per color group.
    # A midpoint-marker trace carries per-branch hover text + inline labels.
    # -----------------------------------------------------------------------
    color_groups: dict[tuple, list] = {}

    mid_x: list[float] = []
    mid_y: list[float] = []
    mid_hover: list[str] = []
    mid_label: list[str] = []
    mid_colors: list[str] = []

    for from_node, to_node, key in graph.edges(keys=True):
        branch_id = (from_node, to_node, key)
        row = branch_map.get(branch_id, {})
        int_branch = graph.edges[from_node, to_node, key]["internal_branch"]
        is_cp = int_branch.model.is_cp()  # use the model's own declaration
        bname = getattr(int_branch, "name", None)

        label, color = (
            _branch_label_and_color(row, is_cp=is_cp)
            if row
            else ("", _ACCENT["cp"] if is_cp else _TL_GRAY)
        )

        hover = (
            _branch_hover(row, from_node, to_node, bname)
            if row
            else f"{from_node} → {to_node}"
        )

        x0, y0 = pos[from_node]
        x1, y1 = pos[to_node]
        color_groups.setdefault((color, is_cp), []).append((x0, y0, x1, y1))

        mid_x.append((x0 + x1) / 2)
        mid_y.append((y0 + y1) / 2)
        mid_hover.append(hover)
        mid_label.append(label)
        mid_colors.append(color)

    edge_traces = []
    for (color, is_cp), segs in color_groups.items():
        x_pts: list = []
        y_pts: list = []
        for x0, y0, x1, y1 in segs:
            x_pts += [x0, x1, None]
            y_pts += [y0, y1, None]
        edge_traces.append(
            go.Scatter(
                x=x_pts,
                y=y_pts,
                mode="lines",
                hoverinfo="none",
                showlegend=False,
                line=dict(
                    color=color,
                    width=3.5 if not is_cp else 2,
                    dash="dot" if is_cp else "solid",
                ),
                opacity=0.65,
            )
        )

    midpoint_trace = go.Scatter(
        x=mid_x,
        y=mid_y,
        mode="markers+text",
        text=mid_label,
        textposition="middle right",
        textfont=dict(family=_FONT, size=10, color=_DIM_COLOR),
        hovertext=mid_hover,
        hoverinfo="text",
        showlegend=False,
        marker=dict(
            size=9,
            color=mid_colors,
            symbol="circle",
            opacity=0.90,
            line=dict(width=1.5, color=_BG),
        ),
    )

    # -----------------------------------------------------------------------
    # Traffic-light legend entries
    # -----------------------------------------------------------------------
    tl_legend = [
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=11, color=_TL_GREEN, symbol="square", line=dict(width=0)),
            name="OK  (< 70 % / vm ±5 %)",
        ),
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=11, color=_TL_YELLOW, symbol="square", line=dict(width=0)),
            name="Warning  (70–90 % / vm ±10 %)",
        ),
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=11, color=_TL_RED, symbol="square", line=dict(width=0)),
            name="Critical  (≥ 90 % / vm > ±10 %)",
        ),
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color=_ACCENT["cp"], width=2, dash="dot"),
            name="Coupling branch (CP)",
        ),
    ]

    # -----------------------------------------------------------------------
    # Assemble  –  render order: edges → midpoints → glow → markers → legend
    # -----------------------------------------------------------------------
    all_traces = (
        edge_traces + [midpoint_trace] + glow_traces + marker_traces + tl_legend
    )

    fig = go.Figure(
        data=all_traces,
        layout=go.Layout(
            title=dict(
                text=title or "Network Result",
                font=dict(family=_FONT, size=18, color=_FONT_COLOR),
                x=0.5,
                xanchor="center",
                y=0.97,
            ),
            paper_bgcolor=_BG,
            plot_bgcolor=_BG,
            hovermode="closest",
            hoverlabel=dict(
                bgcolor=_PANEL,
                bordercolor=_BORDER,
                font=dict(family=_FONT, size=12, color=_FONT_COLOR),
                namelength=-1,
            ),
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                showline=False,
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                showline=False,
                scaleanchor="x",  # equal aspect ratio keeps shapes undistorted
            ),
            font=dict(family=_FONT, color=_FONT_COLOR),
            autosize=True,
            margin=dict(l=30, r=200, t=60, b=30),
            legend=dict(
                title=dict(
                    text="Legend",
                    font=dict(family=_FONT, size=12, color=_DIM_COLOR),
                ),
                x=1.02,
                y=1.0,
                xanchor="left",
                yanchor="top",
                bgcolor="rgba(246, 248, 250, 0.95)",
                bordercolor=_BORDER,
                borderwidth=1,
                font=dict(family=_FONT, size=11, color=_FONT_COLOR),
                itemsizing="constant",
                tracegroupgap=6,
            ),
        ),
    )

    if write_to is not None:
        fig.write_image(write_to)

    return fig

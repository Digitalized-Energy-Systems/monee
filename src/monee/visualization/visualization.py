"""Annotated interactive graph visualization for :class:`~monee.model.Network`.

Entry point: :func:`plot_network`.
"""

import networkx as nx
import plotly.graph_objects as go

import monee.model as mm  # kept for Network type hint only
from monee.model.core import Intermediate, IntermediateEq, Var

# Import shared theme and layout helpers from result_visualization so the two
# functions always look identical.
from monee.visualization.result_visualization import (
    _ACCENT,
    _BG,
    _BORDER,
    _DIM_COLOR,
    _FONT,
    _FONT_COLOR,
    _GRID_LABEL,
    _GRID_SYMBOL,
    _PANEL,
    _TL_GRAY,
    _compute_layout,
    _fmt,
    _grid_type,
    _sep,
)

_SKIP_ATTRS: frozenset[str] = frozenset({"active", "independent", "ignored"})

# Pyomo-like solver objects – hide these from hover text (they carry no useful
# design-time information for the user).
_SOLVER_TYPES = (Var, Intermediate, IntermediateEq)


# ---------------------------------------------------------------------------
# Model parameter extraction
# ---------------------------------------------------------------------------


def _model_params(model) -> dict:
    """Extract scalar constructor parameters from a model instance.

    Skips private attributes and solver-variable objects (Var, Intermediate).
    Returns only plain Python scalars (int, float, bool, str).
    """
    params: dict = {}
    try:
        attrs = vars(model)
    except TypeError:
        return params
    for name, val in attrs.items():
        if name.startswith("_"):
            continue
        if isinstance(val, _SOLVER_TYPES):
            continue
        if callable(val):
            continue
        if isinstance(val, (int, float, bool, str)):
            params[name] = val
    return params


# ---------------------------------------------------------------------------
# Inline labels shown directly on the graph
# ---------------------------------------------------------------------------


def _node_label(int_node) -> str:
    """Short text placed above a node marker."""
    model = int_node.model
    bkv = getattr(model, "base_kv", None)
    if bkv is not None:
        try:
            return f"{float(bkv):.4g} kV"
        except (TypeError, ValueError):
            pass
    return ""


def _branch_label(int_branch) -> str:
    """Short text placed at a branch midpoint."""
    model = int_branch.model
    parts: list[str] = []

    d = getattr(model, "diameter_m", None)
    if d is not None:
        try:
            mm_val = float(d) * 1000
            parts.append(f"⌀{mm_val:.0f}mm")
        except (TypeError, ValueError):
            pass

    length = getattr(model, "length_m", None)
    if length is not None:
        try:
            lm = float(length)
            parts.append(f"{lm / 1000:.4g}km" if lm >= 1000 else f"{lm:.4g}m")
        except (TypeError, ValueError):
            pass

    if not parts:
        # PowerLine fallback: total resistance
        r = getattr(model, "r_ohm_per_m", None)
        if r is not None and length is not None:
            try:
                parts.append(f"{float(r) * float(length):.3g} Ω")
            except (TypeError, ValueError):
                pass

    return "  ".join(parts)


# ---------------------------------------------------------------------------
# Rich hover text
# ---------------------------------------------------------------------------


def _node_hover(int_node, children: list) -> str:
    """HTML hover for a network node, showing all scalar model parameters."""
    model = int_node.model
    typename = type(model).__name__
    nid = getattr(int_node, "id", "?")
    nname = getattr(int_node, "name", None)

    if nname:
        header = (
            f"<b>{nname}</b>  <span style='color:{_DIM_COLOR}'>{typename} #{nid}</span>"
        )
    else:
        header = f"<b>{typename} #{nid}</b>"

    lines = [header, _sep()]
    for k, v in _model_params(model).items():
        if k in _SKIP_ATTRS:
            continue
        lines.append(
            f"<span style='color:{_DIM_COLOR}'>{k}</span>&nbsp;&nbsp;<b>{_fmt(v)}</b>"
        )

    if children:
        lines.append("<br>" + _sep("children"))
        for child in children:
            ctype = type(child.model).__name__
            cparams = _model_params(child.model)
            vals = "  ".join(
                f"<span style='color:{_DIM_COLOR}'>{k}</span>&nbsp;{_fmt(v)}"
                for k, v in cparams.items()
                if k not in _SKIP_ATTRS
            )
            lines.append(f"<i>[{ctype}]</i>&nbsp;&nbsp;{vals}")

    return "<br>".join(lines)


def _branch_hover(int_branch, from_id, to_id) -> str:
    """HTML hover for a network branch, showing all scalar model parameters."""
    model = int_branch.model
    typename = type(model).__name__
    bname = getattr(int_branch, "name", None)

    if bname:
        header = f"<b>{bname}</b>  <span style='color:{_DIM_COLOR}'>{typename}</span>"
    else:
        header = f"<b>{typename}</b>"

    lines = [
        header,
        f"<span style='color:{_DIM_COLOR}'>{from_id} → {to_id}</span>",
        _sep(),
    ]
    for k, v in _model_params(model).items():
        if k in _SKIP_ATTRS:
            continue
        lines.append(
            f"<span style='color:{_DIM_COLOR}'>{k}</span>&nbsp;&nbsp;<b>{_fmt(v)}</b>"
        )

    return "<br>".join(lines)


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------


def plot_network(
    network: mm.Network,
    title: str | None = None,
    show_children: bool = True,
    use_monee_positions: bool = False,
    write_to: str | None = None,
) -> go.Figure:
    """Plot a :class:`~monee.model.Network` as an annotated interactive graph.

    Nodes are colored by energy-carrier type (electricity, heat/water, gas,
    coupling point).  Branch midpoints show compact parameter labels.  Hovering
    over any element reveals the full set of scalar model parameters.

    Args:
        network: The :class:`~monee.model.Network` to visualise.
        title: Figure title.  Defaults to ``"Network"``.
        show_children: Show attached child components (loads, generators, …)
            in the parent node's hover tooltip.
        use_monee_positions: Use stored ``node.position`` coordinates instead
            of the automatic graph layout.
        write_to: Optional path to export the figure (PDF / PNG / SVG).

    Returns:
        A :class:`plotly.graph_objects.Figure`.
    """
    graph: nx.Graph = network._network_internal
    pos = _compute_layout(graph, network, use_monee_positions)

    # -----------------------------------------------------------------------
    # Node data – collected per grid type
    # -----------------------------------------------------------------------
    grid_data: dict[str, dict] = {
        g: {"x": [], "y": [], "hover": [], "labels": []}
        for g in ("power", "water", "gas", "cp")
    }

    for node_id in graph.nodes:
        int_node = graph.nodes[node_id]["internal_node"]
        gtype = "cp" if not int_node.independent else _grid_type(int_node.grid)

        x, y = pos[node_id]
        children = network.childs_by_ids(int_node.child_ids) if show_children else []

        d = grid_data[gtype]
        d["x"].append(x)
        d["y"].append(y)
        d["hover"].append(_node_hover(int_node, children))
        d["labels"].append(_node_label(int_node))

    # Build node traces: soft glow behind each marker
    glow_traces: list = []
    marker_traces: list = []

    for gtype, d in grid_data.items():
        if not d["x"]:
            continue
        color = _ACCENT[gtype]

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
                    color=color,
                    opacity=0.10,
                    line=dict(width=0),
                ),
            )
        )

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
                    color=color,
                    opacity=0.75,
                    line=dict(width=3, color=color),
                ),
            )
        )

    # -----------------------------------------------------------------------
    # Branch traces
    # Grouped by (color, is_cp) for efficient rendering; one midpoint trace
    # carries per-branch hover text and inline labels.
    # -----------------------------------------------------------------------
    color_groups: dict[tuple, list] = {}

    mid_x: list[float] = []
    mid_y: list[float] = []
    mid_hover: list[str] = []
    mid_label: list[str] = []
    mid_colors: list[str] = []

    for from_node, to_node, key in graph.edges(keys=True):
        int_branch = graph.edges[from_node, to_node, key]["internal_branch"]
        is_cp = int_branch.model.is_cp()

        if is_cp:
            color = _ACCENT["cp"]
        else:
            int_node_from = graph.nodes[from_node]["internal_node"]
            gtype = (
                _grid_type(int_node_from.grid) if int_node_from.independent else "cp"
            )
            color = _ACCENT.get(gtype, _TL_GRAY)

        x0, y0 = pos[from_node]
        x1, y1 = pos[to_node]
        color_groups.setdefault((color, is_cp), []).append((x0, y0, x1, y1))

        mid_x.append((x0 + x1) / 2)
        mid_y.append((y0 + y1) / 2)
        mid_hover.append(_branch_hover(int_branch, from_node, to_node))
        mid_label.append(_branch_label(int_branch))
        mid_colors.append(color)

    edge_traces: list = []
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
                opacity=0.55,
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
            opacity=0.85,
            line=dict(width=1.5, color=_BG),
        ),
    )

    # -----------------------------------------------------------------------
    # Legend
    # -----------------------------------------------------------------------
    legend_entries: list = []
    for gtype, label in _GRID_LABEL.items():
        legend_entries.append(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(
                    size=11,
                    color=_ACCENT[gtype],
                    symbol=_GRID_SYMBOL[gtype],
                    line=dict(width=2, color=_ACCENT[gtype]),
                ),
                name=label,
            )
        )
    legend_entries.append(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color=_ACCENT["cp"], width=2, dash="dot"),
            name="Coupling branch (CP)",
        )
    )

    # -----------------------------------------------------------------------
    # Assemble – render order: edges → midpoints → glow → markers → legend
    # -----------------------------------------------------------------------
    all_traces = (
        edge_traces + [midpoint_trace] + glow_traces + marker_traces + legend_entries
    )

    fig = go.Figure(
        data=all_traces,
        layout=go.Layout(
            title=dict(
                text=title or "Network",
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
                scaleanchor="x",
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

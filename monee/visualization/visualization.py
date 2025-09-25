import networkx as nx
import networkx.drawing.nx_agraph as nxd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

import monee.model as mm

pio.templates["publish"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family="sans-serif", size=19),
    )
)
pio.templates["publish3"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family="sans-serif", size=19),
    )
)
pio.templates["publish2"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family="sans-serif", size=13),
    )
)
pio.templates["publish1"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family="sans-serif", size=9),
    )
)

GRID_NAME_TO_SHIFT_X = {
    "power": 0,
    "water": 0.0003,
    "gas": 0.0006,
    "None": 0.0003,
    None: 0.0003,
}
GRID_NAME_TO_SHIFT_Y = {
    "power": 0,
    "water": 0.0003,
    "gas": 0.0006,
    "None": -0.0003,
    None: -0.0003,
}


def _retr_grid_name(grid):
    if isinstance(grid, list):
        return "None"
    return grid.name


def plot_network(
    network: mm.Network,
    color_df=None,
    color_name=None,
    color_legend_text=None,
    title=None,
    plot_node_characteristics=True,
    template="plotly_white+publish",
    without_nodes=False,
    use_monee_positions=False,
    write_to="net.pdf",
):
    graph: nx.Graph = network._network_internal
    pos = {}
    if not use_monee_positions:
        pos = nxd.pygraphviz_layout(graph, prog="fdp")
    x_edges = []
    y_edges = []
    color_edges = []
    cp_edge = []
    for from_node, to_node, uid in graph.edges:
        from_m_node = network.node_by_id(from_node)
        to_m_node = network.node_by_id(to_node)
        if use_monee_positions:
            add_to_from_x = GRID_NAME_TO_SHIFT_X[_retr_grid_name(from_m_node.grid)]
            add_to_from_y = GRID_NAME_TO_SHIFT_Y[_retr_grid_name(from_m_node.grid)]
            add_to_to_x = GRID_NAME_TO_SHIFT_X[_retr_grid_name(to_m_node.grid)]
            add_to_to_y = GRID_NAME_TO_SHIFT_Y[_retr_grid_name(to_m_node.grid)]
            x0, y0 = (
                from_m_node.position[0] + add_to_from_x,
                from_m_node.position[1] + add_to_from_y,
            )
            x1, y1 = (
                to_m_node.position[0] + add_to_to_x,
                to_m_node.position[1] + add_to_to_y,
            )
            pos[from_node] = (x0, y0)
            pos[to_node] = (x1, y1)
        else:
            x0, y0 = pos[from_node]
            x1, y1 = pos[to_node]
        if color_df is not None:
            color_data = 0
            color_data_list = list(
                color_df.loc[
                    color_df["id"] == f"branch:({from_node}, {to_node}, {uid})"
                ][color_name]
            )
            if len(color_data_list) > 0:
                color_data = color_data_list[0]
            color_edges.append(color_data)
        x_edges.append([x0, x1, None])
        y_edges.append([y0, y1, None])
        branch = network.get_branch_between(from_node, to_node)
        cp_edge.append(isinstance(branch.model, mm.MultiGridBranchModel))
    node_x_power = []
    node_y_power = []
    node_color_power = []
    node_text_power = []
    node_x_heat = []
    node_y_heat = []
    node_color_heat = []
    node_text_heat = []
    node_x_gas = []
    node_y_gas = []
    node_color_gas = []
    node_text_gas = []
    node_cp_x = []
    node_cp_y = []
    node_color_cp = []
    node_text_cp = []
    for node in graph.nodes:
        node_id = f"node:{node}"
        x, y = pos[node]
        node_data = graph.nodes[node]
        int_node = node_data["internal_node"]
        color_data = 0
        if color_df is not None:
            color_data_list = list(color_df.loc[color_df["id"] == node_id][color_name])
            if len(color_data_list) > 0:
                color_data = color_data_list[0]

        if not int_node.independent:
            node_cp_x.append(x)
            node_cp_y.append(y)
            node_color_cp.append(color_data)
            node_text = ""
            node_text_cp.append(node_text)
        elif "Water" in str(type(int_node.grid)):
            node_x_heat.append(x)
            node_y_heat.append(y)
            node_color_heat.append(color_data)
            node_text = ""
            if plot_node_characteristics:
                node_text = f"<b>{round(mm.value(int_node.model.t_k), 2)}</b>"
            node_text_heat.append(node_text)
        elif "Gas" in str(type(int_node.grid)):
            node_x_gas.append(x)
            node_y_gas.append(y)
            node_color_gas.append(color_data)
            node_text = ""
            if plot_node_characteristics:
                node_text = f"<b>{round(mm.value(int_node.model.mass_flow), 2)}</b>"
            node_text_gas.append(node_text)
        elif "Power" in str(type(int_node.grid)):
            node_x_power.append(x)
            node_y_power.append(y)
            node_color_power.append(color_data)
            node_text = ""
            if plot_node_characteristics:
                node_text = f"<b>{round(mm.value(int_node.model.p_mw), 2)}</b>"
            node_text_power.append(node_text)

    max_color_val = max(
        color_edges
        if without_nodes
        else node_color_gas
        + node_color_cp
        + node_color_heat
        + node_color_power
        + color_edges
    )
    edge_traces = []
    coloraxis_v = None if color_df is None else "coloraxis"
    for i in range(len(x_edges)):
        edge_text = "" if color_df is None else f"{color_edges[i]}"
        edge_traces.append(
            go.Scatter(
                x=x_edges[i],
                y=y_edges[i],
                line=dict(
                    dash="dash" if cp_edge[i] else "solid",
                    width=3,
                    color="rgb(0,0,0)"
                    if color_df is None
                    else px.colors.sample_colorscale(
                        px.colors.sequential.Sunsetdark,
                        (color_edges[i] / max_color_val) + min(color_edges),
                    )[0],
                ),
                hoverinfo="text",
                mode="lines",
                text=edge_text,
                showlegend=False,
                marker=dict(
                    coloraxis=coloraxis_v,
                ),
            )
        )
    edge_traces.append(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color="black", width=3, dash="solid"),
            name="Branches",
            showlegend=True,
        )
    )
    edge_traces.append(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color="black", width=3, dash="dash"),
            name="CP Branches",
            showlegend=True,
        )
    )

    # cp
    node_trace_cp = go.Scatter(
        x=node_cp_x,
        y=node_cp_y,
        mode="markers+text",
        textposition="top center",
        hoverinfo="text",
        text=node_text_cp,
        legendgroup="points",
        showlegend=True,
        name="CP",
        marker=dict(
            color=node_color_cp,
            symbol="diamond",
            size=12,
            coloraxis=coloraxis_v,
            line=dict(width=3, color="#616161"),
        ),
    )

    # heat
    node_trace_heat = go.Scatter(
        x=node_x_heat,
        y=node_y_heat,
        mode="markers+text",
        textposition="top center",
        hoverinfo="text",
        text=node_text_heat,
        legendgroup="points",
        showlegend=True,
        name="Heat",
        marker=dict(
            color=node_color_heat,
            symbol="pentagon",
            size=12,
            coloraxis=coloraxis_v,
            line=dict(width=3, color="#d62728"),
        ),
    )
    # power
    node_trace_power = go.Scatter(
        x=node_x_power,
        y=node_y_power,
        mode="markers+text",
        textposition="top center",
        hoverinfo="text",
        text=node_text_power,
        legendgroup="points",
        showlegend=True,
        name="Electricity",
        marker=dict(
            color=node_color_power,
            symbol="square",
            size=12,
            coloraxis=coloraxis_v,
            line=dict(width=3, color="#2ca02c"),
        ),
    )
    # gas
    node_trace_gas = go.Scatter(
        x=node_x_gas,
        y=node_y_gas,
        mode="markers+text",
        textposition="top center",
        hoverinfo="text",
        text=node_text_gas,
        legendgroup="points",
        showlegend=True,
        name="Gas",
        marker=dict(
            color=node_color_gas,
            symbol="triangle-up",
            size=12,
            coloraxis=coloraxis_v,
            line=dict(width=3, color="#1f77b4"),
        ),
    )

    fig = go.Figure(
        data=edge_traces
        + (
            [
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(
                        coloraxis=coloraxis_v,
                        showscale=True,
                    ),
                    hoverinfo="none",
                )
            ]
            if without_nodes
            else [
                node_trace_heat,
                node_trace_power,
                node_trace_gas,
                node_trace_cp,
            ]
        ),
        layout=go.Layout(
            title=title,
            hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            template=template,
        ),
    )
    fig.update_layout(
        height=400,
        width=600,
        margin={"l": 20, "b": 30, "r": 10, "t": 30},
        xaxis_title="",
        legend={"title": color_legend_text},
        yaxis_title="",
        title=title,
        coloraxis_colorbar=dict(
            title=color_legend_text,
        ),
    )
    if color_df is not None:
        fig.layout.coloraxis.showscale = True
        fig.layout.coloraxis.colorscale = "Sunsetdark"
        fig.layout.coloraxis.reversescale = False
        fig.layout.coloraxis.colorbar.thickness = 15
        fig.layout.coloraxis.colorbar.xanchor = "left"
        fig.layout.coloraxis.colorbar.outlinewidth = 2
        fig.layout.coloraxis.colorbar.outlinecolor = "#888"
        fig.layout.coloraxis.cmin = min(
            node_color_gas
            + node_color_cp
            + node_color_heat
            + node_color_power
            + color_edges
        )
        fig.layout.coloraxis.cmax = max_color_val
    if write_to is not None:
        fig.write_image(write_to)
    return fig

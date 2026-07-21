import math
from types import SimpleNamespace

import numpy as np
import pandas as pd

from monee.visualization.result_visualization import (
    _TL_GRAY,
    _TL_GREEN,
    _TL_RED,
    _TL_YELLOW,
    _branch_label_and_color,
    _branch_result_map,
    _line_color,
    _node_result_map,
)
from monee.visualization.visualization import _adaptive_marker_px, _model_params


def test_line_color_thresholds_in_percent():
    assert _line_color(50) == _TL_GREEN
    assert _line_color(69.9) == _TL_GREEN
    assert _line_color(70) == _TL_YELLOW
    assert _line_color(89.9) == _TL_YELLOW
    assert _line_color(90) == _TL_RED
    assert _line_color(150) == _TL_RED


def test_line_color_invalid_values_are_gray():
    assert _line_color(math.nan) == _TL_GRAY
    assert _line_color(None) == _TL_GRAY
    assert _line_color("not-a-number") == _TL_GRAY


def test_branch_label_and_color_converts_pu_to_percent():
    label, color = _branch_label_and_color({"loading_from_pu": 0.5})
    assert label == "50%"
    assert color == _TL_GREEN

    label, color = _branch_label_and_color({"loading_from_pu": 0.75})
    assert label == "75%"
    assert color == _TL_YELLOW

    label, color = _branch_label_and_color({"loading_from_pu": 0.95})
    assert label == "95%"
    assert color == _TL_RED


def test_branch_label_and_color_loading_to_pu_fallback():
    label, color = _branch_label_and_color({"loading_to_pu": 1.0})
    assert label == "100%"
    assert color == _TL_RED


def test_branch_label_and_color_nan_loading_falls_through():
    label, color = _branch_label_and_color(
        {"loading_from_pu": math.nan, "mass_flow_kgs": 2.5}
    )
    assert label == "2.5 kg/s"
    assert color == _TL_GREEN


def test_branch_label_and_color_empty_row():
    assert _branch_label_and_color({}) == ("", _TL_GRAY)
    _, color = _branch_label_and_color({}, is_cp=True)
    assert color != _TL_GRAY


def test_node_and_branch_result_maps_split_mixed_frames():
    df = pd.DataFrame(
        {
            "id": [0, (0, 1, 0), 1],
            "vm_pu": [1.0, math.nan, 0.99],
        }
    )
    result = SimpleNamespace(dataframes={"Mixed": df})

    node_map = _node_result_map(result)
    assert set(node_map) == {0, 1}

    branch_map = _branch_result_map(result)
    assert (0, 1, 0) in branch_map
    assert (1, 0, 0) in branch_map  # reversed alias


def test_model_params_accepts_numpy_scalars():
    model = SimpleNamespace(
        a=np.int64(3),
        b=np.int32(4),
        c=np.float32(1.5),
        d=2.0,
        e="x",
        _private=1,
    )
    params = _model_params(model)
    assert params == {"a": 3, "b": 4, "c": np.float32(1.5), "d": 2.0, "e": "x"}


def test_adaptive_marker_px_single_edge_not_clamped_to_floor():
    import networkx as nx

    graph = nx.Graph()
    graph.add_edge(0, 1)
    pos = {0: (0.0, 0.0), 1: (1.0, 1.0)}
    px = _adaptive_marker_px(graph, pos)
    assert 8.0 <= px <= 20.0
    assert px == 20.0  # single long edge should hit the upper cap, not the floor

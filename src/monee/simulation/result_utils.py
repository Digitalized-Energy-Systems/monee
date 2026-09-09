"""Shared rendering and result-query helpers for simulation result objects.

Both :class:`~monee.simulation.multi_period.MultiPeriodResult` and
:class:`~monee.simulation.timeseries.TimeseriesResult` expose the same kind of
per-type statistics tables and attribute/component queries; the helpers here
are parameterized by an iterable of ``(label, {type_name: DataFrame})`` frames
and an index factory so each class keeps its own index semantics.
"""

from __future__ import annotations

import pandas

from monee.solver.core import TABLE_CSS as _TABLE_CSS
from monee.solver.core import display_df as _display_df


def build_attribute_frame(frames, model_type, attribute: str, make_index):
    """DataFrame of *attribute* values: rows=frames, cols=component ids.
    Raises ``KeyError`` for an unknown model type or attribute."""
    rows = []
    labels = []
    for label, dfs in frames:
        raw_df = dfs[model_type.__name__]
        # Column labels are component ids so callers can do df[bus_id].
        if "id" in raw_df.columns:
            rows.append(dict(zip(raw_df["id"], raw_df[attribute])))
        else:
            rows.append(raw_df[attribute].to_dict())
        labels.append(label)
    return pandas.DataFrame(rows, index=make_index(labels))


#: Level names given to the ``(from_node_id, to_node_id, key)`` branch id when
#: :func:`id_indexed` turns a branch result frame into a MultiIndex.
BRANCH_ID_LEVELS = ("from_node_id", "to_node_id", "key")


def id_indexed(df: pandas.DataFrame) -> pandas.DataFrame:
    """Copy of *df* re-indexed by its ``id`` column, columns untouched.

    Node and child frames get a plain ``Index`` named ``id``, so a row is
    reached with ``df.loc[node_id]`` and a map such as ``net.pp_bus_to_node``
    joins directly. Branch ids are ``(from_node_id, to_node_id, key)`` tuples,
    which a plain index cannot serve (``df.loc[(4, 7, 0)]`` is read by pandas as
    a multi-axis key and raises ``IndexingError``), so a frame whose ids are all
    equal-length tuples gets a ``MultiIndex`` with those levels named and
    ``df.loc[(4, 7, 0)]`` returns the row.

    The ``id`` column is kept, so dropping the index reproduces the input frame.
    A frame without an ``id`` column (e.g. the empty frame for a model type that
    is not in the result) is returned unchanged.
    """
    if "id" not in df.columns:
        return df.copy()
    ids = list(df["id"])
    widths = {len(i) for i in ids if isinstance(i, tuple)}
    if ids and len(widths) == 1 and all(isinstance(i, tuple) for i in ids):
        width = widths.pop()
        names = BRANCH_ID_LEVELS if width == len(BRANCH_ID_LEVELS) else None
        return df.set_index(pandas.MultiIndex.from_tuples(ids, names=names))
    return df.set_index(pandas.Index(ids, name="id"))


def id_mask(df, component_id):
    try:
        return df["id"] == component_id
    except (ValueError, TypeError):
        # Tuple branch ids vs a scalar id column trigger a broadcasting error.
        return df["id"].apply(lambda x: x == component_id)


def split_id_key(key):
    """Split an accessor key into ``(component_id, model_type)``; a bare id
    yields ``(key, None)``. ``result[id, SomeModel]`` restricts the lookup to
    one result table, which is how a caller resolves an id shared by several
    categories."""
    if isinstance(key, tuple) and len(key) == 2 and isinstance(key[1], type):
        return key
    return key, None


def match_frames(dfs: dict, component_id, attribute: str = None, model_type=None):
    """``[(type_name, row)]`` for every result table of *dfs* holding
    *component_id* (and *attribute*), restricted to *model_type* if given."""
    wanted = None if model_type is None else model_type.__name__
    matches = []
    for type_name, df in dfs.items():
        if wanted is not None and type_name != wanted:
            continue
        if "id" not in df.columns:
            continue
        if attribute is not None and attribute not in df.columns:
            continue
        rows = df[id_mask(df, component_id)]
        if not rows.empty:
            matches.append((type_name, rows.iloc[0]))
    return matches


def ambiguous_id_error(component_id, type_names, attribute: str = None):
    """Component ids are handed out per category (node, child, branch,
    compound), so the same number addresses one component of each."""
    what = f" carrying {attribute!r}" if attribute else ""
    pick = type_names[-1]
    if attribute is None:
        example = f"result[{component_id!r}, mm.{pick}]"
    else:
        example = (
            f"result.get_result_for_id({component_id!r}, {attribute!r}, mm.{pick})"
        )
    return ValueError(
        f"Component id {component_id!r} matches {len(type_names)} result "
        f"tables{what} ({', '.join(type_names)}); component ids are unique "
        "within a category only, so one node, one child and one branch can "
        "share the same number. Select one of them with model_type=<class>, "
        f"for example {example}. See the concepts/data_model docs page."
    )


def find_attr_value(dfs: dict, component_id, attribute: str, model_type=None):
    matches = match_frames(dfs, component_id, attribute, model_type)
    if len(matches) > 1:
        raise ambiguous_id_error(component_id, [name for name, _ in matches], attribute)
    if not matches:
        return False, None
    return True, matches[0][1][attribute]


def build_id_series(frames, component_id, attribute: str, make_index, model_type=None):
    """Series of *attribute* for *component_id*, ``None`` where absent."""
    values = []
    labels = []
    for label, dfs in frames:
        found, value = find_attr_value(dfs, component_id, attribute, model_type)
        values.append(value if found else None)
        labels.append(label)
    return pandas.Series(values, index=make_index(labels), name=attribute)


def build_component_frame(frames, component_id, make_index, model_type=None):
    """All result attributes for *component_id*, one row per frame where it
    appears. Raises ``KeyError`` when the component is never found and
    ``ValueError`` when the id matches more than one component type."""
    rows: list[dict] = []
    labels: list = []
    for label, dfs in frames:
        matches = match_frames(dfs, component_id, model_type=model_type)
        if len(matches) > 1:
            raise ambiguous_id_error(component_id, [name for name, _ in matches])
        if not matches:
            continue
        row = _display_df(matches[0][1].to_frame().T).iloc[0]
        rows.append({k: v for k, v in row.items() if k != "id"})
        labels.append(label)
    if not rows:
        raise KeyError(component_id)
    return pandas.DataFrame(rows, index=make_index(labels))


def wrap_result_html(name: str, subtitle_html: str, sections: list[str]) -> str:
    header = (
        f"<div style='font-weight:bold;font-size:1.05em;padding:4px 0 8px'>"
        f"{name} &nbsp;"
        f"{subtitle_html}</div>"
    )
    return (
        f"{_TABLE_CSS}"
        f"<div class='monee-result'>"
        f"{header}" + "\n".join(sections) + "</div>"
    )


def build_type_stats_html(
    type_dfs: dict[str, list[pandas.DataFrame]],
    unit: str,
) -> list[str]:
    """Return one ``<details>`` HTML section per component type.

    *type_dfs* maps a component-type name to the list of per-aggregation-unit
    DataFrames for that type (one DataFrame per period or per step). *unit* is
    the singular aggregation-unit word ("period" or "step") used in the
    "aggregated over N unit(s)" caption; ``len(dfs)`` supplies the count.
    """
    sections = []
    for type_name, dfs in type_dfs.items():
        n_comp = len(dfs[0])
        plural = "instance" if n_comp == 1 else "instances"
        combined = pandas.concat(dfs, ignore_index=True)
        vis = _display_df(combined).drop(columns=["id", "node_id"], errors="ignore")
        num_cols = vis.select_dtypes(include="number").columns.tolist()
        stat_rows = []
        for col in num_cols:
            vals = combined[col].dropna()
            if vals.empty:
                continue
            stat_rows.append(
                {
                    "attribute": col,
                    "min": f"{float(vals.min()):.4g}",
                    "mean": f"{float(vals.mean()):.4g}",
                    "max": f"{float(vals.max()):.4g}",
                }
            )
        if stat_rows:
            tbl = pandas.DataFrame(stat_rows).to_html(index=False, border=0, classes=[])
        else:
            tbl = "<em style='color:#888'>(no numeric attributes)</em>"
        sections.append(
            f"<details open style='margin-bottom:6px'>"
            f"<summary style='cursor:pointer;font-weight:bold;color:#333;"
            f"padding:2px 0'>{type_name} "
            f"<span style='color:#999;font-weight:normal'>({n_comp} {plural})</span>"
            f"</summary>"
            f"<div style='color:#888;font-size:.82em;padding:1px 0 3px'>"
            f"aggregated over {len(dfs)} {unit}{'s' if len(dfs) != 1 else ''}"
            f"</div>{tbl}</details>"
        )
    return sections

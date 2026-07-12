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


def find_attr_value(dataframes, component_id, attribute: str):
    for df in dataframes:
        if "id" in df.columns and attribute in df.columns:
            try:
                mask = df["id"] == component_id
            except (ValueError, TypeError):
                mask = df["id"].apply(lambda x: x == component_id)
            row = df[mask]
            if not row.empty:
                return True, row.iloc[0][attribute]
    return False, None


def build_id_series(frames, component_id, attribute: str, make_index):
    """Series of *attribute* for *component_id*, ``None`` where absent."""
    values = []
    labels = []
    for label, dfs in frames:
        found, value = find_attr_value(dfs.values(), component_id, attribute)
        values.append(value if found else None)
        labels.append(label)
    return pandas.Series(values, index=make_index(labels), name=attribute)


def build_component_frame(frames, component_id, make_index):
    """All result attributes for *component_id*, one row per frame where it
    appears. Raises ``KeyError`` when the component is never found."""
    rows: list[dict] = []
    labels: list = []
    for label, dfs in frames:
        for df in dfs.values():
            if "id" not in df.columns:
                continue
            mask = df["id"] == component_id
            if not mask.any():
                continue
            row = _display_df(df[mask].iloc[0].to_frame().T).iloc[0]
            rows.append({k: v for k, v in row.items() if k != "id"})
            labels.append(label)
            break
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

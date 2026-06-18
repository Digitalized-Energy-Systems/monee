"""Shared rendering helpers for simulation result objects.

Both :class:`~monee.simulation.multi_period.MultiPeriodResult` and
:class:`~monee.simulation.timeseries.TimeseriesResult` render the same kind of
per-type statistics tables in their ``_repr_html_``; the only difference is
whether the aggregation unit is a "period" or a "step".
"""

from __future__ import annotations

import pandas

from monee.solver.core import display_df as _display_df


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

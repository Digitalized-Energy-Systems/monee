import logging
from abc import ABC
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas

from monee.model import Network
from monee.model.core import Var, tracked
from monee.model.core import value as _model_value
from monee.simulation.core import solve
from monee.simulation.step_state import StepState
from monee.solver.core import _TABLE_CSS, _col_summary, _display_df

_log = logging.getLogger(__name__)


class TimeseriesData:
    """
    Holds time-varying attribute values for network components.

    Series are registered by component type and lookup key (id or name), then
    applied to the corresponding model attributes before each solve step.

    All series must have the same length.  Mismatched lengths raise
    ``ValueError`` at registration time so errors are caught before the run.
    """

    def __init__(self):
        self._node_id_to_series: dict[Any, dict[str, list]] = {}
        self._child_id_to_series: dict[Any, dict[str, list]] = {}
        self._child_name_to_series: dict[str, dict[str, list]] = {}
        self._branch_id_to_series: dict[Any, dict[str, list]] = {}
        self._branch_name_to_series: dict[str, dict[str, list]] = {}
        self._compound_id_to_series: dict[Any, dict[str, list]] = {}
        self._compound_name_to_series: dict[str, dict[str, list]] = {}
        self._length: int | None = None

    def _validate_length(self, series) -> None:
        n = len(series)
        if self._length is None:
            self._length = n
        elif n != self._length:
            raise ValueError(
                f"Series length {n} does not match existing length {self._length}. "
                "All series must have the same length."
            )

    def _add_to(self, target_dict, key_one, key_two, series) -> None:
        self._validate_length(series)
        if key_one not in target_dict:
            target_dict[key_one] = {}
        target_dict[key_one][key_two] = series

    def add_node_series(self, node_id, attribute: str, series) -> None:
        """Register a time-varying attribute for a node model (by id)."""
        self._add_to(self._node_id_to_series, node_id, attribute, series)

    def add_child_series(self, child_id, attribute: str, series) -> None:
        """Register a time-varying attribute for a child model (by id)."""
        self._add_to(self._child_id_to_series, child_id, attribute, series)

    def add_child_series_by_name(self, child_name: str, attribute: str, series) -> None:
        """Register a time-varying attribute for a child model (by name)."""
        self._add_to(self._child_name_to_series, child_name, attribute, series)

    def add_branch_series(self, branch_id, attribute: str, series) -> None:
        """Register a time-varying attribute for a branch model (by id)."""
        self._add_to(self._branch_id_to_series, branch_id, attribute, series)

    def add_branch_series_by_name(
        self, branch_name: str, attribute: str, series
    ) -> None:
        """Register a time-varying attribute for a branch model (by name)."""
        self._add_to(self._branch_name_to_series, branch_name, attribute, series)

    def add_compound_series(self, compound_id, attribute: str, series) -> None:
        """Register a time-varying attribute for a compound model (by id)."""
        self._add_to(self._compound_id_to_series, compound_id, attribute, series)

    def add_compound_series_by_name(
        self, compound_name: str, attribute: str, series
    ) -> None:
        """Register a time-varying attribute for a compound model (by name)."""
        self._add_to(self._compound_name_to_series, compound_name, attribute, series)

    @classmethod
    def from_dataframe(
        cls,
        df: pandas.DataFrame,
        component_type: str,
        component_id=None,
        component_name: str = None,
    ) -> "TimeseriesData":
        """
        Build a ``TimeseriesData`` from a pandas DataFrame.

        Each column of *df* is treated as a time-varying attribute.  Rows are
        timesteps.

        Args:
            df: DataFrame where each column is an attribute name.
            component_type: ``'node'``, ``'child'``, ``'branch'``, or
                ``'compound'``.
            component_id: Component identifier (id-based lookup).
            component_name: Component name (name-based lookup; not available
                for nodes).

        Returns:
            A new ``TimeseriesData`` with all columns registered.

        Example::

            df = pandas.DataFrame({'p_mw': [...], 'q_mvar': [...]})
            td = TimeseriesData.from_dataframe(df, 'child', component_id=load_id)
        """
        _by_id = {
            "node": "add_node_series",
            "child": "add_child_series",
            "branch": "add_branch_series",
            "compound": "add_compound_series",
        }
        _by_name = {
            "child": "add_child_series_by_name",
            "branch": "add_branch_series_by_name",
            "compound": "add_compound_series_by_name",
        }
        td = cls()
        if component_id is not None:
            if component_type not in _by_id:
                raise ValueError(f"Unknown component_type: {component_type!r}")
            adder = getattr(td, _by_id[component_type])
            for col in df.columns:
                adder(component_id, col, df[col].tolist())
        elif component_name is not None:
            if component_type not in _by_name:
                raise ValueError(
                    f"component_type {component_type!r} does not support name lookup"
                )
            adder = getattr(td, _by_name[component_type])
            for col in df.columns:
                adder(component_name, col, df[col].tolist())
        else:
            raise ValueError("Either component_id or component_name must be provided")
        return td

    @property
    def length(self) -> int | None:
        """Number of timesteps, inferred from registered series lengths."""
        return self._length

    @staticmethod
    def _set_model_attr(model, attr: str, value) -> None:
        """Set *attr* on *model* to *value*.

        * For plain (non-``Var``) attributes the value is replaced directly.
        * For ordinary ``Var`` instances the value is updated in place so that
          the type and bounds are preserved.
        * For ``tracked`` instances the value *and* both bounds are set to the
          series value, effectively pinning the variable at that setpoint for
          this step while keeping the ``tracked`` type so the solved value is
          still recorded in ``StepState`` for inter-step coupling.
        """
        current = getattr(model, attr, None)
        if type(current) is tracked:
            current.value = value
            current.min = value
            current.max = value
        elif isinstance(current, Var):
            current.value = value
        else:
            setattr(model, attr, value)

    def apply_to_node(self, node, timestep: int) -> None:
        """Apply registered series values for *node* at *timestep*."""
        if node.id in self._node_id_to_series:
            for attr, series in self._node_id_to_series[node.id].items():
                self._set_model_attr(node.model, attr, series[timestep])

    def apply_to_child(self, child, timestep: int) -> None:
        """Apply registered series values for *child* at *timestep*."""
        if child.id in self._child_id_to_series:
            for attr, series in self._child_id_to_series[child.id].items():
                self._set_model_attr(child.model, attr, series[timestep])
        if child.name in self._child_name_to_series:
            for attr, series in self._child_name_to_series[child.name].items():
                self._set_model_attr(child.model, attr, series[timestep])

    def apply_to_branch(self, branch, timestep: int) -> None:
        """Apply registered series values for *branch* at *timestep*."""
        if branch.id in self._branch_id_to_series:
            for attr, series in self._branch_id_to_series[branch.id].items():
                self._set_model_attr(branch.model, attr, series[timestep])
        if branch.name in self._branch_name_to_series:
            for attr, series in self._branch_name_to_series[branch.name].items():
                self._set_model_attr(branch.model, attr, series[timestep])

    def apply_to_compound(self, compound, timestep: int) -> None:
        """Apply registered series values for *compound* at *timestep*."""
        if compound.id in self._compound_id_to_series:
            for attr, series in self._compound_id_to_series[compound.id].items():
                self._set_model_attr(compound.model, attr, series[timestep])
        if compound.name in self._compound_name_to_series:
            for attr, series in self._compound_name_to_series[compound.name].items():
                self._set_model_attr(compound.model, attr, series[timestep])

    def apply_to_network(self, net: Network, timestep: int) -> None:
        """Apply all registered series to *net* at *timestep*."""
        for node in net.nodes:
            self.apply_to_node(node, timestep)
            for child in net.childs_by_ids(node.child_ids):
                self.apply_to_child(child, timestep)
        for branch in net.branches:
            self.apply_to_branch(branch, timestep)
        for compound in net.compounds:
            self.apply_to_compound(compound, timestep)

    @staticmethod
    def _merge_component_data(target: dict, source: dict) -> dict:
        """
        Attribute-level merge of two ``{component_id: {attr: series}}`` dicts.

        For each component id in *source*: if absent from *target*, add it
        wholesale.  If present, merge attribute dicts with *target* winning on
        conflicts (self-wins semantics).
        """
        result = dict(target)
        for comp_id, attrs in source.items():
            if comp_id in result:
                result[comp_id] = {**attrs, **result[comp_id]}
            else:
                result[comp_id] = dict(attrs)
        return result

    def extend(self, td: "TimeseriesData") -> None:
        """
        Merge *td* into this ``TimeseriesData``.

        For components present in both, attributes from *self* take priority
        on conflicts.  Components present only in *td* are added wholesale.

        Raises ``ValueError`` if the two objects have incompatible lengths.
        """
        if (
            td._length is not None
            and self._length is not None
            and td._length != self._length
        ):
            raise ValueError(
                f"Cannot extend: incoming TimeseriesData has length {td._length} "
                f"but this object has length {self._length}."
            )
        self._node_id_to_series = self._merge_component_data(
            self._node_id_to_series, td._node_id_to_series
        )
        self._child_id_to_series = self._merge_component_data(
            self._child_id_to_series, td._child_id_to_series
        )
        self._child_name_to_series = self._merge_component_data(
            self._child_name_to_series, td._child_name_to_series
        )
        self._branch_id_to_series = self._merge_component_data(
            self._branch_id_to_series, td._branch_id_to_series
        )
        self._branch_name_to_series = self._merge_component_data(
            self._branch_name_to_series, td._branch_name_to_series
        )
        self._compound_id_to_series = self._merge_component_data(
            self._compound_id_to_series, td._compound_id_to_series
        )
        self._compound_name_to_series = self._merge_component_data(
            self._compound_name_to_series, td._compound_name_to_series
        )
        if self._length is None:
            self._length = td._length

    def __add__(self, other: "TimeseriesData") -> "TimeseriesData":
        new_td = TimeseriesData()
        new_td.extend(self)
        new_td.extend(other)
        return new_td

    @property
    def child_id_data(self):
        return self._child_id_to_series

    @property
    def child_name_data(self):
        return self._child_name_to_series

    @property
    def branch_id_data(self):
        return self._branch_id_to_series

    @property
    def compound_id_data(self):
        return self._compound_id_to_series


@dataclass
class StepResult:
    """
    Wraps the outcome of a single timeseries step.

    Attributes:
        step: Zero-based step index.
        result: The ``SolverResult`` for this step, or ``None`` if *failed* or
            *skipped*.
        failed: ``True`` if the solve raised an exception.
        skipped: ``True`` if the solve was not attempted (``solve_flag=False``).
        error: The exception that caused the failure, or ``None``.
    """

    step: int
    result: Any
    failed: bool = False
    skipped: bool = False
    error: Exception | None = None


class TimeseriesResult:
    """
    Holds the per-step results of a timeseries simulation run.

    Steps that failed (convergence error, infeasibility) are represented by
    ``StepResult`` entries with ``failed=True`` and ``result=None``.  They
    are excluded from DataFrame queries but accessible via ``failed_steps``.
    """

    def __init__(
        self,
        step_results: list[StepResult],
        datetime_index: pandas.DatetimeIndex | None = None,
    ) -> None:
        self._step_results = step_results
        self._datetime_index = datetime_index
        self._cache: dict[tuple, pandas.DataFrame] = {}

    @property
    def step_results(self) -> list[StepResult]:
        """All ``StepResult`` objects, including failed steps."""
        return self._step_results

    @property
    def raw(self) -> list:
        """
        Successful ``SolverResult`` objects in step order.

        Kept for backward compatibility.  Prefer ``step_results``.
        """
        return [
            sr.result for sr in self._step_results if not sr.failed and not sr.skipped
        ]

    @property
    def failed_steps(self) -> list[int]:
        """List of step indices that failed to converge."""
        return [sr.step for sr in self._step_results if sr.failed]

    def _successful(self) -> list[StepResult]:
        return [sr for sr in self._step_results if not sr.failed and not sr.skipped]

    def _make_index(self, step_indices: list[int]) -> pandas.Index:
        if self._datetime_index is not None:
            return self._datetime_index[step_indices]
        return pandas.RangeIndex(len(step_indices))

    def _create_result_for(self, model_type, attribute: str) -> pandas.DataFrame:
        rows = []
        step_indices = []
        for sr in self._successful():
            raw_df = sr.result.dataframes[model_type.__name__]
            # Use component IDs as column names so callers can do df[bus_id]
            if "id" in raw_df.columns:
                row = dict(zip(raw_df["id"], raw_df[attribute]))
            else:
                row = raw_df[attribute].to_dict()
            rows.append(row)
            step_indices.append(sr.step)
        df = pandas.DataFrame(rows, index=self._make_index(step_indices))
        self._cache[model_type, attribute] = df
        return df

    def get_result_for(self, model_type, attribute: str) -> pandas.DataFrame:
        """Return a DataFrame of *attribute* values across all successful steps.

        One row per step, one column per component — **columns are labelled by
        component id** so you can select a specific instance with
        ``df[bus_id]`` instead of relying on positional indices.

        Args:
            model_type: The model class (e.g. ``mm.PowerLoad``, ``mm.Bus``).
            attribute: The attribute name (e.g. ``'p_mw'``).

        Example::

            vm_df = ts_result.get_result_for(mm.Bus, "vm_pu")
            print(vm_df[bus_home_id])   # time-series for one bus
        """
        if (model_type, attribute) in self._cache:
            return self._cache[model_type, attribute]
        return self._create_result_for(model_type, attribute)

    def __getitem__(self, component_id) -> pandas.DataFrame:
        """Return all result attributes for *component_id* across every step.

        Each column is one result attribute; each row is one successful step
        (indexed by step number or datetime if a ``datetime_index`` was
        provided).  Internal bookkeeping columns (``active``, ``independent``,
        ``ignored``) are excluded.

        Raises :exc:`KeyError` if *component_id* is not found in any step.

        Example::

            df = ts_result[bus_home_id]
            print(df["vm_pu"])          # voltage series
            print(df["va_degree"].min()) # worst angle
        """
        rows: list[dict] = []
        step_indices: list[int] = []
        for sr in self._successful():
            for df in sr.result.dataframes.values():
                if "id" not in df.columns:
                    continue
                mask = df["id"] == component_id
                if not mask.any():
                    continue
                row = _display_df(df[mask].iloc[0].to_frame().T).iloc[0]
                rows.append({k: v for k, v in row.items() if k != "id"})
                step_indices.append(sr.step)
                break
        if not rows:
            raise KeyError(component_id)
        return pandas.DataFrame(rows, index=self._make_index(step_indices))

    def get_result_for_id(self, component_id, attribute: str) -> pandas.Series:
        """
        Return a ``Series`` of *attribute* values for a specific component
        across all successful steps.

        Args:
            component_id: The component's id (as stored in the ``id`` column
                of the result DataFrames).
            attribute: The attribute name to retrieve.

        Returns:
            A ``Series`` indexed by step index (or datetime if a
            ``datetime_index`` was provided to ``run()``).  Failed steps are
            excluded.  A ``None`` entry is emitted for a step where the
            component is absent (e.g. ignored due to islanding).
        """
        values = []
        step_indices = []
        for sr in self._successful():
            found = False
            for df in sr.result.dataframes.values():
                if "id" in df.columns and attribute in df.columns:
                    row = df[df["id"] == component_id]
                    if not row.empty:
                        values.append(row.iloc[0][attribute])
                        found = True
                        break
            if not found:
                values.append(None)
            step_indices.append(sr.step)
        return pandas.Series(
            values, index=self._make_index(step_indices), name=attribute
        )

    def summary(self):
        return repr(self)

    def __repr__(self) -> str:
        n_total = len(self._step_results)
        n_failed = len(self.failed_steps)
        n_skipped = sum(1 for sr in self._step_results if sr.skipped)

        status_parts = [f"{n_total} step{'s' if n_total != 1 else ''}"]
        if n_failed:
            status_parts.append(f"{n_failed} failed")
        if n_skipped:
            status_parts.append(f"{n_skipped} skipped")

        SEP = "─" * 68
        lines = [f"TimeseriesResult  {' · '.join(status_parts)}", SEP]

        # Component-type summary from first successful step
        successful = self._successful()
        if successful:
            for type_name, df in successful[0].result.dataframes.items():
                n_comp = len(df)
                # Aggregate key numeric attrs across all steps for this type
                all_dfs = [
                    sr.result.dataframes[type_name]
                    for sr in successful
                    if type_name in sr.result.dataframes
                ]
                combined = pandas.concat(all_dfs, ignore_index=True)
                vis_num = (
                    _display_df(combined)
                    .drop(columns=["id", "node_id"], errors="ignore")
                    .select_dtypes(include="number")
                )
                parts = []
                for col in vis_num.columns:
                    s = _col_summary(vis_num[col])
                    if s is None:
                        continue
                    parts.append(f"{col} ∈ {s}" if "[" in s else f"{col} = {s}")
                row = f"  {type_name:<22} ×{n_comp:>2}"
                if parts:
                    row += "  │  " + "  ·  ".join(parts[:3])
                lines.append(row)
        else:
            lines.append("  (no successful steps)")

        lines.append(SEP)
        return "\n".join(lines)

    def __str__(self) -> str:
        """Full per-type table dump showing the last successful step's data.

        Printing all N steps inline would be impractical; the last step gives
        a concrete snapshot of the network state.  Use ``get_result_for()`` or
        ``self[component_id]`` to retrieve the full time-series programmatically.
        """
        n_total = len(self._step_results)
        n_failed = len(self.failed_steps)
        successful = self._successful()

        status_parts = [f"{n_total} step{'s' if n_total != 1 else ''}"]
        if n_failed:
            status_parts.append(f"{n_failed} failed")
        title = f"TimeseriesResult  {' · '.join(status_parts)}"

        if not successful:
            return title + "\n  (no successful steps)"

        last = successful[-1]
        SEP = "─" * 68
        lines = [title, f"  [showing step {last.step}]"]
        for type_name, df in last.result.dataframes.items():
            vis = _display_df(df)
            n = len(vis)
            plural = "instance" if n == 1 else "instances"
            lines.append("")
            lines.append(f"  {type_name}  ({n} {plural})")
            lines.append("  " + SEP)
            table = vis.to_string(index=False, float_format=lambda x: f"{x:.4g}")
            for line in table.splitlines():
                lines.append("  " + line)
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        n_total = len(self._step_results)
        n_failed = len(self.failed_steps)
        n_skipped = sum(1 for sr in self._step_results if sr.skipped)
        n_ok = n_total - n_failed - n_skipped

        step_info = f"{n_total} step{'s' if n_total != 1 else ''}"
        extra_parts = []
        if n_ok < n_total:
            extra_parts.append(f"<span style='color:#090'>{n_ok} ok</span>")
        if n_failed:
            extra_parts.append(f"<span style='color:#c00'>{n_failed} failed</span>")
        if n_skipped:
            extra_parts.append(f"<span style='color:#888'>{n_skipped} skipped</span>")
        status_html = " &nbsp;·&nbsp; ".join(extra_parts) if extra_parts else ""

        sections = []
        successful = self._successful()
        if successful:
            # Collect all step dataframes per type
            type_dfs: dict[str, list[pandas.DataFrame]] = {}
            for sr in successful:
                for type_name, df in sr.result.dataframes.items():
                    type_dfs.setdefault(type_name, []).append(df)

            for type_name, dfs in type_dfs.items():
                n_comp = len(dfs[0])
                plural = "instance" if n_comp == 1 else "instances"
                combined = pandas.concat(dfs, ignore_index=True)

                # Build per-attribute stats table aggregated over all steps
                vis = _display_df(combined).drop(
                    columns=["id", "node_id"], errors="ignore"
                )
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
                    stats_df = pandas.DataFrame(stat_rows)
                    tbl = stats_df.to_html(index=False, border=0, classes=[])
                else:
                    tbl = "<em style='color:#888'>(no numeric attributes)</em>"

                sections.append(
                    f"<details open style='margin-bottom:6px'>"
                    f"<summary style='cursor:pointer;font-weight:bold;color:#333;"
                    f"padding:2px 0'>{type_name} "
                    f"<span style='color:#999;font-weight:normal'>"
                    f"({n_comp} {plural})</span></summary>"
                    f"<div style='color:#888;font-size:.82em;padding:1px 0 3px'>"
                    f"aggregated over {len(dfs)} step{'s' if len(dfs) != 1 else ''}"
                    f"</div>{tbl}</details>"
                )

        header = (
            f"<div style='font-weight:bold;font-size:1.05em;padding:4px 0 8px'>"
            f"TimeseriesResult &nbsp;"
            f"<span style='font-weight:normal;color:#555'>{step_info}</span>"
            + (f" &nbsp;·&nbsp; {status_html}" if status_html else "")
            + "</div>"
        )
        return (
            f"{_TABLE_CSS}"
            f"<div class='monee-result'>"
            f"{header}" + "\n".join(sections) + "</div>"
        )


def _attrs_to_track(model) -> list:
    """
    Return the list of attribute names whose solved values should be recorded
    in ``StepState`` for *model*.

    Two protocols are supported (both may coexist):

    * **``tracked`` vars** — attributes that are currently ``tracked``
      instances on the model (restored from ``tracked`` to ``tracked`` during
      solver withdrawal so this check works post-solve).
    * **``inter_step_vars()``** — explicit string-list method, kept for
      backward compatibility.
    """
    attrs = [k for k, v in model.__dict__.items() if type(v) is tracked]
    if hasattr(model, "inter_step_vars"):
        for attr in model.inter_step_vars():
            if attr not in attrs:
                attrs.append(attr)
    return attrs


def _extract_step_state(state: StepState, net: Network) -> None:
    """
    Walk the solved network and record the values of all tracked attributes
    into *state*.

    Called after each timestep's solve + withdraw so that the values stored
    are plain Python floats ready for the next step.
    """
    for node in net.nodes:
        if node.ignored:
            continue
        for attr in _attrs_to_track(node.model):
            v = getattr(node.model, attr, None)
            if v is not None:
                state.set(node.id, attr, _model_value(v))
        for child in net.childs_by_ids(node.child_ids):
            if child.ignored:
                continue
            for attr in _attrs_to_track(child.model):
                v = getattr(child.model, attr, None)
                if v is not None:
                    state.set(child.id, attr, _model_value(v))
    for branch in net.branches:
        if branch.ignored:
            continue
        for attr in _attrs_to_track(branch.model):
            v = getattr(branch.model, attr, None)
            if v is not None:
                state.set(branch.id, attr, _model_value(v))
    for compound in net.compounds:
        if compound.ignored:
            continue
        for attr in _attrs_to_track(compound.model):
            v = getattr(compound.model, attr, None)
            if v is not None:
                state.set(compound.id, attr, _model_value(v))


def apply_to_by_id(component, data: dict, timestep: int) -> None:
    if component.id in data:
        for attr, series in data[component.id].items():
            setattr(component.model, attr, series[timestep])


def apply_to_child(child, timeseries_data: TimeseriesData, timestep: int) -> None:
    timeseries_data.apply_to_child(child, timestep)


def apply_to_branch(branch, timeseries_data: TimeseriesData, timestep: int) -> None:
    timeseries_data.apply_to_branch(branch, timestep)


def apply_to_compound(compound, timeseries_data: TimeseriesData, timestep: int) -> None:
    timeseries_data.apply_to_compound(compound, timestep)


class StepHook(ABC):
    """
    Base class for objects that receive callbacks before and after each
    timeseries step.

    Implement one or both of ``pre_run`` / ``post_run`` — both are optional.
    The base class provides silent no-ops so subclasses override only what they
    need.

    Both callbacks receive:

    - *net*: the current-step network copy (timeseries data already applied).
    - *base_net*: the original unmodified base network.
    - *step*: zero-based step index.
    - *step_state*: ``StepState`` carrying inter-step solved values (readable
      and writable).

    ``post_run`` additionally receives:

    - *step_result*: ``StepResult`` for this step; ``step_result.failed`` is
      ``True`` if the solve failed.
    """

    def pre_run(
        self,
        net: Network,
        base_net: Network,
        step: int,
        step_state: StepState,
    ) -> None:
        """Called before the step's solve.  *net* already has timeseries data applied."""

    def post_run(
        self,
        net: Network,
        base_net: Network,
        step: int,
        step_state: StepState,
        step_result: StepResult,
    ) -> None:
        """Called after the step's solve (whether it succeeded or failed)."""


def run(
    net: Network,
    timeseries_data: TimeseriesData,
    steps: int | None = None,
    step_hooks: list[StepHook | Callable] | None = None,
    solver=None,
    optimization_problem=None,
    solve_flag: bool = True,
    on_step_error: str = "raise",
    progress_callback: Callable[[int, int], None] | None = None,
    datetime_index: pandas.DatetimeIndex | None = None,
) -> TimeseriesResult:
    """
    Run a timeseries simulation over *net*.

    For each step the network is copied, registered timeseries values are
    applied to component models, the network is solved, and results are
    collected.

    Args:
        net: Base network.  Not modified; a fresh copy is made each step.
        timeseries_data: Per-component attribute series.
        steps: Number of steps to simulate.  Defaults to
            ``timeseries_data.length`` when omitted.  Must not exceed the
            series length.
        step_hooks: Hooks called before and after each step.  Items may be
            ``StepHook`` subclasses or plain callables
            ``(net_copy, base_net, step) -> None`` called in the post-step
            position.
        solver: Solver instance.  If ``None``, the default GEKKO solver is
            used.
        optimization_problem: Optional optimization problem passed to the
            solver.
        solve_flag: If ``False``, timeseries data is applied and hooks are
            called but no solve is attempted.  Useful for dry-run testing.
        on_step_error: What to do when a step fails to converge.
            ``'raise'`` (default) — re-raise the exception immediately.
            ``'skip'`` — record the failure and continue to the next step.
        progress_callback: Called after each step as
            ``progress_callback(step, total_steps)``.
        datetime_index: Optional ``pd.DatetimeIndex`` aligned to the steps.
            Used as the row index of result DataFrames.

    Returns:
        A ``TimeseriesResult`` containing per-step outcomes.
    """
    if steps is None:
        steps = timeseries_data.length
        if steps is None:
            raise ValueError(
                "Cannot infer step count: no series registered and 'steps' not provided."
            )
    if timeseries_data.length is not None and steps > timeseries_data.length:
        raise ValueError(
            f"'steps' ({steps}) exceeds the length of the registered series "
            f"({timeseries_data.length}).  Either register longer series or "
            f"reduce 'steps'."
        )
    if step_hooks is None:
        step_hooks = []
    if on_step_error not in ("raise", "skip"):
        raise ValueError(
            f"on_step_error must be 'raise' or 'skip', got {on_step_error!r}"
        )

    step_results: list[StepResult] = []
    step_state = StepState()

    for step in range(steps):
        net_copy = net.copy()
        timeseries_data.apply_to_network(net_copy, step)

        for hook in step_hooks:
            if isinstance(hook, StepHook):
                hook.pre_run(net_copy, net, step, step_state)

        if solve_flag:
            try:
                result = solve(
                    net_copy,
                    optimization_problem=optimization_problem,
                    solver=solver,
                    step_state=step_state,
                )
                _extract_step_state(step_state, result.network)
                sr = StepResult(step=step, result=result)
            except Exception as exc:
                if on_step_error == "raise":
                    raise
                _log.warning("Step %d failed: %s", step, exc)
                sr = StepResult(step=step, result=None, failed=True, error=exc)
        else:
            sr = StepResult(step=step, result=None, skipped=True)

        step_results.append(sr)

        for hook in step_hooks:
            if isinstance(hook, StepHook):
                hook.post_run(net_copy, net, step, step_state, sr)
            else:
                hook(net_copy, net, step)

        if progress_callback is not None:
            progress_callback(step, steps)

    return TimeseriesResult(step_results, datetime_index=datetime_index)

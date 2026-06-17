import logging
from abc import ABC
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas

from monee.model import Network
from monee.model.core import Var
from monee.simulation.core import solve
from monee.simulation.step_state import StepState

# Shared result-rendering helpers, imported from the solver's public reporting
# surface (the simulation layer renders the same kind of result tables).
from monee.solver.core import TABLE_CSS as _TABLE_CSS
from monee.solver.core import col_summary as _col_summary
from monee.solver.core import display_df as _display_df
from monee.solver.dispatch import resolve_solver

_log = logging.getLogger(__name__)

_STEP_FAILED_MSG = "Step %d failed: %s"


class TimeseriesData:
    """Time-varying attribute values applied to model objects before each step.
    All registered series must share a length (validated on add)."""

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

    def add_objective_data(self, child_id, attribute: str, series) -> None:
        """Alias for :meth:`add_child_series`; signals the attribute feeds an objective."""
        self.add_child_series(child_id, attribute, series)

    def add_objective_data_by_name(
        self, child_name: str, attribute: str, series
    ) -> None:
        """Like :meth:`add_objective_data` but matches by component name."""
        self.add_child_series_by_name(child_name, attribute, series)

    @classmethod
    def from_dataframe(
        cls,
        df: pandas.DataFrame,
        component_type: str,
        component_id=None,
        component_name: str = None,
    ) -> "TimeseriesData":
        """Build from a DataFrame (cols=attrs, rows=timesteps). ``component_type``
        is ``node|child|branch|compound``; pass id or name."""
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
        """Set *attr*; for Vars pin value/min/max so it's fixed but stays a Var
        (still discoverable via StepState)."""
        current = getattr(model, attr, None)
        if isinstance(current, Var):
            current.value = value
            current.min = value
            current.max = value
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
        """Attribute-level merge with target-wins semantics on conflicts."""
        result = dict(target)
        for comp_id, attrs in source.items():
            if comp_id in result:
                result[comp_id] = {**attrs, **result[comp_id]}
            else:
                result[comp_id] = dict(attrs)
        return result

    def extend(self, td: "TimeseriesData") -> None:
        """Merge *td*; self wins on attribute conflicts. Raises on length mismatch."""
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
    """Outcome of a single timeseries step."""

    step: int
    result: Any
    failed: bool = False
    skipped: bool = False
    error: Exception | None = None


class TimeseriesResult:
    """Per-step results. Failed steps are excluded from DataFrame queries but
    accessible via :attr:`failed_steps`."""

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
        return self._step_results

    @property
    def raw(self) -> list:
        """Successful ``SolverResult`` objects in step order (legacy; prefer ``step_results``)."""
        return [
            sr.result for sr in self._step_results if not sr.failed and not sr.skipped
        ]

    @property
    def failed_steps(self) -> list[int]:
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
            # Column labels are component ids so callers can do df[bus_id].
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
        """DataFrame of *attribute* values: rows=steps, cols=component ids."""
        if (model_type, attribute) in self._cache:
            return self._cache[model_type, attribute]
        return self._create_result_for(model_type, attribute)

    def __getitem__(self, component_id) -> pandas.DataFrame:
        """All result attributes for *component_id* across every successful step."""
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

    @staticmethod
    def _find_attr_value(dataframes, component_id, attribute: str):
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

    def get_result_for_id(self, component_id, attribute: str) -> pandas.Series:
        """Series of *attribute* for *component_id* across successful steps.
        Yields ``None`` where the component is absent (e.g. islanded out)."""
        values = []
        step_indices = []
        for sr in self._successful():
            found, value = self._find_attr_value(
                sr.result.dataframes.values(), component_id, attribute
            )
            values.append(value if found else None)
            step_indices.append(sr.step)
        return pandas.Series(
            values, index=self._make_index(step_indices), name=attribute
        )

    def summary(self):
        return repr(self)

    def __repr__(self) -> str:  # NOSONAR
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
        """Per-type tables from the last successful step (full TS via ``get_result_for``)."""
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

    def _repr_html_(self) -> str:  # NOSONAR
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


class StepHook(ABC):
    """Pre/post-step callbacks for timeseries runs. Both methods are optional no-ops."""

    def pre_run(
        self,
        base_net: Network,
        step: int,
        step_state: StepState,
    ) -> None:
        """Called before the per-step network copy and timeseries application."""

    def post_run(
        self,
        net: Network,
        step: int,
        step_state: StepState,
        step_result: StepResult,
        base_net: Network,
    ) -> None:
        """Called after the step's solve (success or failure)."""


def _is_casadi_solver(solver) -> bool:
    """True if *solver* is a CasADi backend instance (without hard-requiring the
    optional casadi package: if casadi is absent the solver cannot be one)."""
    try:
        from monee.solver.casadi import CasADiSolver
    except ImportError:
        return False
    return isinstance(solver, CasADiSolver)


_TEMPORAL_METHODS = (
    "inter_step_equations",
    "inter_temporal_equations",
    "inter_period_equations",
)


def _network_has_temporal_coupling(net: Network) -> bool:  # NOSONAR
    """True if any component model/formulation or extension contributes
    inter-step temporal coupling (storage SOC, ramp limits, linepack, LTC).

    Models and formulations only define the temporal methods when they actually
    couple across steps (the base classes do not), so ``hasattr`` is a reliable
    signal there. Extensions declare no-op temporal methods on their base, so we
    instead check whether the method is *overridden*.
    """
    from monee.model.extension.core import NetworkAspect

    def _comp_temporal(comp) -> bool:
        if any(hasattr(comp.model, m) for m in _TEMPORAL_METHODS):
            return True
        formulation = getattr(comp, "formulation", None)
        return formulation is not None and any(
            hasattr(formulation, m) for m in _TEMPORAL_METHODS
        )

    for node in net.nodes:
        if _comp_temporal(node):
            return True
        for child in net.childs_by_ids(node.child_ids):
            if _comp_temporal(child):
                return True
    for branch in net.branches:
        if _comp_temporal(branch):
            return True
    for compound in net.compounds:
        if _comp_temporal(compound):
            return True
    for ext in net.extensions:
        if any(
            getattr(type(ext), m, None) is not getattr(NetworkAspect, m, None)
            for m in _TEMPORAL_METHODS
        ):
            return True
    return False


def _casadi_reuse_eligible(
    net, solver, optimization_problem, step_hooks, timeseries_data, solver_kwargs
) -> bool:
    """Whether the build-once / re-solve CasADi graph-reuse path can replace the
    per-step rebuild loop and produce identical results.

    The parametric reuse driver only handles memory-less, id-addressed series on
    a plain power flow, so it is gated to exactly those cases; anything else
    (optimisation problem, step hooks observing temporal state, temporal coupling
    in the network, name-addressed or compound series, or solver kwargs it cannot
    honour) falls back to the standard loop.
    """
    if not _is_casadi_solver(solver):
        return False
    if optimization_problem is not None or step_hooks:
        return False
    # The reuse driver carries no inter-step state, so any temporal coupling
    # (storage/linepack/LTC) must go through the per-step loop instead.
    if _network_has_temporal_coupling(net):
        return False
    # CasADiTimeseries declares only id-addressed child/node/branch series as
    # parameters; name-addressed and compound series are not wired.
    td = timeseries_data
    if (
        td._child_name_to_series
        or td._branch_name_to_series
        or td._compound_id_to_series
        or td._compound_name_to_series
    ):
        return False
    # Only kwargs the reuse driver understands may be present.
    if set(solver_kwargs) - {"simulation", "formulation"}:
        return False
    return True


def _run_casadi_reuse(
    net,
    timeseries_data,
    steps,
    on_step_error,
    progress_callback,
    datetime_index,
    solver_kwargs,
) -> "TimeseriesResult":
    """Build the CasADi NLP + IPOPT solver once and re-solve each step with a
    warm start (no per-step rebuild/recompile). Equivalent results to the
    standard loop for the gated cases (see :func:`_casadi_reuse_eligible`)."""
    from monee.solver.casadi import CasADiTimeseries

    ts = CasADiTimeseries(
        net,
        timeseries_data,
        formulation=solver_kwargs.get("formulation"),
        simulation=solver_kwargs.get("simulation", False),
        steps=steps,
    )
    step_results: list[StepResult] = []
    for step in range(steps):
        try:
            sr = StepResult(step=step, result=ts.step_result(step))
        except Exception as exc:
            if on_step_error == "raise":
                raise
            _log.warning(_STEP_FAILED_MSG, step, exc)
            sr = StepResult(step=step, result=None, failed=True, error=exc)
        step_results.append(sr)
        if progress_callback is not None:
            progress_callback(step, steps)
    return TimeseriesResult(step_results, datetime_index=datetime_index)


def _is_gurobipy_solver(solver) -> bool:
    """True if *solver* is the native gurobipy backend instance (without
    hard-requiring gurobipy: if it's absent the solver cannot be one)."""
    try:
        from monee.solver.gurobipy import GurobipySolver
    except ImportError:
        return False
    return isinstance(solver, GurobipySolver)


def _extension_needs_step0_structure_switch(net) -> bool:
    """True if any active extension emits a STRUCTURALLY different first-step
    equation (e.g. LumpedThermalCapacitance(first_step_steady_state=True) emits
    ``net_heat == 0`` at step 0 vs the inertia equation later). The build-once
    parameter model cannot switch equation structure between steps (its carried
    state is a fixed Var that is never None), so such cases must use the rebuild
    loop. Detected generically via a private flag rather than by type."""
    for ext in net.extensions:
        if getattr(ext, "_first_step_steady_state", False):
            return True
    return False


def _gurobipy_reuse_eligible(
    net,
    solver,
    optimization_problem,
    step_hooks,
    timeseries_data,
    solver_kwargs,
    datetime_index,
) -> bool:
    """Whether the build-once / re-bound-per-step gurobipy model-reuse path can
    replace the per-step rebuild loop and produce identical results.

    Unlike the CasADi fast path, the gurobipy driver *does* wire inter-step
    temporal coupling (storage SoC, the LTC extension, linepack) as per-step
    parameters, so temporally-coupled networks stay eligible. It is gated out of
    the cases it does not reproduce identically: an optimisation problem, step
    hooks observing temporal state, name-addressed or compound series (only
    id-addressed child/node/branch series are declared as parameters), solver
    kwargs it cannot honour, and a custom ``datetime_index`` (the persistent
    model bakes ``dt_h`` in at build time, so non-default/variable step spacing
    is left to the rebuild loop).
    """
    if not _is_gurobipy_solver(solver):
        return False
    if optimization_problem is not None or step_hooks:
        return False
    if datetime_index is not None:
        return False
    # Build-once param model can't reproduce an extension's step-0 structural
    # switch (e.g. LTC first_step_steady_state); leave those to the rebuild loop.
    if _extension_needs_step0_structure_switch(net):
        return False
    td = timeseries_data
    if (
        td._child_name_to_series
        or td._branch_name_to_series
        or td._compound_id_to_series
        or td._compound_name_to_series
    ):
        return False
    if set(solver_kwargs) - {"simulation", "formulation"}:
        return False
    return True


def _run_gurobipy_reuse(
    net,
    solver,
    timeseries_data,
    steps,
    on_step_error,
    progress_callback,
    datetime_index,
    solver_kwargs,
) -> "TimeseriesResult":
    """Build the gurobipy model once and re-bound + re-solve each step (carrying
    state and the integer solution forward) instead of rebuilding every step.
    Equivalent results to the standard loop for the gated cases (see
    :func:`_gurobipy_reuse_eligible`)."""
    from monee.solver.gurobipy import GurobipyTimeseries

    ts = GurobipyTimeseries(
        net,
        timeseries_data,
        formulation=solver_kwargs.get("formulation"),
        simulation=solver_kwargs.get("simulation", False),
        steps=steps,
        params=solver._params,
    )
    step_results: list[StepResult] = []
    for step in range(steps):
        try:
            sr = StepResult(step=step, result=ts.step_result(step))
        except Exception as exc:
            if on_step_error == "raise":
                raise
            _log.warning(_STEP_FAILED_MSG, step, exc)
            sr = StepResult(step=step, result=None, failed=True, error=exc)
        step_results.append(sr)
        if progress_callback is not None:
            progress_callback(step, steps)
    return TimeseriesResult(step_results, datetime_index=datetime_index)


def run(  # NOSONAR
    net: Network,
    timeseries_data: TimeseriesData | None = None,
    steps: int | None = None,
    step_hooks: list[StepHook | Callable] | None = None,
    solver=None,
    backend: str | None = None,
    optimization_problem=None,
    solve_flag: bool = True,
    on_step_error: str = "raise",
    progress_callback: Callable[[int, int], None] | None = None,
    datetime_index: pandas.DatetimeIndex | None = None,
    **solver_kwargs,
) -> TimeseriesResult:
    """Run a timeseries simulation: copy net, apply timeseries, solve, collect.

    ``steps`` defaults to ``timeseries_data.length``. ``on_step_error='skip'``
    records a failed StepResult and continues. ``solver_kwargs`` are forwarded
    to ``solver.solve(...)``.
    """
    if steps is None and timeseries_data is None:
        raise ValueError(
            "Without timeseries data, the number of steps *steps* needs to be provided."
        )
    if steps is None:
        steps = timeseries_data.length
        if steps is None:
            raise ValueError(
                "Cannot infer step count: no series registered and 'steps' not provided."
            )
    if timeseries_data is None:
        timeseries_data = TimeseriesData()
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

    # Resolve solver/backend once up-front; reused across every step.
    resolved_solver = resolve_solver(solver, backend=backend)

    # CasADi graph-reuse fast path: build the NLP + IPOPT solver once and
    # re-solve each step with a warm start instead of rebuilding/recompiling the
    # whole model every step. Gated to the cases that produce identical results
    # (plain power flow, id-addressed memory-less series, no hooks/optimisation).
    if solve_flag and _casadi_reuse_eligible(
        net,
        resolved_solver,
        optimization_problem,
        step_hooks,
        timeseries_data,
        solver_kwargs,
    ):
        _log.debug("run_timeseries: using CasADi build-once graph-reuse fast path")
        return _run_casadi_reuse(
            net,
            timeseries_data,
            steps,
            on_step_error,
            progress_callback,
            datetime_index,
            solver_kwargs,
        )

    # gurobipy model-reuse fast path: build the gurobipy model once and re-bound
    # the time-varying inputs (and carried inter-step state) per step instead of
    # reconstructing the whole model every step. Gated to the cases that produce
    # identical results (see :func:`_gurobipy_reuse_eligible`).
    if solve_flag and _gurobipy_reuse_eligible(
        net,
        resolved_solver,
        optimization_problem,
        step_hooks,
        timeseries_data,
        solver_kwargs,
        datetime_index,
    ):
        _log.debug("run_timeseries: using gurobipy build-once model-reuse fast path")
        return _run_gurobipy_reuse(
            net,
            resolved_solver,
            timeseries_data,
            steps,
            on_step_error,
            progress_callback,
            datetime_index,
            solver_kwargs,
        )

    step_results: list[StepResult] = []
    step_state = StepState()

    for step in range(steps):
        if datetime_index is not None:
            # First step uses the first interval (matching the multi-period
            # engine's _resolve_dt_h); later steps use the prior interval. Both
            # engines then agree on dt_h for identical inputs. Falls back to the
            # default 1.0 only when a single timestamp leaves no interval.
            if step > 0:
                delta = datetime_index[step] - datetime_index[step - 1]
                step_state.dt_h = delta.total_seconds() / 3600.0
            elif len(datetime_index) > 1:
                delta = datetime_index[1] - datetime_index[0]
                step_state.dt_h = delta.total_seconds() / 3600.0

        for hook in step_hooks:
            if isinstance(hook, StepHook):
                hook.pre_run(net, step, step_state)

        net_copy = net.copy()
        timeseries_data.apply_to_network(net_copy, step)

        if solve_flag:
            try:
                result = solve(
                    net_copy,
                    optimization_problem=optimization_problem,
                    solver=resolved_solver,
                    step_state=step_state,
                    **solver_kwargs,
                )
                step_state.push(result.network)
                sr = StepResult(step=step, result=result)
            except Exception as exc:
                if on_step_error == "raise":
                    raise
                _log.warning(_STEP_FAILED_MSG, step, exc)
                sr = StepResult(step=step, result=None, failed=True, error=exc)
        else:
            sr = StepResult(step=step, result=None, skipped=True)

        step_results.append(sr)

        for hook in step_hooks:
            if isinstance(hook, StepHook):
                hook.post_run(net_copy, step, step_state, sr, net)
            else:
                hook(net_copy, step, step_state, sr, net)

        if progress_callback is not None:
            progress_callback(step, steps)

    return TimeseriesResult(step_results, datetime_index=datetime_index)

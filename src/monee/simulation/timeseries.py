import logging
import warnings
from abc import ABC
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas

from monee.model import Network
from monee.model.core import (
    Branch,
    Child,
    Compound,
    CompoundModel,
    Const,
    Node,
    Var,
)
from monee.simulation.core import solve
from monee.simulation.result_utils import (
    build_attribute_frame as _build_attribute_frame,
)
from monee.simulation.result_utils import (
    build_component_frame as _build_component_frame,
)
from monee.simulation.result_utils import (
    build_id_series as _build_id_series,
)
from monee.simulation.result_utils import (
    build_type_stats_html as _build_type_stats_html,
)
from monee.simulation.result_utils import (
    split_id_key as _split_id_key,
)
from monee.simulation.result_utils import (
    wrap_result_html as _wrap_result_html,
)
from monee.simulation.step_state import StepState

# Shared result-rendering helpers, imported from the solver's public reporting
# surface (the simulation layer renders the same kind of result tables).
from monee.solver.core import col_summary as _col_summary
from monee.solver.core import display_df as _display_df
from monee.solver.dispatch import resolve_solver

_log = logging.getLogger(__name__)

_STEP_FAILED_MSG = "Step %d failed: %s"

_SERIES_ATTRS = (
    "_node_id_to_series",
    "_child_id_to_series",
    "_child_name_to_series",
    "_branch_id_to_series",
    "_branch_name_to_series",
    "_compound_id_to_series",
    "_compound_name_to_series",
)


def _unsuccessful_solve_error(step: int, result) -> RuntimeError:
    """Error for a solver that reported failure via ``success=False`` instead
    of raising (Pyomo/GurobiPy style)."""
    details = []
    for attr in ("solver_status", "termination_condition"):
        val = getattr(result, attr, None)
        if val is not None:
            details.append(f"{attr}={val}")
    suffix = f" ({', '.join(details)})" if details else ""
    return RuntimeError(
        f"Step {step}: solver reported an unsuccessful solve (success=False){suffix}"
    )


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
        # A compound attribute is only a build-time seed; without this push the
        # new value never reaches the sub-component carrying the equations.
        if isinstance(model, CompoundModel):
            model.sync()

    def _apply_series(self, comp, timestep: int, id_dict, name_dict=None) -> None:
        if comp.id in id_dict:
            for attr, series in id_dict[comp.id].items():
                self._set_model_attr(comp.model, attr, series[timestep])
        if name_dict is not None and comp.name in name_dict:
            for attr, series in name_dict[comp.name].items():
                self._set_model_attr(comp.model, attr, series[timestep])

    def apply_to_node(self, node, timestep: int) -> None:
        """Apply registered series values for *node* at *timestep*."""
        self._apply_series(node, timestep, self._node_id_to_series)

    def apply_to_child(self, child, timestep: int) -> None:
        """Apply registered series values for *child* at *timestep*."""
        self._apply_series(
            child, timestep, self._child_id_to_series, self._child_name_to_series
        )

    def apply_to_branch(self, branch, timestep: int) -> None:
        """Apply registered series values for *branch* at *timestep*."""
        self._apply_series(
            branch, timestep, self._branch_id_to_series, self._branch_name_to_series
        )

    def apply_to_compound(self, compound, timestep: int) -> None:
        """Apply registered series values for *compound* at *timestep*."""
        self._apply_series(
            compound,
            timestep,
            self._compound_id_to_series,
            self._compound_name_to_series,
        )

    _dispatch = {
        Node: apply_to_node,
        Child: apply_to_child,
        Branch: apply_to_branch,
        Compound: apply_to_compound,
    }

    def apply_to_network(self, net: Network, timestep: int) -> None:
        """Apply all registered series to *net* at *timestep*."""
        for comp in net.iter_all_components():
            self._dispatch[type(comp)](self, comp, timestep)

    def _registered(self) -> tuple[tuple[str, dict], ...]:
        return (
            ("node id", self._node_id_to_series),
            ("child id", self._child_id_to_series),
            ("child name", self._child_name_to_series),
            ("branch id", self._branch_id_to_series),
            ("branch name", self._branch_name_to_series),
            ("compound id", self._compound_id_to_series),
            ("compound name", self._compound_name_to_series),
        )

    def series_labels(self) -> list[str]:
        """Human-readable ``"<kind> <key>"`` labels of all registered series."""
        return [f"{kind} {key!r}" for kind, d in self._registered() for key in d]

    def slice(self, start: int, stop: int) -> "TimeseriesData":
        """New :class:`TimeseriesData` with every registered series cut to the
        positional window ``[start, stop)``. The explicit way to fit a long
        profile to a shorter run horizon (see also :meth:`head`)."""
        if self._length is None:
            raise ValueError(
                "Cannot slice: no series registered. See the how-to/timeseries "
                "docs page."
            )
        if not 0 <= start < stop <= self._length:
            raise ValueError(
                f"slice({start}, {stop}) is out of range for series of length "
                f"{self._length}; 0 <= start < stop <= length is required. "
                "See the how-to/timeseries docs page."
            )
        new_td = TimeseriesData()
        for attr in _SERIES_ATTRS:
            # list(...) first: a pandas Series would slice by label, not by
            # position, and later per-step reads are positional.
            setattr(
                new_td,
                attr,
                {
                    key: {a: list(series)[start:stop] for a, series in attrs.items()}
                    for key, attrs in getattr(self, attr).items()
                },
            )
        new_td._length = stop - start
        return new_td

    def head(self, n: int) -> "TimeseriesData":
        """First *n* steps of every registered series (``slice(0, n)``)."""
        return self.slice(0, n)

    def unbound_keys(self, net: Network) -> list[str]:
        """Registered keys (ids and names) that match no component of *net*,
        as human-readable ``"<kind> <key>"`` labels."""
        ids: dict[type, set] = {
            Node: set(),
            Child: set(),
            Branch: set(),
            Compound: set(),
        }
        names: dict[type, set] = {Child: set(), Branch: set(), Compound: set()}
        for comp in net.iter_all_components():
            cat = type(comp)
            ids[cat].add(comp.id)
            name = getattr(comp, "name", None)
            if name is not None and cat in names:
                names[cat].add(name)
        registered = (
            ("node id", self._node_id_to_series, ids[Node]),
            ("child id", self._child_id_to_series, ids[Child]),
            ("child name", self._child_name_to_series, names[Child]),
            ("branch id", self._branch_id_to_series, ids[Branch]),
            ("branch name", self._branch_name_to_series, names[Branch]),
            ("compound id", self._compound_id_to_series, ids[Compound]),
            ("compound name", self._compound_name_to_series, names[Compound]),
        )
        return [
            f"{kind} {key!r}"
            for kind, series_dict, present in registered
            for key in series_dict
            if key not in present
        ]

    def validate_bound(self, net: Network, strict: bool = True) -> list[str]:
        """Check that every registered series binds to a component of *net* and
        return the unmatched keys. Raises ``KeyError`` listing them; the run
        drivers pass ``strict=False``, which logs them as a warning instead."""
        unbound = self.unbound_keys(net)
        if unbound:
            message = (
                "Timeseries series registered for components that do not exist "
                f"in the network: {', '.join(sorted(unbound))}. "
                "Such a series is never applied. "
                "See the how-to/timeseries docs page."
            )
            if strict:
                raise KeyError(message)
            warnings.warn(message, stacklevel=2)
        return unbound

    def var_targets(self, net: Network) -> list[tuple[str, Any, str, Any]]:
        """Registered series whose target attribute is a solver ``Var`` on the
        bound component's model, as ``(kind, key, attribute, model)`` tuples.
        In a simulation such a series does not act as an input: the solver
        owns the Var's value (see :func:`run`)."""
        lookups = {
            Node: (("node id", self._node_id_to_series, None),),
            Child: (
                ("child id", self._child_id_to_series, None),
                ("child name", self._child_name_to_series, "name"),
            ),
            Branch: (
                ("branch id", self._branch_id_to_series, None),
                ("branch name", self._branch_name_to_series, "name"),
            ),
            Compound: (
                ("compound id", self._compound_id_to_series, None),
                ("compound name", self._compound_name_to_series, "name"),
            ),
        }
        hits: list[tuple[str, Any, str, Any]] = []
        for comp in net.iter_all_components():
            for kind, series_dict, key_attr in lookups[type(comp)]:
                key = comp.id if key_attr is None else getattr(comp, key_attr, None)
                if key is None or key not in series_dict:
                    continue
                for attr in series_dict[key]:
                    if isinstance(getattr(comp.model, attr, None), Var):
                        hits.append((kind, key, attr, comp.model))
        return hits

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
        """Merge *td*; self wins on attribute conflicts. Unequal lengths merge
        by truncating every series (on both sides) to the shorter length, with
        a single warning; use :meth:`head`/:meth:`slice` to pick the window
        explicitly instead."""
        if (
            td._length is not None
            and self._length is not None
            and td._length != self._length
        ):
            shorter = min(self._length, td._length)
            warnings.warn(
                f"Merging TimeseriesData of unequal lengths ({self._length} "
                f"and {td._length}): every series is truncated to the shorter "
                f"length {shorter}. Truncate explicitly with head({shorter}) "
                "or slice(start, stop) to choose the window. See the "
                "how-to/timeseries docs page.",
                stacklevel=2,
            )
            if td._length > shorter:
                td = td.head(shorter)
            else:
                trimmed = self.head(shorter)
                for attr in _SERIES_ATTRS:
                    setattr(self, attr, getattr(trimmed, attr))
                self._length = shorter
        for attr in _SERIES_ATTRS:
            setattr(
                self,
                attr,
                self._merge_component_data(getattr(self, attr), getattr(td, attr)),
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
    def branch_id_data(self):
        return self._branch_id_to_series

    @property
    def compound_id_data(self):
        return self._compound_id_to_series

    @property
    def child_name_data(self):
        return self._child_name_to_series


@dataclass
class StepResult:
    """Outcome of a single timeseries step.

    :class:`~monee.simulation.stepper.Stepper` fills the timing fields: *dt_h*
    is the interval the caller asked for, *effective_dt_h* the one the solve
    integrated over (they differ when a failed step's interval was carried
    forward) and *t_h* the clock time at the start of the step. They stay
    ``None`` for :func:`monee.run_timeseries` results, whose interval is the
    run-wide ``dt_h`` / ``datetime_index``."""

    step: int
    result: Any
    failed: bool = False
    skipped: bool = False
    error: Exception | None = None
    dt_h: float | None = None
    effective_dt_h: float | None = None
    t_h: float | None = None


class TimeseriesResult:
    """Per-step results. Failed steps are excluded from DataFrame queries but
    accessible via :attr:`failed_steps`."""

    def __init__(
        self,
        step_results: list[StepResult],
        datetime_index: pandas.DatetimeIndex | None = None,
        backend_used: str | None = None,
        solver_used: str | None = None,
        index_offset: int = 0,
    ) -> None:
        self._step_results = step_results
        self._datetime_index = datetime_index
        # Subtracted from absolute step numbers before indexing datetime_index;
        # the Stepper uses it to accept a window-sized index under max_history.
        self._index_offset = index_offset
        #: Backend and solver convention selected for this timeseries run
        #: (see :func:`monee.solver.dispatch.resolve_solver`).
        self.backend_used = backend_used
        self.solver_used = solver_used
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
            if self._index_offset:
                step_indices = [s - self._index_offset for s in step_indices]
            return self._datetime_index[step_indices]
        # Label rows by step number (not positionally) so rows stay aligned
        # after skipped/failed steps or bounded history.
        return pandas.Index(step_indices)

    def _frames(self) -> list[tuple[int, dict]]:
        return [(sr.step, sr.result.dataframes) for sr in self._successful()]

    def get_result_for(self, model_type, attribute: str) -> pandas.DataFrame:
        """DataFrame of *attribute* values: rows=steps, cols=component ids."""
        if (model_type, attribute) in self._cache:
            return self._cache[model_type, attribute]
        df = _build_attribute_frame(
            self._frames(), model_type, attribute, self._make_index
        )
        self._cache[model_type, attribute] = df
        return df

    def __getitem__(self, key) -> pandas.DataFrame:
        """All result attributes for a component across every successful step.
        *key* is the component id, or ``(id, model_type)`` when the id is
        shared by several component categories."""
        component_id, model_type = _split_id_key(key)
        return _build_component_frame(
            self._frames(), component_id, self._make_index, model_type
        )

    def get_result_for_id(
        self, component_id, attribute: str, model_type=None
    ) -> pandas.Series:
        """Series of *attribute* for *component_id* across successful steps.
        Yields ``None`` where the component is absent (e.g. islanded out).
        Component ids are unique per category only; *model_type* selects one
        when the id matches several."""
        return _build_id_series(
            self._frames(), component_id, attribute, self._make_index, model_type
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

        SEP = "-" * 68
        lines = [f"TimeseriesResult  {' | '.join(status_parts)}", SEP]

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
                    parts.append(f"{col} in {s}" if "[" in s else f"{col} = {s}")
                row = f"  {type_name:<22} x{n_comp:>2}"
                if parts:
                    row += "  |  " + "  ;  ".join(parts[:3])
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
        title = f"TimeseriesResult  {' | '.join(status_parts)}"

        if not successful:
            return title + "\n  (no successful steps)"

        last = successful[-1]
        SEP = "-" * 68
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
        status_html = " &nbsp;&middot;&nbsp; ".join(extra_parts) if extra_parts else ""

        sections = []
        successful = self._successful()
        if successful:
            # Collect all step dataframes per type
            type_dfs: dict[str, list[pandas.DataFrame]] = {}
            for sr in successful:
                for type_name, df in sr.result.dataframes.items():
                    type_dfs.setdefault(type_name, []).append(df)

            sections = _build_type_stats_html(type_dfs, "step")

        return _wrap_result_html(
            "TimeseriesResult",
            f"<span style='font-weight:normal;color:#555'>{step_info}</span>"
            + (f" &nbsp;&middot;&nbsp; {status_html}" if status_html else ""),
            sections,
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

    for comp in net.iter_all_components():
        if _comp_temporal(comp):
            return True
    for ext in net.extensions:
        if any(
            getattr(type(ext), m, None) is not getattr(NetworkAspect, m, None)
            for m in _TEMPORAL_METHODS
        ):
            return True
    return False


def _declares_var(model, depth: int = 1) -> bool:
    """True if *model* (or, up to *depth* levels down, one of its sub-models)
    carries a solver ``Var`` attribute."""
    for value in vars(model).values():
        if isinstance(value, Var):
            return True
        if depth and hasattr(value, "__dict__") and _declares_var(value, depth - 1):
            return True
    return False


def _temporal_replay_violations(net: Network, step_state, step: int) -> list:
    """Result warnings for ``inter_temporal_equations`` that evaluate to a
    constant ``False`` at *step*.

    Every term of such a constraint is a prescribed (replayed) value, so the
    backends drop it as structurally infeasible (``Equation``'s ``eq is False``
    branch) and the step still reports success. Detecting it here keeps a
    prescribed series that violates the model's own temporal constraint from
    passing silently.

    Only models that declare no ``Var`` are examined: for those the equations
    built here are exactly the ones the backend builds after variable injection,
    so a ``False`` really is a decided-and-violated constraint. A model with a
    ``Var`` would compare its un-injected placeholder and yield ``False`` for
    any constraint at all.
    """
    from monee.solver.core import ResultWarning, as_iter

    entries = []

    def check(component):
        for owner, args in (
            (component.model, (step_state, component.id)),
            (component.formulation, (component.model, step_state, component.id)),
        ):
            method = getattr(owner, "inter_temporal_equations", None)
            if method is None:
                continue
            if _declares_var(component.model):
                return
            try:
                eqs = list(as_iter(method(*args)))
            except Exception:  # noqa: BLE001 - the solve reports it properly
                continue
            for index, eq in enumerate(eqs):
                if eq is False:
                    entries.append(
                        ResultWarning(
                            "temporal",
                            f"step {step}: inter_temporal_equations[{index}] of "
                            f"{type(component.model).__name__} is violated by the "
                            "prescribed series; every term is a replayed value, so "
                            "the constraint is dropped instead of enforced. See "
                            "the how-to/timeseries docs page on prescribed-replay "
                            "semantics.",
                            component=str(component.id),
                        )
                    )

    for component in net.iter_all_components():
        if not getattr(component, "ignored", False):
            check(component)
    return entries


def _report_temporal_replay_violations(entries, result, strict: bool) -> None:
    """Attach *entries* to ``result.warnings`` and surface them; under *strict*
    raise the same :class:`ValidationError` a strict solve would raise."""
    from monee.solver.core import ValidationError

    if not entries:
        return
    result.warnings.extend(entries)
    if strict:
        raise ValidationError(result.warnings)
    warnings.warn(
        "Prescribed timeseries values violate inter-temporal constraints: "
        + "; ".join(str(entry) for entry in entries),
        stacklevel=4,
    )


def _dt_h_at_step(datetime_index, step: int) -> float | None:
    """Inter-step interval in hours at *step* derived from *datetime_index*:
    later steps use the interval to the prior timestamp, the first step borrows
    the first interval, and ``None`` signals that a single timestamp leaves no
    interval (callers keep their default of 1.0). Shared by the sequential
    timeseries loop and the multi-period engine so both agree on dt_h for
    identical inputs."""
    if step > 0:
        delta = datetime_index[step] - datetime_index[step - 1]
    elif len(datetime_index) > 1:
        delta = datetime_index[1] - datetime_index[0]
    else:
        return None
    return delta.total_seconds() / 3600.0


def _td_has_name_or_compound_series(td: TimeseriesData) -> bool:
    """True if *td* carries series the build-once reuse drivers do not wire
    (only id-addressed child/node/branch series are declared as parameters)."""
    return bool(
        td._child_name_to_series
        or td._branch_name_to_series
        or td._compound_id_to_series
        or td._compound_name_to_series
    )


def _only_reuse_kwargs(solver_kwargs) -> bool:
    """True if only kwargs the reuse drivers understand are present."""
    return not (set(solver_kwargs) - {"simulation", "formulation"})


def _reuse_step_loop(ts, steps, on_step_error, progress_callback) -> list["StepResult"]:
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
    return step_results


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
    if _td_has_name_or_compound_series(timeseries_data):
        return False
    return _only_reuse_kwargs(solver_kwargs)


def _run_casadi_reuse(
    net,
    solver,
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
        solver_options=solver.solver_options,
    )
    step_results = _reuse_step_loop(ts, steps, on_step_error, progress_callback)
    return TimeseriesResult(
        step_results,
        datetime_index=datetime_index,
        backend_used="casadi",
        solver_used="ipopt",
    )


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
    dt_h=None,
) -> bool:
    """Whether the build-once / re-bound-per-step gurobipy model-reuse path can
    replace the per-step rebuild loop and produce identical results.

    Unlike the CasADi fast path, the gurobipy driver *does* wire inter-step
    temporal coupling (storage SoC, the LTC extension, linepack) as per-step
    parameters, so temporally-coupled networks stay eligible. It is gated out of
    the cases it does not reproduce identically: an optimisation problem, step
    hooks observing temporal state, name-addressed or compound series (only
    id-addressed child/node/branch series are declared as parameters), solver
    kwargs it cannot honour, and an explicit ``dt_h`` or a custom
    ``datetime_index`` (the persistent model bakes ``dt_h`` in at build time, so
    non-default/variable step spacing is left to the rebuild loop).
    """
    if not _is_gurobipy_solver(solver):
        return False
    if optimization_problem is not None or step_hooks:
        return False
    if datetime_index is not None or dt_h is not None:
        return False
    # Build-once param model can't reproduce an extension's step-0 structural
    # switch (e.g. LTC first_step_steady_state); leave those to the rebuild loop.
    if _extension_needs_step0_structure_switch(net):
        return False
    if _td_has_name_or_compound_series(timeseries_data):
        return False
    return _only_reuse_kwargs(solver_kwargs)


def _run_gurobipy_reuse(
    net,
    solver,
    timeseries_data,
    steps,
    on_step_error,
    progress_callback,
    datetime_index,
    solver_kwargs,
) -> "TimeseriesResult | None":
    """Build the gurobipy model once and re-bound + re-solve each step (carrying
    state and the integer solution forward) instead of rebuilding every step.
    Equivalent results to the standard loop for the gated cases (see
    :func:`_gurobipy_reuse_eligible`). Returns ``None`` when the build-once
    driver cannot represent the case (e.g. compound temporal state or a deeper
    lookback than one step), so the caller falls back to the rebuild loop."""
    from monee.solver.gurobipy import GurobipyTimeseries

    try:
        ts = GurobipyTimeseries(
            net,
            timeseries_data,
            formulation=solver_kwargs.get("formulation"),
            simulation=solver_kwargs.get("simulation", False),
            steps=steps,
            params=solver._params,
        )
    except (ValueError, NotImplementedError) as exc:
        _log.debug(
            "run_timeseries: gurobipy build-once driver unavailable (%s); "
            "falling back to the per-step rebuild loop",
            exc,
        )
        return None
    try:
        step_results = _reuse_step_loop(ts, steps, on_step_error, progress_callback)
    finally:
        ts.dispose()
    return TimeseriesResult(
        step_results,
        datetime_index=datetime_index,
        backend_used="gurobipy",
        solver_used="gurobi",
    )


def _resolve_steps(steps: int | None, timeseries_data: TimeseriesData | None) -> int:
    """Return *steps*; infer from ``timeseries_data.length`` when omitted.
    Shared by the sequential timeseries loop and the multi-period engine."""
    length = None if timeseries_data is None else timeseries_data.length
    if steps is None:
        if length is None:
            raise ValueError(
                "Cannot infer step count: no series registered and 'steps' "
                "not provided."
            )
        return length
    if length is not None and length < steps:
        raise ValueError(
            f"'steps' ({steps}) exceeds the length of the registered series "
            f"({length}).  Either register longer series or reduce 'steps'."
        )
    if length is not None and length > steps:
        # An explicit steps= (the only way to reach this branch) is a
        # deliberate truncation, so this is a log note, not a warning; a
        # length mismatch created by merging unequal series still warns at
        # merge time (see TimeseriesData.extend).
        labels = timeseries_data.series_labels()
        shown = ", ".join(labels[:5]) + (", ..." if len(labels) > 5 else "")
        _log.info(
            "Registered series have length %d but the run covers %d step(s); "
            "the values past step %d of %s are ignored. Truncate explicitly "
            "with timeseries_data.head(%d) or .slice(start, stop). See the "
            "how-to/timeseries docs page.",
            length,
            steps,
            steps - 1,
            shown,
            steps,
        )
    return steps


def _settable_inputs(model) -> list[str]:
    return sorted(
        key
        for key, val in vars(model).items()
        if not key.startswith("_")
        and (isinstance(val, Const) or isinstance(val, (int, float)))
    )


_VAR_SERIES_CONTEXTS = {
    "simulation": (
        "a simulation run",
        "the model stays non-square and every step reports a dof finding",
    ),
    "multi_period": (
        "a multi-period run",
        "every period is pinned at once, which commonly makes the "
        "single-shot solve infeasible",
    ),
}


def _warn_on_var_series_targets(
    timeseries_data: TimeseriesData, net: Network, context: str = "simulation"
) -> None:
    """Warn once at run start when a registered series targets a solver Var:
    the series then only pins the Var through its bounds instead of driving it
    as a plain parameter. Fires whenever a series binds to a component whose
    model declares that attribute as a ``Var``, on both the sequential
    (``run_timeseries``) and the single-shot (``run_multi_period``) path;
    *context* selects the consequence clause and does not change when it
    fires."""
    hits = timeseries_data.var_targets(net)
    if not hits:
        return
    run_kind, consequence = _VAR_SERIES_CONTEXTS[context]
    details = []
    for kind, key, attr, model in hits:
        inputs = _settable_inputs(model)
        shown = ", ".join(inputs[:8]) + (", ..." if len(inputs) > 8 else "")
        details.append(
            f"{kind} {key!r} attribute {attr!r} is a solved variable (Var) on "
            f"{type(model).__name__} (settable inputs: {shown or 'none'})"
        )
    node_hint = (
        " A node quantity is usually driven via the input of an attached "
        "boundary child (for example an ExtHydrGrid's 't_k')."
        if any(kind == "node id" for kind, _, _, _ in hits)
        else ""
    )
    warnings.warn(
        f"Timeseries series target solved variables in {run_kind}: "
        + "; ".join(details)
        + ". Such a series only pins the Var through its bounds; "
        + consequence
        + ". Declare the "
        "attribute as a plain float to make it a settable input, or pass an "
        "optimization problem that controls the Var." + node_hint + " See the "
        "how-to/timeseries docs page.",
        stacklevel=3,
    )


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
    dt_h: float | None = None,
    **solver_kwargs,
) -> TimeseriesResult:
    """Run a timeseries simulation: copy net, apply timeseries, solve, collect.

    ``steps`` defaults to ``timeseries_data.length``. Registered series that
    bind to no component of *net* are reported once, before the first step, as
    a warning (:meth:`TimeseriesData.validate_bound` raises on them instead).
    A series that targets a solver ``Var`` attribute warns up front in a
    simulation run: such a series only pins the Var through its bounds and
    leaves the model non-square (a dof finding per step); a series meant as an
    input should target a settable input, i.e. a plain float or ``Const``
    attribute.
    An ``inter_temporal_equations`` entry whose terms are all prescribed or
    replayed values cannot be enforced by the solve; it is reported as a
    ``temporal`` entry in the step's ``SolverResult.warnings`` (and raised as a
    ``ValidationError`` under ``strict=True``) instead of being dropped
    silently. See the how-to/timeseries docs page.
    ``on_step_error='skip'``
    records a failed StepResult and continues; a step whose solver reports an
    unsuccessful solve (``result.success == False``) counts as failed exactly
    like a raising solver. ``solver_kwargs`` are forwarded to
    ``solver.solve(...)``.

    A plain flow run (``optimization_problem=None``) defaults to
    ``simulation=True``, so a single step matches :func:`monee.run_energy_flow`
    on the same network; pass ``simulation=False`` for the
    optimize-the-feasibility-problem path.

    ``dt_h`` sets the constant inter-step interval in hours seen by
    ``inter_step_equations`` (default ``None`` keeps the StepState default of
    1.0). ``datetime_index`` overrides ``dt_h`` with per-step intervals.

    ``step_hooks`` entries may be :class:`StepHook` instances (receiving both
    ``pre_run`` and ``post_run``) or plain callables, which only receive the
    post-run call ``hook(net_copy, step, step_state, step_result, base_net)``.
    """
    steps = _resolve_steps(steps, timeseries_data)
    if timeseries_data is None:
        timeseries_data = TimeseriesData()
    timeseries_data.validate_bound(net, strict=False)
    if optimization_problem is None:
        solver_kwargs.setdefault("simulation", True)
    if solve_flag and solver_kwargs.get("simulation"):
        _warn_on_var_series_targets(timeseries_data, net)
    if step_hooks is None:
        step_hooks = []
    if on_step_error not in ("raise", "skip"):
        raise ValueError(
            f"on_step_error must be 'raise' or 'skip', got {on_step_error!r}"
        )
    if dt_h is not None and dt_h <= 0:
        raise ValueError(f"dt_h must be positive; got {dt_h}.")
    if datetime_index is not None:
        if len(datetime_index) < steps:
            raise ValueError(
                f"datetime_index length ({len(datetime_index)}) is less than "
                f"steps ({steps})."
            )
        if dt_h is not None:
            _log.warning(
                "Both dt_h and datetime_index were provided; dt_h will be "
                "ignored and step durations will be derived from datetime_index."
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
            resolved_solver,
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
        dt_h,
    ):
        _log.debug("run_timeseries: using gurobipy build-once model-reuse fast path")
        reuse_result = _run_gurobipy_reuse(
            net,
            resolved_solver,
            timeseries_data,
            steps,
            on_step_error,
            progress_callback,
            datetime_index,
            solver_kwargs,
        )
        if reuse_result is not None:
            return reuse_result

    step_results: list[StepResult] = []
    step_state = StepState()
    if dt_h is not None and datetime_index is None:
        step_state.dt_h = dt_h

    for step in range(steps):
        if datetime_index is not None:
            interval = _dt_h_at_step(datetime_index, step)
            if interval is not None:
                step_state.dt_h = interval

        for hook in step_hooks:
            if isinstance(hook, StepHook):
                hook.pre_run(net, step, step_state)

        net_copy = net.copy()
        timeseries_data.apply_to_network(net_copy, step)

        if solve_flag:
            try:
                # Evaluated before the solve: the backend sees exactly these
                # prescribed values, and solving may rebind the attributes.
                temporal_entries = _temporal_replay_violations(
                    net_copy, step_state, step
                )
                result = solve(
                    net_copy,
                    optimization_problem=optimization_problem,
                    solver=resolved_solver,
                    step_state=step_state,
                    **solver_kwargs,
                )
                # Backends that report failure instead of raising (Pyomo,
                # GurobiPy) must hit the same failure contract as raising
                # backends: no push, on_step_error honoured.
                if getattr(result, "success", True) is False:
                    raise _unsuccessful_solve_error(step, result)
                _report_temporal_replay_violations(
                    temporal_entries, result, bool(solver_kwargs.get("strict"))
                )
                step_state.push(result.network, step=step)
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

    return TimeseriesResult(
        step_results,
        datetime_index=datetime_index,
        backend_used=getattr(resolved_solver, "backend_name", None),
        solver_used=getattr(resolved_solver, "solver_name", None),
    )

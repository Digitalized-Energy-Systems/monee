import functools
import logging
import math
import operator
import warnings as _warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import SimpleNamespace

import networkx as nx
import pandas

from monee.model import (
    Child,
    Const,
    ExtHydrGrid,
    ExtPowerGrid,
    GenericModel,
    HeatExchanger,
    Intermediate,
    IntermediateEq,
    MultiGridBranchModel,
    MultiGridCompoundModel,
    Network,
    Node,
    PostProcess,
    Var,
    WaterPipe,
)
from monee.model.child import GridFormingMixin
from monee.model.grid import GasGrid, PowerGrid, WaterGrid
from monee.model.node import Junction
from monee.problem.core import OptimizationProblem

_log = logging.getLogger(__name__)

#: Internal bookkeeping columns omitted from pretty-printed output.
_META_COLS: frozenset[str] = frozenset({"active", "independent"})


def _display_df(df: pandas.DataFrame) -> pandas.DataFrame:
    """Return *df* with internal bookkeeping columns removed."""
    return df.drop(columns=[c for c in _META_COLS if c in df.columns])


#: Reporting-only tolerance for regulation columns: a solved regulation that
#: misses [0, 1] by at most this much is solver tolerance noise and is snapped
#: to the bound in the result frames. The model keeps the solver's own value.
REGULATION_NOISE_FLOOR = 1e-6


def _clamp_regulation_noise(dataframes) -> None:
    """Snap regulation noise within :data:`REGULATION_NOISE_FLOOR` of 0 or 1
    onto the bound, in place, in the reported frames only. Values further out
    are left alone: they are a real violation and reported as one."""
    for df in dataframes.values():
        if not isinstance(df, pandas.DataFrame) or "regulation" not in df.columns:
            continue
        values = pandas.to_numeric(df["regulation"], errors="coerce")
        low = values.between(-REGULATION_NOISE_FLOOR, 0.0, inclusive="left")
        high = values.between(1.0, 1.0 + REGULATION_NOISE_FLOOR, inclusive="neither")
        if low.any():
            df.loc[low, "regulation"] = 0.0
        if high.any():
            df.loc[high, "regulation"] = 1.0


def _col_summary(series: pandas.Series) -> str | None:
    """One-line numeric summary for a single attribute column.

    Returns ``'val'`` for a constant column, ``'[lo, hi]'`` when values vary,
    or ``None`` when the series is empty or entirely NaN.
    """
    vals = series.dropna()
    if vals.empty:
        return None
    lo, hi = float(vals.min()), float(vals.max())
    if abs(hi - lo) < 1e-9 * max(1.0, abs(hi)):
        return f"{lo:.4g}"
    return f"[{lo:.4g}, {hi:.4g}]"


def _summarize_numeric_cols(num: pandas.DataFrame) -> list[str]:
    """Per-column one-line summaries for the repr table row."""
    parts = []
    for col in num.columns:
        s = _col_summary(num[col])
        if s is None:
            continue
        parts.append(f"{col} in {s}" if "[" in s else f"{col} = {s}")
    return parts


_TABLE_CSS = (
    "<style>"
    ".monee-result table{border-collapse:collapse;font-size:.88em;margin-top:4px}"
    ".monee-result th{background:#e8e8e8;padding:3px 10px;border:1px solid #ccc;"
    "text-align:right;font-weight:600}"
    ".monee-result td{padding:3px 10px;border:1px solid #ddd;text-align:right;"
    "white-space:nowrap}"
    ".monee-result tr:nth-child(even) td{background:#f6f6f6}"
    "</style>"
)

# Public, documented aliases for the shared result-rendering helpers above.
# The simulation layer (timeseries / multi_period) renders the same kind of
# result tables, so these are part of the solver's public reporting surface
# rather than backend-private internals.  The underscore-prefixed names are
# retained for backwards compatibility.
display_df = _display_df
col_summary = _col_summary
TABLE_CSS = _TABLE_CSS

#: Footer under a printed/rendered result: rows of a snapshot frame are
#: positional, the component id sits in the 'id' column.
_ID_COLUMN_HINT = (
    "rows are positional; the 'id' column holds the component id "
    "(result.get(Type, index='id') indexes rows by id instead)"
)


@dataclass
class ResultWarning:
    """One typed finding from :func:`validate_result`.

    ``category`` is a stable machine-readable key (``energy_balance``,
    ``served_delta``, ``relaxation``, ``integrality``, ``pruned``,
    ``bound_active``, ``dof``), ``component`` names the affected component when
    the finding is local, ``value`` carries the finding's magnitude where one
    exists (imbalance, shortfall, gap).
    """

    category: str
    message: str
    component: str | None = None
    value: float | None = None

    def __str__(self) -> str:
        loc = f" {self.component}:" if self.component is not None else ""
        return f"[{self.category}]{loc} {self.message}"


class ValidationError(RuntimeError):
    """Raised by ``solve(..., strict=True)`` when result validation attached
    warnings; carries the entries as ``.warnings``."""

    def __init__(self, warnings_list):
        self.warnings = list(warnings_list)
        entries = "\n".join(f"  - {w}" for w in self.warnings)
        super().__init__(
            f"strict mode: the solve succeeded but result validation flagged "
            f"{len(self.warnings)} finding(s):\n{entries}\n"
            "Re-run without strict=True to inspect result.warnings; the "
            "how-to/diagnose_infeasibility docs page explains each category."
        )


@dataclass
class SolverResult:
    """Outcome of a single solve. ``objective=0.0`` for plain energy flow;
    ``None`` when not meaningful (e.g. ``MultiPeriodResult.get_period_result``).

    Objective split
    ---------------
    ``objective`` is the value the solver minimised: the user objectives *plus*
    the formulation's own tightening terms (e.g. the heat-exchanger duty pull,
    which is an unscaled MW quantity). ``user_objective`` reports the user part
    alone and ``aux_objective`` the formulation part, on backends that keep the
    two buckets apart (CasADi, Pyomo, native Gurobi); both are ``None``
    elsewhere. When they differ, ``user_objective`` is the number the user asked
    to minimise; comparing ``objective`` against an external reference double
    counts the tightening terms.

    Validation warnings
    -------------------
    ``warnings`` holds the typed :class:`ResultWarning` entries attached by
    :func:`validate_result` after every solve (empty on a clean result).
    ``solve(..., strict=True)`` raises :class:`ValidationError` instead of
    returning them. The how-to/diagnose_infeasibility docs page explains each
    category.

    Termination metadata
    --------------------
    ``solver_status`` and ``termination_condition`` mirror the Pyomo
    ``SolverStatus`` / ``TerminationCondition`` strings (or ``None`` if
    not populated by the backend). They let downstream consumers
    distinguish a converged-optimal solution from a *witness* incumbent
    Gurobi returns when it hits a time limit - the witness has
    ``success=True`` and looks identical otherwise. Callers that care
    about convergence (e.g. the MC resilience pipeline, which must drop
    aborted samples rather than averaging in a non-converged shed value)
    can inspect ``termination_condition`` to detect the time-limit case.

    Feasibility
    -----------
    ``violations`` reports variable *bound* violations only. ``residuals``
    reports the largest violation of the equation system itself at the solved
    point (``None`` on back-ends that cannot evaluate it). A converged solver
    can return a point that satisfies every bound while missing an equality by
    a physically relevant amount, so an empty ``violations`` dict alone is not
    a feasibility guarantee.

    A ``regulation`` column that misses [0, 1] by at most
    :data:`REGULATION_NOISE_FLOOR` is snapped onto the bound in ``dataframes``;
    the component itself keeps the solver's own value.

    Frame index
    -----------
    The frames in ``dataframes`` are indexed positionally and carry the
    component id in their ``id`` column, so ``df.loc[node_id]`` addresses a row
    number, not a component. Timeseries and multi-period frames are the other
    way round (rows are steps/periods, *columns* are component ids), which is
    what makes the positional snapshot index easy to misread. ``get(model_type,
    index="id")`` returns the same frame indexed by component id for joining
    against id maps; the default stays positional.
    """

    network: Network
    dataframes: dict[str, pandas.DataFrame]
    objective: float | None
    success: bool
    violations: dict[str, float] = field(default_factory=dict)
    solver_status: str | None = None
    termination_condition: str | None = None
    mode_used: str | None = None
    infeasibility_report: object | None = None
    backend_used: str | None = None
    solver_used: str | None = None
    user_objective: float | None = None
    aux_objective: float | None = None
    residuals: float | None = None
    warnings: list = field(default_factory=list)

    def __post_init__(self):
        _clamp_regulation_noise(self.dataframes)

    def summary(self):
        return repr(self)

    def full(self):
        return self.network.as_result_dataframe_dict_str()

    def get(self, model_type, index: str = "position") -> pandas.DataFrame:
        """Result DataFrame for *model_type* (typo-safe vs. dict access).

        ``index="position"`` (the default) returns the stored frame itself: rows
        carry a positional ``RangeIndex`` and the component id lives in the
        ``id`` column. ``index="id"`` returns a copy indexed by that column, so
        ``df.loc[node_id]`` selects the row of that component and a map such as
        ``net.pp_bus_to_node`` joins without a silent off-by-one. Branch frames
        become a ``MultiIndex`` over ``(from_node_id, to_node_id, key)``, the
        branch id tuple. The ``id`` column is kept either way. See the
        concepts/data_model docs page.
        """
        df = self.dataframes.get(model_type.__name__, pandas.DataFrame())
        if index == "position":
            return df
        if index == "id":
            from monee.simulation.result_utils import id_indexed

            return id_indexed(df)
        raise ValueError(
            f"index={index!r} is not a result index mode; use 'position' (the "
            "default positional RangeIndex, id in the 'id' column) or 'id' "
            "(rows indexed by component id). See the concepts/data_model "
            "docs page."
        )

    def __getitem__(self, key) -> pandas.Series:
        """Return the result row for a component. *key* is the component id, or
        ``(id, model_type)`` when the id is shared by several component
        categories. Raises ``KeyError`` if missing and ``ValueError`` when the
        id matches more than one component type."""
        from monee.simulation.result_utils import (
            ambiguous_id_error,
            match_frames,
            split_id_key,
        )

        component_id, model_type = split_id_key(key)
        matches = match_frames(self.dataframes, component_id, model_type=model_type)
        if len(matches) > 1:
            raise ambiguous_id_error(component_id, [name for name, _ in matches])
        if not matches:
            raise KeyError(component_id)
        return matches[0][1]

    def _objective_label(self) -> str:
        if self.objective is None or abs(self.objective) == 0.0:
            return ""
        label = f"objective = {self.objective:.6g}"
        if self.user_objective is not None:
            label += f" (user {self.user_objective:.6g})"
        return label

    def _title(self) -> str:
        title = "SolverResult"
        label = self._objective_label()
        if label:
            title += f"  ({label})"
        return title

    def __repr__(self) -> str:
        # ASCII only: a cp1252 Windows console cannot encode box-drawing chars.
        SEP = "-" * 68
        lines = [self._title(), SEP]
        for type_name, df in self.dataframes.items():
            n = len(df)
            vis = _display_df(df).drop(columns=["id", "node_id"], errors="ignore")
            num = vis.select_dtypes(include="number")
            parts = _summarize_numeric_cols(num)
            row = f"  {type_name:<22} {n:>2}"
            if parts:
                row += "  |  " + "  .  ".join(parts[:4])
            lines.append(row)
        lines.append(SEP)
        if self.violations:
            lines.append("  VIOLATIONS:")
            for key, mag in self.violations.items():
                lines.append(f"    {key}: {mag:.4g}")
            lines.append(SEP)
        return "\n".join(lines)

    def __str__(self) -> str:
        """Full per-type table dump (``print(result)``); ``repr`` gives the summary."""
        SEP = "-" * 68
        lines = [self._title()]
        for type_name, df in self.dataframes.items():
            vis = _display_df(df)
            n = len(vis)
            plural = "instance" if n == 1 else "instances"
            lines.append("")
            lines.append(f"  {type_name}  ({n} {plural})")
            lines.append("  " + SEP)
            table = vis.to_string(index=False, float_format=lambda x: f"{x:.4g}")
            for line in table.splitlines():
                lines.append("  " + line)
        if any("id" in df.columns for df in self.dataframes.values()):
            lines.append("")
            lines.append("  " + _ID_COLUMN_HINT)
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        obj_extra = ""
        label = self._objective_label()
        if label:
            obj_extra = (
                f" &nbsp;<span style='color:#888;font-weight:normal'>{label}</span>"
            )
        sections = []
        for type_name, df in self.dataframes.items():
            vis = _display_df(df)
            n = len(vis)
            plural = "instance" if n == 1 else "instances"
            tbl = vis.to_html(
                index=False,
                border=0,
                classes=[],
                na_rep="-",
                float_format=lambda x: f"{x:.5g}",
            )
            sections.append(
                f"<details open style='margin-bottom:6px'>"
                f"<summary style='cursor:pointer;font-weight:bold;color:#333;"
                f"padding:2px 0'>{type_name} "
                f"<span style='color:#999;font-weight:normal'>({n} {plural})</span>"
                f"</summary>{tbl}</details>"
            )
        if any("id" in df.columns for df in self.dataframes.values()):
            sections.append(
                f"<div style='color:#888;font-size:.8em;padding:2px 0'>"
                f"{_ID_COLUMN_HINT}</div>"
            )
        return (
            f"{_TABLE_CSS}"
            f"<div class='monee-result'>"
            f"<div style='font-weight:bold;font-size:1.05em;padding:4px 0 8px'>"
            f"SolverResult{obj_extra}</div>" + "\n".join(sections) + "</div>"
        )

    def plot(
        self,
        title: str | None = None,
        show_children: bool = True,
        use_monee_positions: bool = False,
        write_to: str | None = None,
    ):
        """Plotly interactive network graph; delegates to :func:`plot_result`."""
        from monee.visualization.result_visualization import plot_result

        return plot_result(
            self,
            title=title,
            show_children=show_children,
            use_monee_positions=use_monee_positions,
            write_to=write_to,
        )


class SinglePeriodSolverProtocol:
    """Documents the contract a solver backend must expose to act as a delegate
    inside :class:`GekkoMultiPeriodSolver` / :class:`PyomoMultiPeriodSolver`.
    Not enforced at runtime."""


class SolverInterface(ABC):
    """Abstract base class for solver backends (GEKKO, Pyomo, ...)."""

    @property
    def backend_name(self) -> str:
        """Backend label recorded on :attr:`SolverResult.backend_used`.
        Concrete backends set ``_backend_name``; unknown custom solvers fall
        back to their class name."""
        return getattr(self, "_backend_name", type(self).__name__)

    @property
    def solver_name(self) -> str | None:
        """Solver label recorded on :attr:`SolverResult.solver_used`.
        Concrete backends set ``_solver_name``; ``None`` when unknown."""
        return getattr(self, "_solver_name", None)

    #: Whether seeding solves with the previous solution speeds this backend
    #: up. Drivers use it as the default for their warm-start behavior (the
    #: Stepper's ``warm_start=None``); an explicit user choice always wins.
    benefits_from_warm_start: bool = True

    def set_warm_start_hint(self, active: bool) -> None:
        """Declare whether the next solve's variable values are a previous
        solution (set by drivers like the Stepper before each solve). Default
        no-op; backends that tune their solver for a warm start override it
        (CasADi switches IPOPT to warm-start options)."""

    @abstractmethod
    def solve(
        self,
        input_network: Network,
        optimization_problem: OptimizationProblem = None,
        draw_debug=False,
        exclude_unconnected_nodes=False,
        step_state=None,
        simulation=False,
        formulation=None,
    ) -> SolverResult:
        """
        Solve the energy-flow / optimisation problem for *input_network*.

        Args:
            input_network: The network to solve.
            optimization_problem: Optional optimisation problem with objectives
                and constraints.  If ``None``, performs a plain energy-flow solve.
            draw_debug: If ``True``, emit debug output from the solver.
            exclude_unconnected_nodes: Legacy flag; prefer islanding config.
            step_state: Inter-step state from the previous timeseries step.
            simulation: If ``True``, square the model (select the simulation
                formulations, pin phantom vars, drop operational flow limits)
                and solve it as a steady-state simulation. Every backend reads
                the flag, so the two settings can converge to different valid
                points on an under-determined network; GEKKO additionally runs
                it as IMODE=1 (falling back to IMODE=3 if not square).
            formulation: Solve-time formulation override - a registry key
                string (``"smooth_nlp"``, ``"convex_miqcqp"``, ...), a
                :class:`~monee.model.formulation.core.NetworkFormulation`, or a
                sequence of either (merged left to right). Overrides the
                network-level ``apply_formulation`` choice; components without
                any choice fall back to ``DEFAULT_SIMULATION_FORMULATION``.

        Returns:
            A :class:`SolverResult` with updated variable values and result
            DataFrames. Every concrete backend runs :func:`validate_result`
            last, attaching :class:`ResultWarning` entries to
            ``result.warnings``; the backends additionally accept
            ``strict=True`` to raise :class:`ValidationError` instead (see the
            how-to/diagnose_infeasibility docs page).
        """

    @abstractmethod
    def _add_equations(self, solver_obj, eqs):
        """Register a list of equations/constraints with the backend solver object."""

    def init_branches(self, branches):
        for branch in branches:
            branch.model.init(branch.grid)

    @staticmethod
    def mark_temporal_components(network, ignored_nodes: set) -> None:
        """Set ``_temporal_active`` on every model carrying a temporal method,
        so static-only constraints can be suppressed when coupling is active."""
        _temporal_methods = frozenset(
            (
                "inter_temporal_equations",
                "inter_step_equations",
                "inter_period_equations",
            )
        )

        def models():
            for node in network.nodes:
                if node.id in ignored_nodes or node.ignored:
                    continue
                yield node.model
                for child in network.childs_by_ids(node.child_ids):
                    if not child.ignored:
                        yield child.model
            for branch in network.branches:
                if not branch.ignored:
                    yield branch.model
            for compound in network.compounds:
                if not compound.ignored:
                    yield compound.model

        for model in models():
            if any(hasattr(model, m) for m in _temporal_methods):
                model._temporal_active = True

    def _collect_temporal_eqs(
        self,
        solver_obj,
        network,
        nodes,
        branches,
        compounds,
        ignored_nodes,
        state,
        mode_method,
    ):
        """Register ``inter_temporal_equations`` and *mode_method* for every
        model/formulation that implements them."""
        methods = ("inter_temporal_equations", mode_method)

        def register(component):
            for method in methods:
                if hasattr(component.model, method):
                    eqs = as_iter(getattr(component.model, method)(state, component.id))
                    self._add_equations(solver_obj, filter_intermediate_eqs(eqs))
                if component.formulation is not None and hasattr(
                    component.formulation, method
                ):
                    eqs = as_iter(
                        getattr(component.formulation, method)(
                            component.model, state, component.id
                        )
                    )
                    self._add_equations(solver_obj, filter_intermediate_eqs(eqs))

        for node in nodes:
            if ignore_node(node, network, ignored_nodes):
                continue
            register(node)
            for child in network.childs_by_ids(node.child_ids):
                if not ignore_child(child, ignored_nodes):
                    register(child)
        for branch in branches:
            if not ignore_branch(branch, network, ignored_nodes):
                register(branch)
        for compound in compounds:
            if not ignore_compound(compound, ignored_nodes):
                register(compound)

    def process_inter_step_equations(
        self,
        solver_obj,
        network: Network,
        nodes,
        branches,
        compounds,
        ignored_nodes: set,
        step_state,
        optimization_problem=None,
        period_index=None,
    ):
        """Collect ``inter_step_equations`` + ``inter_temporal_equations`` plus
        user temporal constraints for timeseries solves."""
        self._collect_temporal_eqs(
            solver_obj,
            network,
            nodes,
            branches,
            compounds,
            ignored_nodes,
            step_state,
            "inter_step_equations",
        )
        self._collect_oxf_temporal_eqs(
            solver_obj, network, step_state, optimization_problem, period_index
        )

    def process_inter_period_equations(
        self,
        solver_obj,
        network: Network,
        nodes,
        branches,
        compounds,
        ignored_nodes: set,
        period_state,
        optimization_problem=None,
        period_index=None,
    ):
        """Collect ``inter_period_equations`` + ``inter_temporal_equations`` plus
        user temporal constraints for multi-period solves."""
        self._collect_temporal_eqs(
            solver_obj,
            network,
            nodes,
            branches,
            compounds,
            ignored_nodes,
            period_state,
            "inter_period_equations",
        )
        self._collect_oxf_temporal_eqs(
            solver_obj, network, period_state, optimization_problem, period_index
        )

    def _collect_oxf_temporal_eqs(
        self, solver_obj, network, temporal_state, optimization_problem, period_index
    ):
        """Evaluate user-defined temporal constraints from the optimization problem."""
        if optimization_problem is None:
            return
        constraints = optimization_problem.constraints
        if constraints is None or not constraints.has_temporal:
            return
        eqs = constraints.all_temporal(
            network, temporal_state, period_index=period_index
        )
        if eqs:
            self._add_equations(solver_obj, filter_intermediate_eqs(eqs))


def as_iter(possible_iter):
    if possible_iter is None:
        raise ValueError("None as result for 'equations' is not allowed!")
    return possible_iter if hasattr(possible_iter, "__iter__") else [possible_iter]


def fold_sum(exprs):
    """Left-fold ``+`` over *exprs*; ``None`` when empty. Unlike ``sum()`` this
    never prepends a ``0 +`` node to the backend expression tree."""
    exprs = list(exprs)
    return functools.reduce(operator.add, exprs) if exprs else None


def add_user_objective(m, obj):
    """Add a *user* objective term, keeping it apart from the formulation's own
    tightening terms where the backend model supports a second bucket
    (``ObjUser``). The formulation terms are unscaled physical quantities, so a
    combined objective value is not the number the user asked to minimise."""
    obj_user = getattr(m, "ObjUser", None)
    if obj_user is None:
        m.Obj(obj)
    else:
        obj_user(obj)


def warn_false_equation(context: str = "", logger=None):
    """Emit the shared always-false-equation warning; *context* is a
    pre-formatted label suffix, *logger* keeps the emitting module's name."""
    (logger or _log).warning(
        "Dropping always-false (structurally infeasible) equation%s; "
        "this usually means a node/branch was over-deactivated.",
        context,
    )


_LEX_REL_TOL = 1e-6
_LEX_ABS_TOL = 1e-9


def lex_cap_slack(s_star: float, mip_gap) -> float:
    """Slack on the phase-2 lexicographic cap so it never excludes a phase-1
    incumbent the solver accepted within its MIPGap tolerance."""
    rel = max(float(mip_gap or 0.0), _LEX_REL_TOL)
    return _LEX_ABS_TOL + rel * max(1.0, abs(s_star))


def sanitize_component_name(name) -> str:
    """Make a string safe for a solver component name (no spaces/brackets)."""
    return (
        str(name)
        .replace("(", "")
        .replace(")", "")
        .replace(", ", "_")
        .replace(" ", "_")
        .replace("[", "")
        .replace("]", "")
    )


def clamp_warm_start(value: Var):
    """Stale start value scrubbed (NaN -> ``None``) and clamped into the Var's
    current bounds; ``None`` when unusable."""
    init = value.value
    if init is None or (isinstance(init, float) and math.isnan(init)):
        return None
    if value.min is not None and init < value.min:
        init = value.min
    if value.max is not None and init > value.max:
        init = value.max
    return init


def snap_to_bounds(val, lb, ub, tol):
    """Snap sub-*tol* bound noise in *val* onto the exact bound."""
    if lb is not None and lb - tol <= val < lb:
        val = lb
    if ub is not None and ub < val <= ub + tol:
        val = ub
    return val


def iter_active_models(network, nodes, branches, compounds, ignored_nodes):
    """Yield the model of every active (non-ignored) component: branches, then
    nodes with their children, then compounds."""
    for branch in branches:
        if not ignore_branch(branch, network, ignored_nodes):
            yield branch.model
    for node in nodes:
        if ignore_node(node, network, ignored_nodes):
            continue
        yield node.model
        for child in network.childs_by_ids(node.child_ids):
            if not ignore_child(child, ignored_nodes):
                yield child.model
    for compound in compounds:
        if not ignore_compound(compound, ignored_nodes):
            yield compound.model


def filter_intermediate_eqs(eqs):
    return [eq for eq in eqs if type(eq) is not IntermediateEq]


def filter_bool_eqs(eqs, context=None):
    """Drop Python-``bool`` sentinel equations, identically for both backends.

    An ``equations()`` body can collapse a constraint to a Python ``bool`` when
    a component is over-deactivated (e.g. an attribute overwritten by a
    :class:`Const` so ``a == b`` evaluates eagerly):

    * ``True``  -> tautology, a no-op; dropped silently.
    * ``False`` -> structurally infeasible (contradictory) constraint. Feeding it
      into model construction either crashes the build (GEKKO) or - if it were
      kept - injects an unsatisfiable constraint. We drop it so the
      load-shedding objective can resolve the situation, but emit a warning so
      the modelling error is surfaced rather than silently hidden.

    *context* is an optional label (e.g. ``"node_3"``) included in the warning.
    """
    kept = []
    for eq in eqs:
        if isinstance(eq, bool):
            if eq is False:
                warn_false_equation(f" in {context}" if context is not None else "")
            continue
        kept.append(eq)
    return kept


def _process_intermediate_eqs(m, model, equations):
    """Replace each ``Intermediate``-typed attribute with the backend's named
    intermediate sub-expression (``m.Intermediate``), for backends whose model
    object exposes that hook (GEKKO, CasADi)."""
    for intermediate_eq in [eq for eq in equations if type(eq) is IntermediateEq]:
        attr_intermediate_var = getattr(model, intermediate_eq.attr)
        eq = (
            intermediate_eq.eq() if callable(intermediate_eq.eq) else intermediate_eq.eq
        )
        if type(attr_intermediate_var) is Intermediate:
            i = m.Intermediate(eq)
            setattr(model, intermediate_eq.attr, i)


class OperatorEquationAssembly:
    """Equation-assembly passes shared by backends whose model object exposes
    operator-overloaded symbols plus GEKKO-style hooks (``m.sin``/``m.cos``/...,
    ``m.Equation(s)``, ``m.Obj``, ``m.Intermediate``): the GEKKO and CasADi
    backends.

    monee's component ``equations()`` bodies are backend-agnostic - they build
    expressions from the injected variable objects and the math hooks passed in -
    so a single set of passes drives both backends. Pyomo and Gurobi construct
    constraints differently (their own intermediate handling / objective
    accumulation) and provide their own overrides instead of using this mixin.

    Hosts must set ``self._simulation`` (the per-solve simulation flag, read by
    the branch pass to drop operational flow limits) and may override
    :meth:`_pwl_impl`.
    """

    def _pwl_impl(self, m):  # NOSONAR
        """Backend piecewise-linear/spline implementation handed to
        ``branch.equations(..., pwl_impl=...)``. ``None`` when the backend has no
        PWL support (e.g. CasADi); GEKKO overrides it with a cubic-spline impl."""
        return None

    def process_internal_oxf_components(self, m, network):
        for constraint in network.constraints:
            m.Equations(
                filter_bool_eqs([constraint(network)], context="network constraint")
            )
        obj = fold_sum(objective(network) for objective in network.objectives)
        if obj is not None:
            add_user_objective(m, obj)

    def process_oxf_components(
        self,
        m,
        network: "Network",
        optimization_problem,
        period_index=None,
    ):
        if optimization_problem.constraints is not None and (
            not optimization_problem.constraints.empty
        ):
            m.Equations(
                filter_bool_eqs(
                    optimization_problem.constraints.all(
                        network, period_index=period_index
                    ),
                    context="optimization problem constraint",
                )
            )
        obj = fold_sum(
            optimization_problem.objectives.all(network, period_index=period_index)
            if optimization_problem.objectives is not None
            else []
        )
        if obj is not None:
            add_user_objective(m, obj)

    def process_equations_compounds(self, m, network, compounds, ignored_nodes):
        for compound in compounds:
            if ignore_compound(compound, ignored_nodes):
                continue

            equations = compound.equations(network)

            for expr in compound.minimize(network, sqrt_impl=m.sqrt):
                m.Obj(expr)

            if equations is not None:
                compound_eqs = filter_bool_eqs(
                    as_iter(equations), context=f"compound_{compound.id}"
                )
                _process_intermediate_eqs(m, compound, compound_eqs)
                m.Equations(filter_intermediate_eqs(compound_eqs))

    def process_equations_nodes_childs(
        self, m, network: "Network", nodes, ignored_nodes
    ):
        for node in nodes:
            if ignore_node(node, network, ignored_nodes):
                continue
            node_childs = network.childs_by_ids(node.child_ids)
            grid = node.grid

            from_branches = [
                network.branch_by_id(branch_id).model
                for branch_id in node.from_branch_ids
                if not ignore_branch(
                    network.branch_by_id(branch_id), network, ignored_nodes
                )
            ]
            to_branches = [
                network.branch_by_id(branch_id).model
                for branch_id in node.to_branch_ids
                if not ignore_branch(
                    network.branch_by_id(branch_id), network, ignored_nodes
                )
            ]
            connected_childs = [
                child.model
                for child in node_childs
                if not ignore_child(child, ignored_nodes)
            ]
            equations = as_iter(
                node.equations(
                    grid,
                    from_branches,
                    to_branches,
                    connected_childs,
                    sin_impl=m.sin,
                    cos_impl=m.cos,
                    if_impl=m.if2,
                    abs_impl=m.abs3,
                    max_impl=m.max2,
                    sign_impl=m.sign2,
                    sqrt_impl=m.sqrt,
                    simulation=self._simulation,
                )
            )
            for expr in node.minimize(
                grid, from_branches, to_branches, connected_childs, sqrt_impl=m.sqrt
            ):
                m.Obj(expr)

            node_eqs = filter_bool_eqs(equations, context=f"node_{node.id}")
            _process_intermediate_eqs(m, node.model, node_eqs)
            m.Equations(filter_intermediate_eqs(node_eqs))

            for child in node_childs:
                if ignore_child(child, ignored_nodes):
                    continue
                child_eqs = filter_bool_eqs(
                    as_iter(
                        child.equations(grid, node.model, simulation=self._simulation)
                    ),
                    context=f"child_{child.id}",
                )

                for expr in child.minimize(grid, node.model, sqrt_impl=m.sqrt):
                    m.Obj(expr)

                _process_intermediate_eqs(m, child.model, child_eqs)
                m.Equations(filter_intermediate_eqs(child_eqs))

    def process_equations_branches(
        self, m, network, branches, ignored_nodes, objs_exprs
    ):
        pwl_impl = self._pwl_impl(m)
        for branch in branches:
            if ignore_branch(branch, network, ignored_nodes):
                continue
            grid = branch.grid

            branch_eqs = as_iter(
                branch.equations(
                    grid,
                    network.node_by_id(branch.from_node_id).model,
                    network.node_by_id(branch.to_node_id).model,
                    sin_impl=m.sin,
                    cos_impl=m.cos,
                    if_impl=m.if3,
                    abs_impl=m.abs3,
                    max_impl=m.max2,
                    sign_impl=m.sign3,
                    log_impl=m.log10,
                    sqrt_impl=m.sqrt,
                    exp_impl=m.exp,
                    pwl_impl=pwl_impl,
                    simulation=self._simulation,
                )
            )

            for expr in branch.minimize(
                grid,
                network.node_by_id(branch.from_node_id).model,
                network.node_by_id(branch.to_node_id).model,
                sqrt_impl=m.sqrt,
            ):
                objs_exprs.append(expr)

            branch_eqs = filter_bool_eqs(branch_eqs, context=f"branch_{branch.id}")
            _process_intermediate_eqs(m, branch.model, branch_eqs)
            m.Equations(filter_intermediate_eqs(branch_eqs))


def inject_nans(target: GenericModel):
    """Replace Var/Const/Intermediate fields with NaN placeholders; zero
    regulation. Covers every solved attribute so pruned components really do
    appear as NaN in the result frames, as the pruning warning promises."""
    for key, value in target.__dict__.items():
        if isinstance(value, Const):
            setattr(target, key, Const(float("nan")))
        if isinstance(value, Var):
            setattr(
                target,
                key,
                Var(float("nan"), max=value.max, min=value.min, name=value.name),
            )
        if isinstance(value, Intermediate):
            setattr(target, key, Intermediate(float("nan")))
    if hasattr(target, "regulation") and not isinstance(target.regulation, Var):
        target.regulation = 0.0


def mark_ignored_components(network, ignored_nodes):
    """Pre-mark ``component.ignored`` so ``optimization_problem._apply`` excludes them."""
    for branch in network.branches:
        if ignore_branch(branch, network, ignored_nodes):
            branch.ignored = True
    for node in network.nodes:
        if ignore_node(node, network, ignored_nodes):
            node.ignored = True
            for child in network.childs_by_ids(node.child_ids):
                child.ignored = True
    _mark_ignored_compounds(network, ignored_nodes)


def _mark_ignored_compounds(network, ignored_nodes):
    for compound in network.compounds:
        if ignore_compound(compound, ignored_nodes):
            compound.ignored = True
            # Children attached to *external* nodes (e.g. SubHG at the heat
            # node of a broken CHPHG) wouldn't be caught by the node-loop
            # above - their host node may still be in the active grid.
            # Mark them directly so ignore_child filters them out.
            for sc in compound.subcomponents:
                if isinstance(sc, Child):
                    sc.ignored = True


def _carrier_node_ids(network: Network, ignored_nodes, grid_type) -> set:
    """Active, non-ignored node ids belonging to a single hydraulic carrier."""
    return {
        n.id
        for n in network.nodes
        if isinstance(n.grid, grid_type) and n.active and n.id not in ignored_nodes
    }


def _carrier_islands(network: Network, node_ids: set, pressure_coupled_only=False):
    """Connected components of the active-pipe subgraph restricted to *node_ids*
    (pipes only ever join same-carrier nodes, so each component is one island).
    ``pressure_coupled_only=True`` drops active heat-exchanger branches, which
    carry mass and temperature but impose no pressure relation - for gauge
    pinning, an HX-fed return loop is its own pressure island."""
    from monee.model.branch import HeatExchanger

    g = nx.Graph()
    g.add_nodes_from(node_ids)
    for branch in network.branches:
        if (
            branch.active
            and branch.from_node_id in node_ids
            and branch.to_node_id in node_ids
            and not (pressure_coupled_only and isinstance(branch.model, HeatExchanger))
        ):
            g.add_edge(branch.from_node_id, branch.to_node_id)
    return nx.connected_components(g)


def _island_grid_forming_node(network: Network, island):
    """Return the id of an island node carrying an active grid-forming child, or
    ``None`` if the island has no pressure/temperature reference."""
    for nid in island:
        node = network.node_by_id(nid)
        childs = network.childs_by_ids(node.child_ids)
        if any(isinstance(c.model, GridFormingMixin) and c.active for c in childs):
            return nid
    return None


def pin_floating_hydraulic_gauges(network: Network, ignored_nodes):
    """Pin one node's pressure per energized gas/water island that has no
    grid-forming source.

    A grid-forming child (``ExtHydrGrid`` / ``GridFormingSource``) pins an
    absolute pressure reference via its ``overwrite``. An island with none
    references pressure only through the (relative) pipe pressure-drop equations
    - its absolute level is a free gauge degree of freedom, which IPOPT can only
    resolve by regularising a rank-deficient block (slow, fragile). Such islands
    arise routinely: e.g. a heat-exchanger-fed return loop, where the coupling HE
    transfers mass and temperature but imposes no pressure drop.

    Pinning any one node's pressure to the carrier nominal removes the
    rank-deficiency without changing flows or pressure *differences* (only the
    arbitrary absolute level). Temperature is left free: where present it is
    referenced through the heat-exchanger coupling, not a nodal pin.
    """
    for grid_type in (GasGrid, WaterGrid):
        ids = _carrier_node_ids(network, ignored_nodes, grid_type)
        for island in _carrier_islands(network, ids, pressure_coupled_only=True):
            if _island_grid_forming_node(network, island) is not None:
                continue
            node = network.node_by_id(min(island))
            nominal = getattr(node.grid, "nominal_pressure_pu", 1.0)
            model = node.model
            if isinstance(getattr(model, "pressure_pu", None), Var):
                model.pressure_pu = Const(nominal)
            if isinstance(getattr(model, "pressure_squared_pu", None), Var):
                model.pressure_squared_pu = Const(nominal**2)


def mark_heat_balance_slacks(network: Network, ignored_nodes):
    """Mark one grid-forming reference node per energized heat island so its
    (dependent) nodal heat balance is dropped - the heat carrier's slack,
    mirroring the free mass-flow / angle slack the other carriers already have.

    The nodal heat balances over a connected island are linearly dependent (one
    is redundant); the grid-forming node absorbs the imbalance. Islands are
    connected components of the active water subgraph; references are
    ``GridFormingMixin`` children, matching the islanding extension's criterion.
    """
    heat_ids = _carrier_node_ids(network, ignored_nodes, WaterGrid)
    if not heat_ids:
        return
    for island in _carrier_islands(network, heat_ids):
        ref = _island_grid_forming_node(network, island)
        if ref is not None:
            network.node_by_id(ref).model._drop_heat_balance = True


def _island_has_free_mass_flow_child(network: Network, island) -> bool:
    """True if any active child on the island leaves its mass flow to the solver
    (slack-like: ``ExtHydrGrid`` / ``ConsumeHydrGrid``), i.e. the island can
    absorb an arbitrary through-flow."""
    for nid in island:
        node = network.node_by_id(nid)
        for child in network.childs_by_ids(node.child_ids):
            if child.active and isinstance(
                getattr(child.model, "mass_flow_kgs", None), Var
            ):
                return True
    return False


def mark_he_flow_prescription(network: Network, ignored_nodes):  # NOSONAR
    """Decide for each dynamic heat exchanger (compound-internal ``SubHE``)
    whether it prescribes its design mass flow or yields to the network.

    A SubHE bridges water islands through its multi-grid control node (the
    islands stay separate because :func:`_carrier_islands` only spans pure
    water nodes). In a supply/return structure every adjacent island has a
    slack child with a free mass flow (``ExtHydrGrid`` / ``ConsumeHydrGrid``),
    so the through-flow is otherwise undetermined and the design flow must
    prescribe it. If any adjacent island lacks such a slack (e.g. a
    fixed-mass-flow ``Sink`` fed only through the compound), that island's
    balance already fixes the through-flow - pinning it to the design flow
    would over-determine the system, so the HE yields and its energy balance
    runs on the actual flow instead (``_he_flow_prescribed = False``)."""
    dyn_he_branches = [
        b
        for b in network.branches
        if b.active and getattr(b.model, "_calc_mass_flow", False)
    ]
    if not dyn_he_branches:
        return
    water_ids = _carrier_node_ids(network, ignored_nodes, WaterGrid)
    node_to_island: dict = {}
    island_free: list = []
    for idx, island in enumerate(_carrier_islands(network, water_ids)):
        island_free.append(_island_has_free_mass_flow_child(network, island))
        for nid in island:
            node_to_island[nid] = idx
    for branch in dyn_he_branches:
        endpoints = [branch.from_node_id, branch.to_node_id]
        # Water-side endpoints contribute their island directly; the other
        # endpoint is the compound's control node - collect the islands of its
        # water neighbours (reached via the compound's transfer branches).
        adjacent = {node_to_island[nid] for nid in endpoints if nid in node_to_island}
        for cn_id in (nid for nid in endpoints if nid not in node_to_island):
            for b in network.branches:
                if not b.active:
                    continue
                if b.from_node_id == cn_id and b.to_node_id in node_to_island:
                    adjacent.add(node_to_island[b.to_node_id])
                elif b.to_node_id == cn_id and b.from_node_id in node_to_island:
                    adjacent.add(node_to_island[b.from_node_id])
        branch.model._he_flow_prescribed = (
            all(island_free[i] for i in adjacent) if adjacent else True
        )


def prepare_solve_network(  # NOSONAR
    input_network: Network,
    optimization_problem=None,
    formulation=None,
    simulation: bool = False,
    exclude_unconnected_nodes: bool = False,
):
    """Backend-agnostic solve-time network preparation.

    Copies *input_network*, runs each extension's ``prepare``, attaches the
    effective formulations, locates the islanding config, computes the ignored
    nodes (and pre-marks ignored components) and finally applies the
    optimization problem. Returns ``(network, ignored_nodes, islanding_config)``.

    The ignored-node computation runs when there is no optimization problem or
    ``exclude_unconnected_nodes`` is set; ``optimization_problem._apply`` runs
    only when a problem is present - mirroring every call site exactly. The
    timeseries drivers reproduce their behaviour by passing the same flags
    (CasADi timeseries: ``optimization_problem=None``; Gurobi timeseries:
    ``exclude_unconnected_nodes=False``).
    """
    from monee.model.extension.islanding.core import NetworkIslandingConfig
    from monee.model.formulation.registry import attach_formulations

    network = input_network.copy()
    # Read by the islanding extension: injection gating / the energisation
    # objective only apply to plain flow solves - an optimization problem
    # brings its own shedding vars and runs _apply after prepare.
    network._solve_has_optimization_problem = optimization_problem is not None
    # Read by inject_vars: phantom degrees of freedom only have to go when the
    # solve is meant to be square.
    network._solve_simulation = bool(simulation)
    for ext in network.extensions:
        ext.prepare(network)
    attach_formulations(network, formulation, simulation=simulation)
    islanding_config = next(
        (e for e in network.extensions if isinstance(e, NetworkIslandingConfig)),
        None,
    )
    ignored_nodes: set = set()
    if optimization_problem is None or exclude_unconnected_nodes:
        ignored_nodes = find_ignored_nodes(network, islanding_config)
        if ignored_nodes:
            mark_ignored_components(network, ignored_nodes)
            described = describe_pruning(network, ignored_nodes)
            if described is not None:
                _warnings.warn(described[0], stacklevel=2)
    if optimization_problem is not None:
        optimization_problem._apply(network)
    return network, ignored_nodes, islanding_config


def apply_child_overwrites(network: Network, nodes, ignored_nodes) -> None:
    """Apply each active child's ``overwrite`` onto its host node's model."""
    for node in nodes:
        if ignore_node(node, network, ignored_nodes):
            continue
        for child in network.childs_by_ids(node.child_ids):
            if child.active:
                child.model.overwrite(node.model, node.grid)


def mark_slacks_and_prescriptions(network: Network, ignored_nodes) -> None:
    """Pin floating hydraulic gauges, mark heat-balance slacks and decide the
    dynamic heat-exchanger flow prescription - the backend-agnostic marking trio
    run after the child overwrites and before variable injection."""
    pin_floating_hydraulic_gauges(network, ignored_nodes)
    mark_heat_balance_slacks(network, ignored_nodes)
    mark_he_flow_prescription(network, ignored_nodes)


def finalize_solution(
    nodes, branches, compounds, network: Network, input_network: Network
) -> dict[str, float]:
    """Post-solve trio: evaluate post-processes, persist solved values back into
    *input_network* (warm start) and return the bound-violation report."""
    apply_post_process_all(nodes, branches, compounds, network)
    persist_solution(network, input_network)
    return compute_bound_violations(nodes, branches, compounds, network)


EQUATION_RESIDUAL_TOL = 1e-4


def warn_on_residuals(
    max_residual: float | None, tol: float = EQUATION_RESIDUAL_TOL
) -> float | None:
    """Warn when the solved point misses the equation system by more than *tol*
    and return *max_residual* unchanged.

    Deliberately advisory: ``success`` stays whatever the solver reported, so
    callers keep the solution and decide themselves.
    """
    if max_residual is not None and max_residual > tol:
        _log.warning(
            "Solved point violates the equation system by up to %.4g (tolerance "
            "%g): the reported flows do not satisfy the model equations, even "
            "though the solver reported success and no bound violations. "
            "Cross-check with another back-end before using the result.",
            max_residual,
            tol,
        )
    return max_residual


def finalize_failed_solution(
    nodes, branches, compounds, network: Network, input_network: Network
) -> dict[str, float]:
    """Post-solve trio for a solve without a usable solution: the withdrawn
    values are meaningless (zeros / scrubbed initialisations), so copy the
    pre-solve values back from the untouched *input_network* instead of
    persisting them (which would clobber warm starts and carried storage
    state), then evaluate post-processes and return the bound-violation
    report."""
    persist_solution(input_network, network)
    apply_post_process_all(nodes, branches, compounds, network)
    return compute_bound_violations(nodes, branches, compounds, network)


def inject_vars(inject_fn, nodes, branches, compounds, network, ignored_nodes):
    """Call ``inject_fn(model, component, category)`` on each active component;
    ignored components get :func:`inject_nans` instead.

    ``category`` in {``branch``, ``node``, ``child``, ``compound``}.

    A branch model may declare ``drop_unused_vars(grids)`` to remove the private
    :class:`Var` attributes its carriers never reference. Injection scans the
    model ``__dict__`` and runs before ``init_branches``, so a multi-carrier
    branch that declares one Var per carrier would otherwise contribute a
    backend variable per carrier it does not have, appearing in no equation.
    The hook runs in simulation mode only (``network._solve_simulation``, set by
    :func:`prepare_solve_network`), where such a variable breaks the squareness
    the solve depends on and, on a multi-energy compound, IPOPT's flat start
    with it. An optimisation solve is well posed with them, and dropping them
    there only reshuffles the variable order, moving which local optimum a
    nonconvex solve reaches.
    """
    prune = getattr(network, "_solve_simulation", False)
    for branch in branches:
        if ignore_branch(branch, network, ignored_nodes):
            branch.ignored = True
            inject_nans(branch.model)
            continue
        drop_unused = getattr(branch.model, "drop_unused_vars", None) if prune else None
        if drop_unused is not None:
            drop_unused(branch.grid)
        inject_fn(branch.model, branch, "branch")

    for node in nodes:
        if ignore_node(node, network, ignored_nodes):
            node.ignored = True
            for child in network.childs_by_ids(node.child_ids):
                child.ignored = True
                inject_nans(child.model)
            inject_nans(node.model)
            continue
        inject_fn(node.model, node, "node")
        _inject_node_childs(inject_fn, node, network, ignored_nodes)

    for compound in compounds:
        if ignore_compound(compound, ignored_nodes):
            compound.ignored = True
            inject_nans(compound.model)
            continue
        inject_fn(compound.model, compound, "compound")


def _inject_node_childs(inject_fn, node, network, ignored_nodes):
    for child in network.childs_by_ids(node.child_ids):
        if ignore_child(child, ignored_nodes):
            child.ignored = True
            inject_nans(child.model)
            continue
        inject_fn(child.model, child, "child")


def withdraw_vars(withdraw_fn, nodes, branches, compounds, network):
    """Call ``withdraw_fn(model)`` on each component to materialise solved Vars."""
    for branch in branches:
        withdraw_fn(branch.model)
    for node in nodes:
        withdraw_fn(node.model)
        for child in network.childs_by_ids(node.child_ids):
            withdraw_fn(child.model)
    for compound in compounds:
        withdraw_fn(compound.model)


def apply_post_process(model: GenericModel) -> None:
    """Evaluate the model's :class:`PostProcess` attributes from its solved
    values - runs after ``withdraw_vars`` and fully outside the solver. Lambdas
    receive a namespace, so they read fields as ``v.vm_pu`` (not ``v["vm_pu"]``)."""
    vals = SimpleNamespace(**model.values)
    for attr in model.__dict__.values():
        if isinstance(attr, PostProcess):
            try:
                attr.value = attr.fn(vals)
            except Exception:
                attr.value = float("nan")


def apply_post_process_all(nodes, branches, compounds, network) -> None:
    """Run :func:`apply_post_process` over every solved component."""
    withdraw_vars(apply_post_process, nodes, branches, compounds, network)


def _is_nan_value(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _copy_var_values(src, dst) -> None:
    """Copy ``.value`` for Var/Intermediate from *src* to *dst*. Intermediates
    must be propagated so e.g. derived ``vm_pu`` isn't silently lost. NaN/None
    sources keep the destination's value: a persisted NaN is never a usable
    warm start (injection scrubs it to 0), while the kept value lets e.g. a
    rejoining islanded component restart from its last solved point."""
    for key, val in src.__dict__.items():
        if isinstance(val, (Var, Intermediate)):
            dst_attr = dst.__dict__.get(key)
            if isinstance(dst_attr, (Var, Intermediate)):
                if _is_nan_value(val.value):
                    continue
                # First write snapshots the pre-solve (declared) value so
                # monee.io.native can serialize networks independently of
                # solve state while .value keeps the warm-start contract.
                if not hasattr(dst_attr, "_start_value"):
                    dst_attr._start_value = dst_attr.value
                dst_attr.value = val.value


def persist_solution(solved_copy: Network, original: Network) -> None:
    """Propagate solved values back so the next ``inject_vars`` warm-starts
    (NaN/None values are skipped, see :func:`_copy_var_values`)."""
    for src_node, dst_node in zip(solved_copy.nodes, original.nodes):
        _copy_var_values(src_node.model, dst_node.model)
        for src_child, dst_child in zip(
            solved_copy.childs_by_ids(src_node.child_ids),
            original.childs_by_ids(dst_node.child_ids),
        ):
            _copy_var_values(src_child.model, dst_child.model)
    for src_branch, dst_branch in zip(solved_copy.branches, original.branches):
        _copy_var_values(src_branch.model, dst_branch.model)
    for src_compound, dst_compound in zip(solved_copy.compounds, original.compounds):
        _copy_var_values(src_compound.model, dst_compound.model)


def _he_duty_shortfall(model, tol: float) -> tuple[float, float] | None:
    """``(unmet duty [MW], duty [MW])`` of a fixed-duty heat exchanger, or ``None``.

    The formulations state the duty as ``q_mw_delivered <= q_mw * on_off``
    (``>=`` for a generator), closed only by an objective pull, so a user
    objective can out-bid it and leave the exchanger under-delivering. Only a
    branch without decision freedom is reported: with a regulation/on_off Var
    the shortfall is a shedding decision, not a defect. A compound-internal
    SubHE carries a Var setpoint, but its ``q_mw`` is pinned by the control
    node's coupling equality, so the solved ``q_mw`` is its duty.

    Noise floor: a shortfall is reported only above ``_SERVED_ABS_TOL``
    (1e-4 MW) and above 0.01% of the duty; sub-0.1 kW gaps are solver
    tolerance, not under-delivery. A shortfall exceeding both still fires.
    """
    delivered = getattr(model, "q_mw_delivered", None)
    q_mw_set = getattr(model, "q_mw_set", None)
    regulation = getattr(model, "regulation", 1)
    on_off = getattr(model, "on_off", 1)
    if not isinstance(delivered, Var) or _is_nan_value(delivered.value):
        return None
    if not all(isinstance(v, (int, float)) for v in (regulation, on_off)):
        return None
    if isinstance(q_mw_set, (int, float)) and not isinstance(q_mw_set, bool):
        duty = q_mw_set * regulation * on_off
    else:
        q_mw = getattr(model, "q_mw", None)
        if not isinstance(q_mw, Var) or _is_nan_value(q_mw.value):
            return None
        duty = q_mw.value * on_off
    shortfall = abs(duty) - abs(delivered.value)
    duty_tol = max(tol, _SERVED_ABS_TOL, 1e-4 * abs(duty))
    return (shortfall, duty) if shortfall > duty_tol else None


def compute_bound_violations(  # NOSONAR
    nodes, branches, compounds, network, tol: float = 1e-6
) -> dict[str, float]:
    """``{"<Type>.<id>.<attr>": magnitude}`` for Var.value violations beyond *tol*,
    plus ``<attr> == "q_mw_delivered_shortfall"`` entries for fixed-duty heat
    exchangers that did not reach their setpoint."""
    violations: dict[str, float] = {}

    def _check(model, label: str) -> None:
        for key, val in model.__dict__.items():
            if not isinstance(val, Var):
                continue
            v = val.value
            if _is_nan_value(v):
                continue
            if val.min is not None and v < val.min - tol:
                violations[f"{label}.{key}"] = val.min - v
            elif val.max is not None and v > val.max + tol:
                violations[f"{label}.{key}"] = v - val.max

    for branch in branches:
        label = f"{type(branch.model).__name__}.{branch.id}"
        _check(branch.model, label)
        if isinstance(branch.model, HeatExchanger):
            duty_gap = _he_duty_shortfall(branch.model, tol)
            if duty_gap is not None:
                shortfall, duty = duty_gap
                violations[f"{label}.q_mw_delivered_shortfall"] = shortfall
                _log.warning(
                    "%s delivers %.4g MW of its %.4g MW setpoint - the duty "
                    "inequality stayed slack (a user objective can out-bid the "
                    "term that closes it). See docs how-to/load_shedding on "
                    "reading this diagnostic.",
                    label,
                    abs(branch.model.q_mw_delivered.value),
                    abs(duty),
                )
    for node in nodes:
        _check(node.model, f"{type(node.model).__name__}.{node.id}")
        for child in network.childs_by_ids(node.child_ids):
            _check(child.model, f"{type(child.model).__name__}.{child.id}")
    for compound in compounds:
        _check(compound.model, f"{type(compound.model).__name__}.{compound.id}")

    return violations


_BALANCE_ABS_TOL = 1e-4
_BINDING_REL_TOL = 1e-6
_SHED_TOL = 1e-6
# Absolute noise floor for served/delivered deltas: 1e-4 MW (0.1 kW) resp.
# 1e-4 kg/s. Chosen from the study evidence (IPOPT leaves O(1e-6) constraint
# noise, e.g. unserved 7.8e-7 MW on a 0.5 MW load); a purely relative 1e-6
# cutoff would not cover it.
_SERVED_ABS_TOL = 1e-4
_MAX_LOCAL_ENTRIES = 10
_BOUND_SKIP_ATTRS = frozenset({"regulation", "on_off", "direction"})
_NODE_BOUND_ATTRS = frozenset(
    {"vm_pu", "t_pu", "pressure_pu", "vm_pu_squared", "pressure_squared_pu"}
)


def _val(v) -> float | None:
    """Plain finite float from a solved attribute; ``None`` when unusable."""
    if isinstance(v, (Var, Const, Intermediate, PostProcess)):
        v = v.value
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    if math.isnan(v):
        return None
    return float(v)


def _factor(model, attr) -> float:
    v = _val(getattr(model, attr, 1.0))
    return 1.0 if v is None else v


def _hydraulic_side_flow(model, side: str) -> float | None:
    direct = _val(getattr(model, f"{side}_mass_flow_kgs", None))
    if direct is not None:
        return direct * _factor(model, "on_off")
    pos = _val(getattr(model, "mass_flow_pos_kgs", None))
    neg = _val(getattr(model, "mass_flow_neg_kgs", None))
    if pos is not None and neg is not None:
        on = _factor(model, "on_off")
        signed = neg - pos if side == "from" else pos - neg
        return signed * on
    return None


def _branch_side_flow(model, side: str, is_power: bool) -> float | None:
    if is_power:
        p = _val(getattr(model, f"p_{side}_mw", None))
        return None if p is None else p * _factor(model, "on_off")
    return _hydraulic_side_flow(model, side)


def _add_branch_balance_terms(network, branch, node_grid, buckets, skip) -> None:
    fg = node_grid.get(branch.from_node_id)
    tg = node_grid.get(branch.to_node_id)
    if fg is None or tg is None or fg is not tg:
        # Carrier boundary (coupler / multi-grid control node): the closure
        # identity does not hold on this carrier - exclude it.
        skip.update(id(g) for g in (fg, tg) if g is not None)
        return
    bucket = buckets[id(fg)]
    is_power = isinstance(fg, PowerGrid)
    for side in ("from", "to"):
        flow = _branch_side_flow(branch.model, side, is_power)
        if flow is None:
            bucket["ok"] = False
            return
        bucket["signed"] += flow
        bucket["mag"] += abs(flow)
    if not is_power:
        # Linepack charging/discharging is storage, not imbalance: the nodal
        # balances carry 0.5 * net_pack_kgs per endpoint (outflow-positive),
        # so the carrier closure identity includes the full net_pack_kgs.
        pack = _val(getattr(branch.model, "net_pack_kgs", None))
        if pack is not None:
            pack *= _factor(branch.model, "on_off")
            bucket["signed"] += pack
            bucket["mag"] += abs(pack)


def _add_child_balance_terms(child, grid, bucket) -> None:
    model = child.model
    attr = "p_mw" if isinstance(grid, PowerGrid) else "mass_flow_kgs"
    if not hasattr(model, attr):
        if isinstance(grid, PowerGrid):
            bucket["ok"] = False
        return
    v = _val(getattr(model, attr))
    if v is None:
        bucket["ok"] = False
        return
    contribution = v * _factor(model, "regulation")
    bucket["signed"] += contribution
    bucket["mag"] += abs(contribution)


def _carrier_balance_entries(network, ignored_nodes) -> list[ResultWarning]:
    """energy_balance findings per carrier. Linepack ``net_pack_kgs`` counts
    as storage charge, not imbalance; the finding still fires when the
    residual including the storage term exceeds
    ``max(_BALANCE_ABS_TOL, 1e-5 * total flow magnitude)``."""
    buckets: dict = {}
    node_grid: dict = {}
    for node in network.nodes:
        if ignore_node(node, network, ignored_nodes):
            continue
        grid = node.grid
        if isinstance(grid, (PowerGrid, GasGrid, WaterGrid)):
            node_grid[node.id] = grid
            buckets.setdefault(
                id(grid), {"grid": grid, "signed": 0.0, "mag": 0.0, "ok": True}
            )
    skip: set = set()
    for branch in network.branches:
        if ignore_branch(branch, network, ignored_nodes):
            continue
        _add_branch_balance_terms(network, branch, node_grid, buckets, skip)
    for node in network.nodes:
        if ignore_node(node, network, ignored_nodes):
            continue
        grid = node_grid.get(node.id)
        if grid is None:
            continue
        for child in network.childs_by_ids(node.child_ids):
            if not ignore_child(child, ignored_nodes):
                _add_child_balance_terms(child, grid, buckets[id(grid)])
    entries = []
    for key, bucket in buckets.items():
        if key in skip or not bucket["ok"]:
            continue
        imbalance = abs(bucket["signed"])
        if imbalance <= max(_BALANCE_ABS_TOL, 1e-5 * bucket["mag"]):
            continue
        grid = bucket["grid"]
        unit = "MW" if isinstance(grid, PowerGrid) else "kg/s"
        entries.append(
            ResultWarning(
                "energy_balance",
                f"carrier balance does not close: net imbalance "
                f"{bucket['signed']:.4g} {unit} across "
                f"{type(grid).__name__}({grid.name}) - the reported flows do "
                "not conserve energy/mass on this carrier",
                component=f"{type(grid).__name__}({grid.name})",
                value=imbalance,
            )
        )
    return entries


def _served_delta_entries(network, ignored_nodes, violations) -> list[ResultWarning]:
    """served_delta findings for demand that was not fully served.

    Suppression rules: a child whose ``regulation`` is an optimisation
    :class:`Var` is skipped entirely (the solved problem made shedding a
    decision, e.g. load-shedding problems and islanding; its optimal
    curtailment is the answer, not a defect), and an unserved amount at or
    below ``max(_SERVED_ABS_TOL, 1e-6 * setpoint)`` is solver noise and
    skipped. The finding still fires when a fixed (non-controllable)
    regulation leaves more than 1e-4 MW resp. kg/s of demand unserved, and
    for heat-exchanger duty shortfalls above the same floor.
    """
    entries = []
    for key, mag in violations.items():
        if key.endswith(".q_mw_delivered_shortfall"):
            entries.append(
                ResultWarning(
                    "served_delta",
                    f"heat exchanger delivers {mag:.4g} MW less than its "
                    "setpoint (the duty inequality stayed slack)",
                    component=key.rsplit(".", 1)[0],
                    value=mag,
                )
            )
    for node in network.nodes:
        if ignore_node(node, network, ignored_nodes):
            continue
        for child in network.childs_by_ids(node.child_ids):
            if ignore_child(child, ignored_nodes):
                continue
            model = child.model
            regulation = getattr(model, "regulation", None)
            if isinstance(regulation, Var):
                continue
            regv = _val(regulation)
            if regv is None or regv >= 1 - _SHED_TOL:
                continue
            for attr in ("p_mw", "mass_flow_kgs", "q_mw_heat"):
                setpoint = _val(getattr(model, attr, None))
                # Load convention: only positive setpoints are demand.
                if setpoint is None or setpoint <= 1e-9:
                    continue
                unserved = setpoint * (1 - regv)
                if unserved <= max(_SERVED_ABS_TOL, 1e-6 * setpoint):
                    break
                entries.append(
                    ResultWarning(
                        "served_delta",
                        f"{attr} setpoint {setpoint:.4g} served at "
                        f"{100 * regv:.1f}% (unserved {unserved:.4g})",
                        component=f"{type(model).__name__}.{child.id}",
                        value=unserved,
                    )
                )
                break
    return entries


_RELAXATION_FLOW_FLOOR_FRACTION = 1e-3
_RELAXATION_FLOW_FLOOR_ABS = 0.02
_INTEGRALITY_ENFORCING_SOLVERS = frozenset({"scip", "gurobi"})


def _relaxation_flow_floor(branch) -> float:
    cap = _val(getattr(getattr(branch, "grid", None), "max_mass_flow_kgs", None))
    if cap is not None and cap > 0:
        return _RELAXATION_FLOW_FLOOR_FRACTION * cap
    return _RELAXATION_FLOW_FLOOR_ABS


def _relaxation_advice(solver_used) -> str:
    used = (solver_used or "").lower()
    if used in _INTEGRALITY_ENFORCING_SOLVERS:
        return (
            f"The relaxation stayed slack even under solver='{used}'; solve "
            "with formulation='smooth_nlp' / 'gas_nlp' for exact physics"
        )
    return (
        "Solve with a MIQCQP solver (e.g. solver='scip') or with "
        "formulation='smooth_nlp' / 'gas_nlp' for exact physics"
    )


def _relaxation_entry(network, solver_used=None) -> ResultWarning | None:
    """Worst untight convex-relaxation gap, or ``None``.

    A branch whose net flow magnitude is below 0.1% of its carrier's
    ``max_mass_flow_kgs`` (0.02 kg/s when the grid has no cap) is skipped:
    the gap is relative and meaningless at negligible flow. The finding
    still fires whenever the flow exceeds that floor and the gap exceeds
    the formulation's ``RELAXATION_GAP_TOL``; the advice never recommends
    the solver that produced the result.
    """
    worst_id, worst_gap, tol = None, 0.0, None
    for branch in network.branches:
        formulation = getattr(branch, "formulation", None)
        gap_of = getattr(formulation, "relaxation_gap", None)
        if gap_of is None:
            continue
        flow = _val(getattr(branch.model, "mass_flow_kgs", None))
        if flow is not None and abs(flow) < _relaxation_flow_floor(branch):
            continue
        gap = gap_of(branch.model)
        if gap > worst_gap:
            worst_id, worst_gap, tol = branch.id, gap, formulation.RELAXATION_GAP_TOL
    if tol is None or worst_gap <= tol:
        return None
    return ResultWarning(
        "relaxation",
        f"convex relaxation not tight: {100 * worst_gap:.1f}% epigraph gap "
        f"(tolerance {100 * tol:.1f}%). {_relaxation_advice(solver_used)}",
        component=f"branch {worst_id}",
        value=worst_gap,
    )


def _uncertified_relaxation_entry(network) -> ResultWarning | None:
    """Worst convex-relaxation slack that no objective term can tighten, or ``None``.

    Distinct from :func:`_relaxation_entry`: that one reports a relaxation the
    model does price and which came out untight anyway. Here the epigraph or
    cone row carries no objective weight at all (a branch-flow SOC cone on a
    zero-resistance branch has no ``r * ell`` loss term), so the slack is
    whatever the solver's MIP gap happened to accept. It is not a physics
    violation, but it does mean the branch certifies no exactness and its
    derived reports are meaningless, so it is surfaced rather than hidden.
    """
    worst_id, worst_slack, tol = None, 0.0, None
    for branch in network.branches:
        formulation = getattr(branch, "formulation", None)
        slack_of = getattr(formulation, "uncertified_relaxation_slack", None)
        if slack_of is None:
            continue
        slack = slack_of(branch, network)
        if slack > worst_slack:
            worst_id, worst_slack, tol = (
                branch.id,
                slack,
                formulation.RELAXATION_GAP_TOL,
            )
    if tol is None or worst_slack <= tol:
        return None
    return ResultWarning(
        "relaxation",
        f"convex relaxation slack by {100 * worst_slack:.1f}% on a branch whose "
        "objective carries no term to tighten it (a branch-flow cone on a "
        "zero-resistance branch). The value is pinned only by the solver's "
        "relative MIP gap (SCIP 'limits/gap', Gurobi 'MIPGap', both preset to "
        "1e-4), so this branch neither certifies nor refutes exactness and its "
        "reported current and loading are not physical. Re-solve with "
        "solver_options={'limits/gap': 0.0} (or MIPGap 0.0) to tighten it; the "
        "concepts/formulations docs page explains the degeneracy",
        component=f"branch {worst_id}",
        value=worst_slack,
    )


_PRUNE_RULE_NO_LEAD = "no active slack or grid-forming child in its island"
_PRUNE_RULE_STUB = "dead-end stub (degree <= 1, no active child)"
_PRUNE_RULE_HEAT_ONLY = (
    "dead-end junction whose only children are heat-only; with no outgoing "
    "pipe or mass-flow child there is no enthalpy stream to balance the heat "
    "against"
)
_PRUNE_MAX_LABELS = 8


def _component_label(component) -> str:
    name = getattr(component, "name", None)
    base = f"{type(component.model).__name__}({component.id})"
    return f"{base} '{name}'" if name else base


def describe_pruning(network, ignored_nodes) -> tuple[str, int] | None:
    """``(message, count)`` naming every user-created component the pruning
    dropped and the rule that dropped it, or ``None`` when nothing was pruned.
    Requires :func:`mark_ignored_components` to have run."""
    rules = getattr(network, "_pruning_rules", {})
    pruned_nodes = [
        n for n in network.nodes if n.active and (n.id in ignored_nodes or n.ignored)
    ]
    pruned_branches = [b for b in network.branches if b.active and b.ignored]
    pruned_childs = [c for c in network.childs if c.active and c.ignored]
    total = len(pruned_nodes) + len(pruned_branches) + len(pruned_childs)
    if total == 0:
        return None

    def rule_of(component) -> str:
        if hasattr(component, "node_id"):
            key = component.node_id
        elif hasattr(component, "from_node_id"):
            key = next(
                (
                    nid
                    for nid in (component.from_node_id, component.to_node_id)
                    if nid in rules
                ),
                None,
            )
        else:
            key = component.id
        return rules.get(key, "attached to an excluded component")

    by_rule: dict[str, list[str]] = {}
    for comp in pruned_nodes + pruned_childs + pruned_branches:
        by_rule.setdefault(rule_of(comp), []).append(_component_label(comp))
    parts = []
    for rule, labels in by_rule.items():
        shown = ", ".join(labels[:_PRUNE_MAX_LABELS])
        if len(labels) > _PRUNE_MAX_LABELS:
            shown += f" and {len(labels) - _PRUNE_MAX_LABELS} more"
        parts.append(f"{shown} [rule: {rule}]")
    message = (
        f"{total} component(s) were excluded from the solve: "
        + "; ".join(parts)
        + ". They appear as NaN/ignored in the result frames. Network.check() "
        "reports this before solving; the how-to/diagnose_infeasibility docs "
        "page explains the pruning rules"
    )
    return message, total


def _pruned_entry(network, ignored_nodes) -> ResultWarning | None:
    described = describe_pruning(network, ignored_nodes)
    if described is None:
        return None
    message, total = described
    return ResultWarning("pruned", message, value=float(total))


def _binding_side(v, lo, hi) -> tuple[str, float] | None:
    for side, bound in (("min", lo), ("max", hi)):
        if bound is None:
            continue
        if abs(v - bound) <= max(1e-8, _BINDING_REL_TOL * max(1.0, abs(bound))):
            return side, bound
    return None


def _is_setpoint_gate(model, key, val, bound) -> bool:
    """True when the binding bound is the component's own setpoint, so
    sitting there is the normal fully-served state, not a constrained
    optimum. Two shapes qualify: islanding-gated injections (Var named
    ``islanding_gated_*`` or listed in ``model._islanding_gated_attrs``),
    whose bounds are the setpoint by construction, and non-grid-forming
    childs at the nonzero end of a ``[0, setpoint]`` / ``[setpoint, 0]``
    gate (the shape ``OptimizationProblem`` setpoint promotion produces).
    bound_active still fires for grid-forming childs (an ext grid at an
    import/export cap), for any Var with two nonzero bounds (true capacity
    ranges, e.g. set via ``prob.bounds()``) and for node-level attributes.
    """
    name = getattr(val, "name", None) or ""
    if name.startswith("islanding_gated_"):
        return True
    if key in getattr(model, "_islanding_gated_attrs", {}):
        return True
    if isinstance(model, GridFormingMixin):
        return False
    if bound == 0:
        return False
    opposite = val.min if bound == val.max else val.max
    return opposite == 0


def _model_bound_entries(model, label, attrs=None) -> list[ResultWarning]:
    flagged: dict[str, ResultWarning] = {}
    for key, val in model.__dict__.items():
        if not isinstance(val, Var) or val.integer:
            continue
        if key in _BOUND_SKIP_ATTRS or key.startswith("_"):
            continue
        if attrs is not None and key not in attrs:
            continue
        if val.min is not None and val.min == val.max:
            continue
        v = _val(val)
        if v is None:
            continue
        hit = _binding_side(v, val.min, val.max)
        if hit is None:
            continue
        side, bound = hit
        if attrs is None and _is_setpoint_gate(model, key, val, bound):
            continue
        flagged[key] = ResultWarning(
            "bound_active",
            f"{key} = {v:.6g} sits at its {side} bound {bound:.6g}; the "
            "bound is constraining the optimum",
            component=label,
            value=v,
        )
    # A squared twin binding alongside its base var is the same finding.
    return [
        entry
        for key, entry in flagged.items()
        if not (key.endswith("_squared") and key.removesuffix("_squared") in flagged)
    ]


def _bound_entries(network, ignored_nodes) -> list[ResultWarning]:
    entries = []
    for node in network.nodes:
        if ignore_node(node, network, ignored_nodes):
            continue
        entries += _model_bound_entries(
            node.model,
            f"{type(node.model).__name__}.{node.id}",
            attrs=_NODE_BOUND_ATTRS,
        )
        for child in network.childs_by_ids(node.child_ids):
            if not ignore_child(child, ignored_nodes):
                entries += _model_bound_entries(
                    child.model, f"{type(child.model).__name__}.{child.id}"
                )
    return entries


def _capped(entries: list, category: str) -> list:
    if len(entries) <= _MAX_LOCAL_ENTRIES:
        return entries
    kept = entries[:_MAX_LOCAL_ENTRIES]
    kept.append(
        ResultWarning(
            category,
            f"... and {len(entries) - _MAX_LOCAL_ENTRIES} more {category} "
            "finding(s) of the same kind",
            value=float(len(entries)),
        )
    )
    return kept


_INTEGRALITY_VERIFY_TOL = 1e-6


def _relaxed_all_integral(relaxed_components) -> bool:
    """True when every relaxed integer Var solved to an integral value
    (within ``_INTEGRALITY_VERIFY_TOL``) and at least one was checkable."""
    seen = False
    for component in relaxed_components:
        for val in component.model.__dict__.values():
            if not isinstance(val, Var) or not val.integer:
                continue
            v = _val(val)
            if v is None:
                continue
            seen = True
            if abs(v - round(v)) > _INTEGRALITY_VERIFY_TOL:
                return False
    return seen


def _mip_solver_example(solver_used) -> str:
    return "gurobi" if (solver_used or "").lower() == "scip" else "scip"


def _integrality_entry(relaxed_components, solver_used=None) -> ResultWarning:
    """integrality finding for relaxed integer variables.

    When every relaxed integer variable verifiably solved to an integral
    value, the result is self-consistent and the message says so instead of
    calling it physically meaningless; the strong wording still fires when
    any relaxed variable is fractional (or its value cannot be checked). The
    recommended back-end never names the solver that produced the result.
    """
    named = ", ".join(
        f"{type(c.model).__name__}({getattr(c, 'id', '?')})"
        for c in relaxed_components[:5]
    )
    more = (
        f" and {len(relaxed_components) - 5} more"
        if len(relaxed_components) > 5
        else ""
    )
    example = _mip_solver_example(solver_used)
    if _relaxed_all_integral(relaxed_components):
        consequence = (
            "All relaxed integer variables solved to integral values (within "
            f"{_INTEGRALITY_VERIFY_TOL:g}), so this result is self-consistent; "
            f"a MIQCQP/MINLP back-end (e.g. solver='{example}') confirms "
            "optimality"
        )
    else:
        consequence = (
            "Results can be physically meaningless (fractional flow "
            f"directions); solve with a MIQCQP/MINLP back-end (e.g. "
            f"solver='{example}') or a binary-free formulation"
        )
    return ResultWarning(
        "integrality",
        f"{len(relaxed_components)} component(s) carry a formulation whose "
        f"integer variables this back-end relaxes: {named}{more}. "
        f"{consequence}",
        value=float(len(relaxed_components)),
    )


_MAX_NAMED_FREE_VARS = 10


def uncovered_free_var_entry(reg, m) -> ResultWarning | None:
    """``dof`` finding for optimization mode, or ``None`` when clean.

    Mirrors the simulation-mode squareness check for the one optimization case
    in which nothing determines the free variables: the model carries more
    variables than equations and no objective term to select among the
    feasible points, so the solver returns an arbitrary one. An optimization
    with an objective is not reported: selecting a point is exactly its job.
    *reg* is the CasADi variable registry and *m* the assembled ``CasModel``.
    """
    import casadi as ca

    if m.obj_terms:
        return None
    n_eq = sum(1 for c in m.cons if c.op == "eq")
    if not m.cons or len(reg) <= n_eq:
        return None
    used = {s.name() for s in ca.symvar(ca.vertcat(*[c.r for c in m.cons]))}
    # Variables no constraint mentions at all are formulation scratch symbols
    # (e.g. an unused squared-voltage alias); counting them would fire on every
    # plain optimization-mode flow.
    n_constrained = sum(1 for r in reg if r["sx"].name() in used)
    if n_constrained <= n_eq:
        return None
    return ResultWarning(
        "dof",
        f"optimization mode without an objective: {n_constrained} variables "
        f"are bound by only {n_eq} equations and no objective selects among "
        "the remaining free ones, so the solved values are one feasible point, "
        "not a unique setpoint. Declare a fixed setpoint as a plain float or "
        "Const, add an equation that pins the Var, or minimise an objective "
        "over it. The how-to/diagnose_infeasibility docs page explains this "
        "category",
        value=float(n_constrained - n_eq),
    )


class ResultValidationSummary(UserWarning):
    """Category of the one-line post-solve validation summary emitted by
    :func:`validate_result`. Silence it with
    ``warnings.filterwarnings("ignore", category=ResultValidationSummary)``.
    An identical summary (same finding profile) from a later solve in the
    same process, e.g. every step of a timeseries or Stepper run, is not
    re-emitted; repeats are counted per message and readable via
    :func:`validation_summary_counts`. A summary with a different finding
    profile always fires again, and every result still carries its full
    ``result.warnings`` list."""


_summary_counts: dict[str, int] = {}


def validation_summary_counts() -> dict[str, int]:
    """``{summary message: times seen}`` for this process, counting the
    suppressed repeats of each validation summary (e.g. across timeseries
    steps)."""
    return dict(_summary_counts)


def reset_validation_summary_counts() -> None:
    """Clear the summary dedup registry so the next identical summary is
    emitted again (e.g. between independent runs in one process)."""
    _summary_counts.clear()


def _emit_validation_summary(n_findings: int, summary: str) -> None:
    message = (
        f"Result validation: {n_findings} finding(s) ({summary}). Inspect "
        "result.warnings; the how-to/diagnose_infeasibility docs page "
        "explains each category. Later solves with this exact finding "
        "profile (e.g. timeseries steps) are counted, not re-reported; see "
        "monee.solver.core.validation_summary_counts()."
    )
    seen = _summary_counts.get(message, 0)
    _summary_counts[message] = seen + 1
    if seen == 0:
        _warnings.warn(message, ResultValidationSummary, stacklevel=3)


def validate_result(  # NOSONAR
    network: Network,
    result: SolverResult,
    *,
    ignored_nodes=None,
    simulation: bool = False,
    strict: bool = False,
    n_vars: int | None = None,
    n_eqs: int | None = None,
    relaxed_integrality=None,
    extra_warnings=None,
) -> SolverResult:
    """Cheap post-solve invariant checks, run by every backend at the end of
    ``solve``. Attaches :class:`ResultWarning` entries to ``result.warnings``,
    emits a one-line :class:`ResultValidationSummary` warning when any were
    found (deduplicated per finding profile, filterable via the ``warnings``
    module) and, under ``strict``, raises :class:`ValidationError` listing
    them. No extra solves are performed; a failed solve is left alone (its
    values are meaningless and the failure is already reported). The
    how-to/diagnose_infeasibility docs page explains each category.
    """
    ignored_nodes = ignored_nodes or set()
    entries: list[ResultWarning] = list(extra_warnings or [])
    if result.success:
        entries += _carrier_balance_entries(network, ignored_nodes)
        entries += _capped(
            _served_delta_entries(network, ignored_nodes, result.violations),
            "served_delta",
        )
        relaxation = _relaxation_entry(network, result.solver_used)
        if relaxation is not None:
            entries.append(relaxation)
        uncertified = _uncertified_relaxation_entry(network)
        if uncertified is not None:
            entries.append(uncertified)
        entries += _capped(_bound_entries(network, ignored_nodes), "bound_active")
        pruned = _pruned_entry(network, ignored_nodes)
        if pruned is not None:
            entries.append(pruned)
        if relaxed_integrality:
            entries.append(_integrality_entry(relaxed_integrality, result.solver_used))
        if (
            simulation
            and n_vars is not None
            and n_eqs is not None
            and n_vars != n_eqs
            and not any(w.category == "dof" for w in entries)
        ):
            entries.append(
                ResultWarning(
                    "dof",
                    f"simulation requested but the model is not square "
                    f"({n_vars} variables, {n_eqs} equations); the solved "
                    "values are one feasible point, not a unique setpoint",
                    value=float(n_vars - n_eqs),
                )
            )
        # Problem-attached validators (e.g. the load-shedding headroom check)
        # registered on the working network by OptimizationProblem._apply hooks.
        for validator in getattr(network, "_result_validators", ()):
            entries += list(validator(network, result))
    result.warnings.extend(entries)
    if result.warnings:
        counts: dict[str, int] = {}
        for w in result.warnings:
            counts[w.category] = counts.get(w.category, 0) + 1
        summary = ", ".join(f"{cat} x{n}" for cat, n in sorted(counts.items()))
        _emit_validation_summary(len(result.warnings), summary)
        if strict:
            raise ValidationError(result.warnings)
    return result


def preview_squareness(
    input_network: Network, formulation=None
) -> tuple[int, int, list[str]]:
    """``(n_vars, n_eqs, unpinned)`` of the simulation-mode model, assembled
    with the CasADi machinery but never solved. ``unpinned`` names injected
    variables that appear in no constraint. Works on a copy; used by
    ``Network.check()`` (see the how-to/diagnose_infeasibility docs page)."""
    import casadi as ca

    from monee.solver.casadi import (
        CasADiSolver,
        CasModel,
        _substitute_relaxed_defaults,
    )

    solver = CasADiSolver()
    solver._simulation = True
    solver._reg = []
    m = CasModel()
    network, ignored_nodes, _ = prepare_solve_network(
        input_network, formulation=formulation, simulation=True
    )
    _substitute_relaxed_defaults(network, formulation, True)
    nodes, branches, compounds = network.nodes, network.branches, network.compounds
    apply_child_overwrites(network, nodes, ignored_nodes)
    mark_slacks_and_prescriptions(network, ignored_nodes)
    inject_vars(solver._inject, nodes, branches, compounds, network, ignored_nodes)
    solver.init_branches(branches)
    solver.process_equations_nodes_childs(m, network, nodes, ignored_nodes)
    solver.process_equations_branches(m, network, branches, ignored_nodes, [])
    solver.process_equations_compounds(m, network, compounds, ignored_nodes)
    solver.process_internal_oxf_components(m, network)
    for ext in network.extensions:
        m.Equations(ext.equations(network, ignored_nodes))
    reg = solver._reg
    n_eqs = sum(1 for c in m.cons if c.op == "eq")
    used = (
        {s.name() for s in ca.symvar(ca.vertcat(*[c.r for c in m.cons]))}
        if m.cons
        else set()
    )
    unpinned = sorted(
        {
            f"{type(r['model']).__name__}.{r['key']}"
            for r in reg
            if r["sx"].name() not in used
        }
    )
    return len(reg), n_eqs, unpinned


def ignore_branch(branch, network: Network, ignored_nodes):
    return (
        (not branch.active)
        or ignore_node(network.node_by_id(branch.id[0]), network, ignored_nodes)
        or ignore_node(network.node_by_id(branch.id[1]), network, ignored_nodes)
    )


def ignore_node(node, network: Network, ignored_nodes):
    ig = (not node.active) or (node.id in ignored_nodes)
    if not node.independent:
        compound = network.compound_of_node(node.id)
        if compound is not None:
            ig = ig or ignore_compound(compound, ignored_nodes)
    return ig


def ignore_child(child, ignored_nodes):
    # ``child.ignored`` is set by ``mark_ignored_components`` when the child
    # belongs to an ignored compound - without consulting it here, the child
    # would still appear in its host node's balance equations as a free Var
    # (e.g. SubHG.heat_mw absorbing arbitrary heat at the heat node).
    return (not child.active) or (child.node_id in ignored_nodes) or child.ignored


def ignore_compound(compound, ignored_nodes):
    ig = not compound.active
    external_broken = any(
        value in ignored_nodes for value in compound.connected_to.values()
    )
    # Internal subcomponent turned off (e.g. user deactivates one of a
    # CHPHG's internal transfer branches): the ControlNode's power balance -
    # degenerate when its from-branches are gone - otherwise collides with
    # its el_mw / heat_mw coupling rows and the LP is infeasible.
    internal_broken = any(
        not getattr(sc, "active", True) for sc in compound.subcomponents
    )
    if external_broken or internal_broken:
        if hasattr(compound.model, "set_active"):
            compound.model.set_active(False)
        else:
            ig = True
    elif hasattr(compound.model, "set_active"):
        compound.model.set_active(True)
    return ig


def generate_real_topology(nx_net):
    net_copy = nx_net.copy()
    # keys=True targets the exact parallel edge - not always key 0.
    for u, v, key, data in nx_net.edges(keys=True, data=True):
        branch = data["internal_branch"]
        if not branch.active or (
            hasattr(branch.model, "on_off")
            and type(branch.model.on_off) is not Var
            and branch.model.on_off == 0
        ):
            net_copy.remove_edge(u, v, key)
    return net_copy


def remove_cps(network: Network):
    relevant_compounds = [
        compound
        for compound in network.compounds
        if isinstance(compound.model, MultiGridCompoundModel)
    ]
    for comp in relevant_compounds:
        network.remove_compound(comp.id)
        if "heat_return_node_id" in comp.connected_to:
            heat_return_node = network.node_by_id(
                comp.connected_to["heat_return_node_id"]
            )
            heat_node = network.node_by_id(comp.connected_to["heat_node_id"])
            network.branch(WaterPipe(0, 0), heat_return_node.id, heat_node.id)

    for branch in network.branches:
        if isinstance(branch.model, MultiGridBranchModel):
            network.remove_branch(branch.id)


def find_ignored_nodes(network: Network, islanding_config=None):  # NOSONAR
    """Return node IDs to exclude from the solve.

    Default: only ExtPowerGrid/ExtHydrGrid children are "leading". With
    *islanding_config*: any :class:`GridFormingMixin` child also counts as
    leading for an islanding-enabled carrier.

    Both cases use the real topology (active branches, on_off not fixed to 0;
    switchable on_off Vars stay connected). A component with no leading child
    can never be energised - the connectivity arcs over its severed branches
    are capped at zero - so keeping it in the solve only leaves unservable
    loads (e.g. an optimization problem's priority floor) that render the
    model infeasible.
    """
    ignored_nodes = set()
    pruning_rules: dict = {}
    network._pruning_rules = pruning_rules
    without_cps = network.copy()
    remove_cps(without_cps)

    topology = generate_real_topology(without_cps._network_internal)

    components = nx.connected_components(topology)
    for component in components:
        component_leading = False
        for node_id in component:
            int_node: Node = topology.nodes[node_id]["internal_node"]
            for child_id in int_node.child_ids:
                child = without_cps.child_by_id(child_id)
                if not child.active:
                    continue
                if isinstance(child.model, ExtPowerGrid | ExtHydrGrid):
                    component_leading = True
                    break
                # With islanding enabled, any GridFormingMixin child also leads.
                if islanding_config is not None and isinstance(
                    child.model, GridFormingMixin
                ):
                    node_grid = int_node.grid
                    carrier_enabled = (
                        (
                            isinstance(node_grid, PowerGrid)
                            and islanding_config.electricity is not None
                        )
                        or (
                            isinstance(node_grid, GasGrid)
                            and islanding_config.gas is not None
                        )
                        or (
                            isinstance(node_grid, WaterGrid)
                            and islanding_config.water is not None
                        )
                    )
                    if carrier_enabled:
                        component_leading = True
                        break
            if component_leading:
                break
        if not component_leading:
            ignored_nodes.update(component)
            for node_id in component:
                pruning_rules[node_id] = _PRUNE_RULE_NO_LEAD

    # Leaf-stub pruning: a node with no active children and degree <= 1 in the
    # remaining active topology is a dead-end pump-target (infeasible LP).
    # Iterate to fixed point. Runs under islanding too: e-variables gate
    # loads and angles/pressures, not the mixing/balance equations that make
    # a childless dead-end junction infeasible, and the topology is the real
    # one in both cases.

    # Compound port children (e.g. SubHG on a CHPHG heat node) carry no
    # demand of their own; they must not keep an otherwise-isolated
    # junction alive.
    compound_port_child_ids = {
        sub.id
        for compound in network.compounds
        for sub in compound.subcomponents
        if isinstance(sub, Child)
    }

    # Attachment ports of active multi-grid compounds: remove_cps strips
    # the compound's transfer branches from the pruning copy, so these
    # nodes look like childless dead-ends here - but in the real solve the
    # compound re-attaches and carries flow through them. Ports whose grid
    # connection is genuinely gone are still caught by the
    # connected-component check above.
    compound_attachment_node_ids = {
        node_id
        for compound in network.compounds
        if compound.active and isinstance(compound.model, MultiGridCompoundModel)
        for node_id in compound.connected_to.values()
    }

    def _has_real_active_child(int_node):
        for cid in int_node.child_ids:
            if cid in compound_port_child_ids:
                continue
            if without_cps.child_by_id(cid).active:
                return True
        return False

    def _has_mass_flow_anchor(int_node):
        """A Junction at degree <= 1 needs a mass-flow-contributing child
        (Sink / Source / ExtHydrGrid) to anchor mass conservation.
        Heat-only children (HeatLoad / HeatGenerator) don't qualify -
        with no outgoing pipe their heat_mw term has no enthalpy
        stream to balance against and the junction becomes infeasible."""
        for cid in int_node.child_ids:
            if cid in compound_port_child_ids:
                continue
            child = without_cps.child_by_id(cid)
            if not child.active:
                continue
            if "mass_flow_kgs" in getattr(child.model, "vars", {}):
                return True
        return False

    while True:
        new_stubs = set()
        for node_id in topology.nodes:
            if node_id in ignored_nodes:
                continue
            if node_id in compound_attachment_node_ids:
                continue
            int_node: Node = topology.nodes[node_id]["internal_node"]
            active_degree = sum(
                1 for nb in topology.neighbors(node_id) if nb not in ignored_nodes
            )
            if active_degree > 1:
                continue
            # Classical leaf stub: no real active children at all.
            if not _has_real_active_child(int_node):
                new_stubs.add(node_id)
                pruning_rules[node_id] = _PRUNE_RULE_STUB
                continue
            # Mass-balance dead-end: Junction at degree <= 1 whose only
            # children are heat-only (heat_mw) - see _has_mass_flow_anchor.
            if isinstance(int_node.model, Junction) and not _has_mass_flow_anchor(
                int_node
            ):
                new_stubs.add(node_id)
                pruning_rules[node_id] = _PRUNE_RULE_HEAT_ONLY
        if not new_stubs:
            break
        ignored_nodes.update(new_stubs)

    return ignored_nodes


# ---------------------------------------------------------------------------
# Inter-step state carriers
# ---------------------------------------------------------------------------
# StepState (sequential timeseries, plain floats) and PeriodState (multi-period,
# live solver vars) live here - in the solver's low-level core - rather than in
# the simulation layer, because both solver backends accept a ``step_state`` and
# must reference these types.  Hosting them here keeps the dependency direction
# one-way (simulation -> solver) and avoids a circular import; the historical
# import path ``monee.simulation.step_state`` re-exports them unchanged.


def _collect_id_matches(net, component_id):
    candidates = []
    for node in net.nodes:
        if node.id == component_id:
            candidates.append(node.model)
        for child in net.childs_by_ids(node.child_ids):
            if child.id == component_id:
                candidates.append(child.model)
    for branch in net.branches:
        if branch.id == component_id:
            candidates.append(branch.model)
    for compound in net.compounds:
        if compound.id == component_id:
            candidates.append(compound.model)
    return candidates


def _find_model(net, component_id, attr=None):
    """Return the model for *component_id*. Disambiguates node/child id collisions
    by preferring a model that actually carries *attr*."""
    candidates = _collect_id_matches(net, component_id)

    if not candidates:
        return None
    if attr is not None and len(candidates) > 1:
        with_attr = [m for m in candidates if hasattr(m, attr)]
        if with_attr:
            return with_attr[0]
    return candidates[0]


def _extract_value(val):
    """Return a plain float from *val* (Var, Intermediate, float, int, or None)."""
    if val is None:
        return None

    if isinstance(val, (Var, Intermediate, PostProcess)):
        return val.value
    if isinstance(val, (int, float)):
        return float(val)
    return val


class InterStepState(ABC):
    """Abstract base. ``step``: negative = relative to current, non-negative = absolute."""

    dt_h: float = 1.0

    @abstractmethod
    def get(self, component_id, attr: str, step: int = -1):
        """Float (StepState) / live var (PeriodState), or None if no data."""

    def has(self, component_id, attr: str) -> bool:
        return self.get(component_id, attr) is not None


class StepState(InterStepState):
    """All previously-solved network copies (sequential timeseries). ``get`` returns floats.

    ``initial_state`` falls back when no prior solve has written the attribute.
    ``max_steps`` caps how many solved networks are retained (``None`` =
    unlimited): each network is a full copy, so an open-ended run (e.g. a
    :class:`~monee.simulation.Stepper` paced by a co-simulation framework)
    would otherwise grow memory without bound. Each push may carry an explicit
    step number (so drivers that skip failed steps keep absolute indices
    aligned with their own step numbering); absolute ``step`` indices keep
    their meaning when old networks are dropped. Reading a dropped or
    never-solved step falls back to ``initial_state``."""

    def __init__(
        self, initial_state: dict | None = None, max_steps: int | None = None
    ) -> None:
        if max_steps is not None and max_steps < 1:
            raise ValueError(f"max_steps must be >= 1 or None, got {max_steps}")
        self._networks: list = []
        self._step_numbers: list[int] = []
        # step number -> absolute position (self._dropped + list index) of its
        # first occurrence, for O(1) absolute lookups.
        self._pos_by_step: dict[int, int] = {}
        self._dropped: int = 0
        self._max_steps = max_steps
        self.dt_h: float = 1.0
        self._initial_state: dict = dict(initial_state) if initial_state else {}

    def push(self, net, step: int | None = None) -> None:
        """Append a solved network; ``step`` records the caller's step number
        for absolute lookups (defaults to the push count)."""
        pos = self._dropped + len(self._networks)
        if step is None:
            step = pos
        self._networks.append(net)
        self._step_numbers.append(step)
        self._pos_by_step.setdefault(step, pos)
        if self._max_steps is not None and len(self._networks) > self._max_steps:
            dropped_step = self._step_numbers[0]
            del self._networks[0]
            del self._step_numbers[0]
            self._dropped += 1
            if self._pos_by_step.get(dropped_step) == self._dropped - 1:
                del self._pos_by_step[dropped_step]
                # Re-point a duplicate step number to its next occurrence so
                # first-occurrence lookup semantics survive the drop.
                try:
                    i = self._step_numbers.index(dropped_step)
                except ValueError:
                    pass
                else:
                    self._pos_by_step[dropped_step] = self._dropped + i

    def _network_for_step(self, step: int):
        if not self._networks:
            return None
        if step < 0:
            return self._networks[step] if -step <= len(self._networks) else None
        # Absolute lookup by recorded step number; dropped/missing -> fallback.
        pos = self._pos_by_step.get(step)
        if pos is None:
            return None
        return self._networks[pos - self._dropped]

    def get(self, component_id, attr: str, step: int = -1):
        net = self._network_for_step(step)
        if net is not None:
            model = _find_model(net, component_id, attr)
            if model is not None:
                val = _extract_value(getattr(model, attr, None))
                if val is not None:
                    return val
        return self._initial_state.get((component_id, attr))

    def __len__(self) -> int:
        """Logical number of solved steps (including dropped ones)."""
        return self._dropped + len(self._networks)

    def __repr__(self) -> str:
        n = len(self)
        return f"StepState({n} step{'s' if n != 1 else ''})"


class PeriodState(InterStepState):
    """All period networks (multi-period). ``get`` returns live solver vars after
    injection, so reading from another period becomes an algebraic cross-period
    constraint. Both past and future absolute indices are accessible.

    .. note::
       Absolute (non-negative) ``step`` indices are **window-local** under
       :func:`~monee.simulation.multi_period.run_mpc`: ``self._networks`` is only
       the current rolling window's networks, so ``step=0`` means the first
       period of *this window*, not the global start of the run. This differs
       from :class:`StepState`, whose absolute indices stay globally meaningful
       via its dropped-prefix bookkeeping. Relative (negative) lookbacks behave
       identically in both. Prefer relative indices in ``inter_step_equations``
       if you need MPC-window-independent behaviour.
    """

    def __init__(
        self,
        networks: list,
        current_t: int,
        dt_h: float = 1.0,
        initial_state: dict | None = None,
    ) -> None:
        self._networks = networks
        self.current_t = current_t
        self.dt_h = dt_h
        self._initial_state: dict = initial_state or {}

    @property
    def T(self) -> int:
        return len(self._networks)

    def get(self, component_id, attr: str, step: int = -1):
        actual_t = (self.current_t + step) if step < 0 else step

        if actual_t < 0:
            key = (component_id, attr)
            if key in self._initial_state:
                return self._initial_state[key]
            return None

        try:
            net = self._networks[actual_t]
        except IndexError:
            return None
        model = _find_model(net, component_id, attr)
        if model is None:
            return None
        return getattr(model, attr, None)

    def __repr__(self) -> str:
        return f"PeriodState(T={self.T}, current_t={self.current_t}, dt_h={self.dt_h})"

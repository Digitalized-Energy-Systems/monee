import logging
import math
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
        parts.append(f"{col} ∈ {s}" if "[" in s else f"{col} = {s}")
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


@dataclass
class SolverResult:
    """Outcome of a single solve. ``objective=0.0`` for plain energy flow;
    ``None`` when not meaningful (e.g. ``MultiPeriodResult.get_period_result``).

    Termination metadata
    --------------------
    ``solver_status`` and ``termination_condition`` mirror the Pyomo
    ``SolverStatus`` / ``TerminationCondition`` strings (or ``None`` if
    not populated by the backend). They let downstream consumers
    distinguish a converged-optimal solution from a *witness* incumbent
    Gurobi returns when it hits a time limit — the witness has
    ``success=True`` and looks identical otherwise. Callers that care
    about convergence (e.g. the MC resilience pipeline, which must drop
    aborted samples rather than averaging in a non-converged shed value)
    can inspect ``termination_condition`` to detect the time-limit case.
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

    def summary(self):
        return repr(self)

    def full(self):
        return self.network.as_result_dataframe_dict_str()

    def get(self, model_type) -> pandas.DataFrame:
        """Result DataFrame for *model_type* (typo-safe vs. dict access)."""
        return self.dataframes.get(model_type.__name__, pandas.DataFrame())

    def __getitem__(self, component_id) -> pandas.Series:
        """Return the result row matching *component_id*. Raises KeyError if missing."""
        for df in self.dataframes.values():
            if "id" in df.columns:
                try:
                    mask = df["id"] == component_id
                except (ValueError, TypeError):
                    # Tuple branch IDs vs scalar id column triggers a broadcasting error.
                    mask = df["id"].apply(lambda x: x == component_id)
                if mask.any():
                    return df[mask].iloc[0]
        raise KeyError(component_id)

    def __repr__(self) -> str:
        SEP = "─" * 68
        title = "SolverResult"
        if self.objective is not None and abs(self.objective) > 0.0:
            title += f"  (objective = {self.objective:.6g})"
        lines = [title, SEP]
        for type_name, df in self.dataframes.items():
            n = len(df)
            vis = _display_df(df).drop(columns=["id", "node_id"], errors="ignore")
            num = vis.select_dtypes(include="number")
            parts = _summarize_numeric_cols(num)
            row = f"  {type_name:<22} {n:>2}"
            if parts:
                row += "  │  " + "  ·  ".join(parts[:4])
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
        title = "SolverResult"
        if self.objective is not None and abs(self.objective) > 0.0:
            title += f"  (objective = {self.objective:.6g})"
        SEP = "─" * 68
        lines = [title]
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
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        obj_extra = ""
        if self.objective is not None and abs(self.objective) > 0.0:
            obj_extra = (
                f" &nbsp;<span style='color:#888;font-weight:normal'>"
                f"objective = {self.objective:.6g}</span>"
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
    """Abstract base class for solver backends (GEKKO, Pyomo, …)."""

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
            simulation: If ``True``, square the model (pin phantom vars, drop
                operational flow limits) and solve it as a steady-state
                simulation. GEKKO runs this as IMODE=1 (falling back to IMODE=3
                if not square); backends without a simulation mode ignore it.
            formulation: Solve-time formulation override - a registry key
                string (``"smooth_nlp"``, ``"convex_miqcqp"``, …), a
                :class:`~monee.model.formulation.core.NetworkFormulation`, or a
                sequence of either (merged left to right). Overrides the
                network-level ``apply_formulation`` choice; components without
                any choice fall back to ``DEFAULT_SIMULATION_FORMULATION``.

        Returns:
            A :class:`SolverResult` with updated variable values and result DataFrames.
        """

    @abstractmethod
    def _add_equations(self, solver_obj, eqs):
        """Register a list of equations/constraints with the backend solver object."""

    def init_branches(self, branches):
        for branch in branches:
            branch.model.init(branch.grid)

    @staticmethod
    def mark_temporal_components(network, ignored_nodes: set) -> None:  # NOSONAR
        """Set ``_temporal_active`` on every model carrying a temporal method,
        so static-only constraints can be suppressed when coupling is active."""
        _temporal_methods = frozenset(
            (
                "inter_temporal_equations",
                "inter_step_equations",
                "inter_period_equations",
            )
        )
        for node in network.nodes:
            if node.id in ignored_nodes or node.ignored:
                continue
            if any(hasattr(node.model, m) for m in _temporal_methods):
                node.model._temporal_active = True
            for child in network.childs_by_ids(node.child_ids):
                if child.ignored:
                    continue
                if any(hasattr(child.model, m) for m in _temporal_methods):
                    child.model._temporal_active = True
        for branch in network.branches:
            if branch.ignored:
                continue
            if any(hasattr(branch.model, m) for m in _temporal_methods):
                branch.model._temporal_active = True
        for compound in network.compounds:
            if compound.ignored:
                continue
            if any(hasattr(compound.model, m) for m in _temporal_methods):
                compound.model._temporal_active = True

    def _collect_temporal_eqs(  # NOSONAR
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
        for node in nodes:
            if ignore_node(node, network, ignored_nodes):
                continue
            for method in methods:
                if hasattr(node.model, method):
                    eqs = as_iter(getattr(node.model, method)(state, node.id))
                    self._add_equations(solver_obj, filter_intermediate_eqs(eqs))
                if node.formulation is not None and hasattr(node.formulation, method):
                    eqs = as_iter(
                        getattr(node.formulation, method)(node.model, state, node.id)
                    )
                    self._add_equations(solver_obj, filter_intermediate_eqs(eqs))
            for child in network.childs_by_ids(node.child_ids):
                if ignore_child(child, ignored_nodes):
                    continue
                for method in methods:
                    if hasattr(child.model, method):
                        eqs = as_iter(getattr(child.model, method)(state, child.id))
                        self._add_equations(solver_obj, filter_intermediate_eqs(eqs))
                    if child.formulation is not None and hasattr(
                        child.formulation, method
                    ):
                        eqs = as_iter(
                            getattr(child.formulation, method)(
                                child.model, state, child.id
                            )
                        )
                        self._add_equations(solver_obj, filter_intermediate_eqs(eqs))
        for branch in branches:
            if ignore_branch(branch, network, ignored_nodes):
                continue
            for method in methods:
                if hasattr(branch.model, method):
                    eqs = as_iter(getattr(branch.model, method)(state, branch.id))
                    self._add_equations(solver_obj, filter_intermediate_eqs(eqs))
                if branch.formulation is not None and hasattr(
                    branch.formulation, method
                ):
                    eqs = as_iter(
                        getattr(branch.formulation, method)(
                            branch.model, state, branch.id
                        )
                    )
                    self._add_equations(solver_obj, filter_intermediate_eqs(eqs))
        for compound in compounds:
            if ignore_compound(compound, ignored_nodes):
                continue
            for method in methods:
                if hasattr(compound.model, method):
                    eqs = as_iter(getattr(compound.model, method)(state, compound.id))
                    self._add_equations(solver_obj, filter_intermediate_eqs(eqs))
                if compound.formulation is not None and hasattr(
                    compound.formulation, method
                ):
                    eqs = as_iter(
                        getattr(compound.formulation, method)(
                            compound.model, state, compound.id
                        )
                    )
                    self._add_equations(solver_obj, filter_intermediate_eqs(eqs))

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


def filter_intermediate_eqs(eqs):
    return [eq for eq in eqs if type(eq) is not IntermediateEq]


def filter_bool_eqs(eqs, context=None):
    """Drop Python-``bool`` sentinel equations, identically for both backends.

    An ``equations()`` body can collapse a constraint to a Python ``bool`` when
    a component is over-deactivated (e.g. an attribute overwritten by a
    :class:`Const` so ``a == b`` evaluates eagerly):

    * ``True``  → tautology, a no-op; dropped silently.
    * ``False`` → structurally infeasible (contradictory) constraint. Feeding it
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
                _log.warning(
                    "Dropping always-false (structurally infeasible) equation%s; "
                    "this usually means a node/branch was over-deactivated.",
                    f" in {context}" if context is not None else "",
                )
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
        obj = None
        for objective in network.objectives:
            if obj is not None:
                obj = obj + objective(network)
            else:
                obj = objective(network)
        if obj is not None:
            m.Obj(obj)

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
        obj = None
        for objective in (
            optimization_problem.objectives.all(network, period_index=period_index)
            if optimization_problem.objectives is not None
            else []
        ):
            if obj is not None:
                obj = obj + objective
            else:
                obj = objective
        if obj is not None:
            m.Obj(obj)

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
                    as_iter(child.equations(grid, node.model)),
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
    """Replace Var/Const fields with NaN placeholders; zero regulation."""
    for key, value in target.__dict__.items():
        if isinstance(value, Const):
            setattr(target, key, Const(float("nan")))
        if isinstance(value, Var):
            setattr(
                target,
                key,
                Var(float("nan"), max=value.max, min=value.min, name=value.name),
            )
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


def _carrier_islands(network: Network, node_ids: set):
    """Connected components of the active-pipe subgraph restricted to *node_ids*
    (pipes only ever join same-carrier nodes, so each component is one island)."""
    g = nx.Graph()
    g.add_nodes_from(node_ids)
    for branch in network.branches:
        if (
            branch.active
            and branch.from_node_id in node_ids
            and branch.to_node_id in node_ids
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
        for island in _carrier_islands(network, ids):
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

    ``category`` ∈ {``branch``, ``node``, ``child``, ``compound``}.
    """
    for branch in branches:
        if ignore_branch(branch, network, ignored_nodes):
            branch.ignored = True
            inject_nans(branch.model)
            continue
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
    for branch in branches:
        apply_post_process(branch.model)
    for node in nodes:
        apply_post_process(node.model)
        for child in network.childs_by_ids(node.child_ids):
            apply_post_process(child.model)
    for compound in compounds:
        apply_post_process(compound.model)


def _copy_var_values(src, dst) -> None:
    """Copy ``.value`` for Var/Intermediate from *src* to *dst*. Intermediates
    must be propagated so e.g. derived ``vm_pu`` isn't silently lost."""
    for key, val in src.__dict__.items():
        if isinstance(val, (Var, Intermediate)):
            dst_attr = dst.__dict__.get(key)
            if isinstance(dst_attr, (Var, Intermediate)):
                dst_attr.value = val.value


def persist_solution(solved_copy: Network, original: Network) -> None:
    """Propagate solved values back so the next ``inject_vars`` warm-starts."""
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


def compute_bound_violations(  # NOSONAR
    nodes, branches, compounds, network, tol: float = 1e-6
) -> dict[str, float]:
    """``{"<Type>.<id>.<attr>": magnitude}`` for Var.value violations beyond *tol*."""
    violations: dict[str, float] = {}

    def _check(model, label: str) -> None:
        for key, val in model.__dict__.items():
            if not isinstance(val, Var):
                continue
            v = val.value
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            if val.min is not None and v < val.min - tol:
                violations[f"{label}.{key}"] = val.min - v
            elif val.max is not None and v > val.max + tol:
                violations[f"{label}.{key}"] = v - val.max

    for branch in branches:
        _check(branch.model, f"{type(branch.model).__name__}.{branch.id}")
    for node in nodes:
        _check(node.model, f"{type(node.model).__name__}.{node.id}")
        for child in network.childs_by_ids(node.child_ids):
            _check(child.model, f"{type(child.model).__name__}.{child.id}")
    for compound in compounds:
        _check(compound.model, f"{type(compound.model).__name__}.{compound.id}")

    return violations


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

    # Leaf-stub pruning: a node with no active children and degree ≤ 1 in the
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
        """A Junction at degree ≤ 1 needs a mass-flow-contributing child
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
                continue
            # Mass-balance dead-end: Junction at degree ≤ 1 whose only
            # children are heat-only (heat_mw) - see _has_mass_flow_anchor.
            if isinstance(int_node.model, Junction) and not _has_mass_flow_anchor(
                int_node
            ):
                new_stubs.add(node_id)
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
        # Absolute lookup by recorded step number; dropped/missing → fallback.
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

"""Multi-period optimization: build a single problem spanning T periods and
solve in one shot. :class:`PeriodState` exposes live solver variables behind
the same API as :class:`StepState`, so ``inter_temporal_equations`` (the hook
that runs in both timeseries and multi-period) works unchanged."""

from __future__ import annotations

import logging

import pandas

from monee.model import Network
from monee.model.core import Var
from monee.model.extension.islanding.core import NetworkIslandingConfig
from monee.model.formulation.registry import attach_formulations
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
    wrap_result_html as _wrap_result_html,
)
from monee.simulation.step_state import PeriodState
from monee.simulation.timeseries import (
    _SERIES_ATTRS,
    TimeseriesData,
    _dt_h_at_step,
    _resolve_steps,
)

# Shared result-rendering helpers, imported from the solver's public reporting
# surface (the simulation layer renders the same kind of result tables).
from monee.solver.core import (
    SolverResult,
    _find_model,
    apply_child_overwrites,
    find_ignored_nodes,
    inject_vars,
    mark_ignored_components,
    mark_slacks_and_prescriptions,
    withdraw_vars,
)
from monee.solver.core import col_summary as _col_summary
from monee.solver.core import display_df as _display_df
from monee.solver.dispatch import GEKKO_SOLVERS, resolve_multi_period_solver

_log = logging.getLogger(__name__)


def _prepare_period(
    network: Network,
    timeseries_data: TimeseriesData | None,
    t: int,
    optimization_problem,
    formulation=None,
) -> tuple[Network, set]:
    """Copy net, apply timeseries for *t*, run extension prepare(), compute
    ignored nodes. Returns ``(net_t, ignored_nodes)`` ready for injection."""
    net_t = network.copy()

    if timeseries_data is not None:
        timeseries_data.apply_to_network(net_t, t)

    # Same stamp prepare_solve_network sets: extension prepare() hooks gated
    # on it (islanding injection gating / energisation objective) must not
    # activate when the optimization problem brings its own shedding vars.
    net_t._solve_has_optimization_problem = optimization_problem is not None
    for ext in net_t.extensions:
        ext.prepare(net_t)

    # Attach formulations and declare their vars on the period copy (same
    # position as the single-period solvers: after prepare, before _apply).
    attach_formulations(net_t, formulation)

    islanding_config = next(
        (e for e in net_t.extensions if isinstance(e, NetworkIslandingConfig)),
        None,
    )

    ignored_nodes: set = set()
    if optimization_problem is None:
        ignored_nodes = find_ignored_nodes(net_t, islanding_config)
        if ignored_nodes:
            mark_ignored_components(net_t, ignored_nodes)

    if optimization_problem is not None:
        optimization_problem._apply(net_t)

    apply_child_overwrites(net_t, net_t.nodes, ignored_nodes)

    # Same backend-agnostic marking trio as the single-period solvers (pin
    # floating hydraulic gauges, mark heat-balance slacks, decide the dynamic
    # HE flow prescription); must run before var injection - the checks rely
    # on monee Var instances.
    mark_slacks_and_prescriptions(net_t, ignored_nodes)

    return net_t, ignored_nodes


def _find_component_var(net_t: Network, comp_id, attr: str):
    """Return ``comp.model.attr`` for *comp_id* in *net_t*, or None. Resolves the
    id through :func:`_find_model` so node/child id collisions pick the component
    that actually carries *attr* - the same disambiguation the cross-period
    coupling uses via :class:`PeriodState`."""
    model = _find_model(net_t, comp_id, attr)
    if model is None:
        return None
    return getattr(model, attr, None)


def _extract_terminal_state(net_t: Network) -> dict:
    """``{(comp_id, attr): value}`` for all Var/numeric attributes; used by
    :func:`run_mpc` to seed the next horizon's ``initial_state``."""
    state: dict = {}

    def _scan(comp_id, model):
        for k, v in model.__dict__.items():
            if isinstance(v, Var):
                state[(comp_id, k)] = v.value
            elif isinstance(v, (int, float)):
                state[(comp_id, k)] = v

    for comp in net_t.iter_all_components():
        _scan(comp.id, comp.model)
    return state


def _validate_state_keys(state: dict | None, net: Network, label: str) -> None:
    """Raise if any (comp_id, attr) key isn't found. ``label`` names the source."""
    if not state:
        return
    for comp_id, attr in state:
        var = _find_component_var(net, comp_id, attr)
        if var is None:
            raise ValueError(
                f"{label}: component {comp_id!r} attribute {attr!r} not found "
                f"in the network.  Check the component id and attribute name."
            )


def _slice_timeseries(td: TimeseriesData, start: int, length: int) -> TimeseriesData:
    """Return a TimeseriesData sliced to ``[start, start+length)``."""
    end = start + length

    def _slice_dict(d: dict) -> dict:
        # Normalize to a list before slicing: a pandas Series keeps its original
        # integer labels under ``series[start:end]``, so the later positional
        # read ``series[timestep]`` (timestep is 0-based within the window) would
        # be label-based and read the wrong row / raise KeyError. ``list(...)``
        # makes both lists and Series slice positionally and consistently.
        return {
            comp_id: {attr: list(series)[start:end] for attr, series in attrs.items()}
            for comp_id, attrs in d.items()
        }

    new_td = TimeseriesData()
    for attr in _SERIES_ATTRS:
        setattr(new_td, attr, _slice_dict(getattr(td, attr)))
    new_td._length = length
    return new_td


class MultiPeriodResult:
    """
    Holds the outcome of a multi-period optimization.

    One :class:`~monee.solver.core.SolverResult`-compatible network copy per
    period, all solved in a single solver invocation.

    Attributes:
        objective: Global objective value at the solution.
        success: ``True`` if the solver reported a feasible solution.
    """

    def __init__(
        self,
        net_copies: list[Network],
        objective: float,
        success: bool,
        datetime_index: pandas.DatetimeIndex | None = None,
        backend_used: str | None = None,
        solver_used: str | None = None,
    ) -> None:
        self._net_copies = net_copies
        self.objective = objective
        self.success = success
        self._datetime_index = datetime_index
        #: Backend and solver convention selected for this multi-period solve
        #: (see :func:`monee.solver.dispatch.resolve_multi_period_solver`).
        self.backend_used = backend_used
        self.solver_used = solver_used
        # Build per-period DataFrames once; queried repeatedly by get_result_for.
        self._period_dfs: list[dict[str, pandas.DataFrame]] = [
            net_t.as_result_dataframe_dict() for net_t in net_copies
        ]

    @property
    def T(self) -> int:  # NOSONAR
        return len(self._net_copies)

    def _make_index(self, _labels=None) -> pandas.Index:
        # The full-period index is used regardless of *_labels*: every period
        # is expected to carry every component, so partial matches surface as a
        # length mismatch rather than being silently reindexed.
        if self._datetime_index is not None:
            return self._datetime_index[: self.T]
        return pandas.RangeIndex(self.T)

    def _frames(self) -> list[tuple[int, dict]]:
        return list(enumerate(self._period_dfs))

    def get_result_for(self, model_type, attribute: str) -> pandas.DataFrame:
        """DataFrame of *attribute*: rows=periods, cols=component ids.
        Raises ``KeyError`` for an unknown model type or attribute."""
        return _build_attribute_frame(
            self._frames(), model_type, attribute, self._make_index
        )

    def get_result_for_id(self, component_id, attribute: str) -> pandas.Series:
        """Series of *attribute* for *component_id* across all periods."""
        return _build_id_series(
            self._frames(), component_id, attribute, self._make_index
        )

    def __getitem__(self, component_id) -> pandas.DataFrame:
        """All result attributes for *component_id*, one row per period."""
        return _build_component_frame(self._frames(), component_id, self._make_index)

    def get_period_result(self, t: int) -> SolverResult:
        """SolverResult for period *t* (``objective`` is None; only the global
        ``MultiPeriodResult.objective`` is tracked)."""
        return SolverResult(
            self._net_copies[t],
            self._period_dfs[t],
            None,
            self.success,
            backend_used=self.backend_used,
            solver_used=self.solver_used,
        )

    def _temporal_lines(self) -> list[str]:  # NOSONAR
        """Compact per-period evolution lines for attributes that vary across
        periods. At most 2 attrs per type."""
        lines = []
        MAX_VALS = 6  # show at most this many period values inline

        for type_name, dfs in self._collect_type_dfs().items():
            df0 = dfs[0]
            num_cols = (
                _display_df(df0)
                .drop(columns=["id", "node_id"], errors="ignore")
                .select_dtypes(include="number")
                .columns.tolist()
            )
            shown = 0
            for col in num_cols:
                if shown >= 2:
                    break
                # Collect per-period mean values for this attribute
                vals = []
                for df in dfs:
                    if col not in df.columns:
                        break
                    nums = df[col].dropna()
                    if nums.empty:
                        break
                    vals.append(float(nums.mean()))
                if len(vals) < len(dfs):
                    continue
                # Only show if values actually vary across periods
                spread = max(vals) - min(vals)
                if spread < 1e-6 * (abs(max(vals)) + 1e-10):
                    continue
                if len(vals) <= MAX_VALS:
                    val_str = "  ".join(f"{v:.3g}" for v in vals)
                else:
                    val_str = (
                        "  ".join(f"{v:.3g}" for v in vals[:3])
                        + "  …  "
                        + f"{vals[-1]:.3g}"
                    )
                lines.append(f"    {type_name}.{col}: [{val_str}]")
                shown += 1
        return lines

    def _collect_type_dfs(self) -> dict[str, list[pandas.DataFrame]]:
        type_dfs: dict[str, list] = {}
        for dfs in self._period_dfs:
            for type_name, df in dfs.items():
                type_dfs.setdefault(type_name, []).append(df)
        return type_dfs

    def _repr_type_row(self, type_name, df) -> str:
        all_dfs = [dfs.get(type_name, pandas.DataFrame()) for dfs in self._period_dfs]
        combined = pandas.concat(all_dfs, ignore_index=True)
        vis = _display_df(combined).drop(columns=["id", "node_id"], errors="ignore")
        num = vis.select_dtypes(include="number")
        parts = []
        for col in num.columns:
            s = _col_summary(num[col])
            if s:
                parts.append(f"{col} ∈ {s}" if "[" in s else f"{col} = {s}")
        row = f"  {type_name:<22} ×{len(df):>2}"
        if parts:
            row += "  │  " + "  ·  ".join(parts[:4])
        return row

    def __repr__(self) -> str:
        SEP = "─" * 68
        status = "ok" if self.success else "FAILED"
        lines = [
            f"MultiPeriodResult  T={self.T}  obj={self.objective:.4g}  [{status}]",
            SEP,
        ]
        if self._period_dfs:
            for type_name, df in self._period_dfs[0].items():
                lines.append(self._repr_type_row(type_name, df))

        # Temporal evolution section - only shown when there are varying attrs
        temporal = self._temporal_lines()
        if temporal:
            lines.append(SEP)
            lines.append("  Temporal evolution (mean over components):")
            lines.extend(temporal)

        lines.append(SEP)
        return "\n".join(lines)

    def _repr_html_(self) -> str:  # NOSONAR
        status_color = "#090" if self.success else "#c00"
        status_text = "ok" if self.success else "failed"
        sections = []
        if self._period_dfs:
            sections = _build_type_stats_html(self._collect_type_dfs(), "period")
        return _wrap_result_html(
            "MultiPeriodResult",
            f"<span style='font-weight:normal;color:#555'>T={self.T} &nbsp;·&nbsp; "
            f"obj={self.objective:.4g} &nbsp;·&nbsp; "
            f"<span style='color:{status_color}'>{status_text}</span></span>",
            sections,
        )


def _assemble_two_pass(
    m,
    single,
    label: str,
    network: Network,
    timeseries_data: TimeseriesData | None,
    steps: int,
    optimization_problem,
    dt_h_list: list[float],
    initial_state: dict | None,
    terminal_state: dict | None,
    formulation,
    inject,
    process_branches,
    sink_objective,
    add_equations,
    add_terminal,
) -> list[Network]:
    """Shared two-pass assembly for the multi-period backends: pass 1 prepares
    per-period network copies and injects backend variables, pass 2 builds
    equations with a :class:`PeriodState` spanning all periods. Backend
    differences are confined to the callbacks: ``inject(net_t, ignored_t, t)``,
    ``process_branches(net_t, ignored_t) -> ctx`` (may collect objective
    expressions), ``sink_objective(ctx)`` (called between the OXF components and
    the inter-period equations), ``add_equations(eqs)`` and
    ``add_terminal(var, target)``."""
    _log.info("Multi-period %s solve: T=%d periods", label, steps)

    # Pass 1: prepare networks and inject variables for all periods.
    net_copies: list[Network] = []
    ignored_list: list[set] = []

    for t in range(steps):
        _log.debug("Preparing period %d/%d", t + 1, steps)
        net_t, ignored_t = _prepare_period(
            network, timeseries_data, t, optimization_problem, formulation
        )
        inject(net_t, ignored_t, t)
        for ext in net_t.extensions:
            ext.activate_timeseries(net_t, ignored_t)
        single.mark_temporal_components(net_t, ignored_t)
        net_copies.append(net_t)
        ignored_list.append(ignored_t)

    # Pass 2: build per-period equations.
    _log.debug("Assembling equations for %d periods", steps)
    for t in range(steps):
        net_t = net_copies[t]
        ignored_t = ignored_list[t]

        period_state = PeriodState(
            net_copies,
            current_t=t,
            dt_h=dt_h_list[t],
            initial_state=initial_state,
        )

        single.init_branches(net_t.branches)
        single.process_equations_nodes_childs(m, net_t, net_t.nodes, ignored_t)
        branch_ctx = process_branches(net_t, ignored_t)
        single.process_equations_compounds(m, net_t, net_t.compounds, ignored_t)

        if optimization_problem is not None:
            single.process_oxf_components(
                m, net_t, optimization_problem, period_index=t
            )
        else:
            single.process_internal_oxf_components(m, net_t)

        sink_objective(branch_ctx)

        single.process_inter_period_equations(
            m,
            net_t,
            net_t.nodes,
            net_t.branches,
            net_t.compounds,
            ignored_t,
            period_state,
            optimization_problem=optimization_problem,
            period_index=t,
        )

        for ext in net_t.extensions:
            add_equations(ext.inter_period_equations(net_t, ignored_t, period_state))
            add_equations(ext.inter_temporal_equations(net_t, ignored_t, period_state))
            add_equations(ext.equations(net_t, ignored_t))

        if terminal_state and t == steps - 1:
            for (comp_id, attr), target in terminal_state.items():
                var = _find_component_var(net_t, comp_id, attr)
                if isinstance(var, (int, float)):
                    # A plain constant would make ``var == target`` a Python bool
                    # (a cryptic backend error or a silent tautology); only a
                    # solver variable / expression can be pinned.
                    _log.warning(
                        "terminal_state target (%r, %r) is a constant, not a "
                        "solver variable; the terminal constraint was skipped. "
                        "Make the attribute controllable to pin it.",
                        comp_id,
                        attr,
                    )
                elif var is not None:
                    add_terminal(var, target)

    return net_copies


# GEKKO multi-period solver


class GekkoMultiPeriodSolver:
    """Multi-period optimizer on GEKKO/IPOPT. Two-pass: inject vars for all T
    periods, then assemble equations with a :class:`PeriodState` that sees all
    periods so ``inter_temporal_equations`` / ``inter_period_equations`` can
    couple them freely."""

    def __init__(self, solver: int = 1):
        self._solver_int = solver
        self._backend_name = "gekko"
        self._solver_name = {v: k for k, v in GEKKO_SOLVERS.items()}.get(
            solver, str(solver)
        )

    def solve_multi_period(
        self,
        network: Network,
        timeseries_data: TimeseriesData | None = None,
        steps: int | None = None,
        optimization_problem=None,
        dt_h: float | list[float] | None = None,
        datetime_index: pandas.DatetimeIndex | None = None,
        initial_state: dict | None = None,
        terminal_state: dict | None = None,
        formulation=None,
    ) -> MultiPeriodResult:
        """Build and solve a multi-period optimization in one GEKKO model.

        ``dt_h`` defaults to 1.0 hour when omitted and may be a list of length T
        (variable step size); ``datetime_index`` derives it from consecutive
        differences. ``initial_state`` / ``terminal_state`` pin attributes at
        t<0 / t=T-1.
        """
        from gekko import GEKKO

        from monee.solver.gekko import GEKKOSolver, _solver_options

        steps = _resolve_steps(steps, timeseries_data)
        dt_h_list = _resolve_dt_h(dt_h, datetime_index, steps)

        m = GEKKO(remote=False)
        m.options.SOLVER = self._solver_int
        m.options.WEB = 0
        m.options.IMODE = 3
        m.solver_options = _solver_options(self._solver_int)

        _single = GEKKOSolver(solver=self._solver_int)
        var_meta: dict[int, Var] = {}

        def _inject(net_t, ignored_t, t):
            inject_vars(
                lambda model, comp, cat, _t=t: GEKKOSolver.inject_gekko_vars_attr(
                    m,
                    model,
                    f"{comp.nid if cat == 'branch' else comp.tid}_t{_t}",
                    var_meta=var_meta,
                ),
                net_t.nodes,
                net_t.branches,
                net_t.compounds,
                net_t,
                ignored_t,
            )

        def _process_branches(net_t, ignored_t):
            objs_exprs: list = []
            _single.process_equations_branches(
                m, net_t, net_t.branches, ignored_t, objs_exprs
            )
            return objs_exprs

        def _sink_objective(objs_exprs):
            if objs_exprs:
                m.Obj(sum(objs_exprs))

        net_copies = _assemble_two_pass(
            m,
            _single,
            "GEKKO",
            network,
            timeseries_data,
            steps,
            optimization_problem,
            dt_h_list,
            initial_state,
            terminal_state,
            formulation,
            inject=_inject,
            process_branches=_process_branches,
            sink_objective=_sink_objective,
            add_equations=m.Equations,
            add_terminal=lambda var, target: m.Equation(var == target),
        )

        _log.info("Solving multi-period problem (T=%d) ...", steps)
        try:
            m.solve(disp=False)
        except Exception as exc:
            terminal_hint = (
                "  • terminal_state constraints may be infeasible given the "
                "horizon length or storage capacity.\n"
                if terminal_state
                else ""
            )
            raise RuntimeError(
                f"Multi-period GEKKO/IPOPT solve failed (T={steps} periods, "
                f"solver={self._solver_int}).\n"
                f"Common causes:\n"
                f"  • Problem is physically infeasible (conflicting bounds or "
                f"insufficient supply).\n"
                f"{terminal_hint}"
                f"  • Numerical scaling - try normalising loads to per-unit or "
                f"reducing T.\n"
                f"Tip: set steps=1 and increase incrementally to isolate the "
                f"first infeasible period."
            ) from exc

        for net_t in net_copies:
            withdraw_vars(
                lambda target: GEKKOSolver.withdraw_gekko_vars_attr(
                    target, var_meta=var_meta
                ),
                net_t.nodes,
                net_t.branches,
                net_t.compounds,
                net_t,
            )

        return MultiPeriodResult(
            net_copies,
            objective=m.options.OBJFCNVAL,
            success=m.options.APPSTATUS == 1,
            datetime_index=datetime_index,
            backend_used=self._backend_name,
            solver_used=self._solver_name,
        )


# Pyomo multi-period solver


class PyomoMultiPeriodSolver:
    """Multi-period optimizer on Pyomo + pluggable MILP/NLP. Same two-pass
    structure as :class:`GekkoMultiPeriodSolver`."""

    def __init__(self, solver_name: str = "scip"):
        self._solver_name = solver_name
        self._backend_name = "pyomo"

    def solve_multi_period(
        self,
        network: Network,
        timeseries_data: TimeseriesData | None = None,
        steps: int | None = None,
        optimization_problem=None,
        dt_h: float | list[float] | None = None,
        datetime_index: pandas.DatetimeIndex | None = None,
        initial_state: dict | None = None,
        terminal_state: dict | None = None,
        formulation=None,
    ) -> MultiPeriodResult:
        """Build and solve a multi-period optimization in a single Pyomo model."""
        import pyomo.environ as pyo
        from pyomo.opt import SolverStatus, TerminationCondition

        from monee.solver.pyo import PyomoSolver

        steps = _resolve_steps(steps, timeseries_data)
        dt_h_list = _resolve_dt_h(dt_h, datetime_index, steps)

        pm = pyo.ConcreteModel()
        pm.cons = pyo.ConstraintList()
        # Split user vs aux objectives so a future lex extension can separate
        # them; multi-period currently solves the single-phase sum.
        pm.user_obj_exprs: list = []
        pm.aux_obj_exprs: list = []

        _single = PyomoSolver()

        def _inject(net_t, ignored_t, t):
            inject_vars(
                lambda model, comp, cat, _t=t: PyomoSolver.inject_pyomo_vars_attr(
                    pm,
                    model,
                    prefix=f"{cat}_{comp.id}_t{_t}",
                ),
                net_t.nodes,
                net_t.branches,
                net_t.compounds,
                net_t,
                ignored_t,
            )

        net_copies = _assemble_two_pass(
            pm,
            _single,
            "Pyomo",
            network,
            timeseries_data,
            steps,
            optimization_problem,
            dt_h_list,
            initial_state,
            terminal_state,
            formulation,
            inject=_inject,
            process_branches=lambda net_t, ignored_t: (
                _single.process_equations_branches(pm, net_t, net_t.branches, ignored_t)
            ),
            sink_objective=lambda _ctx: None,
            add_equations=lambda eqs: _single._add_equations(pm, eqs),
            add_terminal=lambda var, target: pm.cons.add(var == target),
        )

        all_exprs = pm.user_obj_exprs + pm.aux_obj_exprs
        obj_expr = sum(all_exprs) if all_exprs else 0
        pm.obj = pyo.Objective(expr=obj_expr, sense=pyo.minimize)

        _log.info("Solving multi-period problem (T=%d) ...", steps)
        solver = pyo.SolverFactory(self._solver_name)
        solve_result = solver.solve(pm)

        _ok_terminations = {
            TerminationCondition.optimal,
            TerminationCondition.locallyOptimal,
            TerminationCondition.globallyOptimal,
            TerminationCondition.feasible,
        }
        _failed = (
            solve_result.solver.status
            not in (
                SolverStatus.ok,
                SolverStatus.warning,
            )
            or solve_result.solver.termination_condition not in _ok_terminations
        )
        if _failed:
            from monee.solver.infeasibility import diagnose_infeasibility

            report = diagnose_infeasibility(
                pm,
                solver_name=self._solver_name,
                compute_mis_flag=False,
            )
            report_str = report.summary()
            _log.warning(
                "Multi-period Pyomo solve failed. Infeasibility report:\n%s",
                report_str,
            )

            terminal_hint = (
                "  • terminal_state constraints may be infeasible given the "
                "horizon length or storage capacity.\n"
                if terminal_state
                else ""
            )
            raise RuntimeError(
                f"Multi-period Pyomo/{self._solver_name} solve failed "
                f"(T={steps} periods, status={solve_result.solver.status}).\n"
                f"Common causes:\n"
                f"  • Problem is physically infeasible (conflicting bounds or "
                f"insufficient supply).\n"
                f"{terminal_hint}"
                f"Tip: set steps=1 and increase incrementally to isolate the "
                f"first infeasible period.\n\n"
                f"Infeasibility diagnostics:\n{report_str}"
            )

        for net_t in net_copies:
            withdraw_vars(
                PyomoSolver.withdraw_pyomo_vars_attr,
                net_t.nodes,
                net_t.branches,
                net_t.compounds,
                net_t,
            )

        return MultiPeriodResult(
            net_copies,
            objective=pyo.value(pm.obj),
            success=not _failed,
            datetime_index=datetime_index,
            backend_used=self._backend_name,
            solver_used=self._solver_name,
        )


def _dt_h_from_datetime_index(
    dt_h: float | list[float] | None,
    datetime_index: pandas.DatetimeIndex,
    steps: int,
) -> list[float]:
    if dt_h is not None:
        _log.warning(
            "Both dt_h and datetime_index were provided; dt_h will be "
            "ignored and step durations will be derived from "
            "datetime_index."
        )
    if len(datetime_index) < steps:
        raise ValueError(
            f"datetime_index length ({len(datetime_index)}) is less than "
            f"steps ({steps})."
        )
    intervals = (_dt_h_at_step(datetime_index, t) for t in range(steps))
    diffs = [1.0 if d is None else d for d in intervals]
    if any(d <= 0 for d in diffs):
        raise ValueError(
            "datetime_index must be strictly increasing; "
            "found non-positive step duration(s)."
        )
    return diffs


def _resolve_dt_h(
    dt_h: float | list[float] | None,
    datetime_index: pandas.DatetimeIndex | None,
    steps: int,
) -> list[float]:
    """Return per-period timestep durations [h]. ``None`` means the default of
    1.0; ``datetime_index`` overrides dt_h."""
    if datetime_index is not None:
        return _dt_h_from_datetime_index(dt_h, datetime_index, steps)
    if dt_h is None:
        dt_h = 1.0
    if isinstance(dt_h, (list, tuple)):
        if len(dt_h) != steps:
            raise ValueError(
                f"dt_h list length ({len(dt_h)}) must equal steps ({steps})."
            )
        result = list(dt_h)
        if any(d <= 0 for d in result):
            bad = [d for d in result if d <= 0]
            raise ValueError(f"All dt_h values must be positive; got {bad}.")
        return result
    dt_h = float(dt_h)
    if dt_h <= 0:
        raise ValueError(f"dt_h must be positive; got {dt_h}.")
    return [dt_h] * steps


def run_multi_period(
    network: Network,
    timeseries_data: TimeseriesData | None = None,
    steps: int | None = None,
    optimization_problem=None,
    solver=None,
    backend: str | None = None,
    dt_h: float | list[float] | None = None,
    datetime_index: pandas.DatetimeIndex | None = None,
    initial_state: dict | None = None,
    terminal_state: dict | None = None,
    formulation=None,
) -> MultiPeriodResult:
    """Run a single-shot multi-period optimisation. Cross-period coupling goes
    through the ``inter_temporal_equations`` and ``inter_period_equations``
    protocols; ``TimeseriesData`` is applied per-period before equations are
    assembled. ``dt_h`` defaults to 1.0 hour when omitted; ``datetime_index``
    overrides it (a warning is logged if both are given)."""
    solver = resolve_multi_period_solver(solver, backend=backend)

    _validate_state_keys(initial_state, network, "initial_state")
    _validate_state_keys(terminal_state, network, "terminal_state")

    return solver.solve_multi_period(
        network,
        timeseries_data=timeseries_data,
        steps=steps,
        optimization_problem=optimization_problem,
        dt_h=dt_h,
        datetime_index=datetime_index,
        initial_state=initial_state,
        terminal_state=terminal_state,
        formulation=formulation,
    )


def run_mpc(
    network: Network,
    timeseries_data: TimeseriesData | None = None,
    total_steps: int | None = None,
    horizon: int = 4,
    execution_steps: int = 1,
    solver=None,
    backend: str | None = None,
    optimization_problem=None,
    dt_h: float | list[float] | None = None,
    datetime_index: pandas.DatetimeIndex | None = None,
    initial_state: dict | None = None,
    terminal_state: dict | None = None,
    formulation=None,
) -> MultiPeriodResult:
    """Rolling-horizon MPC. Each iteration solves a *horizon*-period problem,
    accepts the first *execution_steps* periods, advances and reseeds initial
    state from the executed terminal state.

    Note: returned ``objective`` is the sum of per-window values and overcounts
    when ``execution_steps < horizon``.

    A window whose solver reports an unsuccessful solve (``success=False``,
    GEKKO style) raises a ``RuntimeError`` identifying the failed window;
    backends that raise on failure (Pyomo) propagate their own error.

    Example::

        result = run_mpc(
            net, td,
            total_steps=24,
            horizon=6,
            execution_steps=1,
            terminal_state={(bat_id, "e_mwh"): 4.0},
        )
        soc = result.get_result_for(mm.ElectricStorage, "e_mwh")
    """
    total_steps = _resolve_steps(total_steps, timeseries_data)
    dt_h_list = _resolve_dt_h(dt_h, datetime_index, total_steps)

    _validate_state_keys(initial_state, network, "initial_state")
    _validate_state_keys(terminal_state, network, "terminal_state")

    solver = resolve_multi_period_solver(solver, backend=backend)

    if execution_steps < 1:
        raise ValueError(f"execution_steps must be >= 1, got {execution_steps}.")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}.")

    all_net_copies: list[Network] = []
    total_objective = 0.0
    current_initial_state = dict(initial_state) if initial_state else None
    offset = 0

    while offset < total_steps:
        remaining = total_steps - offset
        actual_window = min(horizon, remaining)

        window_td = (
            _slice_timeseries(timeseries_data, offset, actual_window)
            if timeseries_data is not None
            else None
        )
        # window_dt already holds the correct per-period durations (derived from
        # datetime_index by _resolve_dt_h when one was supplied). Drive the
        # window solve from window_dt alone and pass datetime_index=None, so the
        # per-window solve does not re-warn about "both dt_h and datetime_index"
        # and does not discard the computed window_dt.
        window_dt = dt_h_list[offset : offset + actual_window]

        # A terminal target pins only the *global* horizon end. Forwarding it to
        # every rolling window would over-constrain intermediate windows (and can
        # make them infeasible); only the final window reaches the global end.
        is_final_window = offset + actual_window >= total_steps
        window_terminal_state = terminal_state if is_final_window else None

        window_result = solver.solve_multi_period(
            network,
            timeseries_data=window_td,
            steps=actual_window,
            optimization_problem=optimization_problem,
            dt_h=window_dt,
            datetime_index=None,
            initial_state=current_initial_state,
            terminal_state=window_terminal_state,
            formulation=formulation,
        )
        # Some backends (GEKKO) report failure via success=False instead of
        # raising; carrying on would seed the next window from garbage state.
        if not window_result.success:
            raise RuntimeError(
                f"run_mpc: window starting at step {offset} (periods "
                f"{offset}..{offset + actual_window - 1} of {total_steps}) "
                f"failed - the solver reported an unsuccessful solve. "
                f"Aborting; state extracted from a failed window would "
                f"poison all subsequent windows."
            )

        n_execute = min(execution_steps, actual_window)
        executed_copies = window_result._net_copies[:n_execute]
        all_net_copies.extend(executed_copies)
        total_objective += window_result.objective

        current_initial_state = _extract_terminal_state(executed_copies[-1])
        offset += n_execute

    # Index covering only executed periods.
    exec_datetime_index = (
        datetime_index[:total_steps] if datetime_index is not None else None
    )

    return MultiPeriodResult(
        all_net_copies,
        objective=total_objective,
        # Every unsuccessful window raises above, so reaching here means all
        # windows succeeded.
        success=True,
        datetime_index=exec_datetime_index,
        backend_used=getattr(solver, "_backend_name", None),
        solver_used=getattr(solver, "_solver_name", None),
    )

"""
Multi-period optimization for monee networks.

Unlike the sequential :func:`~monee.simulation.timeseries.run` runner, which
solves one NLP per timestep and passes only *scalar floats* between steps,
multi-period optimization builds a **single large problem** that spans T
periods and solves it in one shot.

Cross-period coupling — storage SoC dynamics, thermal mass (LTC), ramp limits,
gas linepack — appears as explicit equality constraints that link solver
variables from adjacent periods.  The key mechanism is :class:`PeriodState`,
which mimics the :class:`~monee.simulation.step_state.StepState` API but
returns live solver variables instead of floats, so existing
``inter_step_equations`` implementations on models and extensions work
**without any modification**.

Usage example::

    import monee
    import monee.model as mm
    from monee.simulation.timeseries import TimeseriesData

    td = TimeseriesData()
    td.add_child_series(load_id, "p_mw", [2.0, 3.0, 2.5, 1.8])

    result = monee.run_multi_period(net, td, optimization_problem=my_opt_problem, dt_h=1.0)
    vm_df = result.get_result_for(mm.Bus, "vm_pu")  # rows=periods, cols=bus ids

Overriding initial conditions::

    result = monee.run_multi_period(net, td,
        initial_state={(bat_id, "e_mwh"): 2.0})

Adding a terminal state constraint (storage must end at least half-full)::

    result = monee.run_multi_period(net, td,
        terminal_state={(bat_id, "e_mwh"): 4.0})  # e_mwh[T-1] == 4.0 MWh

Rolling-horizon MPC::

    result = monee.run_mpc(net, td, total_steps=24, horizon=6, execution_steps=1)

Solver selection::

    from monee.simulation.multi_period import GekkoMultiPeriodSolver, PyomoMultiPeriodSolver

    result = monee.run_multi_period(net, td, solver=GekkoMultiPeriodSolver())
    result = monee.run_multi_period(net, td, solver=PyomoMultiPeriodSolver())
"""

from __future__ import annotations

import logging

import pandas

from monee.model import Network
from monee.model.extension.islanding.core import NetworkIslandingConfig
from monee.simulation.step_state import PeriodState
from monee.simulation.timeseries import TimeseriesData
from monee.solver.core import (
    _TABLE_CSS,
    SolverResult,
    _col_summary,
    _display_df,
    find_ignored_nodes,
    ignore_node,
    inject_vars,
    mark_ignored_components,
    withdraw_vars,
)
from monee.solver.dispatch import resolve_multi_period_solver

_log = logging.getLogger(__name__)


def _prepare_period(
    network: Network,
    timeseries_data: TimeseriesData | None,
    t: int,
    optimization_problem,
) -> tuple[Network, set]:
    """Copy *network*, apply timeseries for step *t*, run ``prepare()``, and
    compute the ignored-node set.

    Returns:
        ``(net_t, ignored_nodes)`` ready for variable injection.
    """
    net_t = network.copy()

    if timeseries_data is not None:
        timeseries_data.apply_to_network(net_t, t)

    for ext in net_t.extensions:
        ext.prepare(net_t)

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

    for node in net_t.nodes:
        if ignore_node(node, net_t, ignored_nodes):
            continue
        for child in net_t.childs_by_ids(node.child_ids):
            if child.active:
                child.model.overwrite(node.model, node.grid)

    return net_t, ignored_nodes


# Internal helpers for terminal constraints and MPC


def _find_component_var(net_t: Network, comp_id, attr: str):
    """Return the (post-injection) model attribute *attr* for *comp_id* in *net_t*.

    Returns ``None`` when the component or attribute is not found.
    """
    for node in net_t.nodes:
        if node.id == comp_id:
            return getattr(node.model, attr, None)
        for child in net_t.childs_by_ids(node.child_ids):
            if child.id == comp_id:
                return getattr(child.model, attr, None)
    for branch in net_t.branches:
        if branch.id == comp_id:
            return getattr(branch.model, attr, None)
    for compound in net_t.compounds:
        if compound.id == comp_id:
            return getattr(compound.model, attr, None)
    return None


def _extract_terminal_state(net_t: Network) -> dict:
    """Return ``{(comp_id, attr): value}`` for all ``Var`` attributes in
    *net_t* after variable withdrawal.

    Used by :func:`run_mpc` to seed the ``initial_state`` of the next horizon.
    All solved variable values are captured so any attribute can be referenced
    in the next window's ``inter_step_equations``.
    """
    from monee.model.core import Var

    state: dict = {}

    def _scan(comp_id, model):
        for k, v in model.__dict__.items():
            if isinstance(v, Var):
                state[(comp_id, k)] = v.value
            elif isinstance(v, (int, float)):
                state[(comp_id, k)] = v

    for node in net_t.nodes:
        _scan(node.id, node.model)
        for child in net_t.childs_by_ids(node.child_ids):
            _scan(child.id, child.model)
    for branch in net_t.branches:
        _scan(branch.id, branch.model)
    for compound in net_t.compounds:
        _scan(compound.id, compound.model)
    return state


def _validate_state_keys(state: dict | None, net: Network, label: str) -> None:
    """Raise :exc:`ValueError` when any key in *state* does not match a component
    in *net*.  *label* is ``"initial_state"`` or ``"terminal_state"`` for the
    error message.
    """
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
    """Return a new :class:`~monee.simulation.timeseries.TimeseriesData` whose
    series are sliced to ``[start : start + length]``.

    Used by :func:`run_mpc` to extract the current window from a full-horizon
    ``TimeseriesData``.
    """
    end = start + length

    def _slice_dict(d: dict) -> dict:
        return {
            comp_id: {attr: series[start:end] for attr, series in attrs.items()}
            for comp_id, attrs in d.items()
        }

    new_td = TimeseriesData()
    new_td._node_id_to_series = _slice_dict(td._node_id_to_series)
    new_td._child_id_to_series = _slice_dict(td._child_id_to_series)
    new_td._child_name_to_series = _slice_dict(td._child_name_to_series)
    new_td._branch_id_to_series = _slice_dict(td._branch_id_to_series)
    new_td._branch_name_to_series = _slice_dict(td._branch_name_to_series)
    new_td._compound_id_to_series = _slice_dict(td._compound_id_to_series)
    new_td._compound_name_to_series = _slice_dict(td._compound_name_to_series)
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
    ) -> None:
        self._net_copies = net_copies
        self.objective = objective
        self.success = success
        self._datetime_index = datetime_index
        # Build per-period DataFrames once; queried repeatedly by get_result_for.
        self._period_dfs: list[dict[str, pandas.DataFrame]] = [
            net_t.as_result_dataframe_dict() for net_t in net_copies
        ]

    @property
    def T(self) -> int:
        """Number of periods."""
        return len(self._net_copies)

    def _make_index(self) -> pandas.Index:
        if self._datetime_index is not None:
            return self._datetime_index[: self.T]
        return pandas.RangeIndex(self.T)

    def get_result_for(self, model_type, attribute: str) -> pandas.DataFrame:
        """Return a DataFrame with one row per period and one column per component.

        Columns are labelled by component id — select a component with
        ``df[component_id]``.

        Args:
            model_type: The model class (e.g. ``mm.Bus``, ``mm.Junction``).
            attribute: The attribute name (e.g. ``"vm_pu"``, ``"t_pu"``).

        Example::

            vm = result.get_result_for(mm.Bus, "vm_pu")
            print(vm[bus_home_id])   # voltage series across all periods
        """
        rows = []
        for dfs in self._period_dfs:
            df = dfs.get(model_type.__name__, pandas.DataFrame())
            if attribute not in df.columns:
                rows.append({})
                continue
            if "id" in df.columns:
                rows.append(dict(zip(df["id"], df[attribute])))
            else:
                rows.append(df[attribute].to_dict())
        return pandas.DataFrame(rows, index=self._make_index())

    def get_result_for_id(self, component_id, attribute: str) -> pandas.Series:
        """Return a :class:`pandas.Series` of *attribute* values for a specific
        component across all periods.

        Args:
            component_id: The component's id.
            attribute: The attribute name to retrieve.

        Returns:
            A ``Series`` indexed by period index (or datetime).

        Example::

            soc = result.get_result_for_id(bat_id, "e_mwh")
        """
        values = []
        for dfs in self._period_dfs:
            found = False
            for df in dfs.values():
                if "id" in df.columns and attribute in df.columns:
                    row = df[df["id"] == component_id]
                    if not row.empty:
                        values.append(row.iloc[0][attribute])
                        found = True
                        break
            if not found:
                values.append(None)
        return pandas.Series(values, index=self._make_index(), name=attribute)

    def __getitem__(self, component_id) -> pandas.DataFrame:
        """Return all result attributes for *component_id* across every period.

        Each column is one result attribute; each row is one period.
        Internal bookkeeping columns are excluded.

        Raises :exc:`KeyError` if *component_id* is not found in any period.

        Example::

            df = result[bat_id]
            print(df["e_mwh"])  # SoC over all periods
        """
        rows: list[dict] = []
        for dfs in self._period_dfs:
            for df in dfs.values():
                if "id" not in df.columns:
                    continue
                mask = df["id"] == component_id
                if not mask.any():
                    continue
                row = _display_df(df[mask].iloc[0].to_frame().T).iloc[0]
                rows.append({k: v for k, v in row.items() if k != "id"})
                break
        if not rows:
            raise KeyError(component_id)
        return pandas.DataFrame(rows, index=self._make_index())

    def get_period_result(self, t: int) -> SolverResult:
        """Return a :class:`~monee.solver.core.SolverResult` for period *t*.

        Useful for inspecting a single period with the same API as a
        single-step solve.  The ``objective`` on the returned result is
        ``None`` because per-period objective breakdown is not tracked;
        use ``result.objective`` on the :class:`MultiPeriodResult` for
        the global value.
        """
        return SolverResult(
            self._net_copies[t],
            self._period_dfs[t],
            None,
            self.success,
        )

    # Repr helpers

    def _temporal_lines(self) -> list[str]:
        """Return compact per-period evolution lines for attributes that vary
        across periods (e.g. storage SoC).  At most 2 attributes per type."""
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

    def __repr__(self) -> str:
        SEP = "─" * 68
        status = "ok" if self.success else "FAILED"
        lines = [
            f"MultiPeriodResult  T={self.T}  obj={self.objective:.4g}  [{status}]",
            SEP,
        ]
        if self._period_dfs:
            for type_name, df in self._period_dfs[0].items():
                all_dfs = [
                    dfs.get(type_name, pandas.DataFrame()) for dfs in self._period_dfs
                ]
                combined = pandas.concat(all_dfs, ignore_index=True)
                vis = _display_df(combined).drop(
                    columns=["id", "node_id"], errors="ignore"
                )
                num = vis.select_dtypes(include="number")
                parts = []
                for col in num.columns:
                    s = _col_summary(num[col])
                    if s:
                        parts.append(f"{col} ∈ {s}" if "[" in s else f"{col} = {s}")
                row = f"  {type_name:<22} ×{len(df):>2}"
                if parts:
                    row += "  │  " + "  ·  ".join(parts[:4])
                lines.append(row)

        # Temporal evolution section — only shown when there are varying attrs
        temporal = self._temporal_lines()
        if temporal:
            lines.append(SEP)
            lines.append("  Temporal evolution (mean over components):")
            lines.extend(temporal)

        lines.append(SEP)
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        status_color = "#090" if self.success else "#c00"
        status_text = "ok" if self.success else "failed"
        sections = []
        if self._period_dfs:
            type_dfs = self._collect_type_dfs()
            for type_name, dfs in type_dfs.items():
                n_comp = len(dfs[0])
                plural = "instance" if n_comp == 1 else "instances"
                combined = pandas.concat(dfs, ignore_index=True)
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
                tbl = (
                    pandas.DataFrame(stat_rows).to_html(
                        index=False, border=0, classes=[]
                    )
                    if stat_rows
                    else "<em style='color:#888'>(no numeric attributes)</em>"
                )
                sections.append(
                    f"<details open style='margin-bottom:6px'>"
                    f"<summary style='cursor:pointer;font-weight:bold;color:#333;"
                    f"padding:2px 0'>{type_name} "
                    f"<span style='color:#999;font-weight:normal'>({n_comp} {plural})</span>"
                    f"</summary>"
                    f"<div style='color:#888;font-size:.82em;padding:1px 0 3px'>"
                    f"aggregated over {len(dfs)} period{'s' if len(dfs) != 1 else ''}"
                    f"</div>{tbl}</details>"
                )
        header = (
            f"<div style='font-weight:bold;font-size:1.05em;padding:4px 0 8px'>"
            f"MultiPeriodResult &nbsp;"
            f"<span style='font-weight:normal;color:#555'>T={self.T} &nbsp;·&nbsp; "
            f"obj={self.objective:.4g} &nbsp;·&nbsp; "
            f"<span style='color:{status_color}'>{status_text}</span></span></div>"
        )
        return (
            f"{_TABLE_CSS}"
            f"<div class='monee-result'>"
            f"{header}" + "\n".join(sections) + "</div>"
        )


# GEKKO multi-period solver


class GekkoMultiPeriodSolver:
    """
    Multi-period optimizer backed by GEKKO / IPOPT.

    Uses a two-pass approach:

    1. **Injection pass** — prepare and inject variables for all T periods into
       the single shared GEKKO model.
    2. **Equation pass** — assemble per-period equations with a
       :class:`~monee.simulation.step_state.PeriodState` that has access to
       *all* period networks (past and future), enabling arbitrary cross-period
       coupling in ``inter_step_equations`` implementations.
    """

    def __init__(self, solver: int = 1):
        self._solver_int = solver

    def solve_multi_period(
        self,
        network: Network,
        timeseries_data: TimeseriesData | None = None,
        steps: int | None = None,
        optimization_problem=None,
        dt_h: float | list[float] = 1.0,
        datetime_index: pandas.DatetimeIndex | None = None,
        initial_state: dict | None = None,
        terminal_state: dict | None = None,
    ) -> MultiPeriodResult:
        """Build and solve a multi-period optimization in a single GEKKO model.

        Args:
            network: Base network.  Not modified; a fresh copy is made per period.
            timeseries_data: Per-component attribute series applied before each
                period's equations are assembled.
            steps: Number of periods *T*.  Inferred from *timeseries_data* length
                when omitted.
            optimization_problem: Optional optimisation problem (objectives and
                constraints) evaluated per period and accumulated globally.
            dt_h: Timestep duration in hours.  Either a single float (constant
                step size) or a list of length *T* for variable step sizes.
                Overridden by *datetime_index* if provided.
            datetime_index: Optional ``pd.DatetimeIndex`` of length *T*.  Step
                durations are derived from consecutive differences.
            initial_state: Optional ``{(component_id, attr): value}`` overrides
                used when ``PeriodState.get()`` is called for a period before
                the horizon start (t < 0).  Pass the terminal state of the
                previous horizon here for rolling-horizon MPC warm-starts.
            terminal_state: Optional ``{(component_id, attr): float}`` equality
                constraints added at the last period (t=T-1).

        Returns:
            A :class:`MultiPeriodResult` with one solved network copy per period.
        """
        from gekko import GEKKO

        from monee.solver.gekko import DEFAULT_SOLVER_OPTIONS, GEKKOSolver

        steps = _resolve_steps(steps, timeseries_data)
        dt_h_list = _resolve_dt_h(dt_h, datetime_index, steps)

        m = GEKKO(remote=False)
        m.options.SOLVER = self._solver_int
        m.options.WEB = 0
        m.options.IMODE = 3
        m.solver_options = DEFAULT_SOLVER_OPTIONS

        _single = GEKKOSolver(solver=self._solver_int)

        _log.info("Multi-period GEKKO solve: T=%d periods", steps)

        # --- Pass 1: prepare all period networks and inject variables ---
        net_copies: list[Network] = []
        ignored_list: list[set] = []

        for t in range(steps):
            _log.debug("Preparing period %d/%d", t + 1, steps)
            net_t, ignored_t = _prepare_period(
                network, timeseries_data, t, optimization_problem
            )
            inject_vars(
                lambda model, comp, cat, _t=t: GEKKOSolver.inject_gekko_vars_attr(
                    m,
                    model,
                    f"{comp.nid if cat == 'branch' else comp.tid}_t{_t}",
                ),
                net_t.nodes,
                net_t.branches,
                net_t.compounds,
                net_t,
                ignored_t,
            )
            for ext in net_t.extensions:
                ext.activate_timeseries(net_t, ignored_t)
            _single.mark_temporal_components(net_t, ignored_t)
            net_copies.append(net_t)
            ignored_list.append(ignored_t)

        # --- Pass 2: build equations for all periods ---
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

            _single.init_branches(net_t.branches)

            objs_exprs: list = []
            _single.process_equations_nodes_childs(m, net_t, net_t.nodes, ignored_t)
            _single.process_equations_branches(
                m, net_t, net_t.branches, ignored_t, objs_exprs
            )
            _single.process_equations_compounds(m, net_t, net_t.compounds, ignored_t)

            if optimization_problem is not None:
                _single.process_oxf_components(
                    m, net_t, optimization_problem, period_index=t
                )
            else:
                _single.process_internal_oxf_components(m, net_t)

            if objs_exprs:
                m.Obj(sum(objs_exprs))

            _single.process_inter_period_equations(
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
                m.Equations(ext.inter_period_equations(net_t, ignored_t, period_state))
                m.Equations(
                    ext.inter_temporal_equations(net_t, ignored_t, period_state)
                )
                m.Equations(ext.equations(net_t, ignored_t))

            # Terminal state constraints — added only at the last period.
            if terminal_state and t == steps - 1:
                for (comp_id, attr), target in terminal_state.items():
                    var = _find_component_var(net_t, comp_id, attr)
                    if var is not None:
                        m.Equation(var == target)

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
                f"  • Numerical scaling — try normalising loads to per-unit or "
                f"reducing T.\n"
                f"Tip: set steps=1 and increase incrementally to isolate the "
                f"first infeasible period."
            ) from exc

        for net_t in net_copies:
            withdraw_vars(
                GEKKOSolver.withdraw_gekko_vars_attr,
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
        )


# Pyomo multi-period solver


class PyomoMultiPeriodSolver:
    """
    Multi-period optimizer backed by Pyomo and a pluggable MILP/NLP solver.

    Uses the same two-pass approach as :class:`GekkoMultiPeriodSolver`.
    """

    def __init__(self, solver_name: str = "scip"):
        self._solver_name = solver_name

    def solve_multi_period(
        self,
        network: Network,
        timeseries_data: TimeseriesData | None = None,
        steps: int | None = None,
        optimization_problem=None,
        dt_h: float | list[float] = 1.0,
        datetime_index: pandas.DatetimeIndex | None = None,
        initial_state: dict | None = None,
        terminal_state: dict | None = None,
    ) -> MultiPeriodResult:
        """Build and solve a multi-period optimization in a single Pyomo model."""
        import pyomo.environ as pyo
        from pyomo.opt import SolverStatus, TerminationCondition

        from monee.solver.pyo import PyomoSolver

        steps = _resolve_steps(steps, timeseries_data)
        dt_h_list = _resolve_dt_h(dt_h, datetime_index, steps)

        pm = pyo.ConcreteModel()
        pm.cons = pyo.ConstraintList()
        # See ``PyomoSolver.solve`` — split user-defined objectives from
        # formulation-level tightening terms so a future lex extension
        # can pull them apart.  Multi-period still solves single-phase
        # (sum of both) today.
        pm.user_obj_exprs: list = []
        pm.aux_obj_exprs: list = []

        _single = PyomoSolver()

        _log.info("Multi-period Pyomo solve: T=%d periods", steps)

        # --- Pass 1: prepare all period networks and inject variables ---
        net_copies: list[Network] = []
        ignored_list: list[set] = []

        for t in range(steps):
            _log.debug("Preparing period %d/%d", t + 1, steps)
            net_t, ignored_t = _prepare_period(
                network, timeseries_data, t, optimization_problem
            )
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
            for ext in net_t.extensions:
                ext.activate_timeseries(net_t, ignored_t)
            _single.mark_temporal_components(net_t, ignored_t)
            net_copies.append(net_t)
            ignored_list.append(ignored_t)

        # --- Pass 2: build equations for all periods ---
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

            _single.init_branches(net_t.branches)
            _single.process_equations_nodes_childs(pm, net_t, net_t.nodes, ignored_t)
            _single.process_equations_branches(pm, net_t, net_t.branches, ignored_t)
            _single.process_equations_compounds(pm, net_t, net_t.compounds, ignored_t)

            if optimization_problem is not None:
                _single.process_oxf_components(
                    pm, net_t, optimization_problem, period_index=t
                )
            else:
                _single.process_internal_oxf_components(pm, net_t)

            _single.process_inter_period_equations(
                pm,
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
                _single._add_equations(
                    pm, ext.inter_period_equations(net_t, ignored_t, period_state)
                )
                _single._add_equations(
                    pm, ext.inter_temporal_equations(net_t, ignored_t, period_state)
                )
                _single._add_equations(pm, ext.equations(net_t, ignored_t))

            # Terminal state constraints — added only at the last period.
            if terminal_state and t == steps - 1:
                for (comp_id, attr), target in terminal_state.items():
                    var = _find_component_var(net_t, comp_id, attr)
                    if var is not None:
                        pm.cons.add(var == target)

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
        )


def _resolve_steps(steps: int | None, timeseries_data: TimeseriesData | None) -> int:
    """Return *steps*, inferring from *timeseries_data* length when omitted."""
    if steps is not None:
        if (
            timeseries_data is not None
            and timeseries_data.length is not None
            and timeseries_data.length < steps
        ):
            raise ValueError(
                f"timeseries_data has {timeseries_data.length} step(s) but "
                f"steps={steps} was requested.  Add more values to the series "
                f"or reduce 'steps'."
            )
        return steps
    if timeseries_data is not None and timeseries_data.length is not None:
        return timeseries_data.length
    raise ValueError(
        "'steps' must be provided when timeseries_data has no registered series."
    )


def _resolve_dt_h(
    dt_h: float | list[float],
    datetime_index: pandas.DatetimeIndex | None,
    steps: int,
) -> list[float]:
    """Return a list of per-period timestep durations in hours."""
    if datetime_index is not None:
        if isinstance(dt_h, (list, tuple)) or dt_h != 1.0:
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
        diffs = [
            (datetime_index[t] - datetime_index[t - 1]).total_seconds() / 3600.0
            if t > 0
            else (datetime_index[1] - datetime_index[0]).total_seconds() / 3600.0
            if steps > 1
            else 1.0
            for t in range(steps)
        ]
        if any(d <= 0 for d in diffs):
            raise ValueError(
                "datetime_index must be strictly increasing; "
                "found non-positive step duration(s)."
            )
        return diffs
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


# Convenience entry point — single-shot multi-period


def run_multi_period(
    network: Network,
    timeseries_data: TimeseriesData | None = None,
    steps: int | None = None,
    optimization_problem=None,
    solver=None,
    backend: str | None = None,
    dt_h: float | list[float] = 1.0,
    datetime_index: pandas.DatetimeIndex | None = None,
    initial_state: dict | None = None,
    terminal_state: dict | None = None,
) -> MultiPeriodResult:
    """
    Run a multi-period optimization over *network*.

    All *T = steps* periods are assembled into a single solver problem and
    solved in one shot.  Cross-period coupling (storage SoC, thermal mass,
    ramp limits) is expressed as equality constraints linking solver variables
    from adjacent periods via the existing ``inter_step_equations`` protocol.

    ``TimeseriesData`` is applied per-period before equations are assembled,
    so time-varying loads, prices, and availability signals are supported
    identically to the sequential timeseries runner.  To pass time-varying
    objective coefficients (e.g. electricity spot prices), register them as
    component attributes via ``timeseries_data.add_child_series(gen_id,
    "price", [...])`` and read ``model.price`` inside the objective lambda.

    Args:
        network: Base network.  Not modified; a fresh copy is made per period.
        timeseries_data: Per-component attribute series.  ``steps`` is inferred
            from its length when *steps* is omitted.
        steps: Number of periods *T*.  Optional if *timeseries_data* has a
            known length.
        optimization_problem: Optional optimisation problem with objectives
            and constraints, evaluated per period and summed globally.
        solver: Either a solver-name string (``"gurobi"``, ``"ipopt"``, …),
            a concrete :class:`GekkoMultiPeriodSolver` /
            :class:`PyomoMultiPeriodSolver` instance, or ``None``
            (default — GEKKO+IPOPT).
        backend: ``"gekko"`` / ``"pyomo"`` to force the modelling backend.
            When ``None`` (default), the backend is auto-routed from *solver*.
        dt_h: Timestep duration in hours.  Either a single float (constant) or
            a list of length *T*.  Overridden by *datetime_index* if provided.
        datetime_index: Optional ``pd.DatetimeIndex`` aligned to the *T* periods.
            Step durations are derived from consecutive differences; the index
            is used to label result DataFrame rows.
        initial_state: Optional ``{(component_id, attr): float}`` overrides for
            the tracked-variable values used in the t=0 inter-step equations.
            Values here take precedence over the ``tracked`` attribute values
            read from *network*.  Useful for warm-starting a rolling-horizon
            solve with the terminal state of the previous horizon.
        terminal_state: Optional ``{(component_id, attr): float}`` equality
            constraints added at the last period (t=T-1).  Each entry pins the
            named ``tracked`` variable to the given value at the end of the
            horizon.  Common use: enforce a minimum end-of-horizon storage
            level to prevent the solver from draining batteries in the final
            period.

    Returns:
        A :class:`MultiPeriodResult` with one solved network copy per period.

    Example::

        from monee.simulation.multi_period import run_multi_period
        from monee.simulation.timeseries import TimeseriesData

        td = TimeseriesData()
        td.add_child_series(bat_id, "p_mw", [1.0, -1.0, 0.5, -0.5])

        result = run_multi_period(
            net, td, dt_h=1.0,
            terminal_state={(bat_id, "e_mwh"): 4.0},  # end at initial SoC
        )
        soc = result.get_result_for(mm.ElectricStorage, "e_mwh")
    """
    solver = resolve_multi_period_solver(solver, backend=backend)

    # Validate user-provided state dicts early, before entering the solver.
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
    )


# Rolling-horizon MPC


def run_mpc(
    network: Network,
    timeseries_data: TimeseriesData | None = None,
    total_steps: int | None = None,
    horizon: int = 4,
    execution_steps: int = 1,
    solver=None,
    backend: str | None = None,
    optimization_problem=None,
    dt_h: float | list[float] = 1.0,
    datetime_index: pandas.DatetimeIndex | None = None,
    initial_state: dict | None = None,
    terminal_state: dict | None = None,
) -> MultiPeriodResult:
    """
    Run a rolling-horizon Model Predictive Control (MPC) simulation.

    At each MPC iteration a *horizon*-period optimisation is solved, the first
    *execution_steps* periods are accepted as the committed dispatch, the
    initial state is updated from the terminal state of those executed periods,
    and the window advances by *execution_steps*.  The process repeats until
    *total_steps* periods have been executed.

    The resulting :class:`MultiPeriodResult` contains exactly *total_steps*
    executed period network copies (the lookahead tail of each horizon is
    discarded).

    .. note::
        The ``objective`` field of the returned result is the sum of the
        per-window objective values.  When *execution_steps* < *horizon* this
        overcounts (each window's lookahead is included), so treat it as an
        approximation.  For exact cost accounting, accumulate per-period costs
        from ``result[component_id]`` after the run.

    Args:
        network: Base network.  Not modified.
        timeseries_data: Full-horizon per-component attribute series.  Must
            cover at least *total_steps* periods.
        total_steps: Total number of periods to execute.  Inferred from
            *timeseries_data* length when omitted.
        horizon: Look-ahead window in periods.  Each solve optimises over
            ``min(horizon, remaining)`` periods.  Larger values improve
            dispatch quality at the cost of a bigger problem per solve.
        execution_steps: Number of periods committed per MPC iteration.
            ``execution_steps=1`` (default) is the classic MPC policy.
            ``execution_steps=horizon`` reduces to open-loop receding-horizon.
        solver: Either a solver-name string, a concrete
            :class:`GekkoMultiPeriodSolver` / :class:`PyomoMultiPeriodSolver`
            instance, or ``None`` (default — GEKKO+IPOPT).
        backend: ``"gekko"`` / ``"pyomo"`` to force the modelling backend.
            When ``None`` (default), the backend is auto-routed from *solver*.
        optimization_problem: Optional optimisation problem passed to each
            horizon solve.
        dt_h: Timestep duration(s) in hours — scalar or list of length
            *total_steps*.
        datetime_index: Optional ``pd.DatetimeIndex`` of length *total_steps*.
        initial_state: Optional ``{(component_id, attr): float}`` seed for the
            very first horizon's t=0 inter-step equations.  Subsequent
            horizons derive their initial state from the executed period results
            automatically.
        terminal_state: Optional terminal constraint passed to every horizon
            solve.  Useful for cyclic storage constraints (enforce the storage
            to return to its initial level at the end of each horizon).

    Returns:
        A :class:`MultiPeriodResult` containing the *total_steps* executed
        period network copies.

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

    # Validate user-provided state dicts early.
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
        window_dt = dt_h_list[offset : offset + actual_window]
        window_idx = (
            datetime_index[offset : offset + actual_window]
            if datetime_index is not None
            else None
        )

        window_result = solver.solve_multi_period(
            network,
            timeseries_data=window_td,
            steps=actual_window,
            optimization_problem=optimization_problem,
            dt_h=window_dt,
            datetime_index=window_idx,
            initial_state=current_initial_state,
            terminal_state=terminal_state,
        )

        n_execute = min(execution_steps, actual_window)
        executed_copies = window_result._net_copies[:n_execute]
        all_net_copies.extend(executed_copies)
        total_objective += window_result.objective

        # Seed next horizon from the terminal state of the last executed period.
        current_initial_state = _extract_terminal_state(executed_copies[-1])
        offset += n_execute

    # Rebuild a datetime index covering only executed periods.
    exec_datetime_index = (
        datetime_index[:total_steps] if datetime_index is not None else None
    )

    return MultiPeriodResult(
        all_net_copies,
        objective=total_objective,
        success=True,
        datetime_index=exec_datetime_index,
    )

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

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
    Var,
    WaterPipe,
)
from monee.model.child import GridFormingMixin
from monee.problem.core import OptimizationProblem

# Display helpers (also imported by simulation.timeseries)

#: Internal bookkeeping columns omitted from pretty-printed output.
_META_COLS: frozenset[str] = frozenset({"active", "independent", "ignored"})


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
        if self.objective is not None and self.objective != 0.0:
            title += f"  (objective = {self.objective:.6g})"
        lines = [title, SEP]
        for type_name, df in self.dataframes.items():
            n = len(df)
            vis = _display_df(df).drop(columns=["id", "node_id"], errors="ignore")
            num = vis.select_dtypes(include="number")
            parts = []
            for col in num.columns:
                s = _col_summary(num[col])
                if s is None:
                    continue
                parts.append(f"{col} ∈ {s}" if "[" in s else f"{col} = {s}")
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
        if self.objective is not None and self.objective != 0.0:
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
        if self.objective is not None and self.objective != 0.0:
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
                na_rep="—",
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

    @abstractmethod
    def solve(
        self,
        input_network: Network,
        optimization_problem: OptimizationProblem = None,
        draw_debug=False,
        exclude_unconnected_nodes=False,
        step_state=None,
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
        raise Exception("None as result for 'equations' is not allowed!")
    return possible_iter if hasattr(possible_iter, "__iter__") else [possible_iter]


def filter_intermediate_eqs(eqs):
    return [eq for eq in eqs if type(eq) is not IntermediateEq]


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
    for compound in network.compounds:
        if ignore_compound(compound, ignored_nodes):
            compound.ignored = True
            # Children attached to *external* nodes (e.g. SubHG at the heat
            # node of a broken CHPHG) wouldn't be caught by the node-loop
            # above — their host node may still be in the active grid.
            # Mark them directly so ignore_child filters them out.
            for sc in compound.subcomponents:
                if isinstance(sc, Child):
                    sc.ignored = True


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
        for child in network.childs_by_ids(node.child_ids):
            if ignore_child(child, ignored_nodes):
                child.ignored = True
                inject_nans(child.model)
                continue
            inject_fn(child.model, child, "child")

    for compound in compounds:
        if ignore_compound(compound, ignored_nodes):
            compound.ignored = True
            inject_nans(compound.model)
            continue
        inject_fn(compound.model, compound, "compound")


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


def compute_bound_violations(
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
    # belongs to an ignored compound — without consulting it here, the child
    # would still appear in its host node's balance equations as a free Var
    # (e.g. SubHG.q_mw_heat absorbing arbitrary heat at the heat node).
    return (not child.active) or (child.node_id in ignored_nodes) or child.ignored


def ignore_compound(compound, ignored_nodes):
    ig = not compound.active
    external_broken = any(
        value in ignored_nodes for value in compound.connected_to.values()
    )
    # Internal subcomponent turned off (e.g. user deactivates one of a
    # CHPHG's internal transfer branches): the ControlNode's power balance —
    # degenerate when its from-branches are gone — otherwise collides with
    # its el_mw / q_mw_heat coupling rows and the LP is infeasible.
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
    # keys=True targets the exact parallel edge — not always key 0.
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


def find_ignored_nodes(network: Network, islanding_config=None):
    """Return node IDs to exclude from the solve.

    Default: active topology, only ExtPowerGrid/ExtHydrGrid children are
    "leading". With *islanding_config*: full topology (backup lines included)
    and any :class:`GridFormingMixin` child counts as leading for an
    islanding-enabled carrier.
    """
    ignored_nodes = set()
    without_cps = network.copy()
    remove_cps(without_cps)

    if islanding_config is not None:
        topology = without_cps._network_internal.copy()
    else:
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
                    from monee.model.grid import GasGrid, PowerGrid, WaterGrid

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
    # Iterate to fixed point. Skipped under islanding (topology there includes
    # inactive backup branches, so degree overstates connectivity).
    if islanding_config is None:
        from monee.model.node import Junction

        # Compound port children (e.g. SubHG on a CHPHG heat node) carry no
        # demand of their own; they must not keep an otherwise-isolated
        # junction alive.
        compound_port_child_ids = {
            sub.id
            for compound in network.compounds
            for sub in compound.subcomponents
            if isinstance(sub, Child)
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
            Heat-only children (HeatLoad / HeatGenerator) don't qualify —
            with no outgoing pipe their q_mw_heat term has no enthalpy
            stream to balance against and the junction becomes infeasible."""
            for cid in int_node.child_ids:
                if cid in compound_port_child_ids:
                    continue
                child = without_cps.child_by_id(cid)
                if not child.active:
                    continue
                if "mass_flow" in getattr(child.model, "vars", {}):
                    return True
            return False

        while True:
            new_stubs = set()
            for node_id in topology.nodes:
                if node_id in ignored_nodes:
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
                # children are heat-only (q_mw_heat) — see _has_mass_flow_anchor.
                if isinstance(int_node.model, Junction) and not _has_mass_flow_anchor(
                    int_node
                ):
                    new_stubs.add(node_id)
            if not new_stubs:
                break
            ignored_nodes.update(new_stubs)

    return ignored_nodes

from abc import ABC, abstractmethod
from dataclasses import dataclass

import networkx as nx
import pandas

from monee.model import (
    Const,
    ExtHydrGrid,
    ExtPowerGrid,
    GenericModel,
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

# ---------------------------------------------------------------------------
# Display helpers (also imported by simulation.timeseries)
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------


@dataclass
class SolverResult:
    """
    The outcome of a single energy-flow or optimisation solve.

    Attributes:
        network: The solved network with all ``Var.value`` attributes updated
            to the solution values.
        dataframes: Per-component-type result tables, keyed by class name
            (e.g. ``"Bus"``, ``"PowerLoad"``).  Each DataFrame has one row
            per component instance and one column per model attribute.
            Prefer :meth:`get` over direct dict access to avoid string-key typos.
        objective: Value of the optimisation objective at the solution.
            ``0.0`` for plain energy-flow (no optimisation problem).
    """

    network: Network
    dataframes: dict[str, pandas.DataFrame]
    objective: float

    def summary(self):
        return repr(self)

    def get(self, model_type) -> pandas.DataFrame:
        """Return the result DataFrame for *model_type*.

        Preferred over ``result.dataframes["ClassName"]`` — avoids string-key
        typos and benefits from IDE autocomplete.

        Args:
            model_type: The model class (e.g. ``mm.PowerLoad``, ``mm.Bus``).

        Returns:
            The result :class:`pandas.DataFrame` for that component type, or an
            empty DataFrame if no instances of that type exist in the network.

        Example::

            df = result.get(mm.PowerLoad)
            print(df["p_mw"])
        """
        return self.dataframes.get(model_type.__name__, pandas.DataFrame())

    def __getitem__(self, component_id) -> pandas.Series:
        """Return the result row for the component with *component_id*.

        Searches all component-type DataFrames and returns the matching row as
        a :class:`pandas.Series`.  Raises :exc:`KeyError` if no component with
        that id is found.

        Example::

            row = result[bus_id]
            print(row["vm_pu"])
        """
        for df in self.dataframes.values():
            if "id" in df.columns:
                mask = df["id"] == component_id
                if mask.any():
                    return df[mask].iloc[0]
        raise KeyError(component_id)

    def __repr__(self) -> str:
        SEP = "─" * 68
        title = "SolverResult"
        if self.objective != 0.0:
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
        return "\n".join(lines)

    def __str__(self) -> str:
        """Full per-type table dump, one section per component class.

        This is what ``print(result)`` renders.  Use ``repr(result)`` (or just
        evaluate ``result`` in a REPL) for the compact one-line-per-type summary.
        """
        title = "SolverResult"
        if self.objective != 0.0:
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
        if self.objective != 0.0:
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
        """Plot this result as an annotated interactive network graph.

        Delegates to :func:`monee.visualization.plot_result`.  Requires
        *plotly* (already a project dependency).

        Args:
            title: Figure title.  Defaults to ``"Network Result"``.
            show_children: Show child components (loads, generators, …) in
                parent-node hover text.  Default ``True``.
            use_monee_positions: Use stored ``node.position`` coordinates
                instead of automatic layout.
            write_to: Optional path to export the figure (PDF / PNG / SVG).

        Returns:
            A :class:`plotly.graph_objects.Figure`.

        Example::

            result = solver.solve(network)
            result.plot()          # display in Jupyter
            result.plot(write_to="result.pdf")
        """
        from monee.visualization.result_visualization import plot_result

        return plot_result(
            self,
            title=title,
            show_children=show_children,
            use_monee_positions=use_monee_positions,
            write_to=write_to,
        )


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

    def process_inter_step_equations(
        self,
        solver_obj,
        network: Network,
        nodes,
        branches,
        compounds,
        ignored_nodes: set,
        step_state,
    ):
        """
        Collect and register inter-step equations from every active model and
        formulation that implements ``inter_step_equations()``.

        Called after regular equation assembly when a non-None *step_state* is
        present.  Models/formulations that don't implement the method are
        silently skipped.
        """
        for node in nodes:
            if ignore_node(node, network, ignored_nodes):
                continue
            if hasattr(node.model, "inter_step_equations"):
                eqs = as_iter(node.model.inter_step_equations(step_state, node.id))
                self._add_equations(solver_obj, filter_intermediate_eqs(eqs))
            if node.formulation is not None and hasattr(
                node.formulation, "inter_step_equations"
            ):
                eqs = as_iter(
                    node.formulation.inter_step_equations(
                        node.model, step_state, node.id
                    )
                )
                self._add_equations(solver_obj, filter_intermediate_eqs(eqs))
            for child in network.childs_by_ids(node.child_ids):
                if ignore_child(child, ignored_nodes):
                    continue
                if hasattr(child.model, "inter_step_equations"):
                    eqs = as_iter(
                        child.model.inter_step_equations(step_state, child.id)
                    )
                    self._add_equations(solver_obj, filter_intermediate_eqs(eqs))
                if child.formulation is not None and hasattr(
                    child.formulation, "inter_step_equations"
                ):
                    eqs = as_iter(
                        child.formulation.inter_step_equations(
                            child.model, step_state, child.id
                        )
                    )
                    self._add_equations(solver_obj, filter_intermediate_eqs(eqs))
        for branch in branches:
            if ignore_branch(branch, network, ignored_nodes):
                continue
            if hasattr(branch.model, "inter_step_equations"):
                eqs = as_iter(branch.model.inter_step_equations(step_state, branch.id))
                self._add_equations(solver_obj, filter_intermediate_eqs(eqs))
            if branch.formulation is not None and hasattr(
                branch.formulation, "inter_step_equations"
            ):
                eqs = as_iter(
                    branch.formulation.inter_step_equations(
                        branch.model, step_state, branch.id
                    )
                )
                self._add_equations(solver_obj, filter_intermediate_eqs(eqs))
        for compound in compounds:
            if ignore_compound(compound, ignored_nodes):
                continue
            if hasattr(compound.model, "inter_step_equations"):
                eqs = as_iter(
                    compound.model.inter_step_equations(step_state, compound.id)
                )
                self._add_equations(solver_obj, filter_intermediate_eqs(eqs))
            if compound.formulation is not None and hasattr(
                compound.formulation, "inter_step_equations"
            ):
                eqs = as_iter(
                    compound.formulation.inter_step_equations(
                        compound.model, step_state, compound.id
                    )
                )
                self._add_equations(solver_obj, filter_intermediate_eqs(eqs))


def as_iter(possible_iter):
    if possible_iter is None:
        raise Exception("None as result for 'equations' is not allowed!")
    return possible_iter if hasattr(possible_iter, "__iter__") else [possible_iter]


def filter_intermediate_eqs(eqs):
    return [eq for eq in eqs if type(eq) is not IntermediateEq]


def inject_nans(target: GenericModel):
    """Replace Var/Const fields on *target* with NaN placeholders (for ignored components)."""
    for key, value in target.__dict__.items():
        if isinstance(value, Const):
            setattr(target, key, Const(float("nan")))
        if isinstance(value, Var):
            setattr(
                target,
                key,
                Var(float("nan"), max=value.max, min=value.min, name=value.name),
            )


def inject_vars(inject_fn, nodes, branches, compounds, network, ignored_nodes):
    """
    Traverse all network components and call *inject_fn* for each active one.

    *inject_fn* has signature ``inject_fn(model, component, category)`` where
    *category* is one of ``"branch"``, ``"node"``, ``"child"``, ``"compound"``.
    Ignored components receive NaN placeholders via :func:`inject_nans` instead.
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
    """
    Traverse all network components and call *withdraw_fn(model)* on each,
    converting backend-specific variable objects back to :class:`Var` instances.
    """
    for branch in branches:
        withdraw_fn(branch.model)
    for node in nodes:
        withdraw_fn(node.model)
        for child in network.childs_by_ids(node.child_ids):
            withdraw_fn(child.model)
    for compound in compounds:
        withdraw_fn(compound.model)


def ignore_branch(branch, network: Network, ignored_nodes):
    return (
        (not branch.active)
        or ignore_node(network.node_by_id(branch.id[0]), network, ignored_nodes)
        or ignore_node(network.node_by_id(branch.id[1]), network, ignored_nodes)
    )


def ignore_node(node, network: Network, ignored_nodes):
    ig = (not node.active) or (node.id in ignored_nodes)
    if not node.independent:
        ig = ig or ignore_compound(network.compound_of_node(node.id), ignored_nodes)
    return ig


def ignore_child(child, ignored_nodes):
    return (not child.active) or (child.node_id in ignored_nodes)


def ignore_compound(compound, ignored_nodes):
    ig = not compound.active
    if any([value in ignored_nodes for value in compound.connected_to.values()]):
        if hasattr(compound.model, "set_active"):
            compound.model.set_active(False)
        else:
            ig = True
    elif hasattr(compound.model, "set_active"):
        compound.model.set_active(True)
    return ig


def generate_real_topology(nx_net):
    net_copy = nx_net.copy()
    # Iterate with keys=True so we can remove the exact edge (not always key 0)
    # when there are parallel branches between the same pair of nodes.
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
        heat_return_node = network.node_by_id(comp.connected_to["heat_return_node_id"])
        heat_node = network.node_by_id(comp.connected_to["heat_node_id"])
        network.branch(WaterPipe(0, 0), heat_return_node.id, heat_node.id)

    for branch in network.branches:
        if isinstance(branch.model, MultiGridBranchModel):
            network.remove_branch(branch.id)


def find_ignored_nodes(network: Network, islanding_config=None):
    """
    Return the set of node IDs that should be excluded from the solve.

    When *islanding_config* is ``None`` (default), the original behaviour is
    preserved: only *active* branches are considered and only components that
    contain an ``ExtPowerGrid`` or ``ExtHydrGrid`` child are kept.

    When *islanding_config* is provided, the function switches to a more
    permissive pre-filter:

    * **All** branches are included in the topology (even inactive / on_off=0
      ones), because a backup line that is currently off may be switched on
      during the solve.  Only nodes with **no path at all** (through any branch)
      to a grid-forming node are pre-removed.
    * Both ``ExtPowerGrid`` / ``ExtHydrGrid`` **and** any child that carries
      ``GridFormingMixin`` are treated as "leading" for the carriers that have
      islanding enabled.
    """
    ignored_nodes = set()
    without_cps = network.copy()
    remove_cps(without_cps)

    if islanding_config is not None:
        # Full topology: every branch is a potential connectivity path.
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
                # Original check: ExtPowerGrid / ExtHydrGrid are always leading.
                if isinstance(child.model, ExtPowerGrid | ExtHydrGrid):
                    component_leading = True
                    break
                # Islanding extension: any GridFormingMixin child is leading for
                # the carrier of its node, provided that carrier has islanding
                # enabled.
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
    return ignored_nodes

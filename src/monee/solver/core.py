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

    def get(self, model_type) -> pandas.DataFrame:
        """
        Return the result DataFrame for *model_type* using the class itself as key.

        This is the preferred alternative to ``result.dataframes["ClassName"]``
        because it avoids string-key typos and benefits from IDE autocomplete.

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

    def __str__(self) -> str:
        result_str = str(self.network)
        result_str += "\n"
        for cls_str, dataframe in self.dataframes.items():
            result_str += cls_str
            result_str += "\n"
            result_str += dataframe.to_string()
            result_str += "\n"
            result_str += "\n"
        return result_str


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

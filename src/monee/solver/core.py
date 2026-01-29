from dataclasses import dataclass

import networkx as nx
import pandas

from monee.model import (
    CHP,
    ExtHydrGrid,
    ExtPowerGrid,
    GasToHeat,
    IntermediateEq,
    MultiGridBranchModel,
    Network,
    Node,
    PowerToHeat,
    Var,
    WaterPipe,
)
from monee.problem.core import OptimizationProblem


@dataclass
class SolverResult:
    """
    No docstring provided.
    """

    network: Network
    dataframes: dict[str, pandas.DataFrame]
    objective: float

    def __str__(self) -> str:
        """
        No docstring provided.
        """
        result_str = str(self.network)
        result_str += "\n"
        for cls_str, dataframe in self.dataframes.items():
            result_str += cls_str
            result_str += "\n"
            result_str += dataframe.to_string()
            result_str += "\n"
            result_str += "\n"
        return result_str


class SolverInterface:
    def solve(
        self,
        input_network: Network,
        optimization_problem: OptimizationProblem = None,
        draw_debug=False,
        exclude_unconnected_nodes=False,
    ):
        pass


def as_iter(possible_iter):
    if possible_iter is None:
        raise Exception("None as result for 'equations' is not allowed!")
    return possible_iter if hasattr(possible_iter, "__iter__") else [possible_iter]


def filter_intermediate_eqs(eqs):
    return [eq for eq in eqs if type(eq) is not IntermediateEq]


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
    for edge in nx_net.edges.data():
        branch = edge[2]["internal_branch"]
        if not branch.active or (
            type(branch.model.on_off) is not Var and branch.model.on_off == 0
        ):
            net_copy.remove_edge(edge[0], edge[1], 0)
    return net_copy


COMPOUND_TYPES_TO_REMOVE = [PowerToHeat, GasToHeat, CHP]


def remove_cps(network: Network):
    relevant_compounds = [
        compound
        for compound in network.compounds
        if type(compound.model) in COMPOUND_TYPES_TO_REMOVE
    ]
    for comp in relevant_compounds:
        network.remove_compound(comp.id)
        heat_return_node = network.node_by_id(comp.connected_to["heat_return_node_id"])
        heat_node = network.node_by_id(comp.connected_to["heat_node_id"])
        network.branch(WaterPipe(0, 0), heat_return_node.id, heat_node.id)

    for branch in network.branches:
        if isinstance(branch.model, MultiGridBranchModel):
            network.remove_branch(branch.id)


def find_ignored_nodes(network: Network):
    ignored_nodes = set()
    without_cps = network.copy()
    remove_cps(without_cps)
    real_topology = generate_real_topology(without_cps._network_internal)
    components = nx.connected_components(real_topology)
    for component in components:
        component_leading = False
        for node in component:
            int_node: Node = real_topology.nodes[node]["internal_node"]
            for child_id in int_node.child_ids:
                child = without_cps.child_by_id(child_id)
                if isinstance(child.model, ExtPowerGrid | ExtHydrGrid):
                    component_leading = True
                    break
            if component_leading:
                break
        if not component_leading:
            ignored_nodes.update(component)
    return ignored_nodes

import math

import scipy.io

from monee.model.branch import *  # noqa needed for generic creation of models
from monee.model.child import *  # noqa needed for generic creation of models
from monee.model.grid import *  # noqa needed for generic creation of models
from monee.model.node import *  # noqa needed for generic creation of models

from .native import native_dict_to_network


def as_controllable(start_value):
    return {"value": start_value, "max": None, "min": None}


def number_of_lines_with_from_to(from_node, to_node, branch_list):
    number = 0
    for branch in branch_list:
        branch_id = branch["id"]
        if branch_id[0] == from_node and branch_id[1] == to_node:
            number += 1
    return number


def read_matpower_data(mat_data):
    mpc = mat_data["mpc"]
    base_mva = mpc["baseMVA"][0][0][0]
    bus_mat = mpc["bus"][0][0]
    branch_mat = mpc["branch"][0][0]
    gen_mat = mpc["gen"][0][0]
    grid_dict_list = {
        "power": {
            "model_type": "PowerGrid",
            "values": {"name": "power", "sn_mva": base_mva[0]},
        }
    }
    node_dict_list = []
    branch_dict_list = []
    child_dict_list = []

    fill_node_dict(bus_mat, node_dict_list, child_dict_list)
    fill_child_dict(gen_mat, node_dict_list, child_dict_list)
    fill_branch_dict(branch_mat, branch_dict_list)

    return native_dict_to_network(
        dict(
            grids=grid_dict_list,
            nodes=node_dict_list,
            childs=child_dict_list,
            branches=branch_dict_list,
        )
    )


def fill_branch_dict(branch_mat, branch_dict_list):
    for i in range(len(branch_mat)):
        branch_dict = {}
        branch_row = branch_mat[i]
        branch_dict["values"] = {}
        branch_dict["grid_id"] = "power"
        branch_dict["id"] = (
            int(branch_row[0]),
            int(branch_row[1]),
            number_of_lines_with_from_to(
                branch_row[0], branch_row[1], branch_dict_list
            ),
        )
        branch_dict["from_node"] = int(branch_row[0])
        branch_dict["to_node"] = int(branch_row[1])
        branch_dict["values"]["br_r"] = branch_row[2]
        branch_dict["values"]["br_x"] = branch_row[3]
        branch_dict["values"]["g_fr"] = 0
        branch_dict["values"]["b_fr"] = branch_row[4] / 2
        branch_dict["values"]["g_to"] = 0
        branch_dict["values"]["b_to"] = branch_row[4] / 2
        branch_dict["values"]["tap"] = 1 if branch_row[8] == 0 else branch_row[8]
        # 1 / (    1 - branch_row[8] )
        # 1 if branch_row[8] == 0 else branch_row[8]
        branch_dict["values"]["shift"] = math.radians(branch_row[9])
        branch_dict["values"]["max_i_ka"] = 0.319
        branch_dict["values"]["on_off"] = 1
        branch_dict["model_type"] = "GenericPowerBranch"
        branch_dict_list.append(branch_dict)


def fill_child_dict(gen_mat, node_dict_list, child_dict_list):
    for i in range(len(gen_mat)):
        child_dict = {}
        gen_row = gen_mat[i]
        child_dict["values"] = {}
        child_dict["id"] = len(child_dict_list)
        # if power output == max power output it is a static generator
        # otherwise it
        if gen_row[1] != gen_row[8] and gen_row[1] == 0:
            child_dict["model_type"] = "ExtPowerGrid"
            child_dict["values"]["p_mw"] = as_controllable(gen_row[1])
            child_dict["values"]["q_mvar"] = as_controllable(gen_row[2])
            child_dict["values"]["vm_pu"] = gen_row[5]
            child_dict["values"]["va_degree"] = 0
        else:
            child_dict["model_type"] = "PowerGenerator"
            child_dict["values"]["p_mw"] = gen_row[1]
            child_dict["values"]["q_mvar"] = gen_row[2]
        for node_dict in node_dict_list:
            if node_dict["id"] == gen_row[0]:
                node_dict["child_ids"].append(child_dict["id"])
        child_dict_list.append(child_dict)


def fill_node_dict(bus_mat, node_dict_list, child_dict_list):
    for i in range(len(bus_mat)):
        node_dict = {}
        bus_row = bus_mat[i]
        node_dict["id"] = int(bus_row[0])
        node_dict["grid_id"] = "power"
        node_dict["values"] = {}
        node_dict["values"]["vm_pu"] = as_controllable(bus_row[7])
        node_dict["values"]["va_degree"] = as_controllable(bus_row[8])
        node_dict["values"]["base_kv"] = bus_row[9]
        node_dict["model_type"] = "Bus"
        node_dict["child_ids"] = []
        node_dict_list.append(node_dict)

        if bus_row[2] != 0 or bus_row[3] != 0:
            node_dict["child_ids"].append(len(child_dict_list))
            model_type = "PowerLoad" if bus_row[2] >= 0 else "PowerGenerator"
            child_dict_list.append(
                dict(
                    id=len(child_dict_list),
                    model_type=model_type,
                    values=dict(p_mw=bus_row[2], q_mvar=bus_row[3]),
                )
            )


def read_matpower_case(file):
    return read_matpower_data(scipy.io.loadmat(file))

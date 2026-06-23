import math
import re

import scipy.io

# Imported so the @model decorator registers them in the model component
# registry that native_dict_to_network resolves the string model_types against.
# These are exactly the concrete model classes this importer constructs (by name).
from monee.model.branch import GenericPowerBranch  # noqa: F401
from monee.model.child import (  # noqa: F401
    ExtPowerGrid,
    PowerGenerator,
    PowerLoad,
    PowerShunt,
)
from monee.model.grid import PowerGrid  # noqa: F401
from monee.model.node import Bus  # noqa: F401

from .native import native_dict_to_network

# MATPOWER BUS_TYPE codes (bus matrix col 2).
REF_BUS_TYPE = 3
ISOLATED_BUS_TYPE = 4


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
    """Build a network from a loaded MATLAB ``.mat`` struct (scipy.io.loadmat)."""
    return _build_network(_mpc_from_mat(mat_data))


def _mpc_from_mat(mat_data):
    """Unwrap scipy's nested struct into the carrier-agnostic ``mpc`` IR."""
    mpc = mat_data["mpc"]
    return {
        "baseMVA": mpc["baseMVA"][0][0][0][0],
        "bus": mpc["bus"][0][0],
        "gen": mpc["gen"][0][0],
        "branch": mpc["branch"][0][0],
    }


def _build_network(mpc):
    """Construct the :class:`Network` from the ``mpc`` IR.

    ``mpc`` is a plain dict ``{"baseMVA": float, "bus"/"gen"/"branch": matrix}``;
    each matrix is any sequence-of-rows (numpy array from ``.mat`` or list of
    lists from ``.m`` text) indexable by ``[row][column]``.
    """
    grid_dict_list = {
        "power": {
            "model_type": "PowerGrid",
            "values": {"name": "power", "sn_mva": mpc["baseMVA"]},
        }
    }
    node_dict_list = []
    branch_dict_list = []
    child_dict_list = []

    # MATPOWER BUS_TYPE (col 2): 1=PQ, 2=PV, 3=ref/slack, 4=isolated.
    bus_type_by_id = {int(row[0]): int(row[1]) for row in mpc["bus"]}
    # Isolated buses carry no in-service connection and no voltage reference;
    # drop them (as MATPOWER does internally) along with anything attached to
    # them, so no dangling node references reach the network builder.
    isolated_bus_ids = {
        bus_id
        for bus_id, bus_type in bus_type_by_id.items()
        if bus_type == ISOLATED_BUS_TYPE
    }
    fill_node_dict(mpc["bus"], node_dict_list, child_dict_list, isolated_bus_ids)
    fill_child_dict(
        mpc["gen"], node_dict_list, child_dict_list, bus_type_by_id, isolated_bus_ids
    )
    fill_branch_dict(mpc["branch"], branch_dict_list, isolated_bus_ids)
    if not any(child["model_type"] == "ExtPowerGrid" for child in child_dict_list):
        raise ValueError(
            "MATPOWER case has no usable slack: no in-service generator sits at a "
            "reference (BUS_TYPE==3) bus, so the network has no voltage/angle "
            "reference and cannot be solved. Mark the reference bus as type 3 and "
            "give it an in-service generator."
        )
    return native_dict_to_network(
        {
            "grids": grid_dict_list,
            "nodes": node_dict_list,
            "childs": child_dict_list,
            "branches": branch_dict_list,
        }
    )


def fill_branch_dict(branch_mat, branch_dict_list, isolated_bus_ids=frozenset()):
    for i in range(len(branch_mat)):
        branch_row = branch_mat[i]
        if (
            int(branch_row[0]) in isolated_bus_ids
            or int(branch_row[1]) in isolated_bus_ids
        ):
            continue
        branch_dict = {}
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
        branch_dict["values"]["br_r_pu"] = branch_row[2]
        branch_dict["values"]["br_x_pu"] = branch_row[3]
        branch_dict["values"]["g_fr_pu"] = 0
        branch_dict["values"]["b_fr_pu"] = branch_row[4] / 2
        branch_dict["values"]["g_to_pu"] = 0
        branch_dict["values"]["b_to_pu"] = branch_row[4] / 2
        branch_dict["values"]["tap"] = 1 if branch_row[8] == 0 else branch_row[8]
        branch_dict["values"]["shift"] = math.radians(branch_row[9])
        branch_dict["values"]["max_i_ka"] = 999  # essentially unbound

        # BR_STATUS (col 11): 1=in-service, 0=out-of-service. Out-of-service
        # branches stay in the network with on_off=0, so the nodal power balance
        # zeroes their flow. Older minimal cases may omit the column.
        branch_dict["values"]["on_off"] = (
            int(branch_row[10]) if len(branch_row) > 10 else 1
        )
        branch_dict["model_type"] = "GenericPowerBranch"
        branch_dict_list.append(branch_dict)


def fill_child_dict(
    gen_mat,
    node_dict_list,
    child_dict_list,
    bus_type_by_id,
    isolated_bus_ids=frozenset(),
):
    slack_assigned = False

    for i in range(len(gen_mat)):
        child_dict = {}
        gen_row = gen_mat[i]
        bus_id = int(gen_row[0])
        if bus_id in isolated_bus_ids:
            continue
        in_service = gen_row[7] > 0  # GEN_STATUS (col 8)
        is_ref = bus_type_by_id.get(bus_id) == REF_BUS_TYPE and not slack_assigned
        child_dict["values"] = {}
        child_dict["id"] = len(child_dict_list)
        if is_ref and in_service:
            slack_assigned = True
            child_dict["model_type"] = "ExtPowerGrid"
            child_dict["values"]["p_mw"] = as_controllable(gen_row[1])
            child_dict["values"]["q_mvar"] = as_controllable(gen_row[2])
            child_dict["values"]["vm_pu"] = gen_row[5]
            child_dict["values"]["va_degree"] = 0
        else:
            child_dict["model_type"] = "PowerGenerator"
            child_dict["values"]["p_mw"] = -gen_row[1]
            child_dict["values"]["q_mvar"] = -gen_row[2]
            if not in_service:
                child_dict["values"]["regulation"] = 0
        for node_dict in node_dict_list:
            if node_dict["id"] == gen_row[0]:
                node_dict["child_ids"].append(child_dict["id"])
        child_dict_list.append(child_dict)


def fill_node_dict(
    bus_mat, node_dict_list, child_dict_list, isolated_bus_ids=frozenset()
):
    for i in range(len(bus_mat)):
        bus_row = bus_mat[i]
        if int(bus_row[0]) in isolated_bus_ids:
            continue
        node_dict = {}
        node_dict["id"] = int(bus_row[0])
        node_dict["grid_id"] = "power"
        node_dict["values"] = {}
        node_dict["values"]["vm_pu"] = as_controllable(bus_row[7])
        node_dict["values"]["va_radians"] = {
            "value": bus_row[8] * math.pi / 180,
            "min": -math.pi,
            "max": math.pi,
        }
        node_dict["values"]["base_kv"] = bus_row[9]
        node_dict["model_type"] = "Bus"
        node_dict["child_ids"] = []
        node_dict_list.append(node_dict)
        if bus_row[2] != 0 or bus_row[3] != 0:
            node_dict["child_ids"].append(len(child_dict_list))
            model_type = "PowerLoad" if bus_row[2] >= 0 else "PowerGenerator"
            child_dict_list.append(
                {
                    "id": len(child_dict_list),
                    "model_type": model_type,
                    "values": {"p_mw": bus_row[2], "q_mvar": bus_row[3]},
                }
            )
        # Bus shunt GS/BS (cols 5-6): constant-admittance P/Q at vm = 1.0 p.u.
        if bus_row[4] != 0 or bus_row[5] != 0:
            node_dict["child_ids"].append(len(child_dict_list))
            child_dict_list.append(
                {
                    "id": len(child_dict_list),
                    "model_type": "PowerShunt",
                    "values": {"gs_mw": bus_row[4], "bs_mvar": bus_row[5]},
                }
            )


def _parse_number(token):
    low = token.lower()
    if low in ("inf", "+inf"):
        return math.inf
    if low == "-inf":
        return -math.inf
    if low == "nan":
        return math.nan
    return float(token)


def _parse_matrix(text, field):
    """Extract ``mpc.<field> = [ ... ];`` as a list of float rows, or None."""
    match = re.search(rf"mpc\.{field}\s*=\s*\[(.*?)\]", text, re.DOTALL)
    if match is None:
        return None
    rows = []
    # savecase ends each row with ';'; tolerate plain-newline-separated rows too.
    for raw in match.group(1).replace(";", "\n").split("\n"):
        row = raw.strip()
        if not row:
            continue
        rows.append([_parse_number(token) for token in row.split()])
    return rows


def _mpc_from_m_text(text):
    """Parse a MATPOWER ``.m`` case file (as produced by ``savecase``) into the IR.

    Out of scope: matrices containing MATLAB expressions (e.g. ``1/3``) rather
    than literal numbers - ``savecase`` never emits those.
    """
    # Strip '%' line comments (incl. the extended-format '%column_names%' rows).
    # The numeric core matrices never contain quoted '%', so this is safe.
    text = re.sub(r"%[^\n]*", "", text)
    base = re.search(r"mpc\.baseMVA\s*=\s*([0-9.eE+\-]+)", text)
    if base is None:
        raise ValueError("Not a MATPOWER case: no 'mpc.baseMVA' assignment found.")
    mpc = {
        "baseMVA": float(base.group(1)),
        "bus": _parse_matrix(text, "bus"),
        "gen": _parse_matrix(text, "gen"),
        "branch": _parse_matrix(text, "branch"),
    }
    missing = [field for field in ("bus", "gen", "branch") if mpc[field] is None]
    if missing:
        raise ValueError(f"MATPOWER case missing required matrices: {missing}.")
    # An empty bus matrix yields a network with no nodes; reject it explicitly
    # rather than silently building an empty Network. (An empty branch matrix is
    # legitimate: a single-bus island has no branches.)
    if not mpc["bus"]:
        raise ValueError("MATPOWER case has an empty 'mpc.bus' matrix.")
    return mpc


def read_matpower_case(file):
    """Read a MATPOWER case from a ``.m`` text file or a ``.mat`` binary file."""
    if str(file).lower().endswith(".m"):
        with open(file, encoding="utf-8") as case_fp:
            return _build_network(_mpc_from_m_text(case_fp.read()))
    return read_matpower_data(scipy.io.loadmat(file))

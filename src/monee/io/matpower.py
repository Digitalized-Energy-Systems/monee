import math
import re
import warnings

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
    VoltageControlledGenerator,
)
from monee.model.core import Var
from monee.model.formulation import EL_NLP_FORMULATION
from monee.model.grid import PowerGrid  # noqa: F401
from monee.model.node import Bus  # noqa: F401
from monee.problem.core import Constraints, Objectives, OptimizationProblem
from monee.problem.utils import line_loading_limit

from .native import native_dict_to_network

# MATPOWER BUS_TYPE codes (bus matrix col 2).
PV_BUS_TYPE = 2
REF_BUS_TYPE = 3
ISOLATED_BUS_TYPE = 4

_SQRT_3 = math.sqrt(3)
# max_i_ka sentinel for branches with RATE_A == 0 (MATPOWER "unlimited").
_UNLIMITED_I_KA = 999


def as_controllable(start_value):
    return {"value": start_value, "max": None, "min": None}


def read_matpower_data(mat_data):
    """Build a network from a loaded MATLAB ``.mat`` struct (scipy.io.loadmat)."""
    return _build_network(_mpc_from_mat(mat_data))


def _mpc_from_mat(mat_data):
    mpc = mat_data["mpc"]

    ir = {
        "baseMVA": mpc["baseMVA"][0][0][0][0],
        "bus": mpc["bus"][0][0],
        "gen": mpc["gen"][0][0],
        "branch": mpc["branch"][0][0],
    }

    fields = mpc.dtype.names if hasattr(mpc, "dtype") else mpc
    if fields is not None and "gencost" in fields:
        ir["gencost"] = mpc["gencost"][0][0]
    return ir


def _assemble_network(
    mpc, node_dict_list, child_dict_list, branch_dict_list, no_slack_message
):
    if not any(child["model_type"] == "ExtPowerGrid" for child in child_dict_list):
        raise ValueError(no_slack_message)
    return native_dict_to_network(
        {
            "grids": {
                "power": {
                    "model_type": "PowerGrid",
                    # Per-unit branch parameters stay on the file's baseMVA; the
                    # AC flow equations scale per-unit power to MW by sn_mva, so
                    # injections are kept in MW (MATPOWER's native unit) without
                    # rebasing.
                    "values": {"name": "power", "sn_mva": mpc["baseMVA"]},
                }
            },
            "nodes": node_dict_list,
            "childs": child_dict_list,
            "branches": branch_dict_list,
        }
    )


def _build_network(mpc):
    node_dict_list = []
    branch_dict_list = []
    child_dict_list = []

    # MATPOWER BUS_TYPE (col 2): 1=PQ, 2=PV, 3=ref/slack, 4=isolated.
    bus_type_by_id = {int(row[0]): int(row[1]) for row in mpc["bus"]}
    base_kv_by_id = {int(row[0]): row[9] for row in mpc["bus"]}

    isolated_bus_ids = {
        bus_id
        for bus_id, bus_type in bus_type_by_id.items()
        if bus_type == ISOLATED_BUS_TYPE
    }
    fill_node_dict(mpc["bus"], node_dict_list, child_dict_list, isolated_bus_ids)
    fill_child_dict(
        mpc["gen"], node_dict_list, child_dict_list, bus_type_by_id, isolated_bus_ids
    )
    fill_branch_dict(mpc["branch"], base_kv_by_id, branch_dict_list, isolated_bus_ids)
    return _assemble_network(
        mpc,
        node_dict_list,
        child_dict_list,
        branch_dict_list,
        "MATPOWER case has no usable slack: no in-service generator sits at a "
        "reference (BUS_TYPE==3) bus, so the network has no voltage/angle "
        "reference and cannot be solved. Mark the reference bus as type 3 and "
        "give it an in-service generator.",
    )


def fill_branch_dict(
    branch_mat, base_kv_by_id, branch_dict_list, isolated_bus_ids=frozenset()
):
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

        rate_a_mva = branch_row[5]
        from_base_kv = base_kv_by_id.get(int(branch_row[0]))
        branch_dict["values"]["max_i_ka"] = (
            rate_a_mva / (_SQRT_3 * from_base_kv)
            if rate_a_mva > 0 and from_base_kv
            else _UNLIMITED_I_KA
        )
        branch_dict["values"]["max_s_mva"] = rate_a_mva if rate_a_mva > 0 else None

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
    ref_assigned = set()  # reference buses already given an ExtPowerGrid slack
    pv_controlled = set()  # PV buses whose voltage a generator already holds
    node_by_id = {node_dict["id"]: node_dict for node_dict in node_dict_list}

    for i in range(len(gen_mat)):
        child_dict = {}
        gen_row = gen_mat[i]
        bus_id = int(gen_row[0])

        if bus_id in isolated_bus_ids:
            continue

        in_service = gen_row[7] > 0  # GEN_STATUS (col 8)
        bus_type = bus_type_by_id.get(bus_id)

        is_ref = bus_type == REF_BUS_TYPE and bus_id not in ref_assigned
        is_pv = bus_type == PV_BUS_TYPE and bus_id not in pv_controlled

        child_dict["values"] = {}
        child_dict["id"] = len(child_dict_list)
        if is_ref and in_service:
            ref_assigned.add(bus_id)
            child_dict["model_type"] = "ExtPowerGrid"
            # Load convention seed: generation Pg/Qg enters as -Pg/-Qg,
            # matching the OPF path in fill_opf_child_dict.
            child_dict["values"]["p_mw"] = as_controllable(-gen_row[1])
            child_dict["values"]["q_mvar"] = as_controllable(-gen_row[2])
            child_dict["values"]["vm_pu"] = gen_row[5]
            child_dict["values"]["va_degree"] = 0
        elif is_pv and in_service:
            pv_controlled.add(bus_id)
            child_dict["model_type"] = "VoltageControlledGenerator"
            child_dict["values"]["p_mw"] = -gen_row[1]
            child_dict["values"]["q_mvar"] = as_controllable(-gen_row[2])
            child_dict["values"]["vm_pu"] = gen_row[5]
        else:
            child_dict["model_type"] = "PowerGenerator"
            child_dict["values"]["p_mw"] = -gen_row[1]
            child_dict["values"]["q_mvar"] = -gen_row[2]
            if not in_service:
                child_dict["values"]["regulation"] = 0

        node_dict = node_by_id.get(bus_id)
        if node_dict is not None:
            node_dict["child_ids"].append(child_dict["id"])

        child_dict_list.append(child_dict)


def fill_node_dict(
    bus_mat,
    node_dict_list,
    child_dict_list,
    isolated_bus_ids=frozenset(),
    vm_limits_by_id=None,
):
    for i in range(len(bus_mat)):
        bus_row = bus_mat[i]
        node_id = int(bus_row[0])

        if node_id in isolated_bus_ids:
            continue

        node_dict = {}
        node_dict["id"] = node_id
        node_dict["grid_id"] = "power"
        node_dict["values"] = {}
        # For OPF the bus voltage magnitude is a bounded decision variable
        # (VMIN/VMAX); for a plain power flow it stays an unbounded seed.
        if vm_limits_by_id is not None and node_id in vm_limits_by_id:
            vm_min, vm_max = vm_limits_by_id[node_id]
            node_dict["values"]["vm_pu"] = {
                "value": bus_row[7],
                "min": vm_min,
                "max": vm_max,
            }
        else:
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

    # Strip '%' line comments (incl. the extended-format '%column_names%' rows).
    # The numeric core matrices never contain quoted '%', so this is safe.
    text = re.sub(r"%[^\n]*", "", text)
    # Join MATLAB '...' line continuations into one logical line.
    text = re.sub(r"\.\.\.[^\n]*\n", " ", text)
    base = re.search(r"mpc\.baseMVA\s*=\s*([0-9.eE+\-]+)", text)

    if base is None:
        raise ValueError("Not a MATPOWER case: no 'mpc.baseMVA' assignment found.")

    mpc = {
        "baseMVA": float(base.group(1)),
        "bus": _parse_matrix(text, "bus"),
        "gen": _parse_matrix(text, "gen"),
        "branch": _parse_matrix(text, "branch"),
        "gencost": _parse_matrix(text, "gencost"),  # None if absent (power-flow case)
    }
    missing = [field for field in ("bus", "gen", "branch") if mpc[field] is None]

    if missing:
        raise ValueError(f"MATPOWER case missing required matrices: {missing}.")
    if not mpc["bus"]:
        raise ValueError("MATPOWER case has an empty 'mpc.bus' matrix.")

    return mpc


def read_mpc(file):
    """Read a MATPOWER ``.m``/``.mat`` case file into the ``mpc`` IR dict."""
    if str(file).lower().endswith(".m"):
        with open(file, encoding="utf-8") as case_fp:
            return _mpc_from_m_text(case_fp.read())
    return _mpc_from_mat(scipy.io.loadmat(file))


def read_matpower_case(file):
    return _build_network(read_mpc(file))


# MATPOWER gencost columns: MODEL, STARTUP, SHUTDOWN, NCOST, then the cost data.
_GENCOST_PIECEWISE = 1  # MODEL == 1: piecewise linear, (p, cost) breakpoints
_GENCOST_POLYNOMIAL = 2  # MODEL == 2: polynomial cost, coefficients high->low degree
_NCOST_COL = 3
_COST_DATA_COL = 4


def _gencost_coeffs(gencost_mat, row):
    """Polynomial cost coefficients for generator *row*, or None.

    A 2-point piecewise-linear cost is exactly linear and is converted to its
    polynomial equivalent; other PWL costs return None (caller warns).
    """
    cost_row = gencost_mat[row]
    model = int(cost_row[0])
    ncost = int(cost_row[_NCOST_COL])
    if model == _GENCOST_POLYNOMIAL:
        return [float(c) for c in cost_row[_COST_DATA_COL : _COST_DATA_COL + ncost]]
    if model == _GENCOST_PIECEWISE and ncost == 2:
        x0, y0, x1, y1 = (
            float(v) for v in cost_row[_COST_DATA_COL : _COST_DATA_COL + 4]
        )
        if x1 != x0:
            slope = (y1 - y0) / (x1 - x0)
            return [slope, y0 - slope * x0]
    return None


def _polynomial(coeffs, p):
    degree = len(coeffs) - 1
    return sum(c * p ** (degree - k) for k, c in enumerate(coeffs))


def fill_opf_child_dict(
    gen_mat,
    gencost_mat,
    node_dict_list,
    child_dict_list,
    bus_type_by_id,
    isolated_bus_ids=frozenset(),
):
    """OPF generator handling: every in-service generator is *dispatchable*.

    Differs from the power-flow :func:`fill_child_dict`: there is no PV
    voltage-holding (OPF optimises voltages within the bus VMIN/VMAX band), and
    each generator's active/reactive power becomes a bounded decision Var
    (``PMIN/PMAX``, ``QMIN/QMAX``) carrying its polynomial ``gencost``. The first
    in-service generator at a reference bus is the slack (``ExtPowerGrid``); all
    other in-service generators are dispatchable ``PowerGenerator`` children;
    out-of-service generators are kept inert.
    """
    ref_assigned = set()
    costless_pwl_gens = []
    node_by_id = {node_dict["id"]: node_dict for node_dict in node_dict_list}
    for i in range(len(gen_mat)):
        gen_row = gen_mat[i]
        bus_id = int(gen_row[0])
        if bus_id in isolated_bus_ids:
            continue
        in_service = gen_row[7] > 0  # GEN_STATUS
        p_max, p_min = gen_row[8], gen_row[9]  # PMAX, PMIN (MW)
        q_max, q_min = gen_row[3], gen_row[4]  # QMAX, QMIN (MVAr)
        coeffs = None if gencost_mat is None else _gencost_coeffs(gencost_mat, i)
        if (
            coeffs is None
            and in_service
            and gencost_mat is not None
            and i < len(gencost_mat)
            and int(gencost_mat[i][0]) == _GENCOST_PIECEWISE
        ):
            costless_pwl_gens.append((i, bus_id))
        is_ref = (
            bus_type_by_id.get(bus_id) == REF_BUS_TYPE and bus_id not in ref_assigned
        )

        child_dict = {"id": len(child_dict_list), "values": {}}
        if not in_service:
            child_dict["model_type"] = "PowerGenerator"
            child_dict["values"]["p_mw"] = -gen_row[1]
            child_dict["values"]["q_mvar"] = -gen_row[2]
            child_dict["values"]["regulation"] = 0
        else:
            # Load convention: generation is negative, so Pg in [PMIN, PMAX]
            # maps to the stored p_mw in [-PMAX, -PMIN] (likewise reactive).
            child_dict["values"]["p_mw"] = {
                "value": -gen_row[1],
                "min": -p_max,
                "max": -p_min,
            }
            child_dict["values"]["q_mvar"] = {
                "value": -gen_row[2],
                "min": -q_max,
                "max": -q_min,
            }
            if coeffs is not None:
                child_dict["values"]["_cost_coeffs"] = coeffs
            if is_ref:
                ref_assigned.add(bus_id)
                child_dict["model_type"] = "ExtPowerGrid"
                child_dict["values"]["vm_pu"] = gen_row[5]
                child_dict["values"]["va_degree"] = 0
            else:
                child_dict["model_type"] = "PowerGenerator"

        node_dict = node_by_id.get(bus_id)
        if node_dict is not None:
            node_dict["child_ids"].append(child_dict["id"])
        child_dict_list.append(child_dict)

    if costless_pwl_gens:
        listing = ", ".join(
            f"gen {gen_idx} (bus {bus_id})" for gen_idx, bus_id in costless_pwl_gens
        )
        warnings.warn(
            "MATPOWER piecewise-linear gencost (MODEL==1) is only supported for "
            "exactly 2 distinct breakpoints (a linear cost); the cost of the "
            "following generators was dropped from the OPF objective, so they "
            f"stay dispatchable at zero cost: {listing}.",
            stacklevel=2,
        )


def build_matpower_opf(mpc, max_loading=1.0, limit_basis="mva"):
    """Build the AC optimal-power-flow problem from a MATPOWER ``mpc`` IR.

    Returns ``(network, problem)``: a :class:`~monee.model.network.Network` with
    dispatchable generators bounded by PMIN/PMAX/QMIN/QMAX and bus voltages
    bounded by VMIN/VMAX, plus an
    :class:`~monee.problem.core.OptimizationProblem` whose objective is the total
    polynomial ``gencost`` and whose constraints cap branch loading at
    *max_loading* (per unit; pass ``None`` to drop line limits).

    *limit_basis* selects how the line limit is enforced: ``"mva"`` (default)
    caps apparent power |S| <= max_loading * RATE_A, matching MATPOWER exactly;
    ``"current"`` caps current at the imported ``max_i_ka`` (stricter below
    nominal voltage). Solve it with :func:`monee.run_energy_flow_optimization`.
    """

    if mpc.get("gencost") is None:
        raise ValueError(
            "MATPOWER case has no 'gencost'; it carries no OPF objective. Use "
            "read_matpower_case() for a plain power flow instead."
        )

    bus_mat = mpc["bus"]
    bus_type_by_id = {int(row[0]): int(row[1]) for row in bus_mat}
    base_kv_by_id = {int(row[0]): row[9] for row in bus_mat}
    isolated_bus_ids = {
        bus_id for bus_id, t in bus_type_by_id.items() if t == ISOLATED_BUS_TYPE
    }
    # VMIN (col 13) / VMAX (col 12) -> bus voltage bounds.
    vm_limits_by_id = {int(row[0]): (row[12], row[11]) for row in bus_mat}

    node_dict_list, branch_dict_list, child_dict_list = [], [], []
    fill_node_dict(
        bus_mat, node_dict_list, child_dict_list, isolated_bus_ids, vm_limits_by_id
    )
    fill_opf_child_dict(
        mpc["gen"],
        mpc["gencost"],
        node_dict_list,
        child_dict_list,
        bus_type_by_id,
        isolated_bus_ids,
    )
    fill_branch_dict(mpc["branch"], base_kv_by_id, branch_dict_list, isolated_bus_ids)

    network = _assemble_network(
        mpc,
        node_dict_list,
        child_dict_list,
        branch_dict_list,
        "MATPOWER OPF case has no usable slack at a reference (BUS_TYPE==3) bus.",
    )

    network.apply_formulation(EL_NLP_FORMULATION)

    base = mpc["baseMVA"]
    power_scale = base if (base and base != 1) else 100.0
    for child in network.childs:
        for attr in ("p_mw", "q_mvar"):
            var = getattr(child.model, attr, None)
            if isinstance(var, Var):
                var.scale = power_scale
    for branch in network.branches:
        for attr in ("p_from_mw", "q_from_mvar", "p_to_mw", "q_to_mvar"):
            var = getattr(branch.model, attr, None)
            if isinstance(var, Var):
                var.scale = power_scale

    problem = OptimizationProblem()
    objectives = Objectives()
    objectives.select(lambda m: hasattr(m, "_cost_coeffs")).calculate(
        # Cost is a function of generation Pg = -p_mw (load convention).
        lambda models: sum(_polynomial(m._cost_coeffs, -m.p_mw) for m in models)
    )
    problem.objectives = objectives
    if max_loading is not None:
        constraints = Constraints()
        constraints.select_types(GenericPowerBranch).equation(
            lambda m: line_loading_limit(m, "from", max_loading, basis=limit_basis)
        ).equation(
            lambda m: line_loading_limit(m, "to", max_loading, basis=limit_basis)
        )
        problem.constraints = constraints
    # The buses already carry their per-bus VMIN/VMAX (fill_node_dict); free the
    # slack voltage so the OPF optimises it within that band, like MATPOWER.
    problem.optimize_bus_voltages()
    return network, problem


def read_matpower_opf_case(file, max_loading=1.0, limit_basis="mva"):
    """Read a MATPOWER ``.m``/``.mat`` OPF case into ``(network, problem)``.

    The file must carry a ``gencost`` matrix. See :func:`build_matpower_opf`.
    """
    return build_matpower_opf(
        read_mpc(file), max_loading=max_loading, limit_basis=limit_basis
    )

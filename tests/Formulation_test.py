import pandapower as pp
import pandas as pd
import numpy as np
import math
from monee import mx, run_energy_flow, run_energy_flow_optimization
import monee.model as mm
#from monee.problem import create_load_shedding_optimization_problem
from monee.solver.gekko import GEKKOSolver
#from monee.model.formulation import (AC_NETWORK_FORMULATION, MISOCP_NETWORK_FORMULATION, QC_NETWORK_FORMULATION)
import matplotlib.pyplot as plt
import networkx as nx

import networkx as nx
import pytest

import monee
#import monee.model as mm
from monee import run_energy_flow, run_timeseries
from monee.model import GasLinepack, LumpedThermalCapacitance
from monee.model.core import value as mvalue
from monee.model.formulation import (
    EL_NLP_FORMULATION,
    EL_QC_FORMULATION,
    EL_MIQC_FORMULATION,

)
from monee.model.grid import DEFAULT_GAS_HHV_MJ_PER_KG
from monee.network import generate_supply_return_mes_based_on_power_net
from monee.problem.min_load_shedding import create_min_load_shedding_problem
from monee.simulation.timeseries import TimeseriesData
from monee.solver import GEKKOSolver


def create_grid_topology(bus_number=11, meshed=1, n_feeders=3, plot = False):
    if bus_number < 2:
        raise ValueError("bus_number must be >= 2")
    if meshed not in (0, 1, 2):
        raise ValueError("meshed must be 0, 1 or 2")

    n_feeders = min(n_feeders, bus_number - 1)

    nodes = list(range(1, bus_number))
    size = len(nodes) // n_feeders
    rest = len(nodes) % n_feeders

    feeders, start = [], 0
    for i in range(n_feeders):
        end = start + size + (1 if i < rest else 0)
        feeders.append(nodes[start:end])
        start = end

    edges = set()

    def add(i, j):
        if i != j:
            edges.add(tuple(sorted((i, j))))

    # radial
    for f in feeders:
        add(0, f[0])
        for i, j in zip(f[:-1], f[1:]):
            add(i, j)

    # low meshed
    if meshed >= 1:
        for a, b in zip(feeders[:-1], feeders[1:]):
            add(a[-1], b[-1])

    # high meshed
    if meshed >= 2:
        for f in feeders:
            for i in range(len(f) - 2):
                add(f[i], f[i + 2])

        for a, b in zip(feeders[:-1], feeders[1:]):
            for i in range(0, min(len(a), len(b)), 2):
                add(a[i], b[i])
    if plot:
        G = nx.Graph()
        G.add_nodes_from(range(bus_number))
        G.add_edges_from(edges)

        pos = {0: (0, (len(feeders) - 1) / 2)}
        for i, feeder in enumerate(feeders):
            for j, bus in enumerate(feeder):
                pos[bus] = (j + 1, len(feeders) - 1 - i)

        plt.figure(figsize=(10, 5))
        nx.draw(G, pos, with_labels=True, node_size=700)
        plt.title(
            f"{bus_number}-bus grid — "
            f"{['radial', 'low meshed', 'high meshed'][meshed]}"
        )
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return sorted(edges), feeders
def monee_create_grid(bus_number=11, meshed=1, n_feeders=3, show_plot=False):
    net = mx.create_multi_energy_network()
    buses = [mx.create_bus(net) for _ in range(bus_number)]

    mx.create_ext_power_grid(net, buses[0])

    line_args = dict(
        length_m=100,
        r_ohm_per_m=7e-5,
        x_ohm_per_m=7e-5,
    )

    connections, feeders = create_grid_topology(
        bus_number, meshed, n_feeders, plot = show_plot,
    )

    line_buses = {}

    for i, j in connections:
        line_id = mx.create_line(net, buses[i], buses[j], **line_args)
        line_buses[line_id] = (buses[i], buses[j])

    gen_buses = [f[0] for f in feeders[1:]]
    feeder_heads = {f[0] for f in feeders}
    load_buses = [
        i for i in range(1, bus_number)
        if i not in feeder_heads
    ]

    for i in load_buses:
        mx.create_power_load(net, buses[i], p_mw=0.5, q_mvar=0.0)

    for i in gen_buses:
        mx.create_power_generator(net, buses[i], p_mw=1.0, q_mvar=0.0)

    return net, line_buses
def pp_create_grid(bus_number=11, meshed=1, n_feeders=3):
    net = pp.create_empty_network()

    buses = [
        pp.create_bus(net, vn_kv=1.0)
        for _ in range(bus_number)
    ]

    pp.create_ext_grid(net, bus=buses[0], vm_pu=1.0)

    line_args = dict(
        length_km=0.1,
        r_ohm_per_km=0.07,
        x_ohm_per_km=0.07,
        c_nf_per_km=0.0,
        max_i_ka=3.19,
    )

    connections, feeders = create_grid_topology(
        bus_number, meshed, n_feeders
    )

    line_buses = {}

    for i, j in connections:
        line_id = pp.create_line_from_parameters(
            net,
            buses[i],
            buses[j],
            **line_args,
        )
        line_buses[line_id] = (buses[i], buses[j])

    gen_buses = [f[0] for f in feeders[1:]]
    feeder_heads = {f[0] for f in feeders}
    load_buses = [
        i for i in range(1, bus_number)
        if i not in feeder_heads
    ]

    for i in load_buses:
        pp.create_load(
            net,
            bus=buses[i],
            p_mw=0.5,
            q_mvar=0.0,
        )

    for i in gen_buses:
        pp.create_sgen(
            net,
            bus=buses[i],
            p_mw=1.0,
            q_mvar=0.0,
        )

    return net, line_buses

def PP_test_power_network(net, AC=True, show_results=False):
    print("Pandapower test_power_network")
    if AC == True:
        pp.runpp(net) #AC test
    else:
        pp.rundcpp(net) #DC
    if show_results == True:
        pd.set_option("display.max_columns", None)

        print("Solver results")
        print("Bus results")
        print(net.res_bus)

        print("External grid results")
        print(net.res_ext_grid)
        print("\nLoad results:")
        print(net.load[["bus"]].join(net.res_load))

        print("\nGenerator results:")
        print(net.sgen[["bus"]].join(net.res_sgen))

        print("\nLine results:")
        print(net.line[["from_bus", "to_bus"]].join(net.res_line))
    # number of result tables
    result_tables = [
        net.res_bus,
        net.res_line,
        net.res_load,
        net.res_sgen,
        net.res_ext_grid
    ]
    return result_tables

def monee_test_power_network(monee_net, show_results=False):
    print("Monee test_power_network")
    monee_result = run_energy_flow(monee_net)
    if show_results == True:
        print("monee results:\n", monee_result)

    return monee_result

def compare_line_results(pp_net, monee_result, tol=1e-2):
    print("---------------------compare_line_results----------------------")
    print("Line comparison")
    pp_df = pp_net.line[["from_bus", "to_bus"]].join(pp_net.res_line)
    pp_df["from_bus"] = pp_df["from_bus"].astype(int)
    pp_df["to_bus"] = pp_df["to_bus"].astype(int)

    m_df = monee_result.dataframes["PowerLine"].copy()
    m_df["from_bus"] = m_df["id"].apply(lambda x: x[0]) # id in from and to bus
    m_df["to_bus"]   = m_df["id"].apply(lambda x: x[1])

    merged = pd.merge(
        pp_df,
        m_df,
        on=["from_bus", "to_bus"],
        suffixes=("_pp", "_monee"),
        how="outer",
        indicator=True
    )
    #column results to be compared
    cols = [
        ("p_from_mw", "p_from_mw"),
        ("p_to_mw", "p_to_mw"),
        ("i_from_ka", "i_from_ka"),
        ("i_to_ka", "i_to_ka"),
    ]

    mismatches = []

    for col_pp, col_m in cols:
        diff = np.abs(merged[f"{col_pp}_pp"] - merged[f"{col_m}_monee"])
        tolerance_violation = merged[diff > tol] #tolerance defined in function call (currently set to 0.01, just random number set by me)

        if not tolerance_violation.empty:
            print(f"\n Value Mismatch in {col_pp}:")
            print(tolerance_violation[[
                "from_bus", "to_bus",
                f"{col_pp}_pp", f"{col_m}_monee"
            ]])
            mismatches.append(col_pp)

    if not mismatches:
        print("All values match within tolerance!")
    else:
        print("\n Mismatches found in:", mismatches)


def pp_opf(net, AC = True, extgrid_bounds = (-10,10), show_details = False):
    net.sgen["p_nominal"] = net.sgen["p_mw"].copy()
    net.load["p_nominal"] = net.load["p_mw"].copy()

    print("-------------------pp_opf--------------------")
    net.sgen["controllable"] = True
    net.load["controllable"] = True
    net.ext_grid["controllable"] = True
    #set bounds
    bounds_el = (0.8, 1.2)#voltage bounds from monee problem

    net.bus["min_vm_pu"] = bounds_el[0]
    net.bus["max_vm_pu"] = bounds_el[1]
    net.load["min_p_mw"] = 0.0
    net.load["max_p_mw"] = net.load["p_mw"]
    net.load["min_q_mvar"] = net.load["q_mvar"]
    net.load["max_q_mvar"] = net.load["q_mvar"]

    net.sgen["min_p_mw"] = 0.0
    net.sgen["max_p_mw"] = net.sgen["p_mw"] #only down regulation possible
    net.sgen["min_q_mvar"] = -net.sgen["p_mw"]
    net.sgen["max_q_mvar"] = net.sgen["p_mw"]

    net.ext_grid["min_p_mw"] = extgrid_bounds[0]
    net.ext_grid["max_p_mw"] = extgrid_bounds[1]


    # Costs from monee load shedding objective
    for i in net.load.index:
        pp.create_poly_cost(net, i, "load", cp1_eur_per_mw=-10.0)
    for i in net.sgen.index:
        pp.create_poly_cost(net, i, "sgen", cp1_eur_per_mw=1.0) #no cost for sgeneration shedding
    for i in net.ext_grid.index:
        pp.create_poly_cost(net, i, "ext_grid", cp1_eur_per_mw=5.0)
    if AC == True:
        pp.runopp(net) #AC
    else: pp.rundcopp(net) #DC
    print(" pandapower OPF ")
    print("Objective:", net.res_cost)
    if show_details == True:
        print("\nLoad results:")
        print(net.load[["bus"]].join(net.res_load))

        print("\ns-generator results:")
        print(net.sgen[["bus"]].join(net.res_sgen))
    return net

def print_monee_summary(monee_result):
    #regulation in loads is =! 1 but p_mw is still 0.5 everywhere?
    print("---------Monee: print summary of total Loads, Generator Ext. Grid power-------------------------")
    print(f"Objective: {monee_result.objective}")
    result_powerloads = monee_result.get(mm.PowerLoad)
    result_powerloads["act_p_mw"] = result_powerloads["p_mw"]*result_powerloads["regulation"]
    prov_loads = result_powerloads["act_p_mw"].sum()
    print("Monee: prov PowerLoads in total are: \n", prov_loads)

    result_generators = monee_result.get(mm.PowerGenerator)
    result_generators["act_p_mw"] = result_generators["p_mw"]*result_generators["regulation"]
    prov_generation = result_generators["act_p_mw"].sum()
    print("Monee: prov Power Generator in total are: \n",prov_generation)

    result_extgrid = monee_result.get(mm.ExtPowerGrid)
    result_extgrid["act_p_mw"] = result_extgrid["p_mw"]*result_extgrid["regulation"]
    prov_p_extgrid = result_extgrid["act_p_mw"].sum()
    print("Monee: prov Power Ext grid in total are: \n",prov_p_extgrid)

def print_pp_summary(pp_net):
    print("---------PandaPower: print summary of total Loads, Generator Ext. Grid power-------------------------")

    # get cost calculation like in monee with deviation penalty
    load_shed = (pp_net.load["p_nominal"] - pp_net.res_load["p_mw"]).sum()
    sgen_dev = (pp_net.res_sgen["p_mw"] - pp_net.sgen["p_nominal"]).abs().sum()
    ext_usage = pp_net.res_ext_grid["p_mw"].abs().sum()

    monee_like_obj = (  # factors coming from monee load_shedding
            10 * load_shed +
            1 * sgen_dev +
            5 * ext_usage
    )
    print("Monee-like Objective with deviation penalty:", monee_like_obj)
    print("PP provided total load :", pp_net.res_load["p_mw"].sum())
    print("PP provided total generation:", pp_net.res_sgen["p_mw"].sum())
    print("PP provided total external grid power:", pp_net.res_ext_grid["p_mw"].sum())

'''def monee_opf(monee_net, extgrid_bounds = (-10,10), show_details = False):
    print("----------------------monee_opf--------------------")
    problem = create_load_shedding_optimization_problem(
        bounds_el=(0.8, 1.2),
        check_pressure=False,  # no gas grid in this network
        check_t=False,  # no heat grid in this network
        ext_grid_el_bounds = extgrid_bounds
    )
    solver = GEKKOSolver()
    solver.options = {}

    #result = solver.solve(monee_net, optimization_problem=problem)
    result = run_energy_flow_optimization(monee_net, problem, solver=solver) #does not run without giving solver
    print(f"Objective: {result.objective:.4f}")
    if show_details == True:
        print("Run OPF for Monee Formulation")
        print(result)
    return result'''

def compare_monee_results(result1, result2):
    lines1 = result1.get(mm.PowerLine)
    lines2 = result2.get(mm.PowerLine)
    #mask = (lines1 != lines2).any(axis=0)
    #lines1_mismatches = lines1.loc[:, mask]
    #lines2_mismatches = lines2.loc[:, mask]

    print(f"lines1 : {lines1}")
    print(f"lines2 : {lines2}")
    common_cols = lines1.columns.intersection(lines2.columns)
    l1 = lines1[common_cols].copy()
    l2 = lines2[common_cols].copy()
    print(f"lines1 columns: {l1.columns}")
    print(f"lines2 columns: {l2.columns}")
    l1.set_index("id", inplace=True)
    l2.set_index("id", inplace=True)
    l1 = l1.drop(columns=["active", "independent", "ignored", "backup"])
    l2 = l2.drop(columns=["active", "independent", "ignored", "backup"])
    print(f"lines1 : {l1}")
    print(f"lines2 : {l2}")
    tol = 1e-5
    cols = l1.columns.intersection(l2.columns)
    cols_to_keep = []

    for col in cols:
        if pd.api.types.is_numeric_dtype(l1[col]):
            col_diff = (l1[col] - l2[col]).abs() > tol
        elif pd.api.types.is_bool_dtype(l1[col]):
            col_diff = l1[col] ^ l2[col]
        else:
            col_diff = l1[col] != l2[col]
        if col_diff.any():
            cols_to_keep.append(col)
    # keep only differing columns
    lines1_mismatches = l1[cols_to_keep]
    lines2_mismatches = l2[cols_to_keep]
    print(f"lines 1 mismatches: {lines1_mismatches.columns} and {lines1_mismatches}")
    print(f"lines 2 mismatches: {lines2_mismatches.columns} and {lines2_mismatches}")
    differences = lines1_mismatches-lines2_mismatches
    avg_difference = differences.abs().mean()
    diff_relativetoAC = avg_difference/lines2_mismatches.abs().mean()
    print(f"average difference: {avg_difference}")
    print(f"relative difference: {diff_relativetoAC}")
    return lines1_mismatches, lines2_mismatches

def plot_column_compare(result1, result2):
    l1, l2 = result1.align(result2)

    x = range(len(l1))  # or use l1.index if you want IDs
    cols = l1.columns.intersection(l2.columns)
    for col in cols:
        plt.figure()
        plt.scatter(x, l1[col], label="AC", marker="o")
        plt.scatter(x, l2[col], label="QC", marker="x")
        plt.title(f"Column: {col}")
        plt.xlabel("Line index")
        plt.ylabel(col)
        plt.legend()
        plt.grid(True)
        plt.show()
def boxplot_comparison(lines1_mismatches, lines2_mismatches):
    differences = lines1_mismatches-lines2_mismatches
    relative_differences = differences / lines2_mismatches.replace(0,float('nan'))
    #plot with absolute diference
    plt.figure(figsize=(12, 6))
    differences[["p_from_mw", "q_from_mvar", "p_to_mw", "q_to_mvar"]].boxplot(rot=90)
    plt.title("Boxplot of Absolute Differences")
    plt.ylabel("Difference")
    plt.tight_layout()
    plt.show()
    #plot with relative differences
    plt.figure(figsize=(12, 6))
    relative_differences[["p_from_mw", "q_from_mvar", "p_to_mw", "q_to_mvar"]].boxplot(rot=90)
    plt.title("Boxplot of Relative Differences to AC")
    plt.ylabel("Relative Difference")
    plt.tight_layout()
    plt.ylim(-100, 100)
    plt.show()

def run_column_comparison(ac_result,qc_result,component_name, id_col="id", label_ac="NLP_AC",label_qc="QC",tol=None, exclude_cols=None,):
    if exclude_cols is None:
        exclude_cols = [
            "active",
            "independent",
            "ignored",
            "backup",
            "on_off",
        ]

    ac_tables = solver_result_to_tables(ac_result)
    print(f"NLP_AC tables for {component_name}: \n {ac_tables[component_name]}")
    qc_tables = solver_result_to_tables(qc_result)
    print(f"QC tables for {component_name}: \n {qc_tables[component_name]}")
    if component_name not in ac_tables:
        raise KeyError(f"{component_name} not found in NLP_AC result. Available: {list(ac_tables)}")

    if component_name not in qc_tables:
        raise KeyError(f"{component_name} not found in QC result. Available: {list(qc_tables)}")

    ac = ac_tables[component_name].copy()
    qc = qc_tables[component_name].copy()

    if id_col in ac.columns and id_col in qc.columns:
        ac = ac.set_index(id_col)
        qc = qc.set_index(id_col)

    ac, qc = ac.align(qc, join="inner", axis=0)

    common_cols = ac.columns.intersection(qc.columns)

    numeric_cols = [
        c for c in common_cols
        if c not in exclude_cols
        and pd.api.types.is_numeric_dtype(ac[c])
        and pd.api.types.is_numeric_dtype(qc[c])
    ]

    if tol is not None:
        numeric_cols = [
            c for c in numeric_cols
            if ((ac[c] - qc[c]).abs() > tol).any()
        ]

    if not numeric_cols:
        print(f"No numeric columns to plot for {component_name}.")
        return

    x = range(len(ac))

    for col in numeric_cols:
        plt.figure(figsize=(10, 5))
        plt.scatter(x, ac[col], label=label_ac, marker="o")
        plt.scatter(x, qc[col], label=label_qc, marker="x")
        plt.title(f"{component_name} — {col}: {label_ac} vs {label_qc}")
        plt.xlabel(component_name)
        plt.ylabel(col)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def run_result_comparison(ac_result, qc_result, component_name, id_col="id",
                          label_ac="NLP_AC", label_qc="QC", tol=None,
                          exclude_cols=None, add_voltage_diagnostics=True):
    exclude_cols = exclude_cols or ["active", "independent", "ignored", "backup", "on_off", "x_ohm_per_m", "r_ohm_per_m", "length_m", "max_i_ka", "shift", "tab", "base_kv", "node_id"  ]
    tables = [solver_result_to_tables(r) for r in (ac_result, qc_result)]

    for label, t in zip((label_ac, label_qc), tables):
        if component_name not in t:
            raise KeyError(f"{component_name} not found in {label}. Available: {list(t)}")

    ac, qc = [t[component_name].copy() for t in tables]
    print(f"{label_ac} tables for {component_name}:\n{ac}")
    print(f"{label_qc} tables for {component_name}:\n{qc}")

    if id_col in ac and id_col in qc:
        ac, qc = ac.set_index(id_col), qc.set_index(id_col)
    ac, qc = ac.align(qc, join="inner", axis=0)
    gap = None

    if add_voltage_diagnostics:
        if "vm_pu" in ac:
            ac["vm_pu_squared_exact"] = ac["vm_pu"] ** 2
        if "vm_pu" in qc:
            qc["vm_pu_squared_exact"] = qc["vm_pu"]
        if "vm_pu" in ac and "vm_pu_squared" in qc:
            ac["vm_pu_squared_comparison"] = ac["vm_pu"] ** 2
            qc["vm_pu_squared_comparison"] = qc["vm_pu_squared"]
        if {"vm_pu", "vm_pu_squared"} <= set(qc):
            gap = qc["vm_pu_squared"] - qc["vm_pu"] ** 2
            print(f"\nQC voltage-square gap:\n{gap}")
            print("Maximum absolute gap:", gap.abs().max())
            print("Minimum gap:", gap.min())

    cols = [c for c in ac.columns.intersection(qc.columns)
            if c not in exclude_cols
            and pd.api.types.is_numeric_dtype(ac[c])
            and pd.api.types.is_numeric_dtype(qc[c])]
    if tol is not None:
        cols = [c for c in cols if ((ac[c] - qc[c]).abs() > tol).any()]
    if not cols:
        print(f"No numeric columns to plot for {component_name}.")
        return

    x = range(len(ac))
    for c in cols:
        d = qc[c] - ac[c]
        print(f"\n{component_name} — {c}: max={d.abs().max()}, mean={d.abs().mean()}")
        plt.figure(figsize=(10, 5))
        plt.scatter(x, ac[c], label=label_ac, marker="o")
        plt.scatter(x, qc[c], label=label_qc, marker="x")
        plt.title(f"{component_name} — {c}: {label_ac} vs {label_qc}")
        plt.xlabel(component_name); plt.ylabel(c)
        plt.xticks(x, ac.index, rotation=45)
        plt.legend(); plt.grid(); plt.tight_layout(); plt.show()

    if gap is not None:
        plt.figure(figsize=(10, 5))
        plt.scatter(x, gap); plt.axhline(0, linestyle="--")
        plt.title(f"{component_name} — QC voltage-square gap")
        plt.xlabel(component_name); plt.ylabel("vm_pu_squared - vm_pu**2")
        plt.xticks(x, gap.index, rotation=45)
        plt.grid(); plt.tight_layout(); plt.show()

def result_plots(
    ac_result,
    qc_result,
    component_name,
    id_col="id",
    tol=1e-6,
    rel_tol = 0.01,
    vmin_col=None,
    vmax_col=None,
):
    ac, qc = [
        solver_result_to_tables(r)[component_name].copy()
        for r in (ac_result, qc_result)
    ]


    if id_col in ac.columns and id_col in qc.columns:
        ac, qc = ac.set_index(id_col), qc.set_index(id_col)

    ac, qc = ac.align(qc, join="inner", axis=0)

    exclude = {
        "active", "independent", "ignored", "backup", "on_off",
        "x_ohm_per_m", "r_ohm_per_m", "length_m", "max_i_ka",
        "max_s_mva", "shift", "tap", "base_kv", "node_id",
        "vm_pu", "vm_pu_squared",
        "vv", "wc", "ws", "cs", "s", "i_qc",
        "v_on_from", "v_on_to",
        "v_sq_p_from", "v_sq_p_to",
        "theta_u", "theta_M",
        "parallel", "b_to_pu", "g_to_pu",
        "b_fr_pu", "g_fr_pu",
        "br_x_pu", "br_r_pu",
    }

    cols = [
        c for c in ac.columns.intersection(qc.columns)
        if c not in exclude
        and pd.api.types.is_numeric_dtype(ac[c])
        and pd.api.types.is_numeric_dtype(qc[c])
    ]

    print(f"\n--- {component_name} ---")

    x = np.arange(len(ac))

    for c in cols:
        d = qc[c] - ac[c]
        tol_prozent = rel_tol * ac[c].abs().mean()
        abs_mean = ac[c].abs().mean()
        if abs_mean < 1e-12:
            mean_rel_diff = 0.0 if d.abs().mean() < 1e-12 else np.inf
        else:
            mean_rel_diff = d.abs().mean() / abs_mean * 100
        print(
            f"{c:20s} "
            f"max diff = {d.abs().max():.6g}, "
            f"mean diff = {d.abs().mean():.6g}, "
            f"mean rel diff = {mean_rel_diff:.3f}%, " 
            f"{'at least one abs diff > tolerance' if (d.abs() > tol_prozent).any() else 'OK'}"
        )

        plt.figure(figsize=(9, 4))
        plt.scatter(x, ac[c], label="AC", marker="o")
        plt.scatter(x, qc[c], label="QC", marker="x")
        plt.xticks(x, ac.index, rotation=45)
        plt.xlabel(component_name)
        plt.ylabel(c)
        plt.title(f"{component_name} — {c}")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

    if "vm_pu" not in ac.columns or "vm_pu" not in qc.columns:
        return

    ac_v = ac["vm_pu"]
    qc_v = qc["vm_pu"]

    d_v = qc_v - ac_v

    print(
        "\nAC v vs QC v\n"
        f"  max abs diff  = {d_v.abs().max():.6g}\n"
        f"  mean abs diff = {d_v.abs().mean():.6g}\n"
        f"  status        = "
        f"{'MISMATCH' if (d_v.abs() > tol).any() else 'OK'}"
    )
    '''
    plt.figure(figsize=(9, 4))
    plt.scatter(x, ac_v, label="AC v", marker="o")
    plt.scatter(x, qc_v, label="QC v", marker="x")
    plt.xticks(x, ac.index, rotation=45)
    plt.xlabel(component_name)
    plt.ylabel("Voltage magnitude [p.u.]")
    plt.title(f"{component_name} — AC v vs QC v")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    '''
    if "vm_pu_squared" not in qc.columns:
        return

    qc_v_tilde = qc["vm_pu_squared"]
    ac_v_sq = ac_v ** 2

    d_v_tilde_ac = qc_v_tilde - ac_v_sq

    print(
        "\nAC v² vs QC ṽ\n"
        f"  max abs diff  = {d_v_tilde_ac.abs().max():.6g}\n"
        f"  mean abs diff = {d_v_tilde_ac.abs().mean():.6g}"
    )

    plt.figure(figsize=(9, 4))
    plt.scatter(x, ac_v_sq, label="AC v²", marker="o")
    plt.scatter(x, qc_v_tilde, label="QC ṽ", marker="x")
    plt.xticks(x, ac.index, rotation=45)
    plt.xlabel(component_name)
    plt.ylabel("Squared voltage [p.u.²]")
    plt.title(f"{component_name} — AC v² vs QC ṽ")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    if (qc_v_tilde < -tol).any():
        print("\nWARNING: QC ṽ contains negative values.")

    qc_v_from_tilde = np.sqrt(qc_v_tilde.clip(lower=0.0))
    d_v_from_tilde = qc_v_from_tilde - ac_v

    print(
        "\nAC v vs sqrt(QC ṽ)\n"
        f"  max abs diff  = {d_v_from_tilde.abs().max():.6g}\n"
        f"  mean abs diff = {d_v_from_tilde.abs().mean():.6g}"
    )

    plt.figure(figsize=(9, 4))
    plt.scatter(x, ac_v, label="AC v", marker="o")
    plt.scatter(x, qc_v_from_tilde, label="sqrt(QC ṽ)", marker="x")
    plt.xticks(x, ac.index, rotation=45)
    plt.xlabel(component_name)
    plt.ylabel("Voltage magnitude [p.u.]")
    plt.title(f"{component_name} — AC v vs sqrt(QC ṽ)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    qc_v_sq = qc_v ** 2
    square_gap = qc_v_tilde - qc_v_sq

    print(
        "\nQC square relaxation gap: ṽ - v²\n"
        f"  min gap       = {square_gap.min():.6g}\n"
        f"  max gap       = {square_gap.max():.6g}\n"
        f"  mean gap      = {square_gap.mean():.6g}\n"
        f"  mean abs gap  = {square_gap.abs().mean():.6g}"
    )

    if (square_gap < -tol).any():
        bad = square_gap[square_gap < -tol]
        print(
            "  INVALID: ṽ < v²\n"
            f"  violating buses: {bad.index.tolist()}"
        )
    else:
        print("  constraint ṽ >= v²: OK")

    plt.figure(figsize=(9, 4))
    plt.scatter(x, qc_v_sq, label="QC v²", marker="o")
    plt.scatter(x, qc_v_tilde, label="QC ṽ", marker="x")
    plt.xticks(x, ac.index, rotation=45)
    plt.xlabel(component_name)
    plt.ylabel("Squared voltage [p.u.²]")
    plt.title(f"{component_name} — QC v² vs ṽ")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(9, 4))
    plt.scatter(x, square_gap, marker="o")
    plt.axhline(0.0, linestyle="--")
    plt.xticks(x, ac.index, rotation=45)
    plt.xlabel(component_name)
    plt.ylabel("ṽ - v² [p.u.²]")
    plt.title(f"{component_name} — QC square relaxation gap between v_tilde and v²")
    plt.grid()
    plt.tight_layout()
    plt.show()

    if vmin_col is not None and vmax_col is not None:
        if vmin_col in qc.columns:
            v_min = qc[vmin_col]
        elif vmin_col in ac.columns:
            v_min = ac[vmin_col]
        else:
            raise KeyError(f"Column '{vmin_col}' not found.")

        if vmax_col in qc.columns:
            v_max = qc[vmax_col]
        elif vmax_col in ac.columns:
            v_max = ac[vmax_col]
        else:
            raise KeyError(f"Column '{vmax_col}' not found.")

        upper_envelope = (
            (v_min + v_max) * qc_v
            - v_min * v_max
        )

        upper_gap = upper_envelope - qc_v_tilde

        print(
            "\nQC square upper-envelope check\n"
            f"  min slack  = {upper_gap.min():.6g}\n"
            f"  mean slack = {upper_gap.mean():.6g}"
        )

        if (upper_gap < -tol).any():
            bad = upper_gap[upper_gap < -tol]
            print(
                "  INVALID: ṽ exceeds upper envelope\n"
                f"  violating buses: {bad.index.tolist()}"
            )
        else:
            print("  upper-envelope constraint: OK")

        max_square_gap = ((v_max - v_min) ** 2) / 4.0

        normalized_gap = pd.Series(
            np.nan,
            index=qc.index,
            dtype=float,
        )

        valid = max_square_gap > tol

        normalized_gap.loc[valid] = (
            square_gap.loc[valid]
            / max_square_gap.loc[valid]
        )

        print(
            "\nNormalized QC square gap\n"
            f"  max  = {normalized_gap.max():.6g}\n"
            f"  mean = {normalized_gap.mean():.6g}"
        )

        '''
        plt.figure(figsize=(9, 4))
        plt.scatter(x, normalized_gap, marker="o")
        plt.axhline(0.0, linestyle="--")
        plt.axhline(1.0, linestyle="--")
        plt.xticks(x, ac.index, rotation=45)
        plt.xlabel(component_name)
        plt.ylabel("Normalized relaxation gap")
        plt.title(f"{component_name} — normalized QC square gap")
        plt.grid()
        plt.tight_layout()
        plt.show()
        '''


def va_diff_and_analysis(ac_result, qc_result, line_buses, tol=1e-6, tol_degree=0.05
):
    ac = solver_result_to_tables(ac_result)
    qc = solver_result_to_tables(qc_result)

    ac_bus = ac["Bus"].set_index("id")
    qc_line = qc["PowerLine"].set_index("id")

    pairs = line_buses.values() if isinstance(line_buses, dict) else line_buses

    va_diff_ac_rad = pd.Series(
        [
            ac_bus.loc[i, "va_radians"] - ac_bus.loc[j, "va_radians"]
            for i, j in pairs
        ],
        index=qc_line.index
    )

    va_diff_qc_rad = qc_line["va_diff"]
    va_diff_lifted_rad = np.arctan2(qc_line["ws"], qc_line["wc"])

    va_diff_ac = np.rad2deg(va_diff_ac_rad)
    va_diff_qc = np.rad2deg(va_diff_qc_rad)
    va_diff_lifted = np.rad2deg(va_diff_lifted_rad)

    sin_gap = qc_line["s"] - np.sin(va_diff_qc_rad)
    cos_gap = qc_line["cs"] - np.cos(va_diff_qc_rad)
    ws_gap = qc_line["ws"] - qc_line["vv"] * qc_line["s"]
    wc_gap = qc_line["wc"] - qc_line["vv"] * qc_line["cs"]

    comparisons = {
        "AC vs QC va_diff": va_diff_qc - va_diff_ac,
        "AC vs QC lifted": va_diff_lifted - va_diff_ac,
        "QC internal": va_diff_lifted - va_diff_qc,
    }

    print("\nVoltage angle difference comparison [degree]")

    for name, d in comparisons.items():
        print(
            f"{name:20s} "
            f"max diff = {d.abs().max():.6g}°, "
            f"mean diff = {d.abs().mean():.6g}°, "
            f"{'MISMATCH' if (d.abs() > tol_degree).any() else 'OK'}"
        )

    print("\nQC relaxation gaps [dimensionless / lifted variable units]")

    print(
        f"{'sin gap':20s} "
        f"max = {sin_gap.abs().max():.6g}, "
        f"mean = {sin_gap.abs().mean():.6g}"
    )

    print(
        f"{'cos gap':20s} "
        f"max = {cos_gap.abs().max():.6g}, "
        f"mean = {cos_gap.abs().mean():.6g}"
    )

    print(
        f"{'ws gap':20s} "
        f"max = {ws_gap.abs().max():.6g}, "
        f"mean = {ws_gap.abs().mean():.6g}"
    )

    print(
        f"{'wc gap':20s} "
        f"max = {wc_gap.abs().max():.6g}, "
        f"mean = {wc_gap.abs().mean():.6g}"
    )

    x = np.arange(len(qc_line))

    plt.figure(figsize=(9, 4))
    plt.scatter(x, va_diff_ac, label="AC Δθ", marker="o")
    #plt.scatter(x, va_diff_qc, label="QC Δθ", marker="x")
    plt.scatter(x,va_diff_lifted,label="QC atan2(ws, wc)",marker="+")

    plt.xticks(x, qc_line.index, rotation=45)
    plt.xlabel("PowerLine")
    plt.ylabel("Voltage angle difference [°]")
    plt.title("PowerLine — Voltage Angle Difference: AC vs QC")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

def compare_currents(
    ac_result,
    qc_result,
    sn_mva,
    base_kv,
    component_name="PowerLine",
    id_col="id",
    rel_tol=0.01,
):
    ac = solver_result_to_tables(ac_result)[component_name].copy()
    qc = solver_result_to_tables(qc_result)[component_name].copy()

    if id_col in ac and id_col in qc:
        ac, qc = ac.set_index(id_col), qc.set_index(id_col)

    ac, qc = ac.align(qc, join="inner", axis=0)

    i_base_ka = sn_mva / (np.sqrt(3) * base_kv)
    qc_i_lifted = np.sqrt(qc["i_qc"].clip(lower=0)) * i_base_ka

    result = pd.DataFrame({
        "AC i_from_ka": ac["i_from_ka"],
        "QC i_from_ka": qc["i_from_ka"],
        "QC i_qc_ka": qc_i_lifted,
    })

    d = result["QC i_from_ka"] - result["AC i_from_ka"]
    tol_prozent = rel_tol * result["AC i_from_ka"].abs().mean()

    print(
        f"{'AC vs QC i_from_ka physical':20s} "
        f"max diff = {d.abs().max():.6g}, "
        f"mean diff = {d.abs().mean():.6g}, "
        f"{'MISMATCH' if (d.abs() > tol_prozent).any() else 'OK'}"
    )

    d = result["QC i_qc_ka"] - result["AC i_from_ka"]

    print(
        f"{'AC (i_from_ka) vs QC (i_qc_ka) lifted':20s} "
        f"max diff = {d.abs().max():.6g}, "
        f"mean diff = {d.abs().mean():.6g}, "
        f"{'MISMATCH' if (d.abs() > tol_prozent).any() else 'OK'}"
    )

    x = np.arange(len(result))

    plt.figure(figsize=(9, 4))
    plt.scatter(x, result["AC i_from_ka"], label="AC", marker="o")
    plt.scatter(x, result["QC i_from_ka"], label="QC physical", marker="x")
    plt.xticks(x, result.index, rotation=45)
    plt.ylabel("Current [kA]")
    plt.title("AC vs QC current from P, Q, v")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(9, 4))
    plt.scatter(x, result["AC i_from_ka"], label="AC", marker="o")
    plt.scatter(x, result["QC i_qc_ka"], label="QC from i_qc", marker="x")
    plt.xticks(x, result.index, rotation=45)
    plt.ylabel("Current [kA]")
    plt.title("AC current vs QC lifted current")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    return result
def solver_result_to_tables(result):
    tables = {}

    for name in dir(result):
        if name.startswith("_"):
            continue

        value = getattr(result, name)

        if isinstance(value, pd.DataFrame):
            tables[name] = value

        elif isinstance(value, dict):
            for key, item in value.items():
                if isinstance(item, pd.DataFrame):
                    tables[str(key)] = item

    if not tables:
        raise ValueError(
            "No pandas DataFrames found in SolverResult. "
            "Print dir(result) and check where monee stores component tables."
        )
    return tables
def find_columns_more_than_percent_different(table1, table2, tol=0.05, id_col="id"):
    df1 = table1.copy()
    df2 = table2.copy()
    if id_col in df1.columns and id_col in df2.columns:
        df1 = df1.set_index(id_col)
        df2 = df2.set_index(id_col)

    # Keep only common rows and common columns
    df1, df2 = df1.align(df2, join="inner", axis=0)
    df1, df2 = df1.align(df2, join="inner", axis=1)

    different_cols = []

    for col in df1.columns:
        if not ( pd.api.types.is_numeric_dtype(df1[col]) and pd.api.types.is_numeric_dtype(df2[col])
        ):
            continue

        a = df1[col]
        b = df2[col]

        abs_diff = (a - b).abs()

        # Relative difference based on max(abs(a), abs(b))
        # This is more symmetric than using only df1 as reference.
        denom = pd.concat([a.abs(), b.abs()], axis=1).max(axis=1)

        rel_diff = abs_diff / denom.replace(0, np.nan)

        # If both are zero, difference is 0.
        # If one is zero and the other is not, rel_diff becomes 1.
        one_zero = (denom == 0) & (abs_diff > 0)
        rel_diff = rel_diff.fillna(0)

        if ((rel_diff > tol) | one_zero).any():
            different_cols.append(col)

    return different_cols


def compare_extracted_tables_more_than_percent( MISCOP_result,QC_result,tol=0.05,id_col="id"):
    ac_tables = solver_result_to_tables(MISCOP_result)
    qc_tables = solver_result_to_tables(QC_result)
    comparison = {}
    common_table_names = ac_tables.keys() & qc_tables.keys()
    for table_name in common_table_names:
        ac_df = ac_tables[table_name]
        qc_df = qc_tables[table_name]

        different_cols = find_columns_more_than_percent_different(
            ac_df,
            qc_df,
            tol=tol,
            id_col=id_col,
        )

        if not different_cols:
            continue

        ac_aligned = ac_df.copy()
        qc_aligned = qc_df.copy()

        if id_col in ac_aligned.columns and id_col in qc_aligned.columns:
            ac_aligned = ac_aligned.set_index(id_col)
            qc_aligned = qc_aligned.set_index(id_col)

        ac_aligned, qc_aligned = ac_aligned.align(qc_aligned, join="inner", axis=0)
        ac_aligned, qc_aligned = ac_aligned.align(qc_aligned, join="inner", axis=1)

        comparison[table_name] = {
            "different_columns": different_cols,
            "ac": ac_aligned[different_cols],
            "qc": qc_aligned[different_cols],
        }

    return comparison
#PP_net = PP_create_two_line_power_example()
#print("--------------Pandapower AC--------------------------")
#PP_net = pp_create_66bus_high_meshed()
#PP_results = PP_test_power_network(PP_net, AC = True, show_results = False)
print("--------------Monee AC--------------------------")
monee_net_NLP_AC, line_buses_AC = monee_create_grid(bus_number =11, meshed= 2, n_feeders = 4, show_plot=True)
monee_net_NLP_AC.apply_formulation(EL_NLP_FORMULATION)
monee_result_NLP_AC = monee_test_power_network(monee_net_NLP_AC, show_results = False)
#For result table comparison

print("--------------Monee QC--------------------------")
monee_net_QC, line_buses_QC = monee_create_grid(bus_number = 11, meshed= 2, n_feeders = 4, show_plot=True)
monee_net_QC.apply_formulation(EL_QC_FORMULATION)
monee_result_QC = monee_test_power_network(monee_net_QC, show_results = False)
#For result table comparison

result_plots(monee_result_NLP_AC, monee_result_QC, "Bus")
result_plots(monee_result_NLP_AC, monee_result_QC, "PowerLine")
result_plots(monee_result_NLP_AC, monee_result_QC, "PowerLoad")
result_plots(monee_result_NLP_AC, monee_result_QC, "PowerGenerator")
va_diff_and_analysis(monee_result_NLP_AC, monee_result_QC, line_buses_AC)
current_comparison = compare_currents(monee_result_NLP_AC, monee_result_QC, sn_mva=1.0, base_kv=1.0)
#column_comparison = compare_extracted_tables_more_than_percent(monee_result_AC, monee_result_QC)
#print(column_comparison)

#todo angle gaps in claude anschauen, sind die schon korrigiert?
# Ggf gedanken über Monee paper und vorstellung von Merkmalen überlegen

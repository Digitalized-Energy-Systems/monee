import pandapower as pp
import pandas as pd
import numpy as np
from monee import mx, run_energy_flow, run_energy_flow_optimization
import monee.model as mm
from monee.problem import create_load_shedding_optimization_problem
from monee.solver.gekko import GEKKOSolver
from monee.model.formulation import (AC_NETWORK_FORMULATION, MISOCP_NETWORK_FORMULATION, QC_NETWORK_FORMULATION)
import matplotlib.pyplot as plt
def PP_create_two_nodes_power_example(source_flow=0.1):
    net = pp.create_empty_network()
    bus0 = pp.create_bus(net, vn_kv=1)
    bus1 = pp.create_bus(net, vn_kv=1)
    #pp.create_sgen(net, bus=bus0, p_mw=4)
    pp.create_ext_grid(net, bus=bus0)
    pp.create_load(net, bus=bus1, p_mw=0.5, q_mvar=0)

    pp.create_line_from_parameters(
        net,
        from_bus=bus0,
        to_bus=bus1,
        length_km=0.1,
        r_ohm_per_km=0.07,
        x_ohm_per_km=0.07,
        c_nf_per_km=0,
        max_i_ka=3.19
    )
    return net

def monee_create_two_nodes_power_example():
    net = mx.create_multi_energy_network()
    bus_0 = mx.create_bus(net)
    bus_1 = mx.create_bus(net)
    mx.create_line(net, bus_0, bus_1, length_m=100, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)
    mx.create_ext_power_grid(net, bus_0)
    mx.create_power_load(net, bus_1, p_mw=0.5, q_mvar=0.0)
    #mx.create_power_generator(net, bus_0, p_mw = 4.0)
    return net

def PP_create_three_string_network():

    net = pp.create_empty_network()
    buses = [pp.create_bus(net, vn_kv=1) for _ in range(20)]
    pp.create_ext_grid(net, bus=buses[0], vm_pu=1.0)
    line_args = dict(
        length_km=0.1,
        r_ohm_per_km=0.07,
        x_ohm_per_km=0.07,
        c_nf_per_km=0,
        max_i_ka=3.19
    )

    pp.create_line_from_parameters(net, buses[0], buses[1], **line_args)
    pp.create_line_from_parameters(net, buses[1], buses[2], **line_args)
    pp.create_line_from_parameters(net, buses[2], buses[3], **line_args)
    pp.create_line_from_parameters(net, buses[3], buses[4], **line_args)
    pp.create_line_from_parameters(net, buses[4], buses[5], **line_args)
    pp.create_line_from_parameters(net, buses[5], buses[6], **line_args)

    pp.create_line_from_parameters(net, buses[0], buses[7], **line_args)
    pp.create_line_from_parameters(net, buses[7], buses[8], **line_args)
    pp.create_line_from_parameters(net, buses[8], buses[9], **line_args)
    pp.create_line_from_parameters(net, buses[9], buses[10], **line_args)
    pp.create_line_from_parameters(net, buses[10], buses[11], **line_args)
    pp.create_line_from_parameters(net, buses[11], buses[12], **line_args)

    pp.create_line_from_parameters(net, buses[0], buses[13], **line_args)
    pp.create_line_from_parameters(net, buses[13], buses[14], **line_args)
    pp.create_line_from_parameters(net, buses[14], buses[15], **line_args)
    pp.create_line_from_parameters(net, buses[15], buses[16], **line_args)
    pp.create_line_from_parameters(net, buses[16], buses[17], **line_args)
    pp.create_line_from_parameters(net, buses[17], buses[18], **line_args)
    pp.create_line_from_parameters(net, buses[18], buses[19], **line_args)

    load_buses = [2,3,5,6,8,10,11,12,14,16,18,19]

    for b in load_buses:
        pp.create_load(net, bus=buses[b], p_mw=0.5, q_mvar=0)

    sgen_buses = [4,9,13,15,17]

    for b in sgen_buses:
        pp.create_sgen(net, bus=buses[b], p_mw=1.0, vm_pu=1.0,max_q_mvar=0.0, min_q_mvar=0.0)
    return net

def monee_create_three_string_network():
    net = mx.create_multi_energy_network()

    buses = [mx.create_bus(net) for _ in range(20)]

    mx.create_ext_power_grid(net, buses[0])

    line_args = dict(length_m=100, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)

    mx.create_line(net, buses[0], buses[1], **line_args)
    mx.create_line(net, buses[1], buses[2], **line_args)
    mx.create_line(net, buses[2], buses[3], **line_args)
    mx.create_line(net, buses[3], buses[4], **line_args)
    mx.create_line(net, buses[4], buses[5], **line_args)
    mx.create_line(net, buses[5], buses[6], **line_args)

    mx.create_line(net, buses[0], buses[7], **line_args)
    mx.create_line(net, buses[7], buses[8], **line_args)
    mx.create_line(net, buses[8], buses[9], **line_args)
    mx.create_line(net, buses[9], buses[10], **line_args)
    mx.create_line(net, buses[10], buses[11], **line_args)
    mx.create_line(net, buses[11], buses[12], **line_args)

    mx.create_line(net, buses[0], buses[13], **line_args)
    mx.create_line(net, buses[13], buses[14], **line_args)
    mx.create_line(net, buses[14], buses[15], **line_args)
    mx.create_line(net, buses[15], buses[16], **line_args)
    mx.create_line(net, buses[16], buses[17], **line_args)
    mx.create_line(net, buses[17], buses[18], **line_args)
    mx.create_line(net, buses[18], buses[19], **line_args)

    load_buses = [2, 3, 5, 6, 8, 10, 11, 12, 14, 16, 18, 19]

    for b in load_buses:
        mx.create_power_load(net, buses[b], p_mw=0.5, q_mvar=0.0)

    gen_buses = [4, 9, 13, 15, 17]

    for b in gen_buses:
        mx.create_power_generator(net, buses[b], p_mw=1.0, q_mvar=0.0)
    return net

def pp_create_11bus_low_meshed():
    net = pp.create_empty_network()
    buses = [pp.create_bus(net, vn_kv=1) for _ in range(11)]

    pp.create_ext_grid(net, bus=buses[0], vm_pu=1.0)

    line_args = dict(
        length_km=0.1,
        r_ohm_per_km=0.07,
        x_ohm_per_km=0.07,
        c_nf_per_km=0,
        max_i_ka=3.19
    )

    # three strings
    pp.create_line_from_parameters(net, buses[0], buses[1], **line_args)
    pp.create_line_from_parameters(net, buses[1], buses[2], **line_args)
    pp.create_line_from_parameters(net, buses[2], buses[3], **line_args)

    pp.create_line_from_parameters(net, buses[0], buses[4], **line_args)
    pp.create_line_from_parameters(net, buses[4], buses[5], **line_args)
    pp.create_line_from_parameters(net, buses[5], buses[6], **line_args)

    pp.create_line_from_parameters(net, buses[0], buses[7], **line_args)
    pp.create_line_from_parameters(net, buses[7], buses[8], **line_args)
    pp.create_line_from_parameters(net, buses[8], buses[9], **line_args)
    pp.create_line_from_parameters(net, buses[9], buses[10], **line_args)

    # low meshing (only ends connected)
    pp.create_line_from_parameters(net, buses[3], buses[6], **line_args)
    pp.create_line_from_parameters(net, buses[6], buses[10], **line_args)

    # loads &generators
    for b in [2, 3, 5, 6, 8, 9, 10]:
        pp.create_load(net, bus=buses[b], p_mw=0.5, q_mvar=0)
    for b in [4, 7]:
        pp.create_sgen(net, bus=buses[b], p_mw=1.0, vm_pu=1.0, max_q_mvar=0.0, min_q_mvar=0.0)
    return net

def monee_create_11bus_low_meshed():
    net = mx.create_multi_energy_network()
    buses = [mx.create_bus(net) for _ in range(11)]

    mx.create_ext_power_grid(net, buses[0])

    line_args = dict(length_m=100, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)

    # strings
    mx.create_line(net, buses[0], buses[1], **line_args)
    mx.create_line(net, buses[1], buses[2], **line_args)
    mx.create_line(net, buses[2], buses[3], **line_args)

    mx.create_line(net, buses[0], buses[4], **line_args)
    mx.create_line(net, buses[4], buses[5], **line_args)
    mx.create_line(net, buses[5], buses[6], **line_args)

    mx.create_line(net, buses[0], buses[7], **line_args)
    mx.create_line(net, buses[7], buses[8], **line_args)
    mx.create_line(net, buses[8], buses[9], **line_args)
    mx.create_line(net, buses[9], buses[10], **line_args)

    # low meshing
    mx.create_line(net, buses[3], buses[6], **line_args)
    mx.create_line(net, buses[6], buses[10], **line_args)

    for b in [2, 3, 5, 6, 8, 9, 10]:
        mx.create_power_load(net, buses[b], p_mw=0.5, q_mvar=0.0)
    for b in [4, 7]:
        mx.create_power_generator(net, buses[b], p_mw=1.0, q_mvar=0.0)
    return net

def pp_create_11bus_high_meshed():
    net = pp.create_empty_network()
    buses = [pp.create_bus(net, vn_kv=1) for _ in range(11)]

    pp.create_ext_grid(net, bus=buses[0], vm_pu=1.0)

    line_args = dict(
        length_km=0.1,
        r_ohm_per_km=0.07,
        x_ohm_per_km=0.07,
        c_nf_per_km=0,
        max_i_ka=3.19
    )

    # three strings
    pp.create_line_from_parameters(net, buses[0], buses[1], **line_args)
    pp.create_line_from_parameters(net, buses[1], buses[2], **line_args)
    pp.create_line_from_parameters(net, buses[2], buses[3], **line_args)

    pp.create_line_from_parameters(net, buses[0], buses[4], **line_args)
    pp.create_line_from_parameters(net, buses[4], buses[5], **line_args)
    pp.create_line_from_parameters(net, buses[5], buses[6], **line_args)

    pp.create_line_from_parameters(net, buses[0], buses[7], **line_args)
    pp.create_line_from_parameters(net, buses[7], buses[8], **line_args)
    pp.create_line_from_parameters(net, buses[8], buses[9], **line_args)
    pp.create_line_from_parameters(net, buses[9], buses[10], **line_args)

    # high meshing
    pp.create_line_from_parameters(net, buses[3], buses[6], **line_args)
    pp.create_line_from_parameters(net, buses[6], buses[10], **line_args)
    pp.create_line_from_parameters(net, buses[7], buses[2], **line_args)
    pp.create_line_from_parameters(net, buses[8], buses[5], **line_args)
    pp.create_line_from_parameters(net, buses[4], buses[1], **line_args)
    # loads &generators
    for b in [2, 3, 5, 6, 8, 9, 10]:
        pp.create_load(net, bus=buses[b], p_mw=0.5, q_mvar=0)
    for b in [4, 7]:
        pp.create_sgen(net, bus=buses[b], p_mw=1.0, vm_pu=1.0, max_q_mvar=0.0, min_q_mvar=0.0)
    return net

def monee_create_11bus_high_meshed():
    net = mx.create_multi_energy_network()
    buses = [mx.create_bus(net) for _ in range(11)]

    mx.create_ext_power_grid(net, buses[0])

    line_args = dict(length_m=100, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)

    # strings
    mx.create_line(net, buses[0], buses[1], **line_args)
    mx.create_line(net, buses[1], buses[2], **line_args)
    mx.create_line(net, buses[2], buses[3], **line_args)

    mx.create_line(net, buses[0], buses[4], **line_args)
    mx.create_line(net, buses[4], buses[5], **line_args)
    mx.create_line(net, buses[5], buses[6], **line_args)

    mx.create_line(net, buses[0], buses[7], **line_args)
    mx.create_line(net, buses[7], buses[8], **line_args)
    mx.create_line(net, buses[8], buses[9], **line_args)
    mx.create_line(net, buses[9], buses[10], **line_args)

    # low meshing
    mx.create_line(net, buses[3], buses[6], **line_args)
    mx.create_line(net, buses[6], buses[10], **line_args)
    mx.create_line(net, buses[7], buses[2], **line_args)
    mx.create_line(net, buses[8], buses[5], **line_args)
    mx.create_line(net, buses[4], buses[1], **line_args)

    for b in [2, 3, 5, 6, 8, 9, 10]:
        mx.create_power_load(net, buses[b], p_mw=0.5, q_mvar=0.0)
    for b in [4, 7]:
        mx.create_power_generator(net, buses[b], p_mw=1.0, q_mvar=0.0)
    return net

def pp_create_66bus_high_meshed():
    net = pp.create_empty_network()
    buses = [pp.create_bus(net, vn_kv=1) for _ in range(66)]
    pp.create_ext_grid(net, bus=buses[0], vm_pu=1.0)
    line_args = dict( length_km=0.1, r_ohm_per_km=0.07,  x_ohm_per_km=0.07, c_nf_per_km=0, max_i_ka=3.19)

    feeders = []
    idx = 1
    for f in range(6):
        chain = []
        prev = 0
        for _ in range(10):  # remaining nodes attached later
            if idx >= 66:
                break
            pp.create_line_from_parameters(net, buses[prev], buses[idx], **line_args)
            chain.append(idx)
            prev = idx
            idx += 1
        feeders.append(chain)

    while idx < 66:
        pp.create_line_from_parameters(net, buses[feeders[-1][-1]], buses[idx], **line_args)
        feeders[-1].append(idx)
        idx += 1

    for chain in feeders:
        for i in range(len(chain) - 2):
            pp.create_line_from_parameters( net, buses[chain[i]], buses[chain[i+2]], **line_args )

    for i in range(len(feeders)):
        for j in range(i+1, len(feeders)):
            for k in range(min(len(feeders[i]), len(feeders[j]))):
                if k % 2 == 0:
                    pp.create_line_from_parameters(net, buses[feeders[i][k]], buses[feeders[j][k]], **line_args)

    for i in range(1, 66, 3):
        j = (i + 7) % 66
        if j != 0:
            pp.create_line_from_parameters(net, buses[i], buses[j], **line_args)

    for i in range(1, 66):
        if i % 2 == 0:
            pp.create_load(net, bus=buses[i], p_mw=0.5, q_mvar=0)

    for i in range(1, 66):
        if i % 10 == 0:
            pp.create_sgen(net, bus=buses[i], p_mw=1.0, vm_pu=1.0, max_q_mvar=0.0, min_q_mvar=0.0)
    return net

def monee_create_66bus_high_meshed():

    net = mx.create_multi_energy_network()
    buses = [mx.create_bus(net) for _ in range(66)]
    mx.create_ext_power_grid(net, buses[0])
    line_args = dict(length_m=100, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)

    feeders = []
    idx = 1
    for f in range(6):
        chain = []
        prev = 0
        for _ in range(10):
            if idx >= 66:
                break
            mx.create_line(net, buses[prev], buses[idx], **line_args)
            chain.append(idx)
            prev = idx
            idx += 1
        feeders.append(chain)
    while idx < 66:
        mx.create_line(net, buses[feeders[-1][-1]], buses[idx], **line_args)
        feeders[-1].append(idx)
        idx += 1

    for chain in feeders:
        for i in range(len(chain) - 2):
            mx.create_line(net, buses[chain[i]], buses[chain[i+2]], **line_args)

    for i in range(len(feeders)):
        for j in range(i+1, len(feeders)):
            for k in range(min(len(feeders[i]), len(feeders[j]))):
                if k % 2 == 0:
                    mx.create_line(
                        net,
                        buses[feeders[i][k]],
                        buses[feeders[j][k]],
                        **line_args
                    )
    for i in range(1, 66, 3):
        j = (i + 7) % 66
        if j != 0:
            mx.create_line(net, buses[i], buses[j], **line_args)
    for i in range(1, 66):
        if i % 2 == 0:
            mx.create_power_load(net, buses[i], p_mw=0.5, q_mvar=0.0)
    for i in range(1, 66):
        if i % 10 == 0:
            mx.create_power_generator(net, buses[i], p_mw=1.0, q_mvar=0.0)
    return net

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

def monee_opf(monee_net, extgrid_bounds = (-10,10), show_details = False):
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
    return result

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
#PP_net = PP_create_two_line_power_example()
print("--------------Pandapower AC--------------------------")
PP_net = pp_create_11bus_high_meshed()
PP_results = PP_test_power_network(PP_net, AC = True, show_results = True)
print("--------------Monee AC--------------------------")
monee_net_AC = monee_create_11bus_high_meshed()
monee_net_AC.apply_formulation(AC_NETWORK_FORMULATION)
monee_result_AC = monee_test_power_network(monee_net_AC, show_results = True)
#For result table comparison

print("--------------Monee QC--------------------------")
monee_net_QC = monee_create_11bus_high_meshed()
monee_net_QC.apply_formulation(QC_NETWORK_FORMULATION)
monee_result_QC = monee_test_power_network(monee_net_QC, show_results = True)
#For result table comparison
'''
print("--------------Result comparison-------------------------")
print("Pandapower AC with Monee AC")
compare_line_results(PP_net, monee_result_AC)
print("Pandapower AC with Monee QC")
compare_line_results(PP_net, monee_result_QC)
print("Monee QC with Monee AC")
#compare_line_results(monee_result_QC, monee_result_AC)
'''
# OPF evaluation
print("-----------------------Monee OPF AC Grid------------------------------")
extgrid_bounds = (-0.5,0.5) #set bounds for both problems, other bounds are set internally (pandapower problem according to monee load_shedding)
monee_opf_result_AC = monee_opf(monee_net_AC, extgrid_bounds= extgrid_bounds, show_details = True)
print("-----------------------Monee OPF QC Grid------------------------------")
extgrid_bounds = (-0.5,0.5) #set bounds for both problems, other bounds are set internally (pandapower problem according to monee load_shedding)
monee_opf_result_QC = monee_opf(monee_net_QC, extgrid_bounds= extgrid_bounds, show_details = True)
print("-----------------------Pandapower OPF AC------------------------------")
pp_opf_net = pp_opf(PP_net,AC = True, extgrid_bounds = extgrid_bounds, show_details=True)
#print_pp_summary(pp_opf_net)
#print("Monee AC")
#print_monee_summary(monee_opf_result_AC)
#print("Monee QC")
#print_monee_summary(monee_opf_result_QC)
print("Compare monee results")
lines1_mismatches , lines2_mismatches = compare_monee_results(monee_opf_result_QC, monee_opf_result_AC)
plot_column_compare(lines1_mismatches, lines2_mismatches)

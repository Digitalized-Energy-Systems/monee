import networkx as nx
import pytest
import simbench

import monee
import monee.model as mm
from monee import run_energy_flow, run_timeseries
from monee.io.from_pandapower import from_pandapower_net
from monee.io.from_simbench import obtain_simbench_profile_by_pp_net
from monee.model import GasLinepack, LumpedThermalCapacitance
from monee.model.core import value as mvalue
from monee.model.formulation import (
    EL_MISOCP_FORMULATION,
    make_gas_milp_pwl_formulation,
    make_heat_convex_milp_formulation,
    make_smooth_nlp_formulation,
)
from monee.model.grid import DEFAULT_GAS_HHV_MJ_PER_KG
from monee.network import generate_supply_return_mes_based_on_power_net
from monee.problem.min_load_shedding import create_min_load_shedding_problem
from monee.simulation.timeseries import TimeseriesData
from monee.solver import GEKKOSolver

# Match the gas grid the generators build (lgas) so the expected CP-output
# computations use the same heating value as the sizing/physics.
GAS_HHV_MJ_PER_KG = DEFAULT_GAS_HHV_MJ_PER_KG


def _carrier_balance(mes):
    """Per-carrier capacity deficits; CP power draws use default efficiencies (p2g 0.7, p2h 0.95)."""
    p_load = sum(c.model.p_mw for c in mes.childs if isinstance(c.model, mm.PowerLoad))
    p_gen = sum(
        abs(c.model.p_mw) for c in mes.childs if isinstance(c.model, mm.PowerGenerator)
    )
    q_load = sum(
        c.model.q_mw_heat for c in mes.childs if isinstance(c.model, mm.HeatLoad)
    )
    q_gen = sum(
        abs(c.model.q_mw_heat)
        for c in mes.childs
        if isinstance(c.model, mm.HeatGenerator)
    )
    g_sink = sum(
        c.model.mass_flow_kgs
        for c in mes.childs
        if isinstance(c.model, mm.Sink) and isinstance(c.grid, mm.GasGrid)
    )
    g_source = sum(
        abs(c.model.mass_flow_kgs)
        for c in mes.childs
        if isinstance(c.model, mm.Source) and isinstance(c.grid, mm.GasGrid)
    )
    chp_p = sum(
        float(c.model.mass_flow_setpoint_kgs)
        * GAS_HHV_MJ_PER_KG
        * float(c.model.efficiency_power)
        for c in mes.compounds
        if "CHP" in type(c.model).__name__
    )
    chp_q = sum(
        float(c.model.mass_flow_setpoint_kgs)
        * GAS_HHV_MJ_PER_KG
        * float(c.model.efficiency_heat)
        for c in mes.compounds
        if "CHP" in type(c.model).__name__
    )
    chp_gas = sum(
        float(c.model.mass_flow_setpoint_kgs)
        for c in mes.compounds
        if "CHP" in type(c.model).__name__
    )
    p2g_g = sum(
        abs(float(b.model.gas_mass_flow_kgs))
        for b in mes.branches
        if "PowerToGas" in type(b.model).__name__
    )
    p2h_q = sum(
        float(b.model.heat_energy_mw)
        for b in mes.branches
        if type(b.model).__name__ == "PowerToHeatHG"
    )
    p2g_p_in = p2g_g * GAS_HHV_MJ_PER_KG / 0.7
    p2h_p_in = p2h_q / 0.95

    p_def = max(0.0, p_load + p2g_p_in + p2h_p_in - p_gen - chp_p)
    q_def = max(0.0, q_load - q_gen - chp_q - p2h_q)
    g_def = max(0.0, g_sink + chp_gas - g_source - p2g_g)
    return (p_def, p_load), (q_def, q_load), (g_def, g_sink)


def _reg_val(model):
    reg = model.regulation
    return float(reg.value if hasattr(reg, "value") else reg)


def _total_shed(solved):
    """Aggregate shed power [MW], heat [MW] and gas [kg/s] of a solved network."""
    p_shed = sum(
        c.model.p_mw * (1 - _reg_val(c.model))
        for c in solved.childs
        if isinstance(c.model, mm.PowerLoad)
    )
    q_shed = sum(
        c.model.q_mw_heat * (1 - _reg_val(c.model))
        for c in solved.childs
        if isinstance(c.model, mm.HeatLoad)
    )
    g_shed = sum(
        c.model.mass_flow_kgs * (1 - _reg_val(c.model))
        for c in solved.childs
        if isinstance(c.model, mm.Sink) and isinstance(c.grid, mm.GasGrid)
    )
    return p_shed, q_shed, g_shed


def _gas_graph(mes):
    g = nx.Graph()
    for n in mes.nodes:
        if isinstance(n.model, mm.Junction) and isinstance(n.grid, mm.GasGrid):
            g.add_node(n.id)
    for b in mes.branches:
        if isinstance(b.model, mm.GasPipe):
            g.add_edge(b.from_node_id, b.to_node_id)
    return g


def _independent_cycles(g):
    return g.number_of_edges() - g.number_of_nodes() + nx.number_connected_components(g)


def _cp_totals(mes):
    """Rated CP outputs: CHP power [MW], P2G gas [kg/s], P2H heat [MW]."""
    chp_p = sum(
        float(c.model.mass_flow_setpoint_kgs)
        * GAS_HHV_MJ_PER_KG
        * float(c.model.efficiency_power)
        for c in mes.compounds
        if "CHP" in type(c.model).__name__
    )
    p2g_kgs = sum(
        abs(float(b.model.gas_mass_flow_kgs))
        for b in mes.branches
        if "PowerToGas" in type(b.model).__name__
    )
    p2h_q = sum(
        float(b.model.heat_energy_mw)
        for b in mes.branches
        if type(b.model).__name__ == "PowerToHeatHG"
    )
    return chp_p, p2g_kgs, p2h_q


def _cp_heat_out_mw(mes):
    """Rated CP heat output: CHP heat plus P2H heat [MW]."""
    chp_heat = sum(
        float(c.model.mass_flow_setpoint_kgs)
        * GAS_HHV_MJ_PER_KG
        * float(c.model.efficiency_heat)
        for c in mes.compounds
        if "CHP" in type(c.model).__name__
    )
    p2h_heat = sum(
        float(b.model.heat_energy_mw)
        for b in mes.branches
        if type(b.model).__name__ == "PowerToHeatHG"
    )
    return chp_heat + p2h_heat


def _primary_power_mw(mes):
    return sum(
        abs(c.model.p_mw) for c in mes.childs if isinstance(c.model, mm.PowerGenerator)
    )


def _primary_gas_kgs(mes):
    return sum(
        abs(c.model.mass_flow_kgs)
        for c in mes.childs
        if isinstance(c.model, mm.Source) and isinstance(c.grid, mm.GasGrid)
    )


def _heat_slack(mes):
    """The single hot (t_k >= 350) water-grid slack of the MES."""
    slacks = [
        c
        for c in mes.childs
        if isinstance(c.model, mm.ExtHydrGrid)
        and isinstance(c.grid, mm.WaterGrid)
        and float(getattr(c.model, "t_k", 0.0)) >= 350
    ]
    assert len(slacks) == 1
    return slacks[0]


def _slack_t_k(mes):
    """t_k of the single water-grid slack of the MES."""
    slacks = [
        c
        for c in mes.childs
        if isinstance(c.model, mm.ExtHydrGrid) and isinstance(c.grid, mm.WaterGrid)
    ]
    assert len(slacks) == 1
    return slacks[0].model.t_k


def _build_storage_timeseries_data(mes, full_el_td, steps, factors):
    """Slice simbench profiles to ``steps`` and add a synthetic heat/gas demand profile."""
    td = TimeseriesData()
    for name, attrs in full_el_td.child_name_data.items():
        for attr, series in attrs.items():
            td.add_child_series_by_name(name, attr, list(series[:steps]))
    for c in mes.childs:
        if isinstance(c.model, mm.HeatLoad):
            base = float(mvalue(c.model.q_mw_heat))
            if base == 0:
                continue
            td.add_child_series(c.id, "q_mw_heat", [base * f for f in factors])
        elif isinstance(c.model, mm.Sink):
            base = float(mvalue(c.model.mass_flow_kgs))
            if base == 0:
                continue
            td.add_child_series(c.id, "mass_flow_kgs", [base * f for f in factors])
    return td


def create_large_lv_simbench(
    density,
    *,
    slack_budget_pct: float | None = None,
    simbench_code: str = "1-LV-rural3--1-no_sw",
    backup_lines_per_sector: int = 0,
    backup_seed: int | None = None,
    cp_size_multiplier: float = 1.0,
    replace_primary_generation: bool = False,
):

    def create():
        net = simbench.get_simbench_net(simbench_code)
        mn = from_pandapower_net(net)
        mes = generate_supply_return_mes_based_on_power_net(
            mn,
            coupling_density=density,
            centralized=False,
            couplings=("chp", "p2g", "p2h"),
            coupling_kwargs={
                "seed": 1,
                "use_hg_variants": True,
                "cp_size_multiplier": cp_size_multiplier,
                "replace_primary_generation": replace_primary_generation,
            },
            heat_kwargs={"node_based_heat_loads": True},
        )
        mes.apply_formulation(EL_MISOCP_FORMULATION)
        return mes

    return create


def create_large_mv_simbench(
    density,
    *,
    slack_budget_pct: float | None = None,
    simbench_code: str = "1-MV-urban--1-no_sw",
    backup_lines_per_sector: int = 0,
    backup_seed: int | None = None,
    cp_size_multiplier: float = 1.0,
    replace_primary_generation: bool = False,
):

    def create():
        net = simbench.get_simbench_net(simbench_code)
        mn = from_pandapower_net(net)
        mes = generate_supply_return_mes_based_on_power_net(
            mn,
            coupling_density=density,
            centralized=False,
            couplings=("chp", "p2g", "p2h"),
            coupling_kwargs={
                "seed": 1,
                "use_hg_variants": True,
                "cp_size_multiplier": cp_size_multiplier,
                "replace_primary_generation": replace_primary_generation,
            },
            heat_kwargs={"node_based_heat_loads": True, "auto_diameter": True},
        )
        for branch in mes.branches:
            model = branch.model
            length = getattr(model, "length_m", None)
            if length is not None and float(length) <= 0.0:
                model.length_m = 1
        mes.apply_formulation(make_smooth_nlp_formulation())
        return mes

    return create


@pytest.mark.pptest
def test_generate_scare():

    # GIVEN
    net = create_large_lv_simbench(0.3)()

    # WHEN
    result = run_energy_flow(net, solver="gurobi")

    # THEN
    assert result.success


@pytest.mark.pptest
def test_generate_synapse():

    # GIVEN
    net = create_large_mv_simbench(0)()

    # WHEN
    result = run_energy_flow(net, solver=GEKKOSolver(solver=3))

    # THEN
    assert result.success


@pytest.mark.pptest
def test_generate_mes():

    # GIVEN
    net = simbench.get_simbench_net("1-LV-rural3--1-no_sw")
    mn = from_pandapower_net(net)
    mes = generate_supply_return_mes_based_on_power_net(
        mn,
        coupling_density=0.20,
        centralized=False,
        couplings=("chp", "p2g", "p2h"),
        coupling_kwargs={
            "seed": 1,
            "use_hg_variants": True,
            "chp_p_share": 2.0,
            "p2g_p_share": 0.3,
            "p2h_p_share": 0.5,
            "cp_size_multiplier": 3.0,
        },
        heat_kwargs={
            "node_based_heat_loads": True,
            "node_heat_gen_share": 3.0,
        },
        gas_kwargs={
            "gas_gen_share": 8.0,
            "mesh_seed": 42,
        },
    )

    # WHEN
    (p_def, p_load), (q_def, q_load), (g_def, g_sink) = _carrier_balance(mes)
    mes.apply_formulation(EL_MISOCP_FORMULATION)
    mes.apply_formulation(make_heat_convex_milp_formulation(num_partitions=1, include_heat_exchangers=False))
    problem = create_min_load_shedding_problem(
        bounds_vm=(0.9, 1.1),
        bounds_pressure=(0.9, 1.1),
        bounds_t=(0.7, 1.3),
        bounds_ext_el=(-0.10, 0.10),
        bounds_ext_gas=(-0.02, 0.02),
        bounds_ext_heat=(-100, 100),
        include_ext_grids=True,
        auto_priority_floor=True,
    )
    result = monee.run_energy_flow_optimization(
        mes, optimization_problem=problem, solver="gurobi"
    )

    # THEN
    assert result.success, "baseline solve must converge before any contingency"

    # Per-carrier balance check - capacity-only, no solve.
    p_self = 1 - p_def / p_load
    q_self = 1 - q_def / q_load
    g_self = 1 - g_def / g_sink
    print(
        f"Self-sufficiency:  P={100 * p_self:.1f} %  "
        f"Q={100 * q_self:.1f} %  G={100 * g_self:.1f} %"
    )
    # Power self-sufficiency is intentionally lower than heat/gas: the coupling
    # builder adds P2G/P2H units that *draw* power and only drains primary
    # PowerGenerators by the CHP output (no compensation for the P2G/P2H draws),
    # so the coupled grid is a net power importer by design. With the parameters
    # above the seed=1 network self-supplies ~76 % of its power.
    assert p_self >= 0.70, f"power self-sufficiency {100 * p_self:.1f} % < 70 %"
    assert q_self >= 0.90, f"heat  self-sufficiency {100 * q_self:.1f} % < 90 %"
    assert g_self >= 0.90, f"gas   self-sufficiency {100 * g_self:.1f} % < 90 %"

    # Baseline min-load-shedding solve must shed nothing.
    p_shed, q_shed, g_shed = _total_shed(result.network)
    assert p_shed < 1e-6, f"baseline power shed {p_shed:.6g} MW > tol"
    assert q_shed < 1e-6, f"baseline heat  shed {q_shed:.6g} MW > tol"
    assert g_shed < 1e-9, f"baseline gas   shed {g_shed:.6g} kg/s > tol"


@pytest.mark.pptest
def test_generate_mes_min_load_shedding():

    # GIVEN
    net = simbench.get_simbench_net("1-LV-rural3--1-no_sw")
    mn = from_pandapower_net(net)
    mes = generate_supply_return_mes_based_on_power_net(
        mn,
        coupling_density=0.5,
        centralized=False,
        couplings=("chp", "p2g", "p2h"),
        coupling_kwargs={"seed": 1, "use_hg_variants": True},
        heat_kwargs={"node_based_heat_loads": True},
    )
    mes.apply_formulation(EL_MISOCP_FORMULATION)
    # num_partitions=4 tightens the piecewise-McCormick relaxation enough to keep
    # junction temperatures inside the [0.75, 1.15] envelope asserted below; with
    # num_partitions=1 the LP corner legitimately drops to the 0.7 problem bound.
    mes.apply_formulation(make_heat_convex_milp_formulation(num_partitions=4, include_heat_exchangers=False))
    problem = create_min_load_shedding_problem(
        bounds_vm=(0.9, 1.5),
        bounds_pressure=(0.9, 1.5),
        bounds_t=(0.7, 1.3),
        bounds_ext_el=(-0.01, 0.01),
        bounds_ext_gas=(-0.01, 0.01),
        bounds_ext_heat=(-100, 100),
        include_ext_grids=True,
        check_vm=True,
        check_pressure=True,
        check_t=True,
        check_lp=True,
        auto_priority_floor=True,
        debug=True,
    )

    # WHEN
    result = monee.run_energy_flow_optimization(
        mes,
        optimization_problem=problem,
        solver="gurobi",
    )

    # THEN
    assert result.success
    assert result is not None
    assert result.objective is not None

    # Only the McCormick-DHS + node-based + HG-variants stack achieves near-full delivery.
    solved = result.network
    heat_loads = [c for c in solved.childs if isinstance(c.model, mm.HeatLoad)]
    total_demand = sum(c.model.q_mw_heat for c in heat_loads)
    delivered = sum(
        c.model.q_mw_heat * float(mvalue(c.model.regulation)) for c in heat_loads
    )
    delivery_ratio = delivered / total_demand if total_demand else 0.0
    assert delivery_ratio >= 0.95, (
        f"expected ≥95 % heat delivery, got {100 * delivery_ratio:.1f} % "
        f"({delivered * 1e3:.1f} kW of {total_demand * 1e3:.1f} kW)"
    )

    # Envelope min is 0.75 (≈ 267 K); small slack for piecewise selector rounding.
    water_juncs = [
        n
        for n in solved.nodes
        if isinstance(n.model, mm.Junction)
        and isinstance(n.grid, mm.WaterGrid)
        and not getattr(n, "ignored", False)
    ]
    t_pus = [float(mvalue(n.model.t_pu)) for n in water_juncs]
    if t_pus:
        assert min(t_pus) >= 0.75 - 1e-3, (
            f"junction t_pu fell below envelope: min={min(t_pus):.4f}"
        )


@pytest.mark.pptest
def test_generate_mes_gas_extra_mesh_pipes_reduce_bridges():

    # GIVEN
    net = simbench.get_simbench_net("1-LV-rural3--1-no_sw")
    mn = from_pandapower_net(net)
    common = dict(
        coupling_density=0.5,
        centralized=False,
        couplings=("chp", "p2g", "p2h"),
        coupling_kwargs={"seed": 1, "use_hg_variants": True},
        heat_kwargs={"node_based_heat_loads": True},
    )

    # WHEN
    mes_tree = generate_supply_return_mes_based_on_power_net(mn, **common)
    mes_mesh = generate_supply_return_mes_based_on_power_net(
        mn, gas_kwargs={"extra_mesh_pipes": 20, "mesh_seed": 42}, **common
    )
    g_tree, g_mesh = _gas_graph(mes_tree), _gas_graph(mes_mesh)

    # THEN
    # Tree baseline: 0 cycles, so every edge is a bridge.
    assert _independent_cycles(g_tree) == 0
    assert len(list(nx.bridges(g_tree))) == g_tree.number_of_edges()

    # Mesh: exactly 20 extra edges create 20 cycles and far fewer bridges.
    assert _independent_cycles(g_mesh) == 20
    assert g_mesh.number_of_edges() == g_tree.number_of_edges() + 20
    assert len(list(nx.bridges(g_mesh))) < 0.6 * g_tree.number_of_edges()


@pytest.mark.pptest
def test_generate_mes_cp_size_multiplier_scales_uniformly():

    # GIVEN
    net = simbench.get_simbench_net("1-LV-rural3--1-no_sw")
    mn = from_pandapower_net(net)
    common = dict(
        coupling_density=0.5,
        centralized=False,
        couplings=("chp", "p2g", "p2h"),
        heat_kwargs={"node_based_heat_loads": True},
    )

    def build(**ck):
        base = {"seed": 1, "use_hg_variants": True}
        base.update(ck)
        return generate_supply_return_mes_based_on_power_net(
            mn, coupling_kwargs=base, **common
        )

    # WHEN
    chp_d, p2g_d, p2h_d = _cp_totals(build())
    chp_1, p2g_1, p2h_1 = _cp_totals(build(cp_size_multiplier=1.0))
    chp_2, p2g_2, p2h_2 = _cp_totals(build(cp_size_multiplier=2.0))
    chp_4, p2g_4, p2h_4 = _cp_totals(build(cp_size_multiplier=4.0))
    chp_s, p2g_s, p2h_s = _cp_totals(build(chp_p_share=1.0))

    # THEN
    # Default and explicit multiplier=1.0 must be identical.
    assert (chp_d, p2g_d, p2h_d) == (chp_1, p2g_1, p2h_1)
    assert chp_d > 0 and p2g_d > 0 and p2h_d > 0

    # Linear scaling across all carriers; 2 % tolerance absorbs per-CP round(..., 6) quantisation.
    REL = 2e-2
    assert chp_2 == pytest.approx(2 * chp_1, rel=REL)
    assert p2g_2 == pytest.approx(2 * p2g_1, rel=REL)
    assert p2h_2 == pytest.approx(2 * p2h_1, rel=REL)
    assert chp_4 == pytest.approx(4 * chp_1, rel=REL)
    assert p2g_4 == pytest.approx(4 * p2g_1, rel=REL)
    assert p2h_4 == pytest.approx(4 * p2h_1, rel=REL)

    # chp_p_share=1.0 (vs default 0.5) doubles CHP only, P2G/P2H untouched.
    assert chp_s == pytest.approx(2 * chp_1, rel=REL)
    assert p2g_s == pytest.approx(p2g_1, rel=1e-9)
    assert p2h_s == pytest.approx(p2h_1, rel=1e-9)


@pytest.mark.pptest
def test_generate_mes_replace_primary_generation_invariant():

    # GIVEN
    net = simbench.get_simbench_net("1-LV-rural3--1-no_sw")
    mn = from_pandapower_net(net)
    common = dict(
        coupling_density=0.5,
        centralized=False,
        couplings=("chp", "p2g", "p2h"),
        heat_kwargs={"node_based_heat_loads": True},
    )

    # WHEN
    mes_add = generate_supply_return_mes_based_on_power_net(
        mn,
        coupling_kwargs={"seed": 1, "use_hg_variants": True},
        **common,
    )
    mes_rep = generate_supply_return_mes_based_on_power_net(
        mn,
        coupling_kwargs={
            "seed": 1,
            "use_hg_variants": True,
            "replace_primary_generation": True,
        },
        **common,
    )

    # THEN
    n_chp_add = sum(1 for c in mes_add.compounds if "CHP" in type(c.model).__name__)
    n_chp_rep = sum(1 for c in mes_rep.compounds if "CHP" in type(c.model).__name__)
    n_p2g_add = sum(
        1 for b in mes_add.branches if "PowerToGas" in type(b.model).__name__
    )
    n_p2g_rep = sum(
        1 for b in mes_rep.branches if "PowerToGas" in type(b.model).__name__
    )
    assert n_chp_add == n_chp_rep > 0, (
        "CP placement should be deterministic across modes"
    )
    assert n_p2g_add == n_p2g_rep > 0

    # Replacement strictly reduces the primary power pool by exactly the rated CP power.
    pri_p_add = _primary_power_mw(mes_add)
    pri_p_rep = _primary_power_mw(mes_rep)
    cp_p, cp_g, _ = _cp_totals(mes_add)
    assert cp_p > 0
    assert pri_p_rep < pri_p_add - 1e-9
    assert pri_p_rep == pytest.approx(pri_p_add - cp_p, abs=1e-9)

    pri_g_add = _primary_gas_kgs(mes_add)
    pri_g_rep = _primary_gas_kgs(mes_rep)
    assert cp_g > 0
    assert pri_g_rep < pri_g_add - 1e-12
    assert pri_g_rep == pytest.approx(pri_g_add - cp_g, abs=1e-9)

    # Without node_heat_gen_share the heat replacement is a no-op: slack stays unbounded.
    assert _heat_slack(mes_add).model.mass_flow_kgs.min is None
    assert _heat_slack(mes_rep).model.mass_flow_kgs.min is None


@pytest.mark.pptest
def test_generate_mes_node_heat_gen_share_and_replacement():

    # GIVEN
    net = simbench.get_simbench_net("1-LV-rural3--1-no_sw")
    mn = from_pandapower_net(net)
    common = dict(
        coupling_density=0.5,
        centralized=False,
        couplings=("chp", "p2g", "p2h"),
    )

    # WHEN
    mes_no_hg = generate_supply_return_mes_based_on_power_net(
        mn,
        coupling_kwargs={"seed": 1, "use_hg_variants": True},
        heat_kwargs={"node_based_heat_loads": True, "node_heat_gen_share": 0.0},
        **common,
    )
    mes_add = generate_supply_return_mes_based_on_power_net(
        mn,
        coupling_kwargs={"seed": 1, "use_hg_variants": True},
        heat_kwargs={"node_based_heat_loads": True},
        **common,
    )
    mes_rep = generate_supply_return_mes_based_on_power_net(
        mn,
        coupling_kwargs={
            "seed": 1,
            "use_hg_variants": True,
            "replace_primary_generation": True,
        },
        heat_kwargs={"node_based_heat_loads": True, "node_heat_gen_share": 1.0},
        **common,
    )

    # THEN
    # share=0.0 disables the distributed heat fleet (legacy).
    assert not [c for c in mes_no_hg.childs if isinstance(c.model, mm.HeatGenerator)], (
        "node_heat_gen_share=0.0 must not create HeatGenerator children"
    )

    # Default share=1.0 distributes HeatGenerators at PowerGenerator buses.
    hg_add = [c for c in mes_add.childs if isinstance(c.model, mm.HeatGenerator)]
    assert hg_add, "node_heat_gen_share > 0 must distribute HeatGenerators"
    total_hg_add = sum(abs(c.model.q_mw_heat) for c in hg_add)
    assert total_hg_add > 0

    # Replacement mode drains the HeatGenerator pool by exactly the CP heat output.
    hg_rep = [c for c in mes_rep.childs if isinstance(c.model, mm.HeatGenerator)]
    total_hg_rep = sum(abs(c.model.q_mw_heat) for c in hg_rep)
    cp_heat = _cp_heat_out_mw(mes_rep)
    assert cp_heat > 0
    assert total_hg_rep < total_hg_add - 1e-9
    assert total_hg_rep == pytest.approx(total_hg_add - cp_heat, abs=1e-9)


@pytest.mark.pptest
def test_generate_mes_supply_slack_t_k_parameter():

    # GIVEN
    net = simbench.get_simbench_net("1-LV-rural3--1-no_sw")
    mn = from_pandapower_net(net)

    # WHEN
    mes_default = generate_supply_return_mes_based_on_power_net(
        mn,
        coupling_kwargs={"seed": 1, "use_hg_variants": True},
        heat_kwargs={"node_based_heat_loads": True},
    )
    mes_lowered = generate_supply_return_mes_based_on_power_net(
        mn,
        coupling_kwargs={"seed": 1, "use_hg_variants": True},
        heat_kwargs={"node_based_heat_loads": True, "supply_slack_t_k": 330.0},
    )

    # THEN
    # 356 is the legacy default.
    assert _slack_t_k(mes_default) == 356
    assert _slack_t_k(mes_lowered) == 330.0


@pytest.mark.pptest
def test_generate_mes_storage_capabilities_timeseries():

    # GIVEN
    pp_net = simbench.get_simbench_net("1-LV-rural3--1-no_sw")
    full_el_td = obtain_simbench_profile_by_pp_net(pp_net)
    mn = from_pandapower_net(pp_net)
    mes = generate_supply_return_mes_based_on_power_net(
        mn,
        coupling_density=0.5,
        centralized=False,
        couplings=("chp", "p2g", "p2h"),
        coupling_kwargs={"seed": 1, "use_hg_variants": True},
        heat_kwargs={"node_based_heat_loads": True},
    )
    mes.apply_formulation(EL_MISOCP_FORMULATION)
    mes.apply_formulation(make_heat_convex_milp_formulation(num_partitions=1, include_heat_exchangers=False))
    mes.apply_formulation(make_gas_milp_pwl_formulation())
    mes.add_extension(GasLinepack())
    mes.add_extension(LumpedThermalCapacitance())

    # Synthetic low / peak / recovery profile so linepack and LTC see a varying signal.
    steps = 3
    factors = [0.7, 1.3, 0.9]
    td = _build_storage_timeseries_data(mes, full_el_td, steps, factors)

    problem = create_min_load_shedding_problem(
        bounds_vm=(0.9, 1.5),
        bounds_pressure=(0.9, 1.5),
        bounds_t=(0.7, 1.3),
        bounds_ext_el=(-5, 5),
        bounds_ext_gas=(-5, 5),
        bounds_ext_heat=(-100, 100),
        include_ext_grids=True,
        check_vm=True,
        check_pressure=True,
        check_t=True,
        check_lp=True,
    )

    # WHEN
    ts_result = run_timeseries(
        mes,
        timeseries_data=td,
        steps=steps,
        optimization_problem=problem,
        solver="gurobi",
    )

    # THEN
    assert not ts_result.failed_steps, (
        f"timeseries had failed steps: {ts_result.failed_steps}"
    )
    assert len(ts_result.step_results) == steps

    # Linepack: positive gas mass on every pipe; at least one pipe packs
    # inter-temporally (anchor step t=0 is allowed to be 0).
    gas_pipes = [b for b in mes.branches if isinstance(b.model, mm.GasPipe)]
    assert gas_pipes, "no gas pipes in generated MES"
    saw_packing = False
    for pipe in gas_pipes:
        lp = ts_result.get_result_for_id(pipe.id, "linepack_kg").dropna()
        npk = ts_result.get_result_for_id(pipe.id, "net_pack_kgs").dropna()
        assert len(lp) == steps, (
            f"linepack_kg missing on pipe {pipe.id}: got {len(lp)} of {steps}"
        )
        assert (lp > 0).all(), (
            f"linepack_kg should be strictly positive (gas mass), got {list(lp)}"
        )
        if len(npk) > 1 and float(npk.iloc[1:].abs().max()) > 1e-9:
            saw_packing = True
    assert saw_packing, (
        "no gas pipe exhibited inter-temporal packing flow; the linepack "
        "extension's temporal coupling does not appear to have been active"
    )

    # LTC: junction t_pu stays inside the McCormick envelope on every step.
    t_low, t_high = 0.7 - 1e-3, 1.3 + 1e-3
    water_junc_ids = [
        n.id
        for n in mes.nodes
        if isinstance(n.model, mm.Junction) and isinstance(n.grid, mm.WaterGrid)
    ]
    assert water_junc_ids, "no water junctions in generated MES"
    saw_junction_series = False
    for jid in water_junc_ids:
        s = ts_result.get_result_for_id(jid, "t_pu").dropna()
        if s.empty:
            continue
        saw_junction_series = True
        assert s.min() >= t_low, (
            f"junction {jid} t_pu fell below envelope: min={s.min():.4f}"
        )
        assert s.max() <= t_high, (
            f"junction {jid} t_pu exceeded envelope: max={s.max():.4f}"
        )
    assert saw_junction_series, "no junction t_pu series recovered from result"

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
    MISOCP_NETWORK_FORMULATION,
    make_mccormick_dhs_formulation,
    make_nl_weymouth_pwl_network_formulation,
    make_smooth_network_formulation,
)
from monee.network import generate_supply_return_mes_based_on_power_net
from monee.problem.min_load_shedding import create_min_load_shedding_problem
from monee.simulation.timeseries import TimeseriesData
from monee.solver import GEKKOSolver

GAS_HHV_MJ_PER_KG = 15.3 * 3.6


def _carrier_balance(mes):
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
        c.model.mass_flow
        for c in mes.childs
        if isinstance(c.model, mm.Sink) and isinstance(c.grid, mm.GasGrid)
    )
    g_source = sum(
        abs(c.model.mass_flow)
        for c in mes.childs
        if isinstance(c.model, mm.Source) and isinstance(c.grid, mm.GasGrid)
    )
    chp_p = sum(
        float(c.model.mass_flow_setpoint)
        * GAS_HHV_MJ_PER_KG
        * float(c.model.efficiency_power)
        for c in mes.compounds
        if "CHP" in type(c.model).__name__
    )
    chp_q = sum(
        float(c.model.mass_flow_setpoint)
        * GAS_HHV_MJ_PER_KG
        * float(c.model.efficiency_heat)
        for c in mes.compounds
        if "CHP" in type(c.model).__name__
    )
    chp_gas = sum(
        float(c.model.mass_flow_setpoint)
        for c in mes.compounds
        if "CHP" in type(c.model).__name__
    )
    p2g_g = sum(
        abs(float(b.model.gas_kgps))
        for b in mes.branches
        if "PowerToGas" in type(b.model).__name__
    )
    p2h_q = sum(
        float(b.model.heat_energy_mw)
        for b in mes.branches
        if type(b.model).__name__ == "PowerToHeatHG"
    )
    # CP power draws - approximate via the default efficiencies the
    # generator function uses (p2g 0.7, p2h 0.95); good to ±5 %.
    p2g_p_in = p2g_g * GAS_HHV_MJ_PER_KG / 0.7
    p2h_p_in = p2h_q / 0.95

    p_def = max(0.0, p_load + p2g_p_in + p2h_p_in - p_gen - chp_p)
    q_def = max(0.0, q_load - q_gen - chp_q - p2h_q)
    g_def = max(0.0, g_sink + chp_gas - g_source - p2g_g)
    return (p_def, p_load), (q_def, q_load), (g_def, g_sink)


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
        mes.apply_formulation(MISOCP_NETWORK_FORMULATION)
        # mes.add_extension(GasLinepack())
        # mes.add_extension(LumpedThermalCapacitance(first_step_steady_state=True))
        # mes.apply_formulation(make_mccormick_dhs_formulation(num_partitions=16))
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
        # Combined smooth AC + gas + water formulation. Solved via
        # run_energy_flow (simulation=True), the solver squares it into a DOF=0
        # IMODE=1 steady state under IPOPT.
        mes.apply_formulation(make_smooth_network_formulation())
        return mes

    return create


@pytest.mark.pptest
def test_generate_scare():
    net = create_large_lv_simbench(0.3)()
    result = run_energy_flow(net, solver="gurobi")

    assert result.success


@pytest.mark.pptest
def test_generate_synapse():
    net = create_large_mv_simbench(0)()
    result = run_energy_flow(net, solver=GEKKOSolver(solver=3))
    assert result.success


@pytest.mark.pptest
def test_generate_mes():
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

    # (1) Per-carrier balance check - capacity-only, no solve.
    (p_def, p_load), (q_def, q_load), (g_def, g_sink) = _carrier_balance(mes)
    p_self = 1 - p_def / p_load
    q_self = 1 - q_def / q_load
    g_self = 1 - g_def / g_sink
    print(
        f"Self-sufficiency:  P={100 * p_self:.1f} %  "
        f"Q={100 * q_self:.1f} %  G={100 * g_self:.1f} %"
    )
    assert p_self >= 0.90, f"power self-sufficiency {100 * p_self:.1f} % < 90 %"
    assert q_self >= 0.90, f"heat  self-sufficiency {100 * q_self:.1f} % < 90 %"
    assert g_self >= 0.90, f"gas   self-sufficiency {100 * g_self:.1f} % < 90 %"

    # (2) Baseline min-load-shedding solve must shed nothing - the
    # restoration metric is meaningful only if shed_baseline = 0.
    mes.apply_formulation(MISOCP_NETWORK_FORMULATION)
    mes.apply_formulation(make_mccormick_dhs_formulation(num_partitions=1))

    problem = create_min_load_shedding_problem(
        bounds_el=(0.9, 1.1),
        bounds_gas=(0.9, 1.1),
        bounds_heat=(0.7, 1.3),
        # Bounds sized at ~10–15 % of per-carrier nominal load.  The
        # numbers below correspond to:
        #   power: ±0.10 MW    (~27 % of 0.374 MW load - covers baseline
        #                       shortfall ≈ 0.005 MW + worst single PowerGen
        #                       loss ≈ 30 kW + worst single CHP power loss
        #                       ≈ 25 kW + headroom for re-dispatch)
        #   gas:   ±0.02 kg/s  (~80 % of 0.025 kg/s sink - covers baseline
        #                       shortfall ≈ 0.002 kg/s + worst single Source
        #                       loss ≈ 4 g/s + worst single CHP intake loss
        #                       + headroom)
        #   heat:  ±100 kg/s   (effectively unbounded; the water slack
        #                       absorbs whatever mass flow the consumer
        #                       sinks require - not a smooth energy dial)
        ext_grid_el_bounds=(-0.10, 0.10),
        ext_grid_gas_bounds=(-0.02, 0.02),
        ext_grid_heat_bounds=(-100, 100),
        include_ext_grids=True,
        auto_priority_floor=True,
    )
    result = monee.run_energy_flow_optimization(
        mes, optimization_problem=problem, solver="gurobi"
    )
    assert result.success, "baseline solve must converge before any contingency"

    solved = result.network

    def _reg_val(model):
        reg = model.regulation
        return float(reg.value if hasattr(reg, "value") else reg)

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
        c.model.mass_flow * (1 - _reg_val(c.model))
        for c in solved.childs
        if isinstance(c.model, mm.Sink) and isinstance(c.grid, mm.GasGrid)
    )
    assert p_shed < 1e-6, f"baseline power shed {p_shed:.6g} MW > tol"
    assert q_shed < 1e-6, f"baseline heat  shed {q_shed:.6g} MW > tol"
    assert g_shed < 1e-9, f"baseline gas   shed {g_shed:.6g} kg/s > tol"


@pytest.mark.pptest
def test_generate_mes_min_load_shedding():
    """Canonical MES min-load-shedding configuration.

    Uses MISOCP for electricity and McCormick-DHS for heat with the
    HeatGenerator-style coupling-point variants - the only combination
    that delivers the full HX demand under min-load-shedding (the linear-
    HX + closing-pipe pairing collapses the supply/return ΔT and meets
    only ~25 % of design).  ``num_partitions=4`` shrinks the McCormick LP
    relaxation gap enough to keep junction temperatures inside the
    ``[267 K, 409 K]`` envelope; without it the LP corner can drop below
    freezing.
    """
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
    mes.apply_formulation(MISOCP_NETWORK_FORMULATION)
    mes.apply_formulation(make_mccormick_dhs_formulation(num_partitions=1))
    # mes.apply_formulation(make_nl_weymouth_pwl_network_formulation())

    problem = create_min_load_shedding_problem(
        bounds_el=(0.9, 1.5),
        bounds_gas=(0.9, 1.5),
        bounds_heat=(0.7, 1.3),
        ext_grid_el_bounds=(-0.01, 0.01),
        ext_grid_gas_bounds=(-0.01, 0.01),
        ext_grid_heat_bounds=(-100, 100),
        include_ext_grids=True,
        check_vm=True,
        check_pressure=True,
        check_temperature=True,
        check_line_loading=True,
        auto_priority_floor=True,
        debug=True,
    )

    # WHEN
    result = monee.run_energy_flow_optimization(
        mes,
        optimization_problem=problem,
        solver="gurobi",
    )
    # print(result)
    # result = run_energy_flow(mes, solver="gurobi")

    # THEN
    assert result is not None
    assert result.success, "min load shedding did not converge"
    assert result.objective is not None
    print(result.objective)
    assert False
    # Aggregate heat delivery should be ≥ 95 % of design - the
    # McCormick-DHS + node-based + HG-variants stack is the only one that
    # achieves near-full delivery.
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
    # Sanity-check the McCormick relaxation: junction temperatures stay
    # inside the envelope (no sub-freezing artifacts).
    water_juncs = [
        n
        for n in solved.nodes
        if isinstance(n.model, mm.Junction)
        and isinstance(n.grid, mm.WaterGrid)
        and not getattr(n, "ignored", False)
    ]
    t_pus = [float(mvalue(n.model.t_pu)) for n in water_juncs]
    if t_pus:
        # Envelope min is 0.75 (≈ 267 K, see ``mes.py``); allow small slack
        # for piecewise selector rounding.
        assert min(t_pus) >= 0.75 - 1e-3, (
            f"junction t_pu fell below envelope: min={min(t_pus):.4f}"
        )


@pytest.mark.pptest
def test_generate_mes_gas_extra_mesh_pipes_reduce_bridges():

    net = simbench.get_simbench_net("1-LV-rural3--1-no_sw")
    mn = from_pandapower_net(net)

    def gas_graph(mes):
        g = nx.Graph()
        for n in mes.nodes:
            if isinstance(n.model, mm.Junction) and isinstance(n.grid, mm.GasGrid):
                g.add_node(n.id)
        for b in mes.branches:
            if isinstance(b.model, mm.GasPipe):
                g.add_edge(b.from_node_id, b.to_node_id)
        return g

    common = dict(
        coupling_density=0.5,
        centralized=False,
        couplings=("chp", "p2g", "p2h"),
        coupling_kwargs={"seed": 1, "use_hg_variants": True},
        heat_kwargs={"node_based_heat_loads": True},
    )

    mes_tree = generate_supply_return_mes_based_on_power_net(mn, **common)
    mes_mesh = generate_supply_return_mes_based_on_power_net(
        mn, gas_kwargs={"extra_mesh_pipes": 20, "mesh_seed": 42}, **common
    )

    g_tree, g_mesh = gas_graph(mes_tree), gas_graph(mes_mesh)
    # Tree baseline: 0 cycles → every edge is a bridge.
    cycles_tree = (
        g_tree.number_of_edges()
        - g_tree.number_of_nodes()
        + nx.number_connected_components(g_tree)
    )
    assert cycles_tree == 0
    assert len(list(nx.bridges(g_tree))) == g_tree.number_of_edges()

    # Mesh: exactly 20 extra edges → 20 cycles, dramatically fewer bridges.
    cycles_mesh = (
        g_mesh.number_of_edges()
        - g_mesh.number_of_nodes()
        + nx.number_connected_components(g_mesh)
    )
    assert cycles_mesh == 20
    assert g_mesh.number_of_edges() == g_tree.number_of_edges() + 20
    bridges_mesh = list(nx.bridges(g_mesh))

    assert len(bridges_mesh) < 0.6 * g_tree.number_of_edges()


@pytest.mark.pptest
def test_generate_mes_cp_size_multiplier_scales_uniformly():
    net = simbench.get_simbench_net("1-LV-rural3--1-no_sw")
    mn = from_pandapower_net(net)
    GAS_HHV_MJ_PER_KG = 15.3 * 3.6

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

    def totals(mes):
        chp_p = sum(
            float(c.model.mass_flow_setpoint)
            * GAS_HHV_MJ_PER_KG
            * float(c.model.efficiency_power)
            for c in mes.compounds
            if "CHP" in type(c.model).__name__
        )
        p2g_kgs = sum(
            abs(float(b.model.gas_kgps))
            for b in mes.branches
            if "PowerToGas" in type(b.model).__name__
        )
        p2h_q = sum(
            float(b.model.heat_energy_mw)
            for b in mes.branches
            if type(b.model).__name__ == "PowerToHeatHG"
        )
        return chp_p, p2g_kgs, p2h_q

    chp_d, p2g_d, p2h_d = totals(build())
    chp_1, p2g_1, p2h_1 = totals(build(cp_size_multiplier=1.0))
    chp_2, p2g_2, p2h_2 = totals(build(cp_size_multiplier=2.0))
    chp_4, p2g_4, p2h_4 = totals(build(cp_size_multiplier=4.0))

    # Default and explicit multiplier=1.0 must be identical (no behavior shift).
    assert (chp_d, p2g_d, p2h_d) == (chp_1, p2g_1, p2h_1)
    assert chp_d > 0 and p2g_d > 0 and p2h_d > 0

    # Linear scaling across all three carriers.  Per-CP ``round(..., 6)``
    # quantisation of mass_flow / heat_energy hits hardest on small P2G
    # mass flows (≈ 1·10⁻⁶ kg/s per unit at LV scale), where one ULP is a
    # ~5 % swing on the unit and ~1 % on the aggregate of ~20 units.  A
    # 2 % relative tolerance is the realistic floor for an aggregate check
    # at this rounding precision.
    REL = 2e-2
    assert chp_2 == pytest.approx(2 * chp_1, rel=REL)
    assert p2g_2 == pytest.approx(2 * p2g_1, rel=REL)
    assert p2h_2 == pytest.approx(2 * p2h_1, rel=REL)
    assert chp_4 == pytest.approx(4 * chp_1, rel=REL)
    assert p2g_4 == pytest.approx(4 * p2g_1, rel=REL)
    assert p2h_4 == pytest.approx(4 * p2h_1, rel=REL)

    # Per-type override: ``chp_p_share=1.0`` (vs default 0.5) doubles CHP
    # rated output but leaves P2G and P2H untouched.
    chp_s, p2g_s, p2h_s = totals(build(chp_p_share=1.0))
    assert chp_s == pytest.approx(2 * chp_1, rel=REL)
    assert p2g_s == pytest.approx(p2g_1, rel=1e-9)
    assert p2h_s == pytest.approx(p2h_1, rel=1e-9)


@pytest.mark.pptest
def test_generate_mes_replace_primary_generation_invariant():
    net = simbench.get_simbench_net("1-LV-rural3--1-no_sw")
    mn = from_pandapower_net(net)

    common = dict(
        coupling_density=0.5,
        centralized=False,
        couplings=("chp", "p2g", "p2h"),
        heat_kwargs={"node_based_heat_loads": True},
    )
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

    GAS_HHV_MJ_PER_KG = 15.3 * 3.6

    def primary_power_mw(mes):
        return sum(
            abs(c.model.p_mw)
            for c in mes.childs
            if isinstance(c.model, mm.PowerGenerator)
        )

    def primary_gas_kgs(mes):
        return sum(
            abs(c.model.mass_flow)
            for c in mes.childs
            if isinstance(c.model, mm.Source) and isinstance(c.grid, mm.GasGrid)
        )

    def cp_power_out_mw(mes):
        total = 0.0
        for compound in mes.compounds:
            m = compound.model
            cls = type(m).__name__
            if "CHP" in cls:
                total += (
                    float(m.mass_flow_setpoint)
                    * GAS_HHV_MJ_PER_KG
                    * float(m.efficiency_power)
                )
        return total

    def cp_gas_out_kgs(mes):
        # PowerToGas stores its rated injection in ``gas_kgps`` (negated under
        # the load convention), so the rated kg/s is ``abs(gas_kgps)``.
        return sum(
            abs(float(b.model.gas_kgps))
            for b in mes.branches
            if "PowerToGas" in type(b.model).__name__
        )

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

    pri_p_add = primary_power_mw(mes_add)
    pri_p_rep = primary_power_mw(mes_rep)
    cp_p = cp_power_out_mw(mes_add)
    assert cp_p > 0
    # Replacement mode strictly reduces the primary power pool ...
    assert pri_p_rep < pri_p_add - 1e-9
    # ... by exactly the rated CP power output (rated-invariant check).
    assert pri_p_rep == pytest.approx(pri_p_add - cp_p, abs=1e-9)

    pri_g_add = primary_gas_kgs(mes_add)
    pri_g_rep = primary_gas_kgs(mes_rep)
    cp_g = cp_gas_out_kgs(mes_add)
    assert cp_g > 0
    assert pri_g_rep < pri_g_add - 1e-12
    assert pri_g_rep == pytest.approx(pri_g_add - cp_g, abs=1e-9)

    # Heat side: in node-based mode with no ``node_heat_gen_share`` there is
    # no primary heat fleet to drain, so the heat replacement is a no-op and
    # the heat supply slack stays unbounded.  Bounding ``max_import_kgs`` is
    # *not* attempted: in node-based DHS the slack's mass flow is set by the
    # consumer sinks (hydraulically determined), so a ``max_import_kgs`` cap
    # is not a smooth heat-power scarcity dial.  See the dedicated
    # ``node_heat_gen_share`` test for the principled heat-side replacement.
    def heat_slack(mes):
        slacks = [
            c
            for c in mes.childs
            if isinstance(c.model, mm.ExtHydrGrid)
            and isinstance(c.grid, mm.WaterGrid)
            and float(getattr(c.model, "t_k", 0.0)) >= 350
        ]
        assert len(slacks) == 1
        return slacks[0]

    assert heat_slack(mes_add).model.mass_flow.min is None
    assert heat_slack(mes_rep).model.mass_flow.min is None  # also unbounded now


@pytest.mark.pptest
def test_generate_mes_node_heat_gen_share_and_replacement():
    net = simbench.get_simbench_net("1-LV-rural3--1-no_sw")
    mn = from_pandapower_net(net)
    GAS_HHV_MJ_PER_KG = 15.3 * 3.6

    common = dict(
        coupling_density=0.5,
        centralized=False,
        couplings=("chp", "p2g", "p2h"),
    )

    # 1) Explicit share=0.0 disables the distributed heat fleet (legacy).
    mes_no_hg = generate_supply_return_mes_based_on_power_net(
        mn,
        coupling_kwargs={"seed": 1, "use_hg_variants": True},
        heat_kwargs={"node_based_heat_loads": True, "node_heat_gen_share": 0.0},
        **common,
    )
    assert not [c for c in mes_no_hg.childs if isinstance(c.model, mm.HeatGenerator)], (
        "node_heat_gen_share=0.0 must not create HeatGenerator children"
    )

    # 2) Default (share=1.0): distributed HeatGenerators at PowerGenerator buses.
    mes_add = generate_supply_return_mes_based_on_power_net(
        mn,
        coupling_kwargs={"seed": 1, "use_hg_variants": True},
        heat_kwargs={"node_based_heat_loads": True},  # default share=1.0
        **common,
    )
    hg_add = [c for c in mes_add.childs if isinstance(c.model, mm.HeatGenerator)]
    assert hg_add, "node_heat_gen_share > 0 must distribute HeatGenerators"
    total_hg_add = sum(abs(c.model.q_mw_heat) for c in hg_add)
    assert total_hg_add > 0

    # 3) Replacement mode drains the HeatGenerator pool by the CP heat output.
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
    hg_rep = [c for c in mes_rep.childs if isinstance(c.model, mm.HeatGenerator)]
    total_hg_rep = sum(abs(c.model.q_mw_heat) for c in hg_rep)

    cp_chp_heat = sum(
        float(c.model.mass_flow_setpoint)
        * GAS_HHV_MJ_PER_KG
        * float(c.model.efficiency_heat)
        for c in mes_rep.compounds
        if "CHP" in type(c.model).__name__
    )
    cp_p2h_heat = sum(
        float(b.model.heat_energy_mw)
        for b in mes_rep.branches
        if type(b.model).__name__ == "PowerToHeatHG"
    )
    cp_heat = cp_chp_heat + cp_p2h_heat
    assert cp_heat > 0

    # Total rated heat (HeatGenerator + CP) must be invariant across modes.
    assert total_hg_rep < total_hg_add - 1e-9
    assert total_hg_rep == pytest.approx(total_hg_add - cp_heat, abs=1e-9)


@pytest.mark.pptest
def test_generate_mes_supply_slack_t_k_parameter():
    net = simbench.get_simbench_net("1-LV-rural3--1-no_sw")
    mn = from_pandapower_net(net)

    def slack_t_k(mes):
        slacks = [
            c
            for c in mes.childs
            if isinstance(c.model, mm.ExtHydrGrid) and isinstance(c.grid, mm.WaterGrid)
        ]
        assert len(slacks) == 1
        return slacks[0].model.t_k

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
    assert slack_t_k(mes_default) == 356  # legacy default
    assert slack_t_k(mes_lowered) == 330.0


@pytest.mark.pptest
def test_generate_mes_storage_capabilities_timeseries():
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
    mes.apply_formulation(MISOCP_NETWORK_FORMULATION)
    mes.apply_formulation(make_mccormick_dhs_formulation(num_partitions=1))
    mes.apply_formulation(make_nl_weymouth_pwl_network_formulation())

    # The two storage-capability extensions under study.
    mes.add_extension(GasLinepack())
    mes.add_extension(LumpedThermalCapacitance())

    steps = 3
    # Synthetic 3-step demand profile to drive the storage response: a low /
    # peak / recovery shape.  Power-load variation is already covered by the
    # simbench profile registered above; here we modulate only heat loads
    # and gas sinks so the LTC and linepack extensions see a varying signal.
    factors = [0.7, 1.3, 0.9]
    td = TimeseriesData()
    # Slice the simbench profiles down to ``steps`` so all series share the
    # same length (the raw simbench profile spans an entire year at 15 min
    # resolution and would otherwise mismatch the manual 3-step series).
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
            base = float(mvalue(c.model.mass_flow))
            if base == 0:
                continue
            td.add_child_series(c.id, "mass_flow", [base * f for f in factors])

    problem = create_min_load_shedding_problem(
        bounds_el=(0.9, 1.5),
        bounds_gas=(0.9, 1.5),
        bounds_heat=(0.7, 1.3),
        ext_grid_el_bounds=(-5, 5),
        ext_grid_gas_bounds=(-5, 5),
        ext_grid_heat_bounds=(-100, 100),
        include_ext_grids=True,
        check_vm=True,
        check_pressure=True,
        check_temperature=True,
        check_line_loading=True,
    )

    ts_result = run_timeseries(
        mes,
        timeseries_data=td,
        steps=steps,
        optimization_problem=problem,
        solver="gurobi",
    )

    # All steps converged.
    assert not ts_result.failed_steps, (
        f"timeseries had failed steps: {ts_result.failed_steps}"
    )
    assert len(ts_result.step_results) == steps

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
        # Drop the anchor step (t=0): there ``net_pack_kgs`` has no previous
        # state to differ from and is allowed to be 0.
        if len(npk) > 1 and float(npk.iloc[1:].abs().max()) > 1e-9:
            saw_packing = True
    assert saw_packing, (
        "no gas pipe exhibited inter-temporal packing flow; the linepack "
        "extension's temporal coupling does not appear to have been active"
    )

    # LTC: junction temperatures stay inside the McCormick envelope across
    # every step (the inertia term must not push them through the bounds)
    # and the inter-step jump in t_pu must be finite - i.e. LTC actually
    # produced a temporally-coupled solve.
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

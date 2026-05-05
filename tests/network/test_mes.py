import pytest
import simbench

import monee
import monee.model as mm
from monee import PyomoSolver, run_energy_flow
from monee.io.from_pandapower import from_pandapower_net
from monee.model.core import value as mvalue
from monee.model.formulation import (
    MISOCP_NETWORK_FORMULATION,
    make_mccormick_dhs_formulation,
    make_nl_weymouth_pwl_network_formulation,
)
from monee.network import generate_supply_return_mes_based_on_power_net
from monee.problem.min_load_shedding import create_min_load_shedding_problem


@pytest.mark.pptest
def test_generate_mes():
    # GIVEN
    net = simbench.get_simbench_net("1-LV-rural3--1-no_sw")

    # WHEN
    mn = from_pandapower_net(net)

    mes = generate_supply_return_mes_based_on_power_net(
        mn,
        coupling_density=0.5,
        centralized=False,
        couplings=("chp", "p2g", "p2h"),
        coupling_kwargs={"seed": 1, "use_hg_variants": False},
        heat_kwargs={"node_based_heat_loads": False},
    )
    mes.apply_formulation(MISOCP_NETWORK_FORMULATION)
    mes.apply_formulation(make_mccormick_dhs_formulation(num_partitions=16))

    res = run_energy_flow(mes, solver=PyomoSolver(), solver_name="gurobi")
    print(res)
    print(res.get(mm.Junction).to_string())
    print(res.get(mm.ExtHydrGrid).to_string())
    print(res.get(mm.HeatLoad).to_string())

    assert False


@pytest.mark.pptest
def test_generate_mes_min_load_shedding():
    """Canonical MES min-load-shedding configuration.

    Uses MISOCP for electricity and McCormick-DHS for heat with the
    HeatGenerator-style coupling-point variants — the only combination
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
    mes.apply_formulation(make_nl_weymouth_pwl_network_formulation())

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
    )

    # WHEN
    result = monee.run_energy_flow_optimization(
        mes,
        optimization_problem=problem,
        solver=PyomoSolver(),
        solver_name="gurobi",
    )
    # print(result)
    # result = run_energy_flow(mes, solver=PyomoSolver(), solver_name="gurobi")

    # THEN
    assert result is not None
    assert result.success, "min load shedding did not converge"
    assert result.objective is not None
    assert False
    # Aggregate heat delivery should be ≥ 95 % of design — the
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

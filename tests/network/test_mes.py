import pytest
import simbench

import monee
import monee.model as mm
from monee import PyomoSolver, run_energy_flow, run_timeseries
from monee.io.from_pandapower import from_pandapower_net
from monee.io.from_simbench import obtain_simbench_profile_by_pp_net
from monee.model import GasLinepack, LumpedThermalCapacitance
from monee.model.core import value as mvalue
from monee.model.formulation import (
    MISOCP_NETWORK_FORMULATION,
    make_mccormick_dhs_formulation,
    make_nl_weymouth_pwl_network_formulation,
)
from monee.network import generate_supply_return_mes_based_on_power_net
from monee.problem.min_load_shedding import create_min_load_shedding_problem
from monee.simulation.timeseries import TimeseriesData


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


@pytest.mark.pptest
def test_generate_mes_storage_capabilities_timeseries():
    """Storage-capability study on the canonical MES timeseries.

    Builds the same MES as ``test_generate_mes_min_load_shedding`` but adds
    the two storage-flavoured network extensions monee ships with:

    * :class:`~monee.model.extension.linepack.GasLinepack` — distributed
      gas storage in pipes (linepack), so the gas grid can buffer short-term
      supply/demand mismatches between timesteps.
    * :class:`~monee.model.extension.ltc.LumpedThermalCapacitance` —
      lumped thermal mass at every water-network junction, giving the heat
      grid genuine inter-step thermal inertia rather than instantaneous
      temperature jumps.

    A short timeseries is then run with the simbench load profiles applied
    to the underlying power loads (via
    :func:`~monee.io.from_simbench.obtain_simbench_profile_by_pp_net`,
    which now binds by aggregated bus name) plus a synthetic demand
    modulation on the heat loads and gas sinks generated by
    ``generate_supply_return_mes_based_on_power_net`` — those carriers
    have no simbench profile of their own, so the storage extensions need
    a manual swing to react against.

    The test demonstrates that this kind of multi-energy storage study is
    expressible in monee end-to-end; it asserts the timeseries solves and
    that the storage variables are populated and active rather than
    benchmarking a specific scenario.
    """
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

    # Min-load-shedding objective with the same loose envelope as the
    # canonical test, plus a wide ext-grid window so the optimiser is not
    # forced to shed demand purely because of step-to-step variation —
    # what we want to observe is whether storage absorbs the swing.
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
        solver=PyomoSolver(),
        solver_name="gurobi",
    )

    # All steps converged.
    assert not ts_result.failed_steps, (
        f"timeseries had failed steps: {ts_result.failed_steps}"
    )
    assert len(ts_result.step_results) == steps

    # GasLinepack: the per-pipe linepack mass must be populated for every
    # gas pipe across every step (the variable is injected by the extension),
    # and on at least one pipe the inter-temporal packing flow ``net_pack_kgs``
    # must be non-zero on a step > 0 — that is the direct signature of the
    # temporal-coupling equation being honoured (in a plain steady-state
    # solve ``net_pack_kgs`` is pinned to 0).  We do not assert a specific
    # swing magnitude: with the wide ext-grid window above the optimiser
    # can largely cover the demand swing from the source side, so the
    # observed pipe response is small but non-trivial.
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
    # and the inter-step jump in t_pu must be finite — i.e. LTC actually
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

"""Heat-exchanger duty reporting (F-020/F-027) and part-load flow (F-018)."""

import pytest

import monee.express as mx
import monee.model as mm
import monee.problem as mp
from monee import run_energy_flow, run_energy_flow_optimization
from monee.model.multi import SubHE

Q_SET_MW = 0.2


def _dhs_net(q_mw_per_hx=(0.15, Q_SET_MW)):
    net = mm.Network()
    dhs = mx.dhs_structure(net, diameter_m=0.15, length_m=200)
    seg = dhs.line(len(q_mw_per_hx), heat_exchanger_q_mw=list(q_mw_per_hx))
    dhs.attach_heat_plant(seg.supply.first, seg.return_.last, t_k=358.0)
    return net, seg.heat_exchangers[-1]


def _hx_price_problem():
    """Objective that pays for the exchanger's temperature drop, i.e. one that
    outbids the duty pull closing ``q_mw_delivered <= q_mw``."""
    problem = mp.OptimizationProblem()
    objectives = mp.Objectives()
    objectives.select(lambda m: isinstance(m, mm.HeatExchangerLoad)).data(
        lambda m: 60.0 * (m.t_in_pu - m.t_out_pu)
    ).calculate(lambda d: sum(d.values()))
    problem.objectives = objectives
    return problem


def test_under_delivering_heat_exchanger_is_reported():
    net, hx_id = _dhs_net((0.0, Q_SET_MW))
    run_energy_flow(net)

    result = run_energy_flow_optimization(net, _hx_price_problem())

    assert result.success
    key = f"HeatExchangerLoad.{hx_id}.q_mw_delivered_shortfall"
    assert key in result.violations
    assert result.violations[key] == pytest.approx(Q_SET_MW, abs=1e-3)


def _chp_net():
    net = mx.create_multi_energy_network()
    gas_junction = mx.create_gas_junction(net)
    mx.create_gas_ext_grid(net, gas_junction)
    bus = mx.create_bus(net)
    mx.create_ext_power_grid(net, bus)
    mx.create_power_load(net, bus, p_mw=1.0, q_mvar=0.0)
    supply = mx.create_water_junction(net)
    return_junction = mx.create_water_junction(net)
    mx.create_ext_hydr_grid(net, supply)
    mx.create_sink(net, return_junction, mass_flow_kgs=15)
    mx.create_chp(
        net,
        bus,
        supply,
        return_junction,
        gas_junction,
        diameter_m=0.15,
        efficiency_power=0.35,
        efficiency_heat=0.45,
        mass_flow_setpoint_kgs=0.1,
    )
    return net


def _sub_he_price_problem():
    problem = mp.OptimizationProblem()
    objectives = mp.Objectives()
    objectives.select(lambda m: isinstance(m, SubHE)).data(
        lambda m: -100.0 * m.q_mw_delivered
    ).calculate(lambda d: sum(d.values()))
    problem.objectives = objectives
    return problem


def test_under_delivering_compound_sub_he_is_reported():
    net = _chp_net()
    duty_mw = run_energy_flow(_chp_net()).dataframes["SubHE"]["q_mw"].iloc[0]

    result = run_energy_flow_optimization(net, _sub_he_price_problem())

    assert result.success
    keys = [k for k in result.violations if k.endswith("q_mw_delivered_shortfall")]
    assert len(keys) == 1
    assert keys[0].startswith("SubHE.")
    assert result.violations[keys[0]] == pytest.approx(abs(duty_mw), abs=1e-3)


def test_served_compound_sub_he_reports_no_shortfall():
    result = run_energy_flow(_chp_net())

    assert result.success
    assert not [k for k in result.violations if "shortfall" in k]


def test_served_heat_exchanger_reports_no_shortfall():
    net, _ = _dhs_net()

    result = run_energy_flow(net)

    assert result.success
    assert not [k for k in result.violations if "shortfall" in k]


def test_user_objective_is_reported_apart_from_the_formulation_terms():
    net, _ = _dhs_net((0.0, Q_SET_MW))
    run_energy_flow(net)

    result = run_energy_flow_optimization(net, _hx_price_problem())

    row = result.dataframes["HeatExchangerLoad"].iloc[0]
    expected_user = 60.0 * (row["t_in_pu"] - row["t_out_pu"])
    assert result.user_objective == pytest.approx(expected_user, rel=1e-6)
    assert result.aux_objective == pytest.approx(
        result.objective - result.user_objective, rel=1e-6
    )
    assert result.user_objective != pytest.approx(result.objective, rel=1e-3)


def test_plain_energy_flow_has_no_user_objective():
    net, _ = _dhs_net()

    result = run_energy_flow(net)

    assert result.user_objective is None


@pytest.mark.parametrize("regulation", [1.0, 0.5, 0.0])
def test_part_load_scales_mass_flow_not_temperature_spread(regulation):
    net, hx_id = _dhs_net()
    net.branch_by_id(hx_id).model.regulation = regulation

    result = run_energy_flow(net)

    assert result.success
    row = result.dataframes["HeatExchangerLoad"].iloc[-1]
    design_kgs = row["mass_flow_design_kgs"]
    assert row["mass_flow_kgs"] == pytest.approx(-design_kgs * regulation, abs=1e-4)
    spread_pu = row["t_from_pu"] - row["t_to_pu"]
    if regulation > 0:
        assert spread_pu == pytest.approx(0.0843, abs=2e-3)
    else:
        assert spread_pu == pytest.approx(0.0, abs=1e-6)

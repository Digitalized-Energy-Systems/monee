"""CasADi/IPOPT on district heating with idle pipes and units switched off.

A pipe or unit at zero flow is the corner where the heat transport turns
singular: an idle pipe delivers water at ground temperature, and an idle
unit's temperatures are undetermined. These tests pin the two consequences
that used to surface on a district: solves failing with
``Error_In_Step_Computation`` as soon as a unit was off, and economic
dispatches stopping at a local optimum with pipes idle and the penalised
return reference covering the losses.
"""

import warnings

import pytest

import monee
import monee.express as mx
import monee.model as mm
import monee.problem as mp
from monee import TimeseriesData
from monee.simulation import run_multi_period
from monee.solver.pyo import _pyscipopt_available

pytest.importorskip("casadi")

from monee.solver import CasADiSolver  # noqa: E402


def _district(price_el=10.0, t_ref_k=356.0, bounds_t_k=(283.15, 364.0)):
    """Boiler at the plant, heat pump at the far end of a two-pipe line, two
    consumers, a plant bypass and the return reference as penalised slack."""
    net = mm.Network(
        el_model=mm.create_power_grid("power"),
        gas_model=mm.create_gas_grid("gas"),
        water_model=mm.create_water_grid("heat", t_ref_k=t_ref_k),
    )
    bus = mx.create_bus(net, base_kv=20)
    mx.create_ext_power_grid(net, bus)
    gas = mx.create_gas_junction(net)
    mx.create_gas_ext_grid(net, gas)
    plant, mid, far, ret = (mx.create_water_junction(net) for _ in range(4))
    mx.create_water_ext_grid(net, ret, t_k=333.15, pressure_pu=1.0)
    for u, v in ((plant, mid), (mid, far)):
        mx.create_water_pipe(net, u, v, diameter_m=0.1, length_m=800)
    mx.create_water_pipe(net, plant, ret, diameter_m=0.1, length_m=50)
    for j in (mid, far):
        mx.create_heat_exchanger(net, j, ret, q_mw=0.6)
    boiler = mx.create_g2h(
        net,
        gas_node_id=gas,
        cold_node_id=ret,
        hot_node_id=plant,
        heat_energy_mw=2.0,
        diameter_m=0.15,
        efficiency=0.9,
    )
    hp = mx.create_p2h(
        net,
        power_node_id=bus,
        cold_node_id=ret,
        hot_node_id=far,
        heat_energy_mw=2.0,
        diameter_m=0.15,
        efficiency=3.0,
    )
    dispatch = mp.create_economic_dispatch_problem(
        ext_grid_cost_default=price_el,
        gas_cost_default=14.0,
        heat_cost_default=1000.0,
        bounds_ext_heat_mw=(-0.5, 0.0),
        bounds_t_k=bounds_t_k,
    )
    return net, dispatch, boiler, hp


@pytest.mark.parametrize(
    ("formulation", "boiler_reg", "hp_reg"),
    [
        ("smooth_nlp", [0.6, 0.6, 0.6], [0.0, 0.0, 0.0]),
        ("smooth_nlp", [0.6, 0.0, 0.3], [0.0, 0.6, 0.3]),
        (None, [0.6, 0.0, 0.3], [0.0, 0.6, 0.3]),
    ],
)
def test_multi_period_converges_with_a_unit_switched_off(
    formulation, boiler_reg, hp_reg
):
    net, _, boiler, hp = _district()
    series = TimeseriesData()
    series.add_compound_series(boiler, "regulation", boiler_reg)
    series.add_compound_series(hp, "regulation", hp_reg)
    plain = mp.OptimizationProblem()
    plain.objectives = mp.Objectives()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_multi_period(
            net,
            series,
            steps=3,
            optimization_problem=plain,
            solver="ipopt",
            formulation=formulation,
        )

    assert result.success
    assert result.termination_condition == "Solve_Succeeded"


def _solve(net, dispatch, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return monee.run_energy_flow_optimization(
            net, dispatch, formulation="smooth_nlp", copy=True, **kwargs
        )


@pytest.mark.skipif(not _pyscipopt_available(), reason="pyscipopt not installed")
def test_dispatch_starts_from_a_simulation_and_reaches_the_global_optimum():
    net, dispatch, _, _ = _district(price_el=10.0)
    exact = monee.run_energy_flow_optimization(
        net, dispatch, solver="scip", formulation="nonconvex_miqcqp", copy=True
    )

    result = _solve(net, dispatch, solver="ipopt")

    assert result.solver_stats["start"] == "simulation"
    assert result.objective == pytest.approx(exact.objective, rel=1e-3)


def test_flat_start_is_kept_on_request_and_as_fallback(monkeypatch):
    net, dispatch, _, _ = _district()

    assert (
        _solve(net, dispatch, solver="ipopt", start="flat").solver_stats["start"]
        == "flat"
    )

    monkeypatch.setattr(CasADiSolver, "_simulation_values", lambda *_: None)
    fallback = _solve(net, dispatch, solver="ipopt")
    assert fallback.success
    assert fallback.solver_stats["start"] == "flat"


@pytest.mark.parametrize("unit", ["boiler", "hp"])
def test_dispatch_converges_with_a_unit_unavailable(unit):
    net, dispatch, boiler, hp = _district()
    net.compound_by_id(boiler if unit == "boiler" else hp).set(regulation=0.0)

    assert _solve(net, dispatch, solver="ipopt").success


@pytest.mark.skipif(not _pyscipopt_available(), reason="pyscipopt not installed")
def test_unit_switched_off_keeps_temperature_bounds_without_t_ref():
    net, dispatch, boiler, _ = _district(t_ref_k=300.0, bounds_t_k=(320.0, 380.0))
    net.compound_by_id(boiler).set(regulation=0.0)

    result = monee.run_energy_flow_optimization(
        net, dispatch, solver="scip", formulation="nonconvex_miqcqp", copy=True
    )

    assert result.success


@pytest.mark.parametrize("unit", ["p2h", "g2h"])
def test_water_through_an_idle_unit_leaves_unheated(unit):
    # GIVEN an idle unit (regulation 0) that alone feeds a fixed 5 kg/s sink
    net = mm.Network(
        el_model=mm.create_power_grid("power"),
        gas_model=mm.create_gas_grid("gas"),
        water_model=mm.create_water_grid("heat", t_ref_k=356.0),
    )
    bus = mx.create_bus(net, base_kv=20)
    mx.create_ext_power_grid(net, bus)
    gas = mx.create_gas_junction(net)
    mx.create_gas_ext_grid(net, gas)
    cold, hot = mx.create_water_junction(net), mx.create_water_junction(net)
    mx.create_water_ext_grid(net, cold, t_k=333.15, pressure_pu=1.0)
    net.child_to(mm.Sink(mass_flow_kgs=5.0), hot)
    common = dict(cold_node_id=cold, hot_node_id=hot, heat_energy_mw=0.5)
    if unit == "p2h":
        mx.create_p2h(
            net,
            power_node_id=bus,
            diameter_m=0.15,
            efficiency=3.0,
            regulation=0.0,
            **common,
        )
    else:
        mx.create_g2h(
            net,
            gas_node_id=gas,
            diameter_m=0.15,
            efficiency=0.9,
            regulation=0.0,
            **common,
        )

    # WHEN
    result = monee.run_energy_flow(net, solver="ipopt")

    # THEN the water passes through at its inlet temperature
    assert result.success
    t_hot = result.get(mm.Junction, index="id").loc[hot, "t_pu"] * 356.0
    assert t_hot == pytest.approx(333.15, abs=1e-3)


def test_unknown_start_is_rejected():
    with pytest.raises(ValueError, match="start"):
        CasADiSolver(start="warm")

"""Post-solve restoration of the epigraph-relaxed Darcy-Weisbach hydraulics.

Only the small epsilon term pulls ``m_sq >= m^2`` tight, so an optimisation can
return pressure drops that are too large. ``restore_exact_heat_pressures``
recomputes the exact pressures on tree islands; meshed islands are reported.
"""

import pytest

import monee.express as mx
import monee.model as mm
import monee.problem as mp
from monee import run_energy_flow_optimization
from monee.model.formulation.miqcqp.nonconvex.heat import (
    BilinearDarcyWeisbachBranchFormulation,
)
from monee.model.phys.core.hydraulics import calc_pipe_area, friction_at_high_re
from monee.solver.pyo import _pyscipopt_available

pytestmark = pytest.mark.skipif(not _pyscipopt_available(), reason="pyscipopt missing")


@pytest.fixture
def slack_epigraph(monkeypatch):
    """Reward the squared flows instead of penalising them, so the solver leaves
    every epigraph at its cap."""
    monkeypatch.setattr(
        BilinearDarcyWeisbachBranchFormulation,
        "minimize",
        lambda self, branch, *args, **kwargs: [
            -1e-3
            * (branch.mass_flow_pos_kgs_squared + branch.mass_flow_neg_kgs_squared)
        ],
    )


def _solve(ring):
    net = mm.Network()
    dhs = mx.dhs_structure(net, diameter_m=0.15, length_m=300, unidirectional=False)
    seg = (
        dhs.ring(4, heat_exchanger_q_mw=0.2)
        if ring
        else dhs.line(3, heat_exchanger_q_mw=[0, 0.3, 0.4])
    )
    dhs.attach_heat_plant(seg.supply.first, seg.return_.first, t_k=358.0)
    problem = mp.OptimizationProblem()
    problem.objectives = mp.Objectives()
    problem.objectives.select(
        lambda m, _c: isinstance(m, mm.HeatExchangerLoad)
    ).calculate(lambda ms: sum(50 * m.q_mw_delivered for m in ms))
    result = run_energy_flow_optimization(
        net, problem, solver="scip", formulation="nonconvex_miqcqp"
    )
    assert result.success
    return result


def test_tree_island_gets_the_exact_pressure_drops(slack_epigraph):
    result = _solve(ring=False)

    grid = next(
        b.grid for b in result.network.branches if isinstance(b.model, mm.WaterPipe)
    )
    pressure = result.get(mm.Junction, index="id")["pressure_pu"]
    for _, pipe in result.get(mm.WaterPipe).iterrows():
        u, v, _ = pipe["id"]
        d, m = pipe["diameter_m"], pipe["mass_flow_kgs"]
        resistance = (
            friction_at_high_re(d, pipe["roughness_m"])
            * pipe["length_m"]
            / d
            / (2 * grid.fluid_density_kg_per_m3 * calc_pipe_area(d) ** 2)
            / grid.pressure_ref_pa
        )
        assert pressure[u] - pressure[v] == pytest.approx(
            -resistance * m * abs(m), abs=1e-9
        )
    assert result.violations == {}
    assert not [w for w in result.warnings if w.category == "relaxation"]


def test_meshed_island_keeps_the_relaxed_state_and_warns(slack_epigraph):
    result = _solve(ring=True)

    entries = [w for w in result.warnings if w.category == "relaxation"]
    assert len(entries) == 1
    assert "heat_nlp" in entries[0].message

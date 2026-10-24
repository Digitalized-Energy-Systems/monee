"""Exactness of the branch-flow MISOCP on two-bus cases the AC model decides."""

import pytest

import monee
import monee.model as mm
from monee.model import (
    Bus,
    ExtPowerGrid,
    GenericPowerBranch,
    Network,
    PowerGrid,
    PowerLoad,
)
from monee.model.formulation import EL_MISOCP_FORMULATION
from monee.solver.pyo import _pyscipopt_available

pytestmark = pytest.mark.skipif(not _pyscipopt_available(), reason="pyscipopt missing")


def _two_bus(load, max_i_ka=None, **ext):
    grid = PowerGrid(name="power", sn_mva=1.0)
    net = Network(grid)
    slack = ExtPowerGrid(p_mw=0, q_mvar=0, vm_pu=1.0, va_degree=0, **ext)
    n0 = net.node(Bus(base_kv=1.0), child_ids=[net.child(slack)], grid=grid)
    n1 = net.node(Bus(base_kv=1.0), child_ids=[net.child(load)], grid=grid)
    line = GenericPowerBranch(
        tap=1, shift=0, br_r_pu=0.017, br_x_pu=0.092,
        g_fr_pu=0.0, b_fr_pu=0.0, g_to_pu=0.0, b_to_pu=0.0,
    )  # fmt: skip
    if max_i_ka is not None:
        line.max_i_ka = max_i_ka
    net.branch(line, n0, n1)
    net.apply_formulation(EL_MISOCP_FORMULATION)
    return net


def test_a_slack_cone_on_a_resistive_branch_is_reported():
    # GIVEN a must-run 0.9 MW injection and no export: AC is infeasible, the
    # relaxation burns the surplus in fictitious losses
    net = _two_bus(PowerLoad(p_mw=-0.9, q_mvar=0.0), max_export_mw=0.0)

    # WHEN
    result = monee.run_energy_flow(net, solver="scip")

    # THEN the slack cone is a relaxation finding
    entries = [w for w in result.warnings if w.category == "relaxation"]
    assert len(entries) == 1
    assert "not AC-feasible" in entries[0].message
    assert entries[0].value > 0.5


def test_a_fixed_line_above_three_times_its_rating_still_solves():
    # GIVEN no loading limit and a line carrying ~5.8x its rating: the
    # switching cap of 3x the rating must not apply to a fixed line
    net = _two_bus(PowerLoad(p_mw=0.9, q_mvar=0.3), max_i_ka=0.1)

    # WHEN
    result = monee.run_energy_flow(net, solver="scip")

    # THEN the MISOCP reproduces the AC state (V = 0.951578 pu at the load)
    assert result.success
    vm = result.get(mm.Bus)["vm_pu"].to_list()
    assert vm[1] == pytest.approx(0.951578, abs=1e-5)
    assert result.get(mm.GenericPowerBranch)["loading_from_pu"].iloc[0] == (
        pytest.approx(5.7559, abs=1e-3)
    )
    assert not [w for w in result.warnings if w.category == "relaxation"]

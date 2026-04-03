"""
Test that inactive branches (on_off=0) in the MISOCP formulation
carry exactly zero current and zero power flow.

Topology: ring B0 → B1 → B2 → B0 where the B2→B0 tie-line is a backup
branch with on_off=0.  All load must flow through B0→B1→B2; the
backup line should be completely quiescent.
"""

import math

import monee.model as mm
from monee.model import Network
from monee.model.branch import PowerLine
from monee.model.child import ExtPowerGrid, PowerLoad
from monee.model.formulation import MISOCP_NETWORK_FORMULATION
from monee.model.grid import PowerGrid
from monee.model.node import Bus
from monee.solver import PyomoSolver


def create_ring_with_backup() -> Network:
    """Three-bus ring; the B2→B0 branch is the inactive backup tie-line."""
    pn = Network(PowerGrid(name="power", sn_mva=1))

    b0 = pn.node(
        Bus(base_kv=1),
        child_ids=[pn.child(ExtPowerGrid(p_mw=0, q_mvar=0, vm_pu=1.0, va_degree=0))],
        grid=mm.EL,
    )
    b1 = pn.node(
        Bus(base_kv=1),
        child_ids=[pn.child(PowerLoad(p_mw=0.5, q_mvar=0))],
        grid=mm.EL,
    )
    b2 = pn.node(
        Bus(base_kv=1),
        child_ids=[pn.child(PowerLoad(p_mw=0.5, q_mvar=0))],
        grid=mm.EL,
    )

    line_kw = dict(r_ohm_per_m=0.00007, x_ohm_per_m=0.00007, parallel=1)
    pn.branch(PowerLine(length_m=500, **line_kw), b0, b1)
    pn.branch(PowerLine(length_m=500, **line_kw), b1, b2)
    pn.branch(PowerLine(length_m=500, backup=True, on_off=0, **line_kw), b2, b0)

    pn.apply_formulation(MISOCP_NETWORK_FORMULATION)
    return pn


def _backup_row(result):
    """Return the result Series for the backup (on_off=0) line."""
    lines = result.get(mm.PowerLine)
    # The backup line has on_off=0; all others have on_off=1
    return lines[lines["on_off"] == 0].iloc[0]


def test_inactive_branch_has_zero_current():
    result = PyomoSolver().solve(create_ring_with_backup())
    backup = _backup_row(result)
    assert math.isclose(backup["current_pu"], 0.0, abs_tol=1e-6), (
        f"Inactive branch current_pu should be 0, got {backup['current_pu']}"
    )


def test_inactive_branch_has_zero_power_flow():
    result = PyomoSolver().solve(create_ring_with_backup())
    backup = _backup_row(result)
    for col in ("p_from_mw", "p_to_mw", "q_from_mvar", "q_to_mvar"):
        assert math.isclose(backup[col], 0.0, abs_tol=1e-4), (
            f"Inactive branch {col} should be 0, got {backup[col]}"
        )


def test_active_lines_carry_full_load():
    """Both loads must be fully served; only active lines carry current."""
    result = PyomoSolver().solve(create_ring_with_backup())

    lines = result.get(mm.PowerLine)
    active = lines[lines["on_off"] == 1]

    # Both active lines must carry non-zero current
    assert (active["current_pu"] > 1e-4).all(), (
        f"Active lines should carry current:\n{active['current_pu']}"
    )

    # All load is served (regulation = 1 for both loads)
    loads = result.get(mm.PowerLoad)
    assert (loads["regulation"] > 0.99).all(), (
        f"All load should be served:\n{loads['regulation']}"
    )

"""The branch-flow formulations must model the pi-model charging admittance."""

import math

import monee.model as mm
from monee.model import Network
from monee.model.branch import GenericPowerBranch
from monee.model.child import ExtPowerGrid, PowerLoad
from monee.model.formulation import EL_MISOCP_FORMULATION
from monee.model.grid import PowerGrid
from monee.model.node import Bus
from monee.solver import PyomoSolver

SN_MVA = 1.0
B_SHUNT = 0.088


def create_two_bus_net(b_shunt: float) -> Network:
    pn = Network(PowerGrid(name="power", sn_mva=SN_MVA))
    b0 = pn.node(
        Bus(base_kv=1.0),
        child_ids=[pn.child(ExtPowerGrid(p_mw=0, q_mvar=0, vm_pu=1.0, va_degree=0))],
        grid=mm.EL,
    )
    b1 = pn.node(
        Bus(base_kv=1.0),
        child_ids=[pn.child(PowerLoad(p_mw=0.9, q_mvar=0.3))],
        grid=mm.EL,
    )
    pn.branch(
        GenericPowerBranch(
            tap=1,
            shift=0,
            br_r_pu=0.017,
            br_x_pu=0.092,
            g_fr_pu=0.0,
            b_fr_pu=b_shunt,
            g_to_pu=0.0,
            b_to_pu=b_shunt,
        ),
        b0,
        b1,
    )
    pn.apply_formulation(EL_MISOCP_FORMULATION)
    return pn


def _solve(b_shunt):
    result = PyomoSolver().solve(create_two_bus_net(b_shunt))
    assert result.success
    branch = result.get(mm.GenericPowerBranch).iloc[0]
    buses = result.get(mm.Bus).sort_values("id")
    return branch, buses["vm_pu_squared"].to_numpy()


def test_series_reactive_balance_accounts_for_charging():
    branch, (w_from, w_to) = _solve(B_SHUNT)

    q_from_pu = branch["q_from_mvar"] / SN_MVA
    q_to_pu = branch["q_to_mvar"] / SN_MVA
    q_ser_from = q_from_pu + B_SHUNT * w_from
    q_ser_to = q_to_pu + B_SHUNT * w_to

    assert math.isclose(
        q_ser_from, 0.092 * branch["current_pu_squared"] - q_ser_to, abs_tol=1e-6
    ), "series reactive loss must be x*ell once the pi-shunt is split off"

    # The terminal flows themselves must NOT satisfy the series relation: the
    # shunt injects reactive power at both ends.
    assert abs(q_from_pu - (0.092 * branch["current_pu_squared"] - q_to_pu)) > 1e-3


def test_series_columns_close_the_cone_gap():
    branch, (w_from, _) = _solve(B_SHUNT)

    p_ser = branch["p_series_from_mw"] / SN_MVA
    q_ser = branch["q_series_from_mvar"] / SN_MVA

    assert math.isclose(
        q_ser, branch["q_from_mvar"] / SN_MVA + B_SHUNT * w_from, abs_tol=1e-8
    )
    cone_gap = w_from * branch["current_pu_squared"] - (p_ser**2 + q_ser**2)
    assert abs(cone_gap) <= 1e-6, (
        "the reported series flows must satisfy the SOC cone exactly"
    )


def test_charging_changes_the_slack_reactive_injection():
    with_shunt, _ = _solve(B_SHUNT)
    without_shunt, _ = _solve(0.0)

    assert abs(with_shunt["q_from_mvar"] - without_shunt["q_from_mvar"]) > 0.01, (
        "zeroing the line charging must change the branch-flow solution"
    )

import math

import monee.model as mm
from monee.model import Network
from monee.model.branch import PowerLine
from monee.model.child import ExtPowerGrid, PowerGenerator, PowerLoad
from monee.model.formulation import MISOCP_NETWORK_FORMULATION
from monee.model.grid import PowerGrid
from monee.model.node import Bus
from monee.problem.min_load_shedding import create_min_load_shedding_problem
from monee.solver import GEKKOSolver, PyomoSolver


def _build_net():
    """Four-bus power network with one deactivated line disconnecting B3."""
    pn = Network(PowerGrid(name="power", sn_mva=1))

    # B0 — generator
    pn.node(
        Bus(base_kv=1),
        child_ids=[pn.child(PowerGenerator(p_mw=1, q_mvar=0))],
        grid=mm.EL,
    )
    # B1 — slack (ext_grid)
    pn.node(
        Bus(base_kv=1),
        child_ids=[pn.child(ExtPowerGrid(p_mw=0, q_mvar=0, vm_pu=1, va_degree=0))],
        grid=mm.EL,
    )
    # B2 — connected load, 1.5 MW
    pn.node(
        Bus(base_kv=1),
        child_ids=[pn.child(PowerLoad(p_mw=1.5, q_mvar=0))],
        grid=mm.EL,
    )
    # B3 — disconnected load, 2 MW (line to B2 is deactivated)
    pn.node(
        Bus(base_kv=1),
        child_ids=[pn.child(PowerLoad(p_mw=2.0, q_mvar=0))],
        grid=mm.EL,
    )

    pn.branch(
        PowerLine(length_m=100, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5, parallel=1),
        0,
        1,  # B0 ── B1
    )
    pn.branch(
        PowerLine(length_m=100, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5, parallel=1),
        1,
        2,  # B1 ── B2
    )
    # Deactivate B2 ── B3: B3 becomes an isolated bus.
    pn.branch_by_id(
        pn.branch(
            PowerLine(length_m=100, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5, parallel=1),
            2,
            3,  # B2 ── B3  (inactive)
        )
    ).active = False

    return pn


def test_plain_solve_disconnected_bus_is_nan():
    """Without load shedding, B3's bus is NaN (pre-filtered by the solver)."""
    pn = _build_net()
    result = GEKKOSolver().solve(pn)

    bus_df = result.dataframes["Bus"]
    assert math.isnan(bus_df["vm_pu"][3]), (
        f"Plain solve: B3 bus should be NaN (disconnected); got {bus_df['vm_pu'][3]}"
    )


def test_load_shedding_with_disconnected_bus():
    """
    Load shedding must not crash and must correctly shed the connected load.

    The disconnected load (B3) gets regulation = 0 from the optimizer
    (the only feasible value since no power can reach B3).
    The connected load (B2) is shed to absorb the generation shortfall.
    """
    pn = _build_net()
    pn.apply_formulation(MISOCP_NETWORK_FORMULATION)
    problem = create_min_load_shedding_problem(
        # Force ext_grid to contribute nothing → only 1 MW generator feeds B2.
        ext_grid_el_bounds=(0, 0),
        include_ext_grids=True,
        # Disable non-electric checks to keep the test focused.
        check_temperature=False,
        check_pressure=False,
    )

    result = PyomoSolver().solve(
        pn, exclude_unconnected_nodes=True, optimization_problem=problem
    )
    print(result.dataframes["PowerLoad"])
    load_df = result.dataframes["PowerLoad"]

    # Connected load (B2, index 0): must be partially shed.
    # With 1 MW generation and 1.5 MW demand → regulation ≈ 1/1.5 ≈ 0.667.
    connected_reg = load_df["regulation"][0]
    assert not math.isnan(connected_reg), "B2 load regulation must not be NaN"
    assert 0.0 < connected_reg < 1.0, (
        f"B2 load must be partially shed; got regulation={connected_reg}"
    )
    assert math.isclose(connected_reg, 1 / 1.5, rel_tol=0.05), (
        f"Expected B2 regulation ≈ 0.667, got {connected_reg}"
    )

    # Disconnected load (B3, index 1): optimizer must set regulation = 0
    # because no power path exists to B3.
    disconnected_reg = load_df["regulation"][1]
    assert math.isclose(disconnected_reg, 0.0, abs_tol=1e-4), (
        "B3 load must be fully shed (regulation=0) since it is disconnected; "
        f"got regulation={disconnected_reg}"
    )

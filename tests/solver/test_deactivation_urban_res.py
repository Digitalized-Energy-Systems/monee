"""Deactivate individual urban-district components; unreachable nodes must be NaN'd, the rest solved."""

import math

import monee.model as mm
import monee.problem as mp
from monee.model.formulation import MISOCP_NETWORK_FORMULATION
from monee.network import create_urban_district_net
from monee.solver import PyomoSolver
from tests.util import assert_junction_nan as _assert_jct_nan
from tests.util import assert_junction_solved as _assert_jct_solved


class _Ids:
    pass


def _build():
    net = create_urban_district_net()

    # Power lines identified by unique length_m
    p_lines = net.branches_by_type(mm.PowerLine)
    line_b0_b1 = next(l for l in p_lines if l.model.length_m == 500)
    line_b1_b2 = next(l for l in p_lines if l.model.length_m == 400)
    line_b2_b3 = next(l for l in p_lines if l.model.length_m == 300)
    line_b2_b4 = next(l for l in p_lines if l.model.length_m == 350)

    # Gas pipes
    g_pipes = net.branches_by_type(mm.GasPipe)
    pipe_g0_g1 = next(p for p in g_pipes if p.model.length_m == 400)
    pipe_g1_g2 = next(p for p in g_pipes if p.model.length_m == 300)
    pipe_g2_g3 = next(p for p in g_pipes if p.model.length_m == 250)
    pipe_g1_g4 = next(p for p in g_pipes if p.model.length_m == 200)

    # Heat branches: supply chain s1-s2-s3 (two 100 m pipes) plus the
    # HeatExchangerLoad consumer bridging supply→return (s3→r1).
    def _ends(branch):
        return {
            net.node_by_id(branch.from_node_id).name,
            net.node_by_id(branch.to_node_id).name,
        }

    w_pipes = net.branches_by_type(mm.WaterPipe)
    pipe_s1_s2 = next(p for p in w_pipes if _ends(p) == {"s1", "s2"})
    pipe_s2_s3 = next(p for p in w_pipes if _ends(p) == {"s2", "s3"})
    he_s3_r1 = net.branches_by_type(mm.HeatExchangerLoad)[0]

    # Cross-domain branches
    p2g = net.branches_by_type(mm.PowerToGas)[0]
    g2p = net.branches_by_type(mm.GasToPower)[0]

    # Compounds
    chp = net.compounds_by_type(mm.CHP)[0]

    # Node IDs derived from branches / compounds
    ids = _Ids()
    ids.b0 = line_b0_b1.from_node_id
    ids.b1 = line_b0_b1.to_node_id
    ids.b2 = line_b1_b2.to_node_id
    ids.b3 = line_b2_b3.to_node_id
    ids.b4 = line_b2_b4.to_node_id

    ids.g0 = pipe_g0_g1.from_node_id
    ids.g1 = pipe_g0_g1.to_node_id
    ids.g2 = pipe_g1_g2.to_node_id
    ids.g3 = pipe_g2_g3.to_node_id
    ids.g4 = pipe_g1_g4.to_node_id

    ids.s1 = pipe_s1_s2.from_node_id
    ids.s2 = pipe_s1_s2.to_node_id
    ids.s3 = pipe_s2_s3.to_node_id
    ids.r1 = he_s3_r1.to_node_id

    # Component references
    ids.line_b0_b1 = line_b0_b1
    ids.line_b1_b2 = line_b1_b2
    ids.line_b2_b3 = line_b2_b3
    ids.line_b2_b4 = line_b2_b4
    ids.pipe_g0_g1 = pipe_g0_g1
    ids.pipe_g1_g2 = pipe_g1_g2
    ids.pipe_g2_g3 = pipe_g2_g3
    ids.pipe_g1_g4 = pipe_g1_g4
    ids.pipe_s1_s2 = pipe_s1_s2
    ids.pipe_s2_s3 = pipe_s2_s3
    ids.he_s3_r1 = he_s3_r1
    ids.p2g = p2g
    ids.g2p = g2p
    ids.chp = chp

    return net, ids


def _bus_vm(result, nid):
    df = result.dataframes["Bus"]
    return df.loc[df["id"] == nid, "vm_pu_squared"].iloc[0]


def _assert_bus_nan(result, nid, label):
    v = _bus_vm(result, nid)
    assert math.isnan(v), f"{label}: expected NaN vm_pu, got {v}"


def _assert_bus_solved(result, nid, label):
    v = _bus_vm(result, nid)
    assert not math.isnan(v), f"{label}: expected solved vm_pu, got NaN"


def _assert_converge(result):
    assert result.success


def _solve(net):
    net.apply_formulation(MISOCP_NETWORK_FORMULATION)
    problem = mp.create_min_load_shedding_problem(
        # Force ext_grid to contribute nothing → only 1 MW generator feeds B2.
        ext_grid_el_bounds=(0, 0),
        include_ext_grids=True,
        # Disable non-electric checks to keep the test focused.
        check_temperature=False,
        check_pressure=False,
    )

    return PyomoSolver().solve(
        net, exclude_unconnected_nodes=True, optimization_problem=problem
    )


def test_deactivate_line_b0_b1():
    # GIVEN
    net, ids = _build()
    ids.line_b0_b1.active = False

    # WHEN
    result = _solve(net)
    print(result)

    # THEN
    _assert_converge(result)

    # B0 (generator bus, leaf) isolated; slack B1 stays solved
    _assert_bus_nan(result, ids.b0, "B0")
    _assert_bus_solved(result, ids.b1, "B1")


def test_deactivate_line_b1_b2():
    # GIVEN
    net, ids = _build()
    ids.line_b1_b2.active = False

    # WHEN
    result = _solve(net)
    print(result.full())

    # THEN
    _assert_converge(result)

    # B2/B3/B4 subtree isolated (no path to slack)
    _assert_bus_nan(result, ids.b2, "B2")
    _assert_bus_nan(result, ids.b3, "B3")
    _assert_bus_nan(result, ids.b4, "B4")

    _assert_bus_solved(result, ids.b1, "B1")


def test_deactivate_line_b2_b3():
    # GIVEN
    net, ids = _build()
    ids.line_b2_b3.active = False

    # WHEN
    result = _solve(net)
    print(result.full())

    # THEN
    _assert_converge(result)

    # B3 (CHP power bus, leaf) isolated
    _assert_bus_nan(result, ids.b3, "B3")
    _assert_bus_solved(result, ids.b2, "B2")


def test_deactivate_line_b2_b4():
    # GIVEN
    net, ids = _build()
    ids.line_b2_b4.active = False

    # WHEN
    result = _solve(net)

    # THEN
    _assert_converge(result)

    # B4 (4 MW load bus, leaf) isolated
    _assert_bus_nan(result, ids.b4, "B4")
    _assert_bus_solved(result, ids.b2, "B2")


def test_deactivate_gas_pipe_g0_g1():
    # GIVEN
    net, ids = _build()
    ids.pipe_g0_g1.active = False

    # WHEN
    result = _solve(net)

    # THEN
    _assert_converge(result)

    # whole gas subtree {G1–G4} isolated; G0 ext stays solved
    _assert_jct_nan(result, ids.g1, "G1")
    _assert_jct_nan(result, ids.g4, "G4")
    _assert_jct_solved(result, ids.g0, "G0")


def test_deactivate_gas_pipe_g1_g2():
    # GIVEN
    net, ids = _build()
    ids.pipe_g1_g2.active = False

    # WHEN
    result = _solve(net)

    # THEN
    _assert_converge(result)

    # G2 (CHP gas node) and G3 isolated
    _assert_jct_nan(result, ids.g2, "G2")
    _assert_jct_nan(result, ids.g3, "G3")

    _assert_jct_solved(result, ids.g1, "G1")
    _assert_jct_solved(result, ids.g4, "G4")


def test_deactivate_gas_pipe_g2_g3():
    # GIVEN
    net, ids = _build()
    ids.pipe_g2_g3.active = False

    # WHEN
    result = _solve(net)

    # THEN
    _assert_converge(result)

    # G3 (G2P gas node, leaf) isolated. G2 stays solved: it is the CHP's gas
    # attachment port and still feeds the CHP via the active G1-G2 pipe
    # (attachment ports of active compounds are exempt from leaf-stub pruning).
    _assert_jct_nan(result, ids.g3, "G3")
    _assert_jct_solved(result, ids.g2, "G2")
    _assert_jct_solved(result, ids.g1, "G1")


def test_deactivate_gas_pipe_g1_g4():
    # GIVEN
    net, ids = _build()
    ids.pipe_g1_g4.active = False

    # WHEN
    result = _solve(net)

    # THEN
    _assert_converge(result)

    # G4 (P2G gas node, leaf) isolated
    _assert_jct_nan(result, ids.g4, "G4")
    _assert_jct_solved(result, ids.g1, "G1")


def test_deactivate_water_pipe_h0_h1():
    # GIVEN
    net, ids = _build()
    ids.pipe_s1_s2.active = False

    # WHEN
    result = _solve(net)

    # THEN
    _assert_converge(result)

    # s2 and s3 become a childless dead-end chain and are leaf-stub pruned.
    # r1 stays solved: connectivity analysis replaces the CHP with a synthetic
    # return→supply pipe r1→s1 (remove_cps), and r1's ConsumeHydrGrid anchors
    # its mass balance. s1 ext stays solved; power and gas intact.
    _assert_jct_nan(result, ids.s2, "s2")
    _assert_jct_nan(result, ids.s3, "s3")
    _assert_jct_solved(result, ids.r1, "r1")
    _assert_jct_solved(result, ids.s1, "s1")
    _assert_bus_solved(result, ids.b1, "B1")
    _assert_jct_solved(result, ids.g0, "G0")


def test_deactivate_water_pipe_h1_h4():
    # GIVEN
    net, ids = _build()
    ids.pipe_s2_s3.active = False

    # WHEN
    result = _solve(net)

    # THEN
    _assert_converge(result)

    # s3 (HE supply node) is cut off; s2 then becomes a childless degree-1
    # stub and is pruned too. s1 (ext grid) and r1 (kept alive via the
    # synthetic CHP return→supply link plus its ConsumeHydrGrid) stay solved.
    _assert_jct_nan(result, ids.s3, "s3")
    _assert_jct_nan(result, ids.s2, "s2")
    _assert_jct_solved(result, ids.s1, "s1")
    _assert_jct_solved(result, ids.r1, "r1")


def test_deactivate_heat_exchanger_h2_h3():
    # GIVEN
    net, ids = _build()
    ids.he_s3_r1.active = False

    # WHEN
    result = _solve(net)

    # THEN
    _assert_converge(result)

    # Without the HE, the supply spur s3 (and then s2) is a childless
    # dead-end chain and gets leaf-stub pruned. r1 is NOT isolated: the
    # connectivity analysis keeps the CHP's return→supply link (remove_cps
    # inserts a synthetic pipe r1→s1) and r1's ConsumeHydrGrid anchors it.
    _assert_jct_nan(result, ids.s3, "s3")
    _assert_jct_nan(result, ids.s2, "s2")
    _assert_jct_solved(result, ids.r1, "r1")
    _assert_jct_solved(result, ids.s1, "s1")


def test_deactivate_p2g():
    # GIVEN
    net, ids = _build()
    ids.p2g.active = False

    # WHEN
    result = _solve(net)

    # THEN
    _assert_converge(result)

    # P2G is a coupling branch: no isolation, both grids stay solved
    _assert_bus_solved(result, ids.b1, "B1")
    _assert_jct_solved(result, ids.g0, "G0")


def test_deactivate_g2p():
    # GIVEN
    net, ids = _build()
    ids.g2p.active = False

    # WHEN
    result = _solve(net)

    # THEN
    _assert_converge(result)

    # G2P is a coupling branch: no isolation, both grids stay solved
    _assert_bus_solved(result, ids.b1, "B1")
    _assert_jct_solved(result, ids.g0, "G0")


def test_deactivate_chp_compound():
    # GIVEN
    net, ids = _build()
    net.deactivate(ids.chp)

    # WHEN
    result = _solve(net)

    # THEN
    _assert_converge(result)

    # heat and gas grids still solved without the CHP
    _assert_jct_solved(result, ids.s2, "s2")
    _assert_jct_solved(result, ids.g1, "G1")

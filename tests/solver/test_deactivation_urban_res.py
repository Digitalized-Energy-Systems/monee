"""Tests: deactivating each individual component in the urban residential district grid.

One component (branch or compound) is deactivated per test.  The solver must:
  * not crash,
  * NaN-out nodes that are no longer reachable from an ExtGrid, and
  * leave the rest of the network solved.

Topology (create_urban_district_net)
-------------------------------------
Power  (20 kV):  B0(gen) – B1(slack) – B2 – B3(CHP bus), B2 – B4(P2H bus)
Gas:             G0(ext) – G1 – G2(CHP) – G3(G2P sink), G1 – G4(P2G sink)
Heat:            H0(ext) – H1 – H4  [H2–H3 via HeatExchangerLoad; H4–H5 via P2H]
CPs:             CHP(G2→B3, H1/H2), P2H(B4, H4/H5), P2G(B0→G4), G2P(G3→B2)

Note on P2G / G2P
------------------
P2G and G2P are MultiGridBranchModel branches and are stripped from the
topology graph by ``remove_cps`` before connectivity is evaluated.
Deactivating them never isolates any node — only the coupling equations are
removed.  Both domain grids remain fully connected.
"""

import math

import monee.model as mm
import monee.problem as mp
from monee.model.formulation import MISOCP_NETWORK_FORMULATION
from monee.network import create_urban_district_net
from monee.solver import PyomoSolver


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

    # Heat branches
    w_pipes = net.branches_by_type(mm.WaterPipe)
    pipe_h0_h1 = next(p for p in w_pipes if p.model.length_m == 150)
    pipe_h1_h4 = next(p for p in w_pipes if p.model.length_m == 100)
    he_h2_h3 = net.branches_by_type(mm.HeatExchangerLoad)[0]

    # Cross-domain branches
    p2g = net.branches_by_type(mm.PowerToGas)[0]
    g2p = net.branches_by_type(mm.GasToPower)[0]

    # Compounds
    chp = net.compounds_by_type(mm.CHP)[0]
    p2h = net.compounds_by_type(mm.PowerToHeat)[0]

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

    ids.h0 = pipe_h0_h1.from_node_id
    ids.h1 = pipe_h0_h1.to_node_id
    ids.h2 = he_h2_h3.from_node_id
    ids.h3 = he_h2_h3.to_node_id
    ids.h4 = pipe_h1_h4.to_node_id
    ids.h5 = p2h.connected_to["heat_return_node_id"]

    # Component references
    ids.line_b0_b1 = line_b0_b1
    ids.line_b1_b2 = line_b1_b2
    ids.line_b2_b3 = line_b2_b3
    ids.line_b2_b4 = line_b2_b4
    ids.pipe_g0_g1 = pipe_g0_g1
    ids.pipe_g1_g2 = pipe_g1_g2
    ids.pipe_g2_g3 = pipe_g2_g3
    ids.pipe_g1_g4 = pipe_g1_g4
    ids.pipe_h0_h1 = pipe_h0_h1
    ids.pipe_h1_h4 = pipe_h1_h4
    ids.he_h2_h3 = he_h2_h3
    ids.p2g = p2g
    ids.g2p = g2p
    ids.chp = chp
    ids.p2h = p2h

    return net, ids


def _bus_vm(result, nid):
    df = result.dataframes["Bus"]
    return df.loc[df["id"] == nid, "vm_pu_squared"].iloc[0]


def _jct_t(result, nid):
    df = result.dataframes["Junction"]
    return df.loc[df["id"] == nid, "t_pu"].iloc[0]


def _ctrl_t(result, name):
    return result.dataframes[name]["t_pu"].iloc[0]


def _assert_bus_nan(result, nid, label):
    v = _bus_vm(result, nid)
    assert math.isnan(v), f"{label}: expected NaN vm_pu, got {v}"


def _assert_bus_solved(result, nid, label):
    v = _bus_vm(result, nid)
    assert not math.isnan(v), f"{label}: expected solved vm_pu, got NaN"


def _assert_jct_nan(result, nid, label):
    v = _jct_t(result, nid)
    assert math.isnan(v), f"{label}: expected NaN t_pu, got {v}"


def _assert_jct_solved(result, nid, label):
    v = _jct_t(result, nid)
    assert not math.isnan(v), f"{label}: expected solved t_pu, got NaN"


def _assert_ctrl_nan(result, name, label):
    v = _ctrl_t(result, name)
    assert math.isnan(v), f"{label}: expected NaN {name}.t_pu, got {v}"


def _assert_ctrl_solved(result, name, label):
    v = _ctrl_t(result, name)
    assert not math.isnan(v), f"{label}: expected solved {name}.t_pu, got NaN"


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
    """Line B0–B1 off → B0 (generator bus, leaf) isolated; slack B1 remains solved."""
    net, ids = _build()
    ids.line_b0_b1.active = False
    result = _solve(net)
    print(result)
    _assert_converge(result)
    _assert_bus_nan(result, ids.b0, "B0")
    _assert_bus_solved(result, ids.b1, "B1")


def test_deactivate_line_b1_b2():
    """Line B1–B2 off → B2/B3/B4 subtree isolated (no path to slack); B1 solved."""
    net, ids = _build()
    ids.line_b1_b2.active = False
    result = _solve(net)
    print(result.full())
    _assert_converge(result)
    _assert_bus_nan(result, ids.b2, "B2")
    _assert_bus_nan(result, ids.b3, "B3")
    _assert_bus_nan(result, ids.b4, "B4")
    _assert_bus_solved(result, ids.b1, "B1")


def test_deactivate_line_b2_b3():
    """Line B2–B3 off → B3 (CHP power bus, leaf) isolated; CHP NaN'd, B2 solved."""
    net, ids = _build()
    ids.line_b2_b3.active = False
    result = _solve(net)
    print(result.full())
    _assert_converge(result)
    _assert_bus_nan(result, ids.b3, "B3")
    _assert_bus_solved(result, ids.b2, "B2")


def test_deactivate_line_b2_b4():
    """Line B2–B4 off → B4 (P2H power bus, leaf) isolated; P2H NaN'd, B2 solved."""
    net, ids = _build()
    ids.line_b2_b4.active = False
    result = _solve(net)
    _assert_converge(result)
    _assert_bus_nan(result, ids.b4, "B4")
    _assert_bus_solved(result, ids.b2, "B2")


def test_deactivate_gas_pipe_g0_g1():
    """Pipe G0–G1 off → whole gas subtree {G1–G4} isolated; G0 ext solved."""
    net, ids = _build()
    ids.pipe_g0_g1.active = False
    result = _solve(net)
    _assert_converge(result)
    _assert_jct_nan(result, ids.g1, "G1")
    _assert_jct_nan(result, ids.g4, "G4")
    _assert_jct_solved(result, ids.g0, "G0")


def test_deactivate_gas_pipe_g1_g2():
    """Pipe G1–G2 off → G2 (CHP gas node) and G3 isolated; G1 and G4 remain solved."""
    net, ids = _build()
    ids.pipe_g1_g2.active = False
    result = _solve(net)
    _assert_converge(result)
    _assert_jct_nan(result, ids.g2, "G2")
    _assert_jct_nan(result, ids.g3, "G3")
    _assert_jct_solved(result, ids.g1, "G1")
    _assert_jct_solved(result, ids.g4, "G4")


def test_deactivate_gas_pipe_g2_g3():
    """Pipe G2–G3 off → G3 (G2P gas node, leaf) isolated; G2 remains solved."""
    net, ids = _build()
    ids.pipe_g2_g3.active = False
    result = _solve(net)
    _assert_converge(result)
    _assert_jct_nan(result, ids.g3, "G3")
    _assert_jct_solved(result, ids.g2, "G2")


def test_deactivate_gas_pipe_g1_g4():
    """Pipe G1–G4 off → G4 (P2G gas node, leaf) isolated; G1 remains solved."""
    net, ids = _build()
    ids.pipe_g1_g4.active = False
    result = _solve(net)
    _assert_converge(result)
    _assert_jct_nan(result, ids.g4, "G4")
    _assert_jct_solved(result, ids.g1, "G1")


def test_deactivate_water_pipe_h0_h1():
    """Pipe H0–H1 off → entire heat subtree {H1–H5} isolated; power and gas intact."""
    net, ids = _build()
    ids.pipe_h0_h1.active = False
    result = _solve(net)
    _assert_converge(result)
    _assert_jct_nan(result, ids.h1, "H1")
    _assert_bus_solved(result, ids.b1, "B1")
    _assert_jct_solved(result, ids.g0, "G0")


def test_deactivate_water_pipe_h1_h4():
    """Pipe H1–H4 off → H4 (P2H heat_node) and H5 isolated; P2H NaN'd, H1 solved."""
    net, ids = _build()
    ids.pipe_h1_h4.active = False
    result = _solve(net)
    _assert_converge(result)
    _assert_jct_nan(result, ids.h4, "H4")
    _assert_jct_solved(result, ids.h1, "H1")


def test_deactivate_heat_exchanger_h2_h3():
    """Heat exchanger H2–H3 off → H3 (return-side Sink, leaf) isolated."""
    net, ids = _build()
    ids.he_h2_h3.active = False
    result = _solve(net)
    _assert_converge(result)
    _assert_jct_nan(result, ids.h3, "H3")


def test_deactivate_p2g():
    """P2G branch off → no isolation; power and gas grids remain fully solved."""
    net, ids = _build()
    ids.p2g.active = False
    result = _solve(net)
    _assert_converge(result)
    _assert_bus_solved(result, ids.b1, "B1")
    _assert_jct_solved(result, ids.g0, "G0")


def test_deactivate_g2p():
    """G2P branch off → no isolation; power and gas grids remain fully solved."""
    net, ids = _build()
    ids.g2p.active = False
    result = _solve(net)
    _assert_converge(result)
    _assert_bus_solved(result, ids.b1, "B1")
    _assert_jct_solved(result, ids.g0, "G0")


def test_deactivate_chp_compound():
    """CHP compound off → CHPControlNode NaN'd; heat and gas grids still solved."""
    net, ids = _build()
    net.deactivate(ids.chp)
    result = _solve(net)
    _assert_converge(result)
    _assert_jct_solved(result, ids.h1, "H1")
    _assert_jct_solved(result, ids.g1, "G1")


def test_deactivate_p2h_compound():
    """P2H compound off → PowerToHeatControlNode NaN'd; heat side still solved."""
    net, ids = _build()
    net.deactivate(ids.p2h)
    result = _solve(net)
    _assert_converge(result)
    _assert_jct_solved(result, ids.h4, "H4")

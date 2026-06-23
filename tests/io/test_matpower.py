import math

import numpy as np
import pytest

from monee.io.matpower import (
    _mpc_from_m_text,
    read_matpower_case,
    read_matpower_data,
)
from monee.model.branch import GenericPowerBranch
from monee.model.child import ExtPowerGrid, PowerGenerator, PowerShunt
from monee.solver.gekko import GEKKOSolver


def _mpc(bus, gen, branch, base_mva=100.0):
    """Wrap raw matrices in the nested struct shape scipy.io.loadmat produces."""

    def cell(value):
        holder = np.empty((1, 1), dtype=object)
        holder[0, 0] = value
        return holder

    return {
        "mpc": {
            "baseMVA": cell(np.array([[base_mva]])),
            "bus": cell(np.array(bus, dtype=float)),
            "gen": cell(np.array(gen, dtype=float)),
            "branch": cell(np.array(branch, dtype=float)),
        }
    }


def test_bus_type_drives_slack_detection():
    # bus cols: id type Pd Qd Gs Bs area Vm Va baseKV zone Vmax Vmin
    bus = [
        [1, 3, 0, 0, 0, 0, 1, 1.0, 0, 110, 1, 1.1, 0.9],  # ref / slack
        [2, 2, 0, 0, 0, 0, 1, 1.0, 0, 110, 1, 1.1, 0.9],  # PV, gen dispatched at 0
        [3, 1, 50, 10, 0, 0, 1, 1.0, 0, 110, 1, 1.1, 0.9],  # PQ load
    ]
    # gen cols: bus Pg Qg Qmax Qmin Vg mBase status Pmax Pmin
    gen = [
        [1, 80, 5, 100, -100, 1.0, 100, 1, 200, 0],  # ref bus -> ExtPowerGrid
        [2, 0, 0, 100, -100, 1.0, 100, 1, 200, 0],  # PG==0 but PV -> PowerGenerator
    ]
    # branch cols: f t r x b rateA rateB rateC tap shift status angmin angmax
    branch = [[1, 3, 0.01, 0.05, 0, 0, 0, 0, 0, 0, 1, -30, 30]]

    net = read_matpower_data(_mpc(bus, gen, branch))

    ext_grids = [c for c in net.childs if isinstance(c.model, ExtPowerGrid)]
    assert len(ext_grids) == 1
    # The single slack lives at the reference bus (BUS_TYPE == 3).
    ref_children = net.childs_by_ids(net.node_by_id(1).child_ids)
    assert any(isinstance(c.model, ExtPowerGrid) for c in ref_children)
    # The PG==0 generator at the PV bus must NOT be misread as a slack.
    pv_node = net.node_by_id(2)
    pv_children = net.childs_by_ids(pv_node.child_ids)
    assert any(isinstance(c.model, PowerGenerator) for c in pv_children)
    assert not any(isinstance(c.model, ExtPowerGrid) for c in pv_children)


def test_out_of_service_gen_and_branch():
    bus = [
        [1, 3, 0, 0, 0, 0, 1, 1.0, 0, 110, 1, 1.1, 0.9],
        [2, 1, 30, 5, 0, 0, 1, 1.0, 0, 110, 1, 1.1, 0.9],
    ]
    gen = [
        [1, 50, 5, 100, -100, 1.0, 100, 1, 200, 0],  # in service
        [2, 40, 0, 100, -100, 1.0, 100, 0, 200, 0],  # GEN_STATUS == 0
    ]
    branch = [
        [1, 2, 0.01, 0.05, 0, 0, 0, 0, 0, 0, 1, -30, 30],  # in service
        [1, 2, 0.02, 0.06, 0, 0, 0, 0, 0, 0, 0, -30, 30],  # BR_STATUS == 0
    ]

    net = read_matpower_data(_mpc(bus, gen, branch))

    off_gens = [
        c
        for c in net.childs
        if isinstance(c.model, PowerGenerator) and c.model.regulation == 0
    ]
    assert len(off_gens) == 1

    on_off_flags = sorted(b.model.on_off for b in net.branches)
    assert on_off_flags == [0, 1]


CASE_M = """\
function mpc = case_test
%% MATPOWER Case Format : Version 2
mpc.version = '2';
mpc.baseMVA = 100.0;

%% bus data
%	bus_i	type	Pd	Qd	Gs	Bs	area	Vm	Va	baseKV	zone	Vmax	Vmin
mpc.bus = [
	1	3	0	0	0	0	1	1.0	0	110	1	1.1	0.9;
	2	2	0	0	0	0	1	1.0	0	110	1	1.1	0.9;
	3	1	50	10	0	0	1	1.0	0	110	1	1.1	0.9;
];

%% generator data
mpc.gen = [
	1	80	5	100	-100	1.0	100	1	200	0;
	2	0	0	100	-100	1.0	100	1	200	0;
];

%% branch data
mpc.branch = [
	1	3	0.01	0.05	0	0	0	0	0	0	1	-30	30;	% in service
	3	2	0.02	0.06	0	0	0	0	0	0	0	-30	30;	% out of service
];
"""


def test_parse_m_text_matrices():
    mpc = _mpc_from_m_text(CASE_M)

    assert mpc["baseMVA"] == 100.0
    assert len(mpc["bus"]) == 3
    assert len(mpc["gen"]) == 2
    assert len(mpc["branch"]) == 2
    # comments stripped, 13 numeric columns recovered for the first bus row
    assert mpc["bus"][0] == [1, 3, 0, 0, 0, 0, 1, 1.0, 0, 110, 1, 1.1, 0.9]
    # trailing '% in service' comment did not leak into the row
    assert len(mpc["branch"][0]) == 13


def test_parse_m_handles_inf_and_nan():
    mpc = _mpc_from_m_text(
        "mpc.baseMVA = 100;\n"
        "mpc.bus = [1 3 0 0 0 0 1 1 0 110 1 Inf -Inf];\n"
        "mpc.gen = [1 0 0 0 0 1 100 1 NaN 0];\n"
        "mpc.branch = [1 1 0 0 0 0 0 0 0 0 1];\n"
    )
    assert mpc["bus"][0][11] == math.inf
    assert mpc["bus"][0][12] == -math.inf
    assert math.isnan(mpc["gen"][0][8])


def test_read_m_case_builds_network(tmp_path):
    case_file = tmp_path / "case_test.m"
    case_file.write_text(CASE_M, encoding="utf-8")

    net = read_matpower_case(str(case_file))

    # Same structural outcomes as the .mat path: a single slack at the ref bus,
    # the PG==0 PV generator stays a generator, and BR_STATUS drives on_off.
    assert len([c for c in net.childs if isinstance(c.model, ExtPowerGrid)]) == 1
    ref_children = net.childs_by_ids(net.node_by_id(1).child_ids)
    assert any(isinstance(c.model, ExtPowerGrid) for c in ref_children)
    assert sorted(b.model.on_off for b in net.branches) == [0, 1]
    assert all(isinstance(b.model, GenericPowerBranch) for b in net.branches)


def test_m_and_mat_paths_agree():
    bus = [
        [1, 3, 0, 0, 0, 0, 1, 1.0, 0, 110, 1, 1.1, 0.9],
        [2, 2, 0, 0, 0, 0, 1, 1.0, 0, 110, 1, 1.1, 0.9],
        [3, 1, 50, 10, 0, 0, 1, 1.0, 0, 110, 1, 1.1, 0.9],
    ]
    gen = [
        [1, 80, 5, 100, -100, 1.0, 100, 1, 200, 0],
        [2, 0, 0, 100, -100, 1.0, 100, 1, 200, 0],
    ]
    branch = [
        [1, 3, 0.01, 0.05, 0, 0, 0, 0, 0, 0, 1, -30, 30],
        [3, 2, 0.02, 0.06, 0, 0, 0, 0, 0, 0, 0, -30, 30],
    ]

    from_mat = read_matpower_data(_mpc(bus, gen, branch))
    from_m = _mpc_from_m_text(CASE_M)

    assert [list(map(float, r)) for r in from_m["bus"]] == bus
    assert len(from_mat.childs) == len(gen) + 1  # +1 load at bus 3


def test_no_reference_bus_raises():
    # All PV buses, no BUS_TYPE==3: the network would have no slack/angle
    # reference, so import must fail loudly rather than build an unsolvable net.
    bus = [
        [1, 2, 10, 2, 0, 0, 1, 1.0, 0, 110, 1, 1.1, 0.9],
        [2, 2, 10, 2, 0, 0, 1, 1.0, 0, 110, 1, 1.1, 0.9],
    ]
    gen = [[1, 50, 0, 100, -100, 1.0, 100, 1, 200, 0]]
    branch = [[1, 2, 0.01, 0.05, 0, 0, 0, 0, 0, 0, 1, -30, 30]]

    with pytest.raises(ValueError, match="no usable slack"):
        read_matpower_data(_mpc(bus, gen, branch))


def test_ref_bus_with_only_out_of_service_gen_raises():
    # The reference bus's single generator is out of service -> no slack.
    bus = [
        [1, 3, 0, 0, 0, 0, 1, 1.0, 0, 110, 1, 1.1, 0.9],
        [2, 1, 10, 2, 0, 0, 1, 1.0, 0, 110, 1, 1.1, 0.9],
    ]
    gen = [[1, 50, 0, 100, -100, 1.0, 100, 0, 200, 0]]  # GEN_STATUS == 0
    branch = [[1, 2, 0.01, 0.05, 0, 0, 0, 0, 0, 0, 1, -30, 30]]

    with pytest.raises(ValueError, match="no usable slack"):
        read_matpower_data(_mpc(bus, gen, branch))


def test_multiple_reference_buses_yield_single_slack():
    # A malformed case marks two buses as type 3; only the first becomes the
    # slack, the second's generator falls back to a fixed PowerGenerator.
    bus = [
        [1, 3, 0, 0, 0, 0, 1, 1.0, 0, 110, 1, 1.1, 0.9],
        [2, 3, 10, 2, 0, 0, 1, 1.0, 0, 110, 1, 1.1, 0.9],
    ]
    gen = [
        [1, 50, 0, 100, -100, 1.0, 100, 1, 200, 0],
        [2, 40, 0, 100, -100, 1.0, 100, 1, 200, 0],
    ]
    branch = [[1, 2, 0.01, 0.05, 0, 0, 0, 0, 0, 0, 1, -30, 30]]

    net = read_matpower_data(_mpc(bus, gen, branch))

    ext_grids = [c for c in net.childs if isinstance(c.model, ExtPowerGrid)]
    assert len(ext_grids) == 1
    ref_children = net.childs_by_ids(net.node_by_id(1).child_ids)
    assert any(isinstance(c.model, ExtPowerGrid) for c in ref_children)
    bus2_children = net.childs_by_ids(net.node_by_id(2).child_ids)
    assert all(not isinstance(c.model, ExtPowerGrid) for c in bus2_children)


def test_empty_bus_matrix_raises():
    with pytest.raises(ValueError, match="empty 'mpc.bus'"):
        _mpc_from_m_text(
            "mpc.baseMVA = 100;\nmpc.bus = [];\nmpc.gen = [];\nmpc.branch = [];\n"
        )


def test_bus_shunt_imported():
    # bus 2 carries GS=3 MW, BS=8 MVAr; bus 3 has neither.
    bus = [
        [1, 3, 0, 0, 0, 0, 1, 1.0, 0, 110, 1, 1.1, 0.9],
        [2, 1, 20, 5, 3, 8, 1, 1.0, 0, 110, 1, 1.1, 0.9],
        [3, 1, 10, 2, 0, 0, 1, 1.0, 0, 110, 1, 1.1, 0.9],
    ]
    gen = [[1, 60, 5, 100, -100, 1.0, 100, 1, 200, 0]]
    branch = [
        [1, 2, 0.01, 0.05, 0, 0, 0, 0, 0, 0, 1, -30, 30],
        [2, 3, 0.01, 0.05, 0, 0, 0, 0, 0, 0, 1, -30, 30],
    ]

    net = read_matpower_data(_mpc(bus, gen, branch))

    shunts = [c for c in net.childs if isinstance(c.model, PowerShunt)]
    assert len(shunts) == 1
    assert shunts[0].model.gs_mw == 3
    assert shunts[0].model.bs_mvar == 8
    # the shunt is attached to bus 2, alongside that bus's load
    bus2_children = net.childs_by_ids(net.node_by_id(2).child_ids)
    assert any(isinstance(c.model, PowerShunt) for c in bus2_children)


def test_isolated_bus_and_attachments_dropped():
    # bus 3 is isolated (type 4); its generator and the branch 2-3 must vanish.
    bus = [
        [1, 3, 0, 0, 0, 0, 1, 1.0, 0, 110, 1, 1.1, 0.9],
        [2, 1, 20, 5, 0, 0, 1, 1.0, 0, 110, 1, 1.1, 0.9],
        [3, 4, 10, 2, 1, 1, 1, 1.0, 0, 110, 1, 1.1, 0.9],
    ]
    gen = [
        [1, 60, 5, 100, -100, 1.0, 100, 1, 200, 0],
        [3, 40, 0, 100, -100, 1.0, 100, 1, 200, 0],  # gen at the isolated bus
    ]
    branch = [
        [1, 2, 0.01, 0.05, 0, 0, 0, 0, 0, 0, 1, -30, 30],
        [2, 3, 0.01, 0.05, 0, 0, 0, 0, 0, 0, 1, -30, 30],  # touches isolated bus
    ]

    net = read_matpower_data(_mpc(bus, gen, branch))

    assert {n.id for n in net.nodes} == {1, 2}
    assert len(net.branches) == 1
    # no child (gen or shunt) survives at the dropped bus 3
    assert all(c.node_id in (1, 2) for c in net.childs)


def test_shunt_solves_with_reactive_injection():
    # A capacitor bank (BS > 0, GS = 0) at the PQ bus injects reactive power:
    # the solved q_mvar = -bs_mvar * vm_pu**2 < 0 in load convention, which lifts
    # the bus voltage above the slack's 1.0 p.u.
    bs_mvar = 5.0
    bus = [
        [1, 3, 0, 0, 0, 0, 1, 1.0, 0, 110, 1, 1.1, 0.9],
        [2, 1, 5, 2, 0, bs_mvar, 1, 1.0, 0, 110, 1, 1.1, 0.9],
    ]
    gen = [[1, 0, 0, 100, -100, 1.0, 100, 1, 200, 0]]
    branch = [[1, 2, 0.01, 0.05, 0, 0, 0, 0, 0, 0, 1, -30, 30]]

    net = read_matpower_data(_mpc(bus, gen, branch))
    result = GEKKOSolver().solve(net)
    assert result.success

    shunt = next(c for c in net.childs if isinstance(c.model, PowerShunt))
    vm_pu = net.node_by_id(2).model.vm_pu.value
    assert shunt.model.p_mw.value == pytest.approx(0.0, abs=1e-6)
    assert shunt.model.q_mvar.value == pytest.approx(-bs_mvar * vm_pu**2, rel=1e-3)
    assert vm_pu > 1.0  # capacitor raises the voltage


def test_import_simbench_net():
    # GIVEN
    network = read_matpower_case("tests/data/1-LV-rural3--1-no_sw.mat")

    # WHEN
    solver = GEKKOSolver()
    result = solver.solve(network)

    # THEN
    assert result.success

    assert network is not None
    assert len(result.dataframes["Bus"]) == 129

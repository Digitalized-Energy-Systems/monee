import math

import numpy as np
import pytest

from monee import run_energy_flow_optimization
from monee.io.matpower import (
    _mpc_from_m_text,
    build_matpower_opf,
    read_matpower_case,
    read_matpower_data,
)
from monee.model.branch import GenericPowerBranch
from monee.model.child import (
    ExtPowerGrid,
    PowerGenerator,
    PowerShunt,
    VoltageControlledGenerator,
)
from monee.solver.casadi import CasADiSolver
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
    # The PG==0 generator at the PV bus must NOT be misread as a slack; it is a
    # voltage-controlled (PV) generator.
    pv_node = net.node_by_id(2)
    pv_children = net.childs_by_ids(pv_node.child_ids)
    assert any(isinstance(c.model, VoltageControlledGenerator) for c in pv_children)
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


def test_multiple_reference_buses_each_get_a_slack():
    # Two reference buses (e.g. multiple transmission infeeds) yield one
    # ExtPowerGrid each; a second generator at the same ref bus stays fixed PQ.
    bus = [
        [1, 3, 0, 0, 0, 0, 1, 1.0, 0, 110, 1, 1.1, 0.9],
        [2, 3, 10, 2, 0, 0, 1, 1.0, 0, 110, 1, 1.1, 0.9],
    ]
    gen = [
        [1, 50, 0, 100, -100, 1.0, 100, 1, 200, 0],  # bus 1 -> slack
        [2, 40, 0, 100, -100, 1.0, 100, 1, 200, 0],  # bus 2 -> slack
        [1, 30, 0, 100, -100, 1.0, 100, 1, 200, 0],  # bus 1 second gen -> fixed PQ
    ]
    branch = [[1, 2, 0.01, 0.05, 0, 0, 0, 0, 0, 0, 1, -30, 30]]

    net = read_matpower_data(_mpc(bus, gen, branch))

    ext_grids = [c for c in net.childs if isinstance(c.model, ExtPowerGrid)]
    assert len(ext_grids) == 2  # one slack per reference bus
    bus1 = net.childs_by_ids(net.node_by_id(1).child_ids)
    assert sum(isinstance(c.model, ExtPowerGrid) for c in bus1) == 1
    assert sum(isinstance(c.model, PowerGenerator) for c in bus1) == 1
    bus2 = net.childs_by_ids(net.node_by_id(2).child_ids)
    assert any(isinstance(c.model, ExtPowerGrid) for c in bus2)


def test_empty_bus_matrix_raises():
    with pytest.raises(ValueError, match="empty 'mpc.bus'"):
        _mpc_from_m_text(
            "mpc.baseMVA = 100;\nmpc.bus = [];\nmpc.gen = [];\nmpc.branch = [];\n"
        )


def test_pv_bus_generator_imported():
    # bus 2: PV with two generators (first holds V, second is fixed PQ).
    # bus 3: PV whose first generator is out of service, second is in service.
    bus = [
        [1, 3, 0, 0, 0, 0, 1, 1.0, 0, 110, 1, 1.1, 0.9],
        [2, 2, 5, 1, 0, 0, 1, 1.0, 0, 110, 1, 1.1, 0.9],
        [3, 2, 5, 1, 0, 0, 1, 1.0, 0, 110, 1, 1.1, 0.9],
    ]
    gen = [
        [1, 0, 0, 100, -100, 1.0, 100, 1, 200, 0],  # slack
        [2, 20, 0, 100, -100, 1.04, 100, 1, 200, 0],  # bus2 -> VCG
        [2, 10, 0, 100, -100, 1.04, 100, 1, 200, 0],  # bus2 second -> fixed PQ
        [3, 15, 0, 100, -100, 1.03, 100, 0, 200, 0],  # bus3 first OFF
        [3, 15, 0, 100, -100, 1.03, 100, 1, 200, 0],  # bus3 second ON -> VCG
    ]
    branch = [
        [1, 2, 0.01, 0.05, 0, 0, 0, 0, 0, 0, 1, -30, 30],
        [2, 3, 0.01, 0.05, 0, 0, 0, 0, 0, 0, 1, -30, 30],
    ]

    net = read_matpower_data(_mpc(bus, gen, branch))

    bus2 = net.childs_by_ids(net.node_by_id(2).child_ids)
    assert sum(isinstance(c.model, VoltageControlledGenerator) for c in bus2) == 1
    assert sum(isinstance(c.model, PowerGenerator) for c in bus2) == 1
    vcg2 = next(c for c in bus2 if isinstance(c.model, VoltageControlledGenerator))
    assert vcg2.model.vm_pu == 1.04  # generator VG setpoint

    bus3 = net.childs_by_ids(net.node_by_id(3).child_ids)
    assert sum(isinstance(c.model, VoltageControlledGenerator) for c in bus3) == 1


def test_pv_bus_holds_voltage():
    # The PV bus holds its generator's voltage setpoint (1.06), distinct from the
    # slack's (1.03) - proving |V| is controlled, not left floating as fixed PQ.
    bus = [
        [1, 3, 0, 0, 0, 0, 1, 1.0, 0, 110, 1, 1.1, 0.9],
        [2, 2, 1, 0.3, 0, 0, 1, 1.0, 0, 110, 1, 1.1, 0.9],
    ]
    gen = [
        [1, 0, 0, 100, -100, 1.03, 100, 1, 200, 0],
        [2, 2, 0, 100, -100, 1.06, 100, 1, 200, 0],
    ]
    branch = [[1, 2, 0.01, 0.05, 0, 0, 0, 0, 0, 0, 1, -30, 30]]

    net = read_matpower_data(_mpc(bus, gen, branch))
    result = GEKKOSolver().solve(net)
    assert result.success

    vm_by_id = dict(
        zip(result.dataframes["Bus"]["id"], result.dataframes["Bus"]["vm_pu"])
    )
    assert vm_by_id[1] == pytest.approx(1.03, rel=1e-4)  # slack setpoint
    assert vm_by_id[2] == pytest.approx(1.06, rel=1e-4)  # PV setpoint held


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


def test_branch_rating_imported_as_current_limit():
    # RATE_A (MVA) becomes a current limit at the from-bus voltage,
    # max_i_ka = RATE_A / (sqrt(3) * base_kv) - matching pandapower. RATE_A == 0
    # stays unbounded.
    base_kv = 110.0
    rate_a = 200.0
    bus = [
        [1, 3, 0, 0, 0, 0, 1, 1.0, 0, base_kv, 1, 1.1, 0.9],
        [2, 1, 5, 1, 0, 0, 1, 1.0, 0, base_kv, 1, 1.1, 0.9],
        [3, 1, 5, 1, 0, 0, 1, 1.0, 0, base_kv, 1, 1.1, 0.9],
    ]
    gen = [[1, 0, 0, 100, -100, 1.0, 100, 1, 200, 0]]
    branch = [
        [1, 2, 0.01, 0.05, 0, rate_a, 0, 0, 0, 0, 1, -30, 30],  # rated
        [1, 3, 0.01, 0.05, 0, 0, 0, 0, 0, 0, 1, -30, 30],  # RATE_A=0 -> unbounded
    ]

    net = read_matpower_data(_mpc(bus, gen, branch))
    max_i_ka_by_to = {b.id[1]: b.model.max_i_ka for b in net.branches}
    assert max_i_ka_by_to[2] == pytest.approx(rate_a / (math.sqrt(3) * base_kv))
    assert max_i_ka_by_to[3] == 999  # unlimited sentinel


def test_branch_rating_kept_verbatim_as_apparent_power_limit():
    # RATE_A is also kept as-is in max_s_mva, the apparent-power (MVA) rating used
    # by the default MVA line-limit basis (|S| <= RATE_A, exactly as in MATPOWER).
    # RATE_A == 0 -> None, which falls back to the current (max_i_ka) basis.
    rate_a = 200.0
    bus = [
        [1, 3, 0, 0, 0, 0, 1, 1.0, 0, 110, 1, 1.1, 0.9],
        [2, 1, 5, 1, 0, 0, 1, 1.0, 0, 110, 1, 1.1, 0.9],
        [3, 1, 5, 1, 0, 0, 1, 1.0, 0, 110, 1, 1.1, 0.9],
    ]
    gen = [[1, 0, 0, 100, -100, 1.0, 100, 1, 200, 0]]
    branch = [
        [1, 2, 0.01, 0.05, 0, rate_a, 0, 0, 0, 0, 1, -30, 30],  # rated
        [1, 3, 0.01, 0.05, 0, 0, 0, 0, 0, 0, 1, -30, 30],  # RATE_A=0 -> None
    ]

    net = read_matpower_data(_mpc(bus, gen, branch))
    max_s_by_to = {b.id[1]: b.model.max_s_mva for b in net.branches}
    assert max_s_by_to[2] == pytest.approx(rate_a)
    assert max_s_by_to[3] is None


def test_ac_nlp_scales_branch_flow_vars_to_grid_base():
    # The AC NLP formulation declares scale = sn_mva on the MW-magnitude branch
    # P/Q flow vars so the backend hands IPOPT O(1) unknowns (per-unit
    # conditioning). The per-unit i_*/loading_* intermediates stay unscaled.
    from monee.model.formulation.nlp.el import AcPolarNlpBranchFormulation

    bus = [
        [1, 3, 0, 0, 0, 0, 1, 1.0, 0, 110, 1, 1.1, 0.9],
        [2, 1, 5, 1, 0, 0, 1, 1.0, 0, 110, 1, 1.1, 0.9],
    ]
    gen = [[1, 0, 0, 100, -100, 1.0, 100, 1, 200, 0]]
    branch = [[1, 2, 0.01, 0.05, 0, 0, 0, 0, 0, 0, 1, -30, 30]]
    net = read_matpower_data(_mpc(bus, gen, branch, base_mva=100.0))
    comp = net.branches[0]
    assert comp.model.p_from_mw.scale == 1.0  # not yet declared

    AcPolarNlpBranchFormulation().ensure_var(comp.model, grid=comp.grid)
    for key in ("p_from_mw", "q_from_mvar", "p_to_mw", "q_to_mvar"):
        assert getattr(comp.model, key).scale == 100.0

    # A multi-grid coupling branch carries grid as a dict, not a PowerGrid; the
    # guard must not raise and must leave the flow vars at scale 1.
    fresh = read_matpower_data(_mpc(bus, gen, branch, base_mva=100.0)).branches[0]
    AcPolarNlpBranchFormulation().ensure_var(fresh.model, grid={"a": object()})
    assert fresh.model.p_from_mw.scale == 1.0


def test_ext_grid_slack_injection_scaled_to_grid_base():
    # The slack's free P/Q injection vars are MW-magnitude; ExtPowerGrid.overwrite
    # declares scale = sn_mva on them, the same per-unit basis as the branch flows.
    import types

    from monee.model.grid import PowerGrid

    ext = ExtPowerGrid(p_mw=10.0, q_mvar=2.0)
    assert ext.p_mw.scale == 1.0  # not yet declared
    ext.overwrite(types.SimpleNamespace(), PowerGrid(name="p", sn_mva=100.0))
    assert ext.p_mw.scale == 100.0
    assert ext.q_mvar.scale == 100.0

    # Non-PowerGrid context (multi-grid node) must not raise and leaves scale at 1.
    ext2 = ExtPowerGrid(p_mw=10.0, q_mvar=2.0)
    ext2.overwrite(types.SimpleNamespace(), {"a": object()})
    assert ext2.p_mw.scale == 1.0


def test_opf_scales_dispatchable_generator_injection_vars():
    # build_matpower_opf declares scale = baseMVA on the dispatchable generators'
    # MW-magnitude P/Q injection vars, the same per-unit basis as the branch
    # flows and the slack, so IPOPT sees O(1) unknowns throughout.
    from monee.model.core import Var

    bus = [
        [1, 3, 0, 0, 0, 0, 1, 1.0, 0, 110, 1, 1.1, 0.9],
        [2, 2, 50, 10, 0, 0, 1, 1.0, 0, 110, 1, 1.1, 0.9],
    ]
    gen = [
        [1, 0, 0, 100, -100, 1.0, 100, 1, 200, 0],
        [2, 0, 0, 100, -100, 1.0, 100, 1, 100, 5],
    ]
    gencost = [[2, 0, 0, 2, 10.0, 0.0], [2, 0, 0, 2, 50.0, 0.0]]
    branch = [[1, 2, 0.01, 0.05, 0, 0, 0, 0, 0, 0, 1, -30, 30]]
    mpc = {
        "baseMVA": 100.0,
        "bus": bus,
        "gen": gen,
        "branch": branch,
        "gencost": gencost,
    }

    net, _ = build_matpower_opf(mpc)
    gens = [c.model for c in net.childs if hasattr(c.model, "_cost_coeffs")]
    assert gens, "expected dispatchable generators carrying cost"
    for gm in gens:
        if isinstance(gm.p_mw, Var):
            assert gm.p_mw.scale == 100.0
        if isinstance(gm.q_mvar, Var):
            assert gm.q_mvar.scale == 100.0


def test_per_unit_base_invariance():
    # MATPOWER injections are in MW while branch parameters are per-unit on
    # baseMVA. The importer keeps both as-is and sets sn_mva=baseMVA; the AC flow
    # equations scale per-unit power to MW by sn_mva. So the same physical grid
    # expressed at baseMVA=1 and baseMVA=100 must solve to identical voltages.
    # Regression for the unit bug that made every baseMVA!=1 case infeasible.
    def build(base, r, x):
        bus = [
            [1, 3, 0, 0, 0, 0, 1, 1.0, 0, 110, 1, 1.1, 0.9],
            [2, 1, 2, 0.5, 0, 0, 1, 1.0, 0, 110, 1, 1.1, 0.9],
        ]
        gen = [[1, 0, 0, 100, -100, 1.0, 100, 1, 200, 0]]
        branch = [[1, 2, r, x, 0, 0, 0, 0, 0, 0, 1, -30, 30]]
        return read_matpower_data(_mpc(bus, gen, branch, base_mva=base))

    # pu impedance on baseMVA=100 is 100x the value on baseMVA=1 for one grid.
    r1 = CasADiSolver().solve(build(1.0, 0.002, 0.01))
    r100 = CasADiSolver().solve(build(100.0, 0.2, 1.0))
    assert r1.success and r100.success

    vm1 = dict(zip(r1.dataframes["Bus"]["id"], r1.dataframes["Bus"]["vm_pu"]))
    vm100 = dict(zip(r100.dataframes["Bus"]["id"], r100.dataframes["Bus"]["vm_pu"]))
    for bus_id in vm1:
        assert vm1[bus_id] == pytest.approx(vm100[bus_id], abs=1e-6)
    assert vm1[2] != pytest.approx(1.0, abs=1e-4)  # the load actually moved vm


def test_opf_builder_economic_dispatch():
    # build_matpower_opf turns gencost + PMIN/PMAX into a monee economic
    # dispatch. The slack is cheap (10 /MW), the bus-2 generator expensive
    # (50 /MW), so the optimum serves the load from the slack and parks the
    # expensive generator at its PMIN (5 MW).
    bus = [
        [1, 3, 0, 0, 0, 0, 1, 1.0, 0, 110, 1, 1.1, 0.9],
        [2, 2, 10, 2, 0, 0, 1, 1.0, 0, 110, 1, 1.1, 0.9],
    ]
    gen = [
        [1, 0, 0, 100, -100, 1.0, 100, 1, 200, 0],  # slack, cheap
        [2, 0, 0, 100, -100, 1.0, 100, 1, 50, 5],  # expensive, PMIN=5
    ]
    gencost = [
        [2, 0, 0, 2, 10.0, 0.0],  # 10 /MW (linear)
        [2, 0, 0, 2, 50.0, 0.0],  # 50 /MW
    ]
    branch = [[1, 2, 0.01, 0.05, 0, 0, 0, 0, 0, 0, 1, -30, 30]]
    mpc = {
        "baseMVA": 100.0,
        "bus": bus,
        "gen": gen,
        "branch": branch,
        "gencost": gencost,
    }

    net, prob = build_matpower_opf(mpc, max_loading=None)
    result = run_energy_flow_optimization(net, prob, solver=CasADiSolver())

    assert result.success
    assert result.objective > 0
    # The expensive bus-2 generator is a dispatchable PowerGenerator pinned to its
    # PMIN (Pg = 5 MW -> p_mw = -5 in load convention).
    gen2 = next(
        c
        for c in net.childs_by_ids(net.node_by_id(2).child_ids)
        if isinstance(c.model, PowerGenerator) and hasattr(c.model, "_cost_coeffs")
    )
    assert -gen2.model.p_mw.value == pytest.approx(5.0, abs=1e-2)


def test_mva_line_limit_binds_at_rate_a():
    # The default MVA basis caps apparent power at RATE_A (|S| <= max_loading *
    # RATE_A). With a cheap slack and an expensive local generator, the optimum
    # imports as much as the line allows and serves the rest locally, so the line
    # apparent power sits right at RATE_A.
    rate_a = 60.0
    bus = [
        [1, 3, 0, 0, 0, 0, 1, 1.0, 0, 110, 1, 1.1, 0.9],
        [2, 2, 100, 0, 0, 0, 1, 1.0, 0, 110, 1, 1.1, 0.9],
    ]
    gen = [
        [1, 0, 0, 200, -200, 1.0, 200, 1, 300, 0],  # cheap slack
        [2, 0, 0, 200, -200, 1.0, 200, 1, 200, 0],  # expensive local
    ]
    gencost = [
        [2, 0, 0, 2, 10.0, 0.0],  # 10 /MW
        [2, 0, 0, 2, 50.0, 0.0],  # 50 /MW
    ]
    branch = [[1, 2, 0.01, 0.05, 0, rate_a, 0, 0, 0, 0, 1, -30, 30]]
    mpc = {
        "baseMVA": 100.0,
        "bus": bus,
        "gen": gen,
        "branch": branch,
        "gencost": gencost,
    }

    # default basis is "mva", default max_loading is 1.0
    net, prob = build_matpower_opf(mpc)
    result = run_energy_flow_optimization(net, prob, solver=CasADiSolver())
    assert result.success

    line = net.branches[0].model
    s_from = math.hypot(line.p_from_mw.value, line.q_from_mvar.value)
    s_to = math.hypot(line.p_to_mw.value, line.q_to_mvar.value)
    # apparent power respects the cap on both ends and is binding at RATE_A
    assert max(s_from, s_to) <= rate_a + 1e-2
    assert max(s_from, s_to) == pytest.approx(rate_a, abs=0.5)


def test_powerflow_slack_seed_uses_load_convention():
    # The slack's free P/Q vars are seeded with -Pg/-Qg (load convention),
    # matching the OPF path in fill_opf_child_dict.
    bus = [
        [1, 3, 0, 0, 0, 0, 1, 1.0, 0, 110, 1, 1.1, 0.9],
        [2, 1, 50, 10, 0, 0, 1, 1.0, 0, 110, 1, 1.1, 0.9],
    ]
    gen = [[1, 80, 5, 100, -100, 1.0, 100, 1, 200, 0]]
    branch = [[1, 2, 0.01, 0.05, 0, 0, 0, 0, 0, 0, 1, -30, 30]]

    net = read_matpower_data(_mpc(bus, gen, branch))

    ext = next(c for c in net.childs if isinstance(c.model, ExtPowerGrid))
    assert ext.model.p_mw.value == -80
    assert ext.model.q_mvar.value == -5


def _two_bus_opf_mpc(gencost):
    bus = [
        [1, 3, 0, 0, 0, 0, 1, 1.0, 0, 110, 1, 1.1, 0.9],
        [2, 2, 50, 10, 0, 0, 1, 1.0, 0, 110, 1, 1.1, 0.9],
    ]
    gen = [
        [1, 0, 0, 100, -100, 1.0, 100, 1, 200, 0],
        [2, 0, 0, 100, -100, 1.0, 100, 1, 100, 0],
    ]
    branch = [[1, 2, 0.01, 0.05, 0, 0, 0, 0, 0, 0, 1, -30, 30]]
    return {
        "baseMVA": 100.0,
        "bus": bus,
        "gen": gen,
        "branch": branch,
        "gencost": gencost,
    }


def test_pwl_gencost_two_points_converted_to_linear():
    # A 2-point PWL cost is exactly linear: (0, 0) -> (100, 1000) is 10 /MW.
    gencost = [
        [2, 0, 0, 2, 10.0, 0.0],
        [1, 0, 0, 2, 0.0, 0.0, 100.0, 1000.0],
    ]

    net, _ = build_matpower_opf(_two_bus_opf_mpc(gencost))

    gen2 = next(
        c
        for c in net.childs_by_ids(net.node_by_id(2).child_ids)
        if isinstance(c.model, PowerGenerator)
    )
    assert gen2.model._cost_coeffs == [10.0, 0.0]


def test_pwl_gencost_multipoint_warns_and_drops_cost():
    gencost = [
        [2, 0, 0, 2, 10.0, 0.0],
        [1, 0, 0, 3, 0.0, 0.0, 50.0, 500.0, 100.0, 1500.0],
    ]

    with pytest.warns(UserWarning, match="piecewise-linear.*bus 2"):
        net, _ = build_matpower_opf(_two_bus_opf_mpc(gencost))

    gen2 = next(
        c
        for c in net.childs_by_ids(net.node_by_id(2).child_ids)
        if isinstance(c.model, PowerGenerator)
    )
    assert not hasattr(gen2.model, "_cost_coeffs")


def test_parse_m_strips_line_continuations():
    mpc = _mpc_from_m_text(
        "mpc.baseMVA = 100;\n"
        "mpc.bus = [1 3 0 0 0 0 1 ...\n"
        "  1.0 0 110 1 1.1 0.9;\n"
        "  2 1 50 10 0 0 1 ... trailing text\n"
        "  1.0 0 110 1 1.1 0.9];\n"
        "mpc.gen = [1 80 5 100 -100 1.0 100 1 200 0];\n"
        "mpc.branch = [1 2 0.01 0.05 0 0 0 0 0 0 1 -30 30];\n"
    )
    assert len(mpc["bus"]) == 2
    assert mpc["bus"][0] == [1, 3, 0, 0, 0, 0, 1, 1.0, 0, 110, 1, 1.1, 0.9]
    assert mpc["bus"][1] == [2, 1, 50, 10, 0, 0, 1, 1.0, 0, 110, 1, 1.1, 0.9]


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

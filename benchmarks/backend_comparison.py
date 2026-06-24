"""Backend-performance comparison for monee.

Two head-to-head solver-backend shoot-outs, each on a spread of representative
test cases. The timer covers the full model-build and solve, that is the
``run_energy_flow`` / ``run_energy_flow_optimization`` / ``run_timeseries``
call. The monee ``Network`` is constructed before the timer starts, so the
network build is excluded; what is timed is everything the backend does:
equation assembly, model construction in the solver, and the solve itself.

Group A: GEKKO vs CasADi (smooth-NLP formulations)
    A1  EL power flow                single sector   (cigre_mv, mv_oberrhein)
    A2  EL OPF (economic dispatch)   single sector   (feeder_40, feeder_80)
    A3  Heat power flow              single sector   (water loop)
    A4  Multi-sector power flow      gas + heat      (g2h coupled net)
    A5  EL timeseries                single sector   (memory-less load profile)
    A6  EL timeseries + storage      single sector   (carried SoC coupling)
    A7  Gas timeseries + linepack    single sector   (carried pipe-mass coupling)

Group B: Pyomo/Gurobi vs native gurobipy ((MI)QCQP / MISOCP formulations)
    B1  EL power flow                single sector   (urban district, MISOCP)
    B2  EL OPF (load shedding)       single sector   (urban district, MISOCP)
    B3  Multi-sector power flow      el + gas + heat (convex MIQCQP)
    B4  Multi-sector OPF (shedding)  el + gas + heat (convex MIQCQP)
    B5  Gas timeseries + linepack    single sector   (carried pipe-mass coupling)
    B6  Heat timeseries + LTC        single sector   (carried thermal-inertia coupling)

Temporal-coupling extension cases (linepack / LTC) are added to both groups
where both backends solve them. The LTC heat timeseries is shown only for the
Gurobi backends: GEKKO's IPOPT diverges on the LTC inter-temporal heat NLP for a
varying demand (CasADi solves it, but a comparison needs both NLP backends), so
LTC is restricted to the (MI)QCQP group where Pyomo/Gurobi and gurobipy agree.

Each case is solved on both backends; the result reports the per-backend time,
the speedup, and a cross-backend correctness metric (the two backends must agree
on the physics / optimum). Results are written to ``results/backend_comparison.csv``
and rendered to an interactive Plotly figure (``results/backend_comparison.html``
plus a static ``.png`` when kaleido is available).

Run:  python benchmarks/backend_comparison.py
"""

from __future__ import annotations

import contextlib
import io
import os
import statistics
import tempfile
import time
import uuid
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandapower as pp
import pandapower.networks as ppn
import pandas as pd
import scipy.io

# pandapower 3.x relocated the MATPOWER exporter out of the top-level
# ``converter`` namespace; import it from wherever it lives.
try:
    from pandapower.converter.matpower import to_mpc
except ImportError:  # pragma: no cover - older pandapower
    from pandapower.converter import to_mpc

import monee.model as mm
from monee import (
    GasLinepack,
    LumpedThermalCapacitance,
    TimeseriesData,
    run_energy_flow,
    run_energy_flow_optimization,
    run_timeseries,
)
from monee.io.from_pandapower import _strip_transformer_vector_group_shifts
from monee.io.matpower import read_matpower_case
from monee.model import Network, Var
from monee.model.branch import GenericPowerBranch
from monee.model.child import ExtPowerGrid, PowerGenerator, PowerLoad
from monee.model.formulation import (
    EL_MISOCP_FORMULATION,
    EL_NLP_FORMULATION,
    GAS_CONVEX_MIQCQP_FORMULATION,
    HEAT_NONCONVEX_MIQCQP_FORMULATION,
    make_gas_nlp_formulation,
    make_heat_convex_milp_formulation,
    make_heat_nlp_formulation,
)
from monee.model.node import Bus
from monee.model.storage import ElectricStorage
from monee.problem import create_min_load_shedding_problem
from monee.problem.core import Constraints, Objectives, OptimizationProblem
from monee.problem.utils import line_loading_limit
from monee.solver import GEKKOSolver, PyomoSolver
from monee.solver.casadi import CasADiSolver
from monee.solver.gurobipy import GurobipySolver

HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, "results")


@contextlib.contextmanager
def _silent():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        yield


def _time(fn, repeats: int = 3):
    """Run the solve ``repeats`` times; return (last_result, median_seconds).

    Only the call is timed, the network is already built.
    """
    times, res = [], None
    for _ in range(repeats):
        with _silent():
            t0 = time.perf_counter()
            res = fn()
            times.append(time.perf_counter() - t0)
    return res, statistics.median(times)


def _mnet(loader):
    """Build a monee Network from a pandapower net via the matpower bridge."""
    net = loader()
    tmp = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.mat")
    to_mpc(net, init="flat", filename=tmp)
    m = read_matpower_case(tmp)
    os.remove(tmp)
    # Normalise transformer vector-group phase shifts that to_mpc exports
    # (e.g. mv_oberrhein's 150 degree shift): a lone shift across one branch
    # collapses the flat-start AC NLP onto the spurious low-voltage root, which
    # both backends then fail to recover from. from_pandapower_net strips these;
    # the matpower bridge does too here.
    _strip_transformer_vector_group_shifts(m)
    return m


# Correctness probes: each returns a 1-D vector of a physical/optimal quantity.
def _probe_bus_vm(res):
    return res.dataframes["Bus"]["vm_pu"].to_numpy(dtype=float)


def _probe_objective(res):
    return np.array([float(res.objective)])


def _probe_ext_hydr_flow(res):
    return res.dataframes["ExtHydrGrid"]["mass_flow_kgs"].to_numpy(dtype=float)


def _probe_ts_bus_vm(res):
    return res.get_result_for(mm.Bus, "vm_pu").to_numpy(dtype=float).flatten()


def _probe_ts_storage_soc(res, sid):
    return res.get_result_for_id(sid, "e_mwh").to_numpy(dtype=float)


def _probe_ts_linepack(res):
    return res.get_result_for(mm.GasPipe, "linepack_kg").to_numpy(dtype=float).flatten()


def _probe_ts_junction_t(res):
    return res.get_result_for(mm.Junction, "t_pu").to_numpy(dtype=float).flatten()


def _rel_or_abs_err(a, b):
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    denom = np.maximum(np.abs(b), 1.0)
    return float(np.nanmax(np.abs(a - b) / denom))


# Economic-dispatch OPF builder (EL, for the NLP backends).
def _poly(coeffs, g):
    n = len(coeffs)
    total = 0.0
    for k, c in enumerate(coeffs):
        p = n - 1 - k
        total = total + (c if p == 0 else (c * g if p == 1 else c * g**p))
    return total


def _feeder(n_bus, lat_len=6):
    net = pp.create_empty_network(sn_mva=1.0)
    b0 = pp.create_bus(net, vn_kv=20.0, min_vm_pu=0.9, max_vm_pu=1.1, name="slack")
    pp.create_ext_grid(
        net,
        b0,
        vm_pu=1.0,
        va_degree=0.0,
        min_p_mw=-1e3,
        max_p_mw=1e3,
        min_q_mvar=-1e3,
        max_q_mvar=1e3,
    )
    made, lateral, j = 1, 0, 0
    while made < n_bus:
        prev = b0
        for _ in range(min(lat_len, n_bus - made)):
            bb = pp.create_bus(net, vn_kv=20.0, min_vm_pu=0.9, max_vm_pu=1.1)
            pp.create_line_from_parameters(
                net,
                prev,
                bb,
                length_km=0.8,
                r_ohm_per_km=0.15,
                x_ohm_per_km=0.2,
                c_nf_per_km=0.0,
                max_i_ka=10.0,
            )
            pp.create_load(net, bb, p_mw=0.3, q_mvar=0.06)
            prev = bb
            made += 1
        if lateral % 2 == 0:
            gi = pp.create_gen(
                net,
                prev,
                p_mw=0.4,
                min_p_mw=0.0,
                max_p_mw=0.8,
                min_q_mvar=-2,
                max_q_mvar=2,
                vm_pu=1.0,
                controllable=True,
            )
            pp.create_poly_cost(net, gi, "gen", cp1_eur_per_mw=10.0 + 5.0 * j)
            j += 1
        lateral += 1
    pp.create_poly_cost(net, 0, "ext_grid", cp1_eur_per_mw=50.0)
    return net


def _build_econ_dispatch(pp_net, max_i_ka=None, max_loading=None):
    """Build (monee Network, OptimizationProblem) for an EL economic dispatch
    from a pandapower net carrying create_gen generators and poly costs.

    ``max_i_ka`` restores the per-branch thermal rating dropped by the matpower
    round-trip (matpower carries no line ratings). ``max_loading`` (per-unit)
    adds a line-loading limit constraint per branch end, the AC analogue of
    pandapower's ``max_loading_percent``, so the OPF is line-constrained.
    """
    tmp = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.mat")
    to_mpc(pp_net, init="flat", filename=tmp)
    mpc = scipy.io.loadmat(tmp)["mpc"]
    genmat, gencost, busmat = mpc["gen"][0][0], mpc["gencost"][0][0], mpc["bus"][0][0]
    specs = {}
    for r in range(len(genmat)):
        bus = int(genmat[r, 0])
        nc = int(gencost[r, 3])
        specs[bus] = (
            [float(x) for x in gencost[r, 4 : 4 + nc]],
            (float(genmat[r, 9]), float(genmat[r, 8])),
            (float(genmat[r, 4]), float(genmat[r, 3])),
        )
    vlim = {
        int(busmat[r, 0]): (float(busmat[r, 12]), float(busmat[r, 11]))
        for r in range(len(busmat))
    }
    mnet = read_matpower_case(tmp)
    os.remove(tmp)
    for node in mnet.nodes:
        if (
            isinstance(node.model, Bus)
            and node.id in vlim
            and isinstance(node.model.vm_pu, Var)
        ):
            node.model.vm_pu.min, node.model.vm_pu.max = vlim[node.id]
    node_of_child = {cid: node.id for node in mnet.nodes for cid in node.child_ids}
    for c in mnet.childs:
        if isinstance(c.model, (PowerGenerator, ExtPowerGrid)):
            coeffs, (pmin, pmax), (qmin, qmax) = specs[node_of_child[c.id]]
            gp = min(max(-float(mm.value(c.model.p_mw)), pmin), pmax)
            c.model.p_mw = Var(-gp, max=-pmin, min=-pmax)
            c.model.q_mvar = Var(0.0, max=-qmin, min=-qmax)
            c.model._cost_coeffs = coeffs
    if max_i_ka is not None:
        for branch in mnet.branches:
            branch.model.max_i_ka = max_i_ka
    mnet.apply_formulation(EL_NLP_FORMULATION)
    prob = OptimizationProblem()
    obj = Objectives()
    obj.select(
        lambda m: (
            isinstance(m, (PowerGenerator, ExtPowerGrid)) and hasattr(m, "_cost_coeffs")
        )
    ).calculate(lambda models: sum(_poly(m._cost_coeffs, -m.p_mw) for m in models))
    prob.objectives = obj
    cons = Constraints()
    if max_loading is not None:
        cons.select_types(GenericPowerBranch).equation(
            lambda m: line_loading_limit(m, "from", max_loading)
        ).equation(lambda m: line_loading_limit(m, "to", max_loading))
    prob.constraints = cons
    return mnet, prob


# Network builders for the remaining cases.
def _water_loop(source_t_k=330):
    """3-junction water loop.

    Inlined from tests.util.create_water_loop so the benchmark runs standalone,
    without the test package on sys.path.
    """
    net = mm.Network()
    n0 = net.node(
        mm.Junction(), mm.WATER, child_ids=[net.child(mm.ExtHydrGrid(t_k=356))]
    )
    n1 = net.node(
        mm.Junction(),
        mm.WATER,
        child_ids=[net.child(mm.Source(mass_flow_kgs=5, t_k=source_t_k))],
    )
    n2 = net.node(
        mm.Junction(), mm.WATER, child_ids=[net.child(mm.Sink(mass_flow_kgs=10))]
    )
    pipe = dict(diameter_m=0.3, length_m=100.0)
    net.branch(mm.WaterPipe(**pipe), n0, n1)
    net.branch(mm.WaterPipe(**pipe), n1, n2)
    net.branch(mm.WaterPipe(**pipe), n2, n0)
    return net


def _g2h_net():
    """Gas (source/ext/sink) and water (supply/return) coupled by a G2H.

    Inlined from tests.util.create_g2h_net.
    """
    import monee.express as mx

    pn = mm.Network()
    gas_grid = mm.create_gas_grid("gas", type="lgas")
    g0 = pn.node(
        mm.Junction(), child_ids=[pn.child(mm.Source(mass_flow_kgs=1))], grid=gas_grid
    )
    g1 = pn.node(mm.Junction(), child_ids=[pn.child(mm.ExtHydrGrid())], grid=gas_grid)
    g2 = pn.node(
        mm.Junction(), child_ids=[pn.child(mm.Sink(mass_flow_kgs=1))], grid=gas_grid
    )
    pn.branch(
        mm.GasPipe(
            diameter_m=0.3, length_m=100, temperature_ext_k=300, roughness_m=0.01
        ),
        g0,
        g1,
    )
    pn.branch(
        mm.GasPipe(
            diameter_m=0.3, length_m=150, temperature_ext_k=300, roughness_m=0.01
        ),
        g0,
        g2,
    )

    w0 = pn.node(
        mm.Junction(),
        grid=mm.WATER_KEY,
        child_ids=[pn.child(mm.Sink(mass_flow_kgs=0.1))],
    )
    w1 = pn.node(
        mm.Junction(), child_ids=[pn.child(mm.ConsumeHydrGrid(1))], grid=mm.WATER_KEY
    )
    w2 = pn.node(mm.Junction(), grid=mm.WATER_KEY)
    w3 = pn.node(
        mm.Junction(), grid=mm.WATER_KEY, child_ids=[pn.child(mm.ExtHydrGrid(t_k=359))]
    )
    pn.branch(mm.WaterPipe(diameter_m=0.15, length_m=100), w0, w1)
    pn.branch(mm.WaterPipe(diameter_m=0.15, length_m=200), w3, w2)

    mx.create_g2h(
        pn,
        gas_node_id=g2,
        heat_node_id=w2,
        heat_return_node_id=w1,
        heat_energy_mw=0.010,
        diameter_m=0.4,
        efficiency=0.9,
    )
    return pn


def _heat_loop_nlp():
    net = _water_loop(source_t_k=330)
    net.apply_formulation(make_heat_nlp_formulation())
    return net


def _g2h_nlp():
    net = _g2h_net()
    net.apply_formulation(make_gas_nlp_formulation())
    net.apply_formulation(make_heat_nlp_formulation())
    return net


def _g2h_miqcqp():
    net = _g2h_net()
    net.apply_formulation(GAS_CONVEX_MIQCQP_FORMULATION)
    net.apply_formulation(HEAT_NONCONVEX_MIQCQP_FORMULATION)
    net.apply_formulation(EL_MISOCP_FORMULATION)
    return net


def _urban_misocp():
    from monee.network import create_urban_district_net

    net = create_urban_district_net()
    net.apply_formulation(EL_MISOCP_FORMULATION)
    return net


def _el_ts_net():
    net = Network(mm.PowerGrid(name="power", sn_mva=1))
    b0 = net.node(
        Bus(base_kv=1),
        grid=mm.EL,
        child_ids=[net.child(ExtPowerGrid(p_mw=0, q_mvar=0, vm_pu=1, va_degree=0))],
    )
    lid = net.child(PowerLoad(p_mw=1.0, q_mvar=0.1))
    b1 = net.node(
        Bus(base_kv=1),
        grid=mm.EL,
        child_ids=[lid, net.child(PowerGenerator(p_mw=0.5, q_mvar=0))],
    )
    net.branch(
        mm.PowerLine(length_m=100, r_ohm_per_m=1e-4, x_ohm_per_m=1e-4, parallel=1),
        b0,
        b1,
    )
    net.apply_formulation(EL_NLP_FORMULATION)
    return net, lid


def _el_storage_ts_net():
    net = Network(mm.PowerGrid(name="power", sn_mva=1))
    b0 = net.node(
        Bus(base_kv=1),
        grid=mm.EL,
        child_ids=[net.child(ExtPowerGrid(p_mw=0, q_mvar=0, vm_pu=1, va_degree=0))],
    )
    lid = net.child(PowerLoad(p_mw=2.0, q_mvar=0.0))
    sid = net.child(
        ElectricStorage(e_mwh_initial=5.0, e_mwh_max=10.0, p_max_mw=2.0), name="storage"
    )
    b1 = net.node(Bus(base_kv=1), grid=mm.EL, child_ids=[lid, sid])
    net.branch(
        mm.PowerLine(length_m=100, r_ohm_per_m=1e-4, x_ohm_per_m=1e-4, parallel=1),
        b0,
        b1,
    )
    net.apply_formulation(EL_NLP_FORMULATION)
    return net, sid


def _gas_linepack_net(formulation):
    """Linear gas net (ext-grid, pipe, sink) with the GasLinepack extension.

    Pipe storage couples gas mass across steps. Returns (net, sink_id).
    """
    net = mm.Network(mm.create_gas_grid("gas", type="lgas"))
    net.activate_grid(mm.GAS)
    src = net.child(mm.ExtHydrGrid())
    sink = net.child(mm.Sink(mass_flow_kgs=0.2))
    n0 = net.node(mm.Junction(), mm.GAS, child_ids=[src])
    n1 = net.node(mm.Junction(), mm.GAS, child_ids=[sink])
    net.branch(mm.GasPipe(diameter_m=0.5, length_m=5000), n0, n1)
    net.add_extension(GasLinepack())
    net.apply_formulation(formulation)
    return net, sink


def _water_ltc_net(formulation):
    """3-junction water loop with the LumpedThermalCapacitance extension.

    Junction thermal inertia couples temperature across steps. Returns
    (net, sink_id).
    """
    net = mm.Network()
    n0 = net.node(
        mm.Junction(), mm.WATER, child_ids=[net.child(mm.ExtHydrGrid(t_k=356))]
    )
    n1 = net.node(
        mm.Junction(),
        mm.WATER,
        child_ids=[net.child(mm.Source(mass_flow_kgs=5, t_k=356))],
    )
    sink = net.child(mm.Sink(mass_flow_kgs=10))
    n2 = net.node(mm.Junction(), mm.WATER, child_ids=[sink])
    pipe = dict(diameter_m=0.15, length_m=100)
    net.branch(mm.WaterPipe(**pipe), n0, n1)
    net.branch(mm.WaterPipe(**pipe), n1, n2)
    net.branch(mm.WaterPipe(**pipe), n2, n0)
    net.add_extension(LumpedThermalCapacitance())
    net.apply_formulation(formulation)
    return net, sink


def _shedding_problem():
    return create_min_load_shedding_problem(
        bounds_vm=(0.5, 1.5),
        bounds_pressure=(0.5, 1.5),
        bounds_t=(0.5, 1.5),
        include_ext_grids=False,
        include_storages=False,
    )


# Suites.
def _n_bus(res):
    try:
        return int(len(res.dataframes["Bus"]))
    except Exception:
        return None


def _row(group, case, sector, problem, ba, ta, ra, bb, tb, rb, probe, size):
    """Assemble one comparison row. *probe* maps a result to a comparison
    vector; correctness is the max rel/abs error between the two backends."""
    try:
        err = _rel_or_abs_err(probe(ra), probe(rb))
    except Exception:
        err = float("nan")
    return dict(
        group=group,
        case=case,
        sector=sector,
        problem=problem,
        backend_a=ba,
        backend_b=bb,
        time_a_ms=round(ta * 1000, 1),
        time_b_ms=round(tb * 1000, 1),
        speedup=round(ta / tb, 2) if tb else float("nan"),
        ok_a=bool(
            getattr(ra, "success", False) or getattr(ra, "failed_steps", None) == []
        ),
        ok_b=bool(
            getattr(rb, "success", False) or getattr(rb, "failed_steps", None) == []
        ),
        cross_err=err,
        size=size,
    )


def run_nlp_suite():
    """Group A: GEKKO vs CasADi on smooth-NLP cases."""
    print("\n--- Group A: GEKKO vs CasADi (smooth NLP) ---")
    rows = []
    A, B = "GEKKO", "CasADi"

    # A1 EL power flow (single sector)
    for name, loader in [
        ("cigre_mv", ppn.create_cigre_network_mv),
        ("mv_oberrhein", ppn.mv_oberrhein),
    ]:
        ng = _mnet(loader)
        rg, tg = _time(
            lambda: run_energy_flow(ng, solver=GEKKOSolver(), simulation=True)
        )
        nc = _mnet(loader)
        rc, tc = _time(
            lambda: run_energy_flow(nc, solver=CasADiSolver(), simulation=True)
        )
        rows.append(
            _row(
                "A",
                f"EL PF · {name}",
                "electricity",
                "power flow",
                A,
                tg,
                rg,
                B,
                tc,
                rc,
                _probe_bus_vm,
                _n_bus(rg),
            )
        )
        print(
            f"  {rows[-1]['case']:28s} {A} {tg * 1000:7.1f}ms  {B} {tc * 1000:7.1f}ms  x{rows[-1]['speedup']}"
        )

    # A2 EL OPF economic dispatch (single sector)
    for n in (40, 80):
        mg, pg = _build_econ_dispatch(_feeder(n))
        rg, tg = _time(
            lambda: run_energy_flow_optimization(mg, pg, solver=GEKKOSolver())
        )
        mc, pc_ = _build_econ_dispatch(_feeder(n))
        rc, tc = _time(
            lambda: run_energy_flow_optimization(mc, pc_, solver=CasADiSolver())
        )
        rows.append(
            _row(
                "A",
                f"EL OPF · feeder_{n}",
                "electricity",
                "optimization",
                A,
                tg,
                rg,
                B,
                tc,
                rc,
                _probe_objective,
                _n_bus(rg),
            )
        )
        print(
            f"  {rows[-1]['case']:28s} {A} {tg * 1000:7.1f}ms  {B} {tc * 1000:7.1f}ms  x{rows[-1]['speedup']}"
        )

    # A3 Heat power flow (single sector)
    rg, tg = _time(
        lambda: run_energy_flow(_heat_loop_nlp(), solver=GEKKOSolver(), simulation=True)
    )
    rc, tc = _time(
        lambda: run_energy_flow(
            _heat_loop_nlp(), solver=CasADiSolver(), simulation=True
        )
    )
    rows.append(
        _row(
            "A",
            "Heat PF · water loop",
            "heat",
            "power flow",
            A,
            tg,
            rg,
            B,
            tc,
            rc,
            _probe_ext_hydr_flow,
            None,
        )
    )
    print(
        f"  {rows[-1]['case']:28s} {A} {tg * 1000:7.1f}ms  {B} {tc * 1000:7.1f}ms  x{rows[-1]['speedup']}"
    )

    # A4 Multi-sector power flow (gas + heat)
    rg, tg = _time(
        lambda: run_energy_flow(_g2h_nlp(), solver=GEKKOSolver(), simulation=False)
    )
    rc, tc = _time(
        lambda: run_energy_flow(_g2h_nlp(), solver=CasADiSolver(), simulation=False)
    )
    rows.append(
        _row(
            "A",
            "Multi-sector PF · g2h",
            "gas+heat",
            "power flow",
            A,
            tg,
            rg,
            B,
            tc,
            rc,
            _probe_ext_hydr_flow,
            None,
        )
    )
    print(
        f"  {rows[-1]['case']:28s} {A} {tg * 1000:7.1f}ms  {B} {tc * 1000:7.1f}ms  x{rows[-1]['speedup']}"
    )

    # A5 EL timeseries (memory-less)
    N = 12
    prof = (1.0 + 0.3 * np.sin(np.linspace(0, 2 * np.pi, N))).tolist()

    def _ts(solver):
        net, lid = _el_ts_net()
        td = TimeseriesData()
        td.add_child_series(lid, "p_mw", prof)
        return run_timeseries(net, td, solver=solver)

    rg, tg = _time(lambda: _ts(GEKKOSolver()), repeats=1)
    rc, tc = _time(lambda: _ts(CasADiSolver()), repeats=1)
    rows.append(
        _row(
            "A",
            f"EL timeseries ({N} steps)",
            "electricity",
            "timeseries",
            A,
            tg,
            rg,
            B,
            tc,
            rc,
            _probe_ts_bus_vm,
            N,
        )
    )
    print(
        f"  {rows[-1]['case']:28s} {A} {tg * 1000:7.1f}ms  {B} {tc * 1000:7.1f}ms  x{rows[-1]['speedup']}"
    )

    # A6 EL timeseries + storage coupling (carried SoC)
    disp = [1.0, -0.5, 0.8, 0.3, -0.6, 1.0, -0.4, 0.5, -0.9, 0.7, -0.3, 0.4]

    def _ts_stor(solver):
        net, sid = _el_storage_ts_net()
        td = TimeseriesData()
        td.add_child_series(sid, "p_mw", disp)
        return run_timeseries(net, td, solver=solver), sid

    (rg, sidg), tg = _time(lambda: _ts_stor(GEKKOSolver()), repeats=1)
    (rc, sidc), tc = _time(lambda: _ts_stor(CasADiSolver()), repeats=1)
    rows.append(
        _row(
            "A",
            f"EL ts + storage ({len(disp)})",
            "electricity",
            "timeseries+coupling",
            A,
            tg,
            rg,
            B,
            tc,
            rc,
            lambda r: _probe_ts_storage_soc(r, sidg if r is rg else sidc),
            len(disp),
        )
    )
    print(
        f"  {rows[-1]['case']:28s} {A} {tg * 1000:7.1f}ms  {B} {tc * 1000:7.1f}ms  x{rows[-1]['speedup']}"
    )

    # A7 Gas timeseries + linepack coupling (carried pipe mass)
    NL = 8
    mdot = [0.2, 0.25, 0.15, 0.3, 0.2, 0.28, 0.18, 0.22]

    def _ts_lp(solver):
        net, sid = _gas_linepack_net(make_gas_nlp_formulation())
        td = TimeseriesData()
        td.add_child_series(sid, "mass_flow_kgs", mdot)
        return run_timeseries(net, td, solver=solver)

    rg, tg = _time(lambda: _ts_lp(GEKKOSolver()), repeats=1)
    rc, tc = _time(lambda: _ts_lp(CasADiSolver()), repeats=1)
    rows.append(
        _row(
            "A",
            f"Gas ts + linepack ({NL})",
            "gas",
            "timeseries+coupling",
            A,
            tg,
            rg,
            B,
            tc,
            rc,
            _probe_ts_linepack,
            NL,
        )
    )
    print(
        f"  {rows[-1]['case']:28s} {A} {tg * 1000:7.1f}ms  {B} {tc * 1000:7.1f}ms  x{rows[-1]['speedup']}"
    )
    return rows


def run_miqcqp_suite():
    """Group B: Pyomo/Gurobi vs native gurobipy on (MI)QCQP / MISOCP cases."""
    print("\n--- Group B: Pyomo/Gurobi vs gurobipy ((MI)QCQP) ---")
    rows = []
    A, B = "pyomo-gurobi", "gurobipy"

    # B1 EL power flow (single sector, MISOCP)
    ra, ta = _time(
        lambda: run_energy_flow(_urban_misocp(), solver=PyomoSolver("gurobi"))
    )
    rb, tb = _time(lambda: run_energy_flow(_urban_misocp(), solver=GurobipySolver()))
    rows.append(
        _row(
            "B",
            "EL PF · urban (MISOCP)",
            "electricity",
            "power flow",
            A,
            ta,
            ra,
            B,
            tb,
            rb,
            _probe_bus_vm,
            _n_bus(ra),
        )
    )
    print(
        f"  {rows[-1]['case']:30s} {A} {ta * 1000:7.1f}ms  {B} {tb * 1000:7.1f}ms  x{rows[-1]['speedup']}"
    )

    # B2 EL OPF load shedding (single sector)
    ra, ta = _time(
        lambda: run_energy_flow_optimization(
            _urban_misocp(), _shedding_problem(), solver=PyomoSolver("gurobi")
        )
    )
    rb, tb = _time(
        lambda: run_energy_flow_optimization(
            _urban_misocp(), _shedding_problem(), solver=GurobipySolver()
        )
    )
    rows.append(
        _row(
            "B",
            "EL OPF · urban shedding",
            "electricity",
            "optimization",
            A,
            ta,
            ra,
            B,
            tb,
            rb,
            _probe_objective,
            _n_bus(ra),
        )
    )
    print(
        f"  {rows[-1]['case']:30s} {A} {ta * 1000:7.1f}ms  {B} {tb * 1000:7.1f}ms  x{rows[-1]['speedup']}"
    )

    # B3 Multi-sector power flow (MIQCQP)
    ra, ta = _time(lambda: run_energy_flow(_g2h_miqcqp(), solver=PyomoSolver("gurobi")))
    rb, tb = _time(lambda: run_energy_flow(_g2h_miqcqp(), solver=GurobipySolver()))
    rows.append(
        _row(
            "B",
            "Multi-sector PF · g2h",
            "el+gas+heat",
            "power flow",
            A,
            ta,
            ra,
            B,
            tb,
            rb,
            _probe_ext_hydr_flow,
            None,
        )
    )
    print(
        f"  {rows[-1]['case']:30s} {A} {ta * 1000:7.1f}ms  {B} {tb * 1000:7.1f}ms  x{rows[-1]['speedup']}"
    )

    # B4 Multi-sector OPF load shedding (MIQCQP)
    ra, ta = _time(
        lambda: run_energy_flow_optimization(
            _g2h_miqcqp(), _shedding_problem(), solver=PyomoSolver("gurobi")
        )
    )
    rb, tb = _time(
        lambda: run_energy_flow_optimization(
            _g2h_miqcqp(), _shedding_problem(), solver=GurobipySolver()
        )
    )
    rows.append(
        _row(
            "B",
            "Multi-sector OPF · g2h",
            "el+gas+heat",
            "optimization",
            A,
            ta,
            ra,
            B,
            tb,
            rb,
            _probe_objective,
            None,
        )
    )
    print(
        f"  {rows[-1]['case']:30s} {A} {ta * 1000:7.1f}ms  {B} {tb * 1000:7.1f}ms  x{rows[-1]['speedup']}"
    )

    # B5 Gas timeseries + linepack coupling (convex MIQCQP)
    NL = 8
    mdot = [0.2, 0.25, 0.15, 0.3, 0.2, 0.28, 0.18, 0.22]

    def _ts_lp(solver):
        net, sid = _gas_linepack_net(GAS_CONVEX_MIQCQP_FORMULATION)
        td = TimeseriesData()
        td.add_child_series(sid, "mass_flow_kgs", mdot)
        return run_timeseries(net, td, solver=solver)

    ra, ta = _time(lambda: _ts_lp(PyomoSolver("gurobi")), repeats=1)
    rb, tb = _time(lambda: _ts_lp(GurobipySolver()), repeats=1)
    rows.append(
        _row(
            "B",
            f"Gas ts + linepack ({NL})",
            "gas",
            "timeseries+coupling",
            A,
            ta,
            ra,
            B,
            tb,
            rb,
            _probe_ts_linepack,
            NL,
        )
    )
    print(
        f"  {rows[-1]['case']:30s} {A} {ta * 1000:7.1f}ms  {B} {tb * 1000:7.1f}ms  x{rows[-1]['speedup']}"
    )

    # B6 Heat timeseries + LTC coupling (convex MILP); Gurobi-only - GEKKO NLP diverges
    NT = 8
    mdot_w = [10.0, 10.0, 6.0, 6.0, 8.0, 12.0, 9.0, 7.0]

    def _ts_ltc(solver):
        net, sid = _water_ltc_net(make_heat_convex_milp_formulation())
        td = TimeseriesData()
        td.add_child_series(sid, "mass_flow_kgs", mdot_w)
        return run_timeseries(net, td, solver=solver)

    ra, ta = _time(lambda: _ts_ltc(PyomoSolver("gurobi")), repeats=1)
    rb, tb = _time(lambda: _ts_ltc(GurobipySolver()), repeats=1)
    rows.append(
        _row(
            "B",
            f"Heat ts + LTC ({NT})",
            "heat",
            "timeseries+coupling",
            A,
            ta,
            ra,
            B,
            tb,
            rb,
            _probe_ts_junction_t,
            NT,
        )
    )
    print(
        f"  {rows[-1]['case']:30s} {A} {ta * 1000:7.1f}ms  {B} {tb * 1000:7.1f}ms  x{rows[-1]['speedup']}"
    )
    return rows


# Plot.
def make_plot(
    df: pd.DataFrame, out_html: str, out_png: str, out_svg: str | None = None
):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    color = {
        "GEKKO": "#9467bd",
        "CasADi": "#2ca02c",
        "pyomo-gurobi": "#1f77b4",
        "gurobipy": "#ff7f0e",
    }
    # The figure is transparent and embedded in both light and dark docs themes,
    # so all text uses one theme-neutral mid-grey (roughly 4:1 contrast on white
    # and on dark) and only the bars carry saturated colour.
    TEXT = "#737373"
    GRID = "rgba(128,128,128,0.22)"
    AXIS_LINE = "rgba(128,128,128,0.5)"
    groups = [
        ("A", "Group A: GEKKO vs CasADi (smooth NLP)"),
        ("B", "Group B: Pyomo/Gurobi vs native gurobipy ((MI)QCQP)"),
    ]

    # Row heights proportional to each group's case count (and figure height
    # proportional to the total) so every bar has the same on-screen thickness
    # even though the groups hold different numbers of cases.
    counts = [int((df.group == g[0]).sum()) for g in groups]
    total_cases = max(sum(counts), 1)

    # No per-panel subplot titles: the group banner on the left names each row's
    # backends and the bottom-row x-axis titles label the columns, so per-row
    # titles would only clutter and overlap the banners.
    fig = make_subplots(
        rows=2,
        cols=2,
        column_widths=[0.62, 0.38],
        row_heights=[c / total_cases for c in counts],
        shared_yaxes=True,  # case labels render once, on the far left only
        horizontal_spacing=0.06,
        vertical_spacing=0.12,
    )

    for r, (g, _title) in enumerate(groups, start=1):
        sub = df[df.group == g].iloc[::-1]  # reversed so the first case lands on top
        cases = sub.case.tolist()
        last_row = r == len(groups)
        ba = sub.backend_a.iloc[0]
        bb = sub.backend_b.iloc[0]
        # solve-time bars, grouped reference vs native
        fig.add_trace(
            go.Bar(
                y=cases,
                x=sub.time_a_ms,
                name=ba,
                orientation="h",
                marker_color=color.get(ba, "#888"),
                marker_line_width=0,
                legendgroup=ba,
                showlegend=True,
                cliponaxis=False,
                text=[f"{v:.0f}" for v in sub.time_a_ms],
                textposition="outside",
                textfont={"size": 23, "color": TEXT},
                hovertemplate=f"{ba}: %{{x:.1f}} ms<extra></extra>",
            ),
            row=r,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                y=cases,
                x=sub.time_b_ms,
                name=bb,
                orientation="h",
                marker_color=color.get(bb, "#888"),
                marker_line_width=0,
                legendgroup=bb,
                showlegend=True,
                cliponaxis=False,
                text=[f"{v:.0f}" for v in sub.time_b_ms],
                textposition="outside",
                textfont={"size": 23, "color": TEXT},
                hovertemplate=f"{bb}: %{{x:.1f}} ms<extra></extra>",
            ),
            row=r,
            col=1,
        )
        spd = sub.speedup
        fig.add_trace(
            go.Bar(
                y=cases,
                x=spd,
                orientation="h",
                showlegend=False,
                cliponaxis=False,
                marker_color=[
                    color.get(bb, "#888") if v >= 1 else "#d62728" for v in spd
                ],
                marker_line_width=0,
                text=[f"×{v:.1f}" for v in spd],
                textposition="outside",
                textfont={"size": 23, "color": TEXT},
                hovertemplate="speedup ×%{x:.2f}<extra></extra>",
            ),
            row=r,
            col=2,
        )
        # headroom so the outside value labels on the longest bars aren't clipped
        tvals = sub[["time_a_ms", "time_b_ms"]].to_numpy(dtype=float)
        fig.update_xaxes(
            type="log",
            row=r,
            col=1,
            title_text="solve time (ms, log)" if last_row else None,
            range=[np.log10(np.nanmin(tvals) * 0.5), np.log10(np.nanmax(tvals) * 3.4)],
        )
        fig.update_xaxes(
            row=r,
            col=2,
            title_text="× faster (native vs reference)" if last_row else None,
            range=[0, float(np.nanmax(spd)) * 1.32],
        )
        fig.add_vline(x=1.0, line_dash="dot", line_color=AXIS_LINE, row=r, col=2)

        # group banner over the left column, anchored to the subplot's domain
        fig.add_annotation(
            text=f"<b>{_title}</b>",
            row=r,
            col=1,
            xref="x domain",
            yref="y domain",
            x=0,
            y=1.0,
            yshift=22,
            showarrow=False,
            font={"size": 27, "color": TEXT},
            xanchor="left",
        )

    # Uniform axis cosmetics on every subplot.
    fig.update_xaxes(
        showgrid=True,
        gridcolor=GRID,
        gridwidth=1,
        zeroline=False,
        showline=True,
        linecolor=AXIS_LINE,
        linewidth=1,
        ticks="outside",
        ticklen=4,
        tickcolor=AXIS_LINE,
        tickfont={"size": 24, "color": TEXT},
        title_font={"size": 26, "color": TEXT},
    )
    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        showline=False,
        tickfont={"size": 24, "color": TEXT},
        automargin=True,
    )

    fig.update_layout(
        title={
            "text": "<b>monee backend performance comparison</b><br>"
            f"<span style='font-size:27px;color:{TEXT}'>solver-backend "
            "shoot-outs across representative cases: solve time and speedup</span>",
            "x": 0.5,
            "xanchor": "center",
            "y": 0.978,
            "yanchor": "top",
            "font": {"size": 35, "color": TEXT},
        },
        barmode="group",
        bargap=0.25,
        bargroupgap=0.1,
        template="plotly_white",
        height=int(70 * total_cases + 110 * len(groups) + 170),
        width=1280,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.012,
            "xanchor": "right",
            "x": 1.0,
            "font": {"size": 25, "color": TEXT},
            "bgcolor": "rgba(0,0,0,0)",
        },
        margin={"l": 235, "r": 60, "t": 215, "b": 70},
        font={"family": "Inter, Segoe UI, Helvetica, Arial", "size": 21, "color": TEXT},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        uniformtext={"mode": "hide", "minsize": 11},
    )

    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"Wrote {out_html}")
    for path, scale in [(out_png, 2), (out_svg, 1)]:
        if not path:
            continue
        try:
            fig.write_image(path, scale=scale)
            print(f"Wrote {path}")
        except Exception as exc:  # kaleido missing / failed
            print(f"(static export skipped for {os.path.basename(path)}: {exc})")


CSV_PATH = os.path.join(RESULTS, "backend_comparison.csv")
HTML_PATH = os.path.join(RESULTS, "backend_comparison.html")
PNG_PATH = os.path.join(RESULTS, "backend_comparison.png")
SVG_PATH = os.path.join(RESULTS, "backend_comparison.svg")


def regenerate_plot():
    """Re-render the Plotly figure from the existing CSV, without solving.

    Use ``--plot-only`` to iterate on the plot without re-running the slow
    benchmark; assumes ``results/backend_comparison.csv`` is fully written.
    """
    if not os.path.exists(CSV_PATH):
        raise SystemExit(
            f"No results CSV at {CSV_PATH}; run the benchmark once "
            "(without --plot-only) to generate it first."
        )
    df = pd.read_csv(CSV_PATH)
    print(f"Re-plotting from {CSV_PATH} ({len(df)} cases)")
    make_plot(df, HTML_PATH, PNG_PATH, SVG_PATH)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    # Warm up each backend once to keep import, licence, and JIT costs out of
    # the timings.
    with _silent():
        for s in (
            GEKKOSolver(),
            CasADiSolver(),
            PyomoSolver("gurobi"),
            GurobipySolver(),
        ):
            try:
                run_energy_flow(
                    _mnet(ppn.create_cigre_network_mv), solver=s, simulation=True
                )
            except Exception:
                pass

    rows = run_nlp_suite() + run_miqcqp_suite()
    df = pd.DataFrame(rows)

    df.to_csv(CSV_PATH, index=False)

    show = df[
        [
            "group",
            "case",
            "problem",
            "backend_a",
            "time_a_ms",
            "backend_b",
            "time_b_ms",
            "speedup",
            "cross_err",
            "ok_a",
            "ok_b",
        ]
    ]
    print("\n=== monee backend comparison ===\n")
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(show.to_string(index=False))
    print(f"\nmax cross-backend error: {df.cross_err.max():.2e}")
    print(f"Wrote {CSV_PATH}")

    make_plot(df, HTML_PATH, PNG_PATH, SVG_PATH)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip all solves; only regenerate the plot from the existing CSV.",
    )
    args = parser.parse_args()
    if args.plot_only:
        regenerate_plot()
    else:
        main()

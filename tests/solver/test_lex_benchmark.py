"""Benchmark: weighted-sum vs lexicographic objective solves.

Runs each min-load-shedding problem N times in each mode on a range
of network sizes (including a simbench-based MES) and reports
median / min / mean wall-clock plus the user-objective value at
the optimum.

Skipped without Gurobi (MISOCP solver needed).  Marked
``-m benchmark`` so it does not slow CI by default.

Weighting convention used in the bench
--------------------------------------

For the head-to-head columns we *keep both modes at the same
weights* (``demand_weight=1e3``, ``generator_weight=0.1``) - that is
an apples-to-apples solve on the same problem definition.  We also
report a separate row for ``lex_O1`` that uses ``demand_weight=1.0``,
``generator_weight=1e-4`` to show what lex looks like at its
"natural" (priority-floor-not-required) scale; the optimum is
identical, only conditioning changes.
"""

from __future__ import annotations

import statistics
import time

import pytest

from monee.model.formulation import (
    MISOCP_NETWORK_FORMULATION,
    make_mccormick_dhs_formulation,
)
from monee.network import (
    create_restoration_benchmark,
    create_urban_district_net,
    generate_supply_return_mes_based_on_power_net,
)
from monee.network.res import create_large_urban_mes_net
from monee.problem.min_load_shedding import create_min_load_shedding_problem
from monee.solver.pyo import PyomoSolver


def _solver_available(name: str) -> bool:
    import pyomo.environ as pyo

    try:
        return pyo.SolverFactory(name).available(exception_flag=False)
    except Exception:
        return False


SOLVER = "gurobi" if _solver_available("gurobi") else None


_SIMBENCH_TEMPLATE = None


def _simbench_mes_net():
    """Build the simbench-rural-MES network referenced in test_mes.py.

    Caches a *template* at module scope (heavy simbench → pandapower →
    MES construction takes ~tens of seconds).  Each benchmark call
    returns a fresh deep copy so per-solve mutations don't leak across
    runs.
    """
    import copy

    import simbench

    from monee.io.from_pandapower import from_pandapower_net

    global _SIMBENCH_TEMPLATE
    if _SIMBENCH_TEMPLATE is None:
        pp_net = simbench.get_simbench_net("1-LV-rural3--1-no_sw")
        mn = from_pandapower_net(pp_net)
        mes = generate_supply_return_mes_based_on_power_net(
            mn,
            coupling_density=0.5,
            centralized=False,
            couplings=("chp", "p2g", "p2h"),
            coupling_kwargs={"seed": 1, "use_hg_variants": True},
            heat_kwargs={"node_based_heat_loads": True},
        )
        mes.apply_formulation(MISOCP_NETWORK_FORMULATION)
        mes.apply_formulation(make_mccormick_dhs_formulation(num_partitions=1))
        _SIMBENCH_TEMPLATE = mes
    return copy.deepcopy(_SIMBENCH_TEMPLATE)


@pytest.fixture(
    scope="module",
    params=[
        ("urban", create_urban_district_net, {}),
        ("large_urban", create_large_urban_mes_net, {"n_districts": 4}),
        ("restoration", create_restoration_benchmark, {"misocp": True}),
        ("simbench_rural_mes", _simbench_mes_net, None),
    ],
    ids=["urban", "large_urban", "restoration", "simbench_rural_mes"],
)
def _network_factory(request):
    name, factory, kwargs = request.param
    return name, factory, kwargs


def _fresh_net(factory, kwargs):
    """Build a fresh network for a benchmark run.

    Factories with kwargs are called with ``**kwargs``.  Factories
    that build the formulation internally (``simbench_rural_mes``,
    ``create_restoration_benchmark(misocp=True)``) pass kwargs=None.
    """
    if kwargs is None:
        return factory()
    net = factory(**kwargs)
    if not getattr(net, "_formulation_applied", False):
        try:
            net.apply_formulation(MISOCP_NETWORK_FORMULATION)
        except Exception:
            pass
    return net


def _bench_once(
    net_factory,
    kwargs,
    *,
    lex: bool,
    weights=(1e3, 0.1),
    auto_priority_floor: bool = False,
):
    """One solve.  Returns (wall-clock, reported solver objective)."""
    demand_w, gen_w = weights
    net = _fresh_net(net_factory, kwargs)
    prob = create_min_load_shedding_problem(
        bounds_el=(0.5, 1.5),
        bounds_gas=(0.5, 1.5),
        bounds_heat=(0.5, 1.5),
        include_ext_grids=False,
        include_storages=False,
        lex_objectives=lex,
        auto_priority_floor=auto_priority_floor,
        demand_weight=demand_w,
        generator_weight=gen_w,
    )
    t0 = time.perf_counter()
    res = PyomoSolver(SOLVER).solve(net, optimization_problem=prob)
    elapsed = time.perf_counter() - t0
    assert res.success, f"solve failed in {'lex' if lex else 'weighted'} mode"
    return elapsed, res.objective


def _stats(xs: list[float]) -> dict[str, float]:
    return {
        "min": min(xs),
        "median": statistics.median(xs),
        "mean": statistics.mean(xs),
        "max": max(xs),
    }


def _fmt_stats(s: dict[str, float]) -> str:
    return (
        f"min={s['min']:.3f}s median={s['median']:.3f}s "
        f"mean={s['mean']:.3f}s max={s['max']:.3f}s"
    )


@pytest.mark.skipif(SOLVER is None, reason="Gurobi (MISOCP) not available")
@pytest.mark.benchmark
def test_benchmark_lex_vs_weighted(_network_factory):
    """Compare wall-clock across three variants on each network:

    - weighted (1e3 / 0.1)       - production baseline
    - lex      (1e3 / 0.1)       - apples-to-apples on the same problem
    - lex_O1   (1.0 / 1e-4)      - lex at its natural (no-floor) scale
    """
    name, factory, kwargs = _network_factory
    repeats = 3

    # Variant tuple: (label, lex, weights, auto_floor).
    # ``weighted_auto`` shows the cost of using the deterministic
    # priority-floor bound instead of the magic ``demand_weight=1e3``
    # default - same algorithm, just safer coefficient choice.
    variants = [
        ("weighted", False, (1e3, 0.1), False),
        ("weighted_auto", False, (1.0, 1e-4), True),
        ("lex_1e3", True, (1e3, 0.1), False),
        ("lex_O1", True, (1.0, 1e-4), False),
    ]

    results = {}
    for label, lex, weights, auto in variants:
        times: list[float] = []
        objs: list[float] = []
        for _ in range(repeats):
            t, obj = _bench_once(
                factory,
                kwargs,
                lex=lex,
                weights=weights,
                auto_priority_floor=auto,
            )
            times.append(t)
            objs.append(obj)
        results[label] = (times, objs)

    print(f"\n[{name}]  (N={repeats})")
    for label, *_ in variants:
        times, objs = results[label]
        print(f"  {label:<14} : {_fmt_stats(_stats(times))}  obj={objs[0]:.6g}")

    base_med = statistics.median(results["weighted"][0])
    auto_med = statistics.median(results["weighted_auto"][0])
    lex_med = statistics.median(results["lex_1e3"][0])
    lex_o1_med = statistics.median(results["lex_O1"][0])
    print(
        f"  ratios  : weighted_auto/weighted={auto_med / base_med:.2f}× "
        f"lex_1e3/weighted={lex_med / base_med:.2f}× "
        f"lex_O1/weighted={lex_o1_med / base_med:.2f}×"
    )

    # Sanity guard only - catches outright bugs (model rebuilt,
    # infinite loop) rather than grading performance.  Real lex
    # overhead is workload-dependent: nearly free on unstressed
    # networks (1–2×) but several × on heavily-stressed problems
    # where phase 2 must re-optimise in a tight degenerate region.
    # Threshold scaled with problem size - generous, intentionally.
    SANITY_FACTOR = 50.0
    for label in ("weighted_auto", "lex_1e3", "lex_O1"):
        med = statistics.median(results[label][0])
        assert med < SANITY_FACTOR * base_med, (
            f"{label} wall-clock {med:.1f}s >> {SANITY_FACTOR}× baseline "
            f"{base_med:.1f}s on '{name}' - likely a bug, not a workload effect"
        )

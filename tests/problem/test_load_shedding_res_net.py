import numpy as np

import monee.model as mm
import monee.problem as mp
from monee import TimeseriesData, run_energy_flow_optimization
from monee.model.formulation import EL_MISOCP_FORMULATION
from monee.network import create_urban_district_net

BOUNDS_EL = (0.9, 1.1)
BOUNDS_HEAT = (0.9, 1.1)
BOUNDS_GAS = (0.9, 1.1)

SEED = 101
TIME_STEPS = 4 * 8  # 32 time steps per run


def _sinusoidal_profile(
    n_steps: int,
    base: float,
    amplitude: float = 0.25,
    noise: float = 0.04,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Bell-shaped daily demand curve with small Gaussian noise, clipped to [50%, 200%] of base."""
    if rng is None:
        rng = np.random.default_rng()
    t = np.linspace(0, 2 * np.pi, n_steps, endpoint=False)
    profile = base * (1.0 + amplitude * np.sin(t - np.pi / 2))
    profile += rng.normal(0, noise * base, n_steps)
    return np.clip(profile, 0.5 * base, 2.0 * base)


def _make_urban_district_timeseries(
    net: mm.Network, n_steps: int = 96, seed: int = 0
) -> TimeseriesData:
    """Demand profiles for the urban district network."""
    rng = np.random.default_rng(seed)
    td = TimeseriesData()
    for c in net.childs_by_type(mm.PowerLoad):
        base = float(mm.value(c.model.p_mw))
        td.add_child_series(
            c.id, "p_mw", _sinusoidal_profile(n_steps, base, amplitude=0.25, rng=rng)
        )
    for c in net.childs_by_type(mm.Sink):
        if c.grid.name == "gas":
            base = float(mm.value(c.model.mass_flow_kgs))
            td.add_child_series(
                c.id,
                "mass_flow_kgs",
                _sinusoidal_profile(n_steps, base, amplitude=0.30, rng=rng),
            )
    return td


def _solve(network):
    problem = mp.create_min_load_shedding_problem(
        bounds_vm=BOUNDS_EL,
        bounds_t=BOUNDS_HEAT,
        bounds_pressure=BOUNDS_GAS,
        # legacy formulation left ext grids unbounded; replicate with non-binding wide bounds
        bounds_ext_el=(-100, 100),
        bounds_ext_gas=(-100, 100),
        bounds_ext_heat=(-100, 100),
        include_ext_grids=True,
    )
    return run_energy_flow_optimization(
        network,
        solver="gurobi",
        optimization_problem=problem,
        exclude_unconnected_nodes=True,
    )


def test_res_with_load_shedding():
    # GIVEN
    net = create_urban_district_net()
    net.apply_formulation(EL_MISOCP_FORMULATION)
    td = _make_urban_district_timeseries(net, n_steps=TIME_STEPS, seed=SEED)
    td.apply_to_network(net, 0)

    # WHEN
    result = _solve(net)

    # THEN
    assert result.success
    assert result is not None

    load_df = result.dataframes["PowerLoad"]
    assert (load_df["regulation"] >= 0).all(), "regulation must be non-negative"
    assert (load_df["regulation"] <= 1).all(), "regulation must be at most 1"

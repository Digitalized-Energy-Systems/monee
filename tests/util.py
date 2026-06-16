"""Shared test helpers; behavior-identical consolidations of duplicated per-file helpers."""

import math

import monee.express as mx
import monee.model as mm

WATER_LOOP_PIPE_D = 0.3  # m
WATER_LOOP_PIPE_L = 100.0  # m


def solver_available(name: str) -> bool:
    import pyomo.environ as pyo

    try:
        return pyo.SolverFactory(name).available(exception_flag=False)
    except Exception:
        return False


def create_g2h_net():
    """Gas (source/ext-grid/sink, two pipes) + water (supply/return) net coupled by a G2H."""
    pn = mm.Network()

    gas_grid = mm.create_gas_grid("gas", type="lgas")
    g_node_0 = pn.node(
        mm.Junction(), child_ids=[pn.child(mm.Source(mass_flow_kgs=1))], grid=gas_grid
    )
    g_node_1 = pn.node(
        mm.Junction(), child_ids=[pn.child(mm.ExtHydrGrid())], grid=gas_grid
    )
    g_node_2 = pn.node(
        mm.Junction(), child_ids=[pn.child(mm.Sink(mass_flow_kgs=1))], grid=gas_grid
    )
    # Realistic gas main: 0.1 mm steel roughness over km-scale lengths. The old
    # net used 100 m pipes with a 10 mm roughness - both unrealistic, and only
    # well-conditioned for the (8x-inflated) pre-fix Weymouth drop. With the
    # corrected coefficient a realistic roughness over km-scale lengths gives a
    # drop large enough to keep the pressure<->flow coupling well-conditioned for
    # the default IMODE=3 solve.
    pn.branch(
        mm.GasPipe(diameter_m=0.3, length_m=1000, temperature_ext_k=300, roughness_m=1e-4),
        g_node_0,
        g_node_1,
    )
    pn.branch(
        mm.GasPipe(diameter_m=0.3, length_m=1500, temperature_ext_k=300, roughness_m=1e-4),
        g_node_0,
        g_node_2,
    )

    w_node_0 = pn.node(
        mm.Junction(), grid=mm.WATER_KEY, child_ids=[pn.child(mm.Sink(mass_flow_kgs=0.1))]
    )
    w_node_1 = pn.node(
        mm.Junction(), child_ids=[pn.child(mm.ConsumeHydrGrid(1))], grid=mm.WATER_KEY
    )
    w_node_2 = pn.node(mm.Junction(), grid=mm.WATER_KEY)
    w_node_3 = pn.node(
        mm.Junction(), grid=mm.WATER_KEY, child_ids=[pn.child(mm.ExtHydrGrid(t_k=359))]
    )
    pn.branch(mm.WaterPipe(diameter_m=0.15, length_m=100), w_node_0, w_node_1)
    pn.branch(mm.WaterPipe(diameter_m=0.15, length_m=200), w_node_3, w_node_2)

    mx.create_g2h(
        pn,
        gas_node_id=g_node_2,
        heat_node_id=w_node_2,
        heat_return_node_id=w_node_1,
        heat_energy_mw=0.010,
        diameter_m=0.4,
        efficiency=0.9,
    )
    return pn


def create_water_loop(source_t_k=None):
    """3-junction loop: ext-grid (n0) -- n1 (Source 5 kg/s) -- n2 (Sink 10 kg/s) -- n0.

    ``source_t_k`` types the source's injection temperature; without it the
    source-junction temperature is structurally underdetermined (fine for
    relative comparisons, fatal for APOPT and exact temperature asserts)."""
    net = mm.Network()
    n0 = net.node(
        mm.Junction(),
        mm.WATER,
        child_ids=[net.child(mm.ExtHydrGrid(t_k=356))],
    )
    n1 = net.node(
        mm.Junction(),
        mm.WATER,
        child_ids=[net.child(mm.Source(mass_flow_kgs=5, t_k=source_t_k))],
    )
    n2 = net.node(
        mm.Junction(),
        mm.WATER,
        child_ids=[net.child(mm.Sink(mass_flow_kgs=10))],
    )
    pipe = dict(diameter_m=WATER_LOOP_PIPE_D, length_m=WATER_LOOP_PIPE_L)
    net.branch(mm.WaterPipe(**pipe), n0, n1)
    net.branch(mm.WaterPipe(**pipe), n1, n2)
    net.branch(mm.WaterPipe(**pipe), n2, n0)
    return net, n0, n1, n2


def assert_junction_nan(result, jct_id, label):
    """t_pu is a Var NaN'd by inject_nans for ignored nodes (t_k is a derived Intermediate, not NaN'd)."""
    df = result.dataframes["Junction"]
    t_pu = df.loc[df["id"] == jct_id, "t_pu"].iloc[0]
    assert math.isnan(t_pu), f"{label}: expected NaN t_pu, got {t_pu}"


def assert_junction_solved(result, jct_id, label):
    df = result.dataframes["Junction"]
    t_pu = df.loc[df["id"] == jct_id, "t_pu"].iloc[0]
    assert not math.isnan(t_pu), f"{label}: junction must be solved (got NaN)"


def assert_control_node_nan(result, model_name, label):
    """Check that the control node row has NaN t_pu (a Var on all CP control nodes)."""
    df = result.dataframes[model_name]
    t_pu = df["t_pu"].iloc[0]
    assert math.isnan(t_pu), f"{label}: {model_name}.t_pu must be NaN (got {t_pu})"


def assert_control_node_solved(result, model_name, label):
    df = result.dataframes[model_name]
    t_pu = df["t_pu"].iloc[0]
    assert not math.isnan(t_pu), f"{label}: {model_name}.t_pu must be solved (got NaN)"


def child_id_by_type(net, model_type):
    """Return the id of the first child whose model is an instance of model_type."""
    for node in net.nodes:
        for child in net.childs_by_ids(node.child_ids):
            if isinstance(child.model, model_type):
                return child.id
    raise AssertionError(f"no child of type {model_type.__name__} found")

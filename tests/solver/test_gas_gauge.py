"""Gauge-vs-absolute gas pressure convention (GasGrid.pressure_ambient_pa).

Absolute pressure in the Weymouth/density physics = node pressure + ambient.
Default ambient is 0 (node pressures are absolute, monee's historical behaviour);
STANDARD_ATMOSPHERE_PA makes them gauge (pandapipes / standard convention).
"""

import math

import pytest

import monee.model as mm
import monee.solver as ms
from monee.model.formulation import (
    GAS_CONVEX_MIQCQP_FORMULATION,
    make_gas_nlp_formulation,
)
from monee.model.grid import STANDARD_ATMOSPHERE_PA


def _gurobipy_available() -> bool:
    try:
        import gurobipy  # noqa: F401

        from monee.solver.gurobipy import GurobipySolver  # noqa: F401

        return True
    except Exception:
        return False


def _gas_pipe_net(ambient_pa, formulation=None):
    g = mm.create_gas_grid("gas", type="lgas")
    g.pressure_ambient_pa = ambient_pa
    net = mm.Network()
    net.set_default_grid(mm.GAS_KEY, g)
    net.activate_grid(mm.GAS)
    n0 = net.node(mm.Junction(), mm.GAS, child_ids=[net.child(mm.ExtHydrGrid())])
    n1 = net.node(
        mm.Junction(), mm.GAS, child_ids=[net.child(mm.Sink(mass_flow_kgs=0.4))]
    )
    net.branch(mm.GasPipe(diameter_m=0.3, length_m=8000, roughness_m=1e-4), n0, n1)
    net.apply_formulation(
        formulation or make_gas_nlp_formulation(friction_model="hybrid")
    )
    return net


def _drop_bar(res):
    p = res.dataframes["Junction"]["pressure_pu"].to_numpy(float)
    return float(p.max() - p.min())


def test_default_ambient_is_zero():
    # The default convention is absolute (no ambient offset).
    assert mm.create_gas_grid("gas", type="lgas").pressure_ambient_pa == 0.0


def test_gauge_reduces_gas_drop():
    # Gauge raises the ABSOLUTE pressure (10 -> ~11.013 bar), so the gas is denser
    # and the per-unit pressure DROP shrinks. With only the ambient changed (same
    # p_ref, same derived Z), the drop scales by the (p_i+p_j) absolute-sum ratio
    # ~ 20 / 22.0265 = 0.908.
    drop_abs = _drop_bar(ms.GEKKOSolver().solve(_gas_pipe_net(0.0)))
    drop_gauge = _drop_bar(ms.GEKKOSolver().solve(_gas_pipe_net(STANDARD_ATMOSPHERE_PA)))
    assert drop_gauge < drop_abs
    assert math.isclose(drop_gauge / drop_abs, 0.908, abs_tol=0.02)


# The convex MIQCQP uses constant (fully-rough) friction, so compare it against
# the NLP on the SAME friction model to isolate the gauge linearization.
def _nlp_constant(ambient_pa):
    return _gas_pipe_net(ambient_pa, formulation=make_gas_nlp_formulation(friction_model="constant"))


@pytest.mark.skipif(not _gurobipy_available(), reason="gurobipy not available")
def test_gauge_convex_miqcqp_matches_nlp():
    # Workaround B: the convex MIQCQP expresses gauge via the LINEARIZED pressure
    # offset (affine in pressure_squared_pu), so it stays conic AND reproduces the
    # exact-sqrt NLP gauge result to well within 1%.
    from monee.solver.gurobipy import GurobipySolver

    nlp = ms.GEKKOSolver().solve(_nlp_constant(STANDARD_ATMOSPHERE_PA))
    miqcqp = GurobipySolver().solve(
        _gas_pipe_net(STANDARD_ATMOSPHERE_PA, formulation=GAS_CONVEX_MIQCQP_FORMULATION)
    )
    assert miqcqp.success
    assert math.isclose(_drop_bar(miqcqp), _drop_bar(nlp), rel_tol=0.01)


@pytest.mark.skipif(not _gurobipy_available(), reason="gurobipy not available")
def test_gauge_off_convex_miqcqp_unchanged_vs_nlp():
    # Sanity: with gauge OFF (default), the convex MIQCQP and the NLP agree, i.e.
    # the gauge plumbing is a pure no-op at the default (on par with NLP default).
    from monee.solver.gurobipy import GurobipySolver

    nlp = ms.GEKKOSolver().solve(_nlp_constant(0.0))
    miqcqp = GurobipySolver().solve(
        _gas_pipe_net(0.0, formulation=GAS_CONVEX_MIQCQP_FORMULATION)
    )
    assert miqcqp.success
    assert math.isclose(_drop_bar(miqcqp), _drop_bar(nlp), rel_tol=0.01)

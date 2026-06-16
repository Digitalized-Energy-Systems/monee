"""Equivalence test for the combined AC flow helper.

``ac.int_flows`` is a common-subexpression-shared rewrite of the four
``int_flow_from_p`` / ``int_flow_from_q`` / ``int_flow_to_p`` / ``int_flow_to_q``
functions (it builds ``vm_from*vm_to``, the angle-difference sin/cos and the
``vm**2`` terms once and reuses them, with the to-direction relying on
``cos(-x)=cos(x)`` / ``sin(-x)=-sin(x)``). It must produce numerically identical
flow equations; this test pins that down over randomized inputs.
"""

import math
import random

import monee.model.phys.nonlinear.ac as ac


class _Capture:
    """Stand-in for the flow decision variable: ``_Capture() == rhs`` returns the
    right-hand-side expression, so we can compare the equations' RHS as plain
    floats instead of boolean ``==`` results."""

    def __eq__(self, rhs):
        return rhs


def _cases():
    rng = random.Random(20240614)
    for _ in range(400):
        r = rng.uniform(0.005, 0.5)
        x = rng.uniform(0.005, 0.8)
        y = 1.0 / complex(r, x)
        yield dict(
            vm_from_pu=rng.uniform(0.9, 1.1),
            vm_to_pu=rng.uniform(0.9, 1.1),
            va_from_rad=rng.uniform(-0.5, 0.5),
            va_to_rad=rng.uniform(-0.5, 0.5),
            g_branch=y.real,
            b_branch=y.imag,
            tap=rng.uniform(0.9, 1.1),
            shift=rng.uniform(-0.3, 0.3),
            g_from=rng.uniform(0.0, 0.05),
            b_from=rng.uniform(-0.05, 0.05),
            g_to_pu=rng.uniform(0.0, 0.05),
            b_to_pu=rng.uniform(-0.05, 0.05),
            on_off=rng.choice([1, 0, 0.5]),
        )


def test_int_flows_matches_individual_functions():
    for c in _cases():
        common = dict(
            vm_from_pu=c["vm_from_pu"],
            vm_to_pu=c["vm_to_pu"],
            va_from_rad=c["va_from_rad"],
            va_to_rad=c["va_to_rad"],
            g_branch=c["g_branch"],
            b_branch=c["b_branch"],
            tap=c["tap"],
            shift=c["shift"],
            on_off=c["on_off"],
        )
        ref_pf = ac.int_flow_from_p(_Capture(), g_from=c["g_from"], **common)
        ref_qf = ac.int_flow_from_q(_Capture(), b_from=c["b_from"], **common)
        ref_pt = ac.int_flow_to_p(_Capture(), g_to_pu=c["g_to_pu"], **common)
        ref_qt = ac.int_flow_to_q(_Capture(), b_to_pu=c["b_to_pu"], **common)

        combined = ac.int_flows(
            _Capture(), _Capture(), _Capture(), _Capture(),
            g_from=c["g_from"], b_from=c["b_from"],
            g_to_pu=c["g_to_pu"], b_to_pu=c["b_to_pu"], **common,
        )

        for got, ref, label in zip(
            combined, (ref_pf, ref_qf, ref_pt, ref_qt), ("p_from", "q_from", "p_to", "q_to")
        ):
            assert math.isclose(got, ref, rel_tol=1e-12, abs_tol=1e-12), (
                f"{label}: combined={got!r} individual={ref!r} for {c!r}"
            )

"""Polar AC power flow: a smooth non-convex NLP (sin/cos in the flow
equations). The exact model - no relaxation involved."""

import math

import numpy as np

import monee.model.phys.nonlinear.ac as opfmodel
from monee.model.core import Intermediate, IntermediateEq, PostProcess

from ..core import BranchFormulation, ChildFormulation, NodeFormulation

SQRT_3 = np.sqrt(3)

# Smoothing scale [MW] for the current-magnitude sqrt - same idiom as
# smooth_abs in model.phys.nonlinear.smooth. Without it, \sqrt{p^2+q^2} has a
# singular Jacobian at exactly zero flow, and the min=0 bounds on i_*_ka
# pin the solver onto that point (e.g. a storage at zero dispatch).
CURRENT_SMOOTHING_EPS_MW = 1e-4


class AcPolarNlpNodeFormulation(NodeFormulation):
    def ensure_var(self, model, simulation=False, grid=None):
        # Some multi-grid control nodes subclass Bus (so they match here) without
        # a vm_pu attribute - only act on real voltage buses.
        if simulation and hasattr(model, "vm_pu"):
            model.vm_pu_squared = PostProcess(lambda v: v.vm_pu**2)


class AcPolarNlpShuntFormulation(ChildFormulation):
    """Voltage-dependent bus shunt for the polar NLP: ``p/q`` scale with ``vm_pu**2``."""

    def equations(self, child, grid, node, **kwargs):
        vm_sq = node.vars["vm_pu"] ** 2
        return [
            child.p_mw == child.gs_mw * vm_sq,
            child.q_mvar == -child.bs_mvar * vm_sq,
        ]


class AcPolarNlpBranchFormulation(BranchFormulation):
    def ensure_var(self, branch, simulation=False, grid=None):
        branch.i_from_ka = Intermediate(0)
        branch.i_to_ka = Intermediate(0)
        branch.loading_from_pu = Intermediate(0)
        branch.loading_to_pu = Intermediate(0)

    def equations(self, branch, grid, from_node_model, to_node_model, **kwargs):
        y = np.linalg.pinv([[branch.br_r_pu + branch.br_x_pu * 1j]])[0][0]
        g, b = (np.real(y), np.imag(y))

        i_from_ka = (
            (branch.p_from_mw**2 + branch.q_from_mvar**2 + CURRENT_SMOOTHING_EPS_MW**2)
            ** 0.5
            / (from_node_model.vars["vm_pu"] * from_node_model.vars["base_kv"])
            / SQRT_3
        )
        i_to_ka = (
            (branch.p_to_mw**2 + branch.q_to_mvar**2 + CURRENT_SMOOTHING_EPS_MW**2)
            ** 0.5
            / (to_node_model.vars["vm_pu"] * to_node_model.vars["base_kv"])
            / SQRT_3
        )

        return [
            # All four P/Q flow equations, sharing the sub-terms common across
            # them (vm_from*vm_to, the angle-difference sin/cos, vm**2) so the
            # symbolic graph they contribute is ~halved (see ac.int_flows).
            *opfmodel.int_flows(
                p_from_var=branch.p_from_mw,
                q_from_var=branch.q_from_mvar,
                p_to_var=branch.p_to_mw,
                q_to_var=branch.q_to_mvar,
                vm_from_pu=from_node_model.vars["vm_pu"],
                vm_to_pu=to_node_model.vars["vm_pu"],
                va_from_rad=from_node_model.vars["va_radians"],
                va_to_rad=to_node_model.vars["va_radians"],
                g_branch=g,
                b_branch=b,
                tap=branch.tap,
                shift=branch.shift,
                cos_impl=kwargs["cos_impl"] if "cos_impl" in kwargs else math.cos,
                sin_impl=kwargs["sin_impl"] if "sin_impl" in kwargs else math.sin,
                g_from=branch.g_fr_pu,
                b_from=branch.b_fr_pu,
                g_to_pu=branch.g_to_pu,
                b_to_pu=branch.b_to_pu,
                on_off=branch.on_off,
            ),
            # Report-only intermediates: inline into a line-loading limit when one
            # references them, otherwise evaluated post-solve for results only.
            IntermediateEq("i_from_ka", i_from_ka),
            IntermediateEq("i_to_ka", i_to_ka),
            IntermediateEq("loading_from_pu", i_from_ka / branch.max_i_ka),
            IntermediateEq("loading_to_pu", i_to_ka / branch.max_i_ka),
        ]

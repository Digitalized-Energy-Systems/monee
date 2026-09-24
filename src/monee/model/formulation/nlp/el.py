"""Polar AC power flow: a smooth non-convex NLP (sin/cos in the flow
equations). The exact model - no relaxation involved."""

import math

import numpy as np

import monee.model.phys.nonlinear.ac as opfmodel
from monee.model.core import Intermediate, IntermediateEq, PostProcess, Var
from monee.model.grid import PowerGrid

from ..core import BranchFormulation, ChildFormulation, NodeFormulation

SQRT_3 = np.sqrt(3)

CURRENT_SMOOTHING_EPS_MW = 1e-4


class AcPolarNlpNodeFormulation(NodeFormulation):
    def ensure_var(self, model, simulation=False, grid=None):
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

        sn = grid.sn_mva if isinstance(grid, PowerGrid) else 1.0
        if sn and not math.isclose(sn, 1.0):
            for key in ("p_from_mw", "q_from_mvar", "p_to_mw", "q_to_mvar"):
                var = getattr(branch, key, None)
                if isinstance(var, Var):
                    var.scale = sn

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
                s_base=grid.sn_mva,
            ),
            IntermediateEq("i_from_ka", i_from_ka),
            IntermediateEq("i_to_ka", i_to_ka),
            IntermediateEq("loading_from_pu", i_from_ka / branch.max_i_ka),
            # max_i_ka is expressed in the from-side voltage basis
            # (io/matpower.py: rate_a/(sqrt3*V_from)); the to-side rated
            # current is max_i_ka*V_from/V_to. Dividing i_to_ka by the raw
            # max_i_ka inflated a transformer's loading_to_pu by the voltage
            # ratio; for a line (equal base_kv) nothing changes.
            IntermediateEq(
                "loading_to_pu",
                i_to_ka
                * to_node_model.vars["base_kv"]
                / (from_node_model.vars["base_kv"] * branch.max_i_ka),
            ),
        ]

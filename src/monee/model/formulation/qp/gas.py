import math

import numpy as np

import monee.model.phys.nonlinear.ac as opfmodel

from ..core import BranchFormulation, NodeFormulation

SQRT_3 = np.sqrt(3)


class ACElectricityNodeFormulation(NodeFormulation):
    pass


class ACElectricityBranchFormulation(BranchFormulation):
    def equations(self, branch, grid, from_node_model, to_node_model, **kwargs):
        y = np.linalg.pinv([[branch.br_r + branch.br_x * 1j]])[0][0]
        g, b = (np.real(y), np.imag(y))

        return [
            opfmodel.int_flow_from_p(
                p_from_var=branch.p_from_mw,
                vm_from_var=from_node_model.vars["vm_pu"],
                vm_to_var=to_node_model.vars["vm_pu"],
                va_from_var=from_node_model.vars["va_radians"],
                va_to_var=to_node_model.vars["va_radians"],
                g_branch=g,
                b_branch=b,
                tap=branch.tap,
                shift=branch.shift,
                cos_impl=kwargs["cos_impl"] if "cos_impl" in kwargs else math.cos,
                sin_impl=kwargs["sin_impl"] if "sin_impl" in kwargs else math.sin,
                g_from=branch.g_fr,
                on_off=branch.on_off,
            ),
            opfmodel.int_flow_from_q(
                q_from_var=branch.q_from_mvar,
                vm_from_var=from_node_model.vars["vm_pu"],
                vm_to_var=to_node_model.vars["vm_pu"],
                va_from_var=from_node_model.vars["va_radians"],
                va_to_var=to_node_model.vars["va_radians"],
                g_branch=g,
                b_branch=b,
                tap=branch.tap,
                shift=branch.shift,
                cos_impl=kwargs["cos_impl"] if "cos_impl" in kwargs else math.cos,
                sin_impl=kwargs["sin_impl"] if "sin_impl" in kwargs else math.sin,
                b_from=branch.b_fr,
                on_off=branch.on_off,
            ),
            opfmodel.int_flow_to_p(
                p_to_var=branch.p_to_mw,
                vm_from_var=from_node_model.vars["vm_pu"],
                vm_to_var=to_node_model.vars["vm_pu"],
                va_from_var=from_node_model.vars["va_radians"],
                va_to_var=to_node_model.vars["va_radians"],
                g_branch=g,
                b_branch=b,
                tap=branch.tap,
                shift=branch.shift,
                cos_impl=kwargs["cos_impl"] if "cos_impl" in kwargs else math.cos,
                sin_impl=kwargs["sin_impl"] if "sin_impl" in kwargs else math.sin,
                g_to=branch.g_to,
                on_off=branch.on_off,
            ),
            opfmodel.int_flow_to_q(
                q_to_var=branch.q_to_mvar,
                vm_from_var=from_node_model.vars["vm_pu"],
                vm_to_var=to_node_model.vars["vm_pu"],
                va_from_var=from_node_model.vars["va_radians"],
                va_to_var=to_node_model.vars["va_radians"],
                g_branch=g,
                b_branch=b,
                tap=branch.tap,
                shift=branch.shift,
                cos_impl=kwargs["cos_impl"] if "cos_impl" in kwargs else math.cos,
                sin_impl=kwargs["sin_impl"] if "sin_impl" in kwargs else math.sin,
                b_to=branch.b_to,
                on_off=branch.on_off,
            ),
            branch.i_from_ka
            == (branch.p_from_mw**2 + branch.q_from_mvar**2)
            / (from_node_model.vars["vm_pu"] * from_node_model.vars["base_kv"])
            / SQRT_3,
            branch.i_to_ka
            == (branch.p_to_mw**2 + branch.q_to_mvar**2)
            / (to_node_model.vars["vm_pu"] * to_node_model.vars["base_kv"])
            / SQRT_3,
        ]

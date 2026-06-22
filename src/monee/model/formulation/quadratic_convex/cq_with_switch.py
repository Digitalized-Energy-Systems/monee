import math

import numpy as np

import monee.model.phys.quadratic_convex.cq_with_switch as opfmodel

from monee.model.core import Var
from ..core import BranchFormulation, NodeFormulation

SQRT_3 = np.sqrt(3)


class QCElectricityNodeFormulation(NodeFormulation):
    def equations(
        self,
        node,
        grid,
        from_branch_models,
        to_branch_models,
        connected_node_models,
        **kwargs,
    ):
        v_min = getattr(node, "min_vm_pu", getattr(node, "v_min", 0.8))
        v_max = getattr(node, "max_vm_pu", getattr(node, "v_max", 1.2))

        return opfmodel.square_relax(
            v_sq_var=node.vm_pu_squared,
            v_var=node.vars["vm_pu"],
            v_min=v_min,
            v_max=v_max,
        )


class QCElectricityBranchFormulation(BranchFormulation):
    def ensure_var(self, branch, simulation=False, grid=None, **kwargs):
        theta_u = getattr(branch, "angmax", getattr(branch, "delta_max", math.pi / 6))
        theta_M = getattr(branch, "big_m_theta", math.pi)

        branch.va_diff = Var(0, min=-theta_M, max=theta_M)
        branch.cs = Var(1, min=0, max=1)
        branch.s = Var(0, min=-1, max=1)

        branch.vv = Var(1, min=0)
        branch.wc = Var(1)
        branch.ws = Var(0)

        branch.i_qc = Var(0, min=0)

        branch.theta_u = theta_u
        branch.theta_M = theta_M

    def equations(self, branch, grid, from_node_model, to_node_model, **kwargs):
        #y = np.linalg.pinv([[branch.br_r + branch.br_x * 1j]])[0][0]
        y = np.linalg.pinv([[branch.br_r_pu + branch.br_x_pu * 1j]])[0][0]
        g, b = (np.real(y), np.imag(y))

        v_from_min = getattr(from_node_model, "min_vm_pu", getattr(from_node_model, "v_min", 0.8))
        v_from_max = getattr(from_node_model, "max_vm_pu", getattr(from_node_model, "v_max", 1.2))
        v_to_min = getattr(to_node_model, "min_vm_pu", getattr(to_node_model, "v_min", 0.8))
        v_to_max = getattr(to_node_model, "max_vm_pu", getattr(to_node_model, "v_max", 1.2))

        vv_min = v_from_min * v_to_min
        vv_max = v_from_max * v_to_max

        eqs = [
            branch.va_diff
            == from_node_model.vars["va_radians"] - to_node_model.vars["va_radians"],
        ]

        eqs += opfmodel.cosine_relax(
            cs_var=branch.cs,
            delta_var=branch.va_diff,
            delta_max=branch.theta_u,
            on_off=branch.on_off,
            delta_big_m=branch.theta_M,
        )

        eqs += opfmodel.sine_relax(
            s_var=branch.s,
            delta_var=branch.va_diff,
            delta_max=branch.theta_u,
            on_off=branch.on_off,
            delta_big_m=branch.theta_M,
        )

        eqs += opfmodel.mccormick_relax(
            product_var=branch.vv,
            x_var=from_node_model.vars["vm_pu"],
            y_var=to_node_model.vars["vm_pu"],
            x_lb=v_from_min,
            x_ub=v_from_max,
            y_lb=v_to_min,
            y_ub=v_to_max,
        )

        eqs += opfmodel.mccormick_relax(
            product_var=branch.wc,
            x_var=branch.vv,
            y_var=branch.cs,
            x_lb=vv_min,
            x_ub=vv_max,
            y_lb=0.0,
            y_ub=1.0,
        )

        eqs += opfmodel.mccormick_relax(
            product_var=branch.ws,
            x_var=branch.vv,
            y_var=branch.s,
            x_lb=vv_min,
            x_ub=vv_max,
            y_lb=-1.0,
            y_ub=1.0,
        )

        eqs += [
            opfmodel.int_flow_from_p(
                p_from_var=branch.p_from_mw,
                v_sq_from_var=from_node_model.vars["vm_pu_squared"],
                wc_var=branch.wc,
                ws_var=branch.ws,
                g_branch=g,
                b_branch=b,
                tap=branch.tap,
                shift=branch.shift,
                g_from=branch.g_fr_pu,
                on_off=branch.on_off,
            ),
            opfmodel.int_flow_from_q(
                q_from_var=branch.q_from_mvar,
                v_sq_from_var=from_node_model.vars["vm_pu_squared"],
                wc_var=branch.wc,
                ws_var=branch.ws,
                g_branch=g,
                b_branch=b,
                tap=branch.tap,
                shift=branch.shift,
                b_from=branch.b_fr_pu,
                on_off=branch.on_off,
            ),
            opfmodel.int_flow_to_p(
                p_to_var=branch.p_to_mw,
                v_sq_to_var=to_node_model.vars["vm_pu_squared"],
                wc_var=branch.wc,
                ws_var=branch.ws,
                g_branch=g,
                b_branch=b,
                tap=branch.tap,
                shift=branch.shift,
                g_to=branch.g_fr_pu, #todo: ist das wirklich so richtig? in branch attributen aber kein to wert aktuell?
                on_off=branch.on_off,
            ),
            opfmodel.int_flow_to_q(
                q_to_var=branch.q_to_mvar,
                v_sq_to_var=to_node_model.vars["vm_pu_squared"],
                wc_var=branch.wc,
                ws_var=branch.ws,
                g_branch=g,
                b_branch=b,
                tap=branch.tap,
                shift=branch.shift,
                b_to=branch.b_fr_pu,#todo ist das wirklich richtig?
                on_off=branch.on_off,
            ),
            opfmodel.current_flow_equation(
                i_var=branch.i_qc,
                v_sq_from_var=from_node_model.vars["vm_pu_squared"],
                v_sq_to_var=to_node_model.vars["vm_pu_squared"],
                wc_var=branch.wc,
                g_branch=g,
                b_branch=b,
            ),
            opfmodel.current_soc_relax(
                p_var=branch.p_from_mw,
                q_var=branch.q_from_mvar,
                v_sq_var=from_node_model.vars["vm_pu_squared"],
                i_var=branch.i_qc,
                v_sq_ub=v_from_max**2,
                on_off=branch.on_off,
            ),
            branch.i_from_ka
            >= kwargs["sqrt_impl"](branch.p_from_mw**2 + branch.q_from_mvar**2)
            / (from_node_model.vars["vm_pu"] * from_node_model.vars["base_kv"] * SQRT_3),
            branch.i_to_ka
            >= kwargs["sqrt_impl"](branch.p_to_mw**2 + branch.q_to_mvar**2)
            / (to_node_model.vars["vm_pu"] * to_node_model.vars["base_kv"] * SQRT_3),
        ]

        return eqs
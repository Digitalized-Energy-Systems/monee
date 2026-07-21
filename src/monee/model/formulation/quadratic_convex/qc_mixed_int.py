import monee.model.phys.quadratic_convex.qc_mixed_int as opfmodel
from monee.model.core import Intermediate, IntermediateEq, Var
from monee.model.formulation.core import BranchFormulation, NodeFormulation
import numpy as np
import math

def _get_angle_limit_rad(branch, grid=None):
    """
    Return the angle interval used by the QC trigonometric envelopes.

    Priority:
        1. Optional branch-specific value.
        2. Optional PowerGrid value.
        3. One formulation-wide default.
    """
    theta_max = getattr(
        branch,
        "max_angle_difference_rad",
        None,
    )

    if theta_max is None and grid is not None:
        theta_max = getattr(
            grid,
            "qc_angle_max_rad",
            None,
        )

    if theta_max is None:
        theta_max = math.pi / 2.0 #setting it to value from paper (they assumed it's much lower anyways but said until here it's assumed to be ok)

    theta_max = float(theta_max)

    opfmodel.validate_angle_limit(theta_max)

    return theta_max

class MIQCNodeFormulation(NodeFormulation):
    def ensure_var(self, model, simulation=False, grid=None):
        if not hasattr(model, "vm_pu"):
            return

        # At this stage, vm_pu is still the monee Var and has min/max.
        vm_min = float(model.vm_pu.min)
        vm_max = float(model.vm_pu.max)

        # Save plain numbers because model.vm_pu later becomes a GEKKO variable.
        model._qc_vm_min = vm_min
        model._qc_vm_max = vm_max

        w_min, w_max = opfmodel.voltage_square_bounds(
            vm_min,
            vm_max,
        )

        model.vm_pu_squared = Var(
            model.vm_pu.value**2,
            min=w_min,
            max=w_max,
        )

    def equations(
        self, model, grid, in_branch_models, out_branch_models,  childs, **kwargs,):
        if not hasattr(model, "vm_pu"):
            return []

        return opfmodel.square_relaxation(
            x_square=model.vars["vm_pu_squared"],
            x=model.vars["vm_pu"],
            x_min=model._qc_vm_min,
            x_max=model._qc_vm_max,
        )
        # Paper Eq. (16): convex envelope of vm_pu_squared = vm_pu**2.
class MIQCBranchFormulation(BranchFormulation):
    def ensure_var(self, branch, simulation=False, grid=None):
        theta_max = _get_angle_limit_rad(branch, grid)

        # Derive all auxiliary-variable bounds in the physical QC module.
        bounds = opfmodel.branch_variable_bounds(
            vm_from_min=grid.vm_pu_min,
            vm_from_max=grid.vm_pu_max,
            vm_to_min=grid.vm_pu_min,
            vm_to_max=grid.vm_pu_max,
            theta_max=theta_max,
        )

        # theta_ij = theta_i - theta_j
        branch.angle_difference_rad = Var(
            0.0,
            min=bounds["angle_min"],
            max=bounds["angle_max"],
        )

        # Lifted approximation of cos(theta_ij).
        branch.cos_angle_difference = Var(
            1.0,
            min=bounds["cos_min"],
            max=bounds["cos_max"],
        )

        # Lifted approximation of sin(theta_ij).
        branch.sin_angle_difference = Var(
            0.0,
            min=bounds["sin_min"],
            max=bounds["sin_max"],
        )

        # vv_ij approximates v_i * v_j.
        branch.vm_product_pu_squared = Var(
            1.0,
            min=bounds["vv_min"],
            max=bounds["vv_max"],
        )

        # wc_ij approximates vv_ij * cos(theta_ij).
        branch.w_cos_pu_squared = Var(
            1.0,
            min=bounds["wc_min"],
            max=bounds["wc_max"],
        )

        # ws_ij approximates vv_ij * sin(theta_ij).
        branch.w_sin_pu_squared = Var(
            0.0,
            min=bounds["ws_min"],
            max=bounds["ws_max"],
        )

        # Squared terminal currents required by strengthened QC
        # Equations (21) and (23).
        branch.current_from_pu_squared = Var(
            0.0,
            min=0.0,
        )

        branch.current_to_pu_squared = Var(
            0.0,
            min=0.0,
        )

        # Report-only quantities, matching the AC formulation.
        branch.i_from_ka = Intermediate(0)
        branch.i_to_ka = Intermediate(0)
        branch.loading_from_pu = Intermediate(0)
        branch.loading_to_pu = Intermediate(0)
        
    def equations(
            self,
            branch,
            grid,
            from_node_model,
            to_node_model,
            **kwargs,
    ):
        theta_max = _get_angle_limit_rad(branch, grid)

        # Convert the branch series impedance into its series admittance.
        y = np.linalg.pinv(
            [[branch.br_r_pu + branch.br_x_pu * 1j]]
        )[0][0]

        g = float(np.real(y))
        b = float(np.imag(y))

        return opfmodel.branch_equations(
            vm_from_pu=from_node_model.vars["vm_pu"],
            vm_to_pu=to_node_model.vars["vm_pu"],
            vm_from_pu_squared=from_node_model.vars[
                "vm_pu_squared"
            ],
            vm_to_pu_squared=to_node_model.vars[
                "vm_pu_squared"
            ],
            va_from_rad=from_node_model.vars["va_radians"],
            va_to_rad=to_node_model.vars["va_radians"],

            angle_difference=branch.angle_difference_rad,
            cos_angle_difference=branch.cos_angle_difference,
            sin_angle_difference=branch.sin_angle_difference,

            vm_product_pu_squared=branch.vm_product_pu_squared,
            w_cos_pu_squared=branch.w_cos_pu_squared,
            w_sin_pu_squared=branch.w_sin_pu_squared,

            current_from_pu_squared=(
                branch.current_from_pu_squared
            ),
            current_to_pu_squared=(
                branch.current_to_pu_squared
            ),

            vm_from_min=from_node_model._qc_vm_min,
            vm_from_max=from_node_model._qc_vm_max,
            vm_to_min=to_node_model._qc_vm_min,
            vm_to_max=to_node_model._qc_vm_max,

            theta_max=theta_max,

            p_from_var=branch.p_from_mw,
            q_from_var=branch.q_from_mvar,
            p_to_var=branch.p_to_mw,
            q_to_var=branch.q_to_mvar,

            g_branch=g,
            b_branch=b,

            tap=branch.tap,
            shift=branch.shift,
        )
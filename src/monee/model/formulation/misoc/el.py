import math

from monee.model.core import Intermediate, IntermediateEq, Var
from monee.model.formulation.core import BranchFormulation, NodeFormulation
from monee.model.phys.misoc.pf import (
    active_power_loss,
    reactive_power_loss,
    soc_rel,
    voltage_drop,
)

SQRT_3 = math.sqrt(3.0)


class MISOCPElectricityNodeFormulation(NodeFormulation):
    def ensure_var(self, node):
        node.vm_pu_squared = Var(1, min=0, max=2.25)
        node.vm_pu = Intermediate(1)

    def equations(
        self,
        node,
        grid,
        from_branch_models,
        to_branch_models,
        connected_node_models,
        **kwargs,
    ):
        return [
            IntermediateEq("vm_pu", kwargs["sqrt_impl"](node.vm_pu_squared)),
        ]


def _branch_tap(branch) -> float:
    """Off-nominal turns ratio for a power branch (1.0 if absent or zero)."""
    tap = getattr(branch, "tap", 1.0) or 1.0
    return float(tap)


def _ell_physics_max(branch, w_max: float) -> float:
    """Upper bound on per-unit squared current derived from voltage bounds alone.

    With an ideal a:1 transformer in series with Z, the from-side voltage
    seen by the impedance is V_i' = V_i / a.  From |I_ij| = |V_i' - V_j| / |Z|
    and |V_i'|, |V_j| <= sqrt(W_max):
        ell_ij <= (2*sqrt(W_max))^2 / |Z|^2 = 4*W_max / (r^2 + x^2).

    The tap drops out because both endpoints are bounded by sqrt(W_max) on
    their own per-unit base, so the tap-adjusted from-side voltage is also
    bounded by sqrt(W_max) (the branch tap is normalised relative to the
    base ratio).
    """
    return 4 * w_max / (branch.br_r**2 + branch.br_x**2)


def _big_m(w_max: float) -> float:
    """Compute a tight big-M bound from the voltage bound alone.

    Substituting the physics-based current bound ell_max = 4*W_max/|Z|^2 into
    the Cauchy-Schwarz result M = (sqrt(W_max) + |Z|*sqrt(ell_max))^2, the
    impedance cancels and M = 9*W_max, independent of branch impedance and tap.
    """
    return 9 * w_max


class MISOCPElectricityBranchFormulation(BranchFormulation):
    def ensure_var(self, branch):
        branch.current_pu = Var(1, min=0)
        # i_*_ka and loading_*_percent are not free decision variables under
        # MISOCP — they are algebraically determined by ``current_pu`` (the
        # per-unit |I|² SOC variable) and the bus voltage base.  Replace the
        # base-model ``Var`` declarations with ``Intermediate`` so:
        #   * the LP/MIP doesn't carry four unconstrained slack vars per
        #     branch (a real correctness bug on top of cluttering the model);
        #   * the post-solve sqrt-based conversion stays a Pyomo Expression
        #     (sqrt is nonlinear; Gurobi's LP/SOC writer would reject it as
        #     a hard constraint, but post-solve evaluation is fine).
        branch.i_from_ka = Intermediate(0)
        branch.i_to_ka = Intermediate(0)
        branch.loading_from_percent = Intermediate(0)
        branch.loading_to_percent = Intermediate(0)

    def minimize(self, branch, grid, from_node_model, to_node_model, **kwargs):
        return [branch.current_pu * branch.br_r]

    def equations(self, branch, grid, from_node_model, to_node_model, **kwargs):
        w_max = grid.vm_pu_max**2
        big_m = _big_m(w_max)
        ell_phys = _ell_physics_max(branch, w_max)
        tap = _branch_tap(branch)
        sqrt_impl = kwargs["sqrt_impl"]
        # Per-unit current base on each side:  I_base_ka = S_base / (√3 · V_base)
        # For trafos the from-side primary current is the secondary current
        # divided by the tap ratio (ideal a:1 transformer).  Lines have
        # tap = 1.0 so both bases collapse to the same value.
        I_base_from = grid.sn_mva / (SQRT_3 * from_node_model.base_kv) / tap
        I_base_to = grid.sn_mva / (SQRT_3 * to_node_model.base_kv)
        i_mag_pu = sqrt_impl(branch.current_pu)
        return [
            branch.current_pu <= ell_phys * branch.on_off,
            voltage_drop(
                from_node_model.vars["vm_pu_squared"],
                to_node_model.vars["vm_pu_squared"],
                branch.vars["p_from_mw"] / grid.sn_mva,
                branch.vars["q_from_mvar"] / grid.sn_mva,
                branch.current_pu,
                branch.br_r,
                branch.br_x,
                tap=tap,
            )
            <= big_m * (1 - branch.on_off),
            voltage_drop(
                from_node_model.vars["vm_pu_squared"],
                to_node_model.vars["vm_pu_squared"],
                branch.vars["p_from_mw"] / grid.sn_mva,
                branch.vars["q_from_mvar"] / grid.sn_mva,
                branch.current_pu,
                branch.br_r,
                branch.br_x,
                tap=tap,
            )
            >= -big_m * (1 - branch.on_off),
            soc_rel(
                from_node_model.vars["vm_pu_squared"],
                branch.vars["p_from_mw"] / grid.sn_mva,
                branch.vars["q_from_mvar"] / grid.sn_mva,
                branch.current_pu,
                tap=tap,
            ),
            active_power_loss(
                branch.vars["p_from_mw"] / grid.sn_mva,
                branch.vars["p_to_mw"] / grid.sn_mva,
                branch.current_pu,
                branch.br_r,
            ),
            reactive_power_loss(
                branch.vars["q_from_mvar"] / grid.sn_mva,
                branch.vars["q_to_mvar"] / grid.sn_mva,
                branch.current_pu,
                branch.br_x,
            ),
            # |I_ka| = √current_pu · I_base_ka.  Same √current_pu on both
            # sides — under the SOC relaxation ``current_pu`` represents the
            # from-side magnitude; lossy branches see a slightly different
            # to-side magnitude (off by r·ell, x·ell), but for diagnostic
            # loading the from-side value is the conventional report.
            IntermediateEq("i_from_ka", i_mag_pu * I_base_from),
            IntermediateEq("i_to_ka", i_mag_pu * I_base_to),
            IntermediateEq(
                "loading_from_percent", i_mag_pu * I_base_from / branch.max_i_ka
            ),
            IntermediateEq(
                "loading_to_percent", i_mag_pu * I_base_to / branch.max_i_ka
            ),
        ]

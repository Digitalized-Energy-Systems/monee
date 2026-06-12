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
    def ensure_var(self, node, simulation=False):
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
    """Off-nominal turns ratio (1.0 if absent or zero)."""
    tap = getattr(branch, "tap", 1.0) or 1.0
    return float(tap)


def _ell_physics_max(branch, w_max: float) -> float:
    """Per-unit |I|² upper bound from voltage bounds: ``4·W_max / |Z|²``."""
    return 4 * w_max / (branch.br_r**2 + branch.br_x**2)


# Headroom factor on the thermal rating for the current_pu bound. The bound is
# a big-M tightening device, not the operational loading constraint (that one
# is line_loading_limit()); 3x rating is far above any acceptable steady-state
# loading while keeping the bound finite on near-zero-impedance lines, where
# the voltage-derived 4*W_max/|Z|^2 explodes (~1e9) and wrecks the matrix
# conditioning badly enough that SCIP/Gurobi spuriously prove infeasibility.
_ELL_THERMAL_HEADROOM = 3.0


def _ell_max(branch, w_max: float, i_base_from: float, i_base_to: float) -> float:
    """Tightest available per-unit |I|² bound: voltage-derived, capped by the
    thermal rating (with headroom) when ``max_i_ka`` is available."""
    ell = _ell_physics_max(branch, w_max)
    max_i_ka = getattr(branch, "max_i_ka", None)
    if max_i_ka and max_i_ka > 0:
        i_base = min(i_base_from, i_base_to)
        ell_thermal = (_ELL_THERMAL_HEADROOM * max_i_ka / i_base) ** 2
        ell = min(ell, ell_thermal)
    return ell


def _big_m(w_max: float) -> float:
    """Cauchy-Schwarz big-M; with ``ell_max = 4·W_max/|Z|²`` this collapses to 9·W_max."""
    return 9 * w_max


class MISOCPElectricityBranchFormulation(BranchFormulation):
    def ensure_var(self, branch, simulation=False):
        branch.current_pu = Var(1, min=0)
        branch.i_from_ka = Intermediate(0)
        branch.i_to_ka = Intermediate(0)
        branch.loading_from_percent = Intermediate(0)
        branch.loading_to_percent = Intermediate(0)

    def minimize(self, branch, grid, from_node_model, to_node_model, **kwargs):
        return [branch.current_pu * branch.br_r]

    def equations(self, branch, grid, from_node_model, to_node_model, **kwargs):
        w_max = grid.vm_pu_max**2
        big_m = _big_m(w_max)
        tap = _branch_tap(branch)
        sqrt_impl = kwargs["sqrt_impl"]
        # I_base_ka = S_base / (√3 · V_base); trafo primary divides by tap.
        I_base_from = grid.sn_mva / (SQRT_3 * from_node_model.base_kv) / tap
        I_base_to = grid.sn_mva / (SQRT_3 * to_node_model.base_kv)
        ell_phys = _ell_max(branch, w_max, I_base_from, I_base_to)
        i_mag_pu = sqrt_impl(branch.current_pu)
        # loading² = current_pu · (I_base/max_i_ka)² is linear in current_pu.
        # Used by line_loading_limit() instead of the sqrt-bearing form.
        branch._misocp_loading_from_scale_squared = (I_base_from / branch.max_i_ka) ** 2
        branch._misocp_loading_to_scale_squared = (I_base_to / branch.max_i_ka) ** 2
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
            # |I_ka| = √current_pu · I_base. current_pu is the from-side magnitude;
            # the to-side report is approximate (off by r·ell, x·ell).
            IntermediateEq("i_from_ka", i_mag_pu * I_base_from),
            IntermediateEq("i_to_ka", i_mag_pu * I_base_to),
            IntermediateEq(
                "loading_from_percent", i_mag_pu * I_base_from / branch.max_i_ka
            ),
            IntermediateEq(
                "loading_to_percent", i_mag_pu * I_base_to / branch.max_i_ka
            ),
        ]

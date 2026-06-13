r"""Branch-flow MISOCP electricity formulation: convex SOC relaxation
:math:`P^2 + Q^2 \le (W/tap^2) \cdot ell` plus big-M switching binaries."""

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


def _to_pu(value_mw, grid):
    """Normalise a branch power var to per-unit on the grid apparent-power base.

    ``p_*_mw`` / ``q_*_mvar`` are stored in MW / MVAr (the polar-AC NLP consumes
    them directly in those units). The branch-flow MISOCP works in per-unit, so
    every power that enters the SOC / voltage-drop / loss equations is divided
    by ``S_base = grid.sn_mva`` here - the single place that MW->pu conversion
    happens.
    """
    return value_mw / grid.sn_mva


class MISOCPElectricityNodeFormulation(NodeFormulation):
    def ensure_var(self, node, simulation=False, grid=None):
        node.vm_pu_squared = Var(1, min=0, max=2.25)
        node.vm_pu = Intermediate(1)

    def equations(
        self,
        node,
        grid,
        from_branch_models,
        to_branch_models,
        connected_child_models,
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
    r"""Per-unit :math:`|I|^2` upper bound from voltage bounds: :math:`4 \cdot W_{max} / |Z|^2`."""
    return 4 * w_max / (branch.br_r_pu**2 + branch.br_x_pu**2)


# Headroom factor on the thermal rating for the current_pu_squared bound. The bound is
# a big-M tightening device, not the operational loading constraint (that one
# is line_loading_limit()); 3x rating is far above any acceptable steady-state
# loading while keeping the bound finite on near-zero-impedance lines, where
# the voltage-derived 4*W_max/|Z|^2 explodes (~1e9) and wrecks the matrix
# conditioning badly enough that SCIP/Gurobi spuriously prove infeasibility.
_ELL_THERMAL_HEADROOM = 3.0


def _ell_max(branch, w_max: float, i_base_from: float, i_base_to: float) -> float:
    r"""Tightest available per-unit :math:`|I|^2` bound: voltage-derived, capped by the
    thermal rating (with headroom) when ``max_i_ka`` is available."""
    ell = _ell_physics_max(branch, w_max)
    max_i_ka = getattr(branch, "max_i_ka", None)
    if max_i_ka and max_i_ka > 0:
        i_base = min(i_base_from, i_base_to)
        ell_thermal = (_ELL_THERMAL_HEADROOM * max_i_ka / i_base) ** 2
        ell = min(ell, ell_thermal)
    return ell


def _big_m(w_max: float) -> float:
    r"""Cauchy-Schwarz big-M; with :math:`ell_{max} = 4 \cdot W_{max}/|Z|^2` this collapses to :math:`9 \cdot W_{max}`."""
    return 9 * w_max


class MISOCPElectricityBranchFormulation(BranchFormulation):
    def ensure_var(self, branch, simulation=False, grid=None):
        branch.current_pu_squared = Var(1, min=0)
        branch.i_from_ka = Intermediate(0)
        branch.i_to_ka = Intermediate(0)
        branch.loading_from_pu = Intermediate(0)
        branch.loading_to_pu = Intermediate(0)

    def minimize(self, branch, grid, from_node_model, to_node_model, **kwargs):
        # r \cdot ell loss term doubles as the relaxation-tightening incentive.
        return [branch.current_pu_squared * branch.br_r_pu]

    def _soc_constraints(self, branch, grid, from_node_model, tap):
        """SOC relaxation; the exact (non-convex) sibling overrides this with
        the equality form."""
        return [
            soc_rel(
                from_node_model.vars["vm_pu_squared"],
                _to_pu(branch.vars["p_from_mw"], grid),
                _to_pu(branch.vars["q_from_mvar"], grid),
                branch.current_pu_squared,
                tap=tap,
            ),
        ]

    def equations(self, branch, grid, from_node_model, to_node_model, **kwargs):
        w_max = grid.vm_pu_max**2
        big_m = _big_m(w_max)
        tap = _branch_tap(branch)
        sqrt_impl = kwargs["sqrt_impl"]
        # I_base_ka = S_base / (\sqrt{3} \cdot V_base); trafo primary divides by tap.
        I_base_from = grid.sn_mva / (SQRT_3 * from_node_model.base_kv) / tap
        I_base_to = grid.sn_mva / (SQRT_3 * to_node_model.base_kv)
        ell_phys = _ell_max(branch, w_max, I_base_from, I_base_to)
        i_mag_pu = sqrt_impl(branch.current_pu_squared)
        # loading^2 = current_pu_squared \cdot (I_base/max_i_ka)^2 is linear in current_pu_squared.
        # Used by line_loading_limit() instead of the sqrt-bearing form.
        branch._misocp_loading_from_scale_squared = (I_base_from / branch.max_i_ka) ** 2
        branch._misocp_loading_to_scale_squared = (I_base_to / branch.max_i_ka) ** 2
        return [
            branch.current_pu_squared <= ell_phys * branch.on_off,
            voltage_drop(
                from_node_model.vars["vm_pu_squared"],
                to_node_model.vars["vm_pu_squared"],
                _to_pu(branch.vars["p_from_mw"], grid),
                _to_pu(branch.vars["q_from_mvar"], grid),
                branch.current_pu_squared,
                branch.br_r_pu,
                branch.br_x_pu,
                tap=tap,
            )
            <= big_m * (1 - branch.on_off),
            voltage_drop(
                from_node_model.vars["vm_pu_squared"],
                to_node_model.vars["vm_pu_squared"],
                _to_pu(branch.vars["p_from_mw"], grid),
                _to_pu(branch.vars["q_from_mvar"], grid),
                branch.current_pu_squared,
                branch.br_r_pu,
                branch.br_x_pu,
                tap=tap,
            )
            >= -big_m * (1 - branch.on_off),
            *self._soc_constraints(branch, grid, from_node_model, tap),
            active_power_loss(
                _to_pu(branch.vars["p_from_mw"], grid),
                _to_pu(branch.vars["p_to_mw"], grid),
                branch.current_pu_squared,
                branch.br_r_pu,
            ),
            reactive_power_loss(
                _to_pu(branch.vars["q_from_mvar"], grid),
                _to_pu(branch.vars["q_to_mvar"], grid),
                branch.current_pu_squared,
                branch.br_x_pu,
            ),
            # |I_ka| = \sqrt{current_pu_squared} \cdot I_base. current_pu_squared is the from-side magnitude;
            # the to-side report is approximate (off by r \cdot ell, x \cdot ell).
            IntermediateEq("i_from_ka", i_mag_pu * I_base_from),
            IntermediateEq("i_to_ka", i_mag_pu * I_base_to),
            IntermediateEq(
                "loading_from_pu", i_mag_pu * I_base_from / branch.max_i_ka
            ),
            IntermediateEq(
                "loading_to_pu", i_mag_pu * I_base_to / branch.max_i_ka
            ),
        ]

"""
McCormick-tightened district-heating formulation (Deng et al., 2021,
https://doi.org/10.1016/j.eng.2021.06.006).

Heating side as a convex relaxation:
* ``H_out = c·m·τ_send``, ``H_in = c·m·τ_recv`` (eq. 9c/9d) → linear nodal
  balance (eq. 9a) including ``H_G`` / ``H_L`` from HeatGenerator/HeatLoad.
* Taylor-linearised heat loss (eq. 9b).
* McCormick envelopes (eq. 17b-17e) relax ``H_out = c·m·τ``;
  ``num_partitions > 1`` upgrades to the piecewise MILP form (eq. 18).

This is a relaxation — the bilinear is not pinned, the objective must drive
``H_out`` toward the surface. Tighten via ``WaterGrid.t_pu_min_env`` /
``t_pu_max_env``. Hydraulics omitted (paper §2.1); pipes are unidirectional.

Boundary children contribute ``c·m·τ_node·t_ref`` using the node's own τ.
"""

import math

import monee.model.phys.nonlinear.hf as ohfmodel
from monee.model.branch import WaterPipe
from monee.model.core import Const, Var
from monee.model.grid import WaterGrid
from monee.model.node import Junction
from monee.model.phys.core.hydraulics import calc_max_mass_flow

from ..core import BranchFormulation, NetworkFormulation, NodeFormulation

C_WATER = ohfmodel.SPECIFIC_HEAT_CAP_WATER


def _branch_m_U(branch, grid):
    """Per-pipe mass-flow upper bound [kg/s]: ``min(grid.f_max, π/4·D²·ρ·v_max)``,
    further capped by ``branch.m_U_design`` when set."""
    explicit = getattr(branch, "m_U_design", None)
    velocity_cap = min(
        grid.f_max,
        calc_max_mass_flow(branch.diameter_m, grid.fluid_density, grid.v_max_mps),
    )
    if explicit is None:
        return velocity_cap
    return min(velocity_cap, float(explicit))


def _t_pu_env_bounds(grid):
    """Per-unit node-temperature envelope ``(τ_L, τ_U)``. Tighten via
    ``grid.t_pu_min_env`` / ``grid.t_pu_max_env`` to shrink McCormick gaps."""
    return (
        getattr(grid, "t_pu_min_env", 0.3),
        getattr(grid, "t_pu_max_env", 2.0),
    )


class MccDHSNodeFormulation(NodeFormulation):
    """Linear nodal heat balance (eq. 9a) plus piecewise-McCormick τ
    disaggregation (eq. 18c/18e/18g). Sets ``_mccormick_dhs_active`` so
    :meth:`Junction.calc_signed_heat_flow` skips its degenerate balance."""

    # Tie-breaker pulling t_pu toward 1.0. Without it, leaf-consumer junctions
    # park on the envelope floor since the H_in envelope is loose for fixed-
    # flow loads. 1e-6 keeps objective distortion well below 1e-3 MIPGap.
    TPU_PULL_EPS = 1e-6

    def __init__(self, num_partitions: int = 1):
        self.num_partitions = num_partitions

    def minimize(
        self,
        node,
        grid,
        from_branch_models,
        to_branch_models,
        connected_child_models,
        **kwargs,
    ):
        return [self.TPU_PULL_EPS * (1.0 - node.vars["t_pu"])]

    def ensure_var(self, model):
        model._mccormick_dhs_active = True
        if self.num_partitions > 1:
            # Underscore prefix hides these from result DataFrames; solver
            # still picks them up via direct __dict__ iteration.
            for s in range(self.num_partitions):
                setattr(
                    model,
                    f"_t_pu_piece_{s}",
                    Var(0, min=0, name=f"t_pu_piece_{s}"),
                )
                setattr(
                    model,
                    f"_piece_y_{s}",
                    Var(0, min=0, max=1, integer=True, name=f"piece_y_{s}"),
                )

    def equations(
        self,
        node,
        grid,
        from_branch_models,
        to_branch_models,
        connected_child_models,
        **kwargs,
    ):
        # MW units; (c·t_ref/1e6) maps kg/s·t_pu → MW.
        scale_mw_per_kgs = C_WATER * grid.t_ref / 1e6

        eqs = []

        # LTC owns its own inter-temporal heat balance using the same H_in/H_out;
        # emitting eq. 9a here too would force T(t) ≡ T(t-1) at LTC junctions.
        ltc_owns_node = getattr(node, "_ltc_active", False)

        if not ltc_owns_node:
            # eq. 9c/9d — sender H_out, receiver H_in.
            h_out_terms = [
                bm.vars["H_out_mw"] * bm.vars.get("on_off", 1)
                for bm in from_branch_models
                if "H_out_mw" in bm.vars
            ]
            h_in_terms = [
                bm.vars["H_in_mw"] * bm.vars.get("on_off", 1)
                for bm in to_branch_models
                if "H_in_mw" in bm.vars
            ]

            # Load convention: HeatGenerator → negative q_mw_heat, HeatLoad → positive.
            q_child_terms = [
                cm.vars["q_mw_heat"] * cm.vars.get("regulation", 1)
                for cm in connected_child_models
                if "q_mw_heat" in cm.vars
            ]

            # Branch q_mw_heat (e.g. GasToHeatHG) absorbed at the TO node.
            q_branch_terms = [
                bm.vars["q_mw_heat"] * bm.vars.get("on_off", 1)
                for bm in to_branch_models
                if "q_mw_heat" in bm.vars
            ]

            # Use the node's own τ; fixed-supply inflow children pin τ via
            # overwrite(), collapsing the t_pu factor to a constant.
            boundary_enthalpy_in = [
                -cm.vars["mass_flow"]
                * cm.vars.get("regulation", 1)
                * scale_mw_per_kgs
                * node.vars["t_pu"]
                for cm in connected_child_models
                if "mass_flow" in cm.vars and "q_mw_heat" not in cm.vars
            ]

            if not (
                h_out_terms
                or h_in_terms
                or q_child_terms
                or q_branch_terms
                or boundary_enthalpy_in
            ):
                print("Warning: you are ignoring enthalpy equation.")
            else:
                # eq. 9a: Σ H_in + Σ H_boundary - Σ H_out = Σ q_child + Σ q_branch
                eqs.append(
                    sum(h_in_terms) + sum(boundary_enthalpy_in) - sum(h_out_terms)
                    == sum(q_child_terms) + sum(q_branch_terms)
                )

        # |S| = 1 uses the plain envelopes assembled on the branch side.
        if self.num_partitions > 1:
            tpu_L, tpu_U = _t_pu_env_bounds(grid)
            S = self.num_partitions
            tpu_pieces = [getattr(node, f"_t_pu_piece_{s}") for s in range(S)]
            y_pieces = [getattr(node, f"_piece_y_{s}") for s in range(S)]

            # 18c: τ_i = Σ τ_{i,s}
            eqs.append(node.vars["t_pu"] == sum(tpu_pieces))
            # 18e: exactly one partition active
            eqs.append(sum(y_pieces) == 1)
            # 18f/18g: τ_{i,s} bracketed to piece s when active, else 0
            for s in range(S):
                tL_s = tpu_L + (tpu_U - tpu_L) * s / S
                tU_s = tpu_L + (tpu_U - tpu_L) * (s + 1) / S
                eqs.append(tpu_pieces[s] >= tL_s * y_pieces[s])
                eqs.append(tpu_pieces[s] <= tU_s * y_pieces[s])

        return eqs


class MccDHSBranchFormulation(BranchFormulation):
    """Per-pipe McCormick-tightened DH formulation. ``H_out = c·m·τ`` is
    relaxed by the four McCormick inequalities (eq. 17b-17e); with
    ``num_partitions > 1`` it becomes piecewise (eq. 18b). Heat loss is
    Taylor-linearised (eq. 9b). Hydraulics omitted."""

    def __init__(self, num_partitions: int = 1):
        self.num_partitions = num_partitions

    def ensure_var(self, model):
        model.H_out_mw = Var(0, name="H_out_mw")
        model.H_in_mw = Var(0, name="H_in_mw")
        model.mass_flow_mag = Var(0, min=0, name="mass_flow_mag")
        # §2.1 fixed flow direction: only mass_flow_neg (m ≥ 0); pinning the
        # binary and mass_flow_pos to Const drops them from the LP.
        model.direction = Const(0)
        model.mass_flow_pos = Const(0)
        if self.num_partitions > 1:
            for s in range(self.num_partitions):
                setattr(
                    model,
                    f"_m_piece_{s}",
                    Var(0, min=0, name=f"m_piece_{s}"),
                )

    def equations(self, branch, grid, from_node_model, to_node_model, **kwargs):
        # vL = 2π·λ·L / ln(r_out/r_in) [W/K] · 1e-6 → MW.
        pipe_outside_r = branch.diameter_m / 2 + branch.insulation_thickness_m
        pipe_inside_r = branch.diameter_m / 2
        vL_mw_per_k = (
            2
            * math.pi
            * branch.lambda_insulation_w_per_k
            * branch.length_m
            / math.log(pipe_outside_r / pipe_inside_r)
            / 1e6
        )

        m_U = _branch_m_U(branch, grid)
        m_L = 0.0
        tpu_L, tpu_U = _t_pu_env_bounds(grid)

        scale_mw = C_WATER * grid.t_ref / 1e6
        t_a_pu = branch.temperature_ext_k / grid.t_ref

        t_pu_send = from_node_model.vars["t_pu"]
        m = branch.mass_flow_neg

        eqs = [
            branch.mass_flow_neg <= m_U * branch.on_off,
            branch.mass_flow_mag == branch.mass_flow_neg,
            # eq. 9b: Taylor-linearised heat loss.
            branch.H_in_mw
            == branch.H_out_mw - vL_mw_per_k * grid.t_ref * (t_pu_send - t_a_pu),
            branch.H_out_mw <= scale_mw * m_U * tpu_U * branch.on_off,
            branch.H_in_mw <= scale_mw * m_U * tpu_U * branch.on_off,
        ]

        if self.num_partitions <= 1:
            # eq. 17b-17e: McCormick envelopes.
            eqs += [
                branch.H_out_mw
                >= scale_mw * (m_L * t_pu_send + tpu_L * m - m_L * tpu_L),
                branch.H_out_mw
                >= scale_mw * (m_U * t_pu_send + tpu_U * m - m_U * tpu_U),
                branch.H_out_mw
                <= scale_mw * (m_L * t_pu_send + tpu_U * m - m_L * tpu_U),
                branch.H_out_mw
                <= scale_mw * (m_U * t_pu_send + tpu_L * m - m_U * tpu_L),
            ]
        else:
            # eq. 18b/18d/18h: piecewise McCormick over τ partition.
            S = self.num_partitions
            y_pieces = [getattr(from_node_model, f"_piece_y_{s}") for s in range(S)]
            tpu_pieces = [
                getattr(from_node_model, f"_t_pu_piece_{s}") for s in range(S)
            ]
            m_pieces = [getattr(branch, f"_m_piece_{s}") for s in range(S)]

            eqs.append(m == sum(m_pieces))
            for s in range(S):
                eqs.append(m_pieces[s] >= m_L * y_pieces[s])
                eqs.append(m_pieces[s] <= m_U * y_pieces[s])

            def _env_sum(m_coef, tau_coef):
                return scale_mw * sum(
                    m_coef * tpu_pieces[s]
                    + tau_coef(s) * m_pieces[s]
                    - m_coef * tau_coef(s) * y_pieces[s]
                    for s in range(S)
                )

            def _tL_s(s):
                return tpu_L + (tpu_U - tpu_L) * s / S

            def _tU_s(s):
                return tpu_L + (tpu_U - tpu_L) * (s + 1) / S

            eqs += [
                branch.H_out_mw >= _env_sum(m_L, _tL_s),
                branch.H_out_mw >= _env_sum(m_U, _tU_s),
                branch.H_out_mw <= _env_sum(m_U, _tL_s),
                branch.H_out_mw <= _env_sum(m_L, _tU_s),
            ]

        return eqs


def mccormick_dhs_gap_bound_mw(branch, grid, num_partitions: int = 1) -> float:
    """Worst-case ``H_out_mw`` gap [MW]:
    ``(c·t_ref/1e6) · m_U · (τ_U - τ_L) / (4·S)``."""
    tpu_L, tpu_U = _t_pu_env_bounds(grid)
    m_U = _branch_m_U(branch, grid)
    return C_WATER * grid.t_ref / 1e6 * m_U * (tpu_U - tpu_L) / (4 * num_partitions)


def mccormick_dhs_gap_bound_k(
    grid, num_partitions: int = 1, branch=None, mass_flow_kgs=None
) -> float:
    """Worst-case gap as a sender-temperature error [K]:
    ``t_ref·(τ_U - τ_L)/(4·S)`` at ``m = m_U``, scaling as ``m_U/m`` below."""
    tpu_L, tpu_U = _t_pu_env_bounds(grid)
    base_k = grid.t_ref * (tpu_U - tpu_L) / (4 * num_partitions)
    if mass_flow_kgs is None:
        return base_k
    if branch is None:
        raise ValueError("branch is required when mass_flow_kgs is provided")
    m_U = _branch_m_U(branch, grid)
    return base_k * m_U / mass_flow_kgs


def make_mccormick_dhs_formulation(num_partitions: int = 1) -> NetworkFormulation:
    """``num_partitions=1`` → plain McCormick LP (eq. 17). ``>1`` → piecewise
    MILP (eq. 18). Raise only when :func:`mccormick_dhs_gap_bound_k` shows the
    LP corner is non-physical on the network at hand."""
    return NetworkFormulation(
        branch_type_to_formulations={
            WaterPipe: MccDHSBranchFormulation(num_partitions=num_partitions),
        },
        node_type_to_formulations={
            (Junction, WaterGrid): MccDHSNodeFormulation(num_partitions=num_partitions),
        },
    )


MCCORMICK_DHS_NETWORK_FORMULATION = make_mccormick_dhs_formulation(num_partitions=1)

"""
McCormick-tightened district-heating formulation (Deng et al., 2021).

Reference: Deng et al., "Optimal Operation of Integrated Heat and Electricity
Systems: A Tightening McCormick Approach", Engineering 2021,
https://doi.org/10.1016/j.eng.2021.06.006

Heating side as a convex (LP/MILP) relaxation:

* Variable substitution ``H_out = c·m·τ_send`` and ``H_in = c·m·τ_recv``
  (eq. 9c/9d) — nodal balance (eq. 9a) becomes linear.
* Taylor-linearized heat loss ``H_in = H_out − v·L·(τ_send − τ_a)`` (eq. 9b).
* McCormick envelopes (eq. 17b-17e) relax ``H_out = c·m·τ`` — the LP form
  (``num_partitions = 1``).
* Piecewise McCormick (eq. 18) partitions the τ domain into ``|S|`` pieces
  with binary selectors and disaggregated ``τ_{i,s}`` / ``m_{ik,s}``;
  ``num_partitions > 1`` upgrades to MILP.
* Nodal balance (eq. 9a) absorbs ``H_G`` / ``H_L`` from
  :class:`HeatGenerator` / :class:`HeatLoad` children.

This is a *relaxation* — ``H_out = c·m·τ`` is not pinned. An objective
must drive the solver toward the bilinear surface; the paper's
bound-contraction iteration (Algorithm 1) is out of scope. Tighten the
τ envelope via ``WaterGrid.t_pu_min_env`` / ``t_pu_max_env``.

Hydraulics (Darcy-Weisbach, Reynolds) are intentionally omitted (paper §2.1):
pipes are constrained only by mass-flow bounds (eq. 1e) and nodal mass
balance (eq. 1b); junction pressures are determined externally by the pump.
Pipes are pinned unidirectional to match the paper's fixed-flow-direction
assumption.

Boundary hydraulic children (``ExtHydrGrid`` / ``Sink`` / ``Source`` /
``ConsumeHydrGrid``) contribute ``c·m·τ_node·t_ref`` using the node's own τ,
so outflow boundaries leave water at the node temperature.
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
    """Per-pipe mass-flow upper bound [kg/s].

    Honours an explicit ``m_U_design`` set by the network builder; otherwise
    falls back to ``min(grid.f_max, π/4·D²·ρ·v_max)``.
    """
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
    """Linear nodal heat balance (paper eq. 9a) plus piecewise-McCormick
    disaggregation of the node temperature (eq. 18c/18e/18g).

    Sets ``_mccormick_dhs_active`` on the junction so
    :meth:`Junction.calc_signed_heat_flow` skips its degenerate
    ``T_n × Σ(ṁ) = 0`` balance. With ``num_partitions > 1`` the node also
    carries |S| binary selectors ``y_{i,s}`` and disaggregated ``τ_{i,s}``
    pieces consumed by the branch formulation (eq. 18b).
    """

    # Tie-breaker pulling t_pu toward 1.0 (supply-slack temperature). The
    # receiver-side envelope on H_in is loose at fixed-flow leaf consumers:
    # the LP satisfies H_in = q_load + c·m·t_node at any low t_node since
    # the envelope doesn't enforce the bilinear downward. Without this,
    # ~3 % of leaf junctions park on the envelope floor (267 K).
    #
    # 1e-6 wins empirically on simbench LV-rural3 + S=20: shrinks objective
    # distortion to ~0.2 % (well below MIPGap=1e-3). 1e-4 distorts visibly
    # and lets junctions overshoot to the ceiling 1.15; 1e-7/1e-8 lose
    # strength.
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
        # See TPU_PULL_EPS.
        return [self.TPU_PULL_EPS * (1.0 - node.vars["t_pu"])]

    def ensure_var(self, model):
        model._mccormick_dhs_active = True
        if self.num_partitions > 1:
            # Underscore prefix hides these from result DataFrames (filtered
            # by ``GenericModel.vars``) while the solver still picks them up
            # via direct ``__dict__`` iteration.
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
        # Solve in MW; conversion factor (c·t_ref/1e6) maps kg/s·t_pu → MW.
        scale_mw_per_kgs = C_WATER * grid.t_ref / 1e6

        eqs = []

        # When :class:`~monee.model.extension.ltc.LumpedThermalCapacitance`
        # has taken over a junction (``_ltc_active``), it already emits an
        # *inter-temporal* heat balance built from the same ``H_in_mw`` /
        # ``H_out_mw`` quantities, plus the thermal-mass term
        # ``ρ·V·(T(t) − T(t−1))/Δt``.  Adding paper eq. 9a here on top of
        # that would enforce the *steady-state* balance simultaneously,
        # which together force the time-derivative term to zero — i.e.
        # T(t) ≡ T(t−1) at every LTC junction, defeating the extension.
        # Skip the steady-state balance for those nodes and let LTC own it.
        ltc_owns_node = getattr(node, "_ltc_active", False)

        if not ltc_owns_node:
            # Pipe enthalpies — paper eq. 9c/9d. Sender carries H_out,
            # receiver H_in.
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

            # Node-based heat children (paper eq. 9a). Load convention:
            # HeatGenerator → negative q_mw_heat, HeatLoad → positive.
            q_child_terms = [
                cm.vars["q_mw_heat"] * cm.vars.get("regulation", 1)
                for cm in connected_child_models
                if "q_mw_heat" in cm.vars
            ]

            # Branch-level q_mw_heat (e.g. GasToHeatHG) absorbed at the TO node.
            q_branch_terms = [
                bm.vars["q_mw_heat"] * bm.vars.get("on_off", 1)
                for bm in to_branch_models
                if "q_mw_heat" in bm.vars
            ]

            # Use the node's own τ so outflow boundaries (Sink) leave water at
            # node temperature. Fixed-supply inflow boundaries pin τ to the
            # child's t_k via ``overwrite``, collapsing the t_pu factor to a
            # constant.
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
                # Paper eq. 9a, load convention:
                #   Σ H_in + Σ H_boundary − Σ H_out = Σ q_child + Σ q_branch
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
    """Per-pipe McCormick-tightened district-heating formulation.

    Adds sender/receiver enthalpies ``H_out_mw`` / ``H_in_mw``. The bilinear
    ``H_out = c·m·τ`` is **not** enforced; the four McCormick inequalities
    (paper eq. 17b-17e) relax it over ``[m_L, m_U] × [τ_L, τ_U]``. With
    ``num_partitions > 1`` they become piecewise (eq. 18b) over the τ
    partition owned by :class:`MccDHSNodeFormulation`. Heat loss is
    Taylor-linearized (eq. 9b). Hydraulics (Δp, Reynolds, friction) are
    intentionally omitted — pipes are constrained only by the mass-flow
    bound (eq. 1e) and nodal mass balance (eq. 1b).
    """

    def __init__(self, num_partitions: int = 1):
        self.num_partitions = num_partitions

    def ensure_var(self, model):
        model.H_out_mw = Var(0, name="H_out_mw")
        model.H_in_mw = Var(0, name="H_in_mw")
        model.mass_flow_mag = Var(0, min=0, name="mass_flow_mag")
        # Paper §2.1 pins flow direction; only m ≥ 0 (mass_flow_neg) is used.
        # Pinning the direction binary and mass_flow_pos to constants drops
        # them from the LP entirely (presolve would only eliminate them
        # after matrix construction).
        model.direction = Const(0)
        model.mass_flow_pos = Const(0)
        if self.num_partitions > 1:
            # See MccDHSNodeFormulation.ensure_var on the underscore prefix.
            for s in range(self.num_partitions):
                setattr(
                    model,
                    f"_m_piece_{s}",
                    Var(0, min=0, name=f"m_piece_{s}"),
                )

    def equations(self, branch, grid, from_node_model, to_node_model, **kwargs):
        # vL = 2π·λ·L / ln(r_out/r_in) [W/K]; ÷1e6 lands the row in MW.
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

        # Paper eq. 1e/9g; honours per-pipe ``m_U_design`` overrides.
        m_U = _branch_m_U(branch, grid)
        m_L = 0.0
        tpu_L, tpu_U = _t_pu_env_bounds(grid)

        scale_mw = C_WATER * grid.t_ref / 1e6
        t_a_pu = branch.temperature_ext_k / grid.t_ref

        t_pu_send = from_node_model.vars["t_pu"]
        # Paper §2.1 fixed flow direction; mass_flow_pos is pinned in
        # ``ensure_var``, so all flow goes through mass_flow_neg (m ≥ 0).
        m = branch.mass_flow_neg

        eqs = [
            # Eq. 1e: mass-flow magnitude bound.
            branch.mass_flow_neg <= m_U * branch.on_off,
            branch.mass_flow_mag == branch.mass_flow_neg,
            # Eq. 9b: Taylor-linearized heat loss.
            branch.H_in_mw
            == branch.H_out_mw - vL_mw_per_k * grid.t_ref * (t_pu_send - t_a_pu),
            # Switch-off gating.
            branch.H_out_mw <= scale_mw * m_U * tpu_U * branch.on_off,
            branch.H_in_mw <= scale_mw * m_U * tpu_U * branch.on_off,
        ]

        if self.num_partitions <= 1:
            # Eq. 17b-17e: McCormick envelopes (lower×2, upper×2).
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
            # Eq. 18b/18d/18h: disaggregate τ (from_node) and m, then stack
            # four envelopes per piece.
            S = self.num_partitions
            y_pieces = [getattr(from_node_model, f"_piece_y_{s}") for s in range(S)]
            tpu_pieces = [
                getattr(from_node_model, f"_t_pu_piece_{s}") for s in range(S)
            ]
            m_pieces = [getattr(branch, f"_m_piece_{s}") for s in range(S)]

            # 18d: m_ik = Σ m_{ik,s}
            eqs.append(m == sum(m_pieces))
            # 18h: m_{ik,s} bracketed to [m_L, m_U] on the active piece
            for s in range(S):
                eqs.append(m_pieces[s] >= m_L * y_pieces[s])
                eqs.append(m_pieces[s] <= m_U * y_pieces[s])

            # 18b: four envelope sums across the partition.
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
    """Worst-case ``H_out_mw`` relaxation gap for one pipe [MW].

    For ``H_out = c·m·τ`` over ``[m_L, m_U] × [τ_L, τ_U]`` with ``m_L = 0``,
    the plain-McCormick gap peaks at the box centre at
    ``c·t_ref·(m_U−m_L)(τ_U−τ_L)/4``. Partitioning τ into ``S`` equal pieces
    divides this by ``S``:

        gap ≤ (c·t_ref/1e6) · (m_U − m_L)(τ_U − τ_L) / (4·S)

    Realized gaps are typically much smaller once the objective drives
    ``H_out`` toward the bilinear surface.
    """
    tpu_L, tpu_U = _t_pu_env_bounds(grid)
    m_U = _branch_m_U(branch, grid)
    return C_WATER * grid.t_ref / 1e6 * m_U * (tpu_U - tpu_L) / (4 * num_partitions)


def mccormick_dhs_gap_bound_k(
    grid, num_partitions: int = 1, branch=None, mass_flow_kgs=None
) -> float:
    """Worst-case relaxation gap as a sender-temperature error [K].

    ``H_out = c·m·τ_pu·t_ref`` gives ``ΔT_K = gap_W / (c·m)``. At ``m = m_U``
    the pipe factor cancels:

        ΔT_K(m_U) = t_ref · (τ_U − τ_L) / (4·S)

    For ``m < m_U`` the equivalent error scales as ``m_U/m``. Pass *branch*
    and ``mass_flow_kgs`` to evaluate at an operating point below the cap.
    """
    tpu_L, tpu_U = _t_pu_env_bounds(grid)
    base_k = grid.t_ref * (tpu_U - tpu_L) / (4 * num_partitions)
    if mass_flow_kgs is None:
        return base_k
    if branch is None:
        raise ValueError("branch is required when mass_flow_kgs is provided")
    m_U = _branch_m_U(branch, grid)
    return base_k * m_U / mass_flow_kgs


def make_mccormick_dhs_formulation(num_partitions: int = 1) -> NetworkFormulation:
    """McCormick-DHS formulation with ``num_partitions`` τ-partitions.

    ``num_partitions == 1``  → plain McCormick (LP, paper eq. 17).
    ``num_partitions > 1``   → piecewise McCormick (MILP, paper eq. 18).

    Default of 1 is empirical: for objectives that already drive ``H_out``
    toward the bilinear surface (load-shedding, fuel cost), refinement adds
    ``S × |junctions|`` binaries and ``4S × |pipes|`` envelope rows without
    measurably tightening the optimum. Raise to 4–10 only when an a-priori
    bound check (see :func:`mccormick_dhs_gap_bound_k`) shows the LP corner
    is non-physical on the network at hand.
    """
    return NetworkFormulation(
        branch_type_to_formulations={
            WaterPipe: MccDHSBranchFormulation(num_partitions=num_partitions),
        },
        node_type_to_formulations={
            (Junction, WaterGrid): MccDHSNodeFormulation(num_partitions=num_partitions),
        },
    )


MCCORMICK_DHS_NETWORK_FORMULATION = make_mccormick_dhs_formulation(num_partitions=1)

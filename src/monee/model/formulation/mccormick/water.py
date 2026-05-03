"""
McCormick-tightened district-heating formulation (Deng et al., 2021).

Reference: Deng et al., "Optimal Operation of Integrated Heat and Electricity
Systems: A Tightening McCormick Approach", Engineering 2021,
https://doi.org/10.1016/j.eng.2021.06.006

This module implements the heating side of the paper as a convex
(LP/MILP) relaxation of the quality-quantity regulated DHS model:

* Variable substitution  ``H_out = c·m·τ_send``  and  ``H_in = c·m·τ_recv``
  (paper eq. 9c/9d) — the nodal balance (eq. 9a) becomes linear.
* First-order Taylor linearization of the exponential heat-loss factor
  ``exp(-vL/(c·m))`` reducing the pipe heat-loss relation to
  ``H_in = H_out − v·L·(τ_send − τ_a)`` (paper eq. 9b).
* Plain McCormick envelopes (paper eq. 17b-17e) relax the sending-end
  bilinear term ``H_out = c·m·τ`` over its finite box — this is the LP
  form of the formulation (``num_partitions = 1``).
* **Piecewise McCormick refinement** (paper eq. 18) partitions the node-
  temperature domain into ``|S|`` disjoint regions and introduces binary
  selectors ``y_{i,s}`` plus disaggregated ``τ_{i,s}`` / ``m_{ik,s}``
  variables.  Each piece carries a tighter McCormick envelope, so the
  overall relaxation gap shrinks as ``|S|`` grows.  Selecting
  ``num_partitions > 1`` upgrades the LP to a MILP.
* Nodal heat balance (paper eq. 9a) with explicit node-based heat
  injection ``H_G`` and withdrawal ``H_L`` provided by
  :class:`~monee.model.child.HeatGenerator` and
  :class:`~monee.model.child.HeatLoad` children.

The formulation is a **relaxation** — it does not pin
``H_out = c·m·τ`` to the exact bilinear.  In an optimisation context
(paper eq. 13a: minimise fuel / exchange / loss costs) the objective
drives the solver toward the bilinear surface, and the paper's
bound-contraction iteration (Algorithm 1) further tightens the
envelopes until the residual ``|H_out − c·m·τ|`` drops below a
tolerance.  That iterative loop is out of scope here; users of this
formulation must either provide a meaningful objective or supply tight
a-priori bounds on the partition domain via ``WaterGrid.t_pu_min_env`` /
``t_pu_max_env``.

Unlike monee's default NL water formulation, the paper deliberately
**omits** Darcy-Weisbach and Reynolds: pipes are constrained only by
mass-flow bounds (eq. 1e) and the nodal mass balance (eq. 1b).  Junction
pressures are therefore left free — the pump determines them outside
the heat-flow optimisation (paper §2.1).

Pipes are forced unidirectional (flow pinned ``from_node → to_node``) to
match the paper's predetermined-flow-direction assumption.

Boundary hydraulic children (``ExtHydrGrid`` / ``Sink`` / ``Source`` /
``ConsumeHydrGrid``) contribute a virtual ``c·m·τ_node·t_ref`` enthalpy
term on the node balance, using the node's own ``τ`` so outflow
boundaries (``Sink``) correctly leave the water at the node
temperature, not a pre-imposed return temperature.
"""

import math

import monee.model.phys.nonlinear.hf as ohfmodel
from monee.model.branch import WaterPipe
from monee.model.core import Var
from monee.model.grid import WaterGrid
from monee.model.node import Junction
from monee.model.phys.core.hydraulics import calc_max_mass_flow

from ..core import BranchFormulation, NetworkFormulation, NodeFormulation

C_WATER = ohfmodel.SPECIFIC_HEAT_CAP_WATER


def _t_pu_env_bounds(grid):
    """Envelope bounds on the per-unit node temperature.  Tighter bounds
    produce tighter McCormick envelopes — users can narrow the domain by
    setting ``grid.t_pu_min_env`` / ``grid.t_pu_max_env``."""
    return (
        getattr(grid, "t_pu_min_env", 0.3),
        getattr(grid, "t_pu_max_env", 2.0),
    )


class MccDHSNodeFormulation(NodeFormulation):
    """Linear nodal heat balance (paper eq. 9a) + piecewise-McCormick
    disaggregation of the node temperature (paper eq. 18c/18e/18g).

    The default :meth:`Junction.calc_signed_heat_flow` forms the degenerate
    ``T_n × Σ(ṁ) = 0`` balance; under this formulation, the linear nodal
    balance (paper eq. 9a) replaces it.  A ``_mccormick_dhs_active`` flag
    on the junction short-circuits the default balance.

    For ``num_partitions > 1`` the node additionally carries |S| binary
    selectors ``y_{i,s}`` and disaggregated pieces ``τ_{i,s}`` so the
    matching branch formulation can assemble per-piece envelopes (eq. 18b).
    """

    def __init__(self, num_partitions: int = 1):
        self.num_partitions = num_partitions

    def ensure_var(self, model):
        model._mccormick_dhs_active = True
        if self.num_partitions > 1:
            # Underscore-prefixed so the piecewise internals are hidden
            # from result DataFrames (filtered by ``GenericModel.vars``)
            # while still picked up by the solver (which iterates
            # ``__dict__`` directly).
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
        c = C_WATER

        # Pipe enthalpies.  "from" branches carry H_out_w at this node
        # (sender); "to" branches carry H_in_w (receiver) — paper eq. 9c/9d.
        h_out_terms = [
            bm.vars["H_out_w"] * bm.vars.get("on_off", 1)
            for bm in from_branch_models
            if "H_out_w" in bm.vars
        ]
        h_in_terms = [
            bm.vars["H_in_w"] * bm.vars.get("on_off", 1)
            for bm in to_branch_models
            if "H_in_w" in bm.vars
        ]

        # Node-based H_G / H_L (paper eq. 9a).  Load convention:
        # HeatGenerator → negative q_w_heat, HeatLoad → positive.
        q_child_terms = [
            cm.vars["q_w_heat"] * cm.vars.get("regulation", 1)
            for cm in connected_child_models
            if "q_w_heat" in cm.vars
        ]

        # Branch-level H_G / H_L.  A multi-grid branch (e.g. GasToHeatHG) may
        # carry a q_w_heat Var absorbed at its TO end, in the same load
        # convention as the child-based injection above.
        q_branch_terms = [
            bm.vars["q_w_heat"] * bm.vars.get("on_off", 1)
            for bm in to_branch_models
            if "q_w_heat" in bm.vars
        ]

        # Boundary enthalpy from mass-transporting children.  Using the
        # node's own τ (not a child-local t_k) is mandatory for outflow
        # boundaries — withdrawn water leaves at the node's temperature.
        # For fixed-supply inflow boundaries the ``overwrite`` hook pins
        # τ to the child's t_k so both interpretations coincide.
        boundary_enthalpy_in = [
            -cm.vars["mass_flow"]
            * cm.vars.get("regulation", 1)
            * c
            * node.vars["t_pu"]
            * grid.t_ref
            for cm in connected_child_models
            if "mass_flow" in cm.vars and "q_w_heat" not in cm.vars
        ]

        if not (
            h_out_terms
            or h_in_terms
            or q_child_terms
            or q_branch_terms
            or boundary_enthalpy_in
        ):
            print("Warning: you are ignoring enthalpy equation.")
            return []

        # Paper eq. (9a), rearranged for load convention:
        #   Σ H_in_pipes + Σ H_boundary_in − Σ H_out_pipes = Σ q_child + Σ q_branch
        eqs = [
            sum(h_in_terms) + sum(boundary_enthalpy_in) - sum(h_out_terms)
            == sum(q_child_terms) + sum(q_branch_terms)
        ]

        # Piecewise disaggregation (paper eq. 18c/18e/18g).  For |S|=1 the
        # plain envelopes (17b-e) assembled on the branch side suffice.
        if self.num_partitions > 1:
            tpu_L, tpu_U = _t_pu_env_bounds(grid)
            S = self.num_partitions
            tpu_pieces = [getattr(node, f"_t_pu_piece_{s}") for s in range(S)]
            y_pieces = [getattr(node, f"_piece_y_{s}") for s in range(S)]

            # 18c: τ_i = Σ τ_{i,s}
            eqs.append(node.vars["t_pu"] == sum(tpu_pieces))
            # 18e: exactly one partition is active
            eqs.append(sum(y_pieces) == 1)
            # 18f+18g: τ_{i,s} bracketed to partition s when active, else 0
            for s in range(S):
                tL_s = tpu_L + (tpu_U - tpu_L) * s / S
                tU_s = tpu_L + (tpu_U - tpu_L) * (s + 1) / S
                eqs.append(tpu_pieces[s] >= tL_s * y_pieces[s])
                eqs.append(tpu_pieces[s] <= tU_s * y_pieces[s])

        return eqs


class MccDHSBranchFormulation(BranchFormulation):
    """Per-pipe McCormick-tightened district-heating formulation.

    Adds ``H_out_w`` / ``H_in_w`` (Watts) for the sender/receiver enthalpy
    flows.  The bilinear ``H_out = c·m·τ`` is **not** enforced directly —
    the four McCormick inequalities (paper eq. 17b-17e) form a convex
    relaxation over the box ``[m_L, m_U] × [τ_L, τ_U]``.  With
    ``num_partitions > 1`` the inequalities are replaced by their
    piecewise variant (paper eq. 18b) over the node-temperature
    partition introduced by :class:`MccDHSNodeFormulation`.

    Heat loss is linearized via first-order Taylor expansion (eq. 9b).

    Hydraulics (pressure drop, Reynolds, piecewise friction) are
    **intentionally omitted** — the paper constrains pipes only by the
    mass-flow bound (eq. 1e) and the nodal mass balance (eq. 1b).
    """

    def __init__(self, num_partitions: int = 1):
        self.num_partitions = num_partitions

    def ensure_var(self, model):
        model.H_out_w = Var(0, name="H_out_w")
        model.H_in_w = Var(0, name="H_in_w")
        model.mass_flow_mag = Var(0, min=0, name="mass_flow_mag")
        if self.num_partitions > 1:
            # Hidden from result DataFrames; see MccDHSNodeFormulation.ensure_var.
            for s in range(self.num_partitions):
                setattr(
                    model,
                    f"_m_piece_{s}",
                    Var(0, min=0, name=f"m_piece_{s}"),
                )

    def equations(self, branch, grid, from_node_model, to_node_model, **kwargs):
        # v·L = 2π·λ·L / ln(r_out/r_in)  — heat-transfer coefficient × length [W/K].
        pipe_outside_r = branch.diameter_m / 2 + branch.insulation_thickness_m
        pipe_inside_r = branch.diameter_m / 2
        vL = (
            2
            * math.pi
            * branch.lambda_insulation_w_per_k
            * branch.length_m
            / math.log(pipe_outside_r / pipe_inside_r)
        )

        # Mass-flow upper bound (paper eq. 1e / 9g).
        m_U = min(
            grid.f_max,
            calc_max_mass_flow(branch.diameter_m, grid.fluid_density, grid.v_max_mps),
        )
        m_L = 0.0
        tpu_L, tpu_U = _t_pu_env_bounds(grid)

        # scale · (m · τ_pu)  gives Watts (τ_pu · t_ref = Kelvin).
        scale = C_WATER * grid.t_ref
        t_a_pu = branch.temperature_ext_k / grid.t_ref

        t_pu_send = from_node_model.vars["t_pu"]
        # Paper §2.1: fixed flow direction.  Pinning direction=0 and
        # mass_flow_pos=0 routes all flow through mass_flow_neg, which we
        # use as the paper's m ≥ 0.
        m = branch.mass_flow_neg

        eqs = [
            branch.direction == 0,
            branch.mass_flow_pos == 0,
            # Paper eq. 1e: mass-flow magnitude bound.
            branch.mass_flow_neg <= m_U * branch.on_off,
            branch.mass_flow_mag == branch.mass_flow_neg,
            # Paper eq. 9b — Taylor-linearized heat loss along the pipe.
            branch.H_in_w == branch.H_out_w - vL * grid.t_ref * (t_pu_send - t_a_pu),
            # Switch-off gating: zero enthalpies when the pipe is off.
            branch.H_out_w <= scale * m_U * tpu_U * branch.on_off,
            branch.H_in_w <= scale * m_U * tpu_U * branch.on_off,
        ]

        if self.num_partitions <= 1:
            # Paper eq. 17b / 17c — lower McCormick envelopes.
            # Paper eq. 17d / 17e — upper McCormick envelopes.
            eqs += [
                branch.H_out_w >= scale * (m_L * t_pu_send + tpu_L * m - m_L * tpu_L),
                branch.H_out_w >= scale * (m_U * t_pu_send + tpu_U * m - m_U * tpu_U),
                branch.H_out_w <= scale * (m_L * t_pu_send + tpu_U * m - m_L * tpu_U),
                branch.H_out_w <= scale * (m_U * t_pu_send + tpu_L * m - m_U * tpu_L),
            ]
        else:
            # Paper eq. 18b/18d/18h — piecewise McCormick.  Disaggregate
            # both the node temperature (owned by the from_node) and the
            # branch mass flow, then stack four envelopes per piece.
            S = self.num_partitions
            y_pieces = [getattr(from_node_model, f"_piece_y_{s}") for s in range(S)]
            tpu_pieces = [
                getattr(from_node_model, f"_t_pu_piece_{s}") for s in range(S)
            ]
            m_pieces = [getattr(branch, f"_m_piece_{s}") for s in range(S)]

            # 18d: m_ik = Σ m_{ik,s}
            eqs.append(m == sum(m_pieces))
            # 18h: m_{ik,s} bracketed to [m_L, m_U] when the piece is active
            for s in range(S):
                eqs.append(m_pieces[s] >= m_L * y_pieces[s])
                eqs.append(m_pieces[s] <= m_U * y_pieces[s])

            # 18b: four envelope sums across the partition.
            def _env_sum(m_coef, tau_coef):
                return scale * sum(
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
                branch.H_out_w >= _env_sum(m_L, _tL_s),
                branch.H_out_w >= _env_sum(m_U, _tU_s),
                branch.H_out_w <= _env_sum(m_U, _tL_s),
                branch.H_out_w <= _env_sum(m_L, _tU_s),
            ]

        return eqs


def mccormick_dhs_gap_bound_w(branch, grid, num_partitions: int = 1) -> float:
    """Worst-case relaxation gap of ``H_out_w`` for one pipe under the
    piecewise McCormick-DHS formulation, in Watts.

    For the bilinear ``H_out = c·m·τ`` over the box
    ``[m_L, m_U] × [τ_L, τ_U]`` (with ``m_L = 0``), the maximum gap of the
    plain McCormick envelope is ``c·t_ref·(m_U−m_L)(τ_U−τ_L)/4``, attained
    at the box center.  Partitioning τ into ``S`` equal pieces shrinks the
    active piece's τ-width to ``(τ_U−τ_L)/S``, so the bound becomes:

        gap ≤ c · t_ref · (m_U − m_L)(τ_U − τ_L) / (4·S)

    The envelope ``[τ_L, τ_U]`` is read from ``grid.t_pu_min_env`` /
    ``grid.t_pu_max_env`` (defaults: 0.3, 2.0).  ``m_U`` is the per-pipe
    cap ``min(grid.f_max, π/4·D²·ρ·v_max)``.

    This is a worst-case bound; the realized gap is typically much smaller
    once an objective drives the solver toward the bilinear surface.
    """
    tpu_L, tpu_U = _t_pu_env_bounds(grid)
    m_U = min(
        grid.f_max,
        calc_max_mass_flow(branch.diameter_m, grid.fluid_density, grid.v_max_mps),
    )
    return C_WATER * grid.t_ref * m_U * (tpu_U - tpu_L) / (4 * num_partitions)


def mccormick_dhs_gap_bound_k(
    grid, num_partitions: int = 1, branch=None, mass_flow_kgs=None
) -> float:
    """Worst-case relaxation gap expressed as an equivalent temperature
    error [K] on the sender temperature ``τ_send``.

    Since ``H_out = c·m·τ_pu·t_ref``, the Watts gap maps to a τ-error via
    ``ΔT_K = gap_W / (c·m)``.  At the upper mass-flow bound ``m_U`` the
    pipe-specific ``m_U`` factor cancels and the bound reduces to a
    branch-independent quantity:

        ΔT_K(m_U) = t_ref · (τ_U − τ_L) / (4·S)

    For ``m < m_U`` the equivalent error grows as ``m_U/m`` — at low flow
    the relaxation has more slack on τ, but the absolute heat error is
    bounded by ``mccormick_dhs_gap_bound_w``.  Pass *branch* together with
    a specific ``mass_flow_kgs`` to evaluate the bound at an operating
    point below the cap.
    """
    tpu_L, tpu_U = _t_pu_env_bounds(grid)
    base_k = grid.t_ref * (tpu_U - tpu_L) / (4 * num_partitions)
    if mass_flow_kgs is None:
        return base_k
    if branch is None:
        raise ValueError("branch is required when mass_flow_kgs is provided")
    m_U = min(
        grid.f_max,
        calc_max_mass_flow(branch.diameter_m, grid.fluid_density, grid.v_max_mps),
    )
    return base_k * m_U / mass_flow_kgs


def make_mccormick_dhs_formulation(num_partitions: int = 1) -> NetworkFormulation:
    """Build a McCormick-DHS network formulation with ``num_partitions``
    nodal-temperature partitions.

    ``num_partitions == 1``  →  plain McCormick (LP relaxation, paper eq. 17).
    ``num_partitions > 1``   →  piecewise McCormick (MILP, paper eq. 18).
    """
    return NetworkFormulation(
        branch_type_to_formulations={
            WaterPipe: MccDHSBranchFormulation(num_partitions=num_partitions),
        },
        node_type_to_formulations={
            (Junction, WaterGrid): MccDHSNodeFormulation(num_partitions=num_partitions),
        },
    )


# Module-level default: plain McCormick (LP).
MCCORMICK_DHS_NETWORK_FORMULATION = make_mccormick_dhs_formulation(num_partitions=1)

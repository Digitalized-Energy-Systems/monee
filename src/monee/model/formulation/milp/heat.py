r"""
McCormick-tightened district-heating formulation (Deng et al., 2021,
https://doi.org/10.1016/j.eng.2021.06.006) plus the fixed-flow heat exchanger.

Heating side as a convex relaxation:

* :math:`H_{out} = c \cdot m \cdot \tau_{send}`, :math:`H_{in} = c \cdot m \cdot \tau_{recv}` (eq. 9c/9d) :math:`\to` linear nodal
  balance (eq. 9a) including ``H_G`` / ``H_L`` from HeatGenerator/HeatLoad.
* Taylor-linearised heat loss (eq. 9b).
* McCormick envelopes (eq. 17b-17e) relax :math:`H_{out} = c \cdot m \cdot \tau`;
  ``num_partitions > 1`` upgrades to the piecewise MILP form (eq. 18).

This is a relaxation - the bilinear is not pinned, the objective must drive
``H_out`` toward the surface. Tighten via ``WaterGrid.t_pu_min_env`` /
``t_pu_max_env``. Hydraulics omitted (paper §2.1); pipes are unidirectional.

Boundary children contribute :math:`c \cdot m \cdot \tau_{node} \cdot t_{ref,k}` using the node's own :math:`\tau`.

The heat-exchanger pair (:class:`McCormickHeatExchangerFormulation` /
:class:`McCormickPassiveHeatExchangerFormulation`) extends the same H-space
balance to HE branches so a network with HEs stays convex end to end.
"""

import math
import warnings

import monee.model.phys.nonlinear.hf as ohfmodel
from monee.model.core import Const, Var
from monee.model.phys.core.hydraulics import calc_max_mass_flow

from ..core import BranchFormulation, NodeFormulation

C_WATER = ohfmodel.SPECIFIC_HEAT_CAP_WATER


class FixedFlowHeatExchangerFormulation(BranchFormulation):
    r"""Active heat exchanger driven by a prescribed (design) mass flow.

    LP while the through-flow is prescribed - the energy balance
    :math:`t_{out} \cdot (m \cdot c \cdot t_{ref,k}) = t_{in} \cdot (m \cdot c \cdot t_{ref,k}) - q_{delivered}` is linear in that
    case. When the surrounding network determines the flow instead
    (``_he_flow_prescribed == False``), the balance runs on the actual flow
    magnitude and turns bilinear.
    """

    def ensure_var(self, model, simulation=False, grid=None):
        m_seed = getattr(model, "_mass_flow_seed_kgs", 0.0)
        q_design = getattr(model, "_q_design_mw", None)

        # Warm-start t_out at the design dT spread (in pu), in the direction the
        # duty drives it: a generator (q < 0) raises the outlet, a load drops
        # it. Keeps the start off the t_pu rail the zero-flow corner forces.
        t_out_seed = 1.0
        t_ref_k = getattr(grid, "t_ref_k", None) if not isinstance(
            grid, (list, tuple)
        ) else None
        if t_ref_k and q_design:
            dt_pu = model._T_delta_design_K / t_ref_k
            t_out_seed = 1.0 + dt_pu if q_design < 0 else 1.0 - dt_pu

        model.t_in_pu = Var(1, min=0, max=2, name="t_in_pu")
        model.t_out_pu = Var(t_out_seed, min=0, max=2, name="t_out_pu")
        # t_to_pu == t_out_pu in equations(); keep its seed consistent.
        if hasattr(model, "t_to_pu"):
            model.t_to_pu.value = t_out_seed
        if isinstance(model.mass_flow_design_kgs, Var):
            model.mass_flow_mag_kgs = Var(m_seed, min=0)
        else:
            model.mass_flow_mag_kgs = Var(
                m_seed, min=0, max=model.mass_flow_design_kgs
            )

        q_seed = q_design or 0.0
        if model.q_mw_set <= 0 or isinstance(model.q_mw_set, Var):
            model._he_is_generator = True
            model.q_mw_delivered = Var(min(q_seed, 0.0), max=0, name="q_mw_delivered")
        elif model.q_mw_set > 0:
            model._he_is_generator = False
            model.q_mw_delivered = Var(max(q_seed, 0.0), min=0, name="q_mw_delivered")

    def minimize(self, branch, grid, from_node_model, to_node_model, **kwargs):
        if branch._he_is_generator:
            return [branch.q_mw_delivered]
        return [-branch.q_mw_delivered]

    def equations(self, branch, grid, from_node_model, to_node_model, **kwargs):
        eqs = [branch.direction == 0]
        eqs += self._he_equations(branch, grid, from_node_model)
        return eqs

    def _he_equations(self, branch, grid, from_node_model):
        cp_mw_per_kgs_K = ohfmodel.SPECIFIC_HEAT_CAP_WATER / 1e6

        # equations() runs after solver-var injection, when mass_flow_design_kgs
        # is no longer a monee Var - the model's construction-time flag is the
        # only reliable dynamic/fixed discriminator here.
        is_dynamic_mf = branch._calc_mass_flow

        # Set by mark_he_flow_prescription() during solver prep; defaults to
        # the prescribing behaviour (supply/return semantics).
        prescribed = getattr(branch, "_he_flow_prescribed", True)

        if is_dynamic_mf and not prescribed:
            # SubHE whose through-flow the surrounding network already
            # determines (e.g. a fixed-mass-flow sink fed only through the
            # compound): q_mw is dictated by the control node and
            # mass_flow_design_kgs is only the sizing value (q_mw at the design
            # temperature spread). Pinning the flow to it would over-determine
            # the system, so the energy balance runs on the actual flow
            # magnitude instead.
            flow_eqs = [
                branch.mass_flow_mag_kgs
                == branch.mass_flow_pos_kgs + branch.mass_flow_neg_kgs
            ]
            balance_flow_kgs = branch.mass_flow_mag_kgs
        else:
            flow_eqs = [
                branch.mass_flow_mag_kgs == branch.mass_flow_design_kgs,
                branch.mass_flow_neg_kgs == branch.mass_flow_design_kgs * branch.on_off,
            ]
            balance_flow_kgs = branch.mass_flow_design_kgs

        eqs = flow_eqs + [
            branch.mass_flow_pos_kgs == 0,
            branch.t_in_pu == from_node_model.vars["t_pu"],
            branch.t_from_pu == from_node_model.vars["t_pu"],
            branch.t_to_pu == branch.t_out_pu,
            branch.t_out_pu * (balance_flow_kgs * cp_mw_per_kgs_K * grid.t_ref_k)
            == branch.t_in_pu * (balance_flow_kgs * cp_mw_per_kgs_K * grid.t_ref_k)
            - branch.q_mw_delivered,
        ]
        if branch._he_is_generator:
            eqs.append(branch.q_mw_delivered >= branch.q_mw * branch.on_off)
        else:
            eqs.append(branch.q_mw_delivered <= branch.q_mw * branch.on_off)
        return eqs


def _branch_m_U(branch, grid):
    r"""Per-pipe mass-flow upper bound [kg/s]: the smaller of ``grid.max_mass_flow_kgs``
    and the velocity cap :math:`\pi/4 \cdot D^2 \cdot \rho \cdot v_{max}`, further capped
    by ``branch.m_U_design`` when set."""
    explicit = getattr(branch, "m_U_design", None)
    velocity_cap = min(
        grid.max_mass_flow_kgs,
        calc_max_mass_flow(
            branch.diameter_m, grid.fluid_density_kg_per_m3, grid.v_max_mps
        ),
    )
    if explicit is None:
        return velocity_cap
    return min(velocity_cap, float(explicit))


def _t_pu_env_bounds(grid):
    r"""Per-unit node-temperature envelope :math:`(\tau_L, \tau_U)`. Tighten via
    ``grid.t_pu_min_env`` / ``grid.t_pu_max_env`` to shrink McCormick gaps."""
    return (
        getattr(grid, "t_pu_min_env", 0.3),
        getattr(grid, "t_pu_max_env", 2.0),
    )


class McCormickHeatNodeFormulation(NodeFormulation):
    r"""Linear nodal heat balance (eq. 9a) plus piecewise-McCormick :math:`\tau`
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

    def ensure_var(self, model, simulation=False, grid=None):
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
        # MW units; (c \cdot t_ref_k/1e6) maps kg/s \cdot t_pu \to MW.
        scale_mw_per_kgs = C_WATER * grid.t_ref_k / 1e6

        eqs = []

        # LTC owns its own inter-temporal heat balance using the same H_in/H_out;
        # emitting eq. 9a here too would force T(t) \equiv T(t-1) at LTC junctions.
        ltc_owns_node = getattr(node, "_ltc_active", False)

        if not ltc_owns_node:
            # eq. 9c/9d - sender H_out, receiver H_in.
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

            # Load convention: HeatGenerator \to negative q_mw, HeatLoad \to positive.
            q_child_terms = [
                cm.vars["q_mw_heat"] * cm.vars.get("regulation", 1)
                for cm in connected_child_models
                if "q_mw_heat" in cm.vars
            ]

            # Branch q_mw (e.g. GasToHeatHG) absorbed at the TO node.
            q_branch_terms = [
                bm.vars["q_mw_heat"] * bm.vars.get("on_off", 1)
                for bm in to_branch_models
                if "q_mw_heat" in bm.vars
            ]

            # Use the node's own \tau; fixed-supply inflow children pin \tau via
            # overwrite(), collapsing the t_pu factor to a constant.
            boundary_enthalpy_in = [
                -cm.vars["mass_flow_kgs"]
                * cm.vars.get("regulation", 1)
                * scale_mw_per_kgs
                * node.vars["t_pu"]
                for cm in connected_child_models
                if "mass_flow_kgs" in cm.vars and "q_mw_heat" not in cm.vars
            ]

            if not (
                h_out_terms
                or h_in_terms
                or q_child_terms
                or q_branch_terms
                or boundary_enthalpy_in
            ):
                warnings.warn(
                    "Node contributes no enthalpy terms; nodal heat balance skipped."
                )
            else:
                # eq. 9a: \sum H_in + \sum H_boundary - \sum H_out = \sum q_child + \sum q_branch
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

            # 18c: \tau_i = \sum \tau_{i,s}
            eqs.append(node.vars["t_pu"] == sum(tpu_pieces))
            # 18e: exactly one partition active
            eqs.append(sum(y_pieces) == 1)
            # 18f/18g: \tau_{i,s} bracketed to piece s when active, else 0
            for s in range(S):
                tL_s = tpu_L + (tpu_U - tpu_L) * s / S
                tU_s = tpu_L + (tpu_U - tpu_L) * (s + 1) / S
                eqs.append(tpu_pieces[s] >= tL_s * y_pieces[s])
                eqs.append(tpu_pieces[s] <= tU_s * y_pieces[s])

        return eqs


class McCormickHeatBranchFormulation(BranchFormulation):
    r"""Per-pipe McCormick-tightened DH formulation. :math:`H_{out} = c \cdot m \cdot \tau` is
    relaxed by the four McCormick inequalities (eq. 17b-17e); with
    ``num_partitions > 1`` it becomes piecewise (eq. 18b). Heat loss is
    Taylor-linearised (eq. 9b). Hydraulics omitted."""

    def __init__(self, num_partitions: int = 1):
        self.num_partitions = num_partitions

    def ensure_var(self, model, simulation=False, grid=None):
        model.H_out_mw = Var(0, name="H_out_mw")
        model.H_in_mw = Var(0, name="H_in_mw")
        model.mass_flow_mag_kgs = Var(0, min=0, name="mass_flow_mag_kgs")
        # §2.1 fixed flow direction: only mass_flow_neg_kgs (m \ge 0); pinning the
        # binary and mass_flow_pos_kgs to Const drops them from the LP.
        model.direction = Const(0)
        model.mass_flow_pos_kgs = Const(0)
        if self.num_partitions > 1:
            for s in range(self.num_partitions):
                setattr(
                    model,
                    f"_m_piece_{s}",
                    Var(0, min=0, name=f"m_piece_{s}"),
                )

    def _heat_balance_eqs(self, branch, grid, from_node_model):
        """eq. 9b: Taylor-linearised insulation heat loss along the pipe."""
        # vL = 2 \pi \cdot \lambda \cdot L / ln(r_out/r_in) [W/K] \cdot 1e-6 \to MW.
        pipe_outside_r = branch.diameter_m / 2 + branch.insulation_thickness_m
        pipe_inside_r = branch.diameter_m / 2
        vL_mw_per_k = (
            2
            * math.pi
            * branch.lambda_insulation_w_per_m_k
            * branch.length_m
            / math.log(pipe_outside_r / pipe_inside_r)
            / 1e6
        )
        t_a_pu = branch.temperature_ext_k / grid.t_ref_k
        t_pu_send = from_node_model.vars["t_pu"]
        return [
            branch.H_in_mw
            == branch.H_out_mw - vL_mw_per_k * grid.t_ref_k * (t_pu_send - t_a_pu),
        ]

    def equations(self, branch, grid, from_node_model, to_node_model, **kwargs):
        m_U = _branch_m_U(branch, grid)
        m_L = 0.0
        tpu_L, tpu_U = _t_pu_env_bounds(grid)

        scale_mw = C_WATER * grid.t_ref_k / 1e6

        t_pu_send = from_node_model.vars["t_pu"]
        m = branch.mass_flow_neg_kgs

        eqs = [
            branch.mass_flow_neg_kgs <= m_U * branch.on_off,
            branch.mass_flow_mag_kgs == branch.mass_flow_neg_kgs,
            branch.H_out_mw <= scale_mw * m_U * tpu_U * branch.on_off,
            branch.H_in_mw <= scale_mw * m_U * tpu_U * branch.on_off,
        ]
        eqs += self._heat_balance_eqs(branch, grid, from_node_model)

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
            # eq. 18b/18d/18h: piecewise McCormick over \tau partition.
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


class McCormickPassiveHeatExchangerFormulation(McCormickHeatBranchFormulation):
    r"""Passive HE in the McCormick H-space: free (unidirectional) mass flow,
    fixed ``q_mw`` deducted exactly from the enthalpy flow instead of the
    pipe's Taylor heat loss. Same algebra as the bilinear formulation's
    ``H_in = H_out - q_mw``, but the :math:`H = c \cdot m \cdot \tau` surface is McCormick-relaxed,
    so the branch stays LP/MILP."""

    def _heat_balance_eqs(self, branch, grid, from_node_model):
        return [branch.H_in_mw == branch.H_out_mw - branch.q_mw]


class McCormickHeatExchangerFormulation(FixedFlowHeatExchangerFormulation):
    r"""Active fixed-flow HE feeding the McCormick nodal heat balance.

    Adds :math:`H_{out,mw} = c \cdot m_{design} \cdot \tau_{send}` (linear - the flow is a constant) and
    ``H_in_mw = H_out_mw - q_delivered`` so the HE's extraction/injection shows
    up in eq. 9a; without these the McCormick junctions would not see the HE at
    all. ``direction`` is pinned to Const so no binary survives; ``on_off``
    switching of the HE flow itself is not modelled here (an off HE degrades to
    a zero-q pass-through at design flow)."""

    def ensure_var(self, model, simulation=False, grid=None):
        super().ensure_var(model, simulation, grid=grid)
        model.H_out_mw = Var(0, name="H_out_mw")
        model.H_in_mw = Var(0, name="H_in_mw")
        model.direction = Const(0)

    def equations(self, branch, grid, from_node_model, to_node_model, **kwargs):
        scale_mw = C_WATER * grid.t_ref_k / 1e6
        eqs = self._he_equations(branch, grid, from_node_model)
        eqs += [
            branch.H_out_mw == scale_mw * branch.mass_flow_design_kgs * branch.t_in_pu,
            branch.H_in_mw == branch.H_out_mw - branch.q_mw_delivered,
        ]
        return eqs


def mccormick_dhs_gap_bound_mw(branch, grid, num_partitions: int = 1) -> float:
    r"""Worst-case ``H_out_mw`` gap [MW]:
    :math:`(c \cdot t_{ref,k}/10^6) \cdot m_U \cdot (\tau_U - \tau_L) / (4 \cdot S)`."""
    tpu_L, tpu_U = _t_pu_env_bounds(grid)
    m_U = _branch_m_U(branch, grid)
    return C_WATER * grid.t_ref_k / 1e6 * m_U * (tpu_U - tpu_L) / (4 * num_partitions)


def mccormick_dhs_gap_bound_k(
    grid, num_partitions: int = 1, branch=None, mass_flow_kgs=None
) -> float:
    r"""Worst-case gap as a sender-temperature error [K]:
    :math:`t_{ref,k} \cdot (\tau_U - \tau_L)/(4 \cdot S)` at :math:`m = m_U`, scaling as ``m_U/m`` below."""
    tpu_L, tpu_U = _t_pu_env_bounds(grid)
    base_k = grid.t_ref_k * (tpu_U - tpu_L) / (4 * num_partitions)
    if mass_flow_kgs is None:
        return base_k
    if branch is None:
        raise ValueError("branch is required when mass_flow_kgs is provided")
    m_U = _branch_m_U(branch, grid)
    return base_k * m_U / mass_flow_kgs

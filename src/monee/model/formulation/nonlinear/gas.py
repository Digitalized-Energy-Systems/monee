import monee.model.phys.core.hydraulics as hydraulicsmodel
import monee.model.phys.nonlinear.gf as ogfmodel
from monee.model.core import Const, Intermediate, IntermediateEq, Var

from ..core import BranchFormulation, NodeFormulation


class NLWeymouthNodeFormulation(NodeFormulation):
    def ensure_var(self, model):
        model.pressure_pa = Intermediate(1000000)
        model.pressure_pu = Intermediate(1)
        model.pressure_squared_pu = Var(1, min=0, max=3, name="pressure_sq_pu")

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
            IntermediateEq(
                "pressure_pu", kwargs["sqrt_impl"](node.pressure_squared_pu)
            ),
            IntermediateEq("pressure_pa", lambda: node.pressure_pu * grid.pressure_ref),
        ]


class NLWeymouthBranchFormulation(BranchFormulation):
    # Coefficient on the ``ε · (m_pos_squared + m_neg_squared)`` tie-breaker
    # added to the global objective by :meth:`minimize`.  The epigraph
    # relaxation ``m² ≤ m_squared`` is convex; this penalty pulls
    # ``m_squared`` down to its lower bound ``m²`` so the relaxation stays
    # tight at the optimum.  ``1e-5`` empirically wins on simbench
    # LV-rural3 with McCormick S=20: small enough that the obj distortion
    # is well below the 1e-3 MIPGap, large enough that Gurobi tightens the
    # LP relaxation and avoids a huge B&B tree on the partition binaries.
    EPIGRAPH_TIGHTENING_EPS = 1e-5

    def ensure_var(self, model):
        # Pin friction to the asymptotic turbulent value (Swamee-Jain at
        # Re → ∞).  ``Const`` collapses at variable injection time, so the
        # branch's ``friction`` and ``reynolds`` fields disappear from the
        # LP entirely — eliminating the friction PWL (SOS2 lambdas), the
        # Reynolds equation, and the bilinear ``friction · m²`` in
        # Weymouth.  Trade-off: pressure drop is under-estimated on
        # lightly-loaded pipes (laminar regime, Re < 2300) — for typical
        # gas distribution Re ≳ 10⁵ so the asymptote is within a few %
        # of the true Colebrook root.
        f_const = hydraulicsmodel.friction_at_high_re(model.diameter_m, model.roughness)
        model.friction = Const(f_const)
        model.reynolds = Const(0.0)

    def minimize(self, branch, grid, from_node_model, to_node_model, **kwargs):
        # Tie-breaker that keeps the epigraph relaxation
        # ``m² ≤ m_squared`` (in :meth:`equations`) tight at the optimum.
        return [
            self.EPIGRAPH_TIGHTENING_EPS
            * (branch.mass_flow_pos_squared + branch.mass_flow_neg_squared)
        ]

    def equations(self, branch, grid, from_node_model, to_node_model, **kwargs):
        branch._pipe_area = hydraulicsmodel.calc_pipe_area(branch.diameter_m)

        # linearize sqrt(p) around nominal pressure
        p0 = grid.nominal_pressure_pu
        x0 = p0**2
        p_from = p0 + (1 / (2 * p0)) * (
            from_node_model.vars["pressure_squared_pu"] - x0
        )
        p_to = p0 + (1 / (2 * p0)) * (to_node_model.vars["pressure_squared_pu"] - x0)
        p_avg = 0.5 * (p_from + p_to)

        # Per-pipe big-M tightening: physical max flow is bounded by the
        # pipe cross-section times the velocity cap.  ``grid.f_max`` is a
        # grid-wide ceiling that is typically far above this physical limit,
        # which loosens Gurobi's LP relaxation on the direction / on_off
        # binaries.  Mirrors what ``NLDarcyWeisbachBranchFormulation`` does
        # for water pipes.  Gas density is computed from the ideal-gas law at
        # the grid's reference conditions; ``v_max_mps`` defaults to 20 m/s
        # (physical erosional-velocity ceiling for gas pipelines).
        gas_density = (
            grid.pressure_ref
            * grid.molar_mass
            / (grid.universal_gas_constant * grid.t_k)
        )
        f_max_local = min(
            grid.f_max,
            hydraulicsmodel.calc_max_mass_flow(
                branch.diameter_m,
                gas_density,
                getattr(grid, "v_max_mps", 20.0),
            ),
        )

        return [
            # Epigraph: ``m² ≤ m_squared`` is convex (m·m is convex,
            # subtract linear m_squared → still convex).  The tie-breaking
            # ε·m_squared term in :meth:`minimize` keeps it tight at the
            # optimum.  Replacing the equality removes 256 non-convex
            # constraints from the model so Gurobi's MISOCP detector can
            # take over the SOC inequalities (otherwise it falls back to
            # spatial branching for the whole problem).
            branch.mass_flow_pos * branch.mass_flow_pos <= branch.mass_flow_pos_squared,
            branch.mass_flow_neg * branch.mass_flow_neg <= branch.mass_flow_neg_squared,
            branch.mass_flow_pos_squared <= f_max_local**2 * branch.direction,
            branch.mass_flow_neg_squared <= f_max_local**2 * (1 - branch.direction),
            branch.mass_flow_pos_squared <= f_max_local**2 * branch.on_off,
            branch.mass_flow_neg_squared <= f_max_local**2 * branch.on_off,
            ogfmodel.pipe_weymouth(
                p_squared_i=from_node_model.vars["pressure_squared_pu"]
                * grid.pressure_ref**2,
                p_squared_j=to_node_model.vars["pressure_squared_pu"]
                * grid.pressure_ref**2,
                f_a_pos_sq=branch.mass_flow_pos_squared,
                f_a_neg_sq=branch.mass_flow_neg_squared,
                diameter_m=branch.diameter_m,
                length_m=branch.length_m,
                t_k=grid.t_k,
                compressibility=grid.compressibility,
                on_off=branch.on_off,
                friction=branch.friction,
                **kwargs,
            ),
            branch.gas_density
            == grid.pressure_ref
            * p_avg
            * grid.molar_mass
            / (grid.universal_gas_constant * grid.t_k),
        ]


class NLWeymouthPWLBranchFormulation(BranchFormulation):
    """Variable-friction Weymouth via per-direction PWL of φ(m) = friction(Re(m))·m².

    Opt-in alternative to :class:`NLWeymouthBranchFormulation` (which
    pins friction to its high-Reynolds asymptote).  Use this when you
    need accurate pressure drops in the **laminar regime** (``Re < 2300``,
    e.g. lightly-loaded distribution pipes) — the asymptotic-friction
    shortcut under-estimates pressure drop by a factor of 5–50× there,
    while this formulation captures the full Colebrook friction.

    The trick: replace the bilinear ``friction · m²`` by two PWLs of
    ``φ(m) = friction(Re(m)) · m²`` — one against ``mass_flow_pos`` and
    one against ``mass_flow_neg`` — and combine them with the same sign
    convention as :func:`ogfmodel.pipe_weymouth`:

        (p_i² − p_j²) · C² · on_off  ==  φ_neg  −  φ_pos

    Both the laminar (``φ ∝ m``) and turbulent (``φ ∝ m²``) regimes
    resolve cleanly with ~12 log-spaced breakpoints.  The ``direction``
    binary still gates which side carries flow (one of ``m_pos`` /
    ``m_neg`` is 0 and the corresponding ``φ`` collapses to 0 via the
    PWL's 0-anchor).  No bilinears, no quadratic equalities — just two
    SOS2 sets per pipe.
    """

    def __init__(self, n_breakpoints: int = 12):
        self.n_breakpoints = n_breakpoints

    def ensure_var(self, model):
        # φ aux Vars capturing friction · m² for each flow direction.
        model.phi_pwl_pos = Var(0, min=0, name="phi_pwl_pos")
        model.phi_pwl_neg = Var(0, min=0, name="phi_pwl_neg")
        # Squared-mass-flow Vars are unused once the PWL handles φ directly.
        model.mass_flow_pos_squared = Const(0.0)
        model.mass_flow_neg_squared = Const(0.0)
        # Reynolds and friction are now embedded in the PWL — no Vars needed.
        model.reynolds = Const(0.0)
        model.friction = Const(0.0)

    def equations(self, branch, grid, from_node_model, to_node_model, **kwargs):
        branch._pipe_area = hydraulicsmodel.calc_pipe_area(branch.diameter_m)

        # Linearise sqrt(p) around nominal pressure (same as the convex form).
        p0 = grid.nominal_pressure_pu
        x0 = p0**2
        p_from = p0 + (1 / (2 * p0)) * (
            from_node_model.vars["pressure_squared_pu"] - x0
        )
        p_to = p0 + (1 / (2 * p0)) * (to_node_model.vars["pressure_squared_pu"] - x0)
        p_avg = 0.5 * (p_from + p_to)

        gas_density = (
            grid.pressure_ref
            * grid.molar_mass
            / (grid.universal_gas_constant * grid.t_k)
        )
        m_max = min(
            grid.f_max,
            hydraulicsmodel.calc_max_mass_flow(
                branch.diameter_m,
                gas_density,
                getattr(grid, "v_max_mps", 20.0),
            ),
        )

        # Pyomo Piecewise requires both ends of the x-Var to be bounded.
        # Slightly slack margin (1.001 ×) avoids tightness against a PWL
        # endpoint.
        branch.mass_flow_pos.setub(m_max * 1.001)
        branch.mass_flow_neg.setub(m_max * 1.001)

        # Build the two φ(m) PWLs: one per flow direction.  The PWL has a
        # 0-anchor so the inactive direction's φ collapses to 0 cleanly.
        xs, ys = hydraulicsmodel.phi_pwl_breakpoints(
            branch.diameter_m,
            branch.roughness,
            grid.dynamic_visc,
            branch._pipe_area,
            m_max,
            self.n_breakpoints,
        )
        kwargs["pwl_impl"].piecewise_eq(
            y=branch.phi_pwl_pos,
            x=branch.mass_flow_pos,
            xs=xs,
            ys=ys,
        )
        kwargs["pwl_impl"].piecewise_eq(
            y=branch.phi_pwl_neg,
            x=branch.mass_flow_neg,
            xs=xs,
            ys=ys,
        )

        C_sq = ogfmodel.calc_C_squared(
            branch.diameter_m,
            branch.length_m,
            grid.t_k,
            grid.compressibility,
        )

        return [
            # Mass-flow direction gating, matching the original Weymouth
            # formulation.  ``direction = 0`` ⇒ forward flow via m_neg.
            branch.mass_flow_pos <= m_max * branch.direction,
            branch.mass_flow_neg <= m_max * (1 - branch.direction),
            branch.mass_flow_pos <= m_max * branch.on_off,
            branch.mass_flow_neg <= m_max * branch.on_off,
            # Weymouth pressure drop (sign matches ``ogfmodel.pipe_weymouth``):
            #   (p_i² − p_j²) · C² · on_off == friction · -(m_pos² - m_neg²)
            #                              == φ_neg − φ_pos
            (
                from_node_model.vars["pressure_squared_pu"]
                - to_node_model.vars["pressure_squared_pu"]
            )
            * grid.pressure_ref**2
            * C_sq
            * branch.on_off
            == branch.phi_pwl_neg - branch.phi_pwl_pos,
            branch.gas_density
            == grid.pressure_ref
            * p_avg
            * grid.molar_mass
            / (grid.universal_gas_constant * grid.t_k),
        ]

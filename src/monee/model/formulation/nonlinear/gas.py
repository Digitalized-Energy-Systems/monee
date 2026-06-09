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
    # Tie-breaker ε on m_squared keeps the convex epigraph
    # ``m² ≤ m_squared`` tight at the optimum. 1e-5: small vs the 1e-3 MIPGap,
    # large enough that Gurobi can tighten the LP relaxation.
    EPIGRAPH_TIGHTENING_EPS = 1e-5

    def ensure_var(self, model):
        # Pin friction to its turbulent asymptote (Swamee-Jain, Re→∞). Const
        # collapses at injection so friction / reynolds / friction-PWL / the
        # friction·m² bilinear all drop out. Under-estimates pressure drop for
        # Re < 2300; gas distribution typically runs Re ≳ 1e5.
        f_const = hydraulicsmodel.friction_at_high_re(model.diameter_m, model.roughness)
        model.friction = Const(f_const)
        model.reynolds = Const(0.0)

    def minimize(self, branch, grid, from_node_model, to_node_model, **kwargs):
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

        # Per-pipe big-M via π/4·D²·ρ·v_max (ideal-gas ρ at reference conditions,
        # v_max defaults to 20 m/s — erosional cap for gas pipelines). Mirrors
        # what NLDarcyWeisbachBranchFormulation does for water.
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
            # Convex epigraph ``m² ≤ m_squared`` (kept tight by ε in minimize());
            # avoids the non-convex equality so Gurobi's MISOCP detector kicks in.
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

    Opt-in alternative to :class:`NLWeymouthBranchFormulation` for laminar-heavy
    pipes (Re < 2300). Two PWLs (one per direction) replace the friction·m²
    bilinear; ``(p_i² - p_j²) · C² · on_off == φ_neg - φ_pos`` matches
    :func:`ogfmodel.pipe_weymouth`'s sign. ~12 log-spaced breakpoints suffice.
    """

    def __init__(self, n_breakpoints: int = 12):
        self.n_breakpoints = n_breakpoints

    def ensure_var(self, model):
        # φ = friction · m² per direction; replaces squared-mf / Reynolds / friction.
        model.phi_pwl_pos = Var(0, min=0, name="phi_pwl_pos")
        model.phi_pwl_neg = Var(0, min=0, name="phi_pwl_neg")
        model.mass_flow_pos_squared = Const(0.0)
        model.mass_flow_neg_squared = Const(0.0)
        model.reynolds = Const(0.0)
        model.friction = Const(0.0)

    def equations(self, branch, grid, from_node_model, to_node_model, **kwargs):
        branch._pipe_area = hydraulicsmodel.calc_pipe_area(branch.diameter_m)

        # Linearise sqrt(p) around nominal pressure.
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

        # Pyomo Piecewise requires bounded x; 1.001× slack avoids endpoint tightness.
        branch.mass_flow_pos.setub(m_max * 1.001)
        branch.mass_flow_neg.setub(m_max * 1.001)

        # Two φ(m) PWLs; 0-anchor collapses the inactive side's φ to 0.
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
            # direction=0 ⇒ forward flow via m_neg.
            branch.mass_flow_pos <= m_max * branch.direction,
            branch.mass_flow_neg <= m_max * (1 - branch.direction),
            branch.mass_flow_pos <= m_max * branch.on_off,
            branch.mass_flow_neg <= m_max * branch.on_off,
            # (p_i² - p_j²) · C² · on_off == φ_neg - φ_pos
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

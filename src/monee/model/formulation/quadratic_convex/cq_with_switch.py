import math
import numpy as np
import monee.model.phys.quadratic_convex.cq_with_switch as opfmodel
from ..core import BranchFormulation, NodeFormulation
from monee.model.core import Intermediate, IntermediateEq, Var
from monee.model.child import PowerGenerator
from monee.problem.core import Objectives, OptimizationProblem

# Paper equations use _pu power, while Monee gives branch flow variables in
# _mw / _mvar. The cq_with_switch helper therefore divides P/Q by grid.sn_mva.

SQRT_3 = np.sqrt(3)
CURRENT_SMOOTHING_EPS_MW = 1e-4 # constant for numerical smoothing

def _is_fixed_one(value): #to check if branch is active (on-off variable)
    return isinstance(value, (int, float, bool)) and float(value) == 1.0


def _get_vm_var(node_model): #read voltage flexible to storing method
    if hasattr(node_model, "vars") and "vm_pu" in node_model.vars: #stored inside vars?
        return node_model.vars["vm_pu"]
    if hasattr(node_model, "vm_pu"): #stored inside attribute
        return node_model.vm_pu
    raise ValueError("Could not find vm_pu variable on node model.")


def _get_v_sq_var(node_model): #read squared voltage variable flexible to storing method. returns lifted squared-voltage which is later connected to v^2
    if hasattr(node_model, "vars") and "vm_pu_squared" in node_model.vars:
        return node_model.vars["vm_pu_squared"]
    if hasattr(node_model, "vm_pu_squared"):
        return node_model.vm_pu_squared
    raise ValueError("Could not find vm_pu_squared variable on node model.")


def _voltage_bounds(node_model):
    """
    get v_i^l and v_i^u
    check that values exist .
    """
    v_min = getattr(node_model, "_qc_v_min", None)
    v_max = getattr(node_model, "_qc_v_max", None)

    if v_min is None or v_max is None:
        raise ValueError("QC voltage bounds were not cached in ensure_var().")
    return v_min, v_max

def _angle_bound(branch):
    """
    Paper active-line angle-difference bound theta^u (in radians).
    """
    theta_u = getattr(branch, "angmax", None) #first choice angmax value
    if theta_u is None: #otherwise check delta_max value
        theta_u = getattr(branch, "delta_max", None)

    if theta_u is None:
        import warnings
        warnings.warn(
            "No explicit value for theta_u (angmax/delta_max) given; "
            "defaulting to pi/6, per the paper's expected angle-difference "
            "range.",
            stacklevel=2,
        ) #default value based on paper expectation
        theta_u = math.pi / 6 #todo try to set to pi/36 -> accord to p.5 of paper
    theta_u = float(theta_u)

    if not (0 < theta_u <= math.pi / 2): #according to paper relaxation only valid for agles smaller that pi/2
        raise ValueError(
            "Paper QC trigonometric relaxations require "
            "0 < theta_u <= pi/2 radians."
        )

    return theta_u


def _big_m_angle_bound(branch, theta_u):
    """
    Paper: switched off branch voltage-angle difference bound. (theta_u is from active line, in switched off case upper bound is theta_u+theta_M and lower bound is -theta_u-theta_M)
    value needs to be provided
    theta^M = sum(theta_ij_u)
    """
    theta_M = getattr(branch, "big_m_theta", None)

    if theta_M is not None:
        theta_M = float(theta_M)

        if theta_M < theta_u:
            raise ValueError("big_m_theta must be >= theta_u." )

        return theta_M

    # Monee create_line() defaults to numeric on_off=1.
    # For a permanently active line, all (1-z)*theta_M terms vanish.
    if _is_fixed_one(getattr(branch, "on_off", 1)):
        return theta_u

    # Only switchable lines need explicit theta^M.
    raise ValueError( "Switchable QC branches require an explicit valid big_m_theta." )

def _switch_copy_relax(copy_var, original_var, on_off, lb, ub):
    """
    Exact convex hull of copy_var = z * original_var for binary z.
    preparation for Eq. (35) (requires participating on/off variables to be zero in the off state). The physical bounds lb/ub
    are the paper voltage bounds.
    """
    z = on_off
    return [
        copy_var >= lb * z,
        copy_var <= ub * z,
        copy_var >= original_var - ub * (1 - z),
        copy_var <= original_var - lb * (1 - z),
    ]


class QCElectricityNodeFormulation(NodeFormulation):
    def ensure_var( #to make sure variables for formulation are there
            self,
            node,
            simulation=False,
            grid=None,
            **kwargs,
    ):
        # Monee stores the actual voltage bounds on vm_pu.min / vm_pu.max.
        # Caching original vm_pu bounds may later be replaced by a solver variable. But needed for convexification
        if node.vm_pu.min is None or node.vm_pu.max is None:
            raise ValueError( "QC requires finite voltage magnitude bounds." )

        node._qc_v_min = float(node.vm_pu.min)
        node._qc_v_max = float(node.vm_pu.max)
    def equations( self, node, grid, from_branch_models,to_branch_models, connected_node_models, **kwargs ):
        v_min = node._qc_v_min
        v_max = node._qc_v_max

        return opfmodel.square_relax( v_sq_var=_get_v_sq_var(node),v_var=_get_vm_var(node), v_min=v_min, v_max=v_max ) #constraints vm_pu and v_welle_pu, does not calcualte them


class QCElectricityBranchFormulation(BranchFormulation):
    def ensure_var(self, branch, simulation=False, grid=None, **kwargs):
        theta_u = _angle_bound(branch) #active-line angle bound
        theta_M = _big_m_angle_bound(branch, theta_u) #angle bound relevant for switchable branches
        # not QC optimization variables, derived from solution later
        branch.i_from_ka = Intermediate(0)
        branch.i_to_ka = Intermediate(0)
        branch.loading_from_pu = Intermediate(0)
        branch.loading_to_pu = Intermediate(0)
        # QC variables
        branch.va_diff = Var(0, min=-theta_M, max=theta_M) #branch voltage-angle difference with bounds
        branch.cs = Var(1, min=0, max=1) #lifted cosine variable ~cos(delta_ij) (0 to 1 because delta -pi/2 to pi/2)
        s_max = math.sin(theta_u) #sine magnitude bounds
        branch.s = Var(0, min=-s_max, max=s_max) #lifted sine variable ~sin(delta_ij)
        branch.v_on_from = Var(1, min=0) # copies depending on switching state
        branch.v_on_to = Var(1, min=0)

        branch.vv = Var(1, min=0) #v_i*v_j
        branch.wc = Var(1, min=0) #v_i*v_j*cos(delta) ~vv*cs
        branch.ws = Var(0) #~vv*s -> can be pos or neg, no min value

        branch.v_sq_p_from = Var(1, min=0) #branch local, similar to lifted squared-voltage variables ~z*v_welle_i, used for swithced formulation
        branch.v_sq_p_to = Var(1, min=0)

        branch.i_qc = Var(0, min=0) # QC auxiliary l_ij from paper; NOT physical current in kA, optimization value
        #no optimization variables, just stored for later to branch
        branch.theta_u = theta_u
        branch.theta_M = theta_M

    def equations( self,branch, grid, from_node_model, to_node_model, **kwargs):
        sn_mva = float(grid.sn_mva)
        if sn_mva <= 0:
            raise ValueError("grid.sn_mva must be > 0.")

        z = branch.on_off
        vm_from = _get_vm_var(from_node_model)
        vm_to = _get_vm_var(to_node_model)
        v_sq_from = _get_v_sq_var(from_node_model) #lifted variables v_welle as approx of v^2
        v_sq_to = _get_v_sq_var(to_node_model) #lifted variable v_welle as approx of v^2
        v_from_min, v_from_max = _voltage_bounds(from_node_model)
        v_to_min, v_to_max = _voltage_bounds(to_node_model)
        theta_u = branch.theta_u
        theta_M = branch.theta_M

        # Active-state bounds McCormick relaxations
        vv_min = v_from_min * v_to_min
        vv_max = v_from_max * v_to_max
        cs_min = math.cos(theta_u)
        cs_max = 1.0
        s_min = -math.sin(theta_u)
        s_max = math.sin(theta_u)
        wc_min = vv_min * cs_min #needed for McCormick envelope equ.35
        wc_max = vv_max #needed for McCormick envelope equ.35
        ws_abs_max = vv_max * s_max #needed for McCormick envelope equ.35

        # Series admittance in p.u.
        impedance = branch.br_r_pu + 1j * branch.br_x_pu

        if abs(impedance) == 0:
            raise ValueError("Branch series impedance cannot be zero.")
        y = 1.0 / impedance #admittance
        g = float(np.real(y))
        b = float(np.imag(y))

        # Voltage-angle difference. already linear
        eqs = [ branch.va_diff == from_node_model.vars["va_radians"]  - to_node_model.vars["va_radians"], ]

        # Paper Eqs. (27)-(28): switched sine and cosine relaxations.
        eqs += opfmodel.cosine_relax( cs_var=branch.cs, delta_var=branch.va_diff, delta_max=theta_u, on_off=z, delta_big_m=theta_M)
        eqs += opfmodel.sine_relax( s_var=branch.s, delta_var=branch.va_diff,delta_max=theta_u, on_off=z, delta_big_m=theta_M )

        # On/off voltage-magnitude. on/off McCormick products are zero when branch is switched off. (equ.35)
        eqs += _switch_copy_relax( copy_var=branch.v_on_from, original_var=vm_from, on_off=z, lb=v_from_min, ub=v_from_max)
        eqs += _switch_copy_relax( copy_var=branch.v_on_to, original_var=vm_to, on_off=z,lb=v_to_min, ub=v_to_max )

        # Sequential McCormick products:
        # vv ~ v_i * v_j
        # wc ~ vv * cos(theta_i - theta_j)
        # ws ~ vv * sin(theta_i - theta_j)
        #
        # NOTE on vv's inputs: paper Eq. (16) defines the continuous vv_ij as
        # the McCormick product of the actual voltage MAGNITUDES v_i, v_j
        # (not the squared/lifted v-hat used for v_sq_p). Appendix A.2.1's
        # on/off vv_ij is the on/off extension of that same quantity, so its
        # two McCormick inputs must also be magnitude-perspective copies
        # (v_on_from / v_on_to, zeroed off-state via _switch_copy_relax) with
        # bounds [v_min, v_max] -- not the squared perspective copies used
        # for v_sq_p_from/to below, whose bounds are [v_min^2, v_max^2] and
        # which enter the power-flow equations directly, not through vv.
        eqs += opfmodel.mccormick_on_off_relax( product_var=branch.vv,
            x_var=branch.v_on_from,
            y_var=branch.v_on_to,
            on_off=z,
            x_lb=v_from_min,
            x_ub=v_from_max,
            y_lb=v_to_min,
            y_ub=v_to_max,
            product_lb=vv_min,
            product_ub=vv_max,
        )

        eqs += opfmodel.mccormick_on_off_relax(
            product_var=branch.wc,
            x_var=branch.vv,
            y_var=branch.cs,
            on_off=z,
            x_lb=vv_min,
            x_ub=vv_max,
            y_lb=cs_min,
            y_ub=cs_max,
            product_lb=wc_min,
            product_ub=wc_max,
        )

        eqs += opfmodel.mccormick_on_off_relax(
            product_var=branch.ws,
            x_var=branch.vv,
            y_var=branch.s,
            on_off=z,
            x_lb=vv_min,
            x_ub=vv_max,
            y_lb=s_min,
            y_ub=s_max,
            product_lb=-ws_abs_max,
            product_ub=ws_abs_max,
        )

        # Perspective squared voltage, paper Equ.(34).
        eqs += opfmodel.perspective_voltage_relax(
            v_sq_p_var=branch.v_sq_p_from,
            v_sq_var=v_sq_from,
            v_min=v_from_min,
            v_max=v_from_max,
            on_off=z,
        )

        eqs += opfmodel.perspective_voltage_relax(
            v_sq_p_var=branch.v_sq_p_to,
            v_sq_var=v_sq_to,
            v_min=v_to_min,
            v_max=v_to_max,
            on_off=z,
        )

        # Explicit zero-off bounds for perspective copies. (physical bus voltage doesn't become 0 for switched of line)
        eqs += [
            branch.v_sq_p_from >= z * v_from_min ** 2,
            branch.v_sq_p_from <= z * v_from_max ** 2,

            branch.v_sq_p_to >= z * v_to_min ** 2,
            branch.v_sq_p_to <= z * v_to_max ** 2,
        ]

        # Power-flow equations, Paper equations use p.u., Monee branch variables use MW / MVAr.
        eqs += [
            opfmodel.int_flow_from_p(
                p_from_var=branch.p_from_mw,
                v_sq_from_var=v_sq_from,
                v_sq_p_var=branch.v_sq_p_from,
                wc_var=branch.wc,
                ws_var=branch.ws,
                g_branch=g,
                b_branch=b,
                tap=branch.tap,
                shift=branch.shift,
                sn_mva=sn_mva,
                g_from=branch.g_fr_pu,
                on_off=z,
            ),

            opfmodel.int_flow_from_q(
                q_from_var=branch.q_from_mvar,
                v_sq_from_var=v_sq_from,
                v_sq_p_var=branch.v_sq_p_from,
                wc_var=branch.wc,
                ws_var=branch.ws,
                g_branch=g,
                b_branch=b,
                tap=branch.tap,
                shift=branch.shift,
                sn_mva=sn_mva,
                b_from=branch.b_fr_pu,
                on_off=z,
            ),

            opfmodel.int_flow_to_p(
                p_to_var=branch.p_to_mw,
                v_sq_to_var=v_sq_to,
                v_sq_p_var=branch.v_sq_p_to,
                wc_var=branch.wc,
                ws_var=branch.ws,
                g_branch=g,
                b_branch=b,
                tap=branch.tap,
                shift=branch.shift,
                sn_mva=sn_mva,
                g_to=branch.g_to_pu,
                on_off=z,
            ),

            opfmodel.int_flow_to_q(
                q_to_var=branch.q_to_mvar,
                v_sq_to_var=v_sq_to,
                v_sq_p_var=branch.v_sq_p_to,
                wc_var=branch.wc,
                ws_var=branch.ws,
                g_branch=g,
                b_branch=b,
                tap=branch.tap,
                shift=branch.shift,
                sn_mva=sn_mva,
                b_to=branch.b_to_pu,
                on_off=z,
            ),
        ]

        #  current, same as AC

        i_from_ka = (
                (
                        branch.p_from_mw ** 2
                        + branch.q_from_mvar ** 2
                        + CURRENT_SMOOTHING_EPS_MW ** 2
                )
                ** 0.5
                / (
                        vm_from
                        * from_node_model.vars["base_kv"]
                )
                / SQRT_3
        )

        i_to_ka = (
                (
                        branch.p_to_mw ** 2
                        + branch.q_to_mvar ** 2
                        + CURRENT_SMOOTHING_EPS_MW ** 2
                )
                ** 0.5
                / (
                        vm_to
                        * to_node_model.vars["base_kv"]
                )
                / SQRT_3
        )

        eqs += [
            IntermediateEq(
                "i_from_ka",
                i_from_ka,
            ),
            IntermediateEq(
                "i_to_ka",
                i_to_ka,
            ),
            IntermediateEq(
                "loading_from_pu",
                i_from_ka / branch.max_i_ka,
            ),
            IntermediateEq(
                "loading_to_pu",
                i_to_ka
                * to_node_model.vars["base_kv"]
                / (
                        from_node_model.vars["base_kv"]
                        * branch.max_i_ka
                ),
            ),
        ]

        # Current-magnitude strengthening. no thermal-limit-dependent constraints
        #   Equ. (21): v_i^2 * l_ij >= p_ij^2 + q_ij^2
        #   Equ. (23): relationship between l_ij and lifted voltages
        # Equs. (30)-(31) are deliberately NOT included here: they are an
        # optional extra tightening the paper presents as "another valid
        # constraint" (Sect. 5.2.4), not part of its actual QC-OTS
        # constraint set (Appendix A.2.1 lists only (3)-(7), (16), (21),
        # (23), (32)-(34)). They remain available as
        # opfmodel.current_switch_relax for callers who want them.
        g_fr = float(getattr(branch, "g_fr_pu", 0.0))
        g_to = float(getattr(branch, "g_to_pu", 0.0))
        b_fr = float(getattr(branch, "b_fr_pu", 0.0))
        b_to = float(getattr(branch, "b_to_pu", 0.0))

        # tolerance; not a paper bound but set
        tol = 1e-12
        plain_branch = (
                abs(float(branch.tap) - 1.0) <= tol
                and abs(float(branch.shift)) <= tol
                and abs(g_fr) <= tol
                and abs(g_to) <= tol
                and abs(b_fr) <= tol
                and abs(b_to) <= tol
        )

        if plain_branch:
            # Paper Eq. (23).
            # Perspective squared-voltage copies used -> inactive branch remains consistent with l = wc = 0
            eqs.append(
                opfmodel.current_flow_equation(
                    i_var=branch.i_qc,
                    v_sq_from_var=branch.v_sq_p_from,
                    v_sq_to_var=branch.v_sq_p_to,
                    wc_var=branch.wc,
                    g_branch=g,
                    b_branch=b,
                )
            )

            # Paper Eq. (21).
            eqs.append(
                opfmodel.current_soc_relax(
                    p_var=branch.p_from_mw,
                    q_var=branch.q_from_mvar,
                    v_sq_var=v_sq_from,
                    i_var=branch.i_qc,
                    sn_mva=sn_mva,
                    tap=1.0,
                )
            )

        elif _is_fixed_one(z):

            # Active transformer / line-charging branch: Appendix-C Eqs. (48)-(49)
            if abs(g_fr) > tol or abs(g_to) > tol:
                raise ValueError(
                    "Paper Eq. (49) does not cover nonzero end conductance "
                    "shunts g_fr/g_to in the form implemented here."
                )

            if abs(b_fr - b_to) > tol:
                raise ValueError(
                    "Paper Appendix C assumes one total line-charging "
                    "susceptance split between both ends."
                )

            b_charge = b_fr + b_to

            # Paper Eq. (49).
            eqs.append(
                opfmodel.current_flow_equation_extended(
                    i_var=branch.i_qc,
                    v_sq_from_var=v_sq_from,
                    v_sq_to_var=v_sq_to,
                    wc_var=branch.wc,
                    ws_var=branch.ws,
                    q_from_var=branch.q_from_mvar,
                    g_branch=g,
                    b_branch=b,
                    tap=branch.tap,
                    shift=branch.shift,
                    b_charge=b_charge,
                    sn_mva=sn_mva,
                )
            )

            # Paper Eq. (48).
            eqs.append(
                opfmodel.current_soc_relax(
                    p_var=branch.p_from_mw,
                    q_var=branch.q_from_mvar,
                    v_sq_var=v_sq_from,
                    i_var=branch.i_qc,
                    sn_mva=sn_mva,
                    tap=branch.tap,
                )
            )

        else:
            # Switchable branch (on_off not fixed to 1) with a non-unity
            # tap, non-zero phase shift, or nonzero end shunts. The paper
            # gives no on/off extension of the transformer/line-charging
            # current relation (48)-(49) -- those are derived only for the
            # permanently-active case (Appendix C). Previously this branch
            # silently added no current-magnitude constraint at all, which
            # quietly weakened the relaxation with no warning. Fail loudly
            # instead: either fix on_off=1 for this branch, or give it
            # zero tap/shift/shunts so it falls into the plain (21)/(23)
            # case above, to use this QC formulation on it.
            raise NotImplementedError(
                "Switchable QC branches with a non-unity tap, nonzero "
                "phase shift, or nonzero end shunts (g_fr/g_to/b_fr/b_to) "
                "are outside the scope of the paper's current-magnitude "
                "relaxation: Eqs. (48)-(49) (Appendix C) are derived only "
                "for permanently-active branches (on_off fixed to 1), and "
                "the paper provides no on/off extension of them. Fix this "
                "branch's on_off to 1, or zero out its tap/shift/shunts, "
                "to use the QC relaxation on it."
            )

        return eqs


def create_qc_opf_problem(
    pmin_default=0.0,
    pmax_default=None,
    debug=False,
):
    """
    Builds the paper's generator fuel-cost objective (Eq. 11) as a Monee
    OptimizationProblem, to be solved together with QCElectricityNodeFormulation
    / QCElectricityBranchFormulation (Eqs. 12-23/27-35/48-49 above), giving a
    complete convex QCQP relaxation of AC-OPF:

        min  sum_i c2_i*(pg_i)^2 + c1_i*pg_i + c0_i      (Eq. 11)
        s.t. QC-relaxed power-flow / voltage / current equations (Sect. 4-5)
             pg_i^l <= pg_i <= pg_i^u                     (Eq. 6)

    Objective/physics are wired separately on purpose: Monee keeps the
    objective as a monee.problem.core.Objectives selection over child
    models (here, PowerGenerator), independent of the Node/BranchFormulation
    machinery that carries the physics -- the same pattern used by Monee's
    built-in create_economic_dispatch_problem (which only implements the
    linear special case, Eq. 10).

    Set each generator's cost coefficients directly, e.g.:
        gen.model.cost_c0 = 0.0
        gen.model.cost_c1 = 20.0
        gen.model.cost_c2 = 0.05
    Generators without these attributes default to c0=0, c1=1, c2=0, i.e.
    Eq. (10): plain active-power (loss) minimization.

    pmax_default, if given, uniformly bounds every PowerGenerator's physical
    output to [pmin_default, pmax_default] (Eq. 6). Leave it None to keep
    each generator's existing bounds and only set them for generators you
    bound individually (e.g. via problem.bounds(...) after calling this).

    Sign convention: Monee stores PowerGenerator.p_mw negative for
    generation (load convention), so the objective and any bounds use
    -model.p_mw to recover the paper's physical p^g_i >= 0.
    """
    problem = OptimizationProblem(debug=debug)

    # Paper Eq. (6): pg^l <= pg <= pg^u -- promote p_mw to a free decision var.
    problem.controllable_generators(["p_mw"])

    if pmax_default is not None:
        problem.bounds(
            (-pmax_default, -pmin_default),
            lambda model, grid: isinstance(model, PowerGenerator),
            ["p_mw"],
        )

    objectives = Objectives()

    # Paper Eq. (11).
    objectives.select(lambda m: isinstance(m, PowerGenerator)).calculate(
        lambda models: sum(
            opfmodel.generation_cost(
                -m.p_mw,
                c0=getattr(m, "cost_c0", 0.0),
                c1=getattr(m, "cost_c1", 1.0),
                c2=getattr(m, "cost_c2", 0.0),
            )
            for m in models
        )
    )

    problem.objectives = objectives
    return problem
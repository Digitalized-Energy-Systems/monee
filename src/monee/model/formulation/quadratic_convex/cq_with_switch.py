"""QC relaxation of the AC power flow:
Implementation according to Hijazi, Coffrin, and Van Hentenryck, "Convex quadratic relaxations for mixed-integer
nonlinear programs in power systems", Math. Prog. Comp. 9 (2017), https://doi.org/10.1007/s12532-016-0112-z.
Equation numbers refer to this paper; the equations themselves are in the physics module cq_with_switch.

Each nonlinear ter gets a lifted variable that is only bounded by convex envelopes, never set equal to the term.
v^2 -> v_sq (v-tilde),
cos / sin(delta) -> cs / s,
v_i * v_j -> vv,
v_i * v_j * cos / sin(delta) -> wc / ws.
The flow equations are linear in the lifted variables.

The paper's equations are in p.u., monee's flow variables are in MW/ MVAr, conversion with grid.sn_mva in the physics module.
"""

import math
import numpy as np
import monee.model.phys.quadratic_convex.cq_with_switch as opfmodel
from ..core import BranchFormulation, NodeFormulation
from monee.model.core import Intermediate, IntermediateEq, Var
from monee.model.child import PowerGenerator
from monee.problem.core import Objectives, OptimizationProblem
from ... import ExtPowerGrid

# Paper equations use _pu power, while Monee gives branch flow variables in
# _mw / _mvar. The cq_with_switch helper therefore divides P/Q by grid.sn_mva.

SQRT_3 = np.sqrt(3)
CURRENT_SMOOTHING_EPS_MW = 1e-4 # constant for numerical smoothing

def _is_fixed_one(value):
    """Return True only for a plain number equal to 1 (line permanently on).
    Solver variables must not reach ==, which would build a constraint
    instead of a bool.
    """
    return isinstance(value, (int, float, bool)) and float(value) == 1.0


def _get_vm_var(node_model):
    """Return the voltage magnitude variable vm_pu of a node model."""
    if hasattr(node_model, "vars") and "vm_pu" in node_model.vars:
        return node_model.vars["vm_pu"]
    if hasattr(node_model, "vm_pu"):
        return node_model.vm_pu
    raise ValueError("Could not find vm_pu variable on node model.")


def _get_v_sq_var(node_model):
    """Return the lifted squared voltage v-tilde (vm_pu_squared) of a node model."""
    if hasattr(node_model, "vars") and "vm_pu_squared" in node_model.vars:
        return node_model.vars["vm_pu_squared"]
    if hasattr(node_model, "vm_pu_squared"):
        return node_model.vm_pu_squared
    raise ValueError("Could not find vm_pu_squared variable on node model.")


def _voltage_bounds(node_model):
    """Return the voltage bounds (v^l, v^u) of a bus, Eq. (5).
    """
    v_min = getattr(node_model, "_qc_v_min", None)
    v_max = getattr(node_model, "_qc_v_max", None)

    if v_min is None or v_max is None:
        raise ValueError("QC voltage bounds were not cached in ensure_var().")
    return v_min, v_max

def _angle_bound(branch):
    """Return theta^u, the angle-difference limit of an active line, Eq. (9), in radians.
    Taken from branch.angmax or branch.delta_max; defaults to pi/2, the largest
    value for which the trigonometric relaxations are valid.
    """
    theta_u = getattr(branch, "angmax", None)
    if theta_u is None:
        theta_u = getattr(branch, "delta_max", None)

    if theta_u is None:
        import warnings
        warnings.warn(
            "No explicit value for theta_u (angmax/delta_max) given; defaulting to pi/2, the largest value the "
            "QC relaxation allows. A tighter bound gives a tighter relaxation (the paper mentions pi/36).",
            stacklevel=2)
        theta_u = math.pi / 2
    theta_u = float(theta_u)

    if not (0 < theta_u <= math.pi / 2):
        raise ValueError("Paper QC trigonometric relaxations require 0 < theta_u <= pi/2 radians.")

    return theta_u


def _big_m_angle_bound(branch, theta_u):
    """Return theta^M, the angle-difference limit of an open line (Sect. 5.2).

    The angle box is +-(z * theta^u + (1 - z) * theta^M), so an open line may take any |delta| <= theta^M. A valid choice is the sum of all theta^u.
    Switchable branches must provide it as branch.big_m_theta.
    For a permanently-on line all (1 - z) terms vanish and theta_u serves as placeholder.
    """
    theta_M = getattr(branch, "big_m_theta", None)

    if theta_M is not None:
        theta_M = float(theta_M)

        if theta_M < theta_u:
            raise ValueError("big_m_theta must be >= theta_u.")

        return theta_M

    # Monee's create_line() sets on_off to the plain number 1.
    if _is_fixed_one(getattr(branch, "on_off", 1)):
        return theta_u

    raise ValueError("Switchable QC branches require an explicit valid big_m_theta.")


def _switch_copy_relax(copy_var, original_var, on_off, lb, ub):
    """Switched copy of a bus variable: convex hull of copy = z * original.
    The copy equals original for z = 1 and is 0 for z = 0. Needed because the on/off McCormick envelope (Eq. 35) requires
    its arguments to be zero for an open line, which a bus voltage never is. Not an equation of the paper, see the comment
    on vv in QCElectricityBranchFormulation.equations.
    lb and ub are the bounds of original (the voltage bounds).
    """
    z = on_off
    return [
        copy_var >= lb * z,
        copy_var <= ub * z,
        copy_var >= original_var - ub * (1 - z),
        copy_var <= original_var - lb * (1 - z),
    ]

class QCElectricityNodeFormulation(NodeFormulation):
    """Voltage relaxation of a bus: v_sq bounded by the envelope of v^2, Eq. (16).
    The nodal power balance comes from Bus.equations(), not from this class.
    """
    def ensure_var(
            self,
            node,
            simulation=False,
            grid=None,
            **kwargs,
    ):
        """Cache the voltage bounds (v^l, v^u) of the bus as floats, Eq. (5).
        vm_pu is replaced by a solver variable at solve time, so the bounds are stored as constants for now.
        """
        if node.vm_pu.min is None or node.vm_pu.max is None:
            raise ValueError( "QC requires finite voltage magnitude bounds." )
        node._qc_v_min = float(node.vm_pu.min)
        node._qc_v_max = float(node.vm_pu.max)

    def equations( self, node, grid, from_branch_models,to_branch_models, connected_node_models, **kwargs ):
        """Return the envelope relating vm_pu and vm_pu_squared, Eq. (16)."""
        v_min = node._qc_v_min
        v_max = node._qc_v_max

        return opfmodel.square_relax( v_sq_var=_get_v_sq_var(node),v_var=_get_vm_var(node), v_min=v_min, v_max=v_max )

class QCElectricityBranchFormulation(BranchFormulation):
    """QC relaxation of a branch, switchable with on_off = z.

        Per branch: the lifted variables cs, s (Eqs. 14, 15; switched 28, 27),
        vv, wc, ws (Eqs. 17-19; switched 35), the perspective copies v_sq_p
        (Eq. 34) and the current l = i_qc (Eqs. 21, 23; 48, 49). The objective is
        the ohmic loss, see minimize().

        Limitations:
        - Switchable branches with tap, shift or shunts raise NotImplementedError:
          the paper gives no on/off form of Eqs. (48)-(49).
        - The thermal-limit strengthenings of the paper's QC-OTS, Eqs. (29)-(31),
          are not included.
        """
    def ensure_var(self, branch, simulation=False, grid=None, **kwargs):
        """Declare the lifted variables and the angle constants for equations()."""
        theta_u = _angle_bound(branch)
        theta_M = _big_m_angle_bound(branch, theta_u)

        # Reported after the solve, not part of the relaxation.
        branch.i_from_ka = Intermediate(0)
        branch.i_to_ka = Intermediate(0)
        branch.loading_from_pu = Intermediate(0)
        branch.loading_to_pu = Intermediate(0)

        # Lifted variables, see the class docstring.
        branch.va_diff = Var(0, min=-theta_M, max=theta_M)  # theta_i - theta_j
        # ~cos(delta), 0 to 1 because |delta| <= pi/2
        branch.cs = Var(1, min=0, max=1)
        s_max = math.sin(theta_u)
        branch.s = Var(0, min=-s_max, max=s_max)  # ~sin(delta)
        # switched copies of the bus voltages (0 for an open line)
        branch.v_on_from = Var(1, min=0)
        branch.v_on_to = Var(1, min=0)

        branch.vv = Var(1, min=0)  # ~v_i * v_j
        branch.wc = Var(1, min=0)  # ~vv * cs
        branch.ws = Var(0)  # ~vv * s, either sign

        # ~z * v_sq of the bus, Eq. (34)
        branch.v_sq_p_from = Var(1, min=0)
        branch.v_sq_p_to = Var(1, min=0)

        # l_ij of Eqs. (21), (23): squared current in p.u., not the reported kA
        branch.i_qc = Var(0, min=0)

        # Constants for equations().
        branch.theta_u = theta_u
        branch.theta_M = theta_M

    def minimize(self, branch, grid, from_node_model, to_node_model, **kwargs):
        """Return the ohmic loss r * l of the branch, in p.u.
        The relaxed feasible set has many points; this term selects the one with minimal loss.
        For a plain branch r * l = p_from + p_to (Prop. 4), so the sum is the network loss, i.e. objective (10) for fixed loads.
        Pushing the loss down drives wc, ws and the current cone toward the AC solution, but the relaxation is not guaranteed to be exact.
        """
        return [branch.i_qc * branch.br_r_pu]

    def equations( self,branch, grid, from_node_model, to_node_model, **kwargs):
        """Return the QC constraints of the branch (Sect. 4, 5, Appendix C)."""
        sn_mva = float(grid.sn_mva)
        if sn_mva <= 0:
            raise ValueError("grid.sn_mva must be > 0.")

        z = branch.on_off
        vm_from = _get_vm_var(from_node_model)
        vm_to = _get_vm_var(to_node_model)
        v_sq_from = _get_v_sq_var(from_node_model)
        v_sq_to = _get_v_sq_var(to_node_model)
        v_from_min, v_from_max = _voltage_bounds(from_node_model)
        v_to_min, v_to_max = _voltage_bounds(to_node_model)
        theta_u = branch.theta_u
        theta_M = branch.theta_M

        # Bounds of the lifted products for an active line (z = 1), used as
        # McCormick bounds below.
        vv_min = v_from_min * v_to_min
        vv_max = v_from_max * v_to_max
        cs_min = math.cos(theta_u)
        cs_max = 1.0
        s_min = -math.sin(theta_u)
        s_max = math.sin(theta_u)
        wc_min = vv_min * cs_min
        wc_max = vv_max
        ws_abs_max = vv_max * s_max

        # Series admittance y = g + jb in p.u.
        impedance = branch.br_r_pu + 1j * branch.br_x_pu

        if abs(impedance) == 0:
            raise ValueError("Branch series impedance cannot be zero.")
        y = 1.0 / impedance
        g = float(np.real(y))
        b = float(np.imag(y))

        # The only equality linking the bus angles to the model. The sine and
        # cosine envelopes bound cs and s given va_diff, but do not pin it to
        # the physics: use atan2(ws, wc) as the branch angle.
        eqs = [
            branch.va_diff
            == from_node_model.vars["va_radians"] - to_node_model.vars["va_radians"],
        ]

        # Sine and cosine relaxations, switched forms Eqs. (27), (28).
        eqs += opfmodel.cosine_relax(
            cs_var=branch.cs,
            delta_var=branch.va_diff,
            delta_max=theta_u,
            on_off=z,
            delta_big_m=theta_M,
        )
        eqs += opfmodel.sine_relax(
            s_var=branch.s,
            delta_var=branch.va_diff,
            delta_max=theta_u,
            on_off=z,
            delta_big_m=theta_M,
        )

        # Switched copies of the bus voltages (0 for an open line), the
        # arguments of vv.
        eqs += _switch_copy_relax(
            copy_var=branch.v_on_from,
            original_var=vm_from,
            on_off=z,
            lb=v_from_min,
            ub=v_from_max,
        )
        eqs += _switch_copy_relax(
            copy_var=branch.v_on_to,
            original_var=vm_to,
            on_off=z,
            lb=v_to_min,
            ub=v_to_max,
        )

        # Sequential McCormick products, Eqs. (17)-(19); on/off form Eq. (35):
        #   vv ~ v_i * v_j,  wc ~ vv * cs,  ws ~ vv * s
        # vv takes the switched magnitude copies (bounds [v_min, v_max]), so
        # that it equals Eq. (17) for z = 1. Appendix A.2.1 prints the squared
        # copies v_sq_p as arguments, which would give v_i^2 * v_j^2 for z = 1;
        # the magnitude copies are used as the assumed intent.
        eqs += opfmodel.mccormick_on_off_relax(
            product_var=branch.vv,
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

        # Perspective copies of the squared bus voltages, Eq. (34).
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

        # Not part of Eq. (34), which alone leaves v_sq_p free in
        # [v_sq - v_max^2, v_sq - v_min^2] for z = 0. These bounds force
        # v_sq_p = 0 for an open line (the bus voltage itself stays nonzero)
        # and give, together with Eq. (34), the convex hull. The paper gets
        # them from the on/off McCormick on v_sq_p; vv uses magnitude copies
        # here, so they are added explicitly.
        eqs += [
            branch.v_sq_p_from >= z * v_from_min ** 2,
            branch.v_sq_p_from <= z * v_from_max ** 2,
            branch.v_sq_p_to >= z * v_to_min ** 2,
            branch.v_sq_p_to <= z * v_to_max ** 2,
        ]

        # Flow equations (12)-(13); with tap and shift (42)-(45).
        # The to-side calls get v_sq_from_var, i.e. Eqs. (43) and (45) as
        # printed, with v_i^2. Physically the to-side flow contains v_j^2:
        # with v_i^2 a line can have negative losses (see int_flow_to_p).
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
                v_sq_from_var=branch.v_sq_p_from,  # v_i^2 as printed
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
                v_sq_from_var=branch.v_sq_p_from,  # v_i^2 as printed
            ),
        ]

        # Reported current in kA from P, Q and V, as in the AC formulation.
        i_from_ka = (
                (branch.p_from_mw ** 2 + branch.q_from_mvar ** 2 + CURRENT_SMOOTHING_EPS_MW ** 2)
                ** 0.5
                / (vm_from * from_node_model.vars["base_kv"])
                / SQRT_3
        )

        i_to_ka = (
                (branch.p_to_mw ** 2 + branch.q_to_mvar ** 2 + CURRENT_SMOOTHING_EPS_MW ** 2)
                ** 0.5
                / (vm_to * to_node_model.vars["base_kv"])
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
            # max_i_ka refers to the from-side voltage, as in the AC formulation
            IntermediateEq(
                "loading_to_pu",
                i_to_ka
                * to_node_model.vars["base_kv"]
                / (from_node_model.vars["base_kv"] * branch.max_i_ka),
            ),
        ]

        # Current l: Eqs. (23), (21) for plain branches, Eqs. (49), (48) for
        # transformer / charging branches. The thermal-limit strengthenings of
        # the strong QC-OTS, Eqs. (29)-(31) and l <= z * l^u, are not included:
        # they need a thermal limit t_ij per branch, which is not available here.
        g_fr = float(getattr(branch, "g_fr_pu", 0.0))
        g_to = float(getattr(branch, "g_to_pu", 0.0))
        b_fr = float(getattr(branch, "b_fr_pu", 0.0))
        b_to = float(getattr(branch, "b_to_pu", 0.0))

        # Numerical tolerance (not from the paper) for classifying a branch as
        # plain: tap = 1, shift = 0, no shunts. Unit conversions can leave
        # 1e-17 instead of exactly 0. It only selects the current relation; the
        # flow equations always use the exact parameters.
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
            # Eq. (23) on the perspective copies, so that l = 0 for an open line.
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
            # Eq. (21) with the bus v_sq: for an open line p = q = l = 0, so the
            # cone reads 0 >= 0.
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
            # Permanently active transformer / line-charging branch, Appendix C.
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

            # Eq. (49)
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

            # Eq. (48)
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
            raise NotImplementedError(
                "Switchable QC branches with a non-unity tap, nonzero phase shift, or nonzero end shunts (g_fr/g_to/b_fr/b_to) "
                "are outside the scope of current-magnitude relaxation: Equations for QC formulation are derived only "
                "for permanently-active branches (on_off fixed to 1), and no on/off extension of them is yet provided. Fix this "
                "branch's on_off to 1, or zero out its tap/shift/shunts, to use the QC relaxation on it."
            )
        return eqs


def create_qc_opf_problem( pmin_default=0.0, pmax_default=None, debug=False):
    """Return the OPF problem with the generation-cost objective, Eq. (11).

    To be solved with the QC node and branch formulations. Minimizes
    c2 * p^2 + c1 * p + c0 summed over all PowerGenerators and ExtPowerGrids,
    with p the supplied power (Monee stores supply as negative p_mw).
    Coefficients are read from the models as cost_c0, cost_c1 and cost_c2
    (defaults 0, 1, 0, i.e. Eq. 10).

    The output of the PowerGenerators is a decision variable, bounded to
    [pmin_default, pmax_default] (Eq. 6) if pmax_default is given; the external
    grid is not bounded. If pmax_default is not given, default bounds apply.
    """
    problem = OptimizationProblem(debug=debug)
    # Eq. (6): free the output as decision variable and bound it to
    # [p_min, p_max]; Monee stores generation as negative p_mw.
    # free generator output as decision variable
    problem.controllable_generators(["p_mw"])
    if pmax_default is not None:
        problem.bounds(
            (-pmax_default, -pmin_default),
            lambda model, grid: isinstance(model, PowerGenerator),
            ["p_mw"],
        )
    # Eq. (11), including the external grid as the generator at the slack bus.
    objectives = Objectives()
    objectives.select(lambda m: isinstance(m, (PowerGenerator, ExtPowerGrid))).calculate(
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
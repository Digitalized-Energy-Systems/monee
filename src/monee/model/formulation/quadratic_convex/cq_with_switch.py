"""QC (quadratic-convex) relaxation of AC power flow, with branch switching.

Bus-injection form: the flow equations are linear in the lifted variables
``w_i = v_i^2``, ``wc = v_i v_j cos(theta_ij)`` and ``ws = v_i v_j sin(theta_ij)``,
which are tied back to ``v`` and ``theta`` by convex envelopes.

Accuracy note - the amplification that makes bound tightness matter here.
``p_from = g*(w_i - wc) - b*ws`` is a *difference of two near-equal quantities*
scaled by the branch conductance. On a distribution feeder ``g = r/|z|^2`` is
O(1e3) per unit, so a relaxation slack ``eps`` on ``wc`` shows up as ``g*eps``
MW of phantom flow: 1e-3 of slack is already several MW. Two consequences,
both handled below:

* the relaxation needs an objective term that pushes it tight - :meth:`minimize`
  contributes the ``r * l`` loss term, exactly as the branch-flow MISOCP
  formulation does. Without it a feasibility solve (or an OPF whose objective
  does not price losses, e.g. load shedding) returns an arbitrary point of the
  relaxed set, which is off by orders of magnitude.
* the trigonometric envelope bounds are taken at their physical range
  (``cs in [cos(theta_u), 1]``, ``s in [-sin(theta_u), sin(theta_u)]``), widened
  to the switching-compatible range only on branches that can actually open -
  where ``cs`` has to be able to reach 0.

The voltage band is the one bound deliberately *not* tightened by default: see
:func:`_v_bounds` - it is a hard constraint, not a relaxation parameter, so the
default tracks the grid's own limits and tightening is opt-in.
"""

import math

import numpy as np

import monee.model.phys.quadratic_convex.cq_with_switch as opfmodel
from monee.model.core import Intermediate, IntermediateEq, PostProcess, Var

from ..core import BranchFormulation, NodeFormulation

SQRT_3 = np.sqrt(3)

# Smoothing scale [MW] for the current-magnitude sqrt, as in nlp/el.py. Without
# it the Jacobian of \sqrt{p^2+q^2} is singular at exactly zero flow - which is
# precisely where an opened branch sits, so an on_off=0 solve dies with
# Invalid_Number_Detected.
CURRENT_SMOOTHING_EPS_MW = 1e-4

# Last-resort envelope band, used only when neither the node nor the grid
# carries voltage limits.
DEFAULT_V_MIN = 0.9
DEFAULT_V_MAX = 1.1


def _v_bounds(node, grid=None, override=(None, None)):
    """Envelope voltage band: explicit *override*, else node
    ``min_vm_pu``/``max_vm_pu``, else the grid's ``vm_pu_min``/``vm_pu_max``,
    else the module default.

    The band is not cosmetic and it is not only a relaxation parameter - it is a
    *hard constraint*. ``square_relax`` pairs ``w >= v^2`` with the chord
    ``w <= (v_max + v_min) v - v_max v_min``, and the two are jointly satisfiable
    only for ``v in [v_min, v_max]``. So the QC feasible set is the AC one
    INTERSECTED with this band: narrow it below the true solution and the model
    goes *infeasible*, it does not merely lose accuracy.

    Hence the default is the grid's own limits (``PowerGrid``: 0.5/1.5), which
    can never cut off a solution the AC NLP would find. Tightening is opt-in,
    via ``min_vm_pu``/``max_vm_pu`` on the buses or ``v_min``/``v_max`` on the
    formulation. It buys accuracy quadratically - every envelope gap scales with
    ``(v_max - v_min)^2``, so 0.9/1.1 is 25x tighter than 0.5/1.5 - but only on
    the ``vm_pu`` lifting variable: ``vm_pu_squared`` (and everything derived
    from it: flows, losses, currents) is already exact at the default band once
    the :meth:`~QCElectricityBranchFormulation.minimize` loss term is in play.
    """
    v_min, v_max = override
    if v_min is None:
        v_min = getattr(node, "min_vm_pu", getattr(node, "v_min", None))
    if v_max is None:
        v_max = getattr(node, "max_vm_pu", getattr(node, "v_max", None))
    if v_min is None and grid is not None:
        v_min = getattr(grid, "vm_pu_min", None)
    if v_max is None and grid is not None:
        v_max = getattr(grid, "vm_pu_max", None)
    return (
        DEFAULT_V_MIN if v_min is None else float(v_min),
        DEFAULT_V_MAX if v_max is None else float(v_max),
    )


def _fixed_closed(branch) -> bool:
    """True iff ``on_off`` is the literal constant 1 for this solve.

    Deliberately evaluated in ``equations()`` rather than in ``ensure_var``:
    ``OptimizationProblem`` promotes ``on_off`` to a binary Var *after*
    ``ensure_var`` has run (``controllable_backup_lines``, or a bare
    ``controllable([("on_off", ...)])`` on any branch), so nothing checked at
    declaration time can tell whether the branch will end up switchable. By the
    time ``equations()`` runs, a switchable branch's ``on_off`` is a solver
    variable and a fixed one is still a plain number.
    """
    on_off = getattr(branch, "on_off", 1)
    return (
        isinstance(on_off, (int, float))
        and not isinstance(on_off, bool)
        and on_off == 1
    )


class QCElectricityNodeFormulation(NodeFormulation):
    def __init__(self, v_min=None, v_max=None):
        """*v_min* / *v_max* override the envelope band for every bus - see
        :func:`_v_bounds` for what the band costs and what it risks."""
        self.v_override = (v_min, v_max)

    def ensure_var(self, node, simulation=False, grid=None):
        # Only real voltage buses; multi-grid control nodes subclass Bus without
        # a vm_pu.
        if not hasattr(node, "vm_pu"):
            return
        v_min, v_max = _v_bounds(node, grid, self.v_override)
        node.vm_pu = Var(1, min=v_min, max=v_max, name="vm_pu")
        node.vm_pu_squared = Var(1, min=v_min**2, max=v_max**2, name="vm_pu_squared")
        # ``vm_pu`` is the McCormick *lifting* variable for the vv = v_i*v_j
        # product; ``square_relax`` only ties it to w by a single chord, so it
        # can sit anywhere in [(w + v_min*v_max)/(v_min + v_max), sqrt(w)] -
        # a gap of up to (v_max - v_min)^2 / 8. ``vm_pu_squared`` carries the
        # physics and is the quantity that matches AC; report the voltage
        # derived from it as well, mirroring what the MISOCP formulation makes
        # its ``vm_pu`` intermediate be.
        node.vm_pu_from_w = PostProcess(lambda v: float(v.vm_pu_squared) ** 0.5)

    def equations(
        self,
        node,
        grid,
        from_branch_models,
        to_branch_models,
        connected_node_models,
        **kwargs,
    ):
        v_min, v_max = _v_bounds(node, grid, self.v_override)

        return opfmodel.square_relax(
            v_sq_var=node.vm_pu_squared,
            v_var=node.vars["vm_pu"],
            v_min=v_min,
            v_max=v_max,
        )


class QCElectricityBranchFormulation(BranchFormulation):
    def __init__(self, v_min=None, v_max=None):
        """*v_min* / *v_max* override the envelope band - pass the same values
        to :class:`QCElectricityNodeFormulation`, the two must agree."""
        self.v_override = (v_min, v_max)

    def ensure_var(self, branch, simulation=False, grid=None, **kwargs):
        theta_u = getattr(branch, "angmax", getattr(branch, "delta_max", math.pi / 6))
        theta_M = getattr(branch, "big_m_theta", math.pi)

        # Switch-safe Var bounds: an open branch is forced to cs = s = 0 by the
        # switched cosine/sine relaxations, and whether this branch can open is
        # not knowable yet (see _fixed_closed). The tighter physical range
        # cs in [cos(theta_u), 1], |s| <= sin(theta_u) is imposed as a
        # *constraint* in equations(), where the answer is known.
        branch.va_diff = Var(0, min=-theta_M, max=theta_M)
        branch.cs = Var(1, min=0, max=1)
        branch.s = Var(0, min=-1, max=1)

        # No voltage-derived bounds here: ``ensure_var`` cannot see the two
        # end nodes, so a per-node band override would silently contradict them
        # and make the model infeasible. The McCormick envelopes in
        # ``equations()`` - which do see both nodes - bound these three anyway.
        branch.vv = Var(1, min=0)
        branch.wc = Var(1)
        branch.ws = Var(0)

        branch.i_qc = Var(0, min=0)
        # Report-only, and pinned by an *equality* below. As free Vars under a
        # one-sided ">=" nothing referenced them and nothing bounded them from
        # above, so the solver left them at ~1e5 kA in every solve.
        branch.i_from_ka = Intermediate(0)
        branch.i_to_ka = Intermediate(0)
        branch.loading_from_pu = Intermediate(0)
        branch.loading_to_pu = Intermediate(0)

        branch.theta_u = theta_u
        branch.theta_M = theta_M

    def minimize(self, branch, grid, from_node_model, to_node_model, **kwargs):
        """Ohmic loss ``r * l``, the term that drives the relaxation tight.

        Same device the branch-flow MISOCP formulation uses. It is what makes
        the ``wc``/``ws`` envelopes and the current SOC bind at the AC-feasible
        point; drop it and the solver is free to return any point of the relaxed
        set - on a 0.5 MW radial feeder that means a ~56 MW slack injection.
        """
        return [branch.i_qc * branch.br_r_pu]

    def equations(self, branch, grid, from_node_model, to_node_model, **kwargs):
        y = np.linalg.pinv([[branch.br_r_pu + branch.br_x_pu * 1j]])[0][0]
        g, b = (np.real(y), np.imag(y))

        v_from_min, v_from_max = _v_bounds(from_node_model, grid, self.v_override)
        v_to_min, v_to_max = _v_bounds(to_node_model, grid, self.v_override)

        vv_min = v_from_min * v_to_min
        vv_max = v_from_max * v_to_max

        # A branch that cannot open never reaches cs = 0, so it gets the
        # physical trigonometric range - which is what the McCormick y-bounds
        # below are built from, and is far tighter than [0, 1] / [-1, 1].
        fixed_closed = _fixed_closed(branch)
        cs_min = math.cos(branch.theta_u) if fixed_closed else 0.0
        s_abs = math.sin(branch.theta_u) if fixed_closed else 1.0

        # Linear line-loading path. i_qc is |I|^2 in per-unit, so
        # loading^2 = i_qc * (I_base / max_i_ka)^2 is LINEAR in i_qc, whereas
        # loading_*_pu carries a sqrt. line_loading_limit() picks this up via
        # the _misocp_loading_*_scale_squared attributes and emits the linear
        # form - without it, check_lp=True makes the model unwritable as an LP
        # and gurobi refuses it ("nonlinear terms that cannot be written to LP
        # format"). Same device the branch-flow MISOCP formulation uses.
        tap = float(getattr(branch, "tap", 1) or 1)
        max_i_ka = getattr(branch, "max_i_ka", None)
        if max_i_ka and max_i_ka > 0:
            i_base_from = grid.sn_mva / (SQRT_3 * from_node_model.base_kv) / tap
            i_base_to = grid.sn_mva / (SQRT_3 * to_node_model.base_kv)
            branch._misocp_loading_from_scale_squared = (i_base_from / max_i_ka) ** 2
            branch._misocp_loading_to_scale_squared = (i_base_to / max_i_ka) ** 2
            branch.current_pu_squared = branch.i_qc

        # Report-only current magnitudes, same construction as nlp/el.py: the
        # sqrt node is shared between i_*_ka and loading_*_pu.
        #
        # Divide by sqrt(w), NOT by the vm_pu lifting variable. vm_pu only has
        # to satisfy the single square_relax chord, so at the default band it
        # sits well under sqrt(w) and the reported currents came out ~14% high.
        # Same choice the MISOCP formulation makes for its i_*_ka intermediates.
        vm_from = kwargs["sqrt_impl"](from_node_model.vars["vm_pu_squared"])
        vm_to = kwargs["sqrt_impl"](to_node_model.vars["vm_pu_squared"])
        i_from_ka = kwargs["sqrt_impl"](
            branch.p_from_mw**2 + branch.q_from_mvar**2 + CURRENT_SMOOTHING_EPS_MW**2
        ) / (vm_from * from_node_model.vars["base_kv"] * SQRT_3)
        i_to_ka = kwargs["sqrt_impl"](
            branch.p_to_mw**2 + branch.q_to_mvar**2 + CURRENT_SMOOTHING_EPS_MW**2
        ) / (vm_to * to_node_model.vars["base_kv"] * SQRT_3)

        eqs = [
            branch.va_diff
            == from_node_model.vars["va_radians"] - to_node_model.vars["va_radians"],
        ]

        if fixed_closed:
            eqs += [
                branch.cs >= cs_min,
                branch.s <= s_abs,
                branch.s >= -s_abs,
            ]

        eqs += opfmodel.cosine_relax(
            cs_var=branch.cs,
            delta_var=branch.va_diff,
            delta_max=branch.theta_u,
            on_off=branch.on_off,
            delta_big_m=branch.theta_M,
        )

        eqs += opfmodel.sine_relax(
            s_var=branch.s,
            delta_var=branch.va_diff,
            delta_max=branch.theta_u,
            on_off=branch.on_off,
            delta_big_m=branch.theta_M,
        )

        eqs += opfmodel.mccormick_relax(
            product_var=branch.vv,
            x_var=from_node_model.vars["vm_pu"],
            y_var=to_node_model.vars["vm_pu"],
            x_lb=v_from_min,
            x_ub=v_from_max,
            y_lb=v_to_min,
            y_ub=v_to_max,
        )

        # y-bounds must match the declared cs/s ranges - passing [0, 1] / [-1, 1]
        # for a non-switchable branch widens each envelope by
        # (vv_max - vv_min) * extra_range / 4, i.e. hundreds of MW once
        # multiplied by g.
        eqs += opfmodel.mccormick_relax(
            product_var=branch.wc,
            x_var=branch.vv,
            y_var=branch.cs,
            x_lb=vv_min,
            x_ub=vv_max,
            y_lb=cs_min,
            y_ub=1.0,
        )

        eqs += opfmodel.mccormick_relax(
            product_var=branch.ws,
            x_var=branch.vv,
            y_var=branch.s,
            x_lb=vv_min,
            x_ub=vv_max,
            y_lb=-s_abs,
            y_ub=s_abs,
        )

        eqs += [
            opfmodel.int_flow_from_p(
                p_from_var=branch.p_from_mw,
                v_sq_from_var=from_node_model.vars["vm_pu_squared"],
                wc_var=branch.wc,
                ws_var=branch.ws,
                g_branch=g,
                b_branch=b,
                tap=branch.tap,
                shift=branch.shift,
                g_from=branch.g_fr_pu,
                on_off=branch.on_off,
            ),
            opfmodel.int_flow_from_q(
                q_from_var=branch.q_from_mvar,
                v_sq_from_var=from_node_model.vars["vm_pu_squared"],
                wc_var=branch.wc,
                ws_var=branch.ws,
                g_branch=g,
                b_branch=b,
                tap=branch.tap,
                shift=branch.shift,
                b_from=branch.b_fr_pu,
                on_off=branch.on_off,
            ),
            opfmodel.int_flow_to_p(
                p_to_var=branch.p_to_mw,
                v_sq_to_var=to_node_model.vars["vm_pu_squared"],
                wc_var=branch.wc,
                ws_var=branch.ws,
                g_branch=g,
                b_branch=b,
                tap=branch.tap,
                shift=branch.shift,
                g_to=branch.g_to_pu,
                on_off=branch.on_off,
            ),
            opfmodel.int_flow_to_q(
                q_to_var=branch.q_to_mvar,
                v_sq_to_var=to_node_model.vars["vm_pu_squared"],
                wc_var=branch.wc,
                ws_var=branch.ws,
                g_branch=g,
                b_branch=b,
                tap=branch.tap,
                shift=branch.shift,
                b_to=branch.b_to_pu,
                on_off=branch.on_off,
            ),
            opfmodel.current_flow_equation(
                i_var=branch.i_qc,
                v_sq_from_var=from_node_model.vars["vm_pu_squared"],
                v_sq_to_var=to_node_model.vars["vm_pu_squared"],
                wc_var=branch.wc,
                g_branch=g,
                b_branch=b,
                tap=branch.tap,
                shift=branch.shift,
                ws_var=branch.ws,
                on_off=branch.on_off,
            ),
            opfmodel.current_soc_relax(
                p_var=branch.p_from_mw,
                q_var=branch.q_from_mvar,
                v_sq_var=from_node_model.vars["vm_pu_squared"],
                i_var=branch.i_qc,
                tap=branch.tap,
                g_from=branch.g_fr_pu,
                b_from=branch.b_fr_pu,
            ),
            opfmodel.voltage_product_soc(
                wc_var=branch.wc,
                ws_var=branch.ws,
                v_sq_from_var=from_node_model.vars["vm_pu_squared"],
                v_sq_to_var=to_node_model.vars["vm_pu_squared"],
            ),
            IntermediateEq("i_from_ka", i_from_ka),
            IntermediateEq("i_to_ka", i_to_ka),
            IntermediateEq("loading_from_pu", i_from_ka / branch.max_i_ka),
            IntermediateEq("loading_to_pu", i_to_ka / branch.max_i_ka),
        ]

        return eqs

"""DC power flow: the linear counterpart of the AC formulation (nlp/el.py).

Source: MATPOWER User's Manual v8.1, Sect. 3.7 (equations are in
monee.model.phys.linear.DC).

What this formulation does:

* Branches: p_from = b * (angle difference), p_to = -p_from (3.23/3.24).
* Buses: voltage magnitude fixed to 1 p.u. (3.19); only the active-power
  balance is kept (3.32).
* Reactive power: set to 0 everywhere, so no free variables remain.
* Shunts: P = G_sh (3.26).

The result is a linear program (LP).
"""

import math

import numpy as np

import monee.model.phys.linear.DC as dcmodel
from monee.model.core import Const, PostProcess, Var
from monee.model.grid import PowerGrid

from ..core import BranchFormulation, ChildFormulation, NodeFormulation

SQRT_3 = np.sqrt(3)


class DcLinearNodeFormulation(NodeFormulation):
    """Bus: |V| = 1 p.u., active-power balance only."""

    def ensure_var(self, model, simulation=False, grid=None):
        # |V| = 1 p.u. (3.19)
        model.vm_pu = Const(1.0)
        model.vm_pu_squared = Const(1.0)
        # Tells Bus.equations to skip the reactive-power balance.
        model._active_power_only = True


class DcLinearChildFormulation(ChildFormulation):
    """Any power-grid child (slack, generator, ...): reactive power is 0.

    Without this, a free Q variable would have no equation left to fix it.
    Loads keep their given Q, but it is not used.
    """

    def ensure_var(self, model, simulation=False, grid=None):
        if isinstance(getattr(model, "q_mvar", None), Var):
            model.q_mvar = Const(0.0)


class DcLinearShuntFormulation(DcLinearChildFormulation):
    """Shunt: P = G_sh (3.26), Q = 0."""

    def equations(self, child, grid, node, **kwargs):
        return [child.p_mw == dcmodel.shunt_p(child.gs_mw)]


class DcLinearBranchFormulation(BranchFormulation):
    """Lossless DC branch (3.23/3.24)."""

    def ensure_var(self, branch, simulation=False, grid=None):
        # No reactive flow in DC.
        branch.q_from_mvar = Const(0.0)
        branch.q_to_mvar = Const(0.0)
        # Current and loading are computed after the solve (see equations()),
        # so they are not solver variables.
        for key in ("i_from_ka", "i_to_ka", "loading_from_pu", "loading_to_pu"):
            setattr(branch, key, PostProcess(lambda v: 0.0))

        # Scale the MW flow variables like the AC formulation does.
        sn = grid.sn_mva if isinstance(grid, PowerGrid) else 1.0
        if sn and not math.isclose(sn, 1.0):
            for key in ("p_from_mw", "p_to_mw"):
                var = getattr(branch, key, None)
                if isinstance(var, Var):
                    var.scale = sn

    def equations(self, branch, grid, from_node_model, to_node_model, **kwargs):
        s_base = grid.sn_mva if isinstance(grid, PowerGrid) else 1.0
        self._set_current_reports(branch, from_node_model, to_node_model)

        return dcmodel.int_flows(
            p_from_var=branch.p_from_mw,
            p_to_var=branch.p_to_mw,
            va_from_rad=from_node_model.vars["va_radians"],
            va_to_rad=to_node_model.vars["va_radians"],
            br_x_pu=branch.br_x_pu,
            tap=branch.tap,
            shift=branch.shift,
            on_off=branch.on_off,
            s_base=s_base,
        )

    @staticmethod
    def _set_current_reports(branch, from_node_model, to_node_model):
        """Current and loading after the solve: I = |P| / (sqrt(3) * V_base),
        since |V| = 1 and Q = 0. max_i_ka refers to the from-side voltage, as in AC."""
        kv_from = float(from_node_model.vars["base_kv"])
        kv_to = float(to_node_model.vars["base_kv"])
        max_i = branch.max_i_ka

        branch.i_from_ka = PostProcess(
            lambda v: abs(v.p_from_mw) / (SQRT_3 * kv_from)
        )
        branch.i_to_ka = PostProcess(lambda v: abs(v.p_to_mw) / (SQRT_3 * kv_to))
        branch.loading_from_pu = PostProcess(
            lambda v: abs(v.p_from_mw) / (SQRT_3 * kv_from * max_i)
        )
        branch.loading_to_pu = PostProcess(
            lambda v: abs(v.p_to_mw) / (SQRT_3 * kv_from * max_i)
        )

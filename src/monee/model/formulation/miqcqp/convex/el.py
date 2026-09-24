r"""Branch-flow MISOCP electricity formulation: convex SOC relaxation
:math:`P^2 + Q^2 \le (W/tap^2) \cdot ell` plus big-M switching binaries."""

import math

from monee.model.core import Intermediate, IntermediateEq, Var, value
from monee.model.formulation.core import (
    BranchFormulation,
    ChildFormulation,
    NodeFormulation,
)
from monee.model.phys.misoc.pf import (
    active_power_loss,
    reactive_power_loss,
    soc_rel,
    voltage_drop,
)

SQRT_3 = math.sqrt(3.0)


def _to_pu(value_mw, grid):
    """Normalise a branch power var to per-unit on the grid apparent-power base.

    ``p_*_mw`` / ``q_*_mvar`` are stored in MW / MVAr (the polar-AC NLP consumes
    them directly in those units). The branch-flow MISOCP works in per-unit, so
    every power that enters the SOC / voltage-drop / loss equations is divided
    by ``S_base = grid.sn_mva`` here - the single place that MW->pu conversion
    happens.
    """
    return value_mw / grid.sn_mva


class MISOCPElectricityNodeFormulation(NodeFormulation):
    def ensure_var(self, node, simulation=False, grid=None):
        vm_min = getattr(grid, "vm_pu_min", 0.5)
        vm_max = getattr(grid, "vm_pu_max", 1.5)
        node.vm_pu_squared = Var(1, min=vm_min**2, max=vm_max**2)
        node.vm_pu = Intermediate(1)

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
            IntermediateEq("vm_pu", kwargs["sqrt_impl"](node.vm_pu_squared)),
        ]


class MISOCPShuntFormulation(ChildFormulation):
    """Voltage-dependent bus shunt for the branch-flow model: ``p/q`` are linear
    in the voltage-squared decision variable ``vm_pu_squared`` (``W``)."""

    def equations(self, child, grid, node, **kwargs):
        w = node.vars["vm_pu_squared"]
        return [
            child.p_mw == child.gs_mw * w,
            child.q_mvar == -child.bs_mvar * w,
        ]


def _branch_tap(branch) -> float:
    """Off-nominal turns ratio (1.0 if absent or zero)."""
    tap = getattr(branch, "tap", 1.0) or 1.0
    return float(tap)


def _ell_physics_max(branch, w_max: float) -> float:
    r"""Per-unit :math:`|I|^2` upper bound from voltage bounds:
    :math:`(1 + 1/tap)^2 \cdot W_{max} / |Z|^2` (from :math:`|V_i/tap - V_j| \le
    (1/tap + 1)\sqrt{W_{max}}`); reduces to :math:`4 \cdot W_{max}/|Z|^2` at tap=1."""
    tap = _branch_tap(branch)
    return (1 + 1 / tap) ** 2 * w_max / (branch.br_r_pu**2 + branch.br_x_pu**2)


_ELL_THERMAL_HEADROOM = 3.0


def _ell_max(branch, w_max: float, i_base_from: float, i_base_to: float) -> float:
    r"""Tightest available per-unit :math:`|I|^2` bound: voltage-derived, capped by the
    thermal rating (with headroom) when ``max_i_ka`` is available."""
    ell = _ell_physics_max(branch, w_max)
    max_i_ka = getattr(branch, "max_i_ka", None)
    if max_i_ka and max_i_ka > 0:
        i_base = min(i_base_from, i_base_to)
        ell_thermal = (_ELL_THERMAL_HEADROOM * max_i_ka / i_base) ** 2
        ell = min(ell, ell_thermal)
    return ell


def _big_m(w_max: float) -> float:
    r"""Cauchy-Schwarz big-M; with :math:`ell_{max} = 4 \cdot W_{max}/|Z|^2` this collapses to :math:`9 \cdot W_{max}`."""
    return 9 * w_max


def _shunt_gate(branch):
    """Multiplier applied to the pi-shunt injection. A switchable branch keeps
    its charging energised here: gating by a binary ``on_off`` Var would make
    the SOC constraint quartic, and the nodal balance (model/node.py) already
    multiplies the terminal flows by ``on_off``."""
    on_off = branch.on_off
    return 1.0 if type(on_off) is Var else on_off


def _series_flows(branch, grid, from_node_model, to_node_model, tap):
    r"""Split the terminal flows into the series flows of the pi-model.

    Matching the AC model (phys/nonlinear/ac.py), the from-side shunt sits behind
    the ideal transformer and sees :math:`W_i/tap^2` while the to-side shunt sees
    :math:`W_j`:
    :math:`P^{ser}_{ij} = P_{ij} - g_{fr} W_i/tap^2`,
    :math:`Q^{ser}_{ij} = Q_{ij} + b_{fr} W_i/tap^2`. All terms are linear in the
    voltage-squared decision variables, so the rotated cone survives.
    """
    gate = _shunt_gate(branch)
    w_from = from_node_model.vars["vm_pu_squared"] / (tap * tap)
    w_to = to_node_model.vars["vm_pu_squared"]
    g_fr = getattr(branch, "g_fr_pu", 0.0)
    b_fr = getattr(branch, "b_fr_pu", 0.0)
    g_to = getattr(branch, "g_to_pu", 0.0)
    b_to = getattr(branch, "b_to_pu", 0.0)
    return (
        _to_pu(branch.vars["p_from_mw"], grid) - gate * g_fr * w_from,
        _to_pu(branch.vars["q_from_mvar"], grid) + gate * b_fr * w_from,
        _to_pu(branch.vars["p_to_mw"], grid) - gate * g_to * w_to,
        _to_pu(branch.vars["q_to_mvar"], grid) + gate * b_to * w_to,
    )


class MISOCPElectricityBranchFormulation(BranchFormulation):
    """Branch-flow (BFM) MISOCP line/trafo model.

    The pi-model charging admittance (``g_fr_pu``/``b_fr_pu``/``g_to_pu``/
    ``b_to_pu``) enters as a shunt injection linear in ``vm_pu_squared``; the SOC,
    voltage-drop and loss equations see the resulting series flows.
    ``current_pu_squared`` and the reported ``i_*_ka``/``loading_*_pu`` are
    therefore series quantities and exclude the charging current.

    Known limitation: the branch-flow model works on voltage/current magnitudes
    only, so a transformer phase shift (``branch.shift``) cannot be represented
    and is silently ignored; only the magnitude ratio ``tap`` enters."""

    RELAXATION_GAP_TOL = 0.01

    def uncertified_relaxation_slack(self, branch, network) -> float:
        r"""Relative SOC-cone residual of a branch whose objective cannot tighten
        the cone, 0.0 otherwise.

        :meth:`minimize` prices the lifted current as :math:`r \cdot \ell`, so on a
        zero-resistance branch (an idealised transformer) nothing pulls the cone
        tight: :math:`\ell` is then pinned only by the voltage-drop row and by
        whatever slack the solver's MIP gap accepts. Such a residual certifies
        nothing about exactness - and the ``i_*_ka`` / ``loading_*_pu`` reported
        for that branch are not physical - so it is kept apart from the genuine
        relaxation gap of the resistive branches.
        """
        model = branch.model
        if value(model.br_r_pu) > 0:
            return 0.0
        w_from = value(network.node_by_id(branch.from_node_id).model.vm_pu_squared)
        ell = value(model.current_pu_squared)
        p_ser = value(model.p_series_from_mw)
        q_ser = value(model.q_series_from_mvar)
        if None in (w_from, ell, p_ser, q_ser):
            return 0.0
        tap = _branch_tap(model)
        lhs = w_from / (tap * tap) * ell
        sn_mva = branch.grid.sn_mva
        return (lhs - (p_ser / sn_mva) ** 2 - (q_ser / sn_mva) ** 2) / max(
            abs(lhs), 1e-6
        )

    def ensure_var(self, branch, simulation=False, grid=None):
        branch.current_pu_squared = Var(1, min=0)
        branch.i_from_ka = Intermediate(0)
        branch.i_to_ka = Intermediate(0)
        branch.loading_from_pu = Intermediate(0)
        branch.loading_to_pu = Intermediate(0)
        branch.p_series_from_mw = Intermediate(0)
        branch.q_series_from_mvar = Intermediate(0)
        branch.p_series_to_mw = Intermediate(0)
        branch.q_series_to_mvar = Intermediate(0)

    def minimize(self, branch, grid, from_node_model, to_node_model, **kwargs):
        # r \cdot ell loss term doubles as the relaxation-tightening incentive.
        return [branch.current_pu_squared * branch.br_r_pu]

    def _soc_constraints(self, branch, from_node_model, p_series_pu, q_series_pu, tap):
        """SOC relaxation; the exact (non-convex) sibling overrides this with
        the equality form."""
        return [
            soc_rel(
                from_node_model.vars["vm_pu_squared"],
                p_series_pu,
                q_series_pu,
                branch.current_pu_squared,
                tap=tap,
            ),
        ]

    def equations(self, branch, grid, from_node_model, to_node_model, **kwargs):
        w_max = grid.vm_pu_max**2
        big_m = _big_m(w_max)
        tap = _branch_tap(branch)
        sqrt_impl = kwargs["sqrt_impl"]
        # I_base_ka = S_base / (\sqrt{3} \cdot V_base); trafo primary divides by tap.
        i_base_from = grid.sn_mva / (SQRT_3 * from_node_model.base_kv) / tap
        i_base_to = grid.sn_mva / (SQRT_3 * to_node_model.base_kv)
        ell_phys = _ell_max(branch, w_max, i_base_from, i_base_to)
        i_mag_pu = sqrt_impl(branch.current_pu_squared)
        # loading^2 = current_pu_squared \cdot (I_base/max_i_ka)^2 is linear in current_pu_squared.
        # Used by line_loading_limit() instead of the sqrt-bearing form.
        # max_i_ka is expressed in the from-side voltage basis (io/matpower.py:
        # rate_a/(sqrt3*V_from)), so the to-side rated current is
        # max_i_ka*V_from/V_to and the to-side loading scale reduces to
        # i_base_from*tap (= sn/(sqrt3*V_from)); dividing i_to_ka by the raw
        # max_i_ka inflated a transformer's loading_to_pu by the voltage ratio.
        # For a line (equal base_kv, tap=1) both forms are identical.
        branch._misocp_loading_from_scale_squared = (i_base_from / branch.max_i_ka) ** 2
        branch._misocp_loading_to_scale_squared = (
            i_base_from * tap / branch.max_i_ka
        ) ** 2
        p_ser_from, q_ser_from, p_ser_to, q_ser_to = _series_flows(
            branch, grid, from_node_model, to_node_model, tap
        )
        vd = voltage_drop(
            from_node_model.vars["vm_pu_squared"],
            to_node_model.vars["vm_pu_squared"],
            p_ser_from,
            q_ser_from,
            branch.current_pu_squared,
            branch.br_r_pu,
            branch.br_x_pu,
            tap=tap,
        )
        return [
            branch.current_pu_squared <= ell_phys * branch.on_off,
            vd <= big_m * (1 - branch.on_off),
            vd >= -big_m * (1 - branch.on_off),
            *self._soc_constraints(
                branch, from_node_model, p_ser_from, q_ser_from, tap
            ),
            active_power_loss(
                p_ser_from,
                p_ser_to,
                branch.current_pu_squared,
                branch.br_r_pu,
            ),
            reactive_power_loss(
                q_ser_from,
                q_ser_to,
                branch.current_pu_squared,
                branch.br_x_pu,
            ),
            # |I_ka| = \sqrt{current_pu_squared} \cdot I_base, i.e. the series current;
            # the charging current of the pi-shunt is not included in either report.
            IntermediateEq("i_from_ka", i_mag_pu * i_base_from),
            IntermediateEq("i_to_ka", i_mag_pu * i_base_to),
            IntermediateEq("loading_from_pu", i_mag_pu * i_base_from / branch.max_i_ka),
            IntermediateEq(
                "loading_to_pu", i_mag_pu * i_base_from * tap / branch.max_i_ka
            ),
            # Series flows of the pi-model (terminal flow minus the shunt
            # injection): the quantities the SOC cone actually constrains, so a
            # cone-gap check reads these columns, not p_from_mw/q_from_mvar.
            IntermediateEq("p_series_from_mw", p_ser_from * grid.sn_mva),
            IntermediateEq("q_series_from_mvar", q_ser_from * grid.sn_mva),
            IntermediateEq("p_series_to_mw", p_ser_to * grid.sn_mva),
            IntermediateEq("q_series_to_mvar", q_ser_to * grid.sn_mva),
        ]

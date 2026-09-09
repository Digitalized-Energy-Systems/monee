"""Electricity-carrier islanding mode and grid-forming generator model."""

from __future__ import annotations

from monee.model.child import NoVarChildModel
from monee.model.core import (
    Const,
    Intermediate,
    PostProcess,
    Var,
    is_plain_number,
    model,
)
from monee.model.grid import PowerGrid
from monee.model.phys.islanding import (
    angle_lower_bound_energized,
    angle_upper_bound_energized,
    source_reference_angle,
)

from .core import (
    GridFormingMixin,
    IslandingMode,
    capacity_limited_island_nodes,
    node_leads_island,
    warn_if_leadership_unstamped,
)

BRANCH_GATING_MODES = ("auto", "all", "off")

VM_NOMINAL_PU = 1.0


def branch_off_when_endpoint_dark(on_off, e_i):
    """on_off <= e_i: a branch touching a de-energised bus carries no flow."""
    return on_off <= e_i


def branch_on_between_live_endpoints(on_off, e_i, e_j):
    """on_off >= e_i + e_j - 1: with both endpoints live the branch stays in
    service, so ``on_off`` is exactly ``e_i AND e_j`` and adds no freedom."""
    return on_off >= e_i + e_j - 1


def voltage_upper_bound_energized(v, v_max, e_i, nominal=VM_NOMINAL_PU):
    """v <= nominal + (v_max - nominal) * e_i: a de-energised bus reports its
    nominal voltage instead of an unconstrained one."""
    return v <= nominal + (v_max - nominal) * e_i


def voltage_lower_bound_energized(v, v_min, e_i, nominal=VM_NOMINAL_PU):
    """v >= nominal - (nominal - v_min) * e_i (see
    :func:`voltage_upper_bound_energized`)."""
    return v >= nominal - (nominal - v_min) * e_i


def _is_decision_value(val) -> bool:
    """True for something a constraint can be written against: a monee Var
    before injection, or the backend's own variable afterwards. Intermediates
    and post-processed reports are neither."""
    return (
        val is not None
        and not isinstance(val, Intermediate | PostProcess | Const)
        and not is_plain_number(val)
    )


def _rebound_symmetric(var, limit: float, attr_name: str) -> None:
    if not isinstance(var, Var):
        raise AttributeError(
            f"'{attr_name}' can only be set while the model still holds a monee "
            "Var; this network is currently injected into a solver backend. Set "
            "the capacity before solving (see how-to/islanding)."
        )
    if limit is None:
        var.min = None
        var.max = None
        return
    var.min = -limit
    var.max = limit


@model
class GridFormingGenerator(NoVarChildModel, GridFormingMixin):
    """Grid-forming generator: variable p_mw/q_mvar (absorbs island imbalance)
    with a pinned vm_pu. Angle is pinned by :class:`ElectricityIslandingMode`.

    ``p_mw_max``/``q_mvar_max`` are positive capacities applied symmetrically
    (``-max <= p_mw <= max``) and stay writable after construction: assigning
    either re-bounds the underlying Var, so a scenario study can change the
    unit capacity in place before the next solve.

    Sign convention: like every child component, p_mw/q_mvar are reported in
    load convention, so a unit injecting 2 MW reports ``p_mw = -2.0`` while its
    ``p_mw_max`` argument is ``+2.0``.

    The vm pin applies only when this child leads its island
    (``_gf_leading``, stamped by :meth:`IslandingMode.stamp_gf_leadership`):
    on an ext-grid-led component an unconditional pin would over-constrain
    the power flow (no voltage gradient, so no transport). While not leading
    the unit holds ``nominal_p_mw``/``nominal_q_mvar`` (0 by default); pass
    ``None`` for either to leave that injection a free, unpriced Var."""

    def __init__(
        self,
        p_mw_max: float,
        q_mvar_max: float,
        vm_pu: float = 1.0,
        nominal_p_mw: float | None = 0.0,
        nominal_q_mvar: float | None = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.p_mw = Var(0, min=-p_mw_max, max=p_mw_max, name="gf_p_mw")
        self.q_mvar = Var(0, min=-q_mvar_max, max=q_mvar_max, name="gf_q_mvar")
        self._vm_pu_setpoint = vm_pu
        self._nominal_p_mw = nominal_p_mw
        self._nominal_q_mvar = nominal_q_mvar

    @property
    def p_mw_max(self):
        return getattr(self.p_mw, "max", None)

    @p_mw_max.setter
    def p_mw_max(self, value) -> None:
        _rebound_symmetric(self.p_mw, value, "p_mw_max")

    @property
    def q_mvar_max(self):
        return getattr(self.q_mvar, "max", None)

    @q_mvar_max.setter
    def q_mvar_max(self, value) -> None:
        _rebound_symmetric(self.q_mvar, value, "q_mvar_max")

    def overwrite(self, node_model, grid) -> None:
        if not self._gf_leading:
            return
        node_model.vm_pu = Const(self._vm_pu_setpoint)
        node_model.vm_pu_squared = Const(self._vm_pu_setpoint**2)

    def equations(self, grid, node, **kwargs):
        # A leading former needs its free balancing Vars. Where an ext grid
        # leads the component, stamp_gf_leadership marks every former
        # non-leading, overwrite() pins nothing, and p_mw/q_mvar would appear
        # in the node balance alone - two degrees of freedom that no equation
        # and no objective pins (plain energy flow carries no objective), so
        # the dispatch is an arbitrary vertex that differs per backend.
        warn_if_leadership_unstamped(self)
        if self._gf_leading:
            return []
        eqs = []
        if self._nominal_p_mw is not None:
            eqs.append(self.p_mw == self._nominal_p_mw)
        if self._nominal_q_mvar is not None:
            eqs.append(self.q_mvar == self._nominal_q_mvar)
        return eqs


class ElectricityIslandingMode(IslandingMode):
    r"""Electricity islanding: connectivity flow plus :math:`\theta=0` at GF buses,
    energisation-gated angle and voltage bounds elsewhere, and branch flows gated
    by the energisation of both endpoints.

    ``branch_gating`` selects which branches get the endpoint gate
    ``on_off = e_from AND e_to`` (see docs/source/concepts/islanding.md):

    * ``"auto"`` (default) - only branches touching a component anchored solely
      by capacity-limited grid-forming units, i.e. where load actually has to be
      dropped. Every other network keeps the previous formulation and its solve
      cost.
    * ``"all"`` - every carrier branch. Exact everywhere, at MIQCQP cost on the
      whole network.
    * ``"off"`` - no branch gating. ``on_off`` stays at its constant, so a
      de-energised bus is pinned to the reference angle rather than detached: it
      can neither import nor pass power through, which limits shedding to node
      sets already separated by open branches.
    """

    carrier_grid_type = PowerGrid
    var_prefix = "el"
    gated_child_attrs = ("p_mw", "q_mvar")

    def __init__(
        self,
        angle_bound: float = 3.15,
        big_m_conn: float | None = None,
        branch_gating: str = "auto",
    ) -> None:
        if branch_gating not in BRANCH_GATING_MODES:
            raise ValueError(
                f"branch_gating must be one of {BRANCH_GATING_MODES}, got "
                f"{branch_gating!r} (see docs/source/concepts/islanding.md)."
            )
        self.angle_bound = angle_bound
        self.big_m_conn = big_m_conn
        self.branch_gating = branch_gating

    def _prepare_node(self, node) -> None:
        # Claim bus-angle management: this mode pins \theta=0 at GF buses and
        # applies energisation-gated bounds elsewhere, so the grid-former must
        # NOT statically pin va_radians in its overwrite().
        node.model._islanding_angle_managed = True

    def _gated_branch_node_ids(self, network) -> set:
        if self.branch_gating == "off":
            return set()
        if self.branch_gating == "all":
            return {
                node.id
                for node in network.nodes
                if isinstance(node.grid, self.carrier_grid_type) and node.active
            }
        return capacity_limited_island_nodes(network, self)

    def _prepare_branches(self, network) -> None:
        """Promote ``on_off`` to a binary decision Var on the gated branches, so
        the equation phase can tie it to the endpoint energisation binaries. A
        branch pinned out of service (``on_off == 0``) is left alone.

        A switchable branch - one whose ``on_off`` is already a decision Var, and
        a ``backup`` line, which ``OptimizationProblem.controllable_backup_lines``
        promotes later in the solve - is only upper-gated: closing it stays the
        user's decision, so the AND lower bound would take that decision away."""
        gated_node_ids = self._gated_branch_node_ids(network)
        if not gated_node_ids:
            return
        for nid in gated_node_ids:
            network.node_by_id(nid).model._islanding_voltage_gated = True
        for branch in network.branches:
            if not (isinstance(branch.grid, self.carrier_grid_type) and branch.active):
                continue
            if (
                branch.from_node_id not in gated_node_ids
                and branch.to_node_id not in gated_node_ids
            ):
                continue
            on_off = branch.model.on_off
            switchable = isinstance(on_off, Var) or getattr(
                branch.model, "backup", False
            )
            if is_plain_number(on_off):
                if on_off == 0:
                    continue
                branch.model.on_off = Var(
                    1, min=0, max=1, integer=True, name="islanding_on_off"
                )
            branch.model._islanding_owns_on_off = not switchable
            branch.model._islanding_branch_gated = True

    def _branch_energisation_equations(self, network, e_vars) -> list:
        eqs = []
        for branch in network.branches:
            if not getattr(branch.model, "_islanding_branch_gated", False):
                continue
            if not branch.active or getattr(branch, "ignored", False):
                continue
            e_from = e_vars.get(branch.from_node_id)
            e_to = e_vars.get(branch.to_node_id)
            if e_from is None or e_to is None:
                continue
            on_off = branch.model.on_off
            eqs.append(branch_off_when_endpoint_dark(on_off, e_from))
            eqs.append(branch_off_when_endpoint_dark(on_off, e_to))
            if branch.model._islanding_owns_on_off:
                eqs.append(branch_on_between_live_endpoints(on_off, e_from, e_to))
        return eqs

    def _voltage_gate_equations(self, node, e) -> list:
        # Once the branch gate detaches a dark bus, no equation determines its
        # voltage any more. Pinning it to nominal keeps the model non-singular
        # (the current intermediates divide by vm_pu) and makes the reported
        # value deterministic; with e = 1 both bounds are the model bounds.
        if not getattr(node.model, "_islanding_voltage_gated", False):
            return []
        vm_min = getattr(node.grid, "vm_pu_min", 0.5)
        vm_max = getattr(node.grid, "vm_pu_max", 1.5)
        eqs = []
        vm = getattr(node.model, "vm_pu", None)
        if _is_decision_value(vm):
            eqs.append(voltage_upper_bound_energized(vm, vm_max, e))
            eqs.append(voltage_lower_bound_energized(vm, vm_min, e))
        w = getattr(node.model, "vm_pu_squared", None)
        if _is_decision_value(w):
            eqs.append(voltage_upper_bound_energized(w, vm_max**2, e))
            eqs.append(voltage_lower_bound_energized(w, vm_min**2, e))
        return eqs

    def add_physical_constraints(
        self, network, gf_nodes, regular_nodes, e_vars
    ) -> list:
        eqs = []
        bounded_nodes = list(regular_nodes)
        for node in gf_nodes:
            # theta=0 only at the island reference; a second pin in the same
            # island over-constrains the flow.
            if node_leads_island(network, node, self):
                eqs.append(source_reference_angle(node.model.va_radians))
            else:
                bounded_nodes.append(node)
        for node in bounded_nodes:
            e = e_vars[node.id]
            eqs.append(
                angle_upper_bound_energized(node.model.va_radians, self.angle_bound, e)
            )
            eqs.append(
                angle_lower_bound_energized(node.model.va_radians, self.angle_bound, e)
            )
            eqs += self._voltage_gate_equations(node, e)
        eqs += self._branch_energisation_equations(network, e_vars)
        return eqs

r"""
Lumped Thermal Capacitance (LTC) extension for water junctions.

Each junction gets thermal mass :math:`\rho \cdot \sum V_{pipe}/2` from connected pipes.
The inertia equation:

.. math::

    \rho \cdot V \cdot (T_{pu}(t) - T_{pu}(t-1))/\Delta t = \text{net\_convective\_heat\_in}

replaces the degenerate :math:`T_n \cdot \text{mass\_balance} = 0` heat balance at LTC nodes.
No-op in single-step solves.
"""

import math

from monee.model.child import GridFormingMixin
from monee.model.core import Var, set_initial_value
from monee.model.grid import WaterGrid
from monee.model.node import Junction
from monee.model.phys.nonlinear.hf import SPECIFIC_HEAT_CAP_WATER

from .core import NetworkAspect


class LumpedThermalCapacitance(NetworkAspect):
    """LTC extension. Nodes with a GridFormingMixin child are excluded.

    First-step anchor precedence (anchored mode, the default - required for NLP
    solvers like GEKKO/IPOPT): ``t_init_overrides[node_id]`` → ``default_t_init``
    → the junction's own ``t_pu`` Var initialiser. Pass ``default_t_init`` near
    the operating mean to skip the warm-up transient.

    ``first_step_steady_state=True`` drops the first-step inertia term (emits
    ``net_heat == 0``); MIP backends only.
    """

    def __init__(
        self,
        default_t_init: float | None = None,
        t_init_overrides: dict | None = None,
        first_step_steady_state: bool = False,
    ):
        self._ltc_rho_v: dict = {}  # {node_id: \rho \cdot V_lumped  [kg]}
        self._ltc_initial_t_pu: dict = {}  # {node_id: initial t_pu value}
        self._default_t_init: float | None = (
            None if default_t_init is None else float(default_t_init)
        )
        self._t_init_overrides: dict = dict(t_init_overrides or {})
        self._first_step_steady_state: bool = bool(first_step_steady_state)

    def prepare(self, network) -> None:  # NOSONAR
        self._ltc_rho_v = {}
        self._ltc_initial_t_pu = {}

        # Identify water junctions without a fixed-T supply (e.g. ExtHydrGrid).
        for node in network.nodes:
            if not (
                isinstance(node.model, Junction) and isinstance(node.grid, WaterGrid)
            ):
                continue
            children = network.childs_by_ids(node.child_ids)
            has_fixed_temp = any(
                isinstance(c.model, GridFormingMixin) for c in children
            )
            if not has_fixed_temp:
                self._ltc_rho_v[node.id] = 0.0

        # Accumulate half-pipe volumes at each end of every water pipe.
        for branch in network.branches:
            if not isinstance(branch.grid, WaterGrid):
                continue
            bm = branch.model
            if not (hasattr(bm, "diameter_m") and hasattr(bm, "length_m")):
                continue
            v_pipe = math.pi / 4 * bm.diameter_m**2 * bm.length_m
            for node_id in (branch.from_node_id, branch.to_node_id):
                if node_id in self._ltc_rho_v:
                    rho = network.node_by_id(node_id).grid.fluid_density_kg_per_m3
                    self._ltc_rho_v[node_id] += rho * v_pipe / 2

        # Resolve first-step anchor and replace t_pu with a fresh tracked Var.
        for node in network.nodes:
            if node.id not in self._ltc_rho_v:
                continue
            junc = node.model
            t_var = junc.t_pu
            var_init = t_var.value if isinstance(t_var, Var) else float(t_var)
            if node.id in self._t_init_overrides:
                t_init = float(self._t_init_overrides[node.id])
            elif self._default_t_init is not None:
                t_init = self._default_t_init
            else:
                t_init = var_init
            self._ltc_initial_t_pu[node.id] = t_init
            junc.t_pu = Var(
                t_init,
                min=t_var.min if isinstance(t_var, Var) else 0,
                max=t_var.max if isinstance(t_var, Var) else 2,
                name="t_pu",
            )

    def activate_timeseries(self, network, ignored_nodes: set, step_state=None) -> None:
        """Mark LTC junctions ``_ltc_active`` so the default degenerate heat
        balance is skipped (avoids a near-singular Jacobian); warm-start t_pu."""
        for node in network.nodes:
            if node.id not in self._ltc_rho_v:
                continue
            if node.id in ignored_nodes or node.ignored:
                continue
            # Only suppress the canonical heat balance when there is thermal mass
            # to replace it; zero-mass junctions emit no inertia equation in
            # inter_temporal_equations (rho_v <= 0 is skipped), so dropping their
            # balance would leave t_pu under-determined.
            if self._ltc_rho_v[node.id] <= 0.0:
                continue
            node.model._ltc_active = True
            # Warm-start; t_pu is a backend variable after inject_vars, so route
            # through set_initial_value (handles gurobipy's .Start vs .value).
            if step_state is not None:
                prev_t = step_state.get(node.id, "t_pu")
                if (
                    prev_t is not None
                    and hasattr(node.model, "t_pu")
                    and node.model.t_pu is not None
                ):
                    set_initial_value(node.model.t_pu, prev_t)

    def inter_temporal_equations(
        self, network, ignored_nodes: set, temporal_state
    ) -> list:
        r""":math:`\rho \cdot V \cdot (T_{pu}(t) - T_{pu}(t-1))/\Delta t = \text{net\_convective\_heat\_in}` per LTC
        junction. ``T_prev`` falls back to the first-step anchor (see class
        docstring) when ``temporal_state.get`` returns None."""
        eqs = []
        dt_s = temporal_state.dt_h * 3600.0

        for node in network.nodes:
            if node.id not in self._ltc_rho_v:
                continue
            if node.ignored:
                continue

            junc = node.model
            rho_v = self._ltc_rho_v[node.id]
            if rho_v <= 0.0:
                continue

            net_heat = self._net_convective_heat(node, network, ignored_nodes)

            t_prev = temporal_state.get(node.id, "t_pu")
            if t_prev is None and self._first_step_steady_state:
                # MIP-only steady-state mode - see class docstring.
                eqs.append(net_heat == 0)
            else:
                if t_prev is None:
                    t_prev = self._ltc_initial_t_pu.get(node.id, 1.0)
                eqs.append(rho_v * (junc.t_pu - t_prev) / dt_s == net_heat)

        return eqs

    def equations(self, network, ignored_nodes: set) -> list:
        return []

    def _net_convective_heat(self, node, network, ignored_nodes: set = frozenset()):  # NOSONAR
        """Net convective heat flow INTO *node* (inflow-positive).

        Two paths:
          * Plain: use the branch's t_from_pu/t_to_pu (heat-loss aware).
          * McCormick: use H_out_mw/H_in_mw / scale; avoids the dangling
            ``t_*_pu`` and the :math:`m \cdot \tau` bilinear McCormick is meant to drop.

        Branches dropped from the solve (``branch.ignored`` or an endpoint in
        ``ignored_nodes``) are skipped so ignored-island pipes do not contribute
        to the node heat balance (branch ids are 3-tuples, so they never match
        the scalar node ids in ``ignored_nodes`` - check endpoints instead).
        """
        t_n = node.model.t_pu
        terms = []
        scale_mw_per_kgs = SPECIFIC_HEAT_CAP_WATER * node.grid.t_ref_k / 1e6

        def _branch_ignored(branch) -> bool:
            return (
                branch.ignored
                or branch.from_node_id in ignored_nodes
                or branch.to_node_id in ignored_nodes
            )

        for branch in network.branches:
            if not isinstance(branch.grid, WaterGrid):
                continue
            if _branch_ignored(branch):
                continue
            bm = branch.model
            bvars = bm.vars
            if "mass_flow_pos_kgs" not in bvars or "mass_flow_neg_kgs" not in bvars:
                continue

            is_mccormick = "H_out_mw" in bvars and "H_in_mw" in bvars

            if branch.from_node_id == node.id:
                if is_mccormick:
                    terms.append(-bvars["H_out_mw"] / scale_mw_per_kgs)
                else:
                    if "t_from_pu" not in bvars:
                        continue
                    on_off = bvars.get("on_off", 1)
                    mpos = bvars["mass_flow_pos_kgs"] * on_off
                    mneg = bvars["mass_flow_neg_kgs"] * on_off
                    terms.append(mpos * bvars["t_from_pu"] - mneg * t_n)

            elif branch.to_node_id == node.id:
                if is_mccormick:
                    terms.append(bvars["H_in_mw"] / scale_mw_per_kgs)
                else:
                    if "t_to_pu" not in bvars:
                        continue
                    on_off = bvars.get("on_off", 1)
                    mpos = bvars["mass_flow_pos_kgs"] * on_off
                    mneg = bvars["mass_flow_neg_kgs"] * on_off
                    terms.append(mneg * bvars["t_to_pu"] - mpos * t_n)

        for child in network.childs_by_ids(node.child_ids):
            cm = child.model
            cvars = cm.vars
            if "mass_flow_kgs" in cvars:
                # Well-mixed: m_ext < 0 = injection \to heat IN = -m_ext \cdot t_n.
                m_ext = cvars["mass_flow_kgs"] * cvars.get("regulation", 1)
                terms.append(-m_ext * t_n)
            if "q_mw_heat" in cvars:
                # Load convention: positive = heat OUT → negate.
                q = cvars["q_mw_heat"] * cvars.get("regulation", 1)
                terms.append(-q / scale_mw_per_kgs)

        # Branch heat_mw (e.g. GasToHeatHG) absorbed at the TO-node.
        for branch in network.branches:
            if _branch_ignored(branch):
                continue
            bm = branch.model
            bvars = bm.vars
            if "q_mw_heat" not in bvars:
                continue
            if branch.to_node_id != node.id:
                continue
            q = bvars["q_mw_heat"] * bvars.get("on_off", 1)
            terms.append(-q / scale_mw_per_kgs)

        return sum(terms) if terms else 0

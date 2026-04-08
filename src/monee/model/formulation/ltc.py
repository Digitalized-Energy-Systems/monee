"""
Lumped Thermal Capacitance (LTC) network extension for water-network junctions.

Each water-network junction is assigned a thermal mass equal to ρ times the
volume of water in the half-pipes that connect to it.  The resulting inertia
term modifies the junction temperature dynamics:

    ρ·V_node · (T_pu(t) - T_pu(t-1)) / Δt  =  Σ(ṁ_in · T_in_arriving - ṁ_out · T_pu)

so that junction temperatures cannot jump instantaneously between timesteps.

The right-hand side is the net convective heat flow into the junction, computed
from the actual arriving pipe temperatures (not the degenerate T·mass_balance
form of the existing junction heat balance).

Usage (one-liner)::

    net.add_extension(LumpedThermalCapacitance())

The extension is a no-op in plain (non-timeseries) solves.  In both timeseries
and multi-period solves the first step/period uses the junction's initial
``t_pu`` value as the previous temperature, so results are consistent with the
steady-state solve.
"""

import math

from monee.model.core import Var
from monee.model.grid import WaterGrid
from monee.model.node import Junction

from .core import NetworkAspect


class LumpedThermalCapacitance(NetworkAspect):
    """
    Lumped-thermal-capacitance extension for water-network junctions.

    Assigns each water-network :class:`~monee.model.node.Junction` a thermal
    mass equal to ``ρ × Σ(V_pipe / 2)`` for every pipe connected to it.  The
    resulting inertia equation is added as an extra constraint that directly
    determines the junction temperature from the previous-step value and the
    net convective heat inflow.

    Nodes with a fixed supply temperature (e.g. those carrying an
    ``ExtHydrGrid`` child) are excluded automatically.

    The extension touches two things automatically:

    * **prepare** — replaces each junction's ``t_pu`` with a fresh
      :class:`~monee.model.core.Var` so bounds are preserved and the solved
      temperature is accessible via :class:`~monee.simulation.step_state.StepState`.
    * **inter_step_equations** — for each active junction, adds the discrete
      thermal-mass equation using the previous-step temperature from
      ``StepState`` (or the initial ``t_pu`` value for the very first step).
    """

    def __init__(self):
        self._ltc_rho_v: dict = {}  # {node_id: ρ·V_lumped  [kg]}
        self._ltc_initial_t_pu: dict = {}  # {node_id: initial t_pu value}
        self._ltc_constrained: set = set()

    # ------------------------------------------------------------------
    # Phase 0: variable preparation (before inject_vars)
    # ------------------------------------------------------------------

    def prepare(self, network) -> None:
        self._ltc_rho_v = {}
        self._ltc_initial_t_pu = {}
        self._ltc_constrained = set()

        # Import here to avoid circular imports at module level.
        from monee.model.child import GridFormingMixin

        # 1. Identify water junctions that do NOT have a fixed-temperature
        #    supply child (e.g. ExtHydrGrid).  Fixed-temperature nodes are
        #    driven externally and must not receive an additional LTC constraint.
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

        # 2. Accumulate half-pipe volumes at each end of every water pipe.
        for branch in network.branches:
            if not isinstance(branch.grid, WaterGrid):
                continue
            bm = branch.model
            if not (hasattr(bm, "diameter_m") and hasattr(bm, "length_m")):
                continue
            v_pipe = math.pi / 4 * bm.diameter_m**2 * bm.length_m
            for node_id in (branch.from_node_id, branch.to_node_id):
                if node_id in self._ltc_rho_v:
                    rho = network.node_by_id(node_id).grid.fluid_density
                    self._ltc_rho_v[node_id] += rho * v_pipe / 2

        # 3. Patch each LTC junction: store initial t_pu value and make it
        #    tracked so its solved value is persisted to StepState.
        for node in network.nodes:
            if node.id not in self._ltc_rho_v:
                continue
            junc = node.model
            t_var = junc.t_pu
            t_init = t_var.value if isinstance(t_var, Var) else float(t_var)
            self._ltc_initial_t_pu[node.id] = t_init
            junc.t_pu = Var(
                t_init,
                min=t_var.min if isinstance(t_var, Var) else 0,
                max=t_var.max if isinstance(t_var, Var) else 2,
                name="t_pu",
            )

    # ------------------------------------------------------------------
    # Phase 1a: timeseries activation (between inject_vars and equations)
    # ------------------------------------------------------------------

    def activate_timeseries(self, network, ignored_nodes: set, step_state=None) -> None:
        """
        Called by the solver when a timeseries (non-None step_state) solve is
        about to begin — AFTER ``prepare()`` and variable injection but BEFORE
        node equations are assembled.

        Sets ``_ltc_active = True`` on each LTC-constrained junction so that
        :meth:`~monee.model.node.Junction.calc_signed_heat_flow` suppresses
        the degenerate heat balance for those nodes.  This prevents the
        near-singular Jacobian that arises when both the degenerate
        ``T_n × mass_balance = 0`` and the LTC equality constraint are active
        simultaneously during IPOPT iteration.

        Not called during plain single-step solves, so single-step results are
        identical whether or not the extension is attached.
        """
        for node in network.nodes:
            if node.id not in self._ltc_rho_v:
                continue
            if node.id in ignored_nodes or node.ignored:
                continue
            node.model._ltc_active = True
            # Warm-start the solver initial value from the previous step so
            # iterative solvers (GEKKO/IPOPT) start close to the solution.
            # At this point t_pu may already be a backend variable (after
            # inject_vars), so we set .value regardless of the type.
            if step_state is not None:
                prev_t = step_state.get(node.id, "t_pu")
                if (
                    prev_t is not None
                    and hasattr(node.model, "t_pu")
                    and node.model.t_pu is not None
                ):
                    node.model.t_pu.value = prev_t

    # ------------------------------------------------------------------
    # Phase 1b: inter-temporal constraints (timeseries + multi-period)
    # ------------------------------------------------------------------

    def inter_temporal_equations(
        self, network, ignored_nodes: set, temporal_state
    ) -> list:
        """
        For each LTC junction, add the discrete thermal-mass equation:

            ρ·V · (T_pu(t) − T_pu(t−1)) / Δt  ==  net_convective_heat_in

        This is called in **both** timeseries (where *temporal_state* is a
        :class:`~monee.simulation.step_state.StepState` returning floats) and
        multi-period optimisation (where *temporal_state* is a
        :class:`~monee.simulation.step_state.PeriodState` returning live solver
        variables), making the thermal inertia effective in both solve modes.

        *T_prev* = ``temporal_state.get(node_id, "t_pu")`` — ``None`` on the
        first step/period, in which case the initial ``t_pu`` value anchors the
        constraint so the very first solve is thermally consistent.

        This equation replaces the degenerate junction heat balance
        (``T_n × mass_balance == 0``) which is suppressed for LTC nodes via
        :meth:`activate_timeseries` to avoid a near-singular Jacobian.
        """
        self._ltc_constrained = set()
        eqs = []
        dt_s = temporal_state.dt_h * 3600.0

        for node in network.nodes:
            if node.id not in self._ltc_rho_v:
                continue
            if node.ignored:
                continue

            junc = node.model
            rho_v = self._ltc_rho_v[node.id]

            # Use stored previous temperature, or fall back to the initial
            # value (so the very first step/period is also thermally consistent).
            t_prev = temporal_state.get(node.id, "t_pu")
            if t_prev is None:
                t_prev = self._ltc_initial_t_pu.get(node.id, 1.0)

            if rho_v > 0.0:
                net_heat = self._net_convective_heat(node, network)
                eqs.append(rho_v * (junc.t_pu - t_prev) / dt_s == net_heat)

            self._ltc_constrained.add(node.id)

        return eqs

    def inter_step_equations(self, network, ignored_nodes: set, step_state) -> list:
        """Delegate to :meth:`inter_temporal_equations` (backward compatibility)."""
        return self.inter_temporal_equations(network, ignored_nodes, step_state)

    # ------------------------------------------------------------------
    # Phase 2: fallback (single-step or no step_state)
    # ------------------------------------------------------------------

    def equations(self, network, ignored_nodes: set) -> list:
        """
        No extra equations are needed for single-step (non-timeseries) solves.
        The junction temperatures are determined by the existing heat balance.
        """
        return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _net_convective_heat(self, node, network):
        """
        Return the solver expression for net convective heat flow INTO *node*.

        Sign convention (same as Junction.calc_signed_heat_flow but negated):
        the existing junction heat-balance uses an *outflow* convention
        (sum = 0, positive = outflow).  Here we need the net *inflow*,
        so each branch term is the negative of the junction balance term.

        For each branch connected to *node*:
        - If *node* is the FROM-end:
          - mpos > 0  (backward flow, to→from): fluid arrives at T_from_pu → heat IN
          - mneg > 0  (forward flow, from→to): fluid leaves at T_n → heat OUT
          → term = mpos * t_from_pu - mneg * T_n
        - If *node* is the TO-end:
          - mneg > 0  (forward flow, from→to): fluid arrives at T_to_pu → heat IN
          - mpos > 0  (backward flow, to→from): fluid leaves at T_n → heat OUT
          → term = mneg * t_to_pu - mpos * T_n

        Children are modelled as injecting/withdrawing at junction temperature
        (well-mixed assumption).  Their mass_flow sign follows load convention:
        negative = injection, so net heat IN = -m_ext * T_n.
        """
        T_n = node.model.t_pu  # solver variable after injection
        terms = []

        for branch in network.branches:
            if not isinstance(branch.grid, WaterGrid):
                continue
            bm = branch.model
            bvars = bm.vars
            if "mass_flow_pos" not in bvars or "mass_flow_neg" not in bvars:
                continue

            on_off = bvars.get("on_off", 1)
            mpos = bvars["mass_flow_pos"] * on_off
            mneg = bvars["mass_flow_neg"] * on_off

            if branch.from_node_id == node.id:
                # This node is FROM-end.
                # mpos > 0 (backward, to→from): fluid arrives at t_from_pu → +heat IN
                # mneg > 0 (forward, from→to):  fluid leaves at T_n        → -heat OUT
                if "t_from_pu" not in bvars:
                    continue
                terms.append(mpos * bvars["t_from_pu"] - mneg * T_n)

            elif branch.to_node_id == node.id:
                # This node is TO-end.
                # mneg > 0 (forward, from→to): fluid arrives at t_to_pu → +heat IN
                # mpos > 0 (backward, to→from): fluid leaves at T_n     → -heat OUT
                if "t_to_pu" not in bvars:
                    continue
                terms.append(mneg * bvars["t_to_pu"] - mpos * T_n)

        for child in network.childs_by_ids(node.child_ids):
            cm = child.model
            cvars = cm.vars
            if "mass_flow" not in cvars:
                continue
            m_ext = cvars["mass_flow"] * cvars.get("regulation", 1)
            # Children inject/withdraw at junction temperature (well-mixed model).
            # Sign: stored mass_flow is negative for injection (source convention),
            # so heat added = -m_ext * T_n (injection adds heat, withdrawal removes it).
            terms.append(-m_ext * T_n)

        return sum(terms) if terms else 0

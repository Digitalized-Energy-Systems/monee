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
from monee.model.phys.nonlinear.hf import SPECIFIC_HEAT_CAP_WATER

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
      ``StepState``.

    First-step / first-period heat balance
    --------------------------------------
    The very first step has no previous-state ``T(t−1)``.  Two modes
    control what LTC emits at that step:

    * **Anchored mode** (default; required for NLP / Newton-path solvers
      such as GEKKO/IPOPT).  The dynamic equation
      ``ρ·V·(T(0) − t_init)/Δt == net_heat`` is emitted with an explicit
      starting value ``t_init`` resolved through three layers,
      most-specific first:

      1. ``t_init_overrides[node_id]`` — per-junction explicit value.
      2. ``default_t_init`` (constructor) — a single value applied to all
         LTC junctions that don't have a specific override.
      3. The junction's own ``t_pu`` ``Var`` initialiser at the time the
         extension is attached (the legacy default — typically ``1.0``
         inherited from :class:`~monee.model.node.Junction`).

      For an *operational* timeseries study, the legacy default forces
      an artificial "warm-up transient" because ``1.0 = supply
      temperature`` rarely matches the network's true operating
      equilibrium.  Pass ``default_t_init`` (e.g. an estimate of the
      steady-state mean junction temperature) to skip that transient.

    * **Steady-state mode** (``first_step_steady_state=True``).  The
      inertia term is dropped at the first step and the LTC equation
      degenerates to ``net_convective_heat_in == 0`` — i.e. the
      network's regular steady-state heat balance.  ``T(0)`` then
      emerges from the surrounding equations rather than being pulled
      toward a fixed anchor.  Step 1 onward uses the standard dynamic
      equation with the solved ``T(0)`` as ``T(t−1)``.  This produces
      cleaner physics (no warm-up artefact at all) but **only converges
      reliably with MIP / LP-relaxation solvers** (Pyomo + Gurobi /
      SCIP); on NLP / Newton solvers the unanchored balance can leave
      the junction temperature locally underdetermined and the solve
      may fail.  Opt in only when you're using a MIP backend.

    Args:
        default_t_init: Optional global default for ``T(t−1)`` at the
            very first step in anchored mode.  ``None`` (default) means
            fall back to each junction's own ``t_pu`` ``Var``
            initialiser.  Ignored when
            ``first_step_steady_state=True``.
        t_init_overrides: Optional ``{node_id: value}`` map — per-junction
            explicit anchor in anchored mode, overriding both
            *default_t_init* and the ``Var`` initialiser.  Ignored when
            ``first_step_steady_state=True``.
        first_step_steady_state: If ``True``, emit
            ``net_convective_heat_in == 0`` at the first step instead of
            the anchored dynamic equation.  Recommended only for MIP
            solvers (Gurobi / SCIP via Pyomo).  Default ``False``.

    Examples
    --------
    Skip the supply-temperature warm-up by anchoring every LTC junction
    near the network's expected operating mean::

        net.add_extension(LumpedThermalCapacitance(default_t_init=0.93))

    Drop the anchor entirely (MIP backend only)::

        net.add_extension(LumpedThermalCapacitance(first_step_steady_state=True))

    Cold-start study with a few hot-water tanks initialised separately::

        net.add_extension(LumpedThermalCapacitance(
            default_t_init=0.4,                 # network mostly cold
            t_init_overrides={tank_id: 1.0},    # tanks pre-heated
        ))
    """

    def __init__(
        self,
        default_t_init: float | None = None,
        t_init_overrides: dict | None = None,
        first_step_steady_state: bool = False,
    ):
        self._ltc_rho_v: dict = {}  # {node_id: ρ·V_lumped  [kg]}
        self._ltc_initial_t_pu: dict = {}  # {node_id: initial t_pu value}
        self._ltc_constrained: set = set()
        self._default_t_init: float | None = (
            None if default_t_init is None else float(default_t_init)
        )
        self._t_init_overrides: dict = dict(t_init_overrides or {})
        self._first_step_steady_state: bool = bool(first_step_steady_state)

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

        # 3. Patch each LTC junction: resolve its first-step anchor (with
        #    precedence: t_init_overrides → default_t_init → existing Var
        #    initialiser) and replace t_pu with a fresh tracked Var so its
        #    solved value is persisted to StepState.
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
        first step/period.  In that case the resolved first-step anchor
        (see precedence in the class docstring) is used: an explicit
        ``t_init_overrides`` value, the ``default_t_init`` constructor
        argument, or the junction's own ``t_pu`` ``Var`` initialiser.  We
        keep an explicit numerical anchor on the first step rather than
        emitting the steady-state balance directly, because Newton/IPOPT
        path solvers (e.g. GEKKO) lose convergence when the heat-balance
        equation is fully unanchored on a degenerate node.

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
            if rho_v <= 0.0:
                self._ltc_constrained.add(node.id)
                continue

            net_heat = self._net_convective_heat(node, network)

            t_prev = temporal_state.get(node.id, "t_pu")
            if t_prev is None and self._first_step_steady_state:
                # No anchor: drop the inertia term and let T(0) emerge
                # from the steady-state heat balance.  MIP-only — see
                # class docstring.
                eqs.append(net_heat == 0)
            else:
                if t_prev is None:
                    # First step / period: use the resolved anchor (see
                    # ``prepare`` for precedence).
                    t_prev = self._ltc_initial_t_pu.get(node.id, 1.0)
                eqs.append(rho_v * (junc.t_pu - t_prev) / dt_s == net_heat)

            self._ltc_constrained.add(node.id)

        return eqs

    def inter_step_equations(self, network, ignored_nodes: set, step_state) -> list:
        """Delegate to :meth:`inter_temporal_equations` (backward compatibility)."""
        return self.inter_temporal_equations(network, ignored_nodes, step_state)

    def equations(self, network, ignored_nodes: set) -> list:
        """
        No extra equations are needed for single-step (non-timeseries) solves.
        The junction temperatures are determined by the existing heat balance.
        """
        return []

    def _net_convective_heat(self, node, network):
        """
        Return the solver expression for net convective heat flow INTO *node*.

        Sign convention (same as Junction.calc_signed_heat_flow but negated):
        the existing junction heat-balance uses an *outflow* convention
        (sum = 0, positive = outflow).  Here we need the net *inflow*,
        so each branch term is the negative of the junction balance term.

        Two formulation paths for the branch contribution:

        * **Plain (nonlinear) path** — the branch model carries
          ``t_from_pu`` / ``t_to_pu`` Vars that the nonlinear water
          formulation pins to the actual pipe-end temperatures (with heat
          loss along the pipe).  We compute the net mass-weighted enthalpy
          flow directly:

          - FROM-end of the branch:
            ``term = mpos · t_from_pu − mneg · T_n``
          - TO-end of the branch:
            ``term = mneg · t_to_pu − mpos · T_n``

        * **McCormick path** — the McCormick branch formulation
          (:mod:`monee.model.formulation.mccormick.water`) deliberately
          works around ``t_from_pu`` / ``t_to_pu`` (they are left as
          dangling Vars) and exposes the relaxed enthalpies ``H_out_mw``
          (sender side) and ``H_in_mw`` (receiver side) instead — those
          are exactly the ``c · m · τ`` quantities, with heat loss
          Taylor-linearised between them.  Reusing the dangling
          ``t_to_pu`` here would (a) silently relax the LTC equation
          because ``t_to_pu`` has no other constraint, and (b) reintroduce
          the bilinear ``m·τ`` products that McCormick was specifically
          designed to eliminate.  We instead use the McCormick enthalpies
          directly:

          - FROM-end:  ``term = − H_out_mw / scale_mw_per_kgs``  (heat OUT)
          - TO-end:    ``term = + H_in_mw  / scale_mw_per_kgs``  (heat IN)

          where ``scale_mw_per_kgs = c · t_ref / 1e6`` converts back from
          MW to ``kg·s⁻¹·t_pu`` units used by the rest of the LTC equation.
          McCormick already gates ``H_*_mw`` with ``on_off`` in its own
          equations, so no extra ``on_off`` factor is needed here.

        Children are modelled as injecting/withdrawing at junction
        temperature (well-mixed assumption).  Their mass_flow sign follows
        load convention: negative = injection, so net heat IN = -m_ext * T_n.
        """
        T_n = node.model.t_pu  # solver variable after injection
        terms = []
        scale_mw_per_kgs = SPECIFIC_HEAT_CAP_WATER * node.grid.t_ref / 1e6

        for branch in network.branches:
            if not isinstance(branch.grid, WaterGrid):
                continue
            bm = branch.model
            bvars = bm.vars
            if "mass_flow_pos" not in bvars or "mass_flow_neg" not in bvars:
                continue

            # McCormick path: use the relaxed enthalpies the branch already
            # carries instead of the dangling pipe-side temperatures.  We key
            # on H_out_mw / H_in_mw rather than a flag so this works whether
            # the network has McCormick partially or fully applied.
            is_mccormick = "H_out_mw" in bvars and "H_in_mw" in bvars

            if branch.from_node_id == node.id:
                if is_mccormick:
                    terms.append(-bvars["H_out_mw"] / scale_mw_per_kgs)
                else:
                    if "t_from_pu" not in bvars:
                        continue
                    on_off = bvars.get("on_off", 1)
                    mpos = bvars["mass_flow_pos"] * on_off
                    mneg = bvars["mass_flow_neg"] * on_off
                    terms.append(mpos * bvars["t_from_pu"] - mneg * T_n)

            elif branch.to_node_id == node.id:
                if is_mccormick:
                    terms.append(bvars["H_in_mw"] / scale_mw_per_kgs)
                else:
                    if "t_to_pu" not in bvars:
                        continue
                    on_off = bvars.get("on_off", 1)
                    mpos = bvars["mass_flow_pos"] * on_off
                    mneg = bvars["mass_flow_neg"] * on_off
                    terms.append(mneg * bvars["t_to_pu"] - mpos * T_n)

        for child in network.childs_by_ids(node.child_ids):
            cm = child.model
            cvars = cm.vars
            if "mass_flow" in cvars:
                m_ext = cvars["mass_flow"] * cvars.get("regulation", 1)
                # Children inject/withdraw at junction temperature (well-mixed model).
                # Sign: stored mass_flow is negative for injection (source convention),
                # so heat added = -m_ext * T_n (injection adds heat, withdrawal removes it).
                terms.append(-m_ext * T_n)
            if "q_mw_heat" in cvars:
                # Node-based heat injection / withdrawal (HeatGenerator, HeatLoad).
                # q_mw_heat [MW] → kg/s·t_pu via division by (c·t_ref/1e6).  Load
                # convention: positive = consumption (heat OUT), negative = generation
                # (heat IN) — we want net heat IN, so negate.
                q = cvars["q_mw_heat"] * cvars.get("regulation", 1)
                terms.append(-q / scale_mw_per_kgs)

        # Branch-level q_mw_heat (e.g. GasToHeatHG): absorbed at the branch's
        # TO-node.  Same sign handling as the child-based q_mw_heat above.
        for branch in network.branches:
            bm = branch.model
            bvars = bm.vars
            if "q_mw_heat" not in bvars:
                continue
            if branch.to_node_id != node.id:
                continue
            q = bvars["q_mw_heat"] * bvars.get("on_off", 1)
            terms.append(-q / scale_mw_per_kgs)

        return sum(terms) if terms else 0

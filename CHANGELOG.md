# Changelog

## 2.3.0 (unreleased)

### Behaviour changes

Results of existing scripts can change through the entries below; each one
fixes a wrong or silently dropped result.

**Physics**

- **Transformer impedance.** `Trafo` computed its per-unit impedance with an
  extra factor `(base_kv_from / base_kv_to)²`: far too small when built LV→HV,
  far too large HV→LV. It is now `vk_percent / 100 * sn_mva / sn_trafo_mva`
  (rated voltages = bus `base_kv`) and matches pandapower in both orientations.
  The dead `vn_trafo_lv` attribute is gone. `Trafo` and `create_trafo` take a
  fixed `tap` (off-nominal ratio at the from-node).
- **Line charging.** `PowerLine`/`create_line` take `c_nf_per_m`; the charging
  susceptance enters the AC, MISOCP and exact branch-flow formulations
  (`PowerGrid.f_hz`, default 50). Under the MISOCP and exact formulations the
  reported `i_*_ka`/`loading_*_pu` and the current-based loading limit now use
  the terminal current (series plus shunt current), as AC and pandapower do;
  without charging nothing changes.
- **Real-gas density.** Gas density, the linepack and its auto-sized initial
  and maximum values include the grid's compressibility `Z`
  (`rho = p M / (Z R T)`, about +2.7 % for the default L-gas grid). The NLP
  linepack uses the exact mean pressure of a steady isothermal pipe. Tests that
  encoded the ideal-gas value were updated.
- **Compound efficiencies.** `efficiency` (P2H, G2H) and
  `efficiency_power`/`efficiency_heat` (CHP, CHPHG) set on the compound, by
  `set` or a compound series, reach the control node; the input nameplate is
  kept and the output scales. Before, the control node kept its old value.
- **Velocity report.** `velocity_mps` uses the solved diameter when
  `diameter_m` is a decision variable, not the start value.
- **MISOCP current bound.** The cap of the lifted current at three times the
  thermal rating now applies only to a switchable branch, where it keeps the
  big-M finite. On a fixed branch it cut AC-feasible states, so a power flow
  over a line above three times its rating came out infeasible.

**Optimization semantics**

- **Heat duty is exact by default.** In an `OptimizationProblem`, every heat
  exchanger now delivers its duty exactly (`heat_duty="exact"`, the
  simulation semantics). Before, a plain problem silently allowed it to
  deliver less, so an objective that valued heat above 1 per MW could turn
  consumers off. `heat_duty="at_most"` keeps the old behaviour;
  `create_min_load_shedding_problem` uses it, since the unmet duty is the heat
  shed there. The internal exchanger of a unit at a plain regulation of 0
  keeps "at most" (an exact zero duty at zero flow is a degenerate row).
- **Coupling units at regulation 0 stay off in the economic dispatch.** A unit
  whose regulation is a plain 0 is no longer made a decision fixed to zero;
  the regulation a unit carries in is its availability.
- **Explicit load weights.** A weight returned by `weight_for_load` is used as
  given; the auto priority floor lifts only `demand_weight` and
  `generator_weight` (before, it scaled every weight, so objective terms added
  by hand lost their price relative to the shed). Generating heat exchangers
  under a callback get `gamma * w_g` instead of `gamma**2 * w_g`. A returned
  weight the lift leaves below the lifted default, so that the load is now
  shed before the default loads, raises a warning naming it.
- **Load weights reach the solve.** `weight_for_load` is called with the
  models of the network passed in, on every entry point:
  `run_energy_flow_optimization` (`copy=True` and `False`), `run_multi_period`,
  `run_timeseries` and the `Stepper`. It got the models of the solve's copy, so
  a callback keyed by the caller's models (a dict) matched no load and every
  load fell back to `demand_weight`. The callback therefore sees the network
  as passed in: values a timeseries, `data_overrides` or the `Stepper`'s
  working copy set per period or step are not visible to it (a weight read
  from `p_mw` is the base value, the same in every period); before, it saw
  them. A network the caller copied with `Network.copy()` is a network of its
  own, keyed by its own models.
- **Load-weight errors propagate.** An exception raised by `weight_for_load`
  reaches the caller. Before, it was swallowed and the load silently got
  `demand_weight`; a callback such as `lambda m: weights[m]` that raised
  `KeyError` for unlisted loads now needs `weights.get`.
- **Exact island heat bound in load shedding.** `bounds_ext_heat_mw` of
  `create_min_load_shedding_problem` writes the island heat branch by branch
  (exchanger duties and pipe losses) when every pipe runs a MIQCQP heat
  formulation. Both forms agree on every feasible point, but the relaxation of
  the nodal form let global solvers create or lose heat and hit time limits.
- **Constraints are no longer dropped.** On CasADi, an `equation` or
  `comp_equation` that returns a list applied none of it; it now applies all,
  and anything that is not a relation raises `TypeError`. `line_loading_limit`
  with a decision rating (`max_i_ka` a Var) is enforced instead of dropped.
  `add_to_controllable` called before the solve is no longer wiped by the
  problem's apply step; with a bare model outside a hook it raises
  `TypeError`. Inside a hook a component of the caller's network is mapped
  onto the network being solved, instead of editing the caller's model.
- **Candidate branches in the node balances.** A branch-flow line
  (MISOCP or exact formulation) with a decision `on_off` enters the bus
  balance as `flow - shunt * (1 - on_off)` instead of `flow * on_off`, and a
  water pipe or heat exchanger with a decision `on_off` enters the junction
  mass balance without the factor (every water formulation gates the flow,
  as the heat balance already assumed). Both agree on
  every binary point; the products gave a relaxation in which a fractional
  candidate line carried power without a balance. On an 8-hour sizing
  instance (3 pipe sizes, 2 cables on each of 2 routes) the root bound of the
  joint MIQCQP rises from 17454 to 25715 EUR (25789 with the design fixed)
  and the gap after 900 s (Gurobi, one thread) falls from 185 to 42 EUR;
  SCIP's first solution of the benchmark net's load shedding is now its
  optimum.
- **Switchable water branches.** A decision `on_off` on a water branch is left
  out of the nodal heat balance (the branch gates its own flow), so candidate
  pipes stay quadratic and solve on `solver="gurobi"`, with
  `LumpedThermalCapacitance` attached too; a fixed `on_off` is multiplied in
  once, as before.

**Solvers and post-solve**

- **Gurobi gap guard.** A native Gurobi solve that ends OPTIMAL while its
  incumbent lies more than ten times MIPGap/MIPGapAbs from its own bound is
  reported as `"warning"`/`"feasible"`, not optimal. The relative tolerance
  is taken of the larger of `|objective|` and the largest objective
  coefficient, so an optimum near zero is not called unproven for a gap that
  is tiny on the objective's scale (a waste-heat dispatch at a zero
  electricity price: gap 1.7e-3 EUR, coefficients up to 1394 EUR), and an
  objective in small units is held to the same test (the false optimum this
  guard was added for: incumbent 894.98 against a bound of -7.20, largest
  coefficient 6389, is flagged with or without `MIPGapAbs`). A two-phase
  lexicographic solve, native or on Pyomo, returns phase 1's outcome when
  that was not optimal, and Pyomo's `solver_stats` describe phase 1 (the user
  tier), not the aux objective of phase 2.
- **IPOPT starts from a simulation.** `CasADiSolver` optimizations first solve
  the network as a simulation at its setpoints and start IPOPT there
  (`start="simulation"`, the new default); `start="flat"` and
  `solver_stats["start"]` give the old start and say which one ran;
  `solver_stats["start_fallback"]` says why a simulation start fell back to
  flat (`"simulation_failed"`, `"optimization_failed"`). The start
  simulation runs with the usual retry chain. Local optima, and so results,
  can differ from 2.2.0.
- **Unconstrained variables on CasADi.** In a single-period or timeseries
  solve, a variable bounded on one side only that no constraint and no
  objective term mentions (the mass flows of a coupler's power-side transfer
  branch) is fixed at its start. The barrier
  pushed it towards infinity (6963 kg/s on a small district) and made IPOPT
  fail or stop at an acceptable level on feasible dispatches: in a
  waste-heat dispatch started from a heat-led schedule, 94 instead of 92 of
  96 hours converge and 90 instead of 86 end within 1 % of Gurobi. IPOPT is
  a local solver, so a few hours end in a different local optimum than
  before (here 2 of 96 worse, 7 better).
- **Idle units in multi-period IPOPT.** The enthalpy row of a CHP, G2H or P2H
  unit at a plain regulation of 0 is replaced by a temperature pin that passes
  the inflowing water through unheated, so multi-period CasADi runs with an
  idle unit converge. Under the McCormick heat balance a P2H keeps its row,
  where the pin would fix its hot junction.
- **Compound units under McCormick.** A running `create_p2h`, `create_g2h` or
  `create_chp` unit on the McCormick heat balance raises
  `NotImplementedError` naming the HG variants; before, it solved infeasible
  or silently wrong, because the balance gets no enthalpy from the unit's
  transfer branch. A unit at a regulation of 0 still solves.
- **Weymouth and heat restore.** After a single-period solve, and per period in
  `run_multi_period`, relaxed gas islands get the exact Weymouth flows and
  pressures for the solved injections (`restore_exact_weymouth`), and heat
  pressure islands whose pipes form a tree with one pinned pressure get exact
  Darcy-Weisbach pressures (`restore_exact_heat_pressures`). The bilinear heat
  hydraulics are epigraph-relaxed (the docstrings claimed exact); a slack
  meshed island gets a `relaxation` warning, whose advice now names the exact
  MIQCQP sibling instead of an NLP. All multi-period back-ends restore per
  period, and `MultiPeriodResult.warnings` / `period_warnings` (and
  `get_period_result(t).warnings`) carry the `relaxation` finding of an
  island the restore had to leave slack. That finding also names a slack
  epigraph on an idle pipe (a pressure drop at zero flow), which the flow
  floor of the check skipped before.
- **Slack MISOCP cones.** A branch-flow cone left slack on a resistive branch
  is a `relaxation` finding: the reported flows, losses and voltages are not
  AC-feasible there (typically a surplus burned in fictitious losses). Before,
  only zero-resistance branches were checked.
- **Time limits on Pyomo.** A limit stop with an incumbent reports `obj`, the
  dual `bound` and `gap` in `solver_stats`, also when there is no incumbent
  (`obj` NaN). `run_multi_period` returns such an incumbent with
  `success=True` and the limit as `termination_condition` instead of raising
  `RuntimeError`; a limit stop without one raises a `RuntimeError` naming the
  termination instead of Pyomo's "Cannot load a SolverResults object".
- **Unsupplied islands.** The pruning no longer bridges water islands through
  an inactive coupler, so a heat network with every substation and producer
  off solves instead of stopping at `Not_Enough_Degrees_Of_Freedom`.
  `SolverResult.dropped_islands` lists the islands dropped for lack of a slack
  or source, and the `pruned` warning now also names the couplers switched off
  because a port lies in a dropped island. An island that still has demand
  keeps the solve successful, but `SolverResult.dropped_demand` gives that
  demand in MW per component and the `pruned` warning names the island and
  its total. A gas-to-heat unit switched off this way reports no gas draw on
  its control node (it reported its setpoint), as the CHP already did.
- **Multi-period results.** Pyomo and GEKKO multi-period results run the
  post-processing (`t_k` and `pressure_pa` were 0), and Pyomo
  reports `solver_status` and `termination_condition`.
- **Validation.** `bound_active` skips the unused voltages of coupler control
  nodes and the free temperature of an idle unit's control node;
  `shed_headroom` stays silent on a proven optimum of Gurobi or SCIP. The
  `dof` checks count a variable with equal bounds (a gas gate closed by zero
  import and export caps) as fixed, not free.
- **Warm starts.** The `Stepper` warm start now reuses each Var's solved
  value; before, it made a step about ten times slower than a cold start, so
  `warm_start=False` is no longer needed. Network files skip that solve state
  unless `include_solve_state=True`. The CasADi build-once timeseries path now
  also runs with `LumpedThermalCapacitance` and `GasLinepack` attached.
- **Messages.** An empty Gurobi IIS no longer claims the model "may be
  unbounded" (the report exists only for a proven infeasible status); it says
  why it is empty or not minimal. A `PostProcess` report such as `t_k` used in
  a constraint raises a `TypeError` that names the decision variable to use
  (`t_pu`). A failed solve no longer reports heat exchangers that "deliver
  0 MW of their setpoint" (it judged the pre-solve values). The warning that
  a zero setpoint locks a promoted variable is logged once per model type and
  attribute in a process, not for every PV unit at night (in a long-lived
  session a later network's zero lock of the same type and attribute is not
  logged again). IPOPT's acceptable-level
  warning about an objective far from optimal is no longer logged for a
  simulation or a solve without objective; the residual check covers them.
  The CasADi build-once timeseries and multi-period paths judge this the
  same way as the per-step solve (the timeseries path muted it for every
  optimization, the multi-period path logged it for objective-free solves).
  `segment_length_m` of zero or less raises `ValueError` on every heat
  formulation that takes it (the MIQCQP and PWL ones failed at solve time).

### Added

- `OptimizationProblem(heat_duty=...)`, `controllable_design` and
  `candidate_branches` for investment decisions shared by all periods.
- `PlugFlowTransport`, a Lagrangian plug-flow heat transport extension. The
  steady first step fills each pipe with 16 parcels of its age profile, so
  the outlet ramps after a flow drop or reversal.
- Pipe heat-loss law options: `make_heat_nlp_formulation(heat_loss="exact")`
  (exponential decay) and `segment_length_m` on the NLP, bilinear MIQCQP
  (`make_heat_nonconvex_miqcqp_formulation`, new) and PWL heat formulations
  (the rational law over equal cells). The default law is unchanged.
- `run_multi_period(..., solver_options=...)` and
  `PyomoMultiPeriodSolver(solver_name, solver_options)`;
  `MultiPeriodResult.solver_stats`.
- `GurobipySolver(iis=False)` skips the IIS of an infeasible solve.
- `solver_stats["start_s"]` on CasADi optimizations: the time spent on the
  start simulation.
- `monee.network.pressure_budget_diameter`, the pressure criterion of the
  automatic pipe sizing (the private `_pressure_budget_diameter` stays an
  alias). `SolverResult.dropped_demand`.
- `CasADiSolver(start=...)`, `Trafo(tap=...)`, `create_line(c_nf_per_m=...)`,
  `PowerGrid.f_hz`, `SolverResult.dropped_islands`.
- `monee.__version__`.

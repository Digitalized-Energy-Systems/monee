import logging
import types

import pyomo.environ as pyo
from pyomo.opt import SolverStatus

from monee.model import (
    Const,
    GenericModel,
    Intermediate,
    IntermediateEq,
    Network,
    Var,
)

# detect islanding config for topology-aware pre-filtering
from monee.model.islanding.core import NetworkIslandingConfig
from monee.problem.core import OptimizationProblem
from monee.simulation.step_state import StepState

from .core import (
    SolverInterface,
    SolverResult,
    as_iter,
    compute_bound_violations,
    filter_intermediate_eqs,
    find_ignored_nodes,
    ignore_branch,
    ignore_child,
    ignore_compound,
    ignore_node,
    inject_vars,
    mark_ignored_components,
    persist_solution,
    withdraw_vars,
)

_log = logging.getLogger(__name__)

DEFAULT_SOLVER_OPTIONS = {}

# Per-solver defaults applied on top of ``DEFAULT_SOLVER_OPTIONS``.  Tuned
# for the McCormick-DHS + MISOCP electrical formulation:
#   ``ScaleFlag=2``  — most aggressive geometric scaling, useful even with a
#                      well-conditioned matrix because the SOC and bilinear
#                      blocks still benefit.
#   ``MIPFocus=2``   — focus on proving optimality, not finding incumbents.
#                      For the multi-energy load-shedding problem the LP
#                      root bound is already at the optimum (gap ~0.001 %),
#                      so the solver's job is to *close the gap*, not to
#                      discover an incumbent.
#   ``MIPGap=1e-3``  — 0.1 % relative gap.  At MW objective scale this is
#                      ~kW precision, which already exceeds the model's
#                      physical accuracy.  The default 1e-4 spends most of
#                      B&B proving an extra digit nobody can use.
#
# ``NumericFocus`` is intentionally left at default (0).  It was useful while
# the matrix range spanned 1e-6…1e+7 (Reynolds PWL breakpoints + pressure_pa
# coefficients), but once those were rescaled the matrix range collapsed to
# ~1e-6…4e+2 and the extra numerical conservatism became net-negative
# (Linear-HX measured ~3× slower with NumericFocus=2 once the matrix was
# fixed).
PER_SOLVER_OPTIONS = {
    "gurobi": {
        "ScaleFlag": 2,
        "MIPFocus": 2,
        "MIPGap": 1e-3,
    },
}


class PyomoPWLImpl:
    def __init__(self, pm: pyo.ConcreteModel, pw_repn: str = "SOS2"):
        self.pm = pm
        self.pw_repn = pw_repn
        # counter to ensure unique component names
        if not hasattr(pm, "_pwl_counter"):
            pm._pwl_counter = 0

    def piecewise_eq(self, y, x, xs, ys, name: str | None = None):
        pm = self.pm
        pm._pwl_counter += 1
        k = pm._pwl_counter
        if name is None:
            name = f"pwl_{k}"

        xs = list(xs)
        ys = list(ys)

        pw = pyo.Piecewise(
            y,
            x,
            pw_pts=xs,
            f_rule=ys,  # list form: y[i] at xs[i]
            pw_constr_type="EQ",
            pw_repn=self.pw_repn,  # "SOS2" (tight), or "BIGM"
            warn_domain_coverage=False,
        )
        setattr(pm, name, pw)


class PyomoSolver(SolverInterface):
    """ """

    def __init__(self):
        pass

    # --------- Injection / Withdrawal ---------

    @staticmethod
    def inject_pyomo_vars_attr(
        pm: pyo.ConcreteModel, target: GenericModel, prefix: str
    ):
        """Replace Var/Const fields on `target` with Pyomo Var / numeric constants.

        ``initialize`` is clamped to the current ``[min, max]`` bounds before
        being handed to Pyomo.  The previous solve's stored value can fall
        outside the *current* bounds when an :class:`OptimizationProblem`
        applies a tighter range (e.g. ``bounds_heat=(0.8, 1.15)`` after a
        prior call solved with the intrinsic ``[0, 2]`` Var range).  Clamping
        keeps the initial guess feasible and suppresses W1002.
        """
        for key, value in target.__dict__.items():
            if isinstance(value, Var):
                init = value.value
                if init is not None:
                    if value.min is not None and init < value.min:
                        init = value.min
                    if value.max is not None and init > value.max:
                        init = value.max
                v = pyo.Var(
                    domain=pyo.Integers if value.integer else pyo.Reals,
                    bounds=(value.min, value.max),
                    initialize=init,
                )
                setattr(pm, f"{prefix}__{key}", v)
                setattr(target, key, v)
            elif type(value) is Const:
                setattr(target, key, float(value.value))

    # Tolerance for snapping near-bound values back into bounds during
    # withdrawal.  Solver outputs commonly carry sub-epsilon noise (e.g.
    # ``-1.4e-16`` for a Var with bound ``[0, None]``).  Re-injecting that
    # noise as ``initialize`` for the next solve triggers W1002 even though
    # the value is numerically zero.
    _BOUND_SNAP_TOL = 1e-9

    @staticmethod
    def withdraw_pyomo_vars_attr(target: GenericModel):
        """Convert Pyomo Var values back into Var objects.

        Sanitises three kinds of noise the next ``inject_pyomo_vars_attr``
        would otherwise pass to Pyomo as ``initialize``:

        - **Integer flag lost** → preserves ``integer=value.is_integer()`` and
          snaps near-integer floats to ``int(round(val))``.  Without this,
          relaxed values like ``1.14e-6`` re-enter as the initial guess for a
          fresh ``pyo.Var(domain=Integers)`` and trigger W1001
          (``not in domain Integers``).
        - **Bound noise** → snaps continuous values within
          ``_BOUND_SNAP_TOL`` of a bound to the bound itself, suppressing
          W1002 (``outside the bounds``) caused by sub-machine-epsilon drift.
        - **Missing value** → falls back to ``0`` when the solver failed and
          ``pyo.value`` is unavailable, so withdrawal cannot raise.
        """
        for key, value in target.__dict__.items():
            if isinstance(value, pyo.Var):
                lb, ub = value.bounds if value.bounds is not None else (None, None)
                is_integer = value.is_integer()
                val = pyo.value(value, exception=False)
                if val is None:
                    val = 0
                if is_integer:
                    val = int(round(val))
                else:
                    tol = PyomoSolver._BOUND_SNAP_TOL
                    if lb is not None and lb - tol <= val < lb:
                        val = lb
                    if ub is not None and ub < val <= ub + tol:
                        val = ub
                setattr(target, key, Var(value=val, min=lb, max=ub, integer=is_integer))
            elif isinstance(value, pyo.Expression):
                expr_val = pyo.value(value, exception=False)
                setattr(
                    target,
                    key,
                    Intermediate(value=expr_val if expr_val is not None else 0),
                )

    # --------- Constraint helpers ---------

    @staticmethod
    def _add_equation(pm, expr, name=None):
        # Trivial bool equations are not valid Pyomo Constraint expressions:
        # True = tautology (no-op); False = structural infeasibility from an
        # over-deactivated node/branch.  Skip both so the load-shedding
        # objective can drive the solution instead of crashing construction.
        if isinstance(expr, bool):
            return
        if name is not None:
            setattr(pm, name, pyo.Constraint(expr=expr))
        else:
            pm.cons.add(expr)

    @staticmethod
    def _sanitize_name(name):
        """Make a string safe for use as a Pyomo component name."""
        return (
            name.replace("(", "").replace(")", "").replace(", ", "_").replace(" ", "_")
        )

    def _add_equations(self, pm, exprs, name_prefix=None):
        for i, e in enumerate(exprs):
            if name_prefix is not None:
                name = self._sanitize_name(f"{name_prefix}_eq_{i}")
            else:
                name = None
            self._add_equation(pm, e, name=name)

    @staticmethod
    def _process_intermediate_eqs(pm, model_obj, equations):
        """
        GEKKO Intermediate: two cases in your original code:
        - If attr is not Intermediate: enforce equality constraint
        - If attr is Intermediate: create an Intermediate and assign it

        In Pyomo, use Expression for intermediates.
        """
        for intermediate_eq in [eq for eq in equations if type(eq) is IntermediateEq]:
            attr_val = getattr(model_obj, intermediate_eq.attr)
            eq = (
                intermediate_eq.eq()
                if isinstance(intermediate_eq.eq, types.FunctionType)
                else intermediate_eq.eq
            )

            # If the target attribute is not "Intermediate" wrapper, force equality:
            if type(attr_val) is Intermediate:
                # Create a Pyomo Expression and attach it
                e = pyo.Expression(expr=eq)

                # Put on pm for uniqueness + easy value extraction
                name = f"expr__{id(model_obj)}__{intermediate_eq.attr}"
                setattr(pm, name, e)
                setattr(model_obj, intermediate_eq.attr, e)

    # --------- Core solve ---------

    def solve(
        self,
        input_network: Network,
        optimization_problem: OptimizationProblem = None,
        exclude_unconnected_nodes: bool = False,
        solver_name: str = "scip",
        debug=False,
        step_state: StepState = None,
    ):
        pm = pyo.ConcreteModel()
        pm.cons = pyo.ConstraintList()
        pm.obj_exprs = []

        network = input_network.copy()

        # Phase 1: add Var placeholders for all NetworkAspect extensions
        for ext in network.extensions:
            ext.prepare(network)

        islanding_config = next(
            (e for e in network.extensions if isinstance(e, NetworkIslandingConfig)),
            None,
        )

        # Compute ignored_nodes BEFORE _apply() so that controllable filters
        # (e.g. controllable_demands) honouring component.ignored correctly
        # exclude disconnected components.
        ignored_nodes = set()
        if optimization_problem is None or exclude_unconnected_nodes:
            ignored_nodes = find_ignored_nodes(network, islanding_config)
            if ignored_nodes:
                mark_ignored_components(network, ignored_nodes)

        if optimization_problem is not None:
            optimization_problem._apply(network)

        nodes = network.nodes
        for node in nodes:
            if ignore_node(node, network, ignored_nodes):
                continue
            for child in network.childs_by_ids(node.child_ids):
                if child.active:
                    child.model.overwrite(node.model, node.grid)

        branches = network.branches
        compounds = network.compounds

        # inject vars
        inject_vars(
            lambda model, comp, cat: PyomoSolver.inject_pyomo_vars_attr(
                pm, model, prefix=f"{cat}_{comp.id}"
            ),
            nodes,
            branches,
            compounds,
            network,
            ignored_nodes,
        )

        # Phase 1.5: let extensions mark nodes before equations are assembled.
        if step_state is not None:
            for ext in network.extensions:
                ext.activate_timeseries(network, ignored_nodes, step_state=step_state)
            self.mark_temporal_components(network, ignored_nodes)

        # init branches
        self.init_branches(branches)

        # build constraints & objectives
        self.process_equations_nodes_childs(pm, network, nodes, ignored_nodes)
        self.process_equations_branches(pm, network, branches, ignored_nodes)
        self.process_equations_compounds(pm, network, compounds, ignored_nodes)

        # OXF components
        if optimization_problem is not None:
            self.process_oxf_components(pm, network, optimization_problem)
        else:
            self.process_internal_oxf_components(pm, network)

        if step_state is not None:
            self.process_inter_step_equations(
                pm,
                network,
                nodes,
                branches,
                compounds,
                ignored_nodes,
                step_state,
                optimization_problem=optimization_problem,
            )
            # Also collect inter-step equations from NetworkAspect extensions.
            for ext in network.extensions:
                self._add_equations(
                    pm, ext.inter_step_equations(network, ignored_nodes, step_state)
                )
                self._add_equations(
                    pm, ext.inter_temporal_equations(network, ignored_nodes, step_state)
                )

        # Phase 2: add NetworkAspect extension equations after variable injection
        for ext in network.extensions:
            self._add_equations(pm, ext.equations(network, ignored_nodes))

        # single objective: sum of collected objective expressions
        pm.obj = pyo.Objective(expr=sum(pm.obj_exprs), sense=pyo.minimize)

        # solve
        solver = pyo.SolverFactory(solver_name)

        for k, v in DEFAULT_SOLVER_OPTIONS.items():
            solver.options[k] = v
        for k, v in PER_SOLVER_OPTIONS.get(solver_name, {}).items():
            solver.options[k] = v

        solve_kwargs = {"tee": debug}
        if getattr(solver, "warm_start_capable", lambda: False)():
            solve_kwargs["warmstart"] = True
        result = solver.solve(pm, **solve_kwargs)

        success = result.solver.status == SolverStatus.ok
        if not success:
            from monee.solver.infeasibility import diagnose_infeasibility

            report = diagnose_infeasibility(
                pm, solver_name=solver_name, compute_mis_flag=False, tol=0.001
            )
            _log.warning(
                "Pyomo solve failed (status=%s). Infeasibility report:\n%s",
                result.solver.status,
                report.summary(),
            )

        # pull values back into your objects
        withdraw_vars(
            PyomoSolver.withdraw_pyomo_vars_attr, nodes, branches, compounds, network
        )
        persist_solution(network, input_network)
        violations = compute_bound_violations(nodes, branches, compounds, network)

        # objective value
        obj_val = pyo.value(pm.obj)

        solver_result = SolverResult(
            network,
            network.as_result_dataframe_dict(),
            obj_val,
            success,
            violations,
        )
        if not success:
            solver_result.infeasibility_report = report
        return solver_result

    # --------- Your original processing rewritten to Pyomo ---------

    def process_internal_oxf_components(self, pm, network):
        for constraint in network.constraints:
            self._add_equation(pm, constraint(network))

        obj = None
        for objective in network.objectives:
            obj = objective(network) if obj is None else (obj + objective(network))
        if obj is not None:
            pm.obj_exprs.append(obj)

    def process_oxf_components(
        self,
        pm,
        network,
        optimization_problem: OptimizationProblem,
        period_index=None,
    ):
        if optimization_problem.constraints is not None and (
            not optimization_problem.constraints.empty
        ):
            self._add_equations(
                pm,
                optimization_problem.constraints.all(
                    network, period_index=period_index
                ),
            )

        obj = None
        for objective in (
            optimization_problem.objectives.all(network, period_index=period_index)
            if optimization_problem.objectives is not None
            else []
        ):
            obj = objective if obj is None else (obj + objective)
        if obj is not None:
            pm.obj_exprs.append(obj)

    def process_equations_compounds(self, pm, network, compounds, ignored_nodes):
        for compound in compounds:
            if ignore_compound(compound, ignored_nodes):
                continue

            equations = compound.equations(network)

            if equations is not None:
                equations = as_iter(equations)
                self._process_intermediate_eqs(pm, compound, equations)
                self._add_equations(
                    pm,
                    filter_intermediate_eqs(equations),
                    name_prefix=f"compound_{compound.id}",
                )

    def process_equations_nodes_childs(
        self, pm, network: Network, nodes, ignored_nodes
    ):
        # Pyomo math operators
        sin_impl = pyo.sin
        cos_impl = pyo.cos
        abs_impl = abs
        sqrt_impl = pyo.sqrt
        log_impl = pyo.log

        for node in nodes:
            if ignore_node(node, network, ignored_nodes):
                continue

            node_childs = network.childs_by_ids(node.child_ids)
            grid = node.grid

            from_branches = [
                network.branch_by_id(bid).model
                for bid in node.from_branch_ids
                if not ignore_branch(network.branch_by_id(bid), network, ignored_nodes)
            ]
            to_branches = [
                network.branch_by_id(bid).model
                for bid in node.to_branch_ids
                if not ignore_branch(network.branch_by_id(bid), network, ignored_nodes)
            ]
            connected_childs = [
                child.model
                for child in node_childs
                if not ignore_child(child, ignored_nodes)
            ]

            equations = as_iter(
                node.equations(
                    grid,
                    from_branches,
                    to_branches,
                    connected_childs,
                    sin_impl=sin_impl,
                    cos_impl=cos_impl,
                    abs_impl=abs_impl,
                    sqrt_impl=sqrt_impl,
                    log_impl=log_impl,
                )
            )

            for expr in node.minimize(
                grid, from_branches, to_branches, connected_childs, sqrt_impl=sqrt_impl
            ):
                pm.obj_exprs.append(expr)

            node_eqs = [eq for eq in equations if not isinstance(eq, bool)]
            self._process_intermediate_eqs(pm, node.model, node_eqs)
            self._add_equations(
                pm, filter_intermediate_eqs(node_eqs), name_prefix=f"node_{node.id}"
            )

            for child in node_childs:
                if ignore_child(child, ignored_nodes):
                    continue
                for expr in child.minimize(grid, node, sqrt_impl=sqrt_impl):
                    pm.obj_exprs.append(expr)
                child_eqs = as_iter(child.equations(grid, node))
                self._process_intermediate_eqs(pm, child.model, child_eqs)
                self._add_equations(
                    pm,
                    filter_intermediate_eqs(child_eqs),
                    name_prefix=f"child_{child.id}",
                )

    def process_equations_branches(self, pm, network, branches, ignored_nodes):
        sin_impl = pyo.sin
        cos_impl = pyo.cos
        abs_impl = abs
        sqrt_impl = pyo.sqrt
        log_impl = pyo.log
        pwl_impl = PyomoPWLImpl(pm, pw_repn="SOS2")  # <-- add this

        def if_impl(*args, **kwargs):
            raise NotImplementedError(
                "Replace GEKKO if2/if3 with a Pyomo Piecewise / big-M formulation."
            )

        def max_impl(*args, **kwargs):
            raise NotImplementedError(
                "Replace GEKKO max2 with Pyomo max_ (or explicit constraints)."
            )

        def sign_impl(*args, **kwargs):
            raise NotImplementedError(
                "Replace GEKKO sign2/sign3 with a Pyomo formulation (often binary)."
            )

        for branch in branches:
            if ignore_branch(branch, network, ignored_nodes):
                continue

            grid = branch.grid

            branch_eqs = as_iter(
                branch.equations(
                    grid,
                    network.node_by_id(branch.from_node_id).model,
                    network.node_by_id(branch.to_node_id).model,
                    sin_impl=sin_impl,
                    cos_impl=cos_impl,
                    if_impl=if_impl,
                    abs_impl=abs_impl,
                    max_impl=max_impl,
                    sign_impl=sign_impl,
                    log_impl=log_impl,
                    sqrt_impl=sqrt_impl,
                    pwl_impl=pwl_impl,
                )
            )

            for expr in branch.minimize(
                grid,
                network.node_by_id(branch.from_node_id).model,
                network.node_by_id(branch.to_node_id).model,
                sqrt_impl=sqrt_impl,
            ):
                pm.obj_exprs.append(expr)

            self._process_intermediate_eqs(pm, branch.model, branch_eqs)
            self._add_equations(
                pm,
                filter_intermediate_eqs(branch_eqs),
                name_prefix=f"branch_{branch.id}",
            )

import types

import pyomo.environ as pyo

from monee.model import (
    Const,
    GenericModel,
    Intermediate,
    IntermediateEq,
    Network,
    Var,
)
from monee.problem.core import OptimizationProblem

from .core import (
    SolverInterface,
    SolverResult,
    as_iter,
    filter_intermediate_eqs,
    find_ignored_nodes,
    ignore_branch,
    ignore_child,
    ignore_compound,
    ignore_node,
)

DEFAULT_SOLVER_OPTIONS = {}


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
        """
        Replace Var/Const fields on `target` with Pyomo Var / numeric constants.
        """
        for key, value in target.__dict__.items():
            if type(value) is Var:
                # Create a unique Pyomo Var on the ConcreteModel
                v = pyo.Var(
                    domain=pyo.Integers if value.integer else pyo.Reals,
                    bounds=(value.min, value.max),
                    initialize=value.value,
                )
                setattr(pm, f"{prefix}__{key}", v)
                setattr(target, key, v)

            elif type(value) is Const:
                # Pyomo can use plain floats; Param is optional.
                setattr(target, key, float(value.value))

    @staticmethod
    def inject_nans(target: GenericModel):
        """
        If a component is ignored, replace its Vars/Consts with NaN placeholders
        (keeps downstream code from crashing when it tries to read attributes).
        """
        for key, value in target.__dict__.items():
            if isinstance(value, Const):
                setattr(target, key, Const(float("nan")))
            if isinstance(value, Var | Const):
                setattr(target, key, Var(float("nan"), max=value.max, min=value.min))

    @staticmethod
    def inject_pyomo_vars(pm, nodes, branches, compounds, network, ignored_nodes):
        for branch in branches:
            if ignore_branch(branch, network, ignored_nodes):
                branch.ignored = True
                PyomoSolver.inject_nans(branch.model)
                continue
            PyomoSolver.inject_pyomo_vars_attr(
                pm, branch.model, prefix=f"branch_{branch.id}"
            )

        for node in nodes:
            if ignore_node(node, network, ignored_nodes):
                node.ignored = True
                for child in network.childs_by_ids(node.child_ids):
                    child.ignored = True
                    PyomoSolver.inject_nans(child.model)
                PyomoSolver.inject_nans(node.model)
                continue

            PyomoSolver.inject_pyomo_vars_attr(pm, node.model, prefix=f"node_{node.id}")

            for child in network.childs_by_ids(node.child_ids):
                if ignore_child(child, ignored_nodes):
                    child.ignored = True
                    PyomoSolver.inject_nans(child.model)
                    continue
                PyomoSolver.inject_pyomo_vars_attr(
                    pm, child.model, prefix=f"child_{child.id}"
                )

        for compound in compounds:
            if ignore_compound(compound, ignored_nodes):
                compound.ignored = True
                PyomoSolver.inject_nans(compound.model)
                continue
            PyomoSolver.inject_pyomo_vars_attr(
                pm, compound.model, prefix=f"compound_{compound.id}"
            )

    @staticmethod
    def withdraw_pyomo_vars_attr(target: GenericModel):
        """
        Convert Pyomo Var values back into your Var objects.
        Constants stay constants.
        """
        for key, value in target.__dict__.items():
            if isinstance(value, pyo.Var):
                lb, ub = value.bounds if value.bounds is not None else (None, None)
                val = pyo.value(value)
                setattr(target, key, Var(value=val, min=lb, max=ub))
            elif isinstance(value, pyo.Expression):
                setattr(target, key, Intermediate(value=pyo.value(value)))

    @staticmethod
    def withdraw_pyomo_vars(nodes, branches, compounds, network):
        for branch in branches:
            PyomoSolver.withdraw_pyomo_vars_attr(branch.model)
        for node in nodes:
            PyomoSolver.withdraw_pyomo_vars_attr(node.model)
            for child in network.childs_by_ids(node.child_ids):
                PyomoSolver.withdraw_pyomo_vars_attr(child.model)
        for compound in compounds:
            PyomoSolver.withdraw_pyomo_vars_attr(compound.model)

    # --------- Constraint helpers ---------

    @staticmethod
    def _add_equation(pm, expr):
        """
        GEKKO m.Equation(expr) becomes pm.cons.add(expr).
        `expr` must be a Pyomo relational expression (==, <=, >=).
        """
        pm.cons.add(expr)

    @staticmethod
    def _add_equations(pm, exprs):
        for e in exprs:
            pm.cons.add(e)

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
        draw_debug: bool = False,
        exclude_unconnected_nodes: bool = False,
        solver_name: str = "scip",
    ):
        pm = pyo.ConcreteModel()
        pm.cons = pyo.ConstraintList()
        pm.obj_exprs = []

        network = input_network.copy()

        if optimization_problem is not None:
            optimization_problem._apply(network)

        ignored_nodes = set()
        if optimization_problem is None or exclude_unconnected_nodes:
            ignored_nodes = find_ignored_nodes(network)

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
        self.inject_pyomo_vars(pm, nodes, branches, compounds, network, ignored_nodes)

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

        # single objective: sum of collected objective expressions
        pm.obj = pyo.Objective(expr=sum(pm.obj_exprs), sense=pyo.minimize)

        # solve
        solver = pyo.SolverFactory(solver_name)

        for k, v in DEFAULT_SOLVER_OPTIONS.items():
            solver.options[k] = v

        solver.solve(pm, tee=True)

        # pull values back into your objects
        self.withdraw_pyomo_vars(nodes, branches, compounds, network)

        # objective value
        obj_val = pyo.value(pm.obj)

        return SolverResult(network, network.as_result_dataframe_dict(), obj_val)

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
        self, pm, network, optimization_problem: OptimizationProblem
    ):
        if optimization_problem.constraints is not None and (
            not optimization_problem.constraints.empty
        ):
            self._add_equations(pm, optimization_problem.constraints.all(network))

        obj = None
        for objective in optimization_problem.objectives.all(network):
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
                self._add_equations(pm, filter_intermediate_eqs(equations))

    def process_equations_nodes_childs(
        self, pm, network: Network, nodes, ignored_nodes
    ):
        # Pyomo math operators
        sin_impl = pyo.sin
        cos_impl = pyo.cos
        abs_impl = abs
        sqrt_impl = pyo.sqrt
        log_impl = pyo.log

        # IMPORTANT:
        # GEKKO's if2/if3/max2/sign2/sign3 do NOT map 1:1 to Pyomo.
        # You must implement these yourself (typically with binaries + big-M or Piecewise),
        # or change your model.equations() to avoid them for Pyomo runs.
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
                    if_impl=if_impl,
                    abs_impl=abs_impl,
                    max_impl=max_impl,
                    sign_impl=sign_impl,
                    sqrt_impl=sqrt_impl,
                    log_impl=log_impl,
                )
            )

            for expr in node.minimize(
                grid, from_branches, to_branches, connected_childs, sqrt_impl=sqrt_impl
            ):
                pm.obj_exprs.append(expr)

            node_eqs = [eq for eq in equations if type(eq) is not bool or not eq]
            self._process_intermediate_eqs(pm, node.model, node_eqs)
            self._add_equations(pm, filter_intermediate_eqs(node_eqs))

            for child in node_childs:
                if ignore_child(child, ignored_nodes):
                    continue
                for expr in child.minimize(grid, node, sqrt_impl=sqrt_impl):
                    pm.obj_exprs.append(expr)
                child_eqs = as_iter(child.equations(grid, node))
                self._process_intermediate_eqs(pm, child.model, child_eqs)
                self._add_equations(pm, filter_intermediate_eqs(child_eqs))

    def init_branches(self, branches):
        for branch in branches:
            branch.model.init(branch.grid)

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
            self._add_equations(pm, filter_intermediate_eqs(branch_eqs))

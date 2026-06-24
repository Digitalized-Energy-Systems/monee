"""In-process CasADi/IPOPT backend for monee.

Motivation
----------
GEKKO's per-solve wall-clock is dominated not by the numerical solve but by
APMonitor re-serialising + re-parsing + re-compiling the whole symbolic model
into a subprocess on *every* solve (~1.9 s for mv_oberrhein, vs ~0.07 s of
actual IPOPT). CasADi builds the NLP once as an in-memory expression graph and
calls IPOPT in-process - no text model, no files, no subprocess.

How it reuses monee
-------------------
monee's formulations are backend-agnostic: they build equations from the
*injected* variable objects and take math hooks (``sin_impl``, ``cos_impl``,
...). So only three pieces are CasADi-specific and everything else (network
prep, ``process_equations_*``, post-processing, result frames) is reused
unchanged:

* :class:`CasSym` - wraps a CasADi ``SX``. Its arithmetic operators build CasADi
  expressions; its ``==`` / ``<=`` / ``>=`` capture a constraint record
  (residual + op) instead of collapsing to a bool. Injected as each model
  variable, so monee's unchanged ``equations()`` bodies produce CasADi
  expressions.
* :class:`CasModel` - quacks like the GEKKO model object for the subset of the
  API monee calls (``sin`` / ``cos`` / ``sqrt`` / ..., ``Equation(s)``, ``Obj``,
  ``Intermediate``). This lets us call monee's existing ``process_equations_*``
  passes verbatim.
* :class:`CasADiSolver` - mirrors :meth:`GEKKOSolver.solve`'s network prep, then
  assembles a single ``nlpsol('ipopt', ...)`` and writes the solution back.

Temporal coupling
-----------------
:class:`CasADiSolver` processes inter-step / inter-temporal equations when a
``step_state`` is supplied, so timeseries with storage SOC, ramp limits and
linepack work through the standard :func:`monee.run_timeseries` per-step loop.
:class:`CasADiMultiPeriodSolver` builds one NLP spanning all T periods with
``inter_period`` coupling (``backend="casadi"`` in
:func:`monee.run_multi_period`).

:class:`CasADiTimeseries` exploits the in-process compiled graph for the
headline win on *memory-less* timeseries: it assembles the NLP + IPOPT solver
**once**, with every time-varying input declared as a CasADi *parameter*, then
re-solves each step by only setting the parameter vector and warm-starting from
the previous step - no rebuild, no recompile, no subprocess per step. (This
build-once reuse does not carry inter-step state, so :func:`monee.run_timeseries`
only takes it for networks without temporal coupling and otherwise uses the
per-step loop above.)

Scope
-----
Electricity AC, gas and heat NLP - power flow, OPF, storage/temporal coupling and
multi-period. Gas/heat tabulated relations (the ``pwl_impl`` hook) are realised
as a smooth cubic B-spline ``interpolant`` (:class:`CasADiCubicSplineImpl`) - the
smooth analogue of the MILP backends' piecewise-linear form and the direct
counterpart of GEKKO's ``cspline``. Integer variables are relaxed (IPOPT is
continuous, exactly like the GEKKO/IPOPT default); cases that must *enforce*
integrality (MINLP) need APOPT or a MIP backend (gurobi/pyomo).
"""

from __future__ import annotations

import logging
import math
import time

import numpy as np

from monee.model import Const, Intermediate, Network, Var
from monee.problem.core import OptimizationProblem

from .core import (
    OperatorEquationAssembly,
    SolverInterface,
    SolverResult,
    apply_child_overwrites,
    apply_post_process_all,
    compute_bound_violations,
    finalize_solution,
    ignore_branch,
    ignore_child,
    ignore_compound,
    ignore_node,
    inject_vars,
    mark_slacks_and_prescriptions,
    prepare_solve_network,
)

try:  # CasADi is an optional backend; keep monee importable without it.
    import casadi as ca
except ImportError:  # pragma: no cover - exercised only when casadi is absent
    ca = None

INF = float("inf")

_log = logging.getLogger(__name__)

_CASADI_MISSING = (
    "The CasADi backend requires the 'casadi' package, which is not installed. "
    "Install it with `pip install casadi` (or `pip install monee[casadi]`)."
)


def _require_casadi():
    if ca is None:  # pragma: no cover - trivial guard
        raise ImportError(_CASADI_MISSING)


class CasADiSolveError(RuntimeError):
    """Raised when a CasADi/IPOPT solve does not converge.

    Mirrors GEKKO's ``GekkoSolveError`` so the failure contract is uniform across
    backends: a non-converged solve RAISES (``run_energy_flow`` propagates it;
    ``run_timeseries`` routes it through ``on_step_error``) instead of silently
    returning a ``success=False`` result the caller might ignore.
    """


def _sx(x):
    return x.e if isinstance(x, CasSym) else x


class _Rel:
    __slots__ = ("r", "op")

    def __init__(self, r, op):
        self.r = r
        self.op = op


class CasSym:  # NOSONAR
    """Thin wrapper over a CasADi ``SX`` so monee's operator-based equation
    bodies build a CasADi graph and ``==`` / ``<=`` / ``>=`` capture
    constraints."""

    __slots__ = ("e",)

    def __init__(self, e):
        self.e = e

    # arithmetic -> CasSym
    def __add__(self, o):
        return CasSym(self.e + _sx(o))

    def __radd__(self, o):
        return CasSym(_sx(o) + self.e)

    def __sub__(self, o):
        return CasSym(self.e - _sx(o))

    def __rsub__(self, o):
        return CasSym(_sx(o) - self.e)

    def __mul__(self, o):
        return CasSym(self.e * _sx(o))

    def __rmul__(self, o):
        return CasSym(_sx(o) * self.e)

    def __truediv__(self, o):
        return CasSym(self.e / _sx(o))

    def __rtruediv__(self, o):
        return CasSym(_sx(o) / self.e)

    def __pow__(self, o):
        return CasSym(self.e ** _sx(o))

    def __rpow__(self, o):
        return CasSym(_sx(o) ** self.e)

    def __neg__(self):
        return CasSym(-self.e)

    # relations -> constraint records (residual {==,<=} 0)
    def __eq__(self, o):
        return _Rel(self.e - _sx(o), "eq")

    def __le__(self, o):
        return _Rel(self.e - _sx(o), "le")

    def __ge__(self, o):
        return _Rel(_sx(o) - self.e, "le")

    def __hash__(self):
        return id(self)


class CasModel:
    """Minimal stand-in for the GEKKO model object used by monee's equation
    passes. Collects CasADi constraints and objective terms."""

    def __init__(self):
        self.cons: list[_Rel] = []
        self.obj_terms: list = []
        self._pwl_count = 0  # unique-name counter for spline interpolants

    # --- math hooks (mirror m.sin / m.cos / ...) ---
    @staticmethod
    def sin(x):
        return CasSym(ca.sin(_sx(x)))

    @staticmethod
    def cos(x):
        return CasSym(ca.cos(_sx(x)))

    @staticmethod
    def sqrt(x):
        return CasSym(ca.sqrt(_sx(x)))

    @staticmethod
    def exp(x):
        return CasSym(ca.exp(_sx(x)))

    @staticmethod
    def log10(x):
        return CasSym(ca.log10(_sx(x)))

    @staticmethod
    def abs3(x):
        return CasSym(ca.fabs(_sx(x)))

    @staticmethod
    def max2(a, b):
        return CasSym(ca.fmax(_sx(a), _sx(b)))

    @staticmethod
    def sign2(x):
        return CasSym(ca.sign(_sx(x)))

    @staticmethod
    def sign3(x):
        return CasSym(ca.sign(_sx(x)))

    @staticmethod
    def if2(cond, a, b):
        return CasSym(ca.if_else(_sx(cond), _sx(a), _sx(b)))

    @staticmethod
    def if3(cond, a, b):
        return CasSym(ca.if_else(_sx(cond), _sx(a), _sx(b)))

    # --- model assembly hooks ---
    def Intermediate(self, expr):  # NOSONAR
        # An intermediate is just a named sub-expression; inline it.
        return expr if isinstance(expr, CasSym) else CasSym(_sx(expr))

    def Equation(self, eq):  # NOSONAR
        if isinstance(eq, _Rel):
            self.cons.append(eq)

    def Equations(self, eqs):  # NOSONAR
        for eq in eqs:
            self.Equation(eq)

    def Obj(self, expr):  # NOSONAR
        self.obj_terms.append(_sx(expr))


class CasADiCubicSplineImpl:
    """``pwl_impl`` for the CasADi backend: binds ``y == spline(x)`` through the
    tabulated samples ``(xs, ys)`` using a CasADi B-spline ``interpolant``.

    This is the *smooth* analogue of the MILP/PWL backends' piecewise-linear
    ``addGenConstrPWL`` and the direct counterpart of GEKKO's ``cspline`` - a
    cubic spline is twice-differentiable, which is what IPOPT (a smooth-NLP
    solver) needs; a kinked PWL would give a discontinuous Jacobian. A cubic
    B-spline needs >= 4 knots; for the rare coarse tables (2-3 breakpoints) it
    drops to the highest feasible degree, falling back to linear interpolation
    where even that is rejected. (The bundled gas/heat PWL formulations default
    to 12 breakpoints, so the cubic path is the normal one.)"""

    def __init__(self, model: CasModel):
        self.m = model

    def piecewise_eq(self, y, x, xs, ys, _name=None):
        xs = [float(v) for v in xs]
        ys = [float(v) for v in ys]
        nm = f"monee_cspline_{self.m._pwl_count}"
        self.m._pwl_count += 1
        degree = min(3, max(1, len(xs) - 1))
        try:
            interp = ca.interpolant(nm, "bspline", [xs], ys, {"degree": [degree]})
        except Exception:
            # CasADi rejects some small-knot/degree combinations (e.g. exactly
            # 3 points at degree 2); fall back to a linear interpolant there.
            interp = ca.interpolant(nm, "linear", [xs], ys)
        self.m.Equation(CasSym(_sx(y)) == CasSym(interp(_sx(x))))


# IPOPT options shared by the single-shot solver and the timeseries driver.
_IPOPT_OPTS = {
    "print_time": 0,
    "ipopt.print_level": 0,
    "ipopt.sb": "yes",
    "ipopt.tol": 1e-6,
    "ipopt.max_iter": 3000,
}


class CasADiSolver(OperatorEquationAssembly, SolverInterface):
    """In-process CasADi/IPOPT backend. Reuses the shared
    :class:`~monee.solver.core.OperatorEquationAssembly` equation passes; only
    variable injection, the solve, and the write-back are CasADi-specific."""

    def __init__(self):
        _require_casadi()
        self._backend_name = "casadi"
        self._solver_name = "ipopt"

        self._simulation: bool = False
        self._reg: list = []

        self.last_build_s = None  # nlpsol construction: graph -> derivative fns
        self.last_solve_s = None  # the IPOPT solve() call
        self.last_engine_s = None  # build + solve (the GEKKO "engine" analogue)
        self.last_iters = None

    def _add_equations(self, m, eqs):
        m.Equations(eqs)

    def _pwl_impl(self, m):
        return CasADiCubicSplineImpl(m)

    def _inject(self, model, comp, cat):  # NOSONAR
        for key, val in list(model.__dict__.items()):  # NOSONAR
            if isinstance(val, Var):
                if val.integer:
                    # IPOPT is continuous; relax integers for this backend.
                    pass
                sx = ca.SX.sym(f"v{len(self._reg)}")
                lo = -INF if val.min is None else float(val.min)
                hi = INF if val.max is None else float(val.max)
                v0 = val.value
                x0 = (
                    0.0
                    if (v0 is None or (isinstance(v0, float) and math.isnan(v0)))
                    else float(v0)
                )

                scale = getattr(val, "scale", 1.0) or 1.0
                phys = sx if scale == 1.0 else scale * sx
                if scale != 1.0:
                    lo = lo if lo == -INF else lo / scale
                    hi = hi if hi == INF else hi / scale
                    x0 = x0 / scale
                self._reg.append(
                    {
                        "model": model,
                        "key": key,
                        "sx": sx,
                        "lb": lo,
                        "ub": hi,
                        "x0": x0,
                        "scale": scale,
                        "name": val.name,
                        "vmin": val.min,
                        "vmax": val.max,
                    }
                )
                setattr(model, key, CasSym(phys))
            elif isinstance(val, Const):
                v = val.value
                if isinstance(v, CasSym):
                    setattr(model, key, v)
                elif isinstance(v, (ca.SX, ca.MX)):
                    setattr(model, key, CasSym(v))
                else:
                    setattr(model, key, float(v))

    def _active_models(self, network, nodes, branches, compounds, ignored_nodes):
        for branch in branches:
            if not ignore_branch(branch, network, ignored_nodes):
                yield branch.model
        for node in nodes:
            if ignore_node(node, network, ignored_nodes):
                continue
            yield node.model
            for child in network.childs_by_ids(node.child_ids):
                if not ignore_child(child, ignored_nodes):
                    yield child.model
        for compound in compounds:
            if not ignore_compound(compound, ignored_nodes):
                yield compound.model

    def solve(  # NOSONAR
        self,
        input_network: Network,
        optimization_problem: OptimizationProblem = None,
        draw_debug=False,
        exclude_unconnected_nodes=False,
        step_state=None,
        simulation=False,
        formulation=None,
    ) -> SolverResult:
        self._simulation = simulation
        self._reg = []
        m = CasModel()

        # --- network prep (mirrors GEKKOSolver.solve) ---
        network, ignored_nodes, _islanding_config = prepare_solve_network(
            input_network,
            optimization_problem=optimization_problem,
            formulation=formulation,
            simulation=simulation,
            exclude_unconnected_nodes=exclude_unconnected_nodes,
        )

        nodes = network.nodes
        apply_child_overwrites(network, nodes, ignored_nodes)

        branches = network.branches
        compounds = network.compounds
        mark_slacks_and_prescriptions(network, ignored_nodes)

        # --- variable injection (CasADi symbols) ---
        inject_vars(self._inject, nodes, branches, compounds, network, ignored_nodes)

        # Inter-step temporal coupling (storage SOC, ramp limits, linepack, ...):
        # let extensions/formulations react to the previous step's solved state
        # and flag temporal models so their static-only equations are suppressed.
        if step_state is not None:
            for ext in network.extensions:
                ext.activate_timeseries(network, ignored_nodes, step_state=step_state)
            self.mark_temporal_components(network, ignored_nodes)

        # --- equation assembly (reuses monee's unchanged passes) ---
        objs_exprs: list = []
        self.init_branches(branches)
        self.process_equations_nodes_childs(m, network, nodes, ignored_nodes)
        self.process_equations_branches(m, network, branches, ignored_nodes, objs_exprs)
        self.process_equations_compounds(m, network, compounds, ignored_nodes)
        if optimization_problem is not None:
            self.process_oxf_components(m, network, optimization_problem)
        else:
            self.process_internal_oxf_components(m, network)

        # Inter-step + inter-temporal equations couple this step's variables to
        # the previous step's solved (float) values held in step_state.
        if step_state is not None:
            self.process_inter_step_equations(
                m,
                network,
                nodes,
                branches,
                compounds,
                ignored_nodes,
                step_state,
                optimization_problem=optimization_problem,
            )
            for ext in network.extensions:
                m.Equations(
                    ext.inter_step_equations(network, ignored_nodes, step_state)
                )
                m.Equations(
                    ext.inter_temporal_equations(network, ignored_nodes, step_state)
                )

        for ext in network.extensions:
            m.Equations(ext.equations(network, ignored_nodes))
        for expr in objs_exprs:
            m.Obj(expr)

        # --- assemble & solve the NLP in-process ---
        reg = self._reg
        X = ca.vertcat(*[r["sx"] for r in reg])
        x0 = ca.DM([r["x0"] for r in reg])
        lbx = ca.DM([r["lb"] for r in reg])
        ubx = ca.DM([r["ub"] for r in reg])
        g = ca.vertcat(*[c.r for c in m.cons]) if m.cons else ca.SX.zeros(0)
        lbg = ca.DM([0.0 if c.op == "eq" else -INF for c in m.cons])
        ubg = ca.DM([0.0 for _ in m.cons])
        f = ca.SX(0)
        for t in m.obj_terms:
            f = f + t

        t0 = time.perf_counter()
        # Common-subexpression elimination: monee's per-element equation bodies
        # rebuild identical sub-terms many times (e.g. the AC flow equations
        # reconstruct ``vm_from*vm_to``, ``cos(va_from-va_to)``, ``vm**2`` across
        # all four P/Q flow functions of every branch). ``ca.cse`` merges those
        # duplicate graph nodes before the autodiff build, which both shrinks the
        # nlpsol construction and speeds per-iteration evaluation.
        f, g = ca.cse([f, g])
        solver = ca.nlpsol(
            "monee_casadi", "ipopt", {"x": X, "f": f, "g": g}, _IPOPT_OPTS
        )
        self.last_build_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        self.last_solve_s = time.perf_counter() - t0
        self.last_engine_s = self.last_build_s + self.last_solve_s
        stats = solver.stats()
        success = bool(stats.get("success", False))
        self.last_iters = stats.get("iter_count")
        if not success:
            raise CasADiSolveError(
                "CasADi/IPOPT solve did not converge "
                f"(return status: {stats.get('return_status')!r})."
            )
        x_opt = np.array(sol["x"]).flatten()

        # --- write the solution back into the model ---
        for i, r in enumerate(reg):
            r["model"].__dict__[r["key"]] = Var(
                value=float(x_opt[i]) * r.get("scale", 1.0),
                min=r["vmin"],
                max=r["vmax"],
                name=r["name"],
            )
        # Evaluate any remaining CasSym attributes (inlined intermediates) in one pass.
        leftover = [
            (model, key, val.e)
            for model in self._active_models(
                network, nodes, branches, compounds, ignored_nodes
            )
            for key, val in model.__dict__.items()
            if isinstance(val, CasSym)
        ]
        if leftover:
            F = ca.Function("intval", [X], [ca.vertcat(*[e for _, _, e in leftover])])
            vals = np.array(F(ca.DM(x_opt))).flatten()
            for (model, key, _), v in zip(leftover, vals):
                model.__dict__[key] = Intermediate(value=float(v))

        violations = finalize_solution(
            nodes, branches, compounds, network, input_network
        )
        objective = float(sol["f"]) if m.obj_terms else 0.0
        return SolverResult(
            network,
            network.as_result_dataframe_dict(),
            objective,
            success,
            violations,
            mode_used="optimization",
            backend_used=self.backend_name,
            solver_used=self.solver_name,
        )


class CasADiTimeseries:
    """Build-once / re-solve-per-step timeseries driver on the CasADi backend.

    The whole point of an in-process compiled backend: the model *structure* is
    identical across timesteps, so we assemble the CasADi graph and the
    ``nlpsol`` solver **once**, with every time-varying input declared as a
    CasADi *parameter* (``p``). Each step then only sets the parameter vector and
    re-solves with a warm start from the previous step - no rebuild, no
    re-compile, no subprocess. Contrast :func:`monee.run_timeseries`, which
    rebuilds and recompiles the entire model every step.

    Scope mirrors :class:`CasADiSolver`: electricity AC, plain power flow
    (``optimization_problem=None``). Time-varying attributes come from a
    :class:`~monee.simulation.timeseries.TimeseriesData` (child/node/branch id
    series). Inter-step temporal coupling is *not* supported - only memory-less
    per-step inputs (loads, generation).
    """

    def __init__(  # NOSONAR
        self,
        input_network,
        timeseries_data,
        formulation=None,
        simulation=True,
        steps=None,
    ):
        _require_casadi()
        self._td = timeseries_data
        cs = CasADiSolver()
        cs._simulation = simulation
        cs._reg = []
        m = CasModel()

        network, ignored, _islanding_config = prepare_solve_network(
            input_network,
            optimization_problem=None,
            formulation=formulation,
            simulation=simulation,
        )
        nodes = network.nodes

        # Declare every time-varying attribute as a CasADi parameter BEFORE the
        # overwrite/inject passes. A boundary child whose pinned setpoint is itself
        # time-varying (e.g. an ExtHydrGrid supply ``t_k``, an ExtPowerGrid
        # ``vm_pu``) folds that attribute into the node boundary inside
        # ``overwrite``; declaring the parameter first means the boundary captures
        # the parameter *symbol* rather than a value frozen at step 0.
        self._params = []  # (model, key, param_sx, series)
        self._declare_params(network, timeseries_data)

        apply_child_overwrites(network, nodes, ignored)
        branches, compounds = network.branches, network.compounds
        mark_slacks_and_prescriptions(network, ignored)

        inject_vars(cs._inject, nodes, branches, compounds, network, ignored)

        objs: list = []
        cs.init_branches(branches)
        cs.process_equations_nodes_childs(m, network, nodes, ignored)
        cs.process_equations_branches(m, network, branches, ignored, objs)
        cs.process_equations_compounds(m, network, compounds, ignored)
        cs.process_internal_oxf_components(m, network)
        for ext in network.extensions:
            m.Equations(ext.equations(network, ignored))
        for expr in objs:
            m.Obj(expr)

        reg = cs._reg
        self._network, self._nodes = network, nodes
        self._branches, self._compounds, self._ignored = branches, compounds, ignored
        self._var_entries = reg
        X = ca.vertcat(*[r["sx"] for r in reg])
        P = (
            ca.vertcat(*[p[2] for p in self._params])
            if self._params
            else ca.SX.zeros(0)
        )
        self._lbx = ca.DM([r["lb"] for r in reg])
        self._ubx = ca.DM([r["ub"] for r in reg])
        self._x = ca.DM([r["x0"] for r in reg])  # warm-start state
        g = ca.vertcat(*[c.r for c in m.cons]) if m.cons else ca.SX.zeros(0)
        self._lbg = ca.DM([0.0 if c.op == "eq" else -INF for c in m.cons])
        self._ubg = ca.DM([0.0 for _ in m.cons])
        f = ca.SX(0)
        for t in m.obj_terms:
            f = f + t
        self._has_obj = bool(m.obj_terms)

        t0 = time.perf_counter()
        # Merge duplicate sub-terms before the autodiff build (see CasADiSolver.solve).
        f, g = ca.cse([f, g])
        self._solver = ca.nlpsol(
            "monee_ts", "ipopt", {"x": X, "f": f, "g": g, "p": P}, _IPOPT_OPTS
        )

        # One compiled function (X, P) -> all inlined-intermediate values, for
        # cheap per-step result extraction.
        var_ids = {(id(r["model"]), r["key"]) for r in reg}
        par_ids = {(id(p[0]), p[1]) for p in self._params}
        self._inter = [
            (model, key, val.e)
            for model in self._active_models()
            for key, val in model.__dict__.items()
            if isinstance(val, CasSym)
            and (id(model), key) not in var_ids
            and (id(model), key) not in par_ids
        ]
        self._f_inter = (
            ca.Function("inter", [X, P], [ca.vertcat(*[e for _, _, e in self._inter])])
            if self._inter
            else None
        )
        self.build_s = time.perf_counter() - t0
        length = timeseries_data.length
        length = length() if callable(length) else length
        self.steps = steps if steps is not None else length
        self.last_solve_total_s = None
        self.last_iters_total = None
        self.last_step_success = None

    # ------------------------------------------------------------------ #
    def _declare_params(self, network, td):  # NOSONAR
        # Mirror TimeseriesData.apply_to_network: iterate the network's components
        # and gather the series that target each one by id AND by name (and the
        # compound dicts). Iterating td's id-only dicts (as before) silently
        # dropped name-addressed child/branch series and all compound series.
        def add(model, attr, series):
            psx = ca.SX.sym(f"p{len(self._params)}")
            self._params.append((model, attr, psx, series))
            setattr(model, attr, CasSym(psx))

        def merged(comp, id_dict, name_dict):
            # id series first, then name series (name wins on attr conflict) so
            # each attr yields exactly one parameter - matches apply_to_*.
            out = {}
            if comp.id in id_dict:
                out.update(id_dict[comp.id])
            name = getattr(comp, "name", None)
            if name is not None and name in name_dict:
                out.update(name_dict[name])
            return out

        for node in network.nodes:
            for attr, series in td._node_id_to_series.get(node.id, {}).items():
                add(node.model, attr, series)
            for child in network.childs_by_ids(node.child_ids):
                for attr, series in merged(
                    child, td._child_id_to_series, td._child_name_to_series
                ).items():
                    add(child.model, attr, series)
        for branch in network.branches:
            for attr, series in merged(
                branch, td._branch_id_to_series, td._branch_name_to_series
            ).items():
                add(branch.model, attr, series)
        for compound in network.compounds:
            for attr, series in merged(
                compound, td._compound_id_to_series, td._compound_name_to_series
            ).items():
                add(compound.model, attr, series)

    def _active_models(self):
        net, ign = self._network, self._ignored
        for branch in self._branches:
            if not ignore_branch(branch, net, ign):
                yield branch.model
        for node in self._nodes:
            if ignore_node(node, net, ign):
                continue
            yield node.model
            for child in net.childs_by_ids(node.child_ids):
                if not ignore_child(child, ign):
                    yield child.model
        for compound in self._compounds:
            if not ignore_compound(compound, ign):
                yield compound.model

    # ------------------------------------------------------------------ #
    def _solve_step(self, t):
        """Set the parameter vector for step *t*, re-solve (warm-started), and
        scatter the solution into the network models. Returns the solve stats."""
        pvals = ca.DM([float(p[3][t]) for p in self._params])
        t0 = time.perf_counter()
        sol = self._solver(
            x0=self._x,
            lbx=self._lbx,
            ubx=self._ubx,
            lbg=self._lbg,
            ubg=self._ubg,
            p=pvals,
        )
        solve_s = time.perf_counter() - t0
        stats = self._solver.stats()
        self._x = sol["x"]  # warm-start the next step
        self._last_objective = float(sol["f"]) if self._has_obj else 0.0

        # scatter into models for result extraction
        xv = np.array(sol["x"]).flatten()
        for i, r in enumerate(self._var_entries):
            r["model"].__dict__[r["key"]] = Var(
                value=float(xv[i]) * r.get("scale", 1.0),
                min=r["vmin"],
                max=r["vmax"],
                name=r["name"],
            )
        for model, key, _psx, series in self._params:
            model.__dict__[key] = float(series[t])
        if self._f_inter is not None:
            ivals = np.array(self._f_inter(sol["x"], pvals)).flatten()
            for (model, key, _), v in zip(self._inter, ivals):
                model.__dict__[key] = Intermediate(value=float(v))
        apply_post_process_all(
            self._nodes, self._branches, self._compounds, self._network
        )
        return solve_s, stats

    def step_result(self, t) -> SolverResult:
        """Solve step *t* and return a :class:`SolverResult` for the network at
        that step (used by the :func:`monee.run_timeseries` reuse fast path)."""
        _solve_s, stats = self._solve_step(t)
        success = bool(stats.get("success", False))
        self.last_step_success = success
        if not success:
            raise CasADiSolveError(
                f"CasADi/IPOPT timeseries step {t} did not converge "
                f"(return status: {stats.get('return_status')!r})."
            )
        violations = compute_bound_violations(
            self._nodes, self._branches, self._compounds, self._network
        )
        return SolverResult(
            self._network,
            self._network.as_result_dataframe_dict(),
            self._last_objective,
            success,
            violations,
            mode_used="optimization",
            backend_used="casadi",
            solver_used="ipopt",
        )

    def run(self):
        """Solve every timestep, reusing the compiled solver. Returns a list of
        per-step ``{type_name: DataFrame}`` dicts (like
        :attr:`SolverResult.dataframes`)."""
        results = []
        solve_total = 0.0
        iters_total = 0
        for t in range(self.steps):
            solve_s, stats = self._solve_step(t)
            solve_total += solve_s
            iters_total += stats.get("iter_count") or 0
            results.append(self._network.as_result_dataframe_dict())
        self.last_solve_total_s = solve_total
        self.last_iters_total = iters_total
        return results


class CasADiMultiPeriodSolver:
    """Multi-period optimizer on in-process CasADi/IPOPT.

    Two-pass, mirroring :class:`~monee.simulation.multi_period.GekkoMultiPeriodSolver`:
    inject CasADi symbols for all T periods into one register, then assemble
    per-period and ``inter_period`` equations with a
    :class:`~monee.simulation.step_state.PeriodState` that sees every period
    (so cross-period constraints become algebraic links between the periods'
    injected variables), and solve the whole horizon as a single ``nlpsol`` NLP.
    """

    def __init__(self):
        _require_casadi()
        self._backend_name = "casadi"
        self._solver_name = "ipopt"
        self.last_build_s = None
        self.last_solve_s = None
        self.last_iters = None

    def solve_multi_period(  # NOSONAR
        self,
        network: Network,
        timeseries_data=None,
        steps: int = None,
        optimization_problem=None,
        dt_h=1.0,
        datetime_index=None,
        initial_state: dict = None,
        terminal_state: dict = None,
        formulation=None,
    ):
        from monee.simulation.multi_period import (
            MultiPeriodResult,
            _find_component_var,
            _prepare_period,
            _resolve_dt_h,
            _resolve_steps,
        )
        from monee.simulation.step_state import PeriodState

        steps = _resolve_steps(steps, timeseries_data)
        dt_h_list = _resolve_dt_h(dt_h, datetime_index, steps)

        cs = CasADiSolver()
        cs._simulation = False
        cs._reg = []
        m = CasModel()

        # Pass 1: prepare networks and inject CasADi symbols for all periods.
        net_copies: list[Network] = []
        ignored_list: list[set] = []
        for t in range(steps):
            net_t, ignored_t = _prepare_period(
                network, timeseries_data, t, optimization_problem, formulation
            )
            inject_vars(
                cs._inject,
                net_t.nodes,
                net_t.branches,
                net_t.compounds,
                net_t,
                ignored_t,
            )
            for ext in net_t.extensions:
                ext.activate_timeseries(net_t, ignored_t)
            cs.mark_temporal_components(net_t, ignored_t)
            net_copies.append(net_t)
            ignored_list.append(ignored_t)

        # Pass 2: build per-period and inter-period equations.
        for t in range(steps):
            net_t, ignored_t = net_copies[t], ignored_list[t]
            period_state = PeriodState(
                net_copies,
                current_t=t,
                dt_h=dt_h_list[t],
                initial_state=initial_state,
            )
            cs.init_branches(net_t.branches)
            per_objs: list = []
            cs.process_equations_nodes_childs(m, net_t, net_t.nodes, ignored_t)
            cs.process_equations_branches(m, net_t, net_t.branches, ignored_t, per_objs)
            cs.process_equations_compounds(m, net_t, net_t.compounds, ignored_t)
            if optimization_problem is not None:
                cs.process_oxf_components(
                    m, net_t, optimization_problem, period_index=t
                )
            else:
                cs.process_internal_oxf_components(m, net_t)
            for expr in per_objs:
                m.Obj(expr)
            cs.process_inter_period_equations(
                m,
                net_t,
                net_t.nodes,
                net_t.branches,
                net_t.compounds,
                ignored_t,
                period_state,
                optimization_problem=optimization_problem,
                period_index=t,
            )
            for ext in net_t.extensions:
                m.Equations(ext.inter_period_equations(net_t, ignored_t, period_state))
                m.Equations(
                    ext.inter_temporal_equations(net_t, ignored_t, period_state)
                )
                m.Equations(ext.equations(net_t, ignored_t))
            if terminal_state and t == steps - 1:
                for (comp_id, attr), target in terminal_state.items():
                    var = _find_component_var(net_t, comp_id, attr)
                    if var is not None:
                        m.Equation(var == target)

        # Assemble and solve the whole horizon as one NLP.
        reg = cs._reg
        X = ca.vertcat(*[r["sx"] for r in reg])
        x0 = ca.DM([r["x0"] for r in reg])
        lbx = ca.DM([r["lb"] for r in reg])
        ubx = ca.DM([r["ub"] for r in reg])
        g = ca.vertcat(*[c.r for c in m.cons]) if m.cons else ca.SX.zeros(0)
        lbg = ca.DM([0.0 if c.op == "eq" else -INF for c in m.cons])
        ubg = ca.DM([0.0 for _ in m.cons])
        f = ca.SX(0)
        for term in m.obj_terms:
            f = f + term

        _log.info("Multi-period CasADi solve: T=%d periods", steps)
        t0 = time.perf_counter()
        solver = ca.nlpsol(
            "monee_casadi_mp", "ipopt", {"x": X, "f": f, "g": g}, _IPOPT_OPTS
        )
        self.last_build_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        self.last_solve_s = time.perf_counter() - t0
        stats = solver.stats()
        success = bool(stats.get("success", False))
        self.last_iters = stats.get("iter_count")
        if not success:
            raise CasADiSolveError(
                "CasADi/IPOPT multi-period solve did not converge "
                f"(return status: {stats.get('return_status')!r})."
            )
        x_opt = np.array(sol["x"]).flatten()

        # Scatter the solution back into every period's models.
        for i, r in enumerate(reg):
            r["model"].__dict__[r["key"]] = Var(
                value=float(x_opt[i]) * r.get("scale", 1.0),
                min=r["vmin"],
                max=r["vmax"],
                name=r["name"],
            )
        leftover = [
            (model, key, val.e)
            for net_t, ignored_t in zip(net_copies, ignored_list)
            for model in cs._active_models(
                net_t, net_t.nodes, net_t.branches, net_t.compounds, ignored_t
            )
            for key, val in model.__dict__.items()
            if isinstance(val, CasSym)
        ]
        if leftover:
            F = ca.Function(
                "intval_mp", [X], [ca.vertcat(*[e for _, _, e in leftover])]
            )
            vals = np.array(F(ca.DM(x_opt))).flatten()
            for (model, key, _), v in zip(leftover, vals):
                model.__dict__[key] = Intermediate(value=float(v))
        for net_t in net_copies:
            apply_post_process_all(net_t.nodes, net_t.branches, net_t.compounds, net_t)

        objective = float(sol["f"]) if m.obj_terms else 0.0
        return MultiPeriodResult(
            net_copies,
            objective=objective,
            success=success,
            datetime_index=datetime_index,
        )

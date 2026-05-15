"""
Conductor — co-simulation driver for monee.

Steps a network forward in time under externally-supplied timestep durations,
maintaining the inter-step state (linepack, LTC, storage SoC, …) automatically.
Fits external simulators that run at variable / sub-second timesteps and want
monee to act as their quasi-static energy-flow engine in lock-step.

The Conductor is the *imperative*, externally-paced complement to
:func:`~monee.simulation.timeseries.run` (fixed loop over a pre-known horizon)
and :func:`~monee.simulation.multi_period.run_multi_period` (single joint
optimisation across T periods).  All three share the same quasi-static-model
plumbing: ``StepState.dt_h``, ``inter_step_equations`` /
``inter_temporal_equations``, and the ``state.push(result.network)`` round-trip
after each solve.

Three usage patterns
--------------------
**Bare external driver — no profile, just data injection per tick.**

    cond = Conductor(net, solver="gurobi")
    while external.running:
        cond.step(
            dt_h=external.next_dt_h(),
            data_overrides={(load_id, "p_mw"): external.read(load_id)},
        )

**TimeseriesData as the data source — index by step.**

    td = TimeseriesData()
    td.add_child_series(load_id, "p_mw", profile_p)
    cond = Conductor(net, solver="gurobi", timeseries_data=td)
    for t in range(steps):
        cond.step(dt_h=1.0, ts_index=t)

**Intermediate sub-steps between TimeseriesData ticks** — pass ``ts_index`` on
the boundary steps, omit it on the intermediate ones (the network keeps
whatever was last applied)::

    for major in range(major_steps):
        cond.step(dt_h=dt_major, ts_index=major)
        for _ in range(intermediate_count):
            cond.step(dt_h=dt_minor)            # finer-grained relaxation

Seed initial conditions for the quasi-static states the very same way as
``run_multi_period``::

    cond = Conductor(
        net,
        solver="gurobi",
        initial_state={(battery_id, "e_mwh"): 2.0, (pipe_id, "linepack_kg"): 500},
    )

The Conductor never mutates the base network — each :meth:`step` works on a
fresh deep-copy.  Set ``net_strategy="reuse"`` only after profiling shows the
copy is a bottleneck.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import pandas

from monee.model import Network
from monee.simulation.step_state import StepState
from monee.simulation.timeseries import (
    StepResult,
    TimeseriesData,
    TimeseriesResult,
)
from monee.solver.dispatch import resolve_solver

_log = logging.getLogger(__name__)


class Conductor:
    """Externally-paced co-simulation driver.

    Holds the base network, a resolved solver, an optional
    :class:`~monee.problem.OptimizationProblem`, an optional pre-loaded
    :class:`~monee.simulation.timeseries.TimeseriesData`, and the persistent
    :class:`~monee.simulation.step_state.StepState` that carries quasi-static
    state between steps.

    Args:
        net: Base network — never mutated; each :meth:`step` makes a deep
            copy and solves on the copy.
        solver: Solver name string (``"gurobi"``, ``"ipopt"``, …), a concrete
            :class:`~monee.solver.core.SolverInterface` instance, or ``None``
            for the GEKKO+IPOPT default.  Resolved once at construction and
            reused.
        backend: ``"gekko"`` / ``"pyomo"`` to force a backend.  See
            :func:`monee.solver.dispatch.resolve_solver` for routing rules.
        optimization_problem: Optional optimisation problem applied to every
            per-step network copy.
        timeseries_data: Optional pre-loaded ``TimeseriesData``.  Callers
            pass ``ts_index`` to :meth:`step` to apply that index's slice.
            Omitted ``ts_index`` leaves the network at its last-applied
            state, which is what you want for intermediate sub-steps.
        initial_state: Optional ``{(component_id, attr): value}`` map
            consulted by ``inter_step_equations`` /
            ``inter_temporal_equations`` whenever no prior solve has
            populated the StepState.  Equivalent semantics to
            ``initial_state`` on
            :func:`~monee.simulation.multi_period.run_multi_period`.
        on_step_error: ``"raise"`` (default) re-raises any per-step solver
            exception immediately — recommended for co-simulation where you
            want to surface failures to the external driver right away.
            ``"skip"`` records a failed :class:`StepResult` and continues.
        **solver_kwargs: Forwarded verbatim to ``solver.solve(...)`` for
            every step (e.g. ``debug=True``).

    Attributes:
        state: The live :class:`StepState`.  Quasi-static models read
            previous-step values through it.
        history: One :class:`StepResult` per :meth:`step` call, in order.
            Failed steps appear with ``failed=True`` and ``result=None``.
        step_count: Number of steps run so far (successful or failed).
        t_h: Cumulative simulated time in hours.
    """

    __slots__ = (
        "_base_net",
        "_solver",
        "_optimization_problem",
        "_timeseries_data",
        "_initial_state",
        "_on_step_error",
        "_solver_kwargs",
        "_state",
        "_history",
        "_t_h",
    )

    def __init__(
        self,
        net: Network,
        *,
        solver=None,
        backend: str | None = None,
        optimization_problem=None,
        timeseries_data: TimeseriesData | None = None,
        initial_state: Mapping[tuple, float] | None = None,
        on_step_error: str = "raise",
        **solver_kwargs: Any,
    ) -> None:
        if on_step_error not in ("raise", "skip"):
            raise ValueError(
                f"on_step_error must be 'raise' or 'skip', got {on_step_error!r}"
            )
        self._base_net = net
        self._solver = resolve_solver(solver, backend=backend)
        self._optimization_problem = optimization_problem
        self._timeseries_data = timeseries_data
        self._initial_state: dict = dict(initial_state) if initial_state else {}
        self._on_step_error = on_step_error
        self._solver_kwargs = dict(solver_kwargs)
        self._state = StepState(initial_state=self._initial_state)
        self._history: list[StepResult] = []
        self._t_h: float = 0.0

    # ── Read-only views ───────────────────────────────────────────────────

    @property
    def state(self) -> StepState:
        """The live :class:`StepState` carrier."""
        return self._state

    @property
    def history(self) -> list[StepResult]:
        """Per-step :class:`StepResult` log, in order.

        Returns a copy of the internal list; mutating it does not affect
        the Conductor.
        """
        return list(self._history)

    @property
    def step_count(self) -> int:
        """Number of steps run so far (including failures)."""
        return len(self._history)

    @property
    def t_h(self) -> float:
        """Cumulative simulated time in hours."""
        return self._t_h

    # ── Core stepping API ─────────────────────────────────────────────────

    def step(
        self,
        dt_h: float,
        *,
        data_overrides: Mapping[tuple, float] | None = None,
        ts_index: int | None = None,
    ) -> StepResult:
        """Advance the simulation by *dt_h* hours.

        Args:
            dt_h: Duration of this step in hours.  Must be strictly positive.
            data_overrides: Optional ``{(component_id, attr): value}`` map
                applied to the step's network copy *after* any
                ``timeseries_data`` slice — so overrides win on conflicts.
                ``Var`` attributes are pinned by value/min/max; plain
                attributes are replaced.  Unknown ids or attributes raise
                :exc:`KeyError` / :exc:`AttributeError`.
            ts_index: Optional index into the registered
                :class:`TimeseriesData`.  Omit to leave the network at the
                state it was last left in — the natural choice for
                intermediate sub-steps between data ticks.

        Returns:
            The :class:`StepResult` for this step.  Also appended to
            :attr:`history`.

        Raises:
            ValueError: If *dt_h* ≤ 0.
            Solver exceptions: propagated when ``on_step_error="raise"``.
        """
        if dt_h <= 0:
            raise ValueError(f"dt_h must be > 0, got {dt_h}")

        net_copy = self._base_net.copy()
        if self._timeseries_data is not None and ts_index is not None:
            self._timeseries_data.apply_to_network(net_copy, ts_index)
        if data_overrides:
            _apply_overrides(net_copy, data_overrides)

        self._state.dt_h = dt_h
        step_idx = self.step_count
        try:
            result = self._solver.solve(
                net_copy,
                optimization_problem=self._optimization_problem,
                step_state=self._state,
                **self._solver_kwargs,
            )
        except Exception as exc:
            if self._on_step_error == "raise":
                raise
            _log.warning("Conductor step %d failed: %s", step_idx, exc)
            sr = StepResult(step=step_idx, result=None, failed=True, error=exc)
            self._history.append(sr)
            self._t_h += dt_h
            return sr

        self._state.push(result.network)
        sr = StepResult(step=step_idx, result=result)
        self._history.append(sr)
        self._t_h += dt_h
        return sr

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        initial_state: Mapping[tuple, float] | None = None,
    ) -> None:
        """Forget all step history and recreate the :class:`StepState`.

        Args:
            initial_state: Optional replacement for the seed map.  ``None``
                keeps the original from construction.
        """
        if initial_state is not None:
            self._initial_state = dict(initial_state)
        self._state = StepState(initial_state=self._initial_state)
        self._history = []
        self._t_h = 0.0

    def to_timeseries_result(
        self,
        datetime_index: pandas.DatetimeIndex | None = None,
    ) -> TimeseriesResult:
        """Wrap the accumulated history as a :class:`TimeseriesResult`.

        Lets co-sim runs reuse all the analysis / plotting code written
        against ``run_timeseries`` output.
        """
        return TimeseriesResult(list(self._history), datetime_index=datetime_index)

    # ── Context-manager support ───────────────────────────────────────────

    def __enter__(self) -> Conductor:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Placeholder — current solver backends hold no externally-visible
        # resources, but the lifecycle hook is here so future ones can.
        return None

    def __repr__(self) -> str:
        return (
            f"Conductor(steps={self.step_count}, t_h={self._t_h:.4g}, "
            f"solver={type(self._solver).__name__})"
        )


def _apply_overrides(net: Network, overrides: Mapping[tuple, float]) -> None:
    """Patch the network with ``{(component_id, attr): value}`` entries.

    Mirrors :meth:`TimeseriesData._set_model_attr` semantics — ``Var``
    attributes are pinned (value/min/max), plain attributes replaced.  An
    unknown id raises :exc:`KeyError`; an unknown attribute raises
    :exc:`AttributeError`.  Both errors are *fatal* (re-raised even under
    ``on_step_error="skip"``) because they indicate a wiring bug, not a
    transient solver failure.
    """
    if not overrides:
        return
    # One pass over the network to build an id → [models] map; cheaper than
    # walking the network per override for any non-trivial override count.
    by_id: dict = {}
    for node in net.nodes:
        by_id.setdefault(node.id, []).append(node.model)
        for child in net.childs_by_ids(node.child_ids):
            by_id.setdefault(child.id, []).append(child.model)
    for branch in net.branches:
        by_id.setdefault(branch.id, []).append(branch.model)
    for compound in net.compounds:
        by_id.setdefault(compound.id, []).append(compound.model)

    for (comp_id, attr), value in overrides.items():
        models = by_id.get(comp_id)
        if not models:
            raise KeyError(
                f"data_overrides: component id {comp_id!r} not found in network"
            )
        applied = False
        for model in models:
            if hasattr(model, attr):
                TimeseriesData._set_model_attr(model, attr, value)
                applied = True
        if not applied:
            raise AttributeError(
                f"data_overrides: attribute {attr!r} not found on any model "
                f"with id {comp_id!r}"
            )

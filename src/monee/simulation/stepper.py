"""Stepper - externally-paced co-simulation adapter.

Steps a network forward in time at caller-supplied dt_h values (the external
framework owns the clock), maintaining inter-step state (linepack, LTC,
storage SoC, ...) via the shared StepState plumbing. Each
:meth:`Stepper.step` works on a fresh network copy.
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


class Stepper:
    """Externally-paced co-simulation adapter. Holds base network, solver,
    optional problem and timeseries data, and the persistent :class:`StepState`.

    ``max_history`` caps how many solved steps are retained (``None`` =
    unlimited): every step keeps a full solved network copy (once in the
    :class:`StepState`, once in the :class:`StepResult` history), so an
    open-ended co-simulation grows memory without bound otherwise. Set it to
    a small number (the longest lookback any ``inter_step_equations`` needs,
    e.g. 8) for long runs; :meth:`to_timeseries_result` then only covers the
    retained window."""

    __slots__ = (
        "_base_net",
        "_solver",
        "_optimization_problem",
        "_timeseries_data",
        "_initial_state",
        "_on_step_error",
        "_max_history",
        "_solver_kwargs",
        "_state",
        "_history",
        "_step_count",
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
        max_history: int | None = None,
        **solver_kwargs: Any,
    ) -> None:
        if on_step_error not in ("raise", "skip"):
            raise ValueError(
                f"on_step_error must be 'raise' or 'skip', got {on_step_error!r}"
            )
        if max_history is not None and max_history < 1:
            raise ValueError(f"max_history must be >= 1 or None, got {max_history}")
        self._base_net = net
        self._solver = resolve_solver(solver, backend=backend)
        self._optimization_problem = optimization_problem
        self._timeseries_data = timeseries_data
        self._initial_state: dict = dict(initial_state) if initial_state else {}
        self._on_step_error = on_step_error
        self._max_history = max_history
        self._solver_kwargs = dict(solver_kwargs)
        self._state = StepState(
            initial_state=self._initial_state, max_steps=max_history
        )
        self._history: list[StepResult] = []
        self._step_count: int = 0
        self._t_h: float = 0.0

    @property
    def state(self) -> StepState:
        return self._state

    @property
    def history(self) -> list[StepResult]:
        """Retained step results (the last ``max_history`` ones, or all)."""
        return list(self._history)

    @property
    def step_count(self) -> int:
        """Total number of step() calls, including dropped and failed ones."""
        return self._step_count

    @property
    def t_h(self) -> float:
        return self._t_h

    def step(
        self,
        dt_h: float,
        *,
        data_overrides: Mapping[tuple, float] | None = None,
        ts_index: int | None = None,
    ) -> StepResult:
        """Advance by *dt_h* hours. ``data_overrides`` are applied after the
        ``ts_index`` slice (overrides win on conflicts)."""
        if dt_h <= 0:
            raise ValueError(f"dt_h must be > 0, got {dt_h}")

        net_copy = self._base_net.copy()
        if self._timeseries_data is not None and ts_index is not None:
            self._timeseries_data.apply_to_network(net_copy, ts_index)
        if data_overrides:
            _apply_overrides(net_copy, data_overrides)

        self._state.dt_h = dt_h
        step_idx = self._step_count
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
            _log.warning("Stepper step %d failed: %s", step_idx, exc)
            sr = StepResult(step=step_idx, result=None, failed=True, error=exc)
            self._record(sr, dt_h)
            return sr

        self._state.push(result.network)
        sr = StepResult(step=step_idx, result=result)
        self._record(sr, dt_h)
        return sr

    def get(self, component_id, attr: str, step: int = -1):
        """Solved value of ``attr`` on component ``component_id`` - by default
        from the most recent successful step (the *get* side of a co-simulation
        adapter's set/step/get contract; ``data_overrides`` is the *set* side).

        ``step`` follows :meth:`StepState.get`: negative = relative to the
        latest solve, non-negative = absolute step index. Returns ``None`` (or
        the ``initial_state`` fallback) when no solve has written the value."""
        return self._state.get(component_id, attr, step=step)

    def _record(self, sr: StepResult, dt_h: float) -> None:
        self._history.append(sr)
        if self._max_history is not None and len(self._history) > self._max_history:
            del self._history[0]
        self._step_count += 1
        self._t_h += dt_h

    def reset(
        self,
        *,
        initial_state: Mapping[tuple, float] | None = None,
    ) -> None:
        """Clear step history and recreate the StepState."""
        if initial_state is not None:
            self._initial_state = dict(initial_state)
        self._state = StepState(
            initial_state=self._initial_state, max_steps=self._max_history
        )
        self._history = []
        self._step_count = 0
        self._t_h = 0.0

    def to_timeseries_result(
        self,
        datetime_index: pandas.DatetimeIndex | None = None,
    ) -> TimeseriesResult:
        """Wrap the retained history as a :class:`TimeseriesResult`. With
        ``max_history`` set this covers only the retained window."""
        return TimeseriesResult(list(self._history), datetime_index=datetime_index)

    def __enter__(self) -> Stepper:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def __repr__(self) -> str:
        return (
            f"Stepper(steps={self._step_count}, t_h={self._t_h:.4g}, "
            f"solver={type(self._solver).__name__})"
        )


def _apply_overrides(net: Network, overrides: Mapping[tuple, float]) -> None:
    """Apply ``{(comp_id, attr): value}`` via :meth:`TimeseriesData._set_model_attr`.
    Unknown id/attr raise (treated as wiring bugs, not transient failures)."""
    if not overrides:
        return
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

"""Conductor — externally-paced co-simulation driver.

Steps a network forward in time at user-supplied dt_h values, maintaining
inter-step state (linepack, LTC, storage SoC, …) via the shared StepState
plumbing. Each :meth:`Conductor.step` works on a fresh network copy.
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
    """Externally-paced co-simulation driver. Holds base network, solver,
    optional problem and timeseries data, and the persistent :class:`StepState`."""

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

    @property
    def state(self) -> StepState:
        return self._state

    @property
    def history(self) -> list[StepResult]:
        return list(self._history)

    @property
    def step_count(self) -> int:
        return len(self._history)

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

    def reset(
        self,
        *,
        initial_state: Mapping[tuple, float] | None = None,
    ) -> None:
        """Clear step history and recreate the StepState."""
        if initial_state is not None:
            self._initial_state = dict(initial_state)
        self._state = StepState(initial_state=self._initial_state)
        self._history = []
        self._t_h = 0.0

    def to_timeseries_result(
        self,
        datetime_index: pandas.DatetimeIndex | None = None,
    ) -> TimeseriesResult:
        """Wrap accumulated history as a :class:`TimeseriesResult`."""
        return TimeseriesResult(list(self._history), datetime_index=datetime_index)

    def __enter__(self) -> Conductor:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def __repr__(self) -> str:
        return (
            f"Conductor(steps={self.step_count}, t_h={self._t_h:.4g}, "
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

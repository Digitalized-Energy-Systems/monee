"""Stepper - externally-paced co-simulation adapter.

Steps a network forward in time at caller-supplied dt_h values (the external
framework owns the clock), maintaining inter-step state (linepack, LTC,
storage SoC, ...) via the shared StepState plumbing. Each
:meth:`Stepper.step` works on a fresh network copy.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import astuple, dataclass, fields
from typing import Any

import pandas

from monee.model import Network
from monee.model.core import (
    Branch,
    Child,
    Compound,
    Intermediate,
    Node,
    PostProcess,
    value,
)
from monee.simulation.step_state import StepState
from monee.simulation.timeseries import (
    StepResult,
    TimeseriesData,
    TimeseriesResult,
    _unsuccessful_solve_error,
)
from monee.solver.dispatch import resolve_solver

_log = logging.getLogger(__name__)

_REMOVE_BY_TYPE = {
    Branch: "remove_branch",
    Node: "remove_node",
    Child: "remove_child",
    Compound: "remove_compound",
}


@dataclass(frozen=True)
class NetworkChange:
    """A single structural change recorded by a :class:`Stepper`.

    ``source`` distinguishes how the change arose: ``"mutation"`` (an explicit
    Stepper mutation call), ``"detected"`` (inferred from the applied input
    network, e.g. a ``data_overrides`` flip), or ``"solver"`` (islanding decided
    during the solve)."""

    step: int
    t_h: float
    kind: str
    component_type: str
    component_id: Any
    name: str | None = None
    source: str = "detected"


def _effective_active(comp) -> bool:
    """Active state honouring both the component flag and a model-level
    ``active`` (which may be a :class:`~monee.model.core.Var`)."""
    if not comp.active:
        return False
    model_active = getattr(comp.model, "active", None)
    if model_active is None:
        return True
    return bool(value(model_active))


def _topology_snapshot(net: Network) -> dict[tuple[str, Any], tuple[str | None, bool]]:
    """Map ``(type_name, id) -> (name, effective_active)`` for every component.
    Keyed on the type too because node and child ids are independent counters
    and can collide."""
    snapshot: dict[tuple[str, Any], tuple[str | None, bool]] = {}
    for comp in net.iter_all_components():
        snapshot[(type(comp).__name__, comp.id)] = (comp.name, _effective_active(comp))
    return snapshot


def _islanded_snapshot(net: Network) -> dict[tuple[str, Any], str | None]:
    """Map ``(type_name, id) -> name`` for components the solver islanded
    (``comp.ignored``) on *net* (typically a solved ``result.network``)."""
    return {
        (type(comp).__name__, comp.id): comp.name
        for comp in net.iter_all_components()
        if getattr(comp, "ignored", False)
    }


def _diff_topology(
    prev: dict[tuple[str, Any], tuple[str | None, bool]],
    cur: dict[tuple[str, Any], tuple[str | None, bool]],
    step: int,
    t_h: float,
    annotations: dict[tuple[str, Any], tuple[str, str]],
) -> list[NetworkChange]:
    """Topology/active deltas between two snapshots. An ``annotations`` entry for
    a key overrides the change ``kind``/``source`` (and is consumed) so an
    explicit ``fail()`` reads as ``failed`` rather than ``deactivated``."""
    changes: list[NetworkChange] = []

    def _emit(key, default_kind, default_source, name):
        type_name, comp_id = key
        kind, source = annotations.pop(key, (default_kind, default_source))
        changes.append(NetworkChange(step, t_h, kind, type_name, comp_id, name, source))

    for key, (name, _active) in cur.items():
        if key not in prev:
            _emit(key, "added", "detected", name)
    for key, (name, _active) in prev.items():
        if key not in cur:
            _emit(key, "removed", "detected", name)
    for key, (name, active) in cur.items():
        if key in prev and prev[key][1] != active:
            _emit(key, "activated" if active else "deactivated", "detected", name)
    return changes


def _diff_islanding(
    prev: dict[tuple[str, Any], str | None],
    cur: dict[tuple[str, Any], str | None],
    step: int,
    t_h: float,
    suppress_islanded: set[tuple[str, Any]],
    suppress_rejoined: set[tuple[str, Any]],
) -> list[NetworkChange]:
    """Islanding deltas: newly ignored -> ``islanded``, no-longer ignored ->
    ``rejoined``. Both sourced ``"solver"``. Keys in the suppress sets are
    skipped (echoes of a user/input change recorded in the same step); their
    dependents' events still appear."""
    changes: list[NetworkChange] = []
    for key, name in cur.items():
        if key not in prev and key not in suppress_islanded:
            changes.append(
                NetworkChange(step, t_h, "islanded", key[0], key[1], name, "solver")
            )
    for key, name in prev.items():
        if key not in cur and key not in suppress_rejoined:
            changes.append(
                NetworkChange(step, t_h, "rejoined", key[0], key[1], name, "solver")
            )
    return changes


def _split_typed_id(component_id, component_type=None):
    """Unpack the ``(type, id)`` id form into ``(component_type, bare_id)``."""
    if (
        component_type is None
        and isinstance(component_id, tuple)
        and len(component_id) == 2
        and isinstance(component_id[0], type)
    ):
        component_type, component_id = component_id
    return component_type, component_id


def _filter_by_type(matches, component_type):
    if component_type is None:
        return matches
    return [
        comp
        for comp in matches
        if type(comp) is component_type or isinstance(comp.model, component_type)
    ]


def _match_components(net: Network, component_id, component_type=None):
    """Return ``(bare_id, matches)`` for *component_id* on *net*.

    ``component_id`` may be a bare id or a ``(type, id)`` tuple where ``type``
    is a container class (:class:`~monee.model.core.Child`, ...) or a model
    class (e.g. ``PowerLoad``); the tuple form is equivalent to passing
    ``component_type`` explicitly."""
    component_type, component_id = _split_typed_id(component_id, component_type)
    matches = [comp for comp in net.iter_all_components() if comp.id == component_id]
    return component_id, _filter_by_type(matches, component_type)


class Stepper:
    """Externally-paced co-simulation adapter. Holds base network, solver,
    optional problem and timeseries data, and the persistent :class:`StepState`.

    ``max_history`` caps how many solved steps are retained (``None`` =
    unlimited): every step keeps a full solved network copy (once in the
    :class:`StepState`, once in the :class:`StepResult` history), so an
    open-ended co-simulation grows memory without bound otherwise. Set it to
    a small number (the longest lookback any ``inter_step_equations`` needs,
    e.g. 8) for long runs; :meth:`to_timeseries_result` then only covers the
    retained window. ``max_changes`` likewise caps the recorded
    :class:`NetworkChange` list (oldest events dropped first)."""

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
        "_carry_dt_h",
        "_work_net",
        "_record_changes",
        "_changes",
        "_max_changes",
        "_changes_dropped",
        "_prev_topology",
        "_prev_islanded",
        "_pending_annotations",
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
        record_changes: bool = True,
        max_changes: int | None = None,
        **solver_kwargs: Any,
    ) -> None:
        if on_step_error not in ("raise", "skip"):
            raise ValueError(
                f"on_step_error must be 'raise' or 'skip', got {on_step_error!r}"
            )
        if max_history is not None and max_history < 1:
            raise ValueError(f"max_history must be >= 1 or None, got {max_history}")
        if max_changes is not None and max_changes < 1:
            raise ValueError(f"max_changes must be >= 1 or None, got {max_changes}")
        self._base_net = net
        # Mutations accumulate on a working copy; the user's net stays untouched
        # and is only the reset() baseline.
        self._work_net = net.copy()
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
        self._carry_dt_h: float = 0.0
        self._record_changes = record_changes
        self._changes: list[NetworkChange] = []
        self._max_changes = max_changes
        self._changes_dropped = 0
        self._prev_topology = (
            _topology_snapshot(self._work_net) if record_changes else {}
        )
        self._prev_islanded: dict[tuple[str, Any], str | None] = {}
        self._pending_annotations: dict[tuple[str, Any], tuple[str, str]] = {}

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
        ``ts_index`` slice (overrides win on conflicts).

        A solver that reports an unsuccessful solve (``result.success ==
        False``) is treated exactly like a raising solver, honouring
        ``on_step_error``.

        Change recording: deltas between the previous step's input network and
        this step's are recorded before the solve and attributed to this step
        - including for a step that fails under ``on_step_error='skip'`` (the
        failed step still appears in history). Under ``'raise'`` the recording
        is rolled back, so a raising step leaves no change events; the deltas
        are re-detected and attributed to the next attempted step.

        Failed-interval time accounting: under ``on_step_error='skip'`` a
        failed step's elapsed interval (including any carried backlog) is
        accumulated into the ``dt_h`` seen by the next successful step's
        inter-step integration, so temporal integration (storage SoC,
        linepack, ...) stays conservative. ``t_h`` always advances by the raw
        ``dt_h``."""
        if dt_h <= 0:
            raise ValueError(f"dt_h must be > 0, got {dt_h}")

        net_copy = self._work_net.copy()
        if self._timeseries_data is not None and ts_index is not None:
            self._timeseries_data.apply_to_network(net_copy, ts_index)
        if data_overrides:
            _apply_overrides(net_copy, data_overrides)

        step_idx = self._step_count
        t_h_at_step = self._t_h
        recording_checkpoint = self._recording_checkpoint()
        topo_changes = self._record_topology_changes(net_copy, step_idx, t_h_at_step)

        solve_dt_h = dt_h + self._carry_dt_h
        self._state.dt_h = solve_dt_h
        try:
            result = self._solver.solve(
                net_copy,
                optimization_problem=self._optimization_problem,
                step_state=self._state,
                **self._solver_kwargs,
            )
            if getattr(result, "success", True) is False:
                raise _unsuccessful_solve_error(step_idx, result)
        except Exception as exc:
            if self._on_step_error == "raise":
                self._rollback_recording(recording_checkpoint)
                raise
            _log.warning("Stepper step %d failed: %s", step_idx, exc)
            self._carry_dt_h = solve_dt_h
            sr = StepResult(step=step_idx, result=None, failed=True, error=exc)
            self._record(sr, dt_h)
            return sr

        self._carry_dt_h = 0.0
        self._record_islanding_changes(
            result.network, step_idx, t_h_at_step, topo_changes
        )
        self._state.push(result.network, step=step_idx)
        sr = StepResult(step=step_idx, result=result)
        self._record(sr, dt_h)
        return sr

    def _recording_checkpoint(self):
        """O(1) w.r.t. history: no copy of the change list, just its length and
        the running trim counter. Rollback truncates the entries appended since
        the checkpoint; entries a ``max_changes`` trim dropped from the front
        during the rolled-back step stay dropped (they would be trimmed again
        as soon as the same deltas are re-recorded)."""
        if not self._record_changes:
            return None
        return (
            len(self._changes),
            self._changes_dropped,
            self._prev_topology,
            dict(self._pending_annotations),
        )

    def _rollback_recording(self, checkpoint) -> None:
        if checkpoint is None:
            return
        n_changes, dropped, prev_topology, pending = checkpoint
        trimmed = self._changes_dropped - dropped
        del self._changes[max(n_changes - trimmed, 0) :]
        self._prev_topology = prev_topology
        self._pending_annotations = pending

    def _append_changes(self, new_changes: list[NetworkChange]) -> None:
        self._changes.extend(new_changes)
        if self._max_changes is not None and len(self._changes) > self._max_changes:
            drop = len(self._changes) - self._max_changes
            del self._changes[:drop]
            self._changes_dropped += drop

    def _record_topology_changes(self, net_copy, step_idx, t_h) -> list[NetworkChange]:
        """Diff the applied input network against the previous step and append
        any topology/active deltas (consuming explicit-mutation annotations)."""
        if not self._record_changes:
            return []
        cur = _topology_snapshot(net_copy)
        new_changes = _diff_topology(
            self._prev_topology, cur, step_idx, t_h, self._pending_annotations
        )
        self._append_changes(new_changes)
        self._prev_topology = cur
        return new_changes

    def _record_islanding_changes(
        self, result_net, step_idx, t_h, topo_changes: list[NetworkChange]
    ) -> None:
        """Diff solver islanding against the previous successful solve. A
        component whose topology change was already recorded this step (user
        mutation or detected input flip) does not get its solver echo
        (``islanded``/``rejoined``) re-reported; its dependents do. Which echo
        counts as such is derived from the component's post-change state rather
        than from the change kind: now-inactive suppresses ``islanded``,
        now-active or removed suppresses ``rejoined``. Genuine solver decisions
        still surface (e.g. a component added active but disconnected is
        reported ``islanded``)."""
        if not self._record_changes:
            return
        suppress_islanded: set[tuple[str, Any]] = set()
        suppress_rejoined: set[tuple[str, Any]] = set()
        # _prev_topology already holds this step's applied snapshot here.
        for change in topo_changes:
            key = (change.component_type, change.component_id)
            entry = self._prev_topology.get(key)
            if entry is not None and not entry[1]:
                suppress_islanded.add(key)
            else:
                suppress_rejoined.add(key)
        cur = _islanded_snapshot(result_net)
        self._append_changes(
            _diff_islanding(
                self._prev_islanded,
                cur,
                step_idx,
                t_h,
                suppress_islanded,
                suppress_rejoined,
            )
        )
        self._prev_islanded = cur

    def get(self, component_id, attr: str, step: int = -1):
        """Solved value of ``attr`` on component ``component_id`` - by default
        from the most recent successful step (the *get* side of a co-simulation
        adapter's set/step/get contract; ``data_overrides`` is the *set* side).

        ``step`` follows :meth:`StepState.get`: negative = relative to the
        latest successful solve, non-negative = the absolute step index as
        reported by ``StepResult.step`` (failed/skipped steps have no entry and
        fall back). Returns ``None`` (or the ``initial_state`` fallback) when
        no solve has written the value."""
        return self._state.get(component_id, attr, step=step)

    @property
    def changes(self) -> list[NetworkChange]:
        """Recorded :class:`NetworkChange` events in occurrence order (empty when
        ``record_changes=False``). Capped only by ``max_changes``, not by
        ``max_history``."""
        return list(self._changes)

    def changes_df(self) -> pandas.DataFrame:
        """Recorded changes as a DataFrame, one column per
        :class:`NetworkChange` field (step, t_h, kind, component_type,
        component_id, name, source)."""
        return pandas.DataFrame(
            [astuple(c) for c in self._changes],
            columns=[f.name for f in fields(NetworkChange)],
        )

    def _resolve(self, component_id, component_type=None):
        """Find ``(container_cls, component)`` for *component_id* on the working
        net. Node and child ids are independent counters and can collide, so an
        ambiguous bare id raises unless *component_type* (a container class like
        :class:`~monee.model.core.Child`, or a model class like ``PowerLoad``)
        or a ``(type, id)`` tuple narrows it. Raises ``KeyError`` if absent (a
        wiring bug, like ``data_overrides``)."""
        bare_id, matches = _match_components(
            self._work_net, component_id, component_type
        )
        if not matches:
            raise KeyError(f"component id {bare_id!r} not found in network")
        if len(matches) > 1:
            kinds = sorted({type(c).__name__ for c in matches})
            raise ValueError(
                f"component id {bare_id!r} is ambiguous across {kinds}; "
                "pass component_type= or a (type, id) tuple to disambiguate"
            )
        return type(matches[0]), matches[0]

    def _annotate_state_change(
        self, cls, comp, kind: str, source: str = "mutation"
    ) -> None:
        """Register the mutation's kind/source for the next step's diff. A
        mutation that lands back on the last recorded state is a no-op (nothing
        will be diffed), so any pending annotation is discarded instead - a
        stale entry would mislabel a later opposite-direction change."""
        if not self._record_changes:
            return
        key = (cls.__name__, comp.id)
        baseline = self._prev_topology.get(key)
        if baseline is not None and baseline[1] == _effective_active(comp):
            self._pending_annotations.pop(key, None)
        else:
            self._pending_annotations[key] = (kind, source)

    def _annotate_removal(self, cls, component_id, source: str = "mutation") -> None:
        """Called only after the network removal succeeded, so a raising
        removal cannot leave a stale annotation behind."""
        if not self._record_changes:
            return
        key = (cls.__name__, component_id)
        if key in self._prev_topology:
            self._pending_annotations[key] = ("removed", source)
        else:
            self._pending_annotations.pop(key, None)

    def _set_active_kind(self, component_id, component_type, active, kind) -> None:
        cls, comp = self._resolve(component_id, component_type)
        if active:
            self._work_net.activate_by_id(cls, comp.id)
        else:
            self._work_net.deactivate_by_id(cls, comp.id)
        self._annotate_state_change(cls, comp, kind)

    def deactivate(self, component_id, component_type=None) -> None:
        """Deactivate a component; sticks across steps. Recorded as
        ``deactivated`` at the next :meth:`step`. Pass *component_type* (e.g.
        ``mm.PowerLoad``) when a bare id is shared by a node and a child."""
        self._set_active_kind(component_id, component_type, False, "deactivated")

    def activate(self, component_id, component_type=None) -> None:
        """Reactivate a previously deactivated component. Recorded as
        ``activated`` at the next :meth:`step`."""
        self._set_active_kind(component_id, component_type, True, "activated")

    def fail(self, component_id, component_type=None) -> None:
        """Like :meth:`deactivate` but recorded as ``failed`` (fault semantics)."""
        self._set_active_kind(component_id, component_type, False, "failed")

    def restore(self, component_id, component_type=None) -> None:
        """Reactivate a failed/deactivated component, recorded as ``restored``."""
        self._set_active_kind(component_id, component_type, True, "restored")

    def remove(self, component_id, component_type=None) -> None:
        """Permanently remove a component (auto-dispatched by type). Recorded as
        ``removed`` at the next :meth:`step`. Cascading removals (e.g. a node's
        incident branches) are recorded as detected ``removed`` events."""
        cls, comp = self._resolve(component_id, component_type)
        getattr(self._work_net, _REMOVE_BY_TYPE[cls])(comp.id)
        self._annotate_removal(cls, comp.id)

    def remove_branch(self, branch_id) -> None:
        """Remove a branch by id (see :meth:`remove`)."""
        self.remove(branch_id, Branch)

    def remove_node(self, node_id) -> None:
        """Remove a node and its incident branches (see :meth:`remove`)."""
        self.remove(node_id, Node)

    def remove_child(self, child_id) -> None:
        """Remove a child by id (see :meth:`remove`)."""
        self.remove(child_id, Child)

    def remove_compound(self, compound_id) -> None:
        """Remove a compound and its subcomponents (see :meth:`remove`)."""
        self.remove(compound_id, Compound)

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
        """Clear step history, recreate the StepState, and discard structural
        mutations (the working net reverts to the base network)."""
        if initial_state is not None:
            self._initial_state = dict(initial_state)
        self._state = StepState(
            initial_state=self._initial_state, max_steps=self._max_history
        )
        self._history = []
        self._step_count = 0
        self._t_h = 0.0
        self._carry_dt_h = 0.0
        self._work_net = self._base_net.copy()
        self._changes = []
        self._changes_dropped = 0
        self._prev_topology = (
            _topology_snapshot(self._work_net) if self._record_changes else {}
        )
        self._prev_islanded = {}
        self._pending_annotations = {}

    def to_timeseries_result(
        self,
        datetime_index: pandas.DatetimeIndex | None = None,
    ) -> TimeseriesResult:
        """Wrap the retained history as a :class:`TimeseriesResult`. With
        ``max_history`` set this covers only the retained window."""
        return TimeseriesResult(
            list(self._history),
            datetime_index=datetime_index,
            backend_used=getattr(self._solver, "backend_name", None),
            solver_used=getattr(self._solver, "solver_name", None),
        )

    def __enter__(self) -> Stepper:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def __repr__(self) -> str:
        changes = f", changes={len(self._changes)}" if self._record_changes else ""
        return (
            f"Stepper(steps={self._step_count}, t_h={self._t_h:.4g}, "
            f"solver={type(self._solver).__name__}{changes})"
        )


def _apply_overrides(net: Network, overrides: Mapping[tuple, float]) -> None:
    """Apply ``{(comp_id, attr): value}`` via :meth:`TimeseriesData._set_model_attr`.

    ``comp_id`` may be a bare id or a ``(type, id)`` tuple (same forms as
    :meth:`Stepper._resolve`). Resolution is strict: among components sharing a
    bare id, only those with a *settable* ``attr`` (computed
    Intermediate/PostProcess attributes do not count) qualify, and more than
    one qualifying match raises rather than silently writing to all of them.
    Unknown id/attr raise (treated as wiring bugs, not transient failures)."""
    if not overrides:
        return

    by_id: dict[Any, list] = {}
    for comp in net.iter_all_components():
        by_id.setdefault(comp.id, []).append(comp)

    for (comp_id, attr), override_value in overrides.items():
        component_type, bare_id = _split_typed_id(comp_id)
        matches = _filter_by_type(by_id.get(bare_id, []), component_type)
        if not matches:
            raise KeyError(
                f"data_overrides: component id {bare_id!r} not found in network"
            )
        settable = [
            comp
            for comp in matches
            if hasattr(comp.model, attr)
            and not isinstance(getattr(comp.model, attr), (Intermediate, PostProcess))
        ]
        if not settable:
            raise AttributeError(
                f"data_overrides: attribute {attr!r} not found on any model "
                f"with id {bare_id!r}"
            )
        if len(settable) > 1:
            kinds = sorted({type(c).__name__ for c in settable})
            raise ValueError(
                f"data_overrides: component id {bare_id!r} with attribute "
                f"{attr!r} is ambiguous across {kinds}; use a "
                f"((type, id), attr) key to disambiguate"
            )
        TimeseriesData._set_model_attr(settable[0].model, attr, override_value)

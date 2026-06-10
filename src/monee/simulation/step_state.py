"""State carriers for timeseries (StepState — floats) and multi-period
(PeriodState — live solver vars). Both expose the same ``get(comp_id, attr)`` /
``dt_h`` API so ``inter_step_equations`` works in both modes unchanged."""

from __future__ import annotations

from abc import ABC, abstractmethod


def _find_model(net, component_id, attr=None):
    """Return the model for *component_id*. Disambiguates node/child id collisions
    by preferring a model that actually carries *attr*."""
    candidates = []
    for node in net.nodes:
        if node.id == component_id:
            candidates.append(node.model)
        for child in net.childs_by_ids(node.child_ids):
            if child.id == component_id:
                candidates.append(child.model)
    for branch in net.branches:
        if branch.id == component_id:
            candidates.append(branch.model)
    for compound in net.compounds:
        if compound.id == component_id:
            candidates.append(compound.model)

    if not candidates:
        return None
    if attr is not None and len(candidates) > 1:
        with_attr = [m for m in candidates if hasattr(m, attr)]
        if with_attr:
            return with_attr[0]
    return candidates[0]


def _extract_value(val):
    """Return a plain float from *val* (Var, Intermediate, float, int, or None)."""
    if val is None:
        return None
    from monee.model.core import Intermediate, PostProcess, Var

    if isinstance(val, (Var, Intermediate, PostProcess)):
        return val.value
    if isinstance(val, (int, float)):
        return float(val)
    return val


class InterStepState(ABC):
    """Abstract base. ``step``: negative = relative to current, non-negative = absolute."""

    dt_h: float = 1.0

    @abstractmethod
    def get(self, component_id, attr: str, step: int = -1):
        """Float (StepState) / live var (PeriodState), or None if no data."""

    def has(self, component_id, attr: str) -> bool:
        return self.get(component_id, attr) is not None


class StepState(InterStepState):
    """All previously-solved network copies (sequential timeseries). ``get`` returns floats.

    ``initial_state`` falls back when no prior solve has written the attribute."""

    def __init__(self, initial_state: dict | None = None) -> None:
        self._networks: list = []
        self.dt_h: float = 1.0
        self._initial_state: dict = dict(initial_state) if initial_state else {}

    def push(self, net) -> None:
        self._networks.append(net)

    def get(self, component_id, attr: str, step: int = -1):
        if self._networks:
            try:
                net = self._networks[step]
            except IndexError:
                net = None
            if net is not None:
                model = _find_model(net, component_id, attr)
                if model is not None:
                    val = _extract_value(getattr(model, attr, None))
                    if val is not None:
                        return val
        return self._initial_state.get((component_id, attr))

    def __len__(self) -> int:
        return len(self._networks)

    def __repr__(self) -> str:
        n = len(self._networks)
        return f"StepState({n} step{'s' if n != 1 else ''})"


class PeriodState(InterStepState):
    """All period networks (multi-period). ``get`` returns live solver vars after
    injection, so reading from another period becomes an algebraic cross-period
    constraint. Both past and future absolute indices are accessible."""

    def __init__(
        self,
        networks: list,
        current_t: int,
        dt_h: float = 1.0,
        initial_state: dict | None = None,
    ) -> None:
        self._networks = networks
        self.current_t = current_t
        self.dt_h = dt_h
        self._initial_state: dict = initial_state or {}

    @property
    def T(self) -> int:
        return len(self._networks)

    def get(self, component_id, attr: str, step: int = -1):
        actual_t = (self.current_t + step) if step < 0 else step

        if actual_t < 0:
            key = (component_id, attr)
            if key in self._initial_state:
                return self._initial_state[key]
            return None

        try:
            net = self._networks[actual_t]
        except IndexError:
            return None
        model = _find_model(net, component_id, attr)
        if model is None:
            return None
        return getattr(model, attr, None)

    def __repr__(self) -> str:
        return f"PeriodState(T={self.T}, current_t={self.current_t}, dt_h={self.dt_h})"

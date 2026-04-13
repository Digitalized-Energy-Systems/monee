"""
Network-copy-based state carriers for timeseries and multi-period solves.

Both classes implement :class:`InterStepState` and expose the same
``get(component_id, attr)`` / ``dt_h`` API so that models implementing
``inter_step_equations`` work identically in sequential timeseries and joint
multi-period optimisation.

``StepState`` (sequential timeseries)
    Accumulates the fully-solved network copies from all previous steps.
    ``get()`` returns plain Python floats extracted from ``Var.value``.

``PeriodState`` (multi-period)
    Holds all T period networks after variable injection (live solver objects).
    ``get()`` returns the solver variable directly, so an equation that reads
    from a ``PeriodState`` becomes an algebraic cross-period constraint inside
    the joint solver problem.

Because both classes return ``None`` when no prior network exists (step 0 /
period 0), model ``inter_step_equations`` implementations use a single
``if prev is None: prev = self._initial_value`` guard for both modes.

Any attribute is accessible — not just variables annotated as ``tracked``.
Models can reference velocity, flows, or any other solved quantity from
adjacent (or any) steps/periods.

Example — a ramp-constrained generator child model::

    class RampGenerator(ChildModel):
        def __init__(self, p_mw, ramp_up, ramp_down, **kwargs):
            super().__init__(**kwargs)
            self.p_mw = Var(-p_mw)
            self.ramp_up = ramp_up
            self.ramp_down = ramp_down

        def inter_temporal_equations(self, temporal_state: InterStepState, component_id, **kwargs):
            prev_p = temporal_state.get(component_id, 'p_mw')
            if prev_p is None:
                return []   # first step — no ramp constraint
            return [
                self.p_mw - prev_p <= self.ramp_up,
                prev_p - self.p_mw <= self.ramp_down,
            ]
"""

from __future__ import annotations

from abc import ABC, abstractmethod


def _find_model(net, component_id, attr=None):
    """Return the model for *component_id* in *net*, or ``None``.

    Node IDs and child IDs are independent namespaces that can collide.
    When *attr* is given and multiple models match, we return the first one
    that actually carries *attr*, disambiguating the common case.
    """
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
    from monee.model.core import Intermediate, Var

    if isinstance(val, (Var, Intermediate)):
        return val.value
    if isinstance(val, (int, float)):
        return float(val)
    return val


class InterStepState(ABC):
    """Abstract base for inter-step / inter-period state carriers.

    Both :class:`StepState` (sequential timeseries) and :class:`PeriodState`
    (multi-period optimisation) implement this interface.  Use as the type hint
    for ``inter_temporal_equations`` (works in both modes), ``inter_step_equations``
    (timeseries only, use :class:`StepState`), or ``inter_period_equations``
    (multi-period only, use :class:`PeriodState`).

    Negative index convention (shared by both subclasses):
        ``-1`` = one step/period before the current one,
        ``-2`` = two steps/periods back, etc.
    Non-negative indices are absolute (``0`` = first step/period).

    Attributes:
        dt_h: Duration of the current timestep/period in hours.
    """

    dt_h: float = 1.0

    @abstractmethod
    def get(self, component_id, attr: str, step: int = -1):
        """Return the value of *attr* on *component_id* at the given step/period.

        Args:
            component_id: Component id to look up.
            attr: Attribute name on the component's model.
            step: Relative (negative) or absolute (non-negative) index.
                ``-1`` = previous step/period (default).

        Returns:
            A float (``StepState``) or a live solver variable (``PeriodState``),
            or ``None`` when no data exists for the requested index.
        """

    def has(self, component_id, attr: str) -> bool:
        """Return ``True`` if a non-``None`` value is available."""
        return self.get(component_id, attr) is not None


class StepState(InterStepState):
    """
    Carries all previously-solved network copies for sequential timeseries.

    Any attribute from any prior step can be queried via :meth:`get`.
    Values are returned as plain Python floats (extracted from ``Var.value``
    after each step's ``withdraw_vars``).

    ``dt_h`` carries the duration of the *current* timestep in hours.
    Defaults to 1.0.
    """

    def __init__(self) -> None:
        self._networks: list = []
        self.dt_h: float = 1.0

    def push(self, net) -> None:
        """Append a fully-solved (post-withdrawal) network copy."""
        self._networks.append(net)

    def get(self, component_id, attr: str, step: int = -1):
        """Return the float value of *attr* on *component_id* at *step*.

        Args:
            component_id: Component id to look up.
            attr: Attribute name on the component's model.
            step: ``-1`` (default) = most recent solved step.
                ``-2`` = two steps back.  ``0`` = first step.

        Returns:
            Float value, or ``None`` when no prior steps exist or the
            component/attribute is absent.
        """
        if not self._networks:
            return None
        try:
            net = self._networks[step]
        except IndexError:
            return None
        model = _find_model(net, component_id, attr)
        if model is None:
            return None
        return _extract_value(getattr(model, attr, None))

    def __len__(self) -> int:
        return len(self._networks)

    def __repr__(self) -> str:
        n = len(self._networks)
        return f"StepState({n} step{'s' if n != 1 else ''})"


class PeriodState(InterStepState):
    """
    Carries all period networks for multi-period optimisation.

    Variable injection for all T periods has already run before this object
    is used, so ``get()`` returns live solver-library objects (GEKKO
    ``GKVariable`` / Pyomo ``Var``).  An equation that reads from a
    ``PeriodState`` therefore becomes an algebraic cross-period constraint
    inside the joint solver problem.

    Any period — past *or* future relative to ``current_t`` — is accessible
    by absolute index, enabling look-ahead constraints such as ramp limits
    that couple non-adjacent periods.

    Attributes:
        dt_h: Timestep duration in hours for the *current* period.
        current_t: Zero-based index of the period whose equations are
            currently being assembled.
        T: Total number of periods in the horizon.
    """

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
        """Total number of periods in the horizon."""
        return len(self._networks)

    def get(self, component_id, attr: str, step: int = -1):
        """Return the solver variable for *attr* on *component_id* at *step*.

        Args:
            component_id: Component id to look up.
            attr: Attribute name on the component's model.
            step: Negative values are relative to ``current_t``:
                ``-1`` (default) = ``current_t - 1``, ``-2`` = ``current_t - 2``,
                etc.  Non-negative values are absolute indices ``0..T-1``,
                giving direct access to any period including future ones.

        Returns:
            A live solver variable from the period's network, or ``None``
            when the effective period index is < 0 (before the horizon
            start) — triggering the initial-condition fallback in models.
            Values in *initial_state* take precedence for the virtual t=-1
            case.
        """
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

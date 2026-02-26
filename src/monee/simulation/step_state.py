"""
Per-step state carrier for coupled timeseries solves.

``StepState`` is passed through the timeseries loop and holds the solved
values of tracked model attributes from the *previous* timestep.  Models and
formulations that want inter-step coupling (e.g. ramp constraints) implement
two optional methods:

* ``inter_step_vars(self) -> list[str]``
      Return the attribute names whose post-solve values should be recorded in
      the state for use in the next timestep.

* ``inter_step_equations(self, prev_state: StepState, component_id, **kwargs) -> list``
      Called after variable injection, before the solver runs.  Return a list
      of relational expressions (``==``, ``<=``, ``>=``) linking the current
      solver variables to the previous-step values fetched from *prev_state*.

Both methods are detected via ``hasattr`` — classes that don't implement them
are completely unaffected.

Example — a ramp-constrained generator child model::

    class RampGenerator(PowerGenerator):
        def __init__(self, p_mw, q_mvar, ramp_up, ramp_down, **kwargs):
            super().__init__(p_mw, q_mvar, **kwargs)
            self.ramp_up = ramp_up
            self.ramp_down = ramp_down

        def inter_step_vars(self):
            return ['p_mw']

        def inter_step_equations(self, prev_state, component_id, **kwargs):
            prev_p = prev_state.get(component_id, 'p_mw')
            if prev_p is None:
                return []   # first timestep — no ramp constraint yet
            return [
                self.p_mw - prev_p <= self.ramp_up,
                prev_p - self.p_mw <= self.ramp_down,
            ]
"""


class StepState:
    """
    Holds the solved values of tracked model attributes from the previous
    timestep.

    Keys are ``(component_id, attribute_name)`` pairs.  Values are plain
    Python floats extracted after ``withdraw_vars`` — solver-library objects
    are never stored here.
    """

    def __init__(self) -> None:
        self._state: dict[tuple, float] = {}

    def get(self, component_id, attr: str, default=None):
        """
        Return the previous-step value for *attr* on *component_id*, or
        *default* (``None``) if not yet recorded (i.e. first timestep).
        """
        return self._state.get((component_id, attr), default)

    def set(self, component_id, attr: str, val: float) -> None:
        """Record a solved value for the next timestep."""
        self._state[(component_id, attr)] = val

    def has(self, component_id, attr: str) -> bool:
        """Return ``True`` if a value has been recorded for this key."""
        return (component_id, attr) in self._state

    def __repr__(self) -> str:
        return f"StepState({len(self._state)} entries)"

from abc import ABC


class NetworkAspect(ABC):
    """
    Solver-agnostic network-level extension.

    Analogous to ``BranchFormulation`` / ``NodeFormulation`` but spanning the
    entire network.  Register with ``network.add_extension(aspect)``.

    Phase 1 - ``prepare(network)``: called *before* variable injection; add
    ``Var`` placeholders to model objects so the injection loop picks them up.

    Phase 2 - ``equations(network, ignored_nodes) → list``: called *after*
    variable injection; return relational expressions (``==``, ``<=``,
    ``>=``) built from injected model attributes.  The solver registers them
    with ``m.Equations(eqs)`` / ``pm.cons.add`` without inspecting their
    content - exactly like branch/node equations.
    """

    def prepare(self, network) -> None:
        """Add Var placeholders before variable injection (no-op by default)."""

    def equations(self, network, ignored_nodes: set) -> list:  # NOSONAR
        """Return solver-agnostic relational expressions (empty by default)."""
        return []

    def inter_step_equations(
        self, network, ignored_nodes: set, step_state
    ) -> list:  # NOSONAR
        """Return expressions coupling current variables to previous-step values
        (timeseries only).  Empty by default."""
        return []

    def inter_period_equations(
        self, network, ignored_nodes: set, period_state
    ) -> list:  # NOSONAR
        """Return expressions coupling current variables to other period values
        (multi-period only).  Empty by default."""
        return []

    def inter_temporal_equations(
        self,
        network,
        ignored_nodes: set,
        temporal_state,  # NOSONAR
    ) -> list:
        """Return expressions coupling current variables to previous values
        (both timeseries and multi-period).  Empty by default."""
        return []

    def activate_timeseries(self, network, ignored_nodes: set, step_state=None) -> None:
        """
        Called after ``prepare()`` and variable injection, before node
        equations are assembled, when a timeseries or multi-period solve begins.

        Override to set flags on model objects that modify equation assembly
        for coupled solves - for example, suppressing a degenerate algebraic
        equation whose role is taken over by an inter-step constraint.

        *step_state* is provided so extensions can warm-start solver variable
        initializations from the previous step's solved values.
        No-op by default.
        """

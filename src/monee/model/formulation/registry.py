"""Formulation registry and solve-time attachment.

Formulations are a property of the *solve*, not of the network data: the
solver backends call :func:`attach_formulations` on their internal network
copy right before model assembly. The registry maps short string keys to the
built-in bundles so a solve can be parameterised as
``solve(net, formulation="convex_miqcqp")`` without imports.

Resolution order per component (most specific wins):

1. a per-component formulation pinned via the ``formulation=`` keyword of the
   :class:`~monee.model.network.Network` builder methods,
2. the ``formulation`` argument handed to the solver,
3. the network-level choice recorded by
   :meth:`~monee.model.network.Network.apply_formulation`,
4. :data:`~monee.model.formulation.bundles.DEFAULT_SIMULATION_FORMULATION`.
"""

from collections.abc import Sequence

from .bundles import (
    CONVEX_MIQCQP_FORMULATION,
    DEFAULT_SIMULATION_FORMULATION,
    EL_MISOCP_FORMULATION,
    EL_NLP_FORMULATION,
    EL_NONCONVEX_MIQCQP_FORMULATION,
    EL_QC_FORMULATION,
    GAS_CONVEX_MIQCQP_FORMULATION,
    GAS_NLP_FORMULATION,
    GAS_NONCONVEX_MIQCQP_FORMULATION,
    HEAT_CONVEX_MILP_FORMULATION,
    HEAT_NLP_FORMULATION,
    HEAT_NONCONVEX_MIQCQP_FORMULATION,
    NONCONVEX_MIQCQP_FORMULATION,
    SMOOTH_NLP_FORMULATION,
    combine,
    make_gas_milp_pwl_formulation,
    make_heat_nonconvex_pwl_formulation,
)
from .core import NetworkFormulation

# str key -> NetworkFormulation, or a zero-arg factory returning one.
FORMULATIONS: dict = {
    "default": DEFAULT_SIMULATION_FORMULATION,
    # bundles
    "smooth_nlp": SMOOTH_NLP_FORMULATION,
    "convex_miqcqp": CONVEX_MIQCQP_FORMULATION,
    "nonconvex_miqcqp": NONCONVEX_MIQCQP_FORMULATION,
    # electricity
    "el_nlp": EL_NLP_FORMULATION,
    "el_misocp": EL_MISOCP_FORMULATION,
    "el_qc": EL_QC_FORMULATION,
    "el_nonconvex_miqcqp": EL_NONCONVEX_MIQCQP_FORMULATION,
    # gas
    "gas_nlp": GAS_NLP_FORMULATION,
    "gas_convex_miqcqp": GAS_CONVEX_MIQCQP_FORMULATION,
    "gas_nonconvex_miqcqp": GAS_NONCONVEX_MIQCQP_FORMULATION,
    "gas_milp_pwl": make_gas_milp_pwl_formulation,
    # water / heat
    "heat_nlp": HEAT_NLP_FORMULATION,
    "heat_convex_milp": HEAT_CONVEX_MILP_FORMULATION,
    "heat_nonconvex_miqcqp": HEAT_NONCONVEX_MIQCQP_FORMULATION,
    "heat_nonconvex_pwl": make_heat_nonconvex_pwl_formulation,
}


def register_formulation(key: str, formulation) -> None:
    """Register *formulation* (a :class:`NetworkFormulation` or a zero-arg
    factory returning one) under *key* for use as a solve-time shortcut."""
    FORMULATIONS[key] = formulation


def resolve_formulation(spec) -> NetworkFormulation | None:
    """Resolve a formulation spec into a single :class:`NetworkFormulation`.

    Accepts ``None`` (no solver-level formulation), a registry key string, a
    :class:`NetworkFormulation`, or a sequence of either - sequences are
    merged left to right via :func:`~monee.model.formulation.bundles.combine`
    (later entries win on type collisions), e.g.
    ``("smooth_nlp", EL_MISOCP_FORMULATION)``.
    """
    if spec is None:
        return None
    if isinstance(spec, NetworkFormulation):
        return spec
    if isinstance(spec, str):
        try:
            entry = FORMULATIONS[spec]
        except KeyError:
            raise KeyError(
                f"Unknown formulation key {spec!r}. Registered keys: "
                f"{sorted(FORMULATIONS)}"
            ) from None
        return entry() if callable(entry) else entry
    if isinstance(spec, Sequence):
        return combine(*[resolve_formulation(item) for item in spec])
    raise TypeError(
        "formulation must be None, a registry key, a NetworkFormulation or a "
        f"sequence of those, got {type(spec).__name__}"
    )


def attach_formulations(network, formulation=None, simulation: bool = False) -> None:
    """Attach the effective formulation to every component of *network* and
    declare its variables (``ensure_var``).

    Called by the solver backends on their internal network copy, after
    extension ``prepare()`` and before problem application / variable
    injection. *formulation* is any spec accepted by
    :func:`resolve_formulation`; it overrides the network-level
    ``apply_formulation`` choice but not per-component pinned formulations.
    """
    solver_nf = resolve_formulation(formulation)

    for component in network.all_components():
        effective = None
        if getattr(component, "formulation_pinned", False):
            effective = component.formulation
        if effective is None and solver_nf is not None:
            effective = solver_nf.lookup(component.model, component.grid)
        if effective is None:
            effective = network.lookup_formulation(component.model, component.grid)
        if effective is None:
            # Manually assigned (component.formulation = ...) or left over from
            # a previous attach when re-solving a solver result network.
            effective = component.formulation
        if effective is None:
            effective = DEFAULT_SIMULATION_FORMULATION.lookup(
                component.model, component.grid
            )
        component.formulation = effective
        if effective is not None:
            effective.ensure_var(
                component.model, simulation=simulation, grid=component.grid
            )

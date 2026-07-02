"""
Cross-cutting helpers used by load-shedding / dispatch problems.
"""

from __future__ import annotations

from monee.model.core import Var
from monee.model.grid import DEFAULT_GAS_HHV_KWH_PER_KG, KGPS_KWHPERKG_TO_MW, GasGrid
from monee.model.multi import (
    CHPControlNode,
    CHPHGControlNode,
    GasToHeatControlNode,
    GasToHeatHG,
    GasToPower,
    PowerToGas,
    PowerToHeatControlNode,
    PowerToHeatHG,
)
from monee.model.node import Bus

# Fallback HHV (kWh/kg) when a gas grid is missing ``higher_heating_value_kwh_per_kg``.
_HHV_DEFAULT = DEFAULT_GAS_HHV_KWH_PER_KG

_UNBOUND_MAX_I_KA = 999.0


def line_loading_limit(branch_model, side: str, max_loading: float, basis: str = "mva"):
    """Cap branch loading at *max_loading* (per unit of the branch rating).

    ``basis="mva"`` (default) caps apparent power as the per-unit loading-squared
    form ``(p**2 + q**2) / (max_loading * max_s_mva)**2 <= 1`` - smooth, and the
    same solution as MATPOWER's |S|^2 <= RATE_A^2. The normalization is what makes
    it O(1): the bare ``p**2 + q**2 <= RATE_A**2`` form has residual/curvature of
    order RATE_A^2 (~1e6-1e8 in MW), which ill-conditions the OPF and balloons the
    IPOPT iteration count. When the branch carries no ``max_s_mva`` (None), it
    falls back to the current basis below, so networks that only have ``max_i_ka``
    keep their existing current-based behaviour.

    ``basis="current"`` caps current as |I| <= max_loading * max_i_ka (the
    original behaviour). Unbounded ratings drop the constraint.
    """
    if side not in ("from", "to"):
        raise ValueError(f"side must be 'from' or 'to', got {side!r}")
    if basis not in ("mva", "apparent", "current"):
        raise ValueError(f"basis must be 'mva' or 'current', got {basis!r}")

    if basis in ("mva", "apparent"):
        max_s_mva = getattr(branch_model, "max_s_mva", None)
        if max_s_mva is not None:
            p = getattr(branch_model, f"p_{side}_mw")
            q = getattr(branch_model, f"q_{side}_mvar")
            limit = max_loading * max_s_mva
            return (p**2 + q**2) / (limit * limit) <= 1.0
        # No MVA rating on this branch: fall through to the current basis.

    max_i_ka = getattr(branch_model, "max_i_ka", None)
    if max_i_ka is None or max_i_ka >= _UNBOUND_MAX_I_KA:
        return True
    scale_attr = f"_misocp_loading_{side}_scale_squared"
    if hasattr(branch_model, scale_attr):
        scale_sq = getattr(branch_model, scale_attr)
        if scale_sq <= 0:
            return True
        return branch_model.current_pu_squared <= (max_loading * max_loading) / scale_sq
    return getattr(branch_model, f"loading_{side}_pu") <= max_loading


def make_node_var_bounds_hook(node_type, attr, squared_attr, bounds):
    """``_controllable_appliables`` hook bounding *attr* (and ``lo²..hi²`` on
    *squared_attr*) on independent nodes of exactly *node_type*. Only Var-typed
    attributes are touched: whichever of the pair the formulation uses as its
    actual decision variable gets the bound, while reporting Intermediates are
    left alone (a static bound there would be a no-op)."""
    lo, hi = bounds

    def _apply_bounds(network):
        for component in network.all_components():
            model = component.model
            if type(model) is not node_type or not component.independent:
                continue
            v = getattr(model, attr, None)
            vsq = getattr(model, squared_attr, None)
            if type(v) is Var:
                v.min, v.max = lo, hi
            if type(vsq) is Var:
                vsq.min, vsq.max = lo * lo, hi * hi

    return _apply_bounds


def make_vm_bounds_hook(bounds_vm):
    """Bound bus voltage magnitudes on the formulation's actual decision
    variable: ``vm_pu_squared`` under the MISOCP relaxation, ``vm_pu`` under
    the AC NLP."""
    return make_node_var_bounds_hook(Bus, "vm_pu", "vm_pu_squared", bounds_vm)


def cp_input_rated_mw(component):  # NOSONAR

    model = component.model
    grid = getattr(component, "grid", None)

    def _hhv():
        gg = None
        if isinstance(grid, dict):
            gg = grid.get(GasGrid)
        elif isinstance(grid, list):
            gg = next((g for g in grid if isinstance(g, GasGrid)), None)
        elif isinstance(grid, GasGrid):
            gg = grid
        return (
            getattr(gg, "higher_heating_value_kwh_per_kg", _HHV_DEFAULT)
            if gg
            else _HHV_DEFAULT
        )

    def _scalar(x):
        # Read the bound (or the value) off a Var; pass scalars through.
        if isinstance(x, Var):
            fallback = x.value if x.value is not None else 0.0
            return x.max if x.max is not None else fallback
        return x if x is not None else 0.0

    # ---- Gas-input CPs ----------------------------------------------------
    if isinstance(model, (CHPControlNode, CHPHGControlNode, GasToHeatControlNode)):
        return (
            "gas",
            abs(_scalar(model.gas_mass_flow_kgs)) * KGPS_KWHPERKG_TO_MW * _hhv(),
        )
    if isinstance(model, GasToPower):
        eff = max(getattr(model, "efficiency", 1.0), 1e-6)
        # el_mw is stored as -p_mw_setpoint (scalar).
        return ("gas", abs(_scalar(model.el_mw)) / eff)
    if isinstance(model, GasToHeatHG):
        eff = max(getattr(model, "efficiency", 1.0), 1e-6)
        return ("gas", abs(_scalar(model.heat_mw)) / eff)

    # ---- Power-input CPs --------------------------------------------------
    if isinstance(model, PowerToHeatControlNode):
        return ("power", abs(_scalar(model.el_mw)))
    if isinstance(model, PowerToHeatHG):
        return ("power", abs(_scalar(model.load_p_mw)))
    if isinstance(model, PowerToGas):
        eff = max(getattr(model, "efficiency", 1.0), 1e-6)
        # gas_mass_flow_kgs stored as -mass_flow_setpoint_kgs; rated input power = output / η.
        return (
            "power",
            abs(_scalar(model.gas_mass_flow_kgs)) * KGPS_KWHPERKG_TO_MW * _hhv() / eff,
        )

    return None

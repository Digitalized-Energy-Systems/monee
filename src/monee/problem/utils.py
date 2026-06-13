"""
Cross-cutting helpers used by load-shedding / dispatch problems.
"""

from __future__ import annotations

from monee.model.core import Var
from monee.model.grid import KGPS_KWHPERKG_TO_MW, GasGrid
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

# Fallback HHV (kWh/kg) when a gas grid is missing ``higher_heating_value_kwh_per_kg``.
_HHV_DEFAULT = 15.3


def line_loading_limit(branch_model, side: str, max_loading: float):
    """LP-writable line-loading constraint.

    AC: ``loading_*_percent ≤ max``. MISOCP: emit
    ``current_pu_squared · scale² ≤ max²`` using the formulation-stashed
    ``_misocp_loading_*_scale_squared`` (the sqrt form is LP-incompatible).
    """
    if side not in ("from", "to"):
        raise ValueError(f"side must be 'from' or 'to', got {side!r}")
    scale_attr = f"_misocp_loading_{side}_scale_squared"
    if hasattr(branch_model, scale_attr):
        scale_sq = getattr(branch_model, scale_attr)
        return branch_model.current_pu_squared * scale_sq <= max_loading * max_loading
    return getattr(branch_model, f"loading_{side}_pu") <= max_loading


def cp_input_rated_mw(component):
    """Return ``(carrier, rated_input_mw)`` for a coupling-point component or
    ``None`` if *component* is not a coupling point.

    ``carrier`` is ``'gas'`` or ``'power'`` (the carrier the CP draws *from*).
    ``rated_input_mw`` is the nameplate input the CP consumes at regulation 1.0.
    Used both by the resilience metric (to account CP curtailment) and by
    ``min_load_shedding`` (to treat CPs as additional loads in the objective).
    """
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
        return getattr(gg, "higher_heating_value_kwh_per_kg", _HHV_DEFAULT) if gg else _HHV_DEFAULT

    def _scalar(x):
        # Read the bound (or the value) off a Var; pass scalars through.
        if isinstance(x, Var):
            return (
                x.max
                if x.max is not None
                else (x.value if x.value is not None else 0.0)
            )
        return x if x is not None else 0.0

    # ---- Gas-input CPs ----------------------------------------------------
    if isinstance(model, (CHPControlNode, CHPHGControlNode, GasToHeatControlNode)):
        return ("gas", abs(_scalar(model.gas_mass_flow_kgs)) * KGPS_KWHPERKG_TO_MW * _hhv())
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
        return ("power", abs(_scalar(model.gas_mass_flow_kgs)) * KGPS_KWHPERKG_TO_MW * _hhv() / eff)

    return None

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

# Fallback HHV (kWh/kg) when a gas grid is missing ``higher_heating_value_kwh_per_kg``.
_HHV_DEFAULT = DEFAULT_GAS_HHV_KWH_PER_KG

_UNBOUND_MAX_I_KA = 999.0


def line_loading_limit(branch_model, side: str, max_loading: float):

    if side not in ("from", "to"):
        raise ValueError(f"side must be 'from' or 'to', got {side!r}")
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

"""
Cross-cutting helpers used by load-shedding / dispatch problems.
"""

from __future__ import annotations


def line_loading_limit(branch_model, side: str, max_loading: float):
    """LP-writable line-loading constraint.

    AC: ``loading_*_percent ≤ max``. MISOCP: emit
    ``current_pu · scale² ≤ max²`` using the formulation-stashed
    ``_misocp_loading_*_scale_squared`` (the sqrt form is LP-incompatible).
    """
    if side not in ("from", "to"):
        raise ValueError(f"side must be 'from' or 'to', got {side!r}")
    scale_attr = f"_misocp_loading_{side}_scale_squared"
    if hasattr(branch_model, scale_attr):
        scale_sq = getattr(branch_model, scale_attr)
        return branch_model.current_pu * scale_sq <= max_loading * max_loading
    return getattr(branch_model, f"loading_{side}_percent") <= max_loading

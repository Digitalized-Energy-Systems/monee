"""
Cross-cutting helpers used by load-shedding / dispatch problems.
"""

from __future__ import annotations


def line_loading_limit(branch_model, side: str, max_loading: float):
    """Return an LP-writable line-loading constraint for *branch_model*.

    Under the AC formulation ``loading_*_percent`` is a free ``Var`` pinned
    by the linear identity ``loading == i / max_i_ka``, so the trivial form
    ``loading <= max`` works in the LP writer.

    Under MISOCP ``loading_*_percent`` is an ``Intermediate`` whose post-solve
    Pyomo ``Expression`` carries ``sqrt(current_pu) · I_base / max_i_ka`` —
    a transcendental term the LP writer rejects.  The formulation stashes a
    pre-computed loading-scale on the branch in ``_misocp_loading_*_scale_squared``
    so the equivalent linear-in-``current_pu`` constraint can be emitted here:

        loading² = current_pu · scale²   ⟹   loading <= max
                                           ⟺   current_pu · scale² <= max²

    Args:
        branch_model: The branch's model object (i.e. ``component.model``).
        side: ``"from"`` or ``"to"``.
        max_loading: Loading limit as a fraction (e.g. ``1.0`` for 100%).
    """
    if side not in ("from", "to"):
        raise ValueError(f"side must be 'from' or 'to', got {side!r}")
    scale_attr = f"_misocp_loading_{side}_scale_squared"
    if hasattr(branch_model, scale_attr):
        scale_sq = getattr(branch_model, scale_attr)
        return branch_model.current_pu * scale_sq <= max_loading * max_loading
    return getattr(branch_model, f"loading_{side}_percent") <= max_loading

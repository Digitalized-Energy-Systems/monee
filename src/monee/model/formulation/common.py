"""Sector node formulations shared across optimization classes.

The junction (node) side of the gas and water/heat sectors is the same for
every branch formulation family: it only declares the pressure state and the
reporting intermediates. The class-specific physics (relaxed / exact / smooth /
PWL) lives entirely on the branch side, so the node formulations are defined
once here and reused by ``nlp``, ``milp`` and ``miqcqp``.
"""

import math

from monee.model.branch import WaterPipe
from monee.model.core import Const, Intermediate, IntermediateEq, PostProcess, Var
from monee.model.phys.core.hydraulics import calc_pipe_area
from monee.model.phys.nonlinear.gf import reference_gas_density
from monee.model.phys.nonlinear.hf import SPECIFIC_HEAT_CAP_WATER

from .core import NodeFormulation


def velocity_post_process(model, fluid_density_kg_per_m3):
    r"""Deterministic velocity report :math:`v = \dot m / (\rho A)` for a pipe-like
    branch, as a :class:`PostProcess` so it adds no solver variable/equation.
    ``fluid_density_kg_per_m3`` is the (reference) density the velocity caps use.
    The area is taken from the solved ``diameter_m``, which a problem may have
    promoted to a decision variable after this report was attached."""
    return PostProcess(
        lambda v, rho=fluid_density_kg_per_m3: (
            (v.mass_flow_pos_kgs - v.mass_flow_neg_kgs)
            / (rho * calc_pipe_area(v.diameter_m))
        )
    )


def ensure_velocity_report(model, grid):
    """Attach the ``velocity_mps`` report using the grid's density: the
    reference gas density for gas grids, ``fluid_density_kg_per_m3`` for
    water/heat grids (the density behind the velocity-based flow caps).
    No-op when the grid exposes neither."""
    if grid is None:
        return
    if hasattr(grid, "molar_mass"):
        model.velocity_mps = velocity_post_process(model, reference_gas_density(grid))
    elif hasattr(grid, "fluid_density_kg_per_m3"):
        model.velocity_mps = velocity_post_process(model, grid.fluid_density_kg_per_m3)
        ensure_heat_loss_report(model, grid)


def ensure_heat_loss_report(model, grid):
    r"""Attach the ``heat_loss_mw`` report of a water pipe: the heat its water
    loses to the ground, :math:`|\dot m| c_p (T_{in} - T_{out})` in MW. NaN
    under formulations without inlet/outlet temperatures (McCormick MILP)."""
    if isinstance(model, WaterPipe):
        model.heat_loss_mw = PostProcess(
            lambda v, k=SPECIFIC_HEAT_CAP_WATER * grid.t_ref_k / 1e6: (
                v.mass_flow_mag_kgs * (v.t_in_pu - v.t_out_pu) * k
            )
        )


_PSQ_SIMULATION_BOUNDS = (0, 3)
# Bounds the junction models declare themselves (node.Junction, multi.py control
# nodes) plus the ones a previous attach left behind: anything else on the
# incoming Var is a user-set limit and wins over the grid's.
_PSQ_DEFAULT_MIN = (0, 0.5, None)
_PSQ_DEFAULT_MAX = (2, 3, None)


def _psq_bounds(model, simulation, grid):
    """Pressure-squared Var bounds: the grid's operational limits in
    optimization mode, the wide simulation range otherwise (a square solve must
    never be cut off by an operational limit)."""
    lo, hi = _PSQ_SIMULATION_BOUNDS
    if simulation or grid is None:
        return lo, hi
    existing = getattr(model, "pressure_squared_pu", None)
    if isinstance(existing, Var) and existing.min not in _PSQ_DEFAULT_MIN:
        lo = existing.min
    else:
        lo = getattr(grid, "pressure_squared_pu_min", lo)
    if isinstance(existing, Var) and existing.max not in _PSQ_DEFAULT_MAX:
        hi = existing.max
    else:
        hi = getattr(grid, "pressure_squared_pu_max", hi)
    return lo, hi


def check_segment_length(segment_length_m):
    """*segment_length_m* if it is ``None`` or positive, else ``ValueError``."""
    if segment_length_m is not None and segment_length_m <= 0:
        raise ValueError(f"segment_length_m must be > 0, got {segment_length_m!r}.")
    return segment_length_m


def heat_loss_cells(length_m, segment_length_m) -> int:
    """Number of equal cells of the segmented insulation loss: the smallest
    power of two whose cells are no longer than ``segment_length_m``; 1 (the
    plain one-cell law) for ``None`` or a pipe that is short enough."""
    if segment_length_m is None or length_m <= segment_length_m:
        return 1
    return 2 ** math.ceil(math.log2(length_m / segment_length_m))


def _cell_var_name(k: int) -> str:
    return "alpha_cell" if k == 0 else f"alpha_cell_pow{2**k}"


def ensure_heat_loss_cells(model, segment_length_m) -> None:
    """Declare the cell attenuation ``alpha_cell`` and its repeated squares
    ``alpha_cell_pow2``, ``alpha_cell_pow4``, ... used by
    :func:`pipe_attenuation_eqs`; nothing for a one-cell pipe."""
    n = heat_loss_cells(model.length_m, segment_length_m)
    for k in range(int(math.log2(n))):
        setattr(
            model,
            _cell_var_name(k),
            Var(0.01 ** (2**k / n), min=0, max=1, name=_cell_var_name(k)),
        )


def pipe_attenuation_eqs(branch, mag, ua_c, segment_length_m=None) -> list:
    r"""Insulation attenuation :math:`\alpha` of a water pipe of :math:`n`
    equal cells (:func:`heat_loss_cells`), each losing heat at its outlet
    temperature: :math:`\beta (|\dot m| + UA/(n c_p)) = |\dot m|`,
    :math:`\alpha = \beta^n`. :math:`\alpha` tends to the exact
    :math:`e^{-UA/(|\dot m| c_p)}` as :math:`n` grows; :math:`n = 1` is the
    plain law. The power is built by repeated squaring, so the law stays
    quadratic with :math:`\log_2 n` extra bilinear equations."""
    n = heat_loss_cells(branch.length_m, segment_length_m)
    if n == 1:
        return [branch.alpha * (mag + ua_c) == mag]
    powers = [getattr(branch, _cell_var_name(k)) for k in range(int(math.log2(n)))]
    eqs = [powers[0] * (mag + ua_c / n) == mag]
    eqs += [b == a * a for a, b in zip(powers, powers[1:])]
    eqs.append(branch.alpha == powers[-1] * powers[-1])
    return eqs


def pipe_outlet_law(branch, grid) -> list:
    r"""Steady insulation loss :math:`t_{out} = T_{ext} + \alpha (t_{in} - T_{ext})`
    of a water pipe; empty while
    :class:`~monee.model.extension.plugflow.PlugFlowTransport` states the
    outlet temperature of the step instead."""
    if getattr(branch, "_plug_flow_active", False):
        return []
    t_ext = branch.temperature_ext_k / grid.t_ref_k
    return [branch.t_out_pu == t_ext + branch.alpha * (branch.t_in_pu - t_ext)]


class GasNodeFormulation(NodeFormulation):
    r"""Gas junction working in pressure-squared space.

    ``pressure_squared_pu`` is the decision variable (Weymouth is linear in
    :math:`p^2`); ``pressure_pu`` is a reporting intermediate.

    Pressure convention: the node pressure (and the reported ``pressure_pa``)
    follows the grid's convention - it is ABSOLUTE when the gas grid's
    ``pressure_ambient_pa`` is 0 (the default) and GAUGE when it is set (e.g. to
    ``STANDARD_ATMOSPHERE_PA``). In gauge mode the Weymouth/density physics add
    the ambient internally; only the reported value stays gauge.
    """

    def ensure_var(self, model, simulation=False, grid=None):
        model.pressure_pa = PostProcess(lambda v: float("nan"))
        model.pressure_pu = Intermediate(1)
        lo, hi = _psq_bounds(model, simulation, grid)
        model.pressure_squared_pu = Var(1, min=lo, max=hi, name="pressure_sq_pu")

        if simulation:
            t = getattr(model, "t_pu", None)
            if isinstance(t, Var):
                model.t_pu = Const(t.value)

    def equations(
        self,
        node,
        grid,
        from_branch_models,
        to_branch_models,
        connected_child_models,
        **kwargs,
    ):
        node.pressure_pa = PostProcess(
            lambda v, ref=grid.pressure_ref_pa: v.pressure_pu * ref
        )
        return [
            IntermediateEq(
                "pressure_pu", kwargs["sqrt_impl"](node.pressure_squared_pu)
            ),
        ]


class WaterNodeFormulation(NodeFormulation):
    """Water/heat junction working directly in pressure space."""

    def ensure_var(self, model, simulation=False, grid=None):
        model.pressure_pa = PostProcess(lambda v: float("nan"))
        model.pressure_pu = Var(1, min=0, max=2, name="pressure_pu")
        model.pressure_squared_pu = Intermediate(1)

    def equations(
        self,
        node,
        grid,
        from_branch_models,
        to_branch_models,
        connected_child_models,
        **kwargs,
    ):
        node.pressure_pa = PostProcess(
            lambda v, ref=grid.pressure_ref_pa: v.pressure_pu * ref
        )
        return []

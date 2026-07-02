"""Sector node formulations shared across optimization classes.

The junction (node) side of the gas and water/heat sectors is the same for
every branch formulation family: it only declares the pressure state and the
reporting intermediates. The class-specific physics (relaxed / exact / smooth /
PWL) lives entirely on the branch side, so the node formulations are defined
once here and reused by ``nlp``, ``milp`` and ``miqcqp``.
"""

from monee.model.core import Const, Intermediate, IntermediateEq, PostProcess, Var
from monee.model.phys.core.hydraulics import calc_pipe_area
from monee.model.phys.nonlinear.gf import reference_gas_density

from .core import NodeFormulation


def velocity_post_process(model, fluid_density_kg_per_m3):
    r"""Deterministic velocity report :math:`v = \dot m / (\rho A)` for a pipe-like
    branch, as a :class:`PostProcess` so it adds no solver variable/equation.
    ``fluid_density_kg_per_m3`` is the (reference) density the velocity caps use."""
    scale = fluid_density_kg_per_m3 * calc_pipe_area(model.diameter_m)
    return PostProcess(
        lambda v, s=scale: (v.mass_flow_pos_kgs - v.mass_flow_neg_kgs) / s
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
        model.pressure_squared_pu = Var(1, min=0, max=3, name="pressure_sq_pu")

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

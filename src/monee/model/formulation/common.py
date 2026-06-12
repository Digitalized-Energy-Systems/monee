"""Sector node formulations shared across optimization classes.

The junction (node) side of the gas and water/heat sectors is the same for
every branch formulation family: it only declares the pressure state and the
reporting intermediates. The class-specific physics (relaxed / exact / smooth /
PWL) lives entirely on the branch side, so the node formulations are defined
once here and reused by ``nlp``, ``milp`` and ``miqcqp``.
"""

from monee.model.core import Const, Intermediate, IntermediateEq, PostProcess, Var

from .core import NodeFormulation


class GasNodeFormulation(NodeFormulation):
    """Gas junction working in pressure-squared space.

    ``pressure_squared_pu`` is the decision variable (Weymouth is linear in
    p²); ``pressure_pu`` is a reporting intermediate.
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
            lambda v, ref=grid.pressure_ref: v.pressure_pu * ref
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
            lambda v, ref=grid.pressure_ref: v.pressure_pu * ref
        )
        return []

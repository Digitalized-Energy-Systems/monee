import math

from .core import ChildModel, Const, Var, model


class GridFormingMixin:
    """
    Marker: this child can serve as the slack/reference for an islanded sub-network.

    Carriers must implement ``overwrite()`` to pin their reference variable.
    Islanding keeps components containing a ``GridFormingMixin`` child in the solve.
    """


class NoVarChildModel(ChildModel):
    """:class:`ChildModel` with only scalar parameters and no equations of its own."""

    def set(self, n, value):
        user_attributes = [
            attr
            for attr in dir(self)
            if not attr.startswith("__") and (not callable(getattr(self, attr)))
        ]
        if n < 0 or n >= len(user_attributes):
            raise IndexError(f"No user-defined attribute at index {n}")
        attr_name = user_attributes[n]
        setattr(self, attr_name, value)

    def equations(self, grid, node, **kwargs):
        return []


@model
class PowerGenerator(NoVarChildModel):
    """Fixed-setpoint active/reactive generator. Constructor takes positive magnitudes; sign is internal."""

    def __init__(self, p_mw, q_mvar, **kwargs) -> None:
        # Compound models pass solver Vars for p_mw — only validate plain numerics.
        if isinstance(p_mw, (int, float)) and p_mw < 0:
            raise ValueError(
                f"PowerGenerator expects a positive generation magnitude; "
                f"got p_mw={p_mw}.  Pass the absolute value — the sign is "
                f"handled internally (load convention)."
            )
        super().__init__(**kwargs)
        self.p_mw = -p_mw
        self.q_mvar = -q_mvar


@model
class ExtPowerGrid(NoVarChildModel, GridFormingMixin):
    """
    External slack-bus connection. Pins vm_pu and va_degree, leaves p_mw/q_mvar
    as free Vars absorbing the island's imbalance. Load convention: positive
    p_mw = import.
    """

    def __init__(
        self,
        p_mw,
        q_mvar,
        vm_pu=1,
        va_degree=0,
        max_import_mw=None,
        max_export_mw=None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.p_mw = Var(
            p_mw,
            min=None if max_import_mw is None else -max_import_mw,
            max=max_export_mw,
            name="ext_grid_p_mw",
        )
        self.q_mvar = Var(q_mvar, name="ext_grid_q_mvar")
        self.vm_pu = vm_pu
        self.va_degree = va_degree

    def overwrite(self, node_model, grid):
        """Pin the bus voltage magnitude and angle to the configured setpoints."""
        node_model.vm_pu = Const(self.vm_pu)
        node_model.vm_pu_squared = Const(self.vm_pu * self.vm_pu)
        node_model.va_degree = Const(self.va_degree)
        # The slack bus is the angle reference: pin the angle decision variable
        # (va_degree is a derived Intermediate) — removes the free global-gauge
        # DOF, improving conditioning and squareness. Skipped when electricity
        # islanding manages bus angles itself (energisation-gated), which it
        # flags in its prepare() before this runs.
        if not getattr(node_model, "_islanding_angle_managed", False):
            node_model.va_radians = Const(self.va_degree * math.pi / 180)


@model
class PowerLoad(NoVarChildModel):
    """Fixed-setpoint power load. Load convention: positive = consumption."""

    def __init__(self, p_mw, q_mvar, **kwargs) -> None:
        super().__init__(**kwargs)
        self.p_mw = p_mw
        self.q_mvar = q_mvar


@model
class Source(NoVarChildModel):
    """Fixed-setpoint mass-flow source. Constructor takes positive magnitude; sign is internal."""

    def __init__(self, mass_flow, **kwargs) -> None:
        # Internal callers may pass solver Vars — only validate plain numerics.
        if isinstance(mass_flow, (int, float)) and mass_flow < 0:
            raise ValueError(
                f"Source expects a positive injection magnitude; "
                f"got mass_flow={mass_flow}.  Pass the absolute value — the "
                f"sign is handled internally (load convention)."
            )
        super().__init__(**kwargs)
        self.mass_flow = -mass_flow


@model
class ExtHydrGrid(NoVarChildModel, GridFormingMixin):
    """
    External hydraulic slack source. Pins pressure (and optionally temperature),
    leaves mass_flow as a free Var. Load convention: negative mass_flow = injection.
    """

    def __init__(
        self,
        mass_flow=-1,
        pressure_pu=1,
        t_k=356,
        max_import_kgs=None,
        max_export_kgs=None,
        pin_temperature=True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.mass_flow = Var(
            mass_flow,
            min=None if max_import_kgs is None else -max_import_kgs,
            max=max_export_kgs,
            name="ext_grid_mass_flow",
        )
        self.pressure_pu = pressure_pu
        self.t_k = t_k
        self.pin_temperature = pin_temperature

    def overwrite(self, node_model, grid):
        """Pin pressure; pin temperature only if ``pin_temperature`` (default True).
        Set False on return-side slacks so T emerges from upstream heat balance."""
        node_model.pressure_pu = Const(self.pressure_pu)
        node_model.pressure_squared_pu = Const(self.pressure_pu**2)
        if self.pin_temperature:
            node_model.t_pu = Const(self.t_k / grid.t_ref)
            node_model.t_k = Const(self.t_k)


@model
class ConsumeHydrGrid(NoVarChildModel):
    """Hydraulic demand point: fixed pressure setpoint plus a free mass_flow Var."""

    def __init__(self, mass_flow=0.1, pressure_pu=1, t_k=293, **kwargs) -> None:
        super().__init__(**kwargs)
        self.mass_flow = Var(
            mass_flow,
            name="consume_ext_grid_mass_flow",
        )
        self.pressure_pu = pressure_pu
        self.t_k = t_k


@model
class HeatGenerator(NoVarChildModel):
    """Node-based heat injection (``H_G,i``). Takes positive magnitude; sign is internal."""

    def __init__(self, q_mw, **kwargs) -> None:
        if isinstance(q_mw, (int, float)) and q_mw < 0:
            raise ValueError(
                f"HeatGenerator expects a positive heat-generation magnitude; "
                f"got q_mw={q_mw}.  Pass the absolute value — the sign is "
                f"handled internally (load convention)."
            )
        super().__init__(**kwargs)
        self.q_mw_heat = -q_mw


@model
class HeatLoad(NoVarChildModel):
    """Node-based heat withdrawal (``H_L,i``). Positive q_mw = consumption."""

    def __init__(self, q_mw, **kwargs) -> None:
        super().__init__(**kwargs)
        self.q_mw_heat = q_mw


@model
class Sink(NoVarChildModel):
    """Fixed-setpoint mass-flow sink. Positive = consumption (load convention)."""

    def __init__(self, mass_flow, **kwargs) -> None:
        super().__init__(**kwargs)
        self.mass_flow = mass_flow

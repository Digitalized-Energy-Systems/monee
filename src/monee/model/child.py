from .core import ChildModel, Const, Var, model


class GridFormingMixin:
    """
    Marker mixin: this child component can serve as the reference node (slack bus /
    pressure reference) for an islanded sub-network.

    Any child class that carries this mixin *must* implement ``overwrite()`` to pin
    the carrier-specific reference variable (voltage angle, pressure, …) on the node
    model it is attached to.  When islanding is enabled, ``find_ignored_nodes`` treats
    components that contain a ``GridFormingMixin`` child as "leading" and keeps them in
    the solve.
    """


class NoVarChildModel(ChildModel):
    """
    A :class:`ChildModel` whose parameters are plain scalars (no ``Var`` decision variables).

    All attributes are fixed constants during a solve.  The ``equations()``
    method returns an empty list because the component's contribution to the
    system equations comes solely through node balance equations that reference
    the stored scalar values.
    """

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
    """
    Fixed-setpoint active/reactive power generator.

    Follows the load convention: internally stores *negative* values so that
    the node balance sees this component as an injection.  The constructor
    accepts positive magnitudes and negates them::

        PowerGenerator(p_mw=5, q_mvar=0)  →  p_mw=-5, q_mvar=0

    Args:
        p_mw (float): Active power output in MW (positive = generation).
        q_mvar (float): Reactive power output in Mvar (positive = generation).
    """

    def __init__(self, p_mw, q_mvar, **kwargs) -> None:
        # isinstance guard: compound models (e.g. CHP) internally construct
        # PowerGenerator with a live solver variable as p_mw. Sign validation
        # only makes sense for plain numeric inputs from user code.
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
    External (slack) power grid connection — the reference bus for an electrical island.

    ``ExtPowerGrid`` pins the bus voltage magnitude and angle to fixed setpoints
    (via :meth:`overwrite`) and exposes ``p_mw`` / ``q_mvar`` as free
    :class:`Var` decision variables that absorb the island's power imbalance.

    Follows the load convention: positive ``p_mw`` / ``q_mvar`` at the node
    represents net *import* from the external grid (consumption perspective).
    The ``p_mw`` / ``q_mvar`` initial values are starting guesses; the solver
    determines the final values.

    Args:
        p_mw (float): Initial active power exchange in MW.
        q_mvar (float): Initial reactive power exchange in Mvar.
        vm_pu (float): Voltage magnitude setpoint in per-unit. Defaults to 1.0.
        va_degree (float): Voltage angle setpoint in degrees. Defaults to 0.0.
        max_import_mw (float | None): Maximum power that can be drawn from the
            external grid (positive direction).  ``None`` = unlimited.
        max_export_mw (float | None): Maximum power that can be fed back into
            the external grid (positive magnitude; internally stored as a lower
            bound ``-max_export_mw``).  ``None`` = unlimited.
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


@model
class PowerLoad(NoVarChildModel):
    """
    Fixed-setpoint active/reactive power load.

    Follows the load convention: positive values represent *consumption*.
    Unlike :class:`PowerGenerator`, the constructor does **not** negate
    the supplied values::

        PowerLoad(p_mw=5, q_mvar=1)  →  p_mw=+5, q_mvar=+1

    Args:
        p_mw (float): Active power demand in MW (positive = consumption).
        q_mvar (float): Reactive power demand in Mvar (positive = consumption).
    """

    def __init__(self, p_mw, q_mvar, **kwargs) -> None:
        super().__init__(**kwargs)
        self.p_mw = p_mw
        self.q_mvar = q_mvar


@model
class Source(NoVarChildModel):
    """
    Fixed-setpoint mass-flow source (injection) for gas or water networks.

    Follows the load convention: internally stores a *negative* mass-flow so
    that the junction balance treats this component as an injection.  The
    constructor accepts positive magnitudes and negates them::

        Source(mass_flow=2.0)  →  self.mass_flow = -2.0

    Args:
        mass_flow (float): Mass flow rate in kg/s to inject (positive = injection).
    """

    def __init__(self, mass_flow, **kwargs) -> None:
        # isinstance guard: same reasoning as PowerGenerator — internal callers
        # may pass solver variables; sign validation only applies to user code.
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
    External hydraulic grid (slack source) — the pressure/temperature reference for a gas or water island.

    ``ExtHydrGrid`` pins the junction pressure and temperature to fixed
    setpoints (via :meth:`overwrite`) and exposes ``mass_flow`` as a free
    :class:`Var` decision variable that absorbs the island's flow imbalance.

    Follows the load convention: the default ``mass_flow=-1`` represents
    *injection* (negative = generation/source).  The solver determines the
    actual mass flow.

    Args:
        mass_flow (float): Initial mass-flow guess in kg/s. Negative = injection
            (source), positive = consumption (sink). Defaults to -1.
        pressure_pu (float): Junction pressure setpoint in per-unit. Defaults to 1.0.
        t_k (float): Supply temperature setpoint in Kelvin. Defaults to 356 K.
        max_import_kgs (float | None): Maximum mass flow the external grid can
            inject into the network (kg/s, positive magnitude; stored as lower
            bound ``-max_import_kgs``).  ``None`` = unlimited.
        max_export_kgs (float | None): Maximum mass flow the external grid can
            absorb from the network (kg/s; stored as upper bound).
            ``None`` = unlimited.
    """

    def __init__(
        self,
        mass_flow=-1,
        pressure_pu=1,
        t_k=356,
        max_import_kgs=None,
        max_export_kgs=None,
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

    def overwrite(self, node_model, grid):
        """Pin the junction pressure and temperature to the configured setpoints."""
        node_model.pressure_pu = Const(self.pressure_pu)
        node_model.pressure_squared_pu = Const(self.pressure_pu**2)
        node_model.t_pu = Const(self.t_k / grid.t_ref)
        node_model.t_k = Const(self.t_k)


@model
class ConsumeHydrGrid(NoVarChildModel):
    """
    Hydraulic demand point (consumption) for gas or water networks.

    Represents a fixed-pressure offtake point (e.g. a building substation or
    a gas consumer).  Pins the junction pressure to a setpoint and applies a
    fixed mass-flow consumption.

    Follows the load convention: internally stores a *negative* mass-flow so
    that the junction balance treats it as withdrawal.  The constructor
    accepts positive magnitudes and negates them::

        ConsumeHydrGrid(mass_flow=0.5)  →  self.mass_flow = -0.5

    Args:
        mass_flow (float): Mass flow rate in kg/s to consume (positive = consumption).
            Defaults to 0.1.
        pressure_pu (float): Junction pressure setpoint in per-unit. Defaults to 1.0.
        t_k (float): Return temperature in Kelvin. Defaults to 293 K.
    """

    def __init__(self, mass_flow=0.1, pressure_pu=1, t_k=293, **kwargs) -> None:
        super().__init__(**kwargs)
        self.mass_flow = Var(
            mass_flow,
            name="consume_ext_grid_mass_flow",
        )
        self.pressure_pu = pressure_pu
        self.t_k = t_k

    def overwrite(self, node_model, grid):
        """Pin the junction pressure to the configured setpoint."""
        node_model.pressure_pu = Const(self.pressure_pu)
        node_model.pressure_squared_pu = Const(self.pressure_pu**2)


@model
class HeatGenerator(NoVarChildModel):
    """Node-based heat injection (``H_G,i``) for district-heating formulations.

    Supplies heat at a junction without routing it through a heat-exchanger
    branch.  Integrated into both the default mass-weighted nodal heat balance
    (:meth:`~monee.model.node.Junction.calc_signed_heat_flow`) and the
    McCormick-DHS nodal balance
    (:class:`~monee.model.formulation.mccormick.water.MccDHSNodeFormulation`).
    Follows monee's load convention: the user-visible magnitude is positive,
    but the internal ``q_w_heat`` is negated so the node balance treats this
    child as an injection.

    Args:
        q_w (float): Heat output in Watts (positive = generation).
    """

    def __init__(self, q_w, **kwargs) -> None:
        if isinstance(q_w, (int, float)) and q_w < 0:
            raise ValueError(
                f"HeatGenerator expects a positive heat-generation magnitude; "
                f"got q_w={q_w}.  Pass the absolute value — the sign is "
                f"handled internally (load convention)."
            )
        super().__init__(**kwargs)
        self.q_w_heat = -q_w


@model
class HeatLoad(NoVarChildModel):
    """Node-based heat withdrawal (``H_L,i``) for district-heating formulations.

    Draws heat at a junction without routing it through a heat-exchanger
    branch.  Integrated into both the default mass-weighted nodal heat balance
    (:meth:`~monee.model.node.Junction.calc_signed_heat_flow`) and the
    McCormick-DHS nodal balance
    (:class:`~monee.model.formulation.mccormick.water.MccDHSNodeFormulation`).
    Load convention: positive ``q_w_heat`` represents consumption.

    Args:
        q_w (float): Heat demand in Watts (positive = consumption).
    """

    def __init__(self, q_w, **kwargs) -> None:
        super().__init__(**kwargs)
        self.q_w_heat = q_w


@model
class Sink(NoVarChildModel):
    """
    Fixed-setpoint mass-flow sink (withdrawal) for gas or water networks.

    Follows the load convention: positive values represent *consumption*.
    Unlike :class:`Source`, the constructor does **not** negate the supplied
    value::

        Sink(mass_flow=2.0)  →  self.mass_flow = +2.0

    Args:
        mass_flow (float): Mass flow rate in kg/s to withdraw (positive = consumption).
    """

    def __init__(self, mass_flow, **kwargs) -> None:
        super().__init__(**kwargs)
        self.mass_flow = mass_flow

import math

from .core import ChildModel, Const, Var, model
from .grid import PowerGrid


class GridFormingMixin:
    """
    Marker: this child can serve as the slack/reference for an islanded sub-network.

    Carriers must implement ``overwrite()`` to pin their reference variable.
    Islanding keeps components containing a ``GridFormingMixin`` child in the solve.
    """


class NoVarChildModel(ChildModel):
    """:class:`ChildModel` with only scalar parameters and no equations of its own."""

    def equations(self, grid, node, **kwargs):
        return []


@model
class PowerGenerator(NoVarChildModel):
    """Fixed-setpoint active/reactive generator. Constructor takes positive magnitudes; sign is internal."""

    def __init__(self, p_mw, q_mvar, **kwargs) -> None:
        # Compound models pass solver Vars for p_mw - only validate plain numerics.
        if isinstance(p_mw, (int, float)) and p_mw < 0:
            raise ValueError(
                f"PowerGenerator expects a positive generation magnitude; "
                f"got p_mw={p_mw}.  Pass the absolute value - the sign is "
                f"handled internally (load convention)."
            )
        super().__init__(**kwargs)
        self.p_mw = -p_mw
        self.q_mvar = -q_mvar


@model
class VoltageControlledGenerator(ChildModel):
    """Voltage-controlled (PV-bus) generator: injects a fixed active power while
    holding the bus voltage magnitude, leaving reactive power free to maintain it.

    This is the MATPOWER PV bus (``BUS_TYPE == 2``): ``P`` is the generator
    dispatch (fixed), ``|V|`` is pinned to the ``vm_pu`` setpoint, ``Q`` is a
    free :class:`~monee.model.core.Var` that the reactive power balance solves,
    and the bus angle stays free (the slack remains the angle reference).

    Contrast with :class:`PowerGenerator` (fixed P *and* Q, i.e. a PQ bus) and
    :class:`ExtPowerGrid` (the slack, which additionally pins the angle). Unlike
    the islanding ``GridFormingGenerator`` it does not float ``P`` to absorb
    imbalance, and it is not grid-forming - a PV bus needs a slack elsewhere.

    The constructor takes a positive generation magnitude ``p_mw``; the sign is
    handled internally (load convention, generation = negative). ``q_mvar`` only
    seeds the free reactive Var.
    """

    def __init__(self, p_mw, vm_pu=1.0, q_mvar=0.0, **kwargs) -> None:
        if isinstance(p_mw, (int, float)) and p_mw < 0:
            raise ValueError(
                f"VoltageControlledGenerator expects a positive generation "
                f"magnitude; got p_mw={p_mw}.  Pass the absolute value - the "
                f"sign is handled internally (load convention)."
            )
        super().__init__(**kwargs)
        self.p_mw = -p_mw
        self.q_mvar = Var(-q_mvar, name="gen_pv_q_mvar")
        self.vm_pu = vm_pu

    def overwrite(self, node_model, grid):
        """Pin the bus voltage magnitude to the setpoint; the angle stays free."""
        node_model.vm_pu = Const(self.vm_pu)
        node_model.vm_pu_squared = Const(self.vm_pu * self.vm_pu)

    def equations(self, grid, node_model, **kwargs):
        return []


@model
class ExtPowerGrid(NoVarChildModel, GridFormingMixin):
    """
    External slack-bus connection. Pins vm_pu and va_degree, leaves p_mw/q_mvar
    as free Vars absorbing the island's imbalance. Load convention: positive
    p_mw = import.

    ``regulate_vm`` controls the voltage magnitude. When ``True`` (the default,
    power-flow semantics) the bus |V| is held at ``vm_pu``. When ``False`` it is
    left as the bus's bounded decision variable and only the reference *angle* is
    pinned - the optimal-power-flow convention (MATPOWER/pandapower ``runopp``
    optimise the slack voltage within [VMIN, VMAX]).
    """

    def __init__(
        self,
        p_mw,
        q_mvar,
        vm_pu=1,
        va_degree=0,
        max_import_mw=None,
        max_export_mw=None,
        regulate_vm=True,
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
        self.regulate_vm = regulate_vm

    def overwrite(self, node_model, grid):
        """Pin the bus angle (always) and, when this slack regulates voltage, the
        bus voltage magnitude too. With ``regulate_vm=False`` the magnitude stays
        the bus's bounded Var so an OPF optimises it within [VMIN, VMAX]."""
        if self.regulate_vm:
            node_model.vm_pu = Const(self.vm_pu)
            node_model.vm_pu_squared = Const(self.vm_pu * self.vm_pu)
        node_model.va_degree = Const(self.va_degree)

        if (
            isinstance(grid, PowerGrid)
            and grid.sn_mva
            and not math.isclose(grid.sn_mva, 1.0)
        ):
            if isinstance(self.p_mw, Var):
                self.p_mw.scale = grid.sn_mva
            if isinstance(self.q_mvar, Var):
                self.q_mvar.scale = grid.sn_mva

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
class PowerShunt(ChildModel):
    r"""Fixed shunt element: a constant admittance :math:`y = g + jb` tied to a
    bus, modelling capacitor banks, reactors and line-charging lumped at a node.

    Unlike a :class:`PowerLoad` (constant power) the draw follows the bus voltage,
    since a fixed admittance carries :math:`S = y^* \cdot |V|^2`:

    .. math::

        p_{mw} = g_{s,mw} \cdot v^2 \qquad q_{mvar} = -b_{s,mvar} \cdot v^2

    with ``v`` the per-unit voltage magnitude. ``gs_mw`` / ``bs_mvar`` are the
    real / reactive power the element would draw at ``v = 1.0`` p.u. and map
    directly to the MATPOWER bus ``GS`` (MW demanded) and ``BS`` (MVAr injected)
    columns.

    Sign (load convention, positive ``p_mw`` / ``q_mvar`` = consumption):

    * ``gs_mw > 0`` is a resistive shunt dissipating real power; it is rarely
      nonzero in practice.
    * ``bs_mvar > 0`` is a capacitor that *injects* reactive power (the minus
      sign above makes ``q_mvar`` negative), raising the local voltage.
    * ``bs_mvar < 0`` is a reactor that absorbs reactive power, lowering it.

    Because the draw scales with ``v^2``, the reactive support a capacitor gives
    falls off exactly when voltage sags - the well-known weakness of fixed shunt
    compensation, and the reason results differ from a constant-power source.

    ``p_mw`` / ``q_mvar`` are decision Vars pinned to the equations above; the
    voltage coupling itself is emitted by the electricity
    :class:`~monee.model.formulation.core.ChildFormulation`, which binds ``v`` to
    ``vm_pu`` (polar AC NLP) or ``v^2`` to ``vm_pu_squared`` (branch-flow MISOCP)
    to keep each formulation in its native variable - the same split branches use.
    """

    def __init__(self, gs_mw, bs_mvar, **kwargs) -> None:
        super().__init__(**kwargs)
        self.gs_mw = gs_mw
        self.bs_mvar = bs_mvar
        self.p_mw = Var(0, name="shunt_p_mw")
        self.q_mvar = Var(0, name="shunt_q_mvar")

    def equations(self, grid, node_model, **kwargs):
        return []


@model
class Source(NoVarChildModel):
    """Fixed-setpoint mass-flow source. Constructor takes positive magnitude; sign is internal.

    ``t_k`` (optional) is the temperature of the injected stream. Without it the
    injection is credited at the junction's own (mixed) temperature.
    """

    def __init__(self, mass_flow_kgs, t_k=None, **kwargs) -> None:
        # Internal callers may pass solver Vars - only validate plain numerics.
        if isinstance(mass_flow_kgs, (int, float)) and mass_flow_kgs < 0:
            raise ValueError(
                f"Source expects a positive injection magnitude; "
                f"got mass_flow_kgs={mass_flow_kgs}.  Pass the absolute value - the "
                f"sign is handled internally (load convention)."
            )
        super().__init__(**kwargs)

        self.mass_flow_kgs = -mass_flow_kgs
        self.injection_t_k = t_k


@model
class ExtHydrGrid(NoVarChildModel, GridFormingMixin):
    """
    External hydraulic slack source. Pins pressure (and optionally temperature),
    leaves mass_flow_kgs as a free Var. Load convention: negative mass_flow_kgs = injection.
    """

    def __init__(
        self,
        mass_flow_kgs=-1,
        pressure_pu=1,
        t_k=356,
        max_import_kgs=None,
        max_export_kgs=None,
        pin_temperature=True,
        free_pressure_bounds=None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.mass_flow_kgs = Var(
            mass_flow_kgs,
            min=None if max_import_kgs is None else -max_import_kgs,
            max=max_export_kgs,
            name="ext_grid_mass_flow",
        )
        self.pressure_pu = pressure_pu
        self.t_k = t_k
        self.pin_temperature = pin_temperature
        self.free_pressure_bounds = free_pressure_bounds

    def overwrite(self, node_model, grid):

        if self.free_pressure_bounds is not None:
            lo, hi = self.free_pressure_bounds

            psq = getattr(node_model, "pressure_squared_pu", None)
            p = getattr(node_model, "pressure_pu", None)
            if type(psq) is Var:
                psq.min, psq.max = lo * lo, hi * hi
            if type(p) is Var:
                p.min, p.max = lo, hi
        else:
            node_model.pressure_pu = Const(self.pressure_pu)
            node_model.pressure_squared_pu = Const(self.pressure_pu**2)
        if self.pin_temperature:
            node_model.t_pu = Const(self.t_k / grid.t_ref_k)
            node_model.t_k = Const(self.t_k)


@model
class ConsumeHydrGrid(NoVarChildModel):
    """Hydraulic demand point with a free ``mass_flow_kgs`` Var absorbing the
    island's imbalance.

    Unlike :class:`ExtHydrGrid` it pins nothing: the node's pressure and
    temperature stay free (the solver pins one gauge per island without a
    grid-forming source, see ``pin_floating_hydraulic_gauges``; pinning here
    too would over-determine such islands). ``pressure_pu`` / ``t_k`` are
    stored as descriptive setpoints only. ``overwrite`` bounds the free mass
    flow to the grid's ``max_mass_flow_kgs`` where no explicit bound is set.
    """

    def __init__(self, mass_flow_kgs=0.1, pressure_pu=1, t_k=293, **kwargs) -> None:
        super().__init__(**kwargs)
        self.mass_flow_kgs = Var(
            mass_flow_kgs,
            name="consume_ext_grid_mass_flow",
        )
        self.pressure_pu = pressure_pu
        self.t_k = t_k

    def overwrite(self, node_model, grid):
        max_flow = getattr(grid, "max_mass_flow_kgs", None)
        if max_flow is None or not isinstance(self.mass_flow_kgs, Var):
            return
        if self.mass_flow_kgs.min is None:
            self.mass_flow_kgs.min = -max_flow
        if self.mass_flow_kgs.max is None:
            self.mass_flow_kgs.max = max_flow


@model
class HeatGenerator(NoVarChildModel):
    """Node-based heat injection (``H_G,i``). Takes positive magnitude; sign is internal."""

    def __init__(self, q_mw, **kwargs) -> None:
        if isinstance(q_mw, (int, float)) and q_mw < 0:
            raise ValueError(
                f"HeatGenerator expects a positive heat-generation magnitude; "
                f"got q_mw={q_mw}.  Pass the absolute value - the sign is "
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

    def __init__(self, mass_flow_kgs, **kwargs) -> None:
        super().__init__(**kwargs)
        self.mass_flow_kgs = mass_flow_kgs

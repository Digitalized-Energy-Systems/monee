from .branch import HeatExchanger
from .child import NoVarChildModel, PowerGenerator, PowerLoad, Sink
from .core import (
    MultiGridBranchModel,
    MultiGridCompoundModel,
    MultiGridNodeModel,
    Node,
    PostProcess,
    Var,
    is_plain_number,
    model,
)
from .grid import KGPS_KWHPERKG_TO_MW, GasGrid, PowerGrid, WaterGrid
from .network import Network
from .node import Bus, Junction
from .phys.core.hydraulics import junction_mass_flow_balance
from .phys.nonlinear.ac import power_balance_equation
from .phys.nonlinear.hf import SPECIFIC_HEAT_CAP_WATER

# Nominal supply-to-return temperature drop assumed when deriving mass-flow
# initial values from a heat setpoint (matches the convention used by the
# network builders). Only affects solver starting points, never the physics.
_NOMINAL_DT_K = 25.0


def _heat_flow_init_kgs(heat_mw) -> float:
    """Expected heat-side mass flow for a heat setpoint at the nominal Î”T.

    APOPT stalls on small-setpoint compounds when every flow Var starts at the
    generic 1.0 placeholder, orders of magnitude from the optimum."""
    if not isinstance(heat_mw, (int, float)) or isinstance(heat_mw, bool):
        return 1.0
    return abs(heat_mw) * 1e6 / (SPECIFIC_HEAT_CAP_WATER * _NOMINAL_DT_K)


def _num_or(value, default: float) -> float:
    return value if is_plain_number(value) else default


def _carries(grids, grid_cls) -> bool:
    return type(grids) is grid_cls or (type(grids) is dict and grid_cls in grids)


@model
class GenericTransferBranch(MultiGridBranchModel):
    def __init__(
        self, flow_init_kgs: float = 1.0, p_init_mw: float = 1.0, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        # Initial values only - compounds pass setpoint-derived magnitudes so
        # solvers don't start orders of magnitude from the optimum.
        self._mass_flow_pos = Var(flow_init_kgs, min=0, name="mass_flow_pos_kgs")
        self._mass_flow_neg = Var(flow_init_kgs, min=0, name="mass_flow_neg_kgs")
        self.on_off = 1
        self._p_mw = Var(p_init_mw, name="transfer_p_mw")
        self._q_mvar = Var(1, name="transfer_q_mvar")
        self._t_from_pu = Var(1, min=0, max=2, name="t_from_pu")
        self._t_to_pu = Var(1, min=0, max=2, name="t_to_pu")

    def is_cp(self):
        return False

    def init(self, grids):
        if _carries(grids, WaterGrid):
            self.mass_flow_pos_kgs = self._mass_flow_pos
            self.mass_flow_neg_kgs = self._mass_flow_neg
            self.t_from_pu = self._t_from_pu
            self.t_to_pu = self._t_to_pu
        if _carries(grids, GasGrid):
            self.mass_flow_pos_kgs = self._mass_flow_pos
            self.mass_flow_neg_kgs = self._mass_flow_neg
            self.gas_mass_flow = self.mass_flow_pos_kgs
        if _carries(grids, PowerGrid):
            self.p_to_mw = self._p_mw
            self.p_from_mw = self._p_mw
            self.q_to_mvar = self._q_mvar
            self.q_from_mvar = self._q_mvar

    def equations(self, grids, from_node_model, to_node_model, **kwargs):
        eqs = []
        if _carries(grids, WaterGrid):
            eqs += [self.t_from_pu == self.t_to_pu]
            eqs += [self.t_from_pu == from_node_model.t_pu]
            eqs += [to_node_model.t_pu == self.t_to_pu]
            eqs += [to_node_model.t_pu == from_node_model.t_pu]
            eqs += [from_node_model.pressure_pu == to_node_model.pressure_pu]
        return eqs


def _is_zeroed(regulation):
    """True when ``ignore_compound``'s ``set_active(False)`` pinned the
    regulation to the plain number 0; a controllable-CP Var never counts."""
    return is_plain_number(regulation) and regulation == 0


def _heat_branches(branch_models):
    return [
        branch
        for branch in branch_models
        if "t_from_pu" in branch.vars or isinstance(branch, SubHE)
    ]


def _gas_branches(branch_models):
    return [branch for branch in branch_models if "gas_mass_flow" in branch.vars]


def _power_branches(branch_models, power_var):
    return [branch for branch in branch_models if power_var in branch.vars]


class _RegulatedCompound:
    """set_active() for compounds steered by a control node's ``regulation``:
    save/zero on deactivate, restore on activate."""

    def set_active(self, activation_flag):
        if activation_flag:

            if not self._active:
                self._active = True
                self._restore_controls()
        else:
            # A repeated deactivate must not clobber the saved regulation
            # with the already-zeroed value.
            if self._active:
                self._old_regulation = self._control_node.regulation
                self._active = False
            self._zero_controls()

    def _restore_controls(self):
        # Don't overwrite when controllable_cps has promoted the attr to a
        # Var - restore only the case where set_active(False) zeroed it.
        if is_plain_number(self._control_node.regulation):
            self._control_node.regulation = self._old_regulation

    def _zero_controls(self):
        self._control_node.regulation = 0


class _GasRegulatedCompound(_RegulatedCompound):
    """:class:`_RegulatedCompound` that additionally zeroes/restores the
    control node's gas dispatch."""

    def _restore_controls(self):
        if is_plain_number(self._control_node.gas_mass_flow_kgs):
            self._control_node.gas_mass_flow_kgs = self.mass_flow_setpoint_kgs
        super()._restore_controls()

    def _zero_controls(self):
        self._control_node.gas_mass_flow_kgs = 0
        super()._zero_controls()


@model
class GasToHeatControlNode(MultiGridNodeModel, Junction):
    def __init__(
        self, gas_mass_flow_kgs, efficiency_heat, hhv, regulation=1, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.efficiency_heat = efficiency_heat
        self._hhv = hhv
        self.regulation = regulation

        self.gas_mass_flow_kgs = gas_mass_flow_kgs
        # Initialize at the setpoint solution: APOPT stalls on tiny-setpoint
        # instances when started from a generic placeholder far from optimum.
        heat_mw_init = (
            -efficiency_heat * gas_mass_flow_kgs * KGPS_KWHPERKG_TO_MW * hhv
            if isinstance(gas_mass_flow_kgs, (int, float))
            else -1e-3
        )
        self.heat_mw = Var(min(heat_mw_init, -1e-6), max=0, name="g2h_heat_mw")

        self.t_k = Var(350, min=200, max=800, name="t_k")
        self.t_pu = Var(1, min=0, max=2, name="t_pu")
        # pressure_pa is post-solve only (= pressure_pu \cdot pressure_ref_pa); compute it
        # outside the solver. Real closure attached in equations() (needs grid).
        self.pressure_pa = PostProcess(lambda v: float("nan"))
        self.pressure_pu = Var(1, min=0, max=2, name="pressure_pu")

    def equations(self, grid, from_branch_models, to_branch_models, childs, **kwargs):
        heat_to_branches = _heat_branches(to_branch_models)
        heat_from_branches = _heat_branches(from_branch_models)
        gas_to_branches = _gas_branches(to_branch_models)

        sub_he = next((b for b in heat_from_branches if isinstance(b, SubHE)), None)
        if sub_he is None:
            if not _is_zeroed(self.regulation):
                raise ValueError(
                    "GasToHeatControlNode requires a SubHE heat-exchanger branch on the "
                    "heat-return side. Build via GasToHeat.create(...) so the SubHE is wired up."
                )
            # Heat side ignored: ignore_compound zeroed the regulation and the
            # solver filtered the SubHE with the ignored heat junctions.
            # Degrade to a gas-only node with zero heat output.
            gas_eqs = self.calc_signed_mass_flow(
                [], gas_to_branches, [Sink(self.gas_mass_flow_kgs * self.regulation)]
            )
            self.pressure_pa = PostProcess(
                lambda v, ref=grid[0].pressure_ref_pa: v.pressure_pu * ref
            )
            return [
                junction_mass_flow_balance(gas_eqs),
                self.heat_mw == 0,
                self.t_pu == 1,
                self.t_pu == self.t_k / grid[0].t_ref_k,
            ]

        gas_eqs = self.calc_signed_mass_flow(
            [], gas_to_branches, [Sink(self.gas_mass_flow_kgs * self.regulation)]
        )
        heat_eqs = self.calc_signed_mass_flow(heat_from_branches, heat_to_branches, [])
        heat_energy_eqs = self.calc_signed_heat_flow(
            heat_from_branches, heat_to_branches, [], None
        )
        self.pressure_pa = PostProcess(
            lambda v, ref=grid[0].pressure_ref_pa: v.pressure_pu * ref
        )
        return [
            junction_mass_flow_balance(heat_eqs),
            junction_mass_flow_balance(heat_energy_eqs),
            junction_mass_flow_balance(gas_eqs),
            sub_he.q_mw
            == -self.efficiency_heat
            * self.gas_mass_flow_kgs
            * self.regulation
            * (KGPS_KWHPERKG_TO_MW * self._hhv),
            self.heat_mw == sub_he.q_mw,
            self.t_pu == self.t_k / grid[0].t_ref_k,
        ]


@model
class PowerToHeatControlNode(MultiGridNodeModel, Junction, Bus):
    def __init__(
        self, load_p_mw, load_q_mvar, efficiency, regulation=1, **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.load_q_mvar = load_q_mvar
        self.efficiency = efficiency
        self.regulation = regulation

        self.el_mw = load_p_mw
        # Initialize at the setpoint solution (see GasToHeatControlNode).
        heat_mw_init = (
            -efficiency * load_p_mw if isinstance(load_p_mw, (int, float)) else -1e-3
        )
        self.heat_mw = Var(min(heat_mw_init, -1e-6), max=0, name="p2h_heat_mw")

        self.t_k = Var(350, min=200, max=800, name="t_k")
        self.t_pu = Var(1, min=0, max=2, name="t_pu")
        self.pressure_squared_pu = Var(1, min=0.5, max=3, name="pressure_squared_pu")
        self.pressure_pu = Var(1, min=0.5, max=3, name="pressure_pu")

    def equations(self, grid, from_branch_models, to_branch_models, childs, **kwargs):
        heat_to_branches = _heat_branches(to_branch_models)
        heat_from_branches = _heat_branches(from_branch_models)
        power_to_branches = _power_branches(to_branch_models, "p_from_mw")

        sub_he = next((b for b in heat_to_branches if isinstance(b, SubHE)), None)
        if sub_he is None:
            if not _is_zeroed(self.regulation):
                raise ValueError(
                    "PowerToHeatControlNode requires a SubHE heat-exchanger branch on the "
                    "heat-supply side. Build via PowerToHeat.create(...) so the SubHE is wired up."
                )
            # Heat side ignored (see GasToHeatControlNode): degrade to a
            # power-only node with zero load and zero heat output.
            power_eqs = self.calc_signed_power_values(
                [],
                power_to_branches,
                [PowerLoad(self.el_mw, self.load_q_mvar, regulation=self.regulation)],
            )
            return [
                self.heat_mw == 0,
                sum(power_eqs[0]) == 0,
                sum(power_eqs[1]) == 0,
                self.t_pu == 1,
                self.t_k == self.t_pu * grid[1].t_ref_k,
            ]

        power_eqs = self.calc_signed_power_values(
            [],
            power_to_branches,
            [PowerLoad(self.el_mw, self.load_q_mvar, regulation=self.regulation)],
        )
        heat_eqs = self.calc_signed_mass_flow(heat_from_branches, heat_to_branches, [])
        heat_energy_eqs = self.calc_signed_heat_flow(
            heat_from_branches, heat_to_branches, [], None
        )
        return [
            junction_mass_flow_balance(heat_eqs),
            junction_mass_flow_balance(heat_energy_eqs),
            sub_he.q_mw == -self.efficiency * self.el_mw * self.regulation,
            self.heat_mw == sub_he.q_mw,
            sum(power_eqs[0]) == 0,
            sum(power_eqs[1]) == 0,
            self.t_k == self.t_pu * grid[1].t_ref_k,
        ]


class SubHE(HeatExchanger):
    """Subordinate heat exchanger used inside compound models (CHP, G2H, P2H)."""


@model
class CHPControlNode(MultiGridNodeModel, Junction, Bus):
    """Control node for a CHP unit; couples power, heat, and gas domains."""

    def __init__(
        self,
        mass_flow_capacity_kgs,
        efficiency_power,
        efficiency_heat,
        hhv,
        q_mvar=0,
        regulation=1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.efficiency_heat = efficiency_heat
        self.efficiency_power = efficiency_power
        self.gen_q_mvar = q_mvar
        self._hhv = hhv
        self.regulation = regulation

        # Initialize at the setpoint solution (see GasToHeatControlNode).
        if isinstance(mass_flow_capacity_kgs, (int, float)):
            el_mw_init = (
                -efficiency_power * mass_flow_capacity_kgs * KGPS_KWHPERKG_TO_MW * hhv
            )
            heat_mw_init = (
                -efficiency_heat * mass_flow_capacity_kgs * KGPS_KWHPERKG_TO_MW * hhv
            )
        else:
            el_mw_init, heat_mw_init = -1, -1e-3
        self.el_mw = Var(min(el_mw_init, -1e-6), max=0, name="chp_el_mw")
        self.gas_mass_flow_kgs = mass_flow_capacity_kgs
        self.heat_mw = Var(min(heat_mw_init, -1e-6), max=0, name="chp_heat_mw")

        self.t_k = Var(350, min=200, max=800, name="t_k")
        self.t_pu = Var(1, min=0, max=2, name="t_pu")
        # pressure_pa is post-solve only (see GasToHeatControlNode).
        self.pressure_pa = PostProcess(lambda v: float("nan"))
        self.pressure_pu = Var(1, min=0, max=2, name="pressure_pu")

    def equations(self, grid, from_branch_models, to_branch_models, childs, **kwargs):
        heat_to_branches = _heat_branches(to_branch_models)
        heat_from_branches = _heat_branches(from_branch_models)
        gas_to_branches = _gas_branches(to_branch_models)
        power_from_branches = _power_branches(from_branch_models, "p_to_mw")

        sub_he = next((b for b in heat_from_branches if isinstance(b, SubHE)), None)
        if sub_he is None:
            if not _is_zeroed(self.regulation):
                raise ValueError(
                    "CHPControlNode requires a SubHE heat-exchanger branch on the "
                    "heat-return side. Build via CHP.create(...) so the SubHE is wired up."
                )
            # Heat side ignored (see GasToHeatControlNode): degrade to a
            # gas+power node with zero generation and zero heat output.
            power_eqs = self.calc_signed_power_values(
                power_from_branches, [], [PowerGenerator(self.el_mw, self.gen_q_mvar)]
            )
            gas_eqs = self.calc_signed_mass_flow(
                [], gas_to_branches, [Sink(self.gas_mass_flow_kgs * self.regulation)]
            )
            self.pressure_pa = PostProcess(
                lambda v, ref=grid[1].pressure_ref_pa: v.pressure_pu * ref
            )
            return [
                junction_mass_flow_balance(gas_eqs),
                power_balance_equation(power_eqs[0]),
                power_balance_equation(power_eqs[1]),
                self.el_mw == 0,
                self.heat_mw == 0,
                self.t_pu == 1,
                self.t_k == self.t_pu * grid[1].t_ref_k,
            ]

        power_eqs = self.calc_signed_power_values(
            power_from_branches, [], [PowerGenerator(self.el_mw, self.gen_q_mvar)]
        )
        gas_eqs = self.calc_signed_mass_flow(
            [], gas_to_branches, [Sink(self.gas_mass_flow_kgs * self.regulation)]
        )
        heat_eqs = self.calc_signed_mass_flow(heat_from_branches, heat_to_branches, [])
        heat_energy_eqs = self.calc_signed_heat_flow(
            heat_from_branches, heat_to_branches, [], None
        )
        self.pressure_pa = PostProcess(
            lambda v, ref=grid[1].pressure_ref_pa: v.pressure_pu * ref
        )
        return [
            junction_mass_flow_balance(heat_eqs),
            junction_mass_flow_balance(heat_energy_eqs),
            junction_mass_flow_balance(gas_eqs),
            power_balance_equation(power_eqs[0]),
            power_balance_equation(power_eqs[1]),
            sub_he.q_mw
            == -self.efficiency_heat
            * self.gas_mass_flow_kgs
            * self.regulation
            * (KGPS_KWHPERKG_TO_MW * self._hhv),
            self.el_mw
            == -self.efficiency_power
            * self.gas_mass_flow_kgs
            * self.regulation
            * (KGPS_KWHPERKG_TO_MW * self._hhv),
            self.heat_mw == sub_he.q_mw,
            self.t_k == self.t_pu * grid[1].t_ref_k,
        ]


@model
class CHP(_GasRegulatedCompound, MultiGridCompoundModel):
    def __init__(
        self,
        diameter_m: float,
        efficiency_power: float,
        efficiency_heat: float,
        mass_flow_setpoint_kgs: float,
        q_mvar_setpoint: float = 0,
        temperature_ext_k: float = 293,
        regulation=1,
    ) -> None:
        self.diameter_m = diameter_m
        self.temperature_ext_k = temperature_ext_k
        self.regulation = regulation
        self.efficiency_power = efficiency_power
        self.efficiency_heat = efficiency_heat
        self.mass_flow_setpoint_kgs = mass_flow_setpoint_kgs
        self.mass_flow_kgs = mass_flow_setpoint_kgs
        self.q_mvar = q_mvar_setpoint
        self._old_regulation = self.regulation
        self._active = True

    def create(
        self,
        network: Network,
        gas_node: Node,  # NOSONAR
        heat_node: Node,  # NOSONAR
        heat_return_node: Node,  # NOSONAR
        power_node: Node,  # NOSONAR
    ):
        self._gas_grid = gas_node.grid
        hhv = gas_node.grid.higher_heating_value_kwh_per_kg
        self._control_node = CHPControlNode(
            self.mass_flow_kgs,
            self.efficiency_power,
            self.efficiency_heat,
            hhv,
            regulation=self.regulation,
        )
        node_id_control = network.node(
            self._control_node,
            grid=[power_node.grid, heat_node.grid, gas_node.grid],
            position=power_node.position,
        )
        heat_mw = (
            self.efficiency_heat
            * _num_or(self.mass_flow_kgs, 1.0)
            * KGPS_KWHPERKG_TO_MW
            * hhv
        )
        el_mw = (
            self.efficiency_power
            * _num_or(self.mass_flow_kgs, 1.0)
            * KGPS_KWHPERKG_TO_MW
            * hhv
        )
        network.branch(
            GenericTransferBranch(flow_init_kgs=_num_or(self.mass_flow_kgs, 1.0)),
            gas_node.id,
            node_id_control,
        )
        network.branch(
            GenericTransferBranch(flow_init_kgs=_heat_flow_init_kgs(heat_mw)),
            heat_node.id,
            node_id_control,
        )
        network.branch(
            SubHE(Var(-heat_mw)),
            node_id_control,
            heat_return_node.id,
            grid=heat_return_node.grid,
        )
        network.branch(
            GenericTransferBranch(p_init_mw=el_mw), node_id_control, power_node.id
        )


@model
class GasToHeat(_RegulatedCompound, MultiGridCompoundModel):
    def __init__(
        self, heat_energy_mw, diameter_m, temperature_ext_k, efficiency, regulation=1
    ) -> None:
        self.diameter_m = diameter_m
        self.temperature_ext_k = temperature_ext_k
        self.efficiency = efficiency
        self.heat_energy_mw = -heat_energy_mw
        self.regulation = regulation
        self._old_regulation = self.regulation
        self._active = True

    def create(
        self,
        network: Network,
        gas_node: Node,
        heat_node: Node,
        heat_return_node: Node,  # NOSONAR
    ):
        self._gas_grid = gas_node.grid
        hhv = gas_node.grid.higher_heating_value_kwh_per_kg
        # m = |q| / (eff \cdot KGPS_KWHPERKG_TO_MW \cdot hhv)
        self.gas_mass_flow_kgs = abs(self.heat_energy_mw) / (
            self.efficiency * KGPS_KWHPERKG_TO_MW * hhv
        )
        self._control_node = GasToHeatControlNode(
            self.gas_mass_flow_kgs,
            self.efficiency,
            hhv,
            regulation=self.regulation,
        )
        node_id_control = network.node(
            self._control_node,
            grid=[heat_node.grid, gas_node.grid],
            position=gas_node.position,
        )
        network.branch(
            GenericTransferBranch(flow_init_kgs=_num_or(self.gas_mass_flow_kgs, 1.0)),
            gas_node.id,
            node_id_control,
        )
        network.branch(
            GenericTransferBranch(
                flow_init_kgs=_heat_flow_init_kgs(self.heat_energy_mw)
            ),
            heat_node.id,
            node_id_control,
        )
        network.branch(
            SubHE(Var(self.heat_energy_mw)),
            node_id_control,
            heat_return_node.id,
            grid=heat_return_node.grid,
        )


@model
class PowerToHeat(_RegulatedCompound, MultiGridCompoundModel):
    def __init__(
        self,
        heat_energy_mw,
        diameter_m,
        temperature_ext_k,
        efficiency,
        q_mvar_setpoint=0,
        regulation=1,
    ) -> None:
        self.diameter_m = diameter_m
        self.temperature_ext_k = temperature_ext_k
        self.efficiency = efficiency
        self.heat_energy_mw = heat_energy_mw
        self.load_p_mw = heat_energy_mw / efficiency
        self.load_q_mvar = q_mvar_setpoint
        self.regulation = regulation
        self._old_regulation = self.regulation
        self._active = True

    def create(
        self,
        network: Network,
        power_node: Node,  # NOSONAR
        heat_node: Node,  # NOSONAR
        heat_return_node: Node,  # NOSONAR
    ):
        self._control_node = PowerToHeatControlNode(
            self.load_p_mw,
            self.load_q_mvar,
            self.efficiency,
            regulation=self.regulation,
        )
        node_id_control = network.node(
            self._control_node,
            grid=[power_node.grid, heat_node.grid],
            position=power_node.position,
        )
        network.branch(
            GenericTransferBranch(p_init_mw=_num_or(self.load_p_mw, 1.0)),
            power_node.id,
            node_id_control,
        )
        network.branch(
            GenericTransferBranch(
                flow_init_kgs=_heat_flow_init_kgs(self.heat_energy_mw)
            ),
            node_id_control,
            heat_return_node.id,
        )
        network.branch(
            SubHE(Var(-self.heat_energy_mw)),
            heat_node.id,
            node_id_control,
            grid=heat_node.grid,
        )


@model
class GasToPower(MultiGridBranchModel):
    def __init__(
        self, efficiency, p_mw_setpoint, q_mvar_setpoint=0, regulation=1
    ) -> None:
        super().__init__()
        self.efficiency = efficiency
        self.el_mw = -p_mw_setpoint
        self.gas_mass_flow_kgs = Var(1, min=0, name="g2p_gas_kgps")

        self.on_off = 1
        self.p_to_mw = Var(-p_mw_setpoint, max=0, name="g2p_p_to_mw")
        self.q_to_mvar = -q_mvar_setpoint
        self.from_mass_flow_kgs = Var(1, min=0, name="g2p_from_mass_flow")
        self.regulation = regulation

    def loss_percent(self):
        return 1 - self.efficiency

    def equations(self, grids, from_node_model, to_node_model, **kwargs):
        return [
            self.p_to_mw == self.regulation * self.el_mw,
            -self.p_to_mw
            == self.efficiency
            * self.from_mass_flow_kgs
            * (KGPS_KWHPERKG_TO_MW * grids[GasGrid].higher_heating_value_kwh_per_kg),
            self.gas_mass_flow_kgs == self.from_mass_flow_kgs,
        ]


@model
class PowerToGas(MultiGridBranchModel):
    def __init__(
        self,
        efficiency,
        mass_flow_setpoint_kgs,
        consume_q_mvar_setpoint=0,
        regulation=1,
    ) -> None:
        super().__init__()
        self.efficiency = efficiency
        self.gas_mass_flow_kgs = -mass_flow_setpoint_kgs
        self.el_mw = Var(1.1, min=0, name="p2g_el_mw")

        self.on_off = 1
        self.p_from_mw = Var(1, min=0, name="p2g_p_from_mw")
        self.q_from_mvar = consume_q_mvar_setpoint
        self.to_mass_flow_kgs = Var(
            self.gas_mass_flow_kgs, max=0, name="p2g_to_mass_flow"
        )
        self.regulation = regulation

    def loss_percent(self):
        return 1 - self.efficiency

    def equations(self, grids, from_node_model, to_node_model, **kwargs):
        return [
            self.to_mass_flow_kgs
            == -self.efficiency
            * self.p_from_mw
            * (
                1
                / (grids[GasGrid].higher_heating_value_kwh_per_kg * KGPS_KWHPERKG_TO_MW)
            ),
            self.p_from_mw >= 0,
            self.p_from_mw == self.el_mw,
            self.gas_mass_flow_kgs * self.regulation
            == -self.efficiency
            * self.el_mw
            * (
                1
                / (grids[GasGrid].higher_heating_value_kwh_per_kg * KGPS_KWHPERKG_TO_MW)
            ),
        ]


@model
class SubHG(NoVarChildModel):
    """Subordinate node-based heat generator used inside :class:`CHPHG`.

    Like HeatGenerator but with q_mw_heat as a Var constrained by the parent
    compound's control-node equations. Two-endpoint HG variants
    (GasToHeatHG / PowerToHeatHG) don't use this - they carry q_mw_heat
    directly on the branch.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.q_mw_heat = Var(-1e-3, name="sub_hg_q_mw_heat")


def _gas_grid_of(grid):
    if isinstance(grid, list):
        return next(g for g in grid if isinstance(g, GasGrid))
    return grid


@model
class CHPHGControlNode(MultiGridNodeModel, Junction, Bus):
    """CHP control node using a node-based HeatGenerator (no HX branch).

    Like :class:`CHPControlNode` but only on power+gas; heat goes through a
    :class:`SubHG` child attached at the heat node by :class:`CHPHG`.
    """

    def __init__(
        self,
        mass_flow_capacity_kgs,
        efficiency_power,
        efficiency_heat,
        hhv,
        sub_hg,
        q_mvar=0,
        regulation=1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.efficiency_heat = efficiency_heat
        self.efficiency_power = efficiency_power
        self.gen_q_mvar = q_mvar
        self._hhv = hhv
        self._sub_hg = sub_hg
        self.regulation = regulation

        self.el_mw = Var(-1, max=0, name="chp_hg_el_mw")
        self.gas_mass_flow_kgs = mass_flow_capacity_kgs
        self.heat_mw = Var(-1e-3, max=0, name="chp_hg_heat_mw")

        self.t_k = Var(350, min=200, max=800, name="t_k")
        self.t_pu = Var(1, min=0, max=2, name="t_pu")
        # pressure_pa is post-solve only (see GasToHeatControlNode).
        self.pressure_pa = PostProcess(lambda v: float("nan"))
        self.pressure_pu = Var(1, min=0, max=2, name="pressure_pu")

    def equations(self, grid, from_branch_models, to_branch_models, childs, **kwargs):
        gas_to_branches = _gas_branches(to_branch_models)
        power_from_branches = _power_branches(from_branch_models, "p_to_mw")

        power_eqs = self.calc_signed_power_values(
            power_from_branches, [], [PowerGenerator(self.el_mw, self.gen_q_mvar)]
        )
        gas_eqs = self.calc_signed_mass_flow(
            [], gas_to_branches, [Sink(self.gas_mass_flow_kgs * self.regulation)]
        )
        gas_grid = _gas_grid_of(grid)
        self.pressure_pa = PostProcess(
            lambda v, ref=gas_grid.pressure_ref_pa: v.pressure_pu * ref
        )
        eqs = [
            junction_mass_flow_balance(gas_eqs),
            power_balance_equation(power_eqs[0]),
            power_balance_equation(power_eqs[1]),
            self.el_mw
            == -self.efficiency_power
            * self.gas_mass_flow_kgs
            * self.regulation
            * (KGPS_KWHPERKG_TO_MW * self._hhv),
        ]
        # The heat side lives on a *separate* heat node reached through the
        # captured ``self._sub_hg`` reference, not the ignore-filtered branch
        # inputs the power/gas rows use. If that node is islanded, the solver
        # NaN-injects the SubHG, leaving ``q_mw_heat`` a placeholder model
        # ``Var`` rather than a live backend variable; splicing it into a
        # constraint mixes an un-injected model Var with backend expressions and
        # the solve crashes. Mirror Gas/PowerToHeatControlNode: when the heat
        # unit is gone, drop the coupling rows and emit zero heat output.
        if isinstance(self._sub_hg.q_mw_heat, Var):
            eqs.append(self.heat_mw == 0)
        else:
            eqs.extend(
                [
                    self._sub_hg.q_mw_heat
                    == -self.efficiency_heat
                    * self.gas_mass_flow_kgs
                    * self.regulation
                    * (KGPS_KWHPERKG_TO_MW * self._hhv),
                    self.heat_mw == self._sub_hg.q_mw_heat,
                ]
            )
        return eqs


@model
class CHPHG(_GasRegulatedCompound, MultiGridCompoundModel):
    """HeatGenerator-based :class:`CHP` variant: heat is injected via a
    :class:`SubHG` child at ``heat_node`` instead of via an HX branch. No
    ``heat_return_node`` / ``diameter_m`` required."""

    def __init__(
        self,
        efficiency_power: float,
        efficiency_heat: float,
        mass_flow_setpoint_kgs: float,
        q_mvar_setpoint: float = 0,
        regulation=1,
    ) -> None:
        self.regulation = regulation
        self.efficiency_power = efficiency_power
        self.efficiency_heat = efficiency_heat
        self.mass_flow_setpoint_kgs = mass_flow_setpoint_kgs
        self.mass_flow_kgs = mass_flow_setpoint_kgs
        self.q_mvar = q_mvar_setpoint
        self._old_regulation = self.regulation
        self._active = True

    def create(
        self,
        network: Network,
        gas_node: Node,  # NOSONAR
        heat_node: Node,  # NOSONAR
        power_node: Node,  # NOSONAR
    ):
        self._gas_grid = gas_node.grid
        hhv = gas_node.grid.higher_heating_value_kwh_per_kg
        self._sub_hg = SubHG()
        self._control_node = CHPHGControlNode(
            self.mass_flow_kgs,
            self.efficiency_power,
            self.efficiency_heat,
            hhv,
            self._sub_hg,
            q_mvar=self.q_mvar,
            regulation=self.regulation,
        )
        node_id_control = network.node(
            self._control_node,
            grid=[power_node.grid, gas_node.grid],
            position=power_node.position,
        )
        network.branch(GenericTransferBranch(), gas_node.id, node_id_control)
        network.branch(GenericTransferBranch(), node_id_control, power_node.id)
        network.child(self._sub_hg, attach_to_node_id=heat_node.id)


@model
class GasToHeatHG(MultiGridBranchModel):
    """Two-endpoint Gasâ†’Heat coupling (gas withdrawal at from-end, q_mw_heat
    injection at to-end). Junction heat balance picks up ``q_mw_heat`` directly."""

    def __init__(self, heat_energy_mw, efficiency, regulation=1) -> None:
        super().__init__()
        self.efficiency = efficiency
        self.heat_energy_mw = -heat_energy_mw

        self.on_off = 1
        self.regulation = regulation
        self.q_mw_heat = Var(self.heat_energy_mw, max=0, name="g2h_hg_q_mw_heat")
        self.from_mass_flow_kgs = Var(1, min=0, name="g2h_hg_from_mass_flow")
        self.gas_mass_flow_kgs = 0  # set in init() once hhv is known

    def init(self, grids):
        hhv = grids[GasGrid].higher_heating_value_kwh_per_kg
        self.gas_mass_flow_kgs = abs(self.heat_energy_mw) / (
            self.efficiency * KGPS_KWHPERKG_TO_MW * hhv
        )

    def loss_percent(self):
        return 1 - self.efficiency

    def equations(self, grids, from_node_model, to_node_model, **kwargs):
        hhv = grids[GasGrid].higher_heating_value_kwh_per_kg
        return [
            self.q_mw_heat
            == -self.efficiency
            * self.gas_mass_flow_kgs
            * self.regulation
            * (KGPS_KWHPERKG_TO_MW * hhv),
            self.from_mass_flow_kgs == self.gas_mass_flow_kgs * self.regulation,
        ]


@model
class PowerToHeatHG(MultiGridBranchModel):
    """Two-endpoint Powerâ†’Heat coupling (p_from_mw at from-end, q_mw_heat at
    to-end). Junction heat balance picks up ``q_mw_heat`` directly."""

    def __init__(
        self, heat_energy_mw, efficiency, q_mvar_setpoint=0, regulation=1
    ) -> None:
        super().__init__()
        self.efficiency = efficiency
        self.heat_energy_mw = heat_energy_mw
        self.load_p_mw = heat_energy_mw / efficiency
        self.load_q_mvar = q_mvar_setpoint

        self.on_off = 1
        self.regulation = regulation
        self.q_mw_heat = Var(-heat_energy_mw, max=0, name="p2h_hg_q_mw_heat")
        self.p_from_mw = Var(self.load_p_mw, min=0, name="p2h_hg_p_from_mw")
        self.q_from_mvar = q_mvar_setpoint

    def loss_percent(self):
        return 1 - self.efficiency

    def equations(self, grids, from_node_model, to_node_model, **kwargs):
        return [
            self.q_mw_heat == -self.efficiency * self.load_p_mw * self.regulation,
            self.p_from_mw == self.load_p_mw * self.regulation,
        ]

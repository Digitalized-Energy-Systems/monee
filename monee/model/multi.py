from .branch import HeatExchanger
from .child import PowerGenerator, PowerLoad, Sink
from .core import (
    MultGridCompoundModel,
    MultiGridBranchModel,
    MultiGridNodeModel,
    Node,
    Var,
    model,
)
from .grid import GasGrid, PowerGrid, WaterGrid
from .network import Network
from .node import Bus, Junction
from .phys.nl.hydraulics import junction_mass_flow_balance
from .phys.nl.opf import power_balance_equation


class MutableFloat(float):
    def __init__(self, val):
        self._val = val

    def __int__(self):
        return self._val

    def __index__(self):
        return self._val

    def __str__(self):
        return str(self._val)

    def __repr__(self):
        return repr(self._val)

    def set(self, val):
        self._val = val


@model
class GenericTransferBranch(MultiGridBranchModel):
    def __init__(self, loss=0, **kwargs) -> None:
        super().__init__(**kwargs)

        self._mass_flow = Var(-1)

        self.on_off = 1
        self._p_mw = Var(1)
        self._q_mvar = Var(1)
        self._t_from_pu = Var(350)
        self._t_to_pu = Var(350)

        self._loss = loss

    def loss_percent(self):
        return self._loss

    def is_cp(self):
        return False

    def _fill_el(self):
        self.p_to_mw = self._p_mw
        self.p_from_mw = self._p_mw
        self.q_to_mvar = self._q_mvar
        self.q_from_mvar = self._q_mvar

    def equations(self, grids, from_node_model, to_node_model, **kwargs):
        eqs = []
        print(grids)
        if type(grids) is WaterGrid or type(grids) is dict and WaterGrid in grids:
            self.mass_flow = self._mass_flow
            self.heat_mass_flow = self._mass_flow
            self.t_from_pu = self._t_from_pu
            self.t_to_pu = self._t_to_pu
            eqs += [self.t_from_pu == self.t_to_pu]
            eqs += [self.t_from_pu == from_node_model.t_pu]
            eqs += [to_node_model.t_pu == self.t_to_pu]
            eqs += [to_node_model.t_pu == from_node_model.t_pu]
            eqs += [from_node_model.pressure_pu == to_node_model.pressure_pu]
        if type(grids) is GasGrid or type(grids) is dict and GasGrid in grids:
            self.mass_flow = self._mass_flow
            self.gas_mass_flow = self._mass_flow
            eqs += [from_node_model.pressure_pu == to_node_model.pressure_pu]
            eqs += [from_node_model.pressure_pa == to_node_model.pressure_pa]
        if type(grids) is PowerGrid or type(grids) is dict and PowerGrid in grids:
            self._fill_el()

        return eqs


@model
class GasToHeatControlNode(MultiGridNodeModel, Junction):
    def __init__(
        self,
        heat_gen_w,
        efficiency_heat,
        hhv,
        regulation=1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.heat_gen_w = heat_gen_w
        self.efficiency_heat = efficiency_heat
        self._hhv = hhv
        self.regulation = regulation

        self.gas_mass_flow = Var(1)

        # eventually overridden
        self.t_k = Var(350)
        self.t_pu = Var(1)
        self.pressure_pa = Var(1000000)
        self.pressure_pu = Var(1)

    def equations(self, grid, from_branch_models, to_branch_models, childs, **kwargs):
        heat_to_branches = [
            branch
            for branch in to_branch_models
            if "heat_mass_flow" in branch.vars or type(branch) is SubHE
        ]
        heat_from_branches = [
            branch
            for branch in from_branch_models
            if "heat_mass_flow" in branch.vars or type(branch) is SubHE
        ]
        gas_to_branches = [
            branch for branch in to_branch_models if "gas_mass_flow" in branch.vars
        ]
        gas_eqs = self.calc_signed_mass_flow(
            [], gas_to_branches, [Sink(self.gas_mass_flow)]
        )
        heat_eqs = self.calc_signed_mass_flow(  #
            heat_from_branches,
            heat_to_branches,
            [],  #
        )
        heat_energy_eqs = self.calc_signed_heat_flow(
            heat_from_branches, heat_to_branches, [], None
        )
        return (
            junction_mass_flow_balance(heat_eqs),
            junction_mass_flow_balance(heat_energy_eqs),
            junction_mass_flow_balance(gas_eqs),
            [branch for branch in heat_from_branches if type(branch) is SubHE][0].q_w
            / 1000000
            == -self.efficiency_heat
            * self.gas_mass_flow
            * self.regulation
            * (3.6 * self._hhv),
            self.heat_gen_w
            == [branch for branch in heat_from_branches if type(branch) is SubHE][
                0
            ].q_w,
            self.t_pu == self.t_k / grid[1].t_ref,
            self.pressure_pu == self.pressure_pa / grid[1].pressure_ref,
        )


@model
class PowerToHeatControlNode(MultiGridNodeModel, Junction, Bus):
    def __init__(
        self, load_p_mw, load_q_mvar, heat_energy_mw, efficiency, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.load_p_mw = load_p_mw
        self.load_q_mvar = load_q_mvar
        self.heat_energy_mw = heat_energy_mw
        self.efficiency = efficiency

        self.t_k = Var(350)
        self.t_pu = Var(1)
        self.pressure_pa = Var(1000000)
        self.pressure_pu = Var(1)

    def equations(self, grid, from_branch_models, to_branch_models, childs, **kwargs):
        heat_to_branches = [
            branch
            for branch in to_branch_models
            if "heat_mass_flow" in branch.vars or type(branch) is SubHE
        ]
        heat_from_branches = [
            branch
            for branch in from_branch_models
            if "heat_mass_flow" in branch.vars or type(branch) is SubHE
        ]
        power_to_branches = [
            branch for branch in to_branch_models if "p_from_mw" in branch.vars
        ]
        power_eqs = self.calc_signed_power_values(
            [], power_to_branches, [PowerLoad(self.load_p_mw, self.load_q_mvar)]
        )
        heat_eqs = self.calc_signed_mass_flow(  #
            heat_from_branches,
            heat_to_branches,
            [],  #
        )
        heat_energy_eqs = self.calc_signed_heat_flow(
            heat_from_branches, heat_to_branches, [], None
        )
        print(grid)
        return (
            junction_mass_flow_balance(heat_eqs),
            junction_mass_flow_balance(heat_energy_eqs),
            [branch for branch in heat_to_branches if type(branch) is SubHE][0].q_w
            / 1000000
            == -self.heat_energy_mw,
            sum(power_eqs[0]) == 0,
            sum(power_eqs[1]) == 0,
            self.heat_energy_mw == self.efficiency * self.load_p_mw,
            self.t_pu == self.t_k / grid[1].t_ref,
            self.pressure_pu == self.pressure_pa / grid[1].pressure_ref,
        )


class SubHE(HeatExchanger):
    pass


@model
class CHPControlNode(MultiGridNodeModel, Junction, Bus):
    def __init__(
        self,
        mass_flow_capacity,
        efficiency_heat,
        efficiency_power,
        hhv,
        q_mvar=0,
        regulation=1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.mass_flow_capacity = mass_flow_capacity
        self.efficiency_heat = efficiency_heat
        self.efficiency_power = efficiency_power
        self.gen_q_mvar = q_mvar
        self._hhv = hhv
        self.regulation = regulation

        self._gen_p_mw = Var(1)
        self.heat_gen_w = Var(1)
        self.el_gen_mw = Var(1)

        # eventually overridden
        self.t_k = Var(350)
        self.t_pu = Var(1)
        self.pressure_pa = Var(1000000)
        self.pressure_pu = Var(1)

    def equations(self, grid, from_branch_models, to_branch_models, childs, **kwargs):
        heat_to_branches = [
            branch
            for branch in to_branch_models
            if "heat_mass_flow" in branch.vars or type(branch) is SubHE
        ]
        heat_from_branches = [
            branch
            for branch in from_branch_models
            if "heat_mass_flow" in branch.vars or type(branch) is SubHE
        ]
        gas_to_branches = [
            branch for branch in to_branch_models if "gas_mass_flow" in branch.vars
        ]
        power_from_branches = [
            branch for branch in from_branch_models if "p_to_mw" in branch.vars
        ]
        power_eqs = self.calc_signed_power_values(
            power_from_branches, [], [PowerGenerator(self._gen_p_mw, self.gen_q_mvar)]
        )
        gas_eqs = self.calc_signed_mass_flow(
            [], gas_to_branches, [Sink(self.mass_flow_capacity)]
        )
        heat_eqs = self.calc_signed_mass_flow(  #
            heat_from_branches,
            heat_to_branches,
            [],  #
        )
        heat_energy_eqs = self.calc_signed_heat_flow(
            heat_from_branches, heat_to_branches, [], None
        )
        return (
            junction_mass_flow_balance(heat_eqs),
            junction_mass_flow_balance(heat_energy_eqs),
            junction_mass_flow_balance(gas_eqs),
            power_balance_equation(power_eqs[0]),
            power_balance_equation(power_eqs[1]),
            [branch for branch in heat_from_branches if type(branch) is SubHE][0].q_w
            / 1000000
            == -self.efficiency_heat
            * self.mass_flow_capacity
            * self.regulation
            * (3.6 * self._hhv),
            self._gen_p_mw
            == -self.efficiency_power
            * self.mass_flow_capacity
            * self.regulation
            * (3.6 * self._hhv),
            self.heat_gen_w
            == [branch for branch in heat_from_branches if type(branch) is SubHE][
                0
            ].q_w,
            self.el_gen_mw == self._gen_p_mw,
            self.t_pu == self.t_k / grid[1].t_ref,
            self.pressure_pu == self.pressure_pa / grid[1].pressure_ref,
        )


@model
class CHP(MultGridCompoundModel):
    def __init__(
        self,
        diameter_m: float,
        efficiency_power: float,
        efficiency_heat: float,
        mass_flow_setpoint: float,
        q_mvar_setpoint: float = 0,
        temperature_ext_k: float = 293,
    ) -> None:
        self.diameter_m = diameter_m
        self.temperature_ext_k = temperature_ext_k
        self.efficiency_power = efficiency_power
        self.efficiency_heat = efficiency_heat

        self.mass_flow_setpoint = mass_flow_setpoint

        self.mass_flow = (
            mass_flow_setpoint
            if type(mass_flow_setpoint) is Var
            else MutableFloat(mass_flow_setpoint)
        )
        self.q_mvar = (
            q_mvar_setpoint
            if type(q_mvar_setpoint) is Var
            else MutableFloat(q_mvar_setpoint)
        )

    def set_active(self, activation_flag):
        if activation_flag:
            self._control_node.gas_consumption = self.mass_flow
        else:
            self._control_node.gas_consumption = 0

    def create(
        self,
        network: Network,
        gas_node: Node,
        heat_node: Node,
        heat_return_node: Node,
        power_node: Node,
    ):
        self._gas_grid = gas_node.grid
        self._control_node = CHPControlNode(
            self.mass_flow,
            self.efficiency_power,
            self.efficiency_heat,
            gas_node.grid.higher_heating_value,
        )
        node_id_control = network.node(
            self._control_node,
            grid=[power_node.grid, heat_node.grid, gas_node.grid],
            position=power_node.position,
        )
        network.branch(
            GenericTransferBranch(),  #
            gas_node.id,  #
            node_id_control,
        )
        network.branch(
            GenericTransferBranch(),
            heat_node.id,
            node_id_control,
        )
        network.branch(
            SubHE(
                Var(0.1),
                self.diameter_m,
            ),
            node_id_control,
            heat_return_node.id,
            grid=heat_return_node.grid,
        )
        network.branch(
            GenericTransferBranch(),
            node_id_control,
            power_node.id,
        )


@model
class GasToHeat(MultGridCompoundModel):
    def __init__(
        self,
        heat_energy_w,
        diameter_m,
        temperature_ext_k,
        efficiency,
    ) -> None:
        self.diameter_m = diameter_m
        self.temperature_ext_k = temperature_ext_k
        self.efficiency = efficiency

        self.heat_energy_w = MutableFloat(-heat_energy_w)

    def create(
        self, network: Network, gas_node: Node, heat_node: Node, heat_return_node: Node
    ):
        self._gas_grid = gas_node.grid
        node_id_control = network.node(
            GasToHeatControlNode(
                self.heat_energy_w,
                self.efficiency,
                gas_node.grid.higher_heating_value,
            ),
            grid=[heat_node.grid, gas_node.grid],
            position=gas_node.position,
        )
        network.branch(
            GenericTransferBranch(),
            gas_node.id,
            node_id_control,
        )
        network.branch(
            GenericTransferBranch(),
            heat_node.id,
            node_id_control,
        )
        network.branch(
            SubHE(
                Var(0.1),
                self.diameter_m,
            ),
            node_id_control,
            heat_return_node.id,
            grid=heat_return_node.grid,
        )


@model
class PowerToHeat(MultGridCompoundModel):
    def __init__(
        self,
        heat_energy_mw,
        diameter_m,
        temperature_ext_k,
        efficiency,
        q_mvar_setpoint=0,
    ) -> None:
        self.diameter_m = diameter_m
        self.temperature_ext_k = temperature_ext_k
        self.efficiency = efficiency

        self.heat_energy_mw = (
            heat_energy_mw
            if type(heat_energy_mw) is Var
            else MutableFloat(heat_energy_mw)
        )
        self.load_p_mw = Var(1)
        self.load_q_mvar = (
            q_mvar_setpoint
            if type(q_mvar_setpoint) is Var
            else MutableFloat(q_mvar_setpoint)
        )

    def set_active(self, activation_flag):
        if activation_flag:
            self._control_node.heat_energy_mw = self.heat_energy_mw
        else:
            self._control_node.heat_energy_mw = 0

    def create(
        self,
        network: Network,
        power_node: Node,
        heat_node: Node,
        heat_return_node: Node,
    ):
        self._control_node = PowerToHeatControlNode(
            self.load_p_mw, self.load_q_mvar, self.heat_energy_mw, self.efficiency
        )
        node_id_control = network.node(
            self._control_node,
            grid=[power_node.grid, heat_node.grid],
            position=power_node.position,
        )
        network.branch(GenericTransferBranch(), power_node.id, node_id_control)
        network.branch(
            GenericTransferBranch(),
            node_id_control,
            heat_return_node.id,
        )
        network.branch(
            SubHE(
                Var(0.1),
                self.diameter_m,
            ),
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
        self.p_mw_capacity = -p_mw_setpoint
        self.mass_flow_capacity = Var(1)

        self.on_off = 1
        self.p_to_mw = Var(-p_mw_setpoint)
        self.q_to_mvar = -q_mvar_setpoint
        self.from_mass_flow = Var(1)
        self.regulation = regulation

    def loss_percent(self):
        return 1 - self.efficiency

    def equations(self, grids, from_node_model, to_node_model, **kwargs):
        return [
            self.p_to_mw == self.regulation * self.p_mw_capacity,
            -self.p_to_mw
            == self.efficiency
            * self.from_mass_flow
            * (3.6 * grids[GasGrid].higher_heating_value),
            -self.p_mw_capacity
            == self.efficiency
            * self.mass_flow_capacity
            * (3.6 * grids[GasGrid].higher_heating_value),
        ]


@model
class PowerToGas(MultiGridBranchModel):
    def __init__(
        self, efficiency, mass_flow_setpoint, consume_q_mvar_setpoint=0, regulation=1
    ) -> None:
        super().__init__()

        self.efficiency = efficiency
        self.mass_flow_capacity = -mass_flow_setpoint
        self.p_mw_capacity = Var(1)

        self.on_off = 1
        self.p_from_mw = Var(1)
        self.q_from_mvar = consume_q_mvar_setpoint
        self.to_mass_flow = Var(self.mass_flow_capacity)
        self.regulation = regulation

    def loss_percent(self):
        return 1 - self.efficiency

    def equations(self, grids, from_node_model, to_node_model, **kwargs):
        return [
            (
                self.to_mass_flow
                == -self.efficiency
                * self.p_from_mw
                * (1 / (grids[GasGrid].higher_heating_value * 3.6))
            ),
            self.p_from_mw > 0,
            self.p_from_mw == self.p_mw_capacity * self.regulation,
            (
                self.mass_flow_capacity
                == -self.efficiency
                * self.p_mw_capacity
                * (1 / (grids[GasGrid].higher_heating_value * 3.6))
            ),
        ]

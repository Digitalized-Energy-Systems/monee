from .branch import HeatExchanger
from .child import PowerGenerator, PowerLoad, Sink
from .core import (
    CompoundModel,
    MultiGridBranchModel,
    Network,
    Node,
    Var,
    model,
)
from .grid import NO_GRID, GasGrid, PowerGrid, WaterGrid
from .node import Bus, Junction


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

        self._mass_flow = Var(1)

        self._p_mw = Var(1)
        self._q_mvar = Var(1)
        self._loss = loss

    def loss_percent(self):
        return self._loss

    def _fill_el(self):
        self.p_to_mw = self._p_mw
        self.p_from_mw = self._p_mw
        self.q_to_mvar = self._q_mvar
        self.q_from_mvar = self._q_mvar

    def equations(self, grids, from_node_model, to_node_model, **kwargs):
        if type(grids) is WaterGrid or type(grids) is dict and WaterGrid in grids:
            self.mass_flow = self._mass_flow
            self.heat_mass_flow = self._mass_flow
        if type(grids) is GasGrid or type(grids) is dict and GasGrid in grids:
            self.mass_flow = self._mass_flow
            self.gas_mass_flow = self._mass_flow
        if type(grids) is PowerGrid or type(grids) is dict and PowerGrid in grids:
            self._fill_el()

        for k, v in from_node_model.vars.items():
            if hasattr(to_node_model, k):
                setattr(to_node_model, k, v)
        return []


@model
class GasToHeatControlNode(Junction):
    def __init__(
        self, gas_consumption, heat_energy_mw, efficiency, hhv, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.gas_consumption = gas_consumption
        self.heat_energy_mw = heat_energy_mw
        self.efficiency = efficiency
        self.hhv = hhv

    def equations(self, grid, from_branch_models, to_branch_models, childs, **kwargs):
        heat_to_branches = [
            branch for branch in to_branch_models if "mass_flow" in branch.vars
        ]
        heat_from_branches = [
            branch for branch in from_branch_models if "mass_flow" in branch.vars
        ]
        gas_to_branches = [
            branch for branch in to_branch_models if "to_mass_flow" in branch.vars
        ]

        return (
            sum(
                self.calc_signed_mass_flow(  #
                    heat_from_branches,
                    heat_to_branches,
                    [],  #
                )  #
            )
            == 0,
            sum(
                self.calc_signed_mass_flow(
                    gas_to_branches, [], [Sink(self.gas_consumption)]
                )
            )
            == 0,
            self.heat_energy_mw
            == self.efficiency * self.gas_consumption * 3.6 * self.hhv,
        )


@model
class PowerToHeatControlNode(Junction, Bus):
    def __init__(
        self, load_p_mw, load_q_mvar, heat_energy_mw, efficiency, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.load_p_mw = load_p_mw
        self.load_q_mvar = load_q_mvar
        self.heat_energy_mw = heat_energy_mw
        self.efficiency = efficiency

    def equations(self, grid, from_branch_models, to_branch_models, childs, **kwargs):
        heat_to_branches = [
            branch for branch in to_branch_models if "mass_flow" in branch.vars
        ]
        heat_from_branches = [
            branch for branch in from_branch_models if "mass_flow" in branch.vars
        ]
        power_to_branches = [
            branch for branch in to_branch_models if "p_from_mw" in branch.vars
        ]
        power_eqs = self.calc_signed_power_values(
            [], power_to_branches, [PowerLoad(self.load_p_mw, self.load_q_mvar)]
        )

        return (
            sum(
                self.calc_signed_mass_flow(  #
                    heat_from_branches,
                    heat_to_branches,
                    [],  #
                )  #
            )
            == 0,
            sum(power_eqs[0]) == 0,
            sum(power_eqs[1]) == 0,
            self.heat_energy_mw == self.efficiency * self.load_p_mw,
        )


@model
class CHPControlNode(Junction, Bus):
    def __init__(
        self,
        gas_consumption,
        efficiency_heat,
        efficiency_power,
        hhv,
        q_mvar=0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.gas_consumption = gas_consumption
        self.efficiency_heat = efficiency_heat
        self.efficiency_power = efficiency_power
        self.gen_p_mw = Var(1)
        self.gen_q_mvar = q_mvar
        self._hhv = hhv

        # eventually overridden
        self.t_k = 0
        self.pressure_pa = 0

    def equations(self, grid, from_branch_models, to_branch_models, childs, **kwargs):
        heat_to_branches = [
            branch
            for branch in to_branch_models
            if "heat_mass_flow" in branch.vars or type(branch) is HeatExchanger
        ]
        heat_from_branches = [
            branch
            for branch in from_branch_models
            if "heat_mass_flow" in branch.vars or type(branch) is HeatExchanger
        ]
        gas_to_branches = [
            branch for branch in to_branch_models if "gas_mass_flow" in branch.vars
        ]
        power_from_branches = [
            branch for branch in from_branch_models if "p_to_mw" in branch.vars
        ]
        power_eqs = self.calc_signed_power_values(
            power_from_branches, [], [PowerGenerator(self.gen_p_mw, self.gen_q_mvar)]
        )

        return (
            sum(
                self.calc_signed_mass_flow(  #
                    heat_from_branches,
                    heat_to_branches,
                    [],  #
                )  #
            )
            == 0,
            sum(
                self.calc_signed_mass_flow(
                    [], gas_to_branches, [Sink(self.gas_consumption)]
                )
            )
            == 0,
            sum(power_eqs[0]) == 0,
            sum(power_eqs[1]) == 0,
            [branch for branch in heat_from_branches if type(branch) is HeatExchanger][
                0
            ].q_w / 1000000
            == -self.efficiency_heat * self.gas_consumption * (1 / (3.6* self._hhv)),
            self.gen_p_mw
            == self.efficiency_power * self.gas_consumption * (3.6 * self._hhv),
        )


@model
class CHP(CompoundModel):
    def __init__(
        self,
        diameter_m: float,
        efficiency_power: float,
        efficiency_heat: float,
        mass_flow_setpoint: float,
        q_mvar_setpoint: float = 0,
        temperature_ext_k: float = 293,
        in_line_operation: bool = True,
    ) -> None:
        self._in_line_operation = in_line_operation
        self.diameter_m = diameter_m
        self.temperature_ext_k = temperature_ext_k
        self.efficiency_power = efficiency_power
        self.efficiency_heat = efficiency_heat

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
            grid=NO_GRID,
            position=power_node.position,
        )
        network.branch(
            GenericTransferBranch(),  #
            gas_node.id,  #
            node_id_control,
        )
        network.branch(
            GenericTransferBranch(),
            heat_return_node.id,
            node_id_control,
        )
        network.branch(
            HeatExchanger(
                Var(0.01), self.diameter_m, in_line_operation=self._in_line_operation
            ),
            node_id_control,
            heat_node.id,
            grid=heat_node.grid,
        )
        network.branch(
            GenericTransferBranch(loss=1 - self.efficiency_power),
            node_id_control,
            power_node.id,
        )


@model
class GasToHeat(CompoundModel):
    def __init__(
        self,
        heat_energy_mw,
        diameter_m,
        temperature_ext_k,
        efficiency,
        in_line_operation=False,
    ) -> None:
        self._in_line_operation = in_line_operation
        self.diameter_m = diameter_m
        self.temperature_ext_k = temperature_ext_k
        self.efficiency = efficiency

        self.heat_energy_mw = MutableFloat(heat_energy_mw)
        self.sink_mass_flow = Var(1)

    def create(
        self, network: Network, gas_node: Node, heat_node: Node, heat_return_node: Node
    ):
        self._gas_grid = gas_node.grid
        node_id_control = network.node(
            GasToHeatControlNode(
                self.sink_mass_flow,
                self.heat_energy_mw,
                self.efficiency,
                gas_node.grid.higher_heating_value,
            ),
            grid=NO_GRID,
            position=gas_node.position,
        )
        network.branch(
            GenericTransferBranch(loss=1 - self.efficiency),
            gas_node.id,
            node_id_control,
        )
        network.branch(
            GenericTransferBranch(),
            heat_return_node.id,
            node_id_control,
        )
        network.branch(
            HeatExchanger(
                self.heat_energy_mw,
                self.diameter_m,
                in_line_operation=self._in_line_operation,
            ),
            node_id_control,
            heat_node.id,
        )


@model
class PowerToHeat(CompoundModel):
    def __init__(
        self,
        heat_energy_mw,
        diameter_m,
        temperature_ext_k,
        efficiency,
        q_mvar_setpoint=0,
        in_line_operation=False,
    ) -> None:
        self._in_line_operation = in_line_operation
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
            grid=NO_GRID,
            position=power_node.position,
        )
        network.branch(GenericTransferBranch(), power_node.id, node_id_control)
        network.branch(
            GenericTransferBranch(),
            heat_return_node.id,
            node_id_control,
        )
        network.branch(
            HeatExchanger(
                self.heat_energy_mw,
                self.diameter_m,
                in_line_operation=self._in_line_operation,
            ),
            node_id_control,
            heat_node.id,
        )


@model
class GasToPower(MultiGridBranchModel):
    def __init__(self, efficiency, p_mw_setpoint, q_mvar_setpoint=0, regulation=1) -> None:
        super().__init__()

        self.efficiency = efficiency
        self.p_mw_setpoint = -p_mw_setpoint

        self.p_to_mw = Var(-p_mw_setpoint)
        self.q_to_mvar = -q_mvar_setpoint
        self.from_mass_flow = Var(1)
        self.regulation = regulation

    def loss_percent(self):
        return 1 - self.efficiency

    def equations(self, grids, from_node_model, to_node_model, **kwargs):
        return [
            self.p_to_mw == self.regulation * self.p_mw_setpoint,
            -self.p_to_mw == self.efficiency * self.from_mass_flow * (3.6 * grids[GasGrid].higher_heating_value)
        ]


@model
class PowerToGas(MultiGridBranchModel):
    def __init__(
        self, efficiency, mass_flow_setpoint, consume_q_mvar_setpoint=0, regulation=1
    ) -> None:
        super().__init__()

        self.efficiency = efficiency
        self.mass_flow_setpoint = -mass_flow_setpoint

        self.p_from_mw = Var(1)
        self.q_from_mvar = consume_q_mvar_setpoint
        self.to_mass_flow = Var(-mass_flow_setpoint)
        self.regulation = regulation

    def loss_percent(self):
        return 1 - self.efficiency

    def equations(self, grids, from_node_model, to_node_model, **kwargs):
        return [(
            -self.to_mass_flow
            == self.efficiency
            * self.p_from_mw
            * (1 / (grids[GasGrid].higher_heating_value * 3.6))
        ), 
            self.p_from_mw > 0, 
            self.mass_flow_setpoint * self.regulation == self.to_mass_flow
        ]

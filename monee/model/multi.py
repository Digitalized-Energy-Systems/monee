from .core import (
    MultiGridBranchModel,
    NodeModel,
    CompoundModel,
    Var,
    Network,
    Node,
    model,
)
from .grid import GasGrid
from .branch import WaterPipe
from .child import PowerLoad, Sink


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
    def equations(self, grids, from_node_model, to_node_model, **kwargs):
        for k, v in from_node_model.__dict__:
            setattr(to_node_model, k, v)


@model
class HeatTransferBranch(WaterPipe):
    def __init__(self, heat_energy_mw, diameter_m, temperature_ext_k) -> None:
        super().__init__(diameter_m, 0.01, temperature_ext_k, 0, 1, diameter_m / 2)

        self.heat_loss = heat_energy_mw


@model
class EmptyControlNode(NodeModel):
    def equations(self, grid, in_branch_models, out_branch_models, childs, **kwargs):
        pass


@model
class CHP(CompoundModel):
    def __init__(
        self,
        diameter_m,
        efficiency,
        mass_flow_setpoint,
        q_mvar_setpoint=0,
        temperature_ext_k=293,
    ) -> None:
        self.diameter_m = diameter_m
        self.temperature_ext_k = temperature_ext_k
        self.efficiency = efficiency

        self.mass_flow = MutableFloat(mass_flow_setpoint)
        self.q_mvar = MutableFloat(q_mvar_setpoint)
        self.p_mw = Var(1)
        self.heat_energy_mw = Var(1)

    def create(
        self,
        network: Network,
        gas_node: Node,
        heat_node: Node,
        heat_return_node: Node,
        power_node: Node,
    ):
        self._gas_to_heat = GasToHeat(
            self.heat_energy_mw,
            self.diameter_m,
            self.temperature_ext_k,
            self.efficiency,
        )
        self._gas_to_heat.create(network, gas_node, heat_node, heat_return_node)

        network.branch(
            GasToPower(self.efficiency, self.p_mw, q_mvar_setpoint=self.q_mvar),
            gas_node.id,
            power_node.id,
        )
        self._gas_grid = gas_node.grid

    def equations(self, network, **kwargs):
        return (
            self.p_mw + self.heat_energy_mw
            <= self.mass_flow * 3.6 / self._gas_grid.higher_heating_value
        )


@model
class GasToHeat(CompoundModel):
    def __init__(
        self, heat_energy_mw, diameter_m, temperature_ext_k, efficiency
    ) -> None:
        self.heat_energy_mw = heat_energy_mw
        self.diameter_m = diameter_m
        self.temperature_ext_k = temperature_ext_k
        self.efficiency = efficiency

        self.sink_mass_flow = Var(1)

    def create(
        self, network: Network, gas_node: Node, heat_node: Node, heat_return_node: Node
    ):
        self._gas_grid = gas_node.grid
        node_id_control = network.node(
            EmptyControlNode(),
            child_ids=[network.child(Sink(mass_flow=self.sink_mass_flow))],
        )
        network.branch(GenericTransferBranch(), gas_node.id, node_id_control)
        network.branch(
            HeatTransferBranch(
                self.heat_energy_mw, self.diameter_m, self.temperature_ext_k
            ),
            heat_return_node.id,
            node_id_control,
        )
        network.branch(
            HeatTransferBranch(
                self.heat_energy_mw, self.diameter_m, self.temperature_ext_k
            ),
            node_id_control,
            heat_node.id,
        )

    def equations(self, network, **kwargs):
        return (
            self.heat_energy_mw
            == self.efficiency
            * self.sink_mass_flow
            * 3600
            / self._gas_grid.higher_heating_value
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
    ) -> None:
        self.diameter_m = diameter_m
        self.temperature_ext_k = temperature_ext_k
        self.efficiency = efficiency

        self.heat_energy_mw = MutableFloat(heat_energy_mw)
        self.load_p_mw = Var(1)
        self.load_q_mvar = MutableFloat(q_mvar_setpoint)

    def create(
        self,
        network: Network,
        power_node: Node,
        heat_node: Node,
        heat_return_node: Node,
    ):
        node_id_control = network.node(
            EmptyControlNode(),
            child_ids=[
                network.child(PowerLoad(p_mw=self.load_p_mw, q_mvar=self.load_q_mvar))
            ],
        )
        network.branch(GenericTransferBranch(), power_node, node_id_control)
        network.branch(
            HeatTransferBranch(
                self.heat_energy_mw, self.diameter_m, self.temperature_ext_k
            ),
            heat_return_node,
            node_id_control,
        )
        network.branch(
            HeatTransferBranch(
                self.heat_energy_mw, self.diameter_m, self.temperature_ext_k
            ),
            node_id_control,
            heat_node,
        )

    def equations(self, network, **kwargs):
        return self.heat_energy_mw == self.efficiency * self.load_p_mw


@model
class GasToPower(MultiGridBranchModel):
    def __init__(self, efficiency, p_mw_setpoint, q_mvar_setpoint=0) -> None:
        super().__init__()

        self.efficiency = efficiency

        self.p_to_mw = p_mw_setpoint
        self.q_to_mvar = q_mvar_setpoint
        self.from_mass_flow = Var(1)

    def equations(self, grids, from_node_model, to_node_model, **kwargs):
        return self.p_to_mw == self.efficiency * self.mass_flow * (
            3.6 * grids[GasGrid].higher_heating_value
        )


@model
class PowerToGas(MultiGridBranchModel):
    def __init__(
        self, efficiency, mass_flow_setpoint, consume_q_mvar_setpoint=0
    ) -> None:
        super().__init__()

        self.efficiency = efficiency

        self.p_from_mw = Var(1)
        self.q_from_mvar = consume_q_mvar_setpoint
        self.to_mass_flow = mass_flow_setpoint

    def equations(self, grids, from_node_model, to_node_model, **kwargs):
        return (
            self.to_mass_flow
            == self.efficiency
            * self.p_from_mw
            * (1 / grids[GasGrid].higher_heating_value * 3.6)
        ), self.p_from_mw > 0

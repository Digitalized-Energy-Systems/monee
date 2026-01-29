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
from .phys.core.hydraulics import junction_mass_flow_balance
from .phys.nonlinear.ac import power_balance_equation


class MutableFloat(float):
    """
    No docstring provided.
    """

    def __init__(self, val):
        self._val = val

    def __int__(self):
        """
        No docstring provided.
        """
        return self._val

    def __index__(self):
        """
        No docstring provided.
        """
        return self._val

    def __str__(self):
        """
        No docstring provided.
        """
        return str(self._val)

    def __repr__(self):
        """
        No docstring provided.
        """
        return repr(self._val)

    def set(self, val):
        """
        No docstring provided.
        """
        self._val = val


@model
class GenericTransferBranch(MultiGridBranchModel):
    """
    No docstring provided.
    """

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
        """
        No docstring provided.
        """
        return self._loss

    def is_cp(self):
        """
        No docstring provided.
        """
        return False

    def init(self, grids):
        """
        No docstring provided.
        """
        if type(grids) is WaterGrid or (type(grids) is dict and WaterGrid in grids):
            self.mass_flow = self._mass_flow
            self.heat_mass_flow = self._mass_flow
            self.t_from_pu = self._t_from_pu
            self.t_to_pu = self._t_to_pu
        if type(grids) is GasGrid or (type(grids) is dict and GasGrid in grids):
            self.mass_flow = self._mass_flow
            self.gas_mass_flow = self._mass_flow
        if type(grids) is PowerGrid or (type(grids) is dict and PowerGrid in grids):
            self.p_to_mw = self._p_mw
            self.p_from_mw = self._p_mw
            self.q_to_mvar = self._q_mvar
            self.q_from_mvar = self._q_mvar

    def equations(self, grids, from_node_model, to_node_model, **kwargs):
        """
        Defines the system of equations for a generic transfer branch, enforcing variable consistency across connected nodes and grids.

        This method generates the physical and operational constraints for a transfer branch that may span multiple grid domains, such as water or gas networks. For water grids, it enforces temperature and pressure continuity between the branch and its connected nodes. The method is typically called during network simulation or optimization to ensure correct variable propagation and coupling between nodes.

        Args:
            grids: The grid object or a dictionary of grid objects relevant to the branch (e.g., WaterGrid, GasGrid).
            from_node_model: The model instance representing the source node connected to the branch.
            to_node_model: The model instance representing the destination node connected to the branch.
            **kwargs: Additional keyword arguments for solver options or equation customization.

        Returns:
            list: A list of equations enforcing variable consistency (e.g., temperature and pressure) across the branch and its nodes. The list may be empty if no relevant grid is present.

        Examples:
            # Called automatically during network simulation:
            eqs = transfer_branch.equations(grids, from_node, to_node)
            # eqs will contain temperature and pressure continuity constraints for water grids.
        """
        eqs = []
        if type(grids) is WaterGrid or (type(grids) is dict and WaterGrid in grids):
            eqs += [self.t_from_pu == self.t_to_pu]
            eqs += [self.t_from_pu == from_node_model.t_pu]
            eqs += [to_node_model.t_pu == self.t_to_pu]
            eqs += [to_node_model.t_pu == from_node_model.t_pu]
            eqs += [from_node_model.pressure_pu == to_node_model.pressure_pu]
            eqs += [from_node_model.pressure_pa == to_node_model.pressure_pa]
        if type(grids) is GasGrid or (type(grids) is dict and GasGrid in grids):
            pass
        return eqs


@model
class GasToHeatControlNode(MultiGridNodeModel, Junction):
    """
    No docstring provided.
    """

    def __init__(
        self, heat_gen_w, efficiency_heat, hhv, regulation=1, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.efficiency_heat = efficiency_heat
        self._hhv = hhv
        self.regulation = regulation

        self.gas_kgps = Var(1)
        self.heat_w = heat_gen_w

        self.t_k = Var(350)
        self.t_pu = Var(1)
        self.pressure_pa = Var(1000000)
        self.pressure_pu = Var(1)

    def equations(self, grid, from_branch_models, to_branch_models, childs, **kwargs):
        """
        No docstring provided.
        """
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
        gas_eqs = self.calc_signed_mass_flow([], gas_to_branches, [Sink(self.gas_kgps)])
        heat_eqs = self.calc_signed_mass_flow(heat_from_branches, heat_to_branches, [])
        heat_energy_eqs = self.calc_signed_heat_flow(
            heat_from_branches, heat_to_branches, [], None
        )
        return [
            junction_mass_flow_balance(heat_eqs),
            junction_mass_flow_balance(heat_energy_eqs),
            junction_mass_flow_balance(gas_eqs),
            [branch for branch in heat_from_branches if type(branch) is SubHE][0].q_w
            / 1000000
            == -self.efficiency_heat
            * self.gas_kgps
            * self.regulation
            * (3.6 * self._hhv),
            self.heat_w
            == [branch for branch in heat_from_branches if type(branch) is SubHE][
                0
            ].q_w,
            self.t_pu == self.t_k / grid[1].t_ref,
            self.pressure_pu == self.pressure_pa / grid[1].pressure_ref,
        ]


@model
class PowerToHeatControlNode(MultiGridNodeModel, Junction, Bus):
    """
    No docstring provided.
    """

    def __init__(
        self, load_p_mw, load_q_mvar, heat_energy_mw, efficiency, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.load_q_mvar = load_q_mvar
        self.efficiency = efficiency

        self.el_mw = load_p_mw
        self.heat_w = heat_energy_mw

        self.t_k = Var(350)
        self.t_pu = Var(1)
        self.pressure_pa = Var(1000000)
        self.pressure_pu = Var(1)

    def equations(self, grid, from_branch_models, to_branch_models, childs, **kwargs):
        """
        No docstring provided.
        """
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
            [], power_to_branches, [PowerLoad(self.el_mw, self.load_q_mvar)]
        )
        heat_eqs = self.calc_signed_mass_flow(heat_from_branches, heat_to_branches, [])
        heat_energy_eqs = self.calc_signed_heat_flow(
            heat_from_branches, heat_to_branches, [], None
        )
        return [
            junction_mass_flow_balance(heat_eqs),
            junction_mass_flow_balance(heat_energy_eqs),
            [branch for branch in heat_to_branches if type(branch) is SubHE][0].q_w
            / 1000000
            == -self.heat_w,
            sum(power_eqs[0]) == 0,
            sum(power_eqs[1]) == 0,
            self.heat_w == self.efficiency * self.el_mw,
            self.t_pu == self.t_k / grid[1].t_ref,
            self.pressure_pu == self.pressure_pa / grid[1].pressure_ref,
        ]


class SubHE(HeatExchanger):
    """
    Represents a subordinate or auxiliary heat exchanger within a multi-energy network model.

    SubHE is a specialized subclass of HeatExchanger used to model secondary or supporting heat exchange processes, such as those found in combined heat and power (CHP), power-to-heat, or gas-to-heat conversion systems. This class is typically used as a building block in compound models where additional heat transfer elements are required to accurately represent energy flows and balances.

    Example:
        sub_he = SubHE(q_mw=-1000, diameter_m=0.3)
        # Integrate sub_he into a network branch or compound system

    Attributes:
        Inherits all attributes from HeatExchanger, such as heat transfer rate, diameter, and temperature settings.
    """


@model
class CHPControlNode(MultiGridNodeModel, Junction, Bus):
    """
    Represents a control node for combined heat and power (CHP) systems, managing energy and mass balances across power, heat, and gas domains.

    CHPControlNode extends MultiGridNodeModel, Junction, and Bus to provide a unified interface for modeling the operational logic and constraints of a CHP unit within a multi-energy network. It tracks key parameters such as fuel mass flow, electrical and thermal efficiencies, and regulation factors, and exposes variables for integration with network branches. Use this class when simulating or optimizing CHP systems that require explicit coupling between gas, heat, and electrical grids.

    Example:
        chp_node = CHPControlNode(
            mass_flow_capacity=1.2,
            efficiency_power=0.35,
            efficiency_heat=0.5,
            hhv=42.5e6,
            q_mvar=0,
            regulation=1
        )
        # Integrate chp_node into a network and call chp_node.equations(...) during simulation

    Parameters:
        mass_flow_capacity: Maximum or setpoint mass flow of fuel (e.g., gas) supplied to the CHP unit.
        efficiency_power: Electrical efficiency (fraction, 0 < value ≤ 1).
        efficiency_heat: Thermal efficiency (fraction, 0 < value ≤ 1).
        hhv: Higher heating value of the fuel (J/kg).
        q_mvar (optional): Reactive power setpoint for the generator. Defaults to 0.
        regulation (optional): Regulation factor for operational flexibility. Defaults to 1.
        **kwargs: Additional keyword arguments for parent class initialization.

    Attributes:
        mass_flow_capacity: Fuel mass flow capacity or setpoint.
        efficiency_power: Electrical efficiency.
        efficiency_heat: Thermal efficiency.
        gen_q_mvar: Reactive power setpoint.
        _hhv: Higher heating value of the fuel.
        regulation: Regulation factor.
        _gen_p_mw: Electrical power generation variable.
        heat_gen_w: Thermal power generation variable.
        el_gen_mw: Electrical power generation variable (duplicate for unified interface).
        el_mw: Unified electrical power variable.
        gas_kgps: Unified gas mass flow variable.
        heat_w: Unified heat power variable.
        t_k: Node temperature (K).
        t_pu: Node temperature (per unit).
        pressure_pa: Node pressure (Pa).
        pressure_pu: Node pressure (per unit).

    Methods:
        equations(grid, from_branch_models, to_branch_models, childs, **kwargs): Defines the system of equations for the CHP node, including mass and energy balances, power and heat conversion, and normalization constraints.
    """

    def __init__(
        self,
        mass_flow_capacity,
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

        self.el_mw = Var(-1)
        self.gas_kgps = mass_flow_capacity
        self.heat_w = Var(-1000)

        self.t_k = Var(350)
        self.t_pu = Var(1)
        self.pressure_pa = Var(1000000)
        self.pressure_pu = Var(1)

    def equations(self, grid, from_branch_models, to_branch_models, childs, **kwargs):
        """
        Defines the system of equations for a combined heat and power (CHP) control node, capturing energy and mass balances across power, heat, and gas domains.

        This method assembles the physical and operational constraints for a CHP node, including mass flow balances for heat and gas, power balance equations, and thermodynamic relationships for energy conversion. It integrates the effects of efficiency, regulation, and fuel properties, and links the node's internal variables to the connected branches. Use this method during network simulation or optimization to ensure the CHP node's behavior is accurately represented within the multi-energy system.

        Args:
            grid: List or collection of grid objects, where grid[1] is expected to be the heat grid for reference values.
            from_branch_models (list): Branch models representing flows entering the node.
            to_branch_models (list): Branch models representing flows leaving the node.
            childs (list): Child component models attached to the node.
            **kwargs: Additional keyword arguments for solver options or equation customization.

        Returns:
            tuple: A tuple of equations representing:
                - Heat mass flow balance at the node.
                - Heat energy flow balance at the node.
                - Gas mass flow balance at the node.
                - Power balance equations (active and reactive).
                - Heat exchanger energy conversion constraint.
                - Electrical power generation constraint.
                - Consistency between internal heat and electrical variables and branch flows.
                - Temperature and pressure normalization constraints.

        Raises:
            IndexError: If no SubHE branch is found in the heat_from_branches list.
            KeyError: If expected variables are missing from branch models.

        Examples:
            # Called automatically during network simulation:
            eqs = chp_control_node.equations(grid, from_branches, to_branches, childs)
        """
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
            power_from_branches, [], [PowerGenerator(self.el_mw, self.gen_q_mvar)]
        )
        gas_eqs = self.calc_signed_mass_flow(
            [], gas_to_branches, [Sink(self.gas_kgps * self.regulation)]
        )
        heat_eqs = self.calc_signed_mass_flow(heat_from_branches, heat_to_branches, [])
        heat_energy_eqs = self.calc_signed_heat_flow(
            heat_from_branches, heat_to_branches, [], None
        )
        return [
            junction_mass_flow_balance(heat_eqs),
            junction_mass_flow_balance(heat_energy_eqs),
            junction_mass_flow_balance(gas_eqs),
            power_balance_equation(power_eqs[0]),
            power_balance_equation(power_eqs[1]),
            [branch for branch in heat_from_branches if type(branch) is SubHE][0].q_w
            / 1000000
            == -self.efficiency_heat
            * self.gas_kgps
            * self.regulation
            * (3.6 * self._hhv),
            self.el_mw
            == -self.efficiency_power
            * self.gas_kgps
            * self.regulation
            * (3.6 * self._hhv),
            self.heat_w
            == [branch for branch in heat_from_branches if type(branch) is SubHE][
                0
            ].q_w,
            self.t_k == self.t_pu * grid[1].t_ref,
            self.pressure_pu == self.pressure_pa / grid[1].pressure_ref,
        ]


@model
class CHP(MultGridCompoundModel):
    """
    No docstring provided.
    """

    def __init__(
        self,
        diameter_m: float,
        efficiency_power: float,
        efficiency_heat: float,
        mass_flow_setpoint: float,
        q_mvar_setpoint: float = 0,
        temperature_ext_k: float = 293,
        regulation=1,
    ) -> None:
        self.diameter_m = diameter_m
        self.temperature_ext_k = temperature_ext_k
        self.regulation = regulation
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

    def create(
        self,
        network: Network,
        gas_node: Node,
        heat_node: Node,
        heat_return_node: Node,
        power_node: Node,
    ):
        """
        No docstring provided.
        """
        self._gas_grid = gas_node.grid
        self._control_node = CHPControlNode(
            self.mass_flow,
            self.efficiency_power,
            self.efficiency_heat,
            gas_node.grid.higher_heating_value,
            regulation=self.regulation,
        )
        node_id_control = network.node(
            self._control_node,
            grid=[power_node.grid, heat_node.grid, gas_node.grid],
            position=power_node.position,
        )
        network.branch(GenericTransferBranch(), gas_node.id, node_id_control)
        network.branch(GenericTransferBranch(), heat_node.id, node_id_control)
        network.branch(
            SubHE(Var(-1000), self.diameter_m),
            node_id_control,
            heat_return_node.id,
            grid=heat_return_node.grid,
        )
        network.branch(GenericTransferBranch(), node_id_control, power_node.id)


@model
class GasToHeat(MultGridCompoundModel):
    """
    No docstring provided.
    """

    def __init__(
        self, heat_energy_w, diameter_m, temperature_ext_k, efficiency
    ) -> None:
        self.diameter_m = diameter_m
        self.temperature_ext_k = temperature_ext_k
        self.efficiency = efficiency
        self.heat_energy_w = MutableFloat(-heat_energy_w)

    def create(
        self, network: Network, gas_node: Node, heat_node: Node, heat_return_node: Node
    ):
        """
        No docstring provided.
        """
        self._gas_grid = gas_node.grid
        node_id_control = network.node(
            GasToHeatControlNode(
                self.heat_energy_w, self.efficiency, gas_node.grid.higher_heating_value
            ),
            grid=[heat_node.grid, gas_node.grid],
            position=gas_node.position,
        )
        network.branch(GenericTransferBranch(), gas_node.id, node_id_control)
        network.branch(GenericTransferBranch(), heat_node.id, node_id_control)
        network.branch(
            SubHE(Var(0.1), self.diameter_m),
            node_id_control,
            heat_return_node.id,
            grid=heat_return_node.grid,
        )


@model
class PowerToHeat(MultGridCompoundModel):
    """
    No docstring provided.
    """

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
        """
        No docstring provided.
        """
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
        """
        No docstring provided.
        """
        self._control_node = PowerToHeatControlNode(
            self.load_p_mw, self.load_q_mvar, self.heat_energy_mw, self.efficiency
        )
        node_id_control = network.node(
            self._control_node,
            grid=[power_node.grid, heat_node.grid],
            position=power_node.position,
        )
        network.branch(GenericTransferBranch(), power_node.id, node_id_control)
        network.branch(GenericTransferBranch(), node_id_control, heat_return_node.id)
        network.branch(
            SubHE(Var(0.1), self.diameter_m),
            heat_node.id,
            node_id_control,
            grid=heat_node.grid,
        )


@model
class GasToPower(MultiGridBranchModel):
    """
    No docstring provided.
    """

    def __init__(
        self, efficiency, p_mw_setpoint, q_mvar_setpoint=0, regulation=1
    ) -> None:
        super().__init__()
        self.efficiency = efficiency
        self.el_mw = -p_mw_setpoint
        self.gas_kgps = Var(1)

        self.on_off = 1
        self.p_to_mw = Var(-p_mw_setpoint)
        self.q_to_mvar = -q_mvar_setpoint
        self.from_mass_flow = Var(1)
        self.regulation = regulation

    def loss_percent(self):
        """
        No docstring provided.
        """
        return 1 - self.efficiency

    def equations(self, grids, from_node_model, to_node_model, **kwargs):
        """
        No docstring provided.
        """
        return [
            self.p_to_mw == self.regulation * self.el_mw,
            -self.p_to_mw
            == self.efficiency
            * self.from_mass_flow
            * (3.6 * grids[GasGrid].higher_heating_value),
            -self.el_mw
            == self.efficiency
            * self.gas_kgps
            * (3.6 * grids[GasGrid].higher_heating_value),
        ]


@model
class PowerToGas(MultiGridBranchModel):
    """
    No docstring provided.
    """

    def __init__(
        self, efficiency, mass_flow_setpoint, consume_q_mvar_setpoint=0, regulation=1
    ) -> None:
        super().__init__()
        self.efficiency = efficiency
        self.gas_kgps = -mass_flow_setpoint
        self.el_mw = Var(1.1)

        self.on_off = 1
        self.p_from_mw = Var(1)
        self.q_from_mvar = consume_q_mvar_setpoint
        self.to_mass_flow = Var(self.gas_kgps)
        self.regulation = regulation

    def loss_percent(self):
        """
        No docstring provided.
        """
        return 1 - self.efficiency

    def equations(self, grids, from_node_model, to_node_model, **kwargs):
        """
        No docstring provided.
        """
        return [
            self.to_mass_flow
            == -self.efficiency
            * self.p_from_mw
            * (1 / (grids[GasGrid].higher_heating_value * 3.6)),
            self.p_from_mw > 0,
            self.p_from_mw == self.el_mw * self.regulation,
            self.gas_kgps
            == -self.efficiency
            * self.el_mw
            * (1 / (grids[GasGrid].higher_heating_value * 3.6)),
        ]

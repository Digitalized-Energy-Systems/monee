from abc import ABC, abstractmethod

import numpy as np

import monee.model.phys.nonlinear.hf as ohfmodel

from .core import BranchModel, Intermediate, IntermediateEq, Var, model
from .grid import GasGrid, PowerGrid, WaterGrid


@model
class GenericPowerBranch(BranchModel):
    """
    No docstring provided.
    """

    def __init__(
        self,
        tap,
        shift,
        br_r,
        br_x,
        g_fr,
        b_fr,
        g_to,
        b_to,
        max_i_ka=3.19,
        backup=False,
        on_off=1,
        **kwargs,
    ) -> None:
        """_summary_

        Args:
            tap (_type_): _description_
            shift (_type_): _description_
            br_r (_type_): resistence
            br_x (_type_): reactance
            g_fr (_type_): from conductance
            b_fr (_type_): from susceptance
            g_to (_type_): to conductance
            b_to (_type_): to susceptance
        """
        super().__init__()
        self.tap = tap
        self.shift = shift
        self.br_r = br_r
        self.br_x = br_x
        self.g_fr = g_fr
        self.b_fr = b_fr
        self.g_to = g_to
        self.b_to = b_to
        self.max_i_ka = max_i_ka
        self.backup = backup
        self.on_off = on_off
        self.p_from_mw = Var(1)
        self.q_from_mvar = Var(1)
        self.i_from_ka = Var(1)
        self.loading_from_percent = Var(1)
        self.p_to_mw = Var(1)
        self.q_to_mvar = Var(1)
        self.i_to_ka = Var(1)
        self.loading_to_percent = Var(1)

    @property
    def loading_percent(self):
        """
        No docstring provided.
        """
        return max(self.loading_to_percent.value, self.loading_from_percent.value)

    def loss_percent(self):
        """
        No docstring provided.
        """
        return abs((self.p_from_mw.value - self.p_to_mw.value) / self.p_from_mw.value)

    def equations(self, grid: PowerGrid, from_node_model, to_node_model, **kwargs):
        """
        No docstring provided.
        """
        return [
            self.loading_to_percent == self.i_to_ka / self.max_i_ka,
            self.loading_from_percent == self.i_from_ka / self.max_i_ka,
        ]


@model
class PowerBranch(GenericPowerBranch, ABC):
    """
    No docstring provided.
    """

    def __init__(self, tap, shift, backup=False, on_off=1, **kwargs) -> None:
        super().__init__(
            tap, shift, 0, 0, 0, 0, 0, 0, backup=backup, on_off=on_off, **kwargs
        )
        self.tap = tap
        self.shift = shift
        self.p_from_mw = Var(1)
        self.q_from_mvar = Var(1)
        self.p_to_mw = Var(1)
        self.q_to_mvar = Var(1)

    @abstractmethod
    def calc_r_x(self, grid, from_node_model, to_node_model):
        """
        No docstring provided.
        """

    def equations(self, grid: PowerGrid, from_node_model, to_node_model, **kwargs):
        """
        No docstring provided.
        """
        self.br_r, self.br_x = self.calc_r_x(grid, from_node_model, to_node_model)
        return super().equations(grid, from_node_model, to_node_model, **kwargs)


@model
class PowerLine(PowerBranch):
    """
    No docstring provided.
    """

    def __init__(
        self,
        length_m,
        r_ohm_per_m,
        x_ohm_per_m,
        parallel,
        backup=False,
        on_off=1,
        **kwargs,
    ) -> None:
        super().__init__(1, 0, backup=backup, on_off=on_off, **kwargs)
        self.length_m = length_m
        self.r_ohm_per_m = r_ohm_per_m
        self.x_ohm_per_m = x_ohm_per_m
        self.parallel = parallel

    def calc_r_x(self, grid: PowerGrid, from_node_model, to_node_model):
        """
        No docstring provided.
        """
        base_r = from_node_model.base_kv**2 / grid.sn_mva
        br_r = self.r_ohm_per_m * self.length_m / base_r / self.parallel
        br_x = self.x_ohm_per_m * self.length_m / base_r / self.parallel
        return (br_r, br_x)


@model
class Trafo(PowerBranch):
    """
    No docstring provided.
    """

    def __init__(
        self, vk_percent=12.2, vkr_percent=0.25, sn_trafo_mva=160, shift=0
    ) -> None:
        super().__init__(1, shift)
        self.vk_percent = vk_percent
        self.vkr_percent = vkr_percent
        self.sn_trafo_mva = sn_trafo_mva
        self.vn_trafo_lv = 1

    def calc_r_x(self, grid: PowerGrid, lv_model, hv_model):
        """
        No docstring provided.
        """
        tap_lv = np.square(lv_model.base_kv / hv_model.base_kv) * grid.sn_mva
        z_sc = self.vk_percent / 100.0 / self.sn_trafo_mva * tap_lv
        r_sc = self.vkr_percent / 100.0 / self.sn_trafo_mva * tap_lv
        x_sc = np.sign(z_sc) * np.sqrt((z_sc**2 - r_sc**2).astype(float))
        return (r_sc, x_sc)

    def equations(self, grid: PowerGrid, from_node_model, to_node_model, **kwargs):
        """
        No docstring provided.
        """
        self.tap = 1
        return super().equations(grid, from_node_model, to_node_model, **kwargs)


def sign(v):
    """
    No docstring provided.
    """
    return 1 if v >= 0 else -1


@model
class WaterPipe(BranchModel):
    """
    No docstring provided.
    """

    def __init__(
        self,
        diameter_m,
        length_m,
        temperature_ext_k=283.15,
        roughness=4.5e-05,
        lambda_insulation_w_per_k=0.025,
        insulation_thickness_m=0.12,
        on_off=1,
        friction=None,
    ) -> None:
        super().__init__()
        self.diameter_m = diameter_m
        self.length_m = length_m
        self.temperature_ext_k = temperature_ext_k
        self.roughness = roughness
        self.lambda_insulation_w_per_k = lambda_insulation_w_per_k
        self.insulation_thickness_m = insulation_thickness_m
        self.on_off = on_off
        self.mass_flow = Intermediate(0.1)
        self.mass_flow_pos = Var(0.1, min=0, name="mass_flow_pos")
        self.mass_flow_neg = Var(0.1, min=0, name="mass_flow_neg")
        self.mass_flow_pos_squared = Var(0, min=0, name="mass_flow_pos_sq")
        self.mass_flow_neg_squared = Var(0, min=0, name="mass_flow_neg_sq")
        self.direction = Var(1, integer=True, min=0, max=1, name="direction")
        self.velocity = Var(1, name="velocity")
        self.q_w = Var(1, name="q_w")
        self.reynolds = Var(1000, min=0, name="reynolds")
        self.t_from_pu = Var(1, min=0, max=3, name="t_from_pu")
        self.t_to_pu = Var(1, min=0, max=3, name="t_to_pu")
        self.friction = Var(1, min=0, name="friction") if friction is None else friction

    def loss_percent(self):
        """
        No docstring provided.
        """
        return abs(self.q_w.value) / (
            abs(self.mass_flow.value)
            * ohfmodel.SPECIFIC_HEAT_CAP_WATER
            * self.t_average_k.value
        )

    def equations(self, grid: WaterGrid, from_node_model, to_node_model, **kwargs):
        return [IntermediateEq("mass_flow", self.mass_flow_pos - self.mass_flow_neg)]


@model
class HeatExchanger(BranchModel):
    """
    No docstring provided.
    """

    def __init__(
        self,
        q_mw,
        diameter_m,
        roughness=0.0001,
        length_m=2.5,
        temperature_ext_k=293,
        regulation=1,
        friction=None,
    ) -> None:
        super().__init__()
        self.diameter_m = diameter_m
        self.temperature_ext_k = temperature_ext_k
        self.roughness = roughness
        self.length_m = length_m
        self.limit = 0.1
        self.active = True
        self.regulation = regulation
        self.on_off = 1
        self.q_w_set = -q_mw * 10**6
        self.q_w = Var(-1000, name="q_w")

        self.mass_flow = Intermediate(0.1)
        self.mass_flow_pos = Var(0, min=0, name="mass_flow_pos")
        self.mass_flow_neg = Var(0, min=0, name="mass_flow_neg")
        self.mass_flow_pos_squared = Var(0, min=0, name="mass_flow_pos_sq")
        self.mass_flow_neg_squared = Var(0, min=0, name="mass_flow_neg_sq")
        self.direction = Var(0, integer=True, min=0, max=1, name="direction")
        self.velocity = Var(1, name="velocity")
        self.reynolds = Var(1000, min=0, name="reynolds")
        self.t_from_pu = Var(1, min=0, max=3, name="t_from_pu")
        self.t_to_pu = Var(1, min=0, max=3, name="t_to_pu")
        self.friction = (
            Var(0.01, min=0, name="friction") if friction is None else friction
        )

    def equations(self, grid: WaterGrid, from_node_model, to_node_model, **kwargs):
        return [
            IntermediateEq("mass_flow", self.mass_flow_pos - self.mass_flow_neg),
            self.q_w == self.q_w_set * self.regulation,
        ]


@model
class HeatExchangerLoad(HeatExchanger):
    """
    No docstring provided.
    """

    def __init__(self, q_mw, diameter_m, temperature_ext_k=293) -> None:
        super().__init__(q_mw, diameter_m, temperature_ext_k)


@model
class HeatExchangerGenerator(HeatExchanger):
    """
    No docstring provided.
    """

    def __init__(self, q_mw, diameter_m, temperature_ext_k=293) -> None:
        super().__init__(q_mw, diameter_m, temperature_ext_k)


@model
class GasPipe(BranchModel):
    """
    No docstring provided.
    """

    def __init__(
        self,
        diameter_m,
        length_m,
        temperature_ext_k=296.15,
        roughness=0.0001,
        on_off=1,
        friction=None,
    ) -> None:
        super().__init__()
        self.diameter_m = diameter_m
        self.length_m = length_m
        self.temperature_ext_k = temperature_ext_k
        self.roughness = roughness
        self.on_off = on_off
        self.mass_flow = Intermediate(0.1)
        self.mass_flow_pos = Var(0, min=0, name="mass_flow_pos")
        self.mass_flow_neg = Var(0, min=0, name="mass_flow_neg")
        self.mass_flow_pos_squared = Var(0, min=0, name="mass_flow_pos_sq")
        self.mass_flow_neg_squared = Var(0, min=0, name="mass_flow_neg_sq")
        self.direction = Var(0, integer=True, min=0, max=1)
        self.velocity = Var(1)
        self.reynolds = Var(1000)
        self.gas_density = Var(1)
        self.friction = Var(1) if friction is None else friction
        self.q_w = 0

    def equations(self, grid: GasGrid, from_node_model, to_node_model, **kwargs):
        """
        No docstring provided.
        """
        return [IntermediateEq("mass_flow", self.mass_flow_pos - self.mass_flow_neg)]

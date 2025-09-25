import math
from abc import ABC, abstractmethod

import numpy as np

import monee.model.phys.nl.hydraulics as hydraulicsmodel
import monee.model.phys.nl.ogf as ogfmodel
import monee.model.phys.nl.ohf as ohfmodel
import monee.model.phys.nl.opf as opfmodel
import monee.model.phys.nl.owf as owfmodel

from .core import BranchModel, Var, model
from .grid import GasGrid, PowerGrid, WaterGrid

SQRT_3 = np.sqrt(3)


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
        y = np.linalg.pinv([[self.br_r + self.br_x * 1j]])[0][0]
        g, b = (np.real(y), np.imag(y))
        return (
            opfmodel.int_flow_from_p(
                p_from_var=self.p_from_mw,
                vm_from_var=from_node_model.vars["vm_pu"],
                vm_to_var=to_node_model.vars["vm_pu"],
                va_from_var=from_node_model.vars["va_radians"],
                va_to_var=to_node_model.vars["va_radians"],
                g_branch=g,
                b_branch=b,
                tap=self.tap,
                shift=self.shift,
                cos_impl=kwargs["cos_impl"] if "cos_impl" in kwargs else math.cos,
                sin_impl=kwargs["sin_impl"] if "sin_impl" in kwargs else math.sin,
                g_from=self.g_fr,
                on_off=self.on_off,
            ),
            opfmodel.int_flow_from_q(
                q_from_var=self.q_from_mvar,
                vm_from_var=from_node_model.vars["vm_pu"],
                vm_to_var=to_node_model.vars["vm_pu"],
                va_from_var=from_node_model.vars["va_radians"],
                va_to_var=to_node_model.vars["va_radians"],
                g_branch=g,
                b_branch=b,
                tap=self.tap,
                shift=self.shift,
                cos_impl=kwargs["cos_impl"] if "cos_impl" in kwargs else math.cos,
                sin_impl=kwargs["sin_impl"] if "sin_impl" in kwargs else math.sin,
                b_from=self.b_fr,
                on_off=self.on_off,
            ),
            opfmodel.int_flow_to_p(
                p_to_var=self.p_to_mw,
                vm_from_var=from_node_model.vars["vm_pu"],
                vm_to_var=to_node_model.vars["vm_pu"],
                va_from_var=from_node_model.vars["va_radians"],
                va_to_var=to_node_model.vars["va_radians"],
                g_branch=g,
                b_branch=b,
                tap=self.tap,
                shift=self.shift,
                cos_impl=kwargs["cos_impl"] if "cos_impl" in kwargs else math.cos,
                sin_impl=kwargs["sin_impl"] if "sin_impl" in kwargs else math.sin,
                g_to=self.g_to,
                on_off=self.on_off,
            ),
            opfmodel.int_flow_to_q(
                q_to_var=self.q_to_mvar,
                vm_from_var=from_node_model.vars["vm_pu"],
                vm_to_var=to_node_model.vars["vm_pu"],
                va_from_var=from_node_model.vars["va_radians"],
                va_to_var=to_node_model.vars["va_radians"],
                g_branch=g,
                b_branch=b,
                tap=self.tap,
                shift=self.shift,
                cos_impl=kwargs["cos_impl"] if "cos_impl" in kwargs else math.cos,
                sin_impl=kwargs["sin_impl"] if "sin_impl" in kwargs else math.sin,
                b_to=self.b_to,
                on_off=self.on_off,
            ),
            self.i_from_ka
            == (self.p_from_mw**2 + self.q_from_mvar**2)
            / (from_node_model.vars["vm_pu"] * from_node_model.vars["base_kv"])
            / SQRT_3,
            self.i_to_ka
            == (self.p_to_mw**2 + self.q_to_mvar**2)
            / (to_node_model.vars["vm_pu"] * to_node_model.vars["base_kv"])
            / SQRT_3,
            self.loading_to_percent == self.i_to_ka / self.max_i_ka,
            self.loading_from_percent == self.i_from_ka / self.max_i_ka,
        )


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
    ) -> None:
        super().__init__()
        self.diameter_m = diameter_m
        self.length_m = length_m
        self.temperature_ext_k = temperature_ext_k
        self.pipe_roughness = roughness
        self.lambda_insulation_w_per_k = lambda_insulation_w_per_k
        self.insulation_thickness_m = insulation_thickness_m
        self.on_off = on_off
        self.mass_flow = Var(0.1)
        self.velocity = Var(1)
        self.q_w = Var(1)
        self.reynolds = Var(1000)
        self.t_average_pu = Var(1)
        self.t_from_pu = Var(1)
        self.t_to_pu = Var(1)

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
        """
        No docstring provided.
        """
        self._pipe_area = hydraulicsmodel.calc_pipe_area(self.diameter_m)
        return (
            hydraulicsmodel.reynolds_equation(
                self.reynolds,
                self.mass_flow,
                self.diameter_m,
                grid.dynamic_visc,
                self._pipe_area,
                kwargs["abs_impl"],
            ),
            owfmodel.darcy_weisbach_equation(
                from_node_model.vars["pressure_pu"] * grid.pressure_ref,
                to_node_model.vars["pressure_pu"] * grid.pressure_ref,
                self.reynolds,
                self.velocity,
                self.length_m,
                self.diameter_m,
                grid.fluid_density,
                self.pipe_roughness,
                on_off=self.on_off,
                **kwargs,
            ),
            hydraulicsmodel.flow_rate_equation(
                mean_flow_velocity=self.velocity,
                flow_rate=self.mass_flow,
                diameter=self.diameter_m,
                fluid_density=grid.fluid_density,
            ),
            ohfmodel.heat_transfer_loss(
                heat_transfer_flow_loss_var=self.q_w,
                t_var=self.t_average_pu * grid.t_ref,
                k_insulation_w_per_k=self.lambda_insulation_w_per_k,
                ext_t=self.temperature_ext_k,
                pipe_length=self.length_m,
                pipe_inside_radius=self.diameter_m / 2,
                pipe_outside_radius=self.diameter_m / 2 + self.insulation_thickness_m,
            ),
            ohfmodel.temp_flow(
                t_in_scaled=from_node_model.vars["t_pu"],
                t_out_scaled=to_node_model.vars["t_pu"],
                heat_loss=self.q_w / grid.t_ref,
                mass_flow=self.mass_flow,
                sign_impl=kwargs["sign_impl"],
            ),
            self.t_average_pu
            == (from_node_model.vars["t_pu"] + to_node_model.vars["t_pu"]) / 2,
            self.t_from_pu == from_node_model.vars["t_pu"],
            self.t_to_pu == to_node_model.vars["t_pu"],
        )


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
    ) -> None:
        super().__init__()
        self.diameter_m = diameter_m
        self.temperature_ext_k = temperature_ext_k
        self.pipe_roughness = roughness
        self.length_m = length_m
        self.limit = 0.1
        self.active = True
        self.regulation = regulation
        self.on_off = 1
        self.q_w_set = -q_mw * 10**6
        self.q_w = Var(-1000)
        self.mass_flow = Var(-0.1)
        self.velocity = Var(-0.01)
        self.reynolds = Var(1000)
        self.t_from_pu = Var(1)
        self.t_to_pu = Var(1)
        self.t_average_pu = Var(1)

    def equations(self, grid: WaterGrid, from_node_model, to_node_model, **kwargs):
        """
        No docstring provided.
        """
        self._pipe_area = hydraulicsmodel.calc_pipe_area(self.diameter_m)
        return [
            hydraulicsmodel.reynolds_equation(
                self.reynolds,
                self.mass_flow,
                self.diameter_m,
                grid.dynamic_visc,
                self._pipe_area,
                kwargs["abs_impl"],
            ),
            owfmodel.darcy_weisbach_equation(
                from_node_model.vars["pressure_pu"] * grid.pressure_ref,
                to_node_model.vars["pressure_pu"] * grid.pressure_ref,
                self.reynolds,
                self.velocity,
                self.length_m,
                self.diameter_m,
                grid.fluid_density,
                self.pipe_roughness,
                on_off=self.on_off,
                use_darcy_friction=True,
                **kwargs,
            ),
            hydraulicsmodel.flow_rate_equation(
                mean_flow_velocity=self.velocity,
                flow_rate=self.mass_flow,
                diameter=self.diameter_m,
                fluid_density=grid.fluid_density,
            ),
            ohfmodel.temp_flow(
                t_in_scaled=from_node_model.vars["t_pu"],
                t_out_scaled=to_node_model.vars["t_pu"],
                heat_loss=self.q_w / grid.t_ref,
                mass_flow=self.mass_flow,
                sign_impl=kwargs["sign_impl"],
            ),
            self.t_from_pu == from_node_model.vars["t_pu"],
            self.t_to_pu == to_node_model.vars["t_pu"],
            self.q_w == self.q_w_set * self.regulation,
            self.t_average_pu
            == (from_node_model.vars["t_pu"] + to_node_model.vars["t_pu"]) / 2,
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
        self, diameter_m, length_m, temperature_ext_k=296.15, roughness=0.0001, on_off=1
    ) -> None:
        super().__init__()
        self.diameter_m = diameter_m
        self.length_m = length_m
        self.temperature_ext_k = temperature_ext_k
        self.pipe_roughness = roughness
        self.on_off = on_off
        self.mass_flow = Var(0.1)
        self.velocity = Var(1)
        self.reynolds = Var(1000)
        self.gas_density = Var(1)
        self.q_w = 0

    def equations(self, grid: GasGrid, from_node_model, to_node_model, **kwargs):
        """
        No docstring provided.
        """
        self._pipe_area = hydraulicsmodel.calc_pipe_area(self.diameter_m)
        return (
            hydraulicsmodel.reynolds_equation(
                self.reynolds,
                self.mass_flow,
                self.diameter_m,
                grid.dynamic_visc,
                self._pipe_area,
                kwargs["abs_impl"],
            ),
            ogfmodel.pipe_weymouth(
                from_node_model.vars["pressure_pu"] * grid.pressure_ref,
                to_node_model.vars["pressure_pu"] * grid.pressure_ref,
                f_a=self.mass_flow,
                rey=self.reynolds,
                diameter_m=self.diameter_m,
                roughness=self.pipe_roughness,
                length_m=self.length_m,
                t_k=grid.t_k,
                compressibility=grid.compressibility,
                on_off=self.on_off,
                **kwargs,
            ),
            hydraulicsmodel.flow_rate_equation(
                mean_flow_velocity=self.velocity,
                flow_rate=self.mass_flow,
                diameter=self.diameter_m,
                fluid_density=self.gas_density,
            ),
            self.gas_density
            == grid.pressure_ref
            * (from_node_model.vars["pressure_pu"] + to_node_model.vars["pressure_pu"])
            / 2
            * grid.molar_mass
            / (grid.universal_gas_constant * grid.t_k),
        )

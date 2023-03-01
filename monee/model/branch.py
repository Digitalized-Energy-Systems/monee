import math
from abc import abstractmethod, ABC

import numpy as np

from .core import BranchModel, Var, model
from .grid import WaterGrid, GasGrid, PowerGrid

import monee.model.phys.nl.owf as owfmodel
import monee.model.phys.nl.ohf as ohfmodel
import monee.model.phys.nl.ogf as ogfmodel
import monee.model.phys.nl.opf as opfmodel
import monee.model.phys.nl.hydraulics as hydraulicsmodel
from monee.model.phys.constant import UNIV_GAS_CONST


@model
class GenericPowerBranch(BranchModel):
    def __init__(self, tap, shift, br_r, br_x, g_fr, b_fr, g_to, b_to) -> None:
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

        self.p_from_mw = Var(1000000)
        self.q_from_mvar = Var(1000000)
        self.p_to_mw = Var(1000000)
        self.q_to_mvar = Var(1000000)

    def equations(self, grid: PowerGrid, from_node_model, to_node_model, **kwargs):
        y = np.linalg.pinv([[self.br_r + self.br_x * 1j]])[0][0]
        g, b = np.real(y), np.imag(y)

        return (
            opfmodel.int_flow_from_p(
                p_from_var=self.p_from_mw,
                vm_from_var=from_node_model.vars["vm_pu"],
                vm_to_var=to_node_model.vars["vm_pu"],
                va_from_var=from_node_model.vars["va_degree"],
                va_to_var=to_node_model.vars["va_degree"],
                g_branch=g,
                b_branch=b,
                tap=self.tap,
                shift=self.shift,
                cos_impl=kwargs["cos_impl"] if "cos_impl" in kwargs else math.cos,
                sin_impl=kwargs["sin_impl"] if "sin_impl" in kwargs else math.sin,
                g_from=self.g_fr,
            ),
            opfmodel.int_flow_from_q(
                q_from_var=self.q_from_mvar,
                vm_from_var=from_node_model.vars["vm_pu"],
                vm_to_var=to_node_model.vars["vm_pu"],
                va_from_var=from_node_model.vars["va_degree"],
                va_to_var=to_node_model.vars["va_degree"],
                g_branch=g,
                b_branch=b,
                tap=self.tap,
                shift=self.shift,
                cos_impl=kwargs["cos_impl"] if "cos_impl" in kwargs else math.cos,
                sin_impl=kwargs["sin_impl"] if "sin_impl" in kwargs else math.sin,
                b_from=self.b_fr,
            ),
            opfmodel.int_flow_to_p(
                p_to_var=self.p_to_mw,
                vm_from_var=from_node_model.vars["vm_pu"],
                vm_to_var=to_node_model.vars["vm_pu"],
                va_from_var=from_node_model.vars["va_degree"],
                va_to_var=to_node_model.vars["va_degree"],
                g_branch=g,
                b_branch=b,
                tap=self.tap,
                shift=self.shift,
                cos_impl=kwargs["cos_impl"] if "cos_impl" in kwargs else math.cos,
                sin_impl=kwargs["sin_impl"] if "sin_impl" in kwargs else math.sin,
                g_to=self.g_to,
            ),
            opfmodel.int_flow_to_q(
                q_to_var=self.q_to_mvar,
                vm_from_var=from_node_model.vars["vm_pu"],
                vm_to_var=to_node_model.vars["vm_pu"],
                va_from_var=from_node_model.vars["va_degree"],
                va_to_var=to_node_model.vars["va_degree"],
                g_branch=g,
                b_branch=b,
                tap=self.tap,
                shift=self.shift,
                cos_impl=kwargs["cos_impl"] if "cos_impl" in kwargs else math.cos,
                sin_impl=kwargs["sin_impl"] if "sin_impl" in kwargs else math.sin,
                b_to=self.b_to,
            ),
        )


@model
class PowerBranch(GenericPowerBranch, ABC):
    def __init__(self, tap, shift) -> None:
        super().__init__(tap, shift, 0, 0, 0, 0, 0, 0)

        self.tap = tap
        self.shift = shift

        self.p_from_mw = Var(1)
        self.q_from_mvar = Var(1)
        self.p_to_mw = Var(1)
        self.q_to_mvar = Var(1)

    @abstractmethod
    def calc_r_x(self, grid, from_node_model, to_node_model):
        pass

    def equations(self, grid: PowerGrid, from_node_model, to_node_model, **kwargs):
        self.br_r, self.br_x = self.calc_r_x(grid, from_node_model, to_node_model)

        return super().equations(grid, from_node_model, to_node_model, **kwargs)


@model
class PowerLine(PowerBranch):
    def __init__(self, length_m, r_ohm_per_m, x_ohm_per_m, parallel) -> None:
        super().__init__(1, 0)

        self.length_m = length_m
        self.r_ohm_per_m = r_ohm_per_m
        self.x_ohm_per_m = x_ohm_per_m
        self.parallel = parallel

    def calc_r_x(self, grid: PowerGrid, from_node_model, to_node_model):
        base_r = from_node_model.base_kv**2 / (grid.sn_mva)
        br_r = self.r_ohm_per_m * self.length_m / base_r / self.parallel
        br_x = self.x_ohm_per_m * self.length_m / base_r / self.parallel
        return br_r, br_x


@model
class Trafo(PowerBranch):
    def __init__(
        self, vk_percent, vkr_percent, sn_trafo_mva, vn_trafo_lv, tap, shift
    ) -> None:
        super().__init__(tap, shift)

        self.vk_percent = vk_percent
        self.vkr_percent = vkr_percent
        self.sn_trafo_mva = sn_trafo_mva
        self.vn_trafo_lv = vn_trafo_lv

    def calc_r_x(self, grid: PowerGrid, from_node_model, to_node_model):
        tap_lv = np.square(self.vn_trafo_lv / from_node_model.base_kv) * grid.sn_mva
        z_sc = self.vk_percent / 100.0 / self.sn_trafo_mva * tap_lv
        r_sc = self.vkr_percent / 100.0 / self.sn_trafo_mva * tap_lv
        x_sc = np.sign(z_sc) * np.sqrt((z_sc**2 - r_sc**2).astype(float))
        return r_sc, x_sc


@model
class WaterPipe(BranchModel):
    def __init__(
        self,
        diameter_m,
        length_m,
        temperature_ext_k,
        pipe_roughness,
        lambda_insulation_w_per_k,
        insulation_thickness_m,
    ) -> None:
        super().__init__()
        self.diameter_m = diameter_m
        self.length_m = length_m
        self.temperature_ext_k = temperature_ext_k
        self.pipe_roughness = pipe_roughness
        self.lambda_insulation_w_per_k = lambda_insulation_w_per_k
        self.insulation_thickness_m = insulation_thickness_m

        self.mass_flow = Var(1)
        self.velocity = Var(1)
        self.heat_loss = Var(1)
        self.heat_flow = Var(1)
        self.reynolds = Var(1000)
        self.heat_transfer_coefficient = Var(0.1)

    def equations(self, grid: WaterGrid, from_node_model, to_node_model, **kwargs):
        self._nikurdse = hydraulicsmodel.calc_nikurdse(
            self.diameter_m, self.pipe_roughness
        )
        self._pipe_area = hydraulicsmodel.calc_pipe_area(self.diameter_m)

        return (
            hydraulicsmodel.reynolds_equation(
                self.reynolds,
                self.mass_flow,
                self.diameter_m,
                grid.dynamic_visc,
                self._pipe_area,
            ),
            owfmodel.darcy_weisbach_equation(
                from_node_model.vars["pressure_pa"],
                to_node_model.vars["pressure_pa"],
                self.reynolds,
                self.velocity,
                self._nikurdse,
                self.length_m,
                self.diameter_m,
                grid.fluid_density,
            ),
            hydraulicsmodel.flow_rate_equation(
                mean_flow_velocity=self.velocity,
                flow_rate=self.mass_flow,
                diameter=self.diameter_m,
            ),
            ohfmodel.heat_transfer_loss(
                heat_transfer_flow_loss_var=self.heat_loss,
                t_var=(from_node_model.vars["t_k"] + to_node_model.vars["t_k"]) / 2,
                alpha_insulation=self.lambda_insulation_w_per_k,
                ext_t=self.temperature_ext_k,
                pipe_length=self.length_m,
                pipe_inside_diameter=self.diameter_m,
                pipe_outside_diameter=self.diameter_m + self.insulation_thickness_m,
            ),
            ohfmodel.heat_transfer_pipe(
                heat_transfer_var=self.heat_flow,
                heat_transfer_flow_loss_var=self.heat_loss,
                t_1_var=from_node_model.vars["t_k"],
                t_2_var=to_node_model.vars["t_k"],
                heat_transfer_coefficient_var=self.heat_transfer_coefficient,
                diameter=self.diameter_m,
            ),
            ohfmodel.heat_transfer_coefficient_inside_pipe(
                self.heat_transfer_coefficient,
                self.reynolds,
                from_node_model.vars["t_k"],
                to_node_model.vars["t_k"],
                self.diameter_m,
            ),
        )


@model
class GasPipe(BranchModel):
    def __init__(
        self,
        diameter_m,
        length_m,
        temperature_ext_k,
        pipe_roughness,
    ) -> None:
        super().__init__()

        self.diameter_m = diameter_m
        self.length_m = length_m
        self.temperature_ext_k = temperature_ext_k
        self.pipe_roughness = pipe_roughness

        self.mass_flow = (Var(1),)
        self.velocity = (Var(1),)
        self.reynolds = (Var(1000),)

    def equations(self, grid: GasGrid, from_node_model, to_node_model, **kwargs):
        self._nikurdse = hydraulicsmodel.calc_nikurdse(
            self.diameter_m, self.pipe_roughness
        )
        self._pipe_area = hydraulicsmodel.calc_pipe_area(self.diameter_m)

        a_0 = ogfmodel.calc_a(
            z=grid.compressibility,
            r=UNIV_GAS_CONST,
            t=grid.gas_temperature,
            m=grid.molar_mass,
        )
        w = ogfmodel.calc_w(
            pipe_length=self.length_m,
            diameter=self.diameter_m,
            mass_flow_zero=1,
            pressure_zero=1,
            a=a_0,
            pipe_area=self._pipe_area,
        )
        return (
            hydraulicsmodel.reynolds_equation(
                self.reynolds,
                self.mass_flow,
                self.diameter_m,
                grid.dynamic_visc,
                self._pipe_area,
            ),
            ogfmodel.pipe_weymouth(
                from_node_model.vars["pressure_pa"],
                to_node_model.vars["pressure_pa"],
                w=w,
                f_a=self.mass_flow,
                rey=self.reynolds,
                nikurdse=self._nikurdse,
            ),
            hydraulicsmodel.flow_rate_equation(
                mean_flow_velocity=self.velocity,
                flow_rate=self.mass_flow,
                diameter=self.diameter_m,
            ),
        )
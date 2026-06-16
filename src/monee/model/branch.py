from abc import ABC, abstractmethod

import numpy as np

import monee.model.phys.nonlinear.hf as ohfmodel

from .core import BranchModel, Intermediate, IntermediateEq, Var, model
from .grid import GasGrid, PowerGrid, WaterGrid


@model
class GenericPowerBranch(BranchModel):
    def __init__(
        self,
        tap,
        shift,
        br_r_pu,
        br_x_pu,
        g_fr_pu,
        b_fr_pu,
        g_to_pu,
        b_to_pu,
        max_i_ka=3.19,
        backup=False,
        on_off=1,
        **kwargs,
    ) -> None:
        super().__init__()
        self.tap = tap
        self.shift = shift
        self.br_r_pu = br_r_pu
        self.br_x_pu = br_x_pu
        self.g_fr_pu = g_fr_pu
        self.b_fr_pu = b_fr_pu
        self.g_to_pu = g_to_pu
        self.b_to_pu = b_to_pu
        self.max_i_ka = max_i_ka
        self.backup = backup
        self.on_off = on_off
        self.p_from_mw = Var(1, name="p_from_mw")
        self.q_from_mvar = Var(1, name="q_from_mvar")
        self.i_from_ka = Var(1, min=0, name="i_from_ka")
        self.loading_from_pu = Var(1, min=0, name="loading_from_pu")
        self.p_to_mw = Var(1, name="p_to_mw")
        self.q_to_mvar = Var(1, name="q_to_mvar")
        self.i_to_ka = Var(1, min=0, name="i_to_ka")
        self.loading_to_pu = Var(1, min=0, name="loading_to_pu")

    @property
    def loading_pu(self):
        return max(self.loading_to_pu.value, self.loading_from_pu.value)

    def loss_percent(self):
        return abs((self.p_from_mw.value - self.p_to_mw.value) / self.p_from_mw.value)

    def equations(self, grid: PowerGrid, from_node_model, to_node_model, **kwargs):
        # loading_*_percent \leftrightarrow i_*_ka identity is owned by the branch formulation
        # (AC: equality; MISOCP: derived from current_pu_squared post-solve).
        return []


@model
class PowerBranch(GenericPowerBranch, ABC):
    def __init__(self, tap, shift, backup=False, on_off=1, **kwargs) -> None:
        super().__init__(
            tap, shift, 0, 0, 0, 0, 0, 0, backup=backup, on_off=on_off, **kwargs
        )
        self.tap = tap
        self.shift = shift
        self.p_from_mw = Var(1, name="p_from_mw")
        self.q_from_mvar = Var(1, name="q_from_mvar")
        self.p_to_mw = Var(1, name="p_to_mw")
        self.q_to_mvar = Var(1, name="q_to_mvar")

    @abstractmethod
    def calc_r_x(self, grid, from_node_model, to_node_model):
        pass

    def equations(self, grid: PowerGrid, from_node_model, to_node_model, **kwargs):
        self.br_r_pu, self.br_x_pu = self.calc_r_x(grid, from_node_model, to_node_model)
        return super().equations(grid, from_node_model, to_node_model, **kwargs)


@model
class PowerLine(PowerBranch):
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
        base_r = from_node_model.base_kv**2 / grid.sn_mva
        br_r_pu = self.r_ohm_per_m * self.length_m / base_r / self.parallel
        br_x_pu = self.x_ohm_per_m * self.length_m / base_r / self.parallel
        return (br_r_pu, br_x_pu)


@model
class Trafo(PowerBranch):
    def __init__(
        self, vk_percent=12.2, vkr_percent=0.25, sn_trafo_mva=160, shift=0
    ) -> None:
        super().__init__(1, shift)
        self.vk_percent = vk_percent
        self.vkr_percent = vkr_percent
        self.sn_trafo_mva = sn_trafo_mva
        self.vn_trafo_lv = 1

    def calc_r_x(self, grid: PowerGrid, lv_model, hv_model):
        tap_lv = np.square(lv_model.base_kv / hv_model.base_kv) * grid.sn_mva
        z_sc = self.vk_percent / 100.0 / self.sn_trafo_mva * tap_lv
        r_sc = self.vkr_percent / 100.0 / self.sn_trafo_mva * tap_lv
        x_sc = np.sign(z_sc) * np.sqrt((z_sc**2 - r_sc**2).astype(float))
        return (r_sc, x_sc)

    def equations(self, grid: PowerGrid, from_node_model, to_node_model, **kwargs):
        self.tap = 1
        return super().equations(grid, from_node_model, to_node_model, **kwargs)


def sign(v):
    return 1 if v >= 0 else -1


@model
class WaterPipe(BranchModel):
    def __init__(
        self,
        diameter_m,
        length_m,
        temperature_ext_k=283.15,
        roughness_m=4.5e-05,
        lambda_insulation_w_per_m_k=0.025,
        insulation_thickness_m=0.12,
        on_off=1,
        friction=None,
        unidirectional=False,
    ) -> None:
        super().__init__()
        self.diameter_m = diameter_m
        self.length_m = length_m
        self.temperature_ext_k = temperature_ext_k
        self.roughness_m = roughness_m
        self.lambda_insulation_w_per_m_k = lambda_insulation_w_per_m_k
        self.insulation_thickness_m = insulation_thickness_m
        self.on_off = on_off
        self.unidirectional = unidirectional
        self.mass_flow_kgs = Intermediate(0.1)
        self.mass_flow_pos_kgs = Var(0.1, min=0, name="mass_flow_pos_kgs")
        self.mass_flow_neg_kgs = Var(0.1, min=0, name="mass_flow_neg_kgs")
        self.mass_flow_pos_kgs_squared = Var(0, min=0, name="mass_flow_pos_kgs_squared")
        self.mass_flow_neg_kgs_squared = Var(0, min=0, name="mass_flow_neg_kgs_squared")
        self.direction = Var(1, integer=True, min=0, max=1, name="direction")
        self.velocity_mps = Var(1, min=-50, max=50, name="velocity_mps")
        self.q_mw = Var(1e-6, name="q_mw")
        # reynolds_scaled is stored as Re/1e6 (see REYNOLDS_SCALE); 1e-3 \approx laminar floor.
        self.reynolds_scaled = Var(1e-3, min=0, max=10, name="reynolds_scaled")
        self.t_from_pu = Var(1, min=0, max=2, name="t_from_pu")
        self.t_to_pu = Var(1, min=0, max=2, name="t_to_pu")
        # friction upper bound 7 covers the PWL leftmost breakpoint (Re=10, 64/10).
        self.friction = (
            Var(0.02, min=0, max=7, name="friction") if friction is None else friction
        )

    def loss_percent(self):
        mass_flow_kgs = abs(self.mass_flow_kgs.value)
        if mass_flow_kgs == 0:
            return 0
        # Average fluid temperature in Kelvin from the per-unit endpoint temps
        # (WaterGrid.t_ref_k is the per-unit reference; the branch has no grid here).
        t_average_k = (
            (self.t_from_pu.value + self.t_to_pu.value) / 2 * WaterGrid.t_ref_k
        )
        return (
            abs(self.q_mw.value)
            * 1e6
            / (mass_flow_kgs * ohfmodel.SPECIFIC_HEAT_CAP_WATER * t_average_k)
        )

    def equations(self, grid: WaterGrid, from_node_model, to_node_model, **kwargs):
        return [
            IntermediateEq(
                "mass_flow_kgs", self.mass_flow_pos_kgs - self.mass_flow_neg_kgs
            )
        ]


@model
class HeatExchanger(BranchModel):
    def __init__(
        self,
        q_mw,
        mass_flow_design_kgs=None,
        T_delta_design_K=30,
        regulation=1,
    ) -> None:
        super().__init__()
        self._calc_mass_flow = False
        self._T_delta_design_K = T_delta_design_K

        self.on_off = 1
        self.regulation = regulation
        self.q_mw_set = -q_mw
        self.q_mw = Var(0, name="q_mw")

        if mass_flow_design_kgs is None:
            if isinstance(q_mw, (int, float)):
                mass_flow_design_kgs = abs(q_mw * 1e6) / (
                    ohfmodel.SPECIFIC_HEAT_CAP_WATER * T_delta_design_K
                )
            else:
                mass_flow_design_kgs = Var(0, name="mass_flow_design_kgs")
                self._calc_mass_flow = True

        self.mass_flow_design_kgs = mass_flow_design_kgs
        self.mass_flow_kgs = Intermediate(0.1)
        self.mass_flow_pos_kgs = Var(0, min=0, name="mass_flow_pos_kgs")
        self.mass_flow_neg_kgs = Var(0, min=0, name="mass_flow_neg_kgs")
        self.direction = Var(0, integer=True, min=0, max=1, name="direction")
        self.t_from_pu = Var(1, min=0, max=2, name="t_from_pu")
        self.t_to_pu = Var(1, min=0, max=2, name="t_to_pu")

    def equations(self, grid: WaterGrid, from_node_model, to_node_model, **kwargs):
        eqs = [
            IntermediateEq(
                "mass_flow_kgs", self.mass_flow_pos_kgs - self.mass_flow_neg_kgs
            ),
        ]
        if self._calc_mass_flow:
            eqs.append(
                self.mass_flow_design_kgs
                == -self.q_mw
                * 1e6
                / (ohfmodel.SPECIFIC_HEAT_CAP_WATER * self._T_delta_design_K)
            )
        else:
            eqs.append(self.q_mw == self.q_mw_set * self.regulation)
        return eqs


@model
class HeatExchangerLoad(HeatExchanger):
    def __init__(self, q_mw, mass_flow_design_kgs=None, regulation=1) -> None:
        super().__init__(
            q_mw, mass_flow_design_kgs=mass_flow_design_kgs, regulation=regulation
        )


@model
class HeatExchangerGenerator(HeatExchanger):
    def __init__(self, q_mw, mass_flow_design_kgs=None, regulation=1) -> None:
        super().__init__(
            q_mw, mass_flow_design_kgs=mass_flow_design_kgs, regulation=regulation
        )


@model
class PassiveHeatExchanger(BranchModel):
    r"""
    Passive heat exchanger injecting/extracting fixed ``q_mw`` into a free-flowing
    water branch. Mass flow is determined by surrounding hydraulics; temperature
    change follows from q_mw and actual mass flow.

    Sign: positive q_mw = load, negative = generator.

    Hydraulics: by default the pressure drop is Darcy-Weisbach friction over
    ``length_m``. Passing ``loss_coefficient`` (zeta) instead switches to the
    pandapipes ``heat_exchanger`` model - a zero-length minor loss
    :math:`\Delta p = \zeta \cdot \tfrac{\rho}{2} v^2` (``length_m`` / friction
    are then ignored; ``loss_coefficient=0`` is a lossless heat injector).
    """

    def __init__(
        self,
        q_mw,
        diameter_m,
        roughness_m=0.0001,
        length_m=2.5,
        temperature_ext_k=293,
        regulation=1,
        friction=None,
        loss_coefficient=None,
    ) -> None:
        super().__init__()
        self.diameter_m = diameter_m
        self.temperature_ext_k = temperature_ext_k
        self.roughness_m = roughness_m
        self.length_m = length_m
        self.loss_coefficient = loss_coefficient
        self.limit = 0.1
        self.regulation = regulation
        self.on_off = 1
        self.q_mw_set = -q_mw
        self.q_mw = Var(-1e-3, name="q_mw")

        self.mass_flow_kgs = Intermediate(0.1)
        self.mass_flow_pos_kgs = Var(0, min=0, name="mass_flow_pos_kgs")
        self.mass_flow_neg_kgs = Var(0, min=0, name="mass_flow_neg_kgs")
        self.mass_flow_pos_kgs_squared = Var(0, min=0, name="mass_flow_pos_kgs_squared")
        self.mass_flow_neg_kgs_squared = Var(0, min=0, name="mass_flow_neg_kgs_squared")
        self.direction = Var(0, integer=True, min=0, max=1, name="direction")
        self.velocity_mps = Var(1, min=-50, max=50, name="velocity_mps")
        # reynolds_scaled = Re/1e6 (see REYNOLDS_SCALE); 1e-3 \approx laminar floor.
        self.reynolds_scaled = Var(1e-3, min=0, max=10, name="reynolds_scaled")
        self.t_from_pu = Var(1, min=0, max=2, name="t_from_pu")
        self.t_to_pu = Var(1, min=0, max=2, name="t_to_pu")
        # friction upper bound 7 covers the PWL leftmost breakpoint (Re=10).
        self.friction = (
            Var(0.01, min=0, max=7, name="friction") if friction is None else friction
        )

    def equations(self, grid: WaterGrid, from_node_model, to_node_model, **kwargs):
        return [
            IntermediateEq(
                "mass_flow_kgs", self.mass_flow_pos_kgs - self.mass_flow_neg_kgs
            ),
            self.q_mw == self.q_mw_set * self.regulation,
        ]


@model
class PassiveHeatExchangerLoad(PassiveHeatExchanger):
    """Passive heat exchanger that consumes heat (load, ``q_mw > 0``)."""

    def __init__(
        self, q_mw, diameter_m, temperature_ext_k=293, loss_coefficient=None
    ) -> None:
        super().__init__(
            q_mw,
            diameter_m,
            temperature_ext_k=temperature_ext_k,
            loss_coefficient=loss_coefficient,
        )


@model
class PassiveHeatExchangerGenerator(PassiveHeatExchanger):
    """Passive heat exchanger that injects heat (generator, ``q_mw < 0``)."""

    def __init__(
        self, q_mw, diameter_m, temperature_ext_k=293, loss_coefficient=None
    ) -> None:
        super().__init__(
            q_mw,
            diameter_m,
            temperature_ext_k=temperature_ext_k,
            loss_coefficient=loss_coefficient,
        )


@model
class GasPipe(BranchModel):
    def __init__(
        self,
        diameter_m,
        length_m,
        temperature_ext_k=296.15,
        roughness_m=0.0001,
        on_off=1,
        friction=None,
    ) -> None:
        super().__init__()
        self.diameter_m = diameter_m
        self.length_m = length_m
        self.temperature_ext_k = temperature_ext_k
        self.roughness_m = roughness_m
        self.on_off = on_off
        self.mass_flow_kgs = Intermediate(0.1)
        self.mass_flow_pos_kgs = Var(0, min=0, name="mass_flow_pos_kgs")
        self.mass_flow_neg_kgs = Var(0, min=0, name="mass_flow_neg_kgs")
        self.mass_flow_pos_kgs_squared = Var(0, min=0, name="mass_flow_pos_kgs_squared")
        self.mass_flow_neg_kgs_squared = Var(0, min=0, name="mass_flow_neg_kgs_squared")
        self.direction = Var(0, integer=True, min=0, max=1)
        self.velocity_mps = Var(1, min=-100, max=100, name="velocity_mps")
        # reynolds_scaled = Re/1e6 (see REYNOLDS_SCALE); 1e-3 \approx laminar floor.
        self.reynolds_scaled = Var(1e-3, min=0, max=10, name="reynolds_scaled")
        self.gas_density_kg_per_m3 = Var(
            1, min=0, max=100, name="gas_density_kg_per_m3"
        )
        self.friction = (
            Var(0.02, min=0, max=7, name="friction") if friction is None else friction
        )
        self.q_mw = 0

    def equations(self, grid: GasGrid, from_node_model, to_node_model, **kwargs):
        return [
            IntermediateEq(
                "mass_flow_kgs", self.mass_flow_pos_kgs - self.mass_flow_neg_kgs
            )
        ]


@model
class GasCompressor(BranchModel):
    """
    Ideal compressor - fixed pressure ratio, unidirectional (suction → discharge).
    Forward flow lives in ``mass_flow_neg_kgs`` to match GasPipe's Weymouth convention.
    """

    def __init__(self, compression_ratio=1.5, max_flow_kgs=10.0) -> None:
        super().__init__()
        self.compression_ratio = compression_ratio
        self.max_flow_kgs = max_flow_kgs
        self.mass_flow_kgs = Intermediate(0.1)
        self.mass_flow_neg_kgs = Var(
            0.1, min=0, max=max_flow_kgs, name="mass_flow_neg_kgs"
        )
        self.on_off = 1

    def equations(self, grid: GasGrid, from_node_model, to_node_model, **kwargs):
        p_sq_from = from_node_model.vars["pressure_squared_pu"]
        p_sq_to = to_node_model.vars["pressure_squared_pu"]
        return [
            IntermediateEq("mass_flow_kgs", -self.mass_flow_neg_kgs),
            self.compression_ratio**2 * p_sq_from == p_sq_to,
        ]

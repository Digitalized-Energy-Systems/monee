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
        return max(self.loading_to_percent.value, self.loading_from_percent.value)

    def loss_percent(self):
        return abs((self.p_from_mw.value - self.p_to_mw.value) / self.p_from_mw.value)

    def equations(self, grid: PowerGrid, from_node_model, to_node_model, **kwargs):
        return [
            self.loading_to_percent == self.i_to_ka / self.max_i_ka,
            self.loading_from_percent == self.i_from_ka / self.max_i_ka,
        ]


@model
class PowerBranch(GenericPowerBranch, ABC):
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
        pass

    def equations(self, grid: PowerGrid, from_node_model, to_node_model, **kwargs):
        self.br_r, self.br_x = self.calc_r_x(grid, from_node_model, to_node_model)
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
        br_r = self.r_ohm_per_m * self.length_m / base_r / self.parallel
        br_x = self.x_ohm_per_m * self.length_m / base_r / self.parallel
        return (br_r, br_x)


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
        self.reynolds = Var(1000, min=0, max=1000000, name="reynolds")
        self.t_from_pu = Var(1, min=0, max=3, name="t_from_pu")
        self.t_to_pu = Var(1, min=0, max=3, name="t_to_pu")
        self.friction = (
            Var(0.2, min=0, max=640000, name="friction")
            if friction is None
            else friction
        )

    def loss_percent(self):
        return abs(self.q_w.value) / (
            abs(self.mass_flow.value)
            * ohfmodel.SPECIFIC_HEAT_CAP_WATER
            * self.t_average_k.value
        )

    def equations(self, grid: WaterGrid, from_node_model, to_node_model, **kwargs):
        return [IntermediateEq("mass_flow", self.mass_flow_pos - self.mass_flow_neg)]


@model
class HeatExchanger(BranchModel):
    def __init__(
        self,
        q_mw,
        diameter_m,
        roughness=0.0001,
        length_m=2.5,
        temperature_ext_k=293,
        regulation=1,
        friction=None,
        mass_flow_design_kgs=None,
        T_delta_design_K=30,
    ) -> None:
        super().__init__()
        self._calc_mass_flow = False
        self._T_delta_design_K = T_delta_design_K

        self.diameter_m = diameter_m
        self.temperature_ext_k = temperature_ext_k
        self.roughness = roughness
        self.length_m = length_m
        self.limit = 0.1
        self.active = True
        self.regulation = regulation
        self.on_off = 1
        self.q_w_set = -q_mw * 10**6
        self.q_w = Var(0, name="q_w")

        if mass_flow_design_kgs is None:
            if isinstance(q_mw, (int, float)):
                mass_flow_design_kgs = abs(q_mw * 10**6) / (
                    ohfmodel.SPECIFIC_HEAT_CAP_WATER * T_delta_design_K
                )
            else:
                mass_flow_design_kgs = Var(0, name="mass_flow_design_kgs")
                self._calc_mass_flow = True

        self.mass_flow_design_kgs = mass_flow_design_kgs
        self.mass_flow = Intermediate(0.1)
        self.mass_flow_pos = Var(0, min=0, name="mass_flow_pos")
        self.mass_flow_neg = Var(0, min=0, name="mass_flow_neg")
        self.mass_flow_pos_squared = Var(0, min=0, name="mass_flow_pos_sq")
        self.mass_flow_neg_squared = Var(0, min=0, name="mass_flow_neg_sq")
        self.direction = Var(0, integer=True, min=0, max=1, name="direction")
        self.velocity = Var(1, name="velocity")
        self.reynolds = Var(1000, min=0, max=1000000, name="reynolds")
        self.t_from_pu = Var(1, min=0, max=3, name="t_from_pu")
        self.t_to_pu = Var(1, min=0, max=3, name="t_to_pu")
        self.friction = (
            Var(0.01, min=0, max=1, name="friction") if friction is None else friction
        )

    def equations(self, grid: WaterGrid, from_node_model, to_node_model, **kwargs):
        eqs = [
            IntermediateEq("mass_flow", self.mass_flow_pos - self.mass_flow_neg),
        ]
        if self._calc_mass_flow:
            eqs.append(
                self.mass_flow_design_kgs
                == -self.q_w
                / (ohfmodel.SPECIFIC_HEAT_CAP_WATER * self._T_delta_design_K)
            )
        else:
            eqs.append(self.q_w == self.q_w_set * self.regulation)
        return eqs


@model
class HeatExchangerLoad(HeatExchanger):
    def __init__(
        self, q_mw, diameter_m, temperature_ext_k=293, mass_flow_design_kgs=None
    ) -> None:
        super().__init__(
            q_mw,
            diameter_m,
            temperature_ext_k=temperature_ext_k,
            mass_flow_design_kgs=mass_flow_design_kgs,
        )


@model
class HeatExchangerGenerator(HeatExchanger):
    def __init__(
        self, q_mw, diameter_m, temperature_ext_k=293, mass_flow_design_kgs=None
    ) -> None:
        super().__init__(
            q_mw,
            diameter_m,
            temperature_ext_k=temperature_ext_k,
            mass_flow_design_kgs=mass_flow_design_kgs,
        )


@model
class PassiveHeatExchanger(BranchModel):
    """
    Passive heat exchanger: injects or extracts a fixed heat power ``q_mw`` into
    the water flow that passes through it.  The mass flow is *not* prescribed —
    it is determined by the surrounding network hydraulics.  The formulation then
    computes the resulting temperature increase (or decrease) from the heat power
    and the actual mass flow.

    Use :class:`PassiveHeatExchangerLoad` / :class:`PassiveHeatExchangerGenerator`
    for the load/generator convenience sub-classes.

    Args:
        q_mw: Heat power in MW.  Positive = heat consumed (load),
              negative = heat injected (generator).
        diameter_m: Inner pipe diameter [m].
        roughness: Pipe wall roughness [m] (default 0.0001).
        length_m: Equivalent pipe length for pressure-drop calc [m] (default 2.5).
        temperature_ext_k: Ambient temperature [K] (default 293).
        regulation: Scaling factor applied to ``q_w_set`` (default 1).
        friction: Pre-computed friction variable (optional).
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
        self.reynolds = Var(1000, min=0, max=1000000, name="reynolds")
        self.t_from_pu = Var(1, min=0, max=3, name="t_from_pu")
        self.t_to_pu = Var(1, min=0, max=3, name="t_to_pu")
        self.friction = (
            Var(0.01, min=0, max=1, name="friction") if friction is None else friction
        )

    def equations(self, grid: WaterGrid, from_node_model, to_node_model, **kwargs):
        return [
            IntermediateEq("mass_flow", self.mass_flow_pos - self.mass_flow_neg),
            self.q_w == self.q_w_set * self.regulation,
        ]


@model
class PassiveHeatExchangerLoad(PassiveHeatExchanger):
    """Passive heat exchanger that consumes heat (load, ``q_mw > 0``)."""

    def __init__(self, q_mw, diameter_m, temperature_ext_k=293) -> None:
        super().__init__(q_mw, diameter_m, temperature_ext_k=temperature_ext_k)


@model
class PassiveHeatExchangerGenerator(PassiveHeatExchanger):
    """Passive heat exchanger that injects heat (generator, ``q_mw < 0``)."""

    def __init__(self, q_mw, diameter_m, temperature_ext_k=293) -> None:
        super().__init__(q_mw, diameter_m, temperature_ext_k=temperature_ext_k)


@model
class GasPipe(BranchModel):
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
        self.reynolds = Var(1000, min=0, max=1000000)
        self.gas_density = Var(1)
        self.friction = Var(1) if friction is None else friction
        self.q_w = 0

    def equations(self, grid: GasGrid, from_node_model, to_node_model, **kwargs):
        return [IntermediateEq("mass_flow", self.mass_flow_pos - self.mass_flow_neg)]


@model
class GasCompressor(BranchModel):
    """
    Ideal gas compressor — raises pressure from suction junction to discharge junction
    by a fixed compression ratio.

    The pressure boost equation uses the same first-order linearisation around
    ``grid.nominal_pressure_pu`` as the Weymouth pipe formulation, keeping the
    overall system linear.  Mass flow is strictly unidirectional (suction →
    discharge); no ``mass_flow_neg`` variable is needed.

    Args:
        compression_ratio (float): Desired outlet/inlet pressure ratio (≥ 1).
        max_flow_kgs (float): Upper bound on mass throughput in kg/s.
    """

    def __init__(self, compression_ratio=1.5, max_flow_kgs=10.0) -> None:
        super().__init__()
        self.compression_ratio = compression_ratio
        self.max_flow_kgs = max_flow_kgs
        self.mass_flow = Intermediate(0.1)
        # Gas convention: forward physical flow (suction→discharge) uses mass_flow_neg,
        # matching the sign convention of GasPipe (Weymouth uses mf_neg for forward flow).
        self.mass_flow_neg = Var(0.1, min=0, max=max_flow_kgs, name="mass_flow_neg")
        self.on_off = 1

    def equations(self, grid: GasGrid, from_node_model, to_node_model, **kwargs):
        p_sq_from = from_node_model.vars["pressure_squared_pu"]
        p_sq_to = to_node_model.vars["pressure_squared_pu"]
        return [
            IntermediateEq("mass_flow", -self.mass_flow_neg),
            # p_to² = ratio² · p_from²  (linear when ratio is a fixed scalar)
            self.compression_ratio**2 * p_sq_from == p_sq_to,
        ]

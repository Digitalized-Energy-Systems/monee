from abc import ABC, abstractmethod

import monee.model as md
from monee.model.grid import KGPS_KWHPERKG_TO_MW
from monee.problem.utils import cp_input_rated_mw

# Matches _SERVED_ABS_TOL in monee.solver.core: 1e-4 MW (0.1 kW) covers the
# O(1e-6..1e-10) constraint noise IPOPT leaves on served loads.
CURTAILMENT_NOISE_FLOOR_MW = 1e-4


def _clip_noise(curtailed: float) -> float:
    if -CURTAILMENT_NOISE_FLOOR_MW < curtailed < 0:
        return 0.0
    return curtailed


class PerformanceMetric(ABC):
    @abstractmethod
    def calc(self, network: md.Network):
        pass


class ResilienceMetric(ABC):
    @abstractmethod
    def gather(self, network: md.Network, step, **kwargs):
        pass

    @abstractmethod
    def calc(self):
        pass


def is_load(component):
    model = component.model
    grid = component.grid
    return (
        isinstance(model, (md.PowerLoad, md.HeatLoad))
        or (isinstance(model, md.Sink) and isinstance(grid, md.GasGrid))
        # Sign-based, not subclass-based: a bare (Passive)HeatExchanger with a
        # consuming setpoint is sheddable (see controllable_demands) and must
        # count here too.
        or md.hx_is_consuming(model)
        or isinstance(model, md.ExtPowerGrid)
        or (isinstance(model, md.ExtHydrGrid) and isinstance(grid, md.GasGrid))
    )


class GeneralResiliencePerformanceMetric(PerformanceMetric):
    def get_relevant_components(self, network: md.Network):
        return [
            component
            for component in network.childs + network.branches
            if is_load(component)
        ]

    def get_coupling_point_components(self, network: md.Network):
        """All coupling-point components: control-node CPs live as compound
        subnodes (in ``network.nodes``) and HG-variant CPs live as branches."""
        return [
            component
            for component in network.nodes + network.branches
            if cp_input_rated_mw(component) is not None
        ]

    def calc(  # NOSONAR
        self, network, inv=False, include_ext_grid=False, include_coupling_points=False
    ):
        """Curtailed demand of the solved *network* as ``(power, heat, gas)`` in MW.

        Ignored or inactive loads count at their full rating, regulated loads at
        ``upper - value * regulation``, and a heat exchanger at the gap between
        its setpoint and the duty it reached. Components in
        ``(-CURTAILMENT_NOISE_FLOOR_MW, 0)`` are clipped to ``0.0`` (solver
        noise); a negative value at or below the floor passes through, since it
        signals a real accounting problem rather than tolerance noise.

        ``include_ext_grid=True`` additionally counts every external grid that
        *feeds in* as curtailed demand: the islanding view, where import is the
        power an islanded network would have had to shed. On a grid-connected
        network that adds the whole substation import to the power component,
        so leave it False unless you are ranking islanding scenarios.
        ``include_coupling_points=True`` adds coupling-point curtailment on the
        input carrier, mirroring the option of the load-shedding problem.
        ``inv=True`` returns the three components negated.
        """
        relevant_components = self.get_relevant_components(network)
        power_load_curtailed = 0
        heat_load_curtailed = 0
        gas_load_curtailed = 0
        for component in relevant_components:
            model = component.model
            if component.ignored or not component.active:
                if isinstance(model, md.PowerLoad):
                    power_load_curtailed += md.upper(model.p_mw)
                if isinstance(model, md.Sink):
                    gas_load_curtailed += (
                        md.upper(model.mass_flow_kgs)
                        * KGPS_KWHPERKG_TO_MW
                        * component.grid.higher_heating_value_kwh_per_kg
                    )
                if md.hx_is_consuming(model):
                    # q_mw itself is a max-less Var, so md.upper would fall
                    # back to its (post-solve) value and misreport curtailment;
                    # use the numeric setpoint instead.
                    heat_load_curtailed += model.q_mw_set
                if isinstance(model, md.HeatLoad):
                    heat_load_curtailed += md.upper(model.q_mw_heat)
                continue
            if isinstance(model, md.ExtHydrGrid) and include_ext_grid:
                # Only count when ext grid feeds in (load would need shedding).
                if md.value(model.mass_flow_kgs) < 0:
                    gas_load_curtailed += (
                        -md.value(model.mass_flow_kgs)
                        * KGPS_KWHPERKG_TO_MW
                        * component.grid.higher_heating_value_kwh_per_kg
                    )
            if isinstance(model, md.ExtPowerGrid) and include_ext_grid:
                if md.value(model.p_mw) < 0:
                    power_load_curtailed += -md.value(model.p_mw)
            if isinstance(model, md.PowerLoad):
                power_load_curtailed += md.upper(model.p_mw) - md.value(
                    model.p_mw
                ) * md.value(model.regulation)
            if isinstance(model, md.Sink):
                gas_load_curtailed += (
                    (
                        md.upper(model.mass_flow_kgs)
                        - md.value(model.mass_flow_kgs) * md.value(model.regulation)
                    )
                    * KGPS_KWHPERKG_TO_MW
                    * component.grid.higher_heating_value_kwh_per_kg
                )
            if md.hx_is_consuming(model):
                # q_mw solves to q_mw_set * regulation, so the shed is the gap
                # to the setpoint (no extra regulation factor).
                heat_load_curtailed += model.q_mw_set - md.value(model.q_mw)
            if isinstance(model, md.HeatLoad):
                heat_load_curtailed += md.upper(model.q_mw_heat) - md.value(
                    model.q_mw_heat
                ) * md.value(model.regulation)

        if include_coupling_points:
            # CP curtailment is accounted on the CP's input carrier ('power' or
            # 'gas') so it mirrors the way min_load_shedding penalises CPs.
            # Output-side service loss is already captured by the downstream
            # load shedding above.
            for component in self.get_coupling_point_components(network):
                carrier_rated = cp_input_rated_mw(component)
                if carrier_rated is None:
                    continue
                carrier, rated_mw = carrier_rated
                model = component.model
                if component.ignored or not component.active:
                    loss = rated_mw
                else:
                    reg = md.value(getattr(model, "regulation", 1))
                    if reg is None:
                        reg = 1.0
                    loss = rated_mw * max(0.0, 1.0 - reg)
                if carrier == "power":
                    power_load_curtailed += loss
                elif carrier == "gas":
                    gas_load_curtailed += loss

        power_load_curtailed = _clip_noise(power_load_curtailed)
        heat_load_curtailed = _clip_noise(heat_load_curtailed)
        gas_load_curtailed = _clip_noise(gas_load_curtailed)

        if inv:
            return (-power_load_curtailed, -heat_load_curtailed, -gas_load_curtailed)
        else:
            return (power_load_curtailed, heat_load_curtailed, gas_load_curtailed)

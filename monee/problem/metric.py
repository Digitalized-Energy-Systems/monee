from abc import ABC, abstractmethod

import monee.model as md


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


class rlist(list):
    def __init__(self, default):
        self._default = default

    def __setitem__(self, key, value):
        if key >= len(self):
            self += [self._default] * (key - len(self) + 1)
        super().__setitem__(key, value)


def is_load(component):
    model = component.model
    grid = component.grid
    return (
        isinstance(model, md.PowerLoad)
        or isinstance(model, md.Sink)
        and isinstance(grid, md.GasGrid)
        or isinstance(model, md.HeatExchangerLoad)
    )


class GeneralResiliencePerformanceMetric(PerformanceMetric):
    def get_relevant_components(self, network: md.Network):
        return [
            component
            for component in network.childs + network.branches
            if is_load(component)
        ]

    def calc(self, network, inv=False):
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
                        (md.upper(model.mass_flow))
                        * 3.6
                        * component.grid.higher_heating_value
                    )
                if isinstance(model, (md.HeatExchangerLoad)):
                    heat_load_curtailed += md.upper(model.q_w) / 10**6
                continue

            if isinstance(model, md.PowerLoad):
                power_load_curtailed += md.upper(model.p_mw) - md.value(
                    model.p_mw
                ) * md.value(model.regulation)
            if isinstance(model, md.Sink):
                gas_load_curtailed += (
                    (
                        md.upper(model.mass_flow)
                        - md.value(model.mass_flow) * md.value(model.regulation)
                    )
                    * 3.6
                    * component.grid.higher_heating_value
                )
            if isinstance(model, (md.HeatExchangerLoad)):
                heat_load_curtailed += (
                    md.upper(model.q_w)
                    - md.value(model.q_w) * md.value(model.regulation)
                ) / 10**6
        if inv:
            return (-power_load_curtailed, -heat_load_curtailed, -gas_load_curtailed)
        else:
            return (power_load_curtailed, heat_load_curtailed, gas_load_curtailed)

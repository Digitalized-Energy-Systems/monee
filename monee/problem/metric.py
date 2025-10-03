from abc import ABC, abstractmethod

import monee.model as md


class PerformanceMetric(ABC):
    """
    No docstring provided.
    """

    @abstractmethod
    def calc(self, network: md.Network):
        """
        No docstring provided.
        """


class ResilienceMetric(ABC):
    """
    No docstring provided.
    """

    @abstractmethod
    def gather(self, network: md.Network, step, **kwargs):
        """
        No docstring provided.
        """

    @abstractmethod
    def calc(self):
        """
        No docstring provided.
        """


class rlist(list):
    """
    No docstring provided.
    """

    def __init__(self, default):
        self._default = default

    def __setitem__(self, key, value):
        """
        No docstring provided.
        """
        if key >= len(self):
            self += [self._default] * (key - len(self) + 1)
        super().__setitem__(key, value)


def is_load(component):
    """
    No docstring provided.
    """
    model = component.model
    grid = component.grid
    return (
        isinstance(model, md.PowerLoad)
        or (isinstance(model, md.Sink) and isinstance(grid, md.GasGrid))
        or isinstance(model, md.HeatExchangerLoad)
        or isinstance(model, md.ExtPowerGrid)
        or isinstance(model, md.ExtHydrGrid)
        and isinstance(grid, md.GasGrid)
    )


class GeneralResiliencePerformanceMetric(PerformanceMetric):
    """
    No docstring provided.
    """

    def get_relevant_components(self, network: md.Network):
        """
        No docstring provided.
        """
        return [
            component
            for component in network.childs + network.branches
            if is_load(component)
        ]

    def calc(self, network, inv=False, include_ext_grid=True):
        """
        No docstring provided.
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
                        md.upper(model.mass_flow)
                        * 3.6
                        * component.grid.higher_heating_value
                    )
                if isinstance(model, md.HeatExchangerLoad):
                    heat_load_curtailed += md.upper(model.q_w) / 10**6
                continue
            if isinstance(model, md.ExtHydrGrid) and include_ext_grid:
                if md.value(model.mass_flow) < 0:
                    # only if ext grid needs to feed in (load would need to be shedded)
                    gas_load_curtailed += (
                        -md.value(model.mass_flow)
                        * 3.6
                        * component.grid.higher_heating_value
                    )
            if isinstance(model, md.ExtPowerGrid) and include_ext_grid:
                if md.value(model.p_mw) < 0:
                    # only if ext grid needs to feed in (load would need to be shedded)
                    power_load_curtailed += -md.value(model.p_mw)
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
            if isinstance(model, md.HeatExchangerLoad):
                heat_load_curtailed += (
                    md.upper(model.q_w)
                    - md.value(model.q_w) * md.value(model.regulation)
                ) / 10**6
        if inv:
            return (-power_load_curtailed, -heat_load_curtailed, -gas_load_curtailed)
        else:
            return (power_load_curtailed, heat_load_curtailed, gas_load_curtailed)

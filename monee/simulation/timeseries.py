from typing import Dict, Any, List, Tuple
from monee.model import Network
from monee import run_energy_flow_optimization

import pandas


class TimeseriesData:
    _child_id_to_series: Dict[Any, Dict[str, List]] = {}
    _child_name_to_series: Dict[str, Dict[str, List]] = {}

    _compound_id_to_series: Dict[Any, Dict[str, List]] = {}

    _branch_id_to_series: Dict[Any, Dict[str, List]] = {}

    def _add_to(self, target_dict, key_one, key_two, value):
        if key_one not in target_dict:
            target_dict[key_one] = {}
        target_dict[key_one][key_two] = value

    def add_compound_series(self, compound_id: int, attribute: str, series: List):
        self._add_to(self._compound_id_to_series, compound_id, attribute, series)

    def add_branch_series(self, branch_id: int, attribute: str, series: List):
        self._add_to(self._branch_id_to_series, branch_id, attribute, series)

    def add_child_series(self, child_id: int, attribute: str, series: List):
        self._add_to(self._child_id_to_series, child_id, attribute, series)

    def add_child_series_by_name(self, child_name: str, attribute: str, series: List):
        self._add_to(self._child_name_to_series, child_name, attribute, series)

    @property
    def child_id_data(self):
        return self._child_id_to_series

    @property
    def child_name_data(self):
        return self._child_name_to_series

    @property
    def branch_id_data(self):
        return self._branch_id_to_series

    @property
    def compound_id_data(self):
        return self._compound_id_to_series


class TimeseriesResult:
    def __init__(self, raw) -> None:
        self._raw_results: List = raw
        self._type_attr_to_result_df: Dict[Tuple[Any, str], pandas.DataFrame] = {}

    def _create_result_for(self, type, attribute: str):
        rows = []
        for raw_result in self._raw_results:
            raw_df = raw_result.dataframes[type.__name__]
            raw_attribute_series_t = raw_df[attribute].transpose()
            rows.append(raw_attribute_series_t.to_dict())
        df = pandas.DataFrame(rows)
        self._type_attr_to_result_df[(type, attribute)] = df
        return df

    def get_result_for(self, type, attribute: str) -> pandas.DataFrame:
        if (type, attribute) in self._type_attr_to_result_df:
            return self._type_attr_to_result_df[(type, attribute)]
        return self._create_result_for(type, attribute)

    @property
    def raw(self):
        return self._raw_results


def apply_to_by_id(component, data, timestep):
    if component.id in data:
        attr_series_dict = data[component.id]
        for attr, series in attr_series_dict.items():
            setattr(component.model, attr, series[timestep])


def apply_to_child(child, timeseries_data, timestep):
    apply_to_by_id(child, timeseries_data.child_id_data, timestep)

    if child.model._ext_data["name"] in timeseries_data.child_name_data:
        attr_series_dict = timeseries_data.data[child.model._ext_data["name"]]
        for attr, series in attr_series_dict.items():
            setattr(child.model, attr, series[timestep])


def apply_to_branch(branch, timeseries_data, timestep):
    apply_to_by_id(branch, timeseries_data.branch_id_data, timestep)


def apply_to_compound(compound, timeseries_data, timestep):
    apply_to_by_id(compound, timeseries_data.branch_id_data, timestep)


def run(
    net: Network,
    timeseries_data: TimeseriesData,
    steps: int,
    solver=None,
    optimization_problem=None,
):
    result_list = []

    for step in range(steps):
        net_copy = net.copy()
        for child in net_copy.childs:
            apply_to_child(child, timeseries_data, step)
        for branch in net_copy.branches:
            apply_to_branch(branch, timeseries_data, step)
        for compound in net_copy.compounds:
            apply_to_compound(compound, timeseries_data, step)

        result_list.append(
            run_energy_flow_optimization(
                net_copy, optimization_problem=optimization_problem, solver=solver
            )
        )
    return TimeseriesResult(result_list)

from typing import Dict, Any, List, Tuple
from monee.model import Network
from monee import run_energy_flow_optimization

import pandas


class TimeseriesData:
    _child_id_to_series: Dict[Any, Dict[str, List]] = {}
    _child_name_to_series: Dict[str, Dict[str, List]] = {}

    def add_child_series(self, child_id: Any, attribute: str, series: List):
        if child_id not in self._child_id_to_series:
            self._child_id_to_series[child_id] = {}
        self._child_id_to_series[child_id][attribute] = series

    def add_child_series_by_name(self, child_name: str, attribute: str, series: List):
        if child_name not in self._child_name_to_series:
            self._child_name_to_series[child_name] = {}
        self._child_name_to_series[child_name][attribute] = series

    @property
    def id_data(self):
        return self._child_id_to_series

    @property
    def name_data(self):
        return self._child_name_to_series


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


def apply_to_child(child, timeseries_data, timestep):
    if child.id in timeseries_data.id_data:
        attr_series_dict = timeseries_data.data[child.id]
        for attr, series in attr_series_dict.items():
            setattr(child.model, attr, series[timestep])
    if child.model._ext_data["name"] in timeseries_data.name_data:
        attr_series_dict = timeseries_data.data[child.model._ext_data["name"]]
        for attr, series in attr_series_dict.items():
            setattr(child.model, attr, series[timestep])


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

        result_list.append(
            run_energy_flow_optimization(
                net_copy, optimization_problem=optimization_problem, solver=solver
            )
        )
    return TimeseriesResult(result_list)

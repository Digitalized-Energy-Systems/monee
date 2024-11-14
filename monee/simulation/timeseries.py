from abc import ABC
from collections.abc import Callable
from typing import Any

import pandas

from monee.model import Network
from monee.simulation.core import solve


def _merge_inner_dicts_to(target_dict, extend_dict):
    for key, inner_dict in target_dict.items():
        for key_two, inner_dict_two in extend_dict.items():
            if key == key_two:
                new_inner_dict = {**inner_dict_two, **inner_dict}
                target_dict[key] = new_inner_dict
    return target_dict


class TimeseriesData:
    _child_id_to_series: dict[Any, dict[str, list]] = {}
    _child_name_to_series: dict[str, dict[str, list]] = {}

    _compound_id_to_series: dict[Any, dict[str, list]] = {}

    _branch_id_to_series: dict[Any, dict[str, list]] = {}

    def _add_to(self, target_dict, key_one, key_two, value):
        if key_one not in target_dict:
            target_dict[key_one] = {}
        target_dict[key_one][key_two] = value

    def add_compound_series(self, compound_id: int, attribute: str, series: list):
        self._add_to(self._compound_id_to_series, compound_id, attribute, series)

    def add_branch_series(self, branch_id: int, attribute: str, series: list):
        self._add_to(self._branch_id_to_series, branch_id, attribute, series)

    def add_child_series(self, child_id: int, attribute: str, series: list):
        self._add_to(self._child_id_to_series, child_id, attribute, series)

    def add_child_series_by_name(self, child_name: str, attribute: str, series: list):
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

    def extend(self, td):
        self._child_id_to_series = {**td.child_id_data, **self._child_id_to_series}
        self._child_name_to_series = {
            **td.child_name_data,
            **self._child_name_to_series,
        }
        self._branch_id_to_series = {**td.branch_id_data, **self._branch_id_to_series}
        self._compound_id_to_series = {
            **td.compound_id_data,
            **self._compound_id_to_series,
        }

        _merge_inner_dicts_to(self._child_id_to_series, td.child_id_data)
        _merge_inner_dicts_to(self._child_name_to_series, td.child_name_data)
        _merge_inner_dicts_to(self._branch_id_to_series, td.branch_id_data)
        _merge_inner_dicts_to(self._compound_id_to_series, td.compound_id_data)

    def __add__(self, other):
        new_td = TimeseriesData()
        new_td.extend(self)
        new_td.extend(other)
        return new_td


class TimeseriesResult:
    def __init__(self, raw) -> None:
        self._raw_results: list = raw
        self._type_attr_to_result_df: dict[tuple[Any, str], pandas.DataFrame] = {}

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

    if child.name in timeseries_data.child_name_data:
        attr_series_dict = timeseries_data.data[child.name]
        for attr, series in attr_series_dict.items():
            setattr(child.model, attr, series[timestep])


def apply_to_branch(branch, timeseries_data, timestep):
    apply_to_by_id(branch, timeseries_data.branch_id_data, timestep)


def apply_to_compound(compound, timeseries_data, timestep):
    apply_to_by_id(compound, timeseries_data.branch_id_data, timestep)


class StepHook(ABC):
    def pre_run(self, net, base_net, step):
        pass

    def post_run(self, net, base_net, step):
        pass


def run(
    net: Network,
    timeseries_data: TimeseriesData,
    steps: int,
    step_hooks: list[StepHook | Callable],
    solver=None,
    optimization_problem=None,
    solve_flag=True,
):
    result_list = []

    for step in range(steps):
        for step_hook in step_hooks:
            if isinstance(step_hook, StepHook):
                step_hook.pre_run(net, step)

        net_copy = net.copy()

        for child in net_copy.childs:
            apply_to_child(child, timeseries_data, step)
        for branch in net_copy.branches:
            apply_to_branch(branch, timeseries_data, step)
        for compound in net_copy.compounds:
            apply_to_compound(compound, timeseries_data, step)

        if solve_flag:
            result_list.append(
                solve(
                    net_copy, optimization_problem=optimization_problem, solver=solver
                )
            )

        for step_hook in step_hooks:
            if isinstance(step_hook, StepHook):
                step_hook.post_run(net_copy, net, step)
            else:
                step_hook(net_copy, net, step)

    return TimeseriesResult(result_list)

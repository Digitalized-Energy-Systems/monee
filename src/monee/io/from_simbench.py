import simbench

import monee.model as md
from monee.io.from_pandapower import from_pandapower_net
from monee.simulation.timeseries import TimeseriesData


def _attr_by_type(t):
    """
    No docstring provided.
    """
    if t == "load":
        return None
    else:
        return "p_mw"


def obtain_simbench_profile_by_pp_net(pp_net) -> TimeseriesData:
    """
    No docstring provided.
    """
    td = TimeseriesData()
    profile_dict = pp_net.profiles
    for t, profile_df in profile_dict.items():
        for name, values in profile_df.items():
            if name == "time":
                continue
            actual_name = name
            attr = _attr_by_type(t)
            if t == "load":
                actual_name = name[:-6]
                attr = "p_mw" if name[-5] == "p" else "q_mvar"
            td.add_child_series_by_name(actual_name, attr, values)
    return td


def obtain_simbench_profile(sb_code) -> TimeseriesData:
    """
    No docstring provided.
    """
    net = simbench.get_simbench_net(sb_code)
    return obtain_simbench_profile_by_pp_net(net)


def obtain_simbench_net(sb_code) -> md.Network:
    """
    No docstring provided.
    """
    net = simbench.get_simbench_net(sb_code)
    return from_pandapower_net(net)


def obtain_simbench_net_with_td(sb_code) -> tuple[md.Network, TimeseriesData]:
    """
    No docstring provided.
    """
    net = simbench.get_simbench_net(sb_code)
    return (from_pandapower_net(net), obtain_simbench_profile_by_pp_net(net))

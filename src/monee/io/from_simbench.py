import simbench

import monee.model as md
from monee.io.from_pandapower import aggregated_pp_load_name, from_pandapower_net
from monee.simulation.timeseries import TimeseriesData


def obtain_simbench_profile_by_pp_net(pp_net) -> TimeseriesData:
    """Build a :class:`TimeseriesData` from a simbench pandapower net.

    Load profiles are scaled per-load (``base_p_mw·profile[t]``) and summed
    per bus to match :func:`from_pandapower_net`'s aggregation, registered
    under :func:`aggregated_pp_load_name`. Non-load profile categories pass
    through under their raw simbench column names.
    """
    td = TimeseriesData()
    profiles = pp_net.profiles

    if "load" in profiles and hasattr(pp_net, "load") and len(pp_net.load):
        load_df = profiles["load"]
        for _bus, group in pp_net.load.groupby("bus", sort=False):
            agg_name = aggregated_pp_load_name(group)
            p_total = None
            q_total = None
            for _, row in group.iterrows():
                profile = row["profile"]
                p_col = f"{profile}_pload"
                q_col = f"{profile}_qload"
                if p_col in load_df.columns:
                    contribution = load_df[p_col].to_numpy() * float(row["p_mw"])
                    p_total = (
                        contribution if p_total is None else p_total + contribution
                    )
                if q_col in load_df.columns:
                    contribution = load_df[q_col].to_numpy() * float(row["q_mvar"])
                    q_total = (
                        contribution if q_total is None else q_total + contribution
                    )
            if p_total is not None:
                td.add_child_series_by_name(agg_name, "p_mw", p_total.tolist())
            if q_total is not None:
                td.add_child_series_by_name(agg_name, "q_mvar", q_total.tolist())

    for t, profile_df in profiles.items():
        if t == "load":
            continue
        for name, values in profile_df.items():
            if name == "time":
                continue
            td.add_child_series_by_name(name, "p_mw", list(values))

    return td


def obtain_simbench_profile(sb_code) -> TimeseriesData:
    net = simbench.get_simbench_net(sb_code)
    return obtain_simbench_profile_by_pp_net(net)


def obtain_simbench_net(sb_code) -> md.Network:
    net = simbench.get_simbench_net(sb_code)
    return from_pandapower_net(net)


def obtain_simbench_net_with_td(sb_code) -> tuple[md.Network, TimeseriesData]:
    net = simbench.get_simbench_net(sb_code)
    return (from_pandapower_net(net), obtain_simbench_profile_by_pp_net(net))

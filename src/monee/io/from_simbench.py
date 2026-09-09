import logging
import warnings

import simbench

import monee.model as md
from monee.io.from_pandapower import (
    _coerce_positive_float,
    aggregated_pp_load_name,
    from_pandapower_net,
)
from monee.simulation.timeseries import TimeseriesData

logger = logging.getLogger(__name__)


def _row_scaling(row) -> float:
    return _coerce_positive_float(row.get("scaling", 1.0), allow_zero=True)


def _is_missing(value) -> bool:
    return value is None or value != value  # NOSONAR NaN


def _find_profile_column(profiles, profile_name):
    for category in ("renewables", "powerplants"):
        df = profiles.get(category)
        if df is not None and profile_name in df.columns:
            return category, df[profile_name].to_numpy()
    for category, df in profiles.items():
        if category in ("load", "storage"):
            continue
        if profile_name in df.columns:
            return category, df[profile_name].to_numpy()
    return None, None


def _attach_generation_series(td, table, profiles, consumed) -> list:
    """Register per-element generation series and report unresolved elements.

    The stored value carries the internal load-convention sign: a
    :class:`~monee.model.child.PowerGenerator` holds ``-p``, and a negative-p
    sgen becomes a :class:`~monee.model.child.PowerLoad` holding ``+|p|``, so
    ``-profile*p_mw*scaling`` is correct for both.
    """
    unresolved = []
    if table is None or not len(table):
        return unresolved
    for _, row in table.iterrows():
        if not row.get("in_service", True):
            continue
        name = row.get("name")
        profile_name = row.get("profile")
        if _is_missing(name) or _is_missing(profile_name):
            continue
        category, values = _find_profile_column(profiles, profile_name)
        if values is None:
            unresolved.append(str(profile_name))
            continue
        consumed.add((category, profile_name))
        scaling = _row_scaling(row)
        p_mw = float(row["p_mw"])
        td.add_child_series_by_name(
            str(name), "p_mw", (-values * p_mw * scaling).tolist()
        )
        q_mvar = float(row.get("q_mvar", 0.0) or 0.0)
        if q_mvar:
            td.add_child_series_by_name(
                str(name), "q_mvar", (-values * q_mvar * scaling).tolist()
            )
    return unresolved


def obtain_simbench_profile_by_pp_net(pp_net) -> TimeseriesData:  # NOSONAR
    """Build a :class:`TimeseriesData` from a simbench pandapower net.

    Load profiles are scaled per-load (``base_p_mw*profile[t]``) and summed
    per bus to match :func:`from_pandapower_net`'s aggregation, registered
    under :func:`aggregated_pp_load_name`. Generation profiles are resolved
    per element of ``net.sgen`` / ``net.gen`` (``-profile[t]*base_p_mw*scaling``,
    the generation sign of the load convention) and registered under the
    element name that :func:`from_pandapower_net` puts on the child. Profile
    columns no element claims pass through under their raw simbench column
    name and are warned about; the ``storage`` category is skipped entirely,
    since its columns are state-of-charge series and not ``p_mw``.
    """
    td = TimeseriesData()
    profiles = getattr(pp_net, "profiles", None)
    if not profiles:
        raise ValueError(
            "The pandapower net carries no simbench profiles ('net.profiles' is "
            "missing or empty). Load the net with simbench.get_simbench_net(...) "
            "or attach the profiles dict before calling this function."
        )

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
                # p_mw*scaling matches the base import, where to_mpc applies
                # the per-load scaling factor.
                scaling = _row_scaling(row)
                if p_col in load_df.columns:
                    contribution = (
                        load_df[p_col].to_numpy() * float(row["p_mw"]) * scaling
                    )
                    p_total = (
                        contribution if p_total is None else p_total + contribution
                    )
                if q_col in load_df.columns:
                    contribution = (
                        load_df[q_col].to_numpy() * float(row["q_mvar"]) * scaling
                    )
                    q_total = (
                        contribution if q_total is None else q_total + contribution
                    )
            if p_total is not None:
                td.add_child_series_by_name(agg_name, "p_mw", p_total.tolist())
            if q_total is not None:
                td.add_child_series_by_name(agg_name, "q_mvar", q_total.tolist())

    consumed = set()
    unresolved = []
    for table_name in ("sgen", "gen"):
        unresolved += _attach_generation_series(
            td, getattr(pp_net, table_name, None), profiles, consumed
        )
    if unresolved:
        logger.warning(
            "simbench import: no profile column found for %d element profile(s) "
            "(%s); the affected generators keep their constant nameplate power.",
            len(unresolved),
            ", ".join(sorted(set(unresolved))),
        )

    # Columns no element claimed still pass through under their raw name, so a
    # hand-built child named after a simbench profile keeps working; 'p_mw' is
    # assumed, which is why the storage category (state of charge) is skipped.
    for t, profile_df in profiles.items():
        if t in ("load", "storage"):
            continue
        registered = []
        for name, values in profile_df.items():
            if name == "time" or (t, name) in consumed:
                continue
            td.add_child_series_by_name(name, "p_mw", list(values))
            registered.append(name)
        if registered:
            logger.warning(
                "simbench import: %d %r profile column(s) (%s) belong to no "
                "in-service element and were registered as 'p_mw' child series "
                "by raw name; these apply only to childs whose name matches "
                "exactly, unmatched series are silently ignored.",
                len(registered),
                t,
                ", ".join(map(str, registered)),
            )

    return td


def obtain_simbench_profile(sb_code) -> TimeseriesData:
    """Fetch the simbench grid *sb_code* and build its :class:`TimeseriesData`.

    Shorthand for ``obtain_simbench_profile_by_pp_net(simbench.get_simbench_net
    (sb_code))``; see :func:`obtain_simbench_profile_by_pp_net` for the profile
    scaling and naming conventions. Use :func:`obtain_simbench_net_with_td`
    when you also need the network, to avoid fetching the grid twice. Covered
    in the how-to guide "Convert from pandapower" (SimBench section).
    """
    net = simbench.get_simbench_net(sb_code)
    return obtain_simbench_profile_by_pp_net(net)


def obtain_simbench_net(sb_code) -> md.Network:
    """Fetch the simbench grid *sb_code* as a monee :class:`~monee.model.Network`.

    Shorthand for ``from_pandapower_net(simbench.get_simbench_net(sb_code))``,
    so everything :func:`~monee.io.from_pandapower.from_pandapower_net`
    documents applies: loads are aggregated per bus under a joined name,
    line/trafo ratings and names are carried onto the branches, and the
    network exposes the ``pp_bus_to_node`` / ``pp_line_to_branch`` /
    ``pp_trafo_to_branch`` lookups. *sb_code* is a simbench grid code such as
    ``"1-LV-rural1--0-no_sw"``. Covered in the how-to guide "Convert from
    pandapower" (SimBench section).
    """
    net = simbench.get_simbench_net(sb_code)
    return from_pandapower_net(net)


def obtain_simbench_net_with_td(sb_code) -> tuple[md.Network, TimeseriesData]:
    """Fetch the simbench grid *sb_code* as ``(Network, TimeseriesData)``.

    One simbench fetch yields both the converted network (see
    :func:`obtain_simbench_net`) and the full-year 15-minute profiles as a
    :class:`TimeseriesData` (see :func:`obtain_simbench_profile_by_pp_net` for
    the scaling and naming conventions). The pair feeds
    :func:`monee.run_timeseries` directly; pass ``steps`` there to simulate
    only part of the horizon. Covered in the how-to guide "Convert from
    pandapower" (SimBench section).

    The pandas FutureWarnings simbench's own csv converter raises during the
    fetch are suppressed here: they are not actionable from monee and they
    bury monee's own conversion warnings.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=FutureWarning, module=r".*simbench.*"
        )
        net = simbench.get_simbench_net(sb_code)
    return (from_pandapower_net(net), obtain_simbench_profile_by_pp_net(net))

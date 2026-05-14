import math
import os
import uuid

import pandapower.converter as pc

from monee.model.child import PowerLoad

from .matpower import read_matpower_case

_SQRT3 = math.sqrt(3.0)


def aggregated_pp_load_name(pp_loads_at_bus) -> str:
    """Stable joined name for the pandapower loads sharing a single bus.

    The matpower converter aggregates every pandapower load on a bus into a
    single :class:`~monee.model.child.PowerLoad`.  We mirror that grouping
    by joining the contributing pandapower load names with ``"+"`` so the
    monee child has a deterministic name that
    :func:`monee.io.from_simbench.obtain_simbench_profile_by_pp_net` can use
    to bind the corresponding (summed) timeseries.
    """
    return "+".join(pp_loads_at_bus["name"].astype(str).tolist())


def _coerce_positive_int(value, default=1):
    if value is None:
        return default
    try:
        fv = float(value)
    except (TypeError, ValueError):
        return default
    if fv != fv or fv <= 0:  # NaN or non-positive
        return default
    return max(1, int(fv))


def _coerce_positive_float(value, default=1.0):
    if value is None:
        return default
    try:
        fv = float(value)
    except (TypeError, ValueError):
        return default
    if fv != fv or fv <= 0:
        return default
    return fv


def _pp_branch_max_i_ka_overrides(net):
    """Build a ``{monee_branch_id -> max_i_ka}`` map from the pandapower
    line / trafo tables, recovering values dropped by the matpower roundtrip.

    The matpower converter only carries an apparent-power rating (``rateA``),
    and our :func:`~monee.io.matpower.fill_branch_dict` hardcodes
    ``max_i_ka = 0.319`` for every emitted branch — so line current limits
    and transformer ratings are wrong for any net imported through this
    path.  This helper builds the authoritative override directly from
    ``net.line`` / ``net.trafo``.

    Mapping convention (verified empirically against ``pc.to_mpc``):

    * Bus index: ``pc.to_mpc`` renumbers pandapower bus indices to a
      contiguous 1-based sequence preserving ``net.bus`` row order, so
      ``mpc_bus_id = net.bus.index.get_loc(pp_bus) + 1``.  For nets whose
      pandapower indices are already ``0..N-1`` this collapses to
      ``pp_bus + 1`` (the LV-rural3 case), but simbench MV / HV grids have
      sparse indices and require the lookup.
    * Branch ordering: lines first in ``net.line`` row order, then trafos
      in ``net.trafo`` row order; each ``(from, to)`` pair gets a per-pair
      parallel-key counter ``0, 1, 2, …`` matching the ``key`` element of
      monee's branch ID tuple ``(from_node_id, to_node_id, key)``.

    Per-row capacity:

    * **Line** — ``max_i_ka × parallel × df``; ``parallel`` reflects
      multiple physical conductors sharing the (from, to) endpoint pair,
      ``df`` is pandapower's derating factor.
    Transformers are intentionally **not** overridden.  ``GenericPowerBranch``
    constrains ``i_from_ka`` and ``i_to_ka`` against the *same* ``max_i_ka``
    scalar (see :class:`~monee.model.branch.GenericPowerBranch.equations`).
    For a Y-Δ trafo the HV-side current at rated MVA is
    ``sn_mva / (√3 · vn_hv_kv)`` and the LV-side current is
    ``sn_mva / (√3 · vn_lv_kv)``; the two differ by the turns ratio (e.g.
    50× for a 20 kV / 0.4 kV unit), so any single ``max_i_ka`` scalar
    saturates one side or the other.  The legacy 0.319 kA placeholder is
    physically wrong for both sides too, but at least doesn't bind the
    solver into infeasibility under normal loading; introducing an
    HV-tight bound would make the LV-side current constraint structurally
    infeasible.  A proper trafo limit requires a model-side change
    (separate ``max_i_from_ka`` / ``max_i_to_ka`` or an apparent-power
    ``max_s_mva``).

    Auxiliary branches that ``pc.to_mpc`` synthesises to model open line
    switches (one extra bus + one extra branch per ``net.switch`` row of
    type ``"l"``) are also not covered here — there is no corresponding
    row in ``net.line`` / ``net.trafo`` to draw a rating from.  They keep
    the legacy ``0.319`` placeholder from
    :func:`~monee.io.matpower.fill_branch_dict` and can be detected after
    import as branches whose ``to_node_id`` exceeds ``len(net.bus)``.
    """
    overrides = {}
    seen = {}

    bus_index = net.bus.index if hasattr(net, "bus") else None
    if bus_index is None:
        return overrides

    def mpc_bus(pp_bus):
        return int(bus_index.get_loc(int(pp_bus))) + 1

    if hasattr(net, "line") and len(net.line):
        for row in net.line.itertuples():
            f = mpc_bus(row.from_bus)
            t = mpc_bus(row.to_bus)
            key = seen.get((f, t), 0)
            seen[(f, t)] = key + 1
            max_i_ka = float(row.max_i_ka)
            parallel = _coerce_positive_int(getattr(row, "parallel", 1))
            df = _coerce_positive_float(getattr(row, "df", 1.0))
            overrides[(f, t, key)] = max_i_ka * parallel * df

    # Trafos: advance the (f, t) parallel-key counter so subsequent lines on
    # the same node pair get the right key, but do *not* emit an override —
    # see the docstring above for why mapping ``sn_mva`` onto a single
    # ``max_i_ka`` scalar is physically inconsistent for a Y-Δ transformer.
    if hasattr(net, "trafo") and len(net.trafo):
        for row in net.trafo.itertuples():
            f = mpc_bus(row.hv_bus)
            t = mpc_bus(row.lv_bus)
            seen[(f, t)] = seen.get((f, t), 0) + 1

    return overrides


def from_pandapower_net(net):
    id_file = uuid.uuid4()
    name_file = f"{id_file}.mat"
    pc.to_mpc(net, init="flat", filename=name_file)
    monee_net = read_matpower_case(name_file)
    os.remove(name_file)
    for node in monee_net.nodes:
        pp_id = node.id - 1
        if len(net.bus) > pp_id:
            node.name = net.bus["name"].iloc[pp_id]
            if hasattr(net, "bus_geodata"):
                node.position = (
                    net.bus_geodata["x"].iloc[pp_id],
                    net.bus_geodata["y"].iloc[pp_id],
                )

    # Override ``max_i_ka`` from the authoritative pandapower line / trafo
    # tables — the matpower roundtrip drops physical current limits (see
    # :func:`_pp_branch_max_i_ka_overrides`).
    overrides = _pp_branch_max_i_ka_overrides(net)
    if overrides:
        for branch in monee_net.branches:
            if not hasattr(branch.model, "max_i_ka"):
                continue
            bid = (branch.from_node_id, branch.to_node_id, branch.id[2])
            if bid in overrides:
                branch.model.max_i_ka = overrides[bid]

    # Tag aggregated PowerLoad children with a deterministic name derived
    # from the contributing pandapower loads, so simbench timeseries can be
    # matched back to them by name.  Buses with no load are silently
    # skipped; buses where matpower produced no PowerLoad child (zero
    # aggregate p/q) are likewise ignored.
    if hasattr(net, "load") and len(net.load):
        nodes_by_id = {n.id: n for n in monee_net.nodes}
        for pp_bus, group in net.load.groupby("bus", sort=False):
            monee_node = nodes_by_id.get(int(pp_bus) + 1)
            if monee_node is None:
                continue
            agg_name = aggregated_pp_load_name(group)
            for child in monee_net.childs_by_ids(monee_node.child_ids):
                if isinstance(child.model, PowerLoad):
                    child.name = agg_name

    return monee_net

============================
Convert from pandapower
============================

If you already have an electrical network defined in
`pandapower <https://www.pandapower.org>`_, monee can import it directly.
The helper function serialises the pandapower network to a temporary
MATPOWER ``.mat`` file and reads it back into monee for you.

.. warning::

   This feature is experimental. The export flattens everything into buses,
   branches and injections, so elements without a monee counterpart survive as
   their electrical effect only, and some of them add auxiliary nodes: a
   three-winding transformer arrives as three branches around an unnamed star
   node, and an open line switch splits its line at an auxiliary bus. The
   import therefore gains one extra unnamed node per open switch: the line
   keeps its pandapower current limit, the stub node carries no child, and the
   solver drops it as a dead end, which marks its result row ``ignored=True``
   with a NaN voltage. The line charging of the disconnected section is lost
   with it, which shifts the slack reactive power slightly (15.713 Mvar instead
   of the 15.696 Mvar pandapower reports on the CIGRE medium voltage grid).
   Close or fuse the switches before converting if you need results that match
   pandapower to the last decimal, and always verify against the original
   network.

----

Prerequisites
=============

pandapower (and simbench, for the section below) are optional
dependencies. Install them via the ``simbench`` extra:

.. code-block:: bash

    pip install monee[simbench]

or install pandapower alone if you only need the converter:

.. code-block:: bash

    pip install pandapower

----

Converting a network
====================

.. code-block:: python

    import pandapower as pp
    import pandapower.networks as pn
    from monee.io.from_pandapower import from_pandapower_net

    # Load a built-in IEEE test case from pandapower
    pp_net = pn.case9()

    # Convert to a monee Network
    net = from_pandapower_net(pp_net)

The returned :class:`~monee.model.Network` is a standard monee electricity
network. Run an energy flow immediately:

.. code-block:: python

    from monee import run_energy_flow

    result = run_energy_flow(net)

Plotting the converted network
------------------------------

``result.plot()`` is the quickest way to eyeball a conversion. It returns a
``plotly.graph_objects.Figure`` of the solved network: buses colored green,
yellow or red by ``vm_pu`` band, branches colored by loading, junctions
labeled with ``pressure_pu``, and the full result row of every component in
its hover text.

.. code-block:: python

    fig = result.plot(
        title="case9 after conversion",   # default "Network Result"
        show_children=True,               # loads and generators in node hover
        use_monee_positions=True,         # stored node.position instead of a spring layout
        write_to="case9.html",            # optional export path
    )
    fig.show()

The converter copies the pandapower bus geodata into ``node.position`` where
the source net has it, so ``use_monee_positions=True`` reproduces the
pandapower layout. Stored positions are used only when every node carries
one; if any node lacks a position the whole plot falls back to a computed
layout.
A ``.html`` path in ``write_to`` writes a self-contained interactive page,
which is what you want for a report. PDF, PNG and SVG paths use static export
and need the optional ``kaleido`` package (``pip install monee[plot]``).

----

What is preserved
=================

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - pandapower element
     - monee equivalent
   * - ``bus``
     - :class:`~monee.model.node.Bus` node (id, name, geodata)
   * - ``line``
     - :class:`~monee.model.branch.GenericPowerBranch` (r, x, b parameters,
       current limit)
   * - ``trafo``
     - :class:`~monee.model.branch.GenericPowerBranch` (tap ratio and phase
       shift, HV side rated current as the limit, see below)
   * - ``ext_grid``
     - :class:`~monee.model.child.ExtPowerGrid` child (the slack)
   * - ``gen`` at a PV bus
     - :class:`~monee.model.child.VoltageControlledGenerator` child (fixed
       active power, pinned voltage magnitude)
   * - ``gen`` at the slack bus, second ``gen`` at the same bus, or an
       out-of-service ``gen``
     - :class:`~monee.model.child.ExtPowerGrid` for the first one at the slack
       bus, :class:`~monee.model.child.PowerGenerator` for the others
   * - ``sgen``
     - :class:`~monee.model.child.PowerGenerator` child, or
       :class:`~monee.model.child.PowerLoad` when ``p_mw`` is negative
   * - ``load``
     - :class:`~monee.model.child.PowerLoad` child (aggregated per bus,
       see below)
   * - ``switch``
     - closed bus-bus switches fuse their two buses into one node; open line
       switches add an auxiliary node (see the warning above)

Elements outside this table keep their electrical effect but lose their
identity: a ``shunt`` (and the shunt part of a ``ward``) becomes a
:class:`~monee.model.child.PowerShunt`, a ``storage`` and the injection part of
a ``ward`` are folded into the bus demand and end up inside the aggregated
:class:`~monee.model.child.PowerLoad` as a fixed draw, an ``impedance`` becomes
another :class:`~monee.model.branch.GenericPowerBranch`, and a ``trafo3w``
becomes three branches around an auxiliary star node. Generator limits and the
cost tables (``poly_cost``, ``pwl_cost``) are dropped unless you convert with
``opf=True``.

Node ids, names and positions
-----------------------------

The MATPOWER format numbers buses from 1, so the monee node id follows the
position of the bus in the exported case, not the pandapower bus index. The two
agree only when the index is contiguous (``0 .. n-1``) and no buses are fused:
on ``1-LV-rural1--0-no_sw`` the bus index ends at 42 while the last node id is
15, and a closed bus-bus switch fuses two buses into one node, which shifts
every id behind it. The importer therefore stores the real lookup on the
returned network:

.. code-block:: python

    net = from_pandapower_net(pp_net)
    node_id = net.pp_bus_to_node[bus]  # pandapower bus index -> monee node id

.. warning::

   Joining that lookup against a snapshot result frame needs the id-indexed
   view. The frames returned by ``result.get(ModelType)`` are indexed
   positionally, exactly like ``result.dataframes[...]``, and carry the
   component id in their ``id`` column, so ``result.get(mm.Bus).loc[node_id]``
   selects row number ``node_id`` and quietly reports a different bus (or
   raises ``KeyError`` once the id runs past the row count). This is the join
   that fusion and the MATPOWER renumbering above are most likely to break.
   Pass ``index="id"`` to index the rows by component id instead:

   .. code-block:: python

       vm = result.get(mm.Bus, index="id")           # rows indexed by node id
       for bus, node_id in net.pp_bus_to_node.items():
           delta = vm.loc[node_id, "vm_pu"] - pp_net.res_bus.vm_pu[bus]

   The default index is unchanged, so existing scripts keep working; only the
   ``index="id"`` view is new. :doc:`../concepts/data_model` describes the two
   index modes and how branch ids behave.

Use it to align ``net_pp.res_bus`` with monee results. Fused buses share one
node, so several bus indices can map to the same id, and the node keeps the
name of the first pandapower bus. Bus names and coordinates are copied onto the
nodes when present. After conversion, ``node.name`` holds the pandapower bus
name and ``node.position`` holds the ``(x, y)`` coordinates. Coordinates come
from ``net.bus.geo`` (the GeoJSON ``Point`` column used by pandapower 3.x) or
from the legacy ``net.bus_geodata`` frame on pandapower 2.x.

.. note::

   pandapower 3.x models a transformer's winding vector group as a branch phase
   shift (a multiple of 30°, for example Dyn5 gives 150°) and exports it through
   ``to_mpc``; pandapower 2.x dropped it. For monee's single-slack PQ power flow
   that shift is a pure reference rotation: it offsets all downstream bus angles
   by a constant and leaves every magnitude (voltages, branch flows, currents)
   unchanged. The importer zeroes these transformer phase shifts so the
   flat-start AC solve stays well-conditioned. Reported magnitudes still match
   pandapower's; only the absolute downstream angle reference differs.

Which branch is which
---------------------

Lines, two winding transformers, three winding transformer arms and
impedances all arrive as :class:`~monee.model.branch.GenericPowerBranch`, so
the class alone cannot tell you which row is the transformer. The importer
therefore marks every electrical branch with a ``kind`` attribute naming its
pandapower origin: ``"line"``, ``"trafo"``, ``"trafo3w"`` (one arm of a three
winding transformer), ``"impedance"``, ``"switch-stub"`` (a line split at the
auxiliary bus of an open switch, see the warning above) or ``"unknown"``. The
attribute survives into the branch result frame as a ``kind`` column:

.. code-block:: python

    result = monee.run_energy_flow(net)
    branches = result.get(mm.GenericPowerBranch)
    trafo_rows = branches[branches["kind"] == "trafo"]
    print(trafo_rows[["id", "loading_from_pu", "loading_to_pu"]])

The pandapower names travel along: each branch created from a ``line``,
``trafo`` or ``impedance`` row keeps that row's ``name`` (for example
``"LV1.101 Line 5"``), which shows up as a ``name`` column in the branch
result frame next to ``kind``. For a join by element index instead of by
name, the network carries branch lookups analogous to ``pp_bus_to_node``:

.. code-block:: python

    branch_id = net.pp_line_to_branch[5]    # pandapower line index -> branch id
    trafo_id = net.pp_trafo_to_branch[0]    # pandapower trafo index -> branch id
    line_row = branches[branches["id"].isin([branch_id])]

A branch id is the ``(from_node_id, to_node_id, key)`` tuple, which a
positional frame cannot select by label at all: ``branches.set_index("id")``
builds an index of tuples and ``.loc[branch_id]`` then raises an
``IndexingError`` because pandas reads the tuple as one key per axis. The
boolean mask above avoids that, and so does the id-indexed view, which lays the
tuple out as a three level index:

.. code-block:: python

    by_id = result.get(mm.GenericPowerBranch, index="id")
    print(by_id.index.names)                 # ['from_node_id', 'to_node_id', 'key']
    loading = by_id.loc[branch_id, "loading_from_pu"]
    from_bus_5 = by_id.loc[net.pp_bus_to_node[5]]   # every branch leaving that node

These lookups also resolve a line that an open switch splits at an auxiliary
bus, where reconstructing the branch from the endpoint buses fails because one
endpoint is the unnamed stub node. A line the importer could not map at all
(see the ``max_i_ka`` warning above) has no entry.

Line current limits
-------------------

The intermediate MATPOWER round-trip drops current limits, so
:func:`~monee.io.from_pandapower.from_pandapower_net` recovers them
afterwards from the original pandapower tables. Each line's limit is
rebuilt as ``max_i_ka * parallel * df`` and written back to the matching
monee branch.

Transformer ratings and loading
-------------------------------

A transformer's ``max_i_ka`` is not a placeholder: it is the real rated
current of the HV side. The MATPOWER export carries the transformer rating as
``RATE_A = sn_mva * max_loading_percent / 100`` and monee converts it with the
HV side voltage base, ``max_i_ka = RATE_A / (sqrt(3) * vn_hv_kv)``. The
SimBench 0.16 MVA 20/0.4 kV distribution transformer therefore arrives with
``0.16 / (sqrt(3) * 20) = 0.00462`` kA, its actual HV side rated current.

Read transformer loading from ``loading_from_pu`` and ``loading_to_pu`` in the
result frame. Both are computed against this rating with the correct voltage
base on each side (the LV side current is compared to the LV side rated
current ``max_i_ka * vn_hv_kv / vn_lv_kv``), so a value of 1.0 means the
transformer runs at ``sn_mva`` and ``loading_pu * sn_mva`` is the apparent
power flowing through it in MVA.

One caveat: pandapower only exports a meaningful ``RATE_A`` when the
transformer has ``max_loading_percent`` set. Without it the export falls back
to a rating at the system base (typically 100 MVA), which makes ``max_i_ka``
and the loading columns meaningless for that transformer. Set
``max_loading_percent`` before converting when you rely on transformer
loading.

.. warning::

   Some tools ship transformer ratings as placeholders, and study briefs
   sometimes tell you to skip the transformer limit for that reason. For
   SimBench grids (and any pandapower net with ``max_loading_percent`` set)
   that folklore is wrong: the imported rating is the real nameplate rating,
   and in a small LV feeder the transformer is often the first constraint to
   bind, well before any line limit or voltage band. Include the
   ``kind == "trafo"`` rows in every loading or hosting capacity check:

   .. code-block:: python

       overloaded = branches[
           (branches["loading_from_pu"] > 1.0) | (branches["loading_to_pu"] > 1.0)
       ]

   Filtering to ``kind == "line"`` first silently drops the binding
   constraint and overstates the hosting capacity.

Load aggregation and naming
---------------------------

All pandapower loads connected to the same bus are aggregated into a
single :class:`~monee.model.child.PowerLoad`. The aggregated child is
named deterministically by joining the original pandapower load names with
``+`` (see :func:`~monee.io.from_pandapower.aggregated_pp_load_name`), for
example ``"LV1.101 Load 1+LV1.101 Load 2"``. This deterministic naming lets
simbench timeseries match back to the aggregated loads by name.

Making a converted case dispatchable
------------------------------------

A plain conversion is a power flow: every generator holds the setpoint it had
in pandapower, and the ``poly_cost`` and ``pwl_cost`` tables together with the
``min_p_mw`` to ``max_p_mw`` limits are dropped (the importer warns when it
drops them). Pass ``opf=True`` to keep them and get an optimisation problem
alongside the network:

.. code-block:: python

    import pandapower.networks as pn
    import monee
    import monee.model as mm
    from monee.io.from_pandapower import from_pandapower_net

    net, problem = from_pandapower_net(pn.case9(), opf=True)
    result = monee.run_energy_flow_optimization(net, problem)
    print(result.get(mm.PowerGenerator)["p_mw"])
    # 0   -134.482654
    # 1    -94.326647
    # Name: p_mw, dtype: float64

Every in-service generator becomes a dispatchable
:class:`~monee.model.child.PowerGenerator` bounded by the pandapower
``min_p_mw`` to ``max_p_mw`` and ``min_q_mvar`` to ``max_q_mvar`` limits and
carrying its polynomial cost, the ext_grid cost lands on the slack, and the bus
voltages are optimised inside their ``min_vm_pu`` to ``max_vm_pu`` band. Powers
follow monee's load convention, so a generation of 134.48 MW is stored as
``-134.48``. ``max_loading`` (per unit of the line rating, ``None`` to drop the
limits) and ``limit_basis`` (``"mva"`` or ``"current"``) are passed straight to
:func:`~monee.io.matpower.read_matpower_opf_case`.

The values above reproduce ``pandapower.runopp`` on the same case to within
about 1e-5 MW. Do not expect bit-identical numbers: the two solvers differ in
formulation details, and static generators are stripped before the MATPOWER
export and re-attached afterwards, so an ``sgen`` stays a fixed injection under
``opf=True`` rather than becoming dispatchable.

.. tip::

   After converting, you can save the result to the native OMEF format using
   :func:`monee.io.native.write_omef_network` so you do not need pandapower
   installed in future sessions. See :doc:`matpower_io` for details.

----

SimBench
========

`SimBench <https://simbench.de>`_ provides standardised benchmark grids
with full-year load profiles. :mod:`monee.io.from_simbench` wraps the
pandapower converter and turns the simbench profiles into a
:class:`~monee.simulation.TimeseriesData` ready for
:func:`~monee.run_timeseries`:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Function
     - Returns
   * - ``obtain_simbench_net(sb_code)``
     - :class:`~monee.model.Network`
   * - ``obtain_simbench_profile(sb_code)``
     - :class:`~monee.simulation.TimeseriesData`
   * - ``obtain_simbench_net_with_td(sb_code)``
     - ``(Network, TimeseriesData)`` from a single simbench fetch
   * - ``obtain_simbench_profile_by_pp_net(pp_net)``
     - :class:`~monee.simulation.TimeseriesData` from an already loaded
       simbench pandapower net

Load profiles are scaled per load (``base_p_mw * profile``) and summed per
bus, matching the per-bus PowerLoad aggregation described above. Each summed
series is registered under the aggregated load name via
:meth:`~monee.simulation.TimeseriesData.add_child_series_by_name`, for both the
``p_mw`` and ``q_mvar`` attributes.

Generation profiles are resolved per element: the profile named in
``net.sgen["profile"]`` (or ``net.gen["profile"]``) is looked up in the
``renewables`` and ``powerplants`` categories and registered under the element
name that the converter puts on the child, scaled to
``-profile * p_mw * scaling``. The minus sign is monee's load convention, so a
PV unit shows negative ``p_mw`` while it generates. Profile columns that no
in-service element claims still pass through under their raw simbench column
name as ``p_mw`` series and are logged as a warning; the ``storage`` category is
skipped, since its columns are state of charge rather than power.

Feeding the result into a timeseries simulation takes four lines:

.. code-block:: python

    from monee import run_timeseries
    from monee.io.from_simbench import obtain_simbench_net_with_td

    net, td = obtain_simbench_net_with_td("1-LV-rural1--0-no_sw")
    ts_result = run_timeseries(net, td, steps=96)

.. note::

   SimBench profiles cover a full year at 15-minute resolution. Pass
   ``steps`` to simulate only the first part of the horizon, as above.

Cross-checking against pandapower
---------------------------------

The warning at the top of this page asks you to verify converted networks
against the original, which for SimBench means running ``pp.runpp`` on the
fetched net. Known trap: simbench 1.6.1 fills the ZIP load columns of
pandapower 3.x nets (``const_z_p_percent``, ``const_i_p_percent``,
``const_z_q_percent``, ``const_i_q_percent``) with NaN, which makes the
Newton iteration diverge with ``LoadflowNotConverged`` and a misleading
``pp.diagnostic`` hint. The monee import is unaffected (it does not run a
pandapower power flow), only your reference comparison breaks. Zero the
columns before calling ``runpp``:

.. code-block:: python

    const_cols = [c for c in pp_net.load.columns if c.startswith("const_")]
    pp_net.load[const_cols] = pp_net.load[const_cols].fillna(0.0)

Alternatively ``pp.runpp(pp_net, voltage_depend_loads=False)`` ignores the
ZIP columns entirely; both give the constant power behaviour the converter
assumes.

See :doc:`timeseries` for everything you can do with the resulting
:class:`~monee.simulation.TimeseriesResult`.

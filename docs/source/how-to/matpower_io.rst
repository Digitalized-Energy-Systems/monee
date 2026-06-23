==========================
Import MATPOWER case files
==========================

monee reads MATPOWER case files, both the ``.m`` text format that ``savecase``
writes and the binary ``.mat`` format, and converts them into a
:class:`~monee.model.Network`. Use this to load standard IEEE test cases
(case9, case30, case118) or any network exported from MATPOWER or MATLAB.

.. note::

   The MATPOWER import only supports electrical (AC) networks. For gas or
   water networks, build the model directly using the :mod:`monee.express` API
   or load from the native JSON format.

----

Reading a MATPOWER file
=======================

.. code-block:: python

    from monee.io.matpower import read_matpower_case

    net = read_matpower_case("case9.m")    # or "case9.mat"

The reader dispatches on the file extension: ``.m`` is parsed as MATPOWER text,
anything else is loaded as a binary ``.mat`` struct. The ``.m`` parser reads the
literal numeric matrices that ``savecase`` emits; it does not evaluate MATLAB
expressions inside a matrix (something ``savecase`` never produces).

The returned :class:`~monee.model.Network` is ready for simulation:

.. code-block:: python

    from monee import run_energy_flow

    result = run_energy_flow(net)

How MATPOWER data is mapped
---------------------------

The importer translates the MATPOWER ``mpc`` struct into monee model classes.
Know these conventions before you optimise:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - MATPOWER data
     - monee mapping
   * - ``bus`` rows
     - One :class:`~monee.model.node.Bus` per row. The bus voltage magnitude
       seeds a controllable ``vm_pu`` variable; the bus angle seeds the
       ``va_radians`` decision :class:`~monee.model.core.Var` (bounded
       :math:`\pm\pi`). ``va_degree`` is not set from the case; it is a
       derived report value (:class:`~monee.model.core.PostProcess`) computed
       from ``va_radians`` after each solve.
   * - ``bus`` demand (``Pd``/``Qd``)
     - A :class:`~monee.model.child.PowerLoad` child at the bus, or a
       :class:`~monee.model.child.PowerGenerator` when ``Pd < 0`` (negative
       demand encodes generation).
   * - ``gen`` rows
     - An :class:`~monee.model.child.ExtPowerGrid` (slack) when ``Pg == 0`` and
       ``Pg != Pmax``. Then ``p_mw`` and ``q_mvar`` become free variables and
       ``vm_pu`` is pinned to the generator voltage setpoint. All other rows
       become fixed :class:`~monee.model.child.PowerGenerator` children.
   * - ``branch`` rows
     - A :class:`~monee.model.branch.GenericPowerBranch` with the pi-model
       parameters of the row: charging susceptance split evenly between both
       ends, ``tap = 1`` when the MATPOWER tap is ``0``, and the phase shift
       converted to radians.

.. note::

   Imported branches get ``max_i_ka = 999`` (effectively unbounded
   thermal limits), because MATPOWER cases carry MVA ratings rather than
   current limits. Set realistic ``max_i_ka`` values on the branch models
   before running any optimisation that is constrained by line loading.

----

Saving and loading the native format
=====================================

monee also has a lightweight JSON native format (OMEF, Open Multi-Energy
Format). Use it to persist a network between sessions without depending on
MATPOWER:

.. code-block:: python

    from monee.io.native import write_omef_network, load_to_network

    # Save to disk
    write_omef_network("my_network.json", net)

    # Load back
    net2 = load_to_network("my_network.json")

To keep the serialized form in memory (for custom storage back-ends, message
passing, or diffing), use the dict-level pair
:func:`~monee.io.native.network_to_native_dict` and
:func:`~monee.io.native.native_dict_to_network` instead of the file-based
functions:

.. code-block:: python

    from monee.io.native import network_to_native_dict, native_dict_to_network

    data = network_to_native_dict(net)   # plain JSON-friendly dict
    net2 = native_dict_to_network(data)  # rebuild the Network

.. tip::

   The native format preserves all node, branch, child, and compound models
   including their parameter values. It does not preserve solved variable
   values: run the energy flow again after loading if you need results.

Format details
--------------

The on-disk structure is version 2
(:data:`~monee.io.native.FORMAT_VERSION`), bumped whenever the structure
changes in a non-additive way. Each model attribute is encoded with full
fidelity:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Attribute type
     - Round-trip behaviour
   * - :class:`~monee.model.core.Var`
     - Stored with ``value``, ``max``, ``min``, ``integer`` flag, and ``name``;
       decision variables keep their bounds exactly.
   * - :class:`~monee.model.core.Const` / :class:`~monee.model.core.Intermediate`
     - Tagged with a reserved ``__type__`` key so pinned constants and
       solver-computed intermediates are restored as the correct type.
   * - :class:`~monee.model.core.PostProcess`
     - Also tagged via ``__type__``, but only the computed value is
       stored: the computation lambda is not serializable and is re-attached
       automatically on the next solve.
   * - Plain scalars, lists, dicts
     - Pass through unchanged.

The reader stays backward compatible: legacy *untagged*
``{"value", "max", "min"}`` dicts (written by older versions and by the
MATPOWER importer) decode to a :class:`~monee.model.core.Var`.

.. note::

   **What is NOT serialized.** The native format captures the network's
   components and their state, but not behaviour attached to the network
   object itself:

   * network-level ``constraint()`` / ``objective()`` callables,
   * extensions registered via :meth:`~monee.model.network.Network.add_extension`
     (linepack, LTC, islanding configuration),
   * non-default formulations applied with
     :meth:`~monee.model.network.Network.apply_formulation`; the defaults are
     re-derived on load.

   Re-register constraints, objectives, and extensions and re-apply your
   formulation after :func:`~monee.io.native.load_to_network`.

Custom models
-------------

Deserialization resolves model classes by name through the registry that the
:func:`~monee.model.model` decorator populates. Custom model classes therefore
round-trip only when decorated:

.. code-block:: python

    import monee.model as mm

    @mm.model
    class MyLoad(mm.ChildModel):
        ...

Loading a file that references an unregistered class raises
:class:`~monee.io.native.PersistenceException` with this hint ("Maybe you
forgot to decorate your model class with @model?"). Import the module defining
your custom models before calling
:func:`~monee.io.native.load_to_network`.

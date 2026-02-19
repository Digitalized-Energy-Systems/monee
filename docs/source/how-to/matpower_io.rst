==========================
Import MATPOWER case files
==========================

monee can read MATPOWER ``.mat`` case files and convert them into a
:class:`~monee.model.Network`. This is useful for quickly loading standard
IEEE test cases (case9, case30, case118, …) or any network exported from
MATPOWER or MATLAB.

.. note::

   The MATPOWER import only supports **electrical (AC) networks**. For gas or
   water networks, build the model directly using the :mod:`monee.express` API
   or load from the native JSON format.

----

Reading a MATPOWER file
=======================

.. code-block:: python

    from monee.io.matpower import read_matpower_case

    net = read_matpower_case("case9.mat")

The function returns a :class:`~monee.model.Network` ready for simulation:

.. code-block:: python

    from monee import run_energy_flow

    result = run_energy_flow(net)

----

Saving and loading the native format
=====================================

monee also supports a lightweight JSON-based native format (**OMEF** —
Open Multi-Energy Format). Use it to persist a network between sessions
without depending on MATPOWER:

.. code-block:: python

    from monee.io.native import write_omef_network, load_to_network

    # Save to disk
    write_omef_network("my_network.json", net)

    # Load back
    net2 = load_to_network("my_network.json")

.. tip::

   The native format preserves all node, branch, child, and compound models
   including their parameter values. It does **not** preserve solved variable
   values — run the energy flow again after loading if you need results.

===========
Quickstart
===========

The easiest way to get a multi-energy grid running is using the express-style API, which conceals the inner structure of the network (nodes, branches, childs).

.. testcode::

    from monee import mx, run_energy_flow

    # create multi-grid container the monee.Network
    net = mx.create_multi_energy_network()

    # electricity grid
    bus_0 = mx.create_bus(net)
    bus_1 = mx.create_bus(net)

    mx.create_line(net, bus_0, bus_1, 100, r_ohm_per_m=0.00007, x_ohm_per_m=0.00007)
    mx.create_ext_power_grid(net, bus_0)
    mx.create_power_load(net, bus_1, 0.1, 0)

    # water-based district heating grid
    junc_0 = mx.create_water_junction(net)
    junc_1 = mx.create_water_junction(net)
    junc_2 = mx.create_water_junction(net)

    mx.create_ext_hydr_grid(net, junc_0)
    mx.create_water_pipe(net, junc_0, junc_1, diameter_m=0.12, length_m=100)
    mx.create_sink(net, junc_2, mass_flow=1)

    # creating connection between el and water grid
    mx.create_p2h(net, bus_1, junc_1, junc_2, heat_energy_mw=0.1, diameter_m=0.1, efficiency=0.9)

    # execute an energy flow calculating the energy flow for the whole MES
    result = run_energy_flow(net)


Object Oriented API
------------------------

For extendability monee has a second API, the object oriented API, which can be used to use custom defined nodes, childs and branches.


.. testcode::

    import monee.model as mm

    # altering the behavior  of mm.Powerload
    @mm.model
    class IndividualPowerLoadModel(mm.PowerLoad):
        def __init__(self, c, **kwargs) -> None:
            super().__init__(mm.Var(1), mm.Var(1), **kwargs) # p_mw, q_mvar are variables now
            self._c = c

        def equations(self, grid, node, **kwargs):
            return [
                # your equations
                self.p_mw == self._c * 10**6,
                self.q_mvar == self._c * 10**-6
            ]

    pn = mm.Network(mm.PowerGrid(name="power", sn_mva=1))

    node_2 = pn.node(
        mm.Bus(base_kv=1), # All equations, calculation defining the bus beavior are implemented in the class Bus
        child_ids=[pn.child(IndividualPowerLoadModel(c=1))], # New Implementation of a load
        grid=mm.EL
    )



Next steps
------------------------

- See the :ref:`Tutorials` for end-to-end workflows.
- Explore the :ref:`API-reference`.

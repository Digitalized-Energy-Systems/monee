import monee.model as mm


# altering the behavior
@mm.model
class IndividualPowerLoadModel(mm.PowerLoad):
    def __init__(self, c, **kwargs) -> None:
        super().__init__(mm.Var(1), mm.Var(1), **kwargs)  # p_mw, q_mvar are variables
        self._c = c

    def equations(self, grid, node, **kwargs):
        return [
            # your equations
            self.p_mw == self._c * 10**6,
            self.q_mvar == self._c * 10**-6,
        ]


def test_new_model():
    # GIVEN
    pn = mm.Network(mm.PowerGrid(name="power", sn_mva=1))

    # WHEN
    node_2 = pn.node(
        mm.Bus(base_kv=1),
        child_ids=[pn.child(IndividualPowerLoadModel(c=1))],
        grid=mm.EL,
    )

    # THEN
    assert node_2 is not None

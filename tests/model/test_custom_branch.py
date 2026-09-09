import monee
import monee.model as mm
from monee import mx


@mm.model
class MinimalLinearValve(mm.BranchModel):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.mass_flow_pos_kgs = mm.Var(0.0, min=0, name="mass_flow_pos_kgs")
        self.mass_flow_neg_kgs = mm.Var(0.1, min=0, name="mass_flow_neg_kgs")

    def equations(self, grid, from_node_model, to_node_model, **kwargs):
        return [
            self.mass_flow_pos_kgs == 0,
            from_node_model.pressure_squared_pu - to_node_model.pressure_squared_pu
            == 0.01 * self.mass_flow_neg_kgs,
        ]


def test_branch_model_defaults_on_off():
    valve = MinimalLinearValve()

    assert valve.on_off == 1
    assert valve.vars["on_off"] == 1


def test_minimal_custom_branch_solves():
    net = mm.Network()
    gas = mx.gas_structure(net, diameter_m=0.3, length_m=500)
    seg = gas.line(3, sink_mass_flow=0.05)
    gas.attach_ext_grid(seg.first)
    net.remove_branch_between(seg.nodes[1], seg.nodes[2])
    net.branch(MinimalLinearValve(), seg.nodes[1], seg.nodes[2])

    result = monee.run_energy_flow(net, formulation="smooth_nlp")

    row = result.get(MinimalLinearValve).iloc[0]
    assert row["on_off"] == 1
    assert abs(row["mass_flow_neg_kgs"] - 0.05) < 1e-6
    assert row["mass_flow_pos_kgs"] < 1e-8

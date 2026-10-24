import pytest

import monee.model as mm
from monee.model.core import GenericModel, Node, component_list, model


def test_tracked_is_var_alias():
    assert mm.tracked is mm.Var


def test_lower_returns_bound_value_or_passthrough():
    assert mm.lower(mm.Var(5.0, min=2.0)) == 2.0
    assert mm.lower(mm.Var(5.0)) == 5.0  # unbounded -> value
    assert mm.lower(3.0) == 3.0  # plain number passes through


def test_model_decorator():
    # GIVEN
    class TestClass:
        pass

    # WHEN
    model(TestClass)

    # THEN
    assert TestClass in component_list


def test_generic_model_vars():
    # GIVEN
    class ConcModel(GenericModel):
        def __init__(self) -> None:
            self.public = "A"
            self._not_public = "B"

    # WHEN
    conc_model = ConcModel()

    # THEN
    assert "public" in conc_model.vars
    assert "_not_public" not in conc_model.vars
    assert "not_public" not in conc_model.vars


def test_node_base():
    # GIVEN
    default_node = Node(1, None, child_ids=None, constraints=None, grid=None)
    node = Node(1, None, child_ids=[1], constraints=[1], grid=None)

    # WHEN
    node.add_from_branch_id("from_branch")
    node.add_to_branch_id("to_branch")

    # THEN
    # None args default to empty lists
    assert default_node.constraints == []
    assert default_node.child_ids == []

    assert node.constraints == [1]
    assert node.child_ids == [1]

    assert node.from_branch_ids == ["from_branch"]
    assert node.to_branch_ids == ["to_branch"]


def test_unknown_kwargs_warn_and_are_kept_in_ext_data():
    with pytest.warns(UserWarning, match=r"PowerLoad ignored unknown keyword"):
        load = mm.PowerLoad(p_mw=1.0, q_mvar=0.0, cost=10.0, typo_kwarg=1)
    assert load._ext_data == {"cost": 10.0, "typo_kwarg": 1}
    assert "cost" not in load.vars


def test_known_kwargs_do_not_warn():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mm.PowerLoad(p_mw=1.0, q_mvar=0.0)
        mm.Junction()


def test_consume_hydr_grid_documents_its_input_only_columns():
    child = mm.ConsumeHydrGrid()

    assert isinstance(child.mass_flow_kgs, mm.Var)
    # pressure_pu / t_k are plain inputs echoed back into the result tables,
    # never solved values.
    assert child.pressure_pu == 1
    assert child.t_k == 293

    doc = mm.ConsumeHydrGrid.__doc__
    assert "inputs echoed back" in doc
    assert "junction table" in doc
    assert "express_structures" in doc


def test_scale_duty_matches_a_heat_exchanger_built_for_the_part_load():
    hx = mm.HeatExchanger(q_mw=-0.4)
    part = mm.HeatExchanger(q_mw=-0.1)

    hx.scale_duty(0.25)

    assert hx.q_mw_set == pytest.approx(part.q_mw_set)
    assert hx.mass_flow_design_kgs == pytest.approx(part.mass_flow_design_kgs)
    assert hx._q_design_mw == pytest.approx(part._q_design_mw)
    assert hx._mass_flow_seed_kgs == pytest.approx(part._mass_flow_seed_kgs)


def test_set_pushes_a_compound_attribute_onto_its_control_node():
    import monee.express as mx

    net = mx.create_multi_energy_network()
    bus = mx.create_bus(net)
    cold, hot = mx.create_water_junction(net), mx.create_water_junction(net)
    compound = net.compound_by_id(
        mx.create_p2h(
            net,
            bus,
            cold_node_id=cold,
            hot_node_id=hot,
            heat_energy_mw=1.0,
            diameter_m=0.2,
            efficiency=3.0,
        )
    )
    node = next(c for c in compound.subcomponents if isinstance(c, Node))

    assert compound.set(regulation=0.3) is compound
    assert compound.model.regulation == 0.3
    assert node.model.regulation == 0.3


@pytest.mark.parametrize(
    "use",
    [
        lambda r: r >= 338.15,
        lambda r: 338.15 <= r,
        lambda r: r - 338.15,
        lambda r: 2 * r,
        lambda r: -r,
    ],
)
def test_a_post_process_in_an_expression_names_its_decision_variable(use):
    """Regression: ``m.t_k >= 338.15`` in a constraint raised the bare
    "'>=' not supported between instances of 'PostProcess' and 'float'"."""
    from monee.model.core import PostProcess

    t_k = PostProcess(lambda v, tref=356.0: v.t_pu * tref)

    with pytest.raises(TypeError, match="computed from instead: t_pu"):
        use(t_k)

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

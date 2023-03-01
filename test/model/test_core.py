
from monee.model.core import *

def test_model_decorator():

    @model
    class TestClass:
        pass

    assert TestClass in component_list

def test_generic_model_vars():
    class ConcModel(GenericModel):
        def __init__(self) -> None:
            self.public = "A"
            self._not_public = "B"
    
    model = ConcModel()
    assert "public" in model.vars
    assert "_not_public" not in model.vars
    assert "not_public" not in model.vars


def test_node_base():
    node = Node(1, None, child_ids=None, constraints=None, grid=None)

    assert node.constraints == []
    assert node.child_ids == []

    node = Node(1, None, child_ids=[1], constraints=[1], grid=None)

    assert node.constraints == [1]
    assert node.child_ids == [1]

    node.add_from_branch("from_branch")
    node.add_to_branch("to_branch")

    assert node.from_branches == ["from_branch"]
    assert node.to_branches == ["to_branch"]

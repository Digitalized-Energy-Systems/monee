
from monee.model.core import model, component_list, GenericModel

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
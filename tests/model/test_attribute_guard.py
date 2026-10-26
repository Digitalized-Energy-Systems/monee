import copy
import warnings

import pytest

import monee.express as mx
import monee.model as mm
from monee.model.core import (
    UnknownAttributeWarning,
    attribute_guard,
    register_optional_attributes,
    set_attribute_guard,
)


@pytest.fixture(autouse=True)
def restore_guard():
    previous = attribute_guard()
    yield
    set_attribute_guard(previous)


class _StubCompound(mm.CompoundModel):
    def __init__(self, p_mw) -> None:
        super().__init__()
        self.p_mw = p_mw

    def create(self, network, connect_node):
        self.child_id = network.child_to(
            mm.PowerLoad(p_mw=self.p_mw, q_mvar=0), connect_node.id
        )


def test_unknown_public_attribute_warns_and_names_the_alternatives():
    gen = mm.PowerGenerator(p_mw=1, q_mvar=0)

    with pytest.warns(UnknownAttributeWarning) as caught:
        gen.p_mw_max = 5.0

    message = str(caught[0].message)
    assert "PowerGenerator" in message
    assert "'p_mw_max'" in message
    assert "'p_mw'" in message and "'q_mvar'" in message
    assert "_p_mw_max" in message
    assert "concepts/data_model" in message
    assert gen.p_mw_max == 5.0


def test_underscore_names_are_never_guarded():
    gen = mm.PowerGenerator(p_mw=1, q_mvar=0)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UnknownAttributeWarning)
        gen._priority_weight = 3.0

    assert gen._priority_weight == 3.0
    assert "_priority_weight" not in gen.vars


def test_attributes_declared_during_init_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error", UnknownAttributeWarning)
        line = mm.PowerLine(length_m=1, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5, parallel=1)
        junction = mm.Junction()

    assert "length_m" in line.vars
    assert junction is not None


def test_property_setter_does_not_warn():
    gen = mm.GridFormingGenerator(p_mw_max=1.2, q_mvar_max=1.2)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UnknownAttributeWarning)
        gen.p_mw_max = 0.4

    assert gen.p_mw.max == 0.4


def test_constructor_parameter_stays_settable_after_construction():
    gen = mm.PowerGenerator(p_mw=1, q_mvar=0)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UnknownAttributeWarning)
        gen.cost = 2.5

    assert gen.cost == 2.5


def test_declaring_solver_state_does_not_warn():
    gen = mm.PowerGenerator(p_mw=1, q_mvar=0)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UnknownAttributeWarning)
        gen.extra_var = mm.Var(1.0)
        gen.extra_const = mm.Const(2.0)

    assert "extra_var" in gen.vars


def test_registered_optional_attribute_is_quiet():
    class _TaggedLoad(mm.PowerLoad):
        pass

    load = _TaggedLoad(p_mw=1, q_mvar=0)
    register_optional_attributes(_TaggedLoad, "asset_tag")

    with warnings.catch_warnings():
        warnings.simplefilter("error", UnknownAttributeWarning)
        load.asset_tag = "L-17"

    assert load.asset_tag == "L-17"
    with pytest.warns(UnknownAttributeWarning):
        mm.PowerLoad(p_mw=1, q_mvar=0).asset_tag = "L-18"


def test_error_mode_raises_and_reports_the_previous_mode():
    gen = mm.PowerGenerator(p_mw=1, q_mvar=0)
    previous = set_attribute_guard("error")

    assert previous == "warn"
    assert attribute_guard() == "error"
    with pytest.raises(AttributeError, match="does not declare the attribute"):
        gen.p_mw_max = 5.0


def test_off_mode_removes_the_hook_and_still_assigns():
    set_attribute_guard("off")
    gen = mm.PowerGenerator(p_mw=1, q_mvar=0)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UnknownAttributeWarning)
        gen.p_mw_max = 5.0

    assert "__setattr__" not in vars(mm.GenericModel)
    assert gen.p_mw_max == 5.0


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown attribute guard mode"):
        set_attribute_guard("loud")


def test_compound_create_may_record_ids_but_later_writes_are_guarded():
    net = mm.Network(mm.PowerGrid(name="power", sn_mva=1))
    bus = net.node(mm.Bus(base_kv=1))
    compound = _StubCompound(p_mw=0.1)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UnknownAttributeWarning)
        net.compound(compound, connect_node_id=bus)

    assert compound.child_id is not None
    with pytest.warns(UnknownAttributeWarning):
        compound.p_mw_setpoint = 0.2


def test_guard_survives_a_model_copy():
    gen = mm.PowerGenerator(p_mw=1, q_mvar=0)
    clone = copy.deepcopy(gen)

    with pytest.warns(UnknownAttributeWarning):
        clone.p_mw_max = 5.0


def test_solving_a_network_emits_no_guard_warnings():
    from monee import run_energy_flow

    net = mm.Network(mm.PowerGrid(name="power", sn_mva=1))
    bus_a = mx.create_bus(net, base_kv=1)
    bus_b = mx.create_bus(net, base_kv=1)
    mx.create_line(net, bus_a, bus_b, length_m=100, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)
    mx.create_ext_power_grid(net, bus_a)
    mx.create_power_load(net, bus_b, p_mw=0.05, q_mvar=0.0)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = run_energy_flow(net)

    assert result.success
    assert [w for w in caught if issubclass(w.category, UnknownAttributeWarning)] == []


def test_the_arming_marker_never_reaches_a_saved_file():
    from monee.io.native import model_to_dict

    gen = mm.PowerGenerator(p_mw=1, q_mvar=0)

    assert "_attrs_locked" not in model_to_dict(gen)

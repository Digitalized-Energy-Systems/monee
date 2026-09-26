"""``cost`` on gas and heat supply models: stored only when given, settable
without the unknown-attribute warning, forwarded by the express creators."""

import warnings

import pytest

import monee.express as mx
import monee.model as mm
from monee.model.core import UnknownAttributeWarning


@pytest.mark.parametrize(
    "build",
    [
        lambda cost: mm.Source(0.1, cost=cost),
        lambda cost: mm.ExtHydrGrid(cost=cost),
        lambda cost: mm.HeatGenerator(0.2, cost=cost),
        lambda cost: mm.HeatExchangerGenerator(0.2, cost=cost),
        lambda cost: mm.PassiveHeatExchangerGenerator(0.2, 0.1, cost=cost),
    ],
    ids=["Source", "ExtHydrGrid", "HeatGenerator", "HX", "PassiveHX"],
)
def test_cost_stored_only_when_given(build):
    assert build(4.0).cost == 4.0
    assert not hasattr(build(None), "cost")


@pytest.mark.parametrize(
    "model",
    [
        mm.Source(0.1),
        mm.ExtHydrGrid(),
        mm.HeatGenerator(0.2),
        mm.HeatExchanger(q_mw=0.1),
        mm.PassiveHeatExchanger(q_mw=0.1, diameter_m=0.1),
    ],
    ids=[
        "Source",
        "ExtHydrGrid",
        "HeatGenerator",
        "HeatExchanger",
        "PassiveHeatExchanger",
    ],
)
def test_cost_settable_without_unknown_attribute_warning(model):
    with warnings.catch_warnings():
        warnings.simplefilter("error", UnknownAttributeWarning)
        model.cost = 3.0


def test_express_forwards_cost():
    net = mx.create_multi_energy_network()
    gj = mx.create_gas_junction(net)
    wj0, wj1 = mx.create_water_junction(net), mx.create_water_junction(net)

    created = {
        mx.create_gas_source(net, gj, mass_flow_kgs=0.1, cost=1): 1,
        mx.create_gas_ext_grid(net, gj, cost=2): 2,
        mx.create_water_ext_grid(net, wj0, cost=3): 3,
        mx.create_heat_generator(net, wj1, q_mw=0.1, cost=4): 4,
    }
    hx = mx.create_heat_exchanger(net, wj0, wj1, q_mw=-0.2, cost=6)
    passive = mx.create_passive_heat_exchanger(
        net, wj1, wj0, q_mw=-0.1, diameter_m=0.1, cost=7
    )

    for child_id, cost in created.items():
        assert net.child_by_id(child_id).model.cost == cost
    assert net.branch_by_id(hx).model.cost == 6
    assert net.branch_by_id(passive).model.cost == 7


@pytest.mark.parametrize(
    "create", [mx.create_heat_exchanger, mx.create_passive_heat_exchanger]
)
def test_consuming_exchanger_rejects_cost(create):
    net = mx.create_multi_energy_network()
    j0, j1 = mx.create_water_junction(net), mx.create_water_junction(net)
    kwargs = {"diameter_m": 0.1} if create is mx.create_passive_heat_exchanger else {}

    with pytest.raises(ValueError, match="consuming exchanger"):
        create(net, j0, j1, q_mw=0.2, cost=5, **kwargs)

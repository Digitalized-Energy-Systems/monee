"""Snapshot result frames stay positionally indexed; ``get(..., index='id')``
is the additive id-indexed access path (batch result-index, R1/R2)."""

import pandas
import pytest

import monee
import monee.express as mx
import monee.model as mm


def _interleaved_net():
    """Buses at node ids 0, 2, 4: the gas junctions in between make the Bus
    frame's row positions differ from the node ids."""
    net = mm.Network(mm.create_power_grid("pg", sn_mva=1))
    gas = mm.create_gas_grid("gg", type="lgas")
    n0 = mx.create_bus(net)
    mx.create_junction(net, grid=gas)
    n1 = mx.create_bus(net)
    mx.create_junction(net, grid=gas)
    n2 = mx.create_bus(net)
    mx.create_ext_power_grid(net, n0)
    mx.create_line(net, n0, n1, length_m=1000, r_ohm_per_m=1e-4, x_ohm_per_m=1e-5)
    mx.create_line(net, n1, n2, length_m=1000, r_ohm_per_m=1e-4, x_ohm_per_m=1e-5)
    mx.create_power_load(net, n2, p_mw=0.02, q_mvar=0.0)
    return net, [n0, n1, n2]


@pytest.fixture(scope="module")
def solved():
    net, node_ids = _interleaved_net()
    with pytest.warns(Warning):
        result = monee.run_energy_flow(net)
    return result, node_ids


def test_default_frame_is_unchanged_and_positional(solved):
    result, node_ids = solved

    # GIVEN/WHEN the documented default access
    default = result.get(mm.Bus)

    # THEN it is the stored frame itself, positionally indexed
    assert default is result.dataframes["Bus"]
    assert isinstance(default.index, pandas.RangeIndex)
    assert list(default.index) == [0, 1, 2]
    assert list(default["id"]) == node_ids


def test_id_index_selects_by_component_id(solved):
    result, node_ids = solved
    default = result.get(mm.Bus)

    by_id = result.get(mm.Bus, index="id")

    assert by_id.index.name == "id"
    assert list(by_id.index) == node_ids
    for node_id in node_ids:
        assert by_id.loc[node_id, "id"] == node_id
    # the frame itself is untouched: same columns, same rows, same values
    assert list(by_id.columns) == list(default.columns)
    assert by_id.reset_index(drop=True).equals(default)
    assert default is result.dataframes["Bus"]


def test_branch_ids_become_a_named_multiindex(solved):
    result, _ = solved
    default = result.get(mm.PowerLine)
    key = default["id"].iloc[0]

    by_id = result.get(mm.PowerLine, index="id")

    assert isinstance(by_id.index, pandas.MultiIndex)
    assert by_id.index.names == ["from_node_id", "to_node_id", "key"]
    # the pandas trap the positional frame leaves open
    with pytest.raises(pandas.errors.IndexingError):
        default.set_index("id").loc[key]
    assert by_id.loc[key, "id"] == key
    assert by_id.loc[key, "p_from_mw"] == default["p_from_mw"].iloc[0]


def test_child_frame_is_id_indexed_and_keeps_node_id(solved):
    result, node_ids = solved

    loads = result.get(mm.PowerLoad, index="id")

    assert loads.index.name == "id"
    assert list(loads["node_id"]) == [node_ids[2]]


def test_unknown_index_mode_raises_naming_both_modes(solved):
    result, _ = solved

    with pytest.raises(ValueError, match="not a result index mode"):
        result.get(mm.Bus, index="ids")


def test_missing_model_type_frame_survives_id_indexing(solved):
    result, _ = solved

    assert result.get(mm.HeatExchanger, index="id").empty


def test_printed_result_points_at_the_id_column(solved):
    result, _ = solved

    text = str(result)

    assert "the 'id' column holds the component id" in text
    assert "index='id'" in text
    text.encode("ascii")
    assert "the 'id' column holds the component id" in result._repr_html_()


@pytest.mark.parametrize(
    "ids,expected_names",
    [
        ([0, 1, 2], ["id"]),
        ([(0, 1), (1, 2)], [None, None]),
        ([(0, 1, 0), 3], ["id"]),
    ],
)
def test_id_indexed_index_shape(ids, expected_names):
    from monee.simulation.result_utils import id_indexed

    df = pandas.DataFrame({"id": ids, "v": range(len(ids))})

    indexed = id_indexed(df)

    assert list(indexed.index.names) == expected_names
    assert indexed.reset_index(drop=True).equals(df)


@pytest.mark.pptest
def test_id_index_joins_against_pp_bus_to_node():
    """A bus-bus switch fusion shifts the bus numbering, so a positional join
    against ``pp_bus_to_node`` picks the wrong row (or raises)."""
    import pandapower as pp

    from monee.io.from_pandapower import from_pandapower_net

    pp_net = pp.create_empty_network()
    buses = [pp.create_bus(pp_net, vn_kv=20.0, name=f"B{i}") for i in range(4)]
    pp.create_ext_grid(pp_net, buses[0])
    for a, b in ((0, 1), (2, 3)):
        pp.create_line_from_parameters(
            pp_net,
            buses[a],
            buses[b],
            length_km=1.0,
            r_ohm_per_km=0.1,
            x_ohm_per_km=0.1,
            c_nf_per_km=0.0,
            max_i_ka=0.4,
        )
    pp.create_switch(pp_net, buses[1], buses[2], et="b", closed=True)
    pp.create_load(pp_net, buses[3], p_mw=0.1, q_mvar=0.02)
    pp.runpp(pp_net)

    net = from_pandapower_net(pp_net)
    result = monee.run_energy_flow(net)
    by_id = result.get(mm.Bus, index="id")

    assert len(by_id) < len(pp_net.bus)  # fusion: fewer nodes than pp buses
    for bus, node_id in net.pp_bus_to_node.items():
        row = by_id.loc[node_id]
        assert row["id"] == node_id
        assert row["vm_pu"] == pytest.approx(pp_net.res_bus.vm_pu[bus], abs=1e-4)

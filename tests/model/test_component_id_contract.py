import monee.express as mx


def build_net():
    net = mx.create_multi_energy_network()
    b0, b1 = mx.create_bus(net), mx.create_bus(net)
    mx.create_line(net, b0, b1, 100, r_ohm_per_m=1e-4, x_ohm_per_m=1e-4)
    mx.create_line(net, b0, b1, 100, r_ohm_per_m=1e-4, x_ohm_per_m=1e-4)
    mx.create_power_load(net, b1, p_mw=0.1, q_mvar=0.0)
    return net


def test_branch_id_is_from_to_key_tuple():
    net = build_net()
    assert [b.id for b in net.branches] == [(0, 1, 0), (0, 1, 1)]


def test_node_and_child_counters_are_independent():
    net = build_net()
    node_ids = {n.id for n in net.nodes}
    child_ids = {c.id for c in net.childs}
    assert 0 in node_ids
    assert 0 in child_ids


def test_result_frames_expose_id_tuple_and_child_node_id():
    net = build_net()
    frames = net.as_dataframe_dict()
    assert list(frames["PowerLine"]["id"]) == [(0, 1, 0), (0, 1, 1)]
    assert list(frames["PowerLoad"]["node_id"]) == [1]

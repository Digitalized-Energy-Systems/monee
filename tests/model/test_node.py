from monee.model.branch import GasPipe, GenericPowerBranch
from monee.model.child import PowerLoad
from monee.model.node import Bus, Intermediate, Junction, Var


def test_bus_vars():
    bus = Bus(base_kv=1)

    assert type(bus.p_mw) is Intermediate
    assert type(bus.q_mvar) is Intermediate
    assert type(bus.vm_pu) is Var
    assert type(bus.va_degree) is Var


def test_bus_eq():
    bus = Bus(base_kv=1)
    to_model = GenericPowerBranch(1, 0, 0, 0, 0, 0, 0, 0)
    to_model.p_to_mw = 10
    to_model.q_to_mvar = 2
    from_model = GenericPowerBranch(1, 0, 0, 0, 0, 0, 0, 0)
    from_model.p_from_mw = 20
    from_model.q_from_mvar = 5

    ap, rp = bus.calc_signed_power_values(
        to_branch_models=[to_model],
        from_branch_models=[from_model],
        connected_node_models=[],
    )

    assert ap == [20, 10]
    assert rp == [5, 2]

    bus.p_mw = 30
    bus.q_mvar = 7
    r1 = bus.p_mw_equation([])
    r2 = bus.q_mvar_equation([])

    assert r1
    assert r2


def test_bus_eq_with_child():
    bus = Bus(base_kv=1)
    to_model = GenericPowerBranch(1, 0, 0, 0, 0, 0, 0, 0)
    to_model.p_to_mw = 10
    to_model.q_to_mvar = 2
    child_model = PowerLoad(p_mw=11, q_mvar=12)

    ap, rp = bus.calc_signed_power_values(
        to_branch_models=[to_model],
        from_branch_models=[],
        connected_node_models=[child_model],
    )

    assert ap == [10, 11]
    assert rp == [2, 12]


def test_junction_vars():
    junction = Junction()

    assert type(junction.pressure_pa) is Intermediate
    assert type(junction.t_k) is Intermediate


def test_junction_mass_flow():
    junction = Junction()

    to_model = GasPipe(diameter_m=10, length_m=10, temperature_ext_k=234, roughness=1)
    to_model.to_mass_flow = 10
    from_model = GasPipe(diameter_m=10, length_m=10, temperature_ext_k=234, roughness=1)
    from_model.from_mass_flow = 3

    mass_flow = junction.calc_signed_mass_flow(
        to_branch_models=[to_model],
        from_branch_models=[from_model],
        connected_node_models=[],
    )

    assert mass_flow[0] == 3
    assert mass_flow[1] == 10

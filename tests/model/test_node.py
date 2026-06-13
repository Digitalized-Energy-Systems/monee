from monee.model.branch import GasPipe, GenericPowerBranch
from monee.model.child import PowerLoad
from monee.model.node import Bus, Junction


def test_bus_eq():
    # GIVEN
    bus = Bus(base_kv=1)
    to_model = GenericPowerBranch(1, 0, 0, 0, 0, 0, 0, 0)
    to_model.p_to_mw = 10
    to_model.q_to_mvar = 2
    from_model = GenericPowerBranch(1, 0, 0, 0, 0, 0, 0, 0)
    from_model.p_from_mw = 20
    from_model.q_from_mvar = 5
    bus.p_mw = 30
    bus.q_mvar = 7

    # WHEN
    ap, rp = bus.calc_signed_power_values(
        to_branch_models=[to_model],
        from_branch_models=[from_model],
        child_models=[],
    )
    r1 = bus.p_mw_equation([])
    r2 = bus.q_mvar_equation([])

    # THEN
    assert ap == [20, 10]
    assert rp == [5, 2]

    assert r1
    assert r2


def test_bus_eq_with_child():
    # GIVEN
    bus = Bus(base_kv=1)
    to_model = GenericPowerBranch(1, 0, 0, 0, 0, 0, 0, 0)
    to_model.p_to_mw = 10
    to_model.q_to_mvar = 2
    child_model = PowerLoad(p_mw=11, q_mvar=12)

    # WHEN
    ap, rp = bus.calc_signed_power_values(
        to_branch_models=[to_model],
        from_branch_models=[],
        child_models=[child_model],
    )

    # THEN
    assert ap == [10, 11]
    assert rp == [2, 12]


def test_junction_mass_flow():
    # GIVEN
    junction = Junction()
    to_model = GasPipe(diameter_m=10, length_m=10, temperature_ext_k=234, roughness_m=1)
    to_model.to_mass_flow_kgs = 10
    from_model = GasPipe(diameter_m=10, length_m=10, temperature_ext_k=234, roughness_m=1)
    from_model.from_mass_flow_kgs = 3

    # WHEN
    mass_flow_kgs = junction.calc_signed_mass_flow(
        to_branch_models=[to_model],
        from_branch_models=[from_model],
        child_models=[],
    )

    # THEN
    assert mass_flow_kgs[0] == 3
    assert mass_flow_kgs[1] == 10

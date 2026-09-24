import pytest

import monee.model as mm
from monee.io.from_cim import cim_objects_to_network


def _obj(cls_name, **attrs):
    obj = type(cls_name, (), {})()
    for key, value in attrs.items():
        setattr(obj, key, value)
    return obj


def _terminal(tn, seq=1):
    return _obj("Terminal", TopologicalNode=tn, sequenceNumber=seq)


def _tn(mrid, base_kv=20.0):
    return _obj(
        "TopologicalNode",
        mRID=mrid,
        name=mrid,
        BaseVoltage=_obj("BaseVoltage", nominalVoltage=base_kv),
    )


def _import(objects):
    with pytest.warns(UserWarning, match="experimental"):
        return cim_objects_to_network(objects)


def test_consuming_machine_stays_consuming():
    tn = _tn("tn1")
    # SSH load reference: p > 0 means the machine consumes.
    machine = _obj("SynchronousMachine", p=5.0, q=1.0, Terminals=[_terminal(tn)])

    net, report = _import({"tn1": tn, "m1": machine})

    loads = [c for c in net.childs if isinstance(c.model, mm.PowerLoad)]
    assert report.loads == 1
    assert report.generators == 0
    assert len(loads) == 1
    assert loads[0].model.p_mw == pytest.approx(5.0)
    assert loads[0].model.q_mvar == pytest.approx(1.0)


def test_injecting_machine_maps_to_generator():
    tn = _tn("tn1")
    machine = _obj("SynchronousMachine", p=-10.0, q=-2.0, Terminals=[_terminal(tn)])

    net, report = _import({"tn1": tn, "m1": machine})

    gens = [c for c in net.childs if isinstance(c.model, mm.PowerGenerator)]
    assert report.generators == 1
    assert len(gens) == 1
    # load convention storage: generation is negative
    assert gens[0].model.p_mw == pytest.approx(-10.0)
    assert gens[0].model.q_mvar == pytest.approx(-2.0)


def test_equivalent_injection_mapped_by_sign():
    tn1, tn2 = _tn("tn1"), _tn("tn2")
    consuming = _obj("EquivalentInjection", p=3.0, q=0.5, Terminals=[_terminal(tn1)])
    injecting = _obj("EquivalentInjection", p=-4.0, q=0.0, Terminals=[_terminal(tn2)])

    net, report = _import({"tn1": tn1, "tn2": tn2, "e1": consuming, "e2": injecting})

    assert report.loads == 1
    assert report.generators == 1
    load = next(c for c in net.childs if isinstance(c.model, mm.PowerLoad))
    assert load.model.p_mw == pytest.approx(3.0)
    gen = next(c for c in net.childs if isinstance(c.model, mm.PowerGenerator))
    assert gen.model.p_mw == pytest.approx(-4.0)


def test_linear_shunt_compensator_mapped_to_power_shunt():
    from monee.model.child import PowerShunt

    tn = _tn("tn1", base_kv=10.0)
    shunt = _obj(
        "LinearShuntCompensator",
        bPerSection=1e-3,
        gPerSection=0.0,
        sections=2,
        Terminals=[_terminal(tn)],
    )

    net, report = _import({"tn1": tn, "s1": shunt})

    assert report.shunts == 1
    model = next(c.model for c in net.childs if isinstance(c.model, PowerShunt))
    # b_total * base_kv^2 = 2e-3 S * 100 kV^2 -> 0.2 MVAr at v = 1 p.u.
    assert model.bs_mvar == pytest.approx(0.2)
    assert model.gs_mw == pytest.approx(0.0)


def test_asynchronous_machine_mapped_to_load():
    tn = _tn("tn1")
    machine = _obj("AsynchronousMachine", p=1.5, q=0.7, Terminals=[_terminal(tn)])

    net, report = _import({"tn1": tn, "m1": machine})

    assert report.loads == 1
    load = next(c for c in net.childs if isinstance(c.model, mm.PowerLoad))
    assert load.model.p_mw == pytest.approx(1.5)
    assert load.model.q_mvar == pytest.approx(0.7)


def test_zero_impedance_line_skipped_with_reason():
    tn1, tn2 = _tn("tn1"), _tn("tn2")
    line = _obj(
        "ACLineSegment",
        r=0.0,
        x=0.0,
        Terminals=[_terminal(tn1, 1), _terminal(tn2, 2)],
    )

    net, report = _import({"tn1": tn1, "tn2": tn2, "l1": line})

    assert report.lines == 0
    assert len(net.branches) == 0
    assert any("singular" in reason for reason in report.skipped)


def test_missing_base_voltage_recorded():
    tn = _obj("TopologicalNode", mRID="tn1", name="tn1")

    _, report = _import({"tn1": tn})

    assert report.buses == 1
    assert any("BaseVoltage" in reason for reason in report.skipped)


def test_switched_out_shunt_with_zero_sections_contributes_nothing():
    from monee.model.child import PowerShunt

    tn = _tn("tn1", base_kv=10.0)
    shunt = _obj(
        "LinearShuntCompensator",
        bPerSection=1e-3,
        gPerSection=2e-4,
        sections=0,
        Terminals=[_terminal(tn)],
    )

    net, report = _import({"tn1": tn, "s1": shunt})

    assert report.shunts == 0
    assert not any(isinstance(c.model, PowerShunt) for c in net.childs)
    assert any("sections=0" in reason for reason in report.skipped)


def test_shunt_without_sections_attribute_defaults_to_one_section():
    from monee.model.child import PowerShunt

    tn = _tn("tn1", base_kv=10.0)
    shunt = _obj(
        "LinearShuntCompensator",
        bPerSection=1e-3,
        gPerSection=0.0,
        Terminals=[_terminal(tn)],
    )

    net, report = _import({"tn1": tn, "s1": shunt})

    assert report.shunts == 1
    model = next(c.model for c in net.childs if isinstance(c.model, PowerShunt))
    assert model.bs_mvar == pytest.approx(0.1)


def test_shunt_without_admittance_data_skipped():
    tn = _tn("tn1")
    shunt = _obj("LinearShuntCompensator", Terminals=[_terminal(tn)])

    _, report = _import({"tn1": tn, "s1": shunt})

    assert report.shunts == 0
    assert any("b/g" in reason for reason in report.skipped)

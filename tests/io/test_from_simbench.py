import pytest


@pytest.mark.pptest
def test_obtain_simbench_profile():
    from monee.io.from_simbench import obtain_simbench_profile

    # GIVEN
    simbench_code = "1-LV-rural3--1-no_sw"

    # WHEN
    td = obtain_simbench_profile(simbench_code)

    # THEN
    # loads register under the aggregated pandapower load name; other types keep raw names
    assert "LV3.101 Load 1" in td.child_name_data
    assert "PV1" in td.child_name_data

    attrs = td.child_name_data["LV3.101 Load 1"]
    assert "p_mw" in attrs and "q_mvar" in attrs

    # series is base p_mw scaled by the profile factor, not a raw 0..1 multiplier
    assert max(attrs["p_mw"]) > 0

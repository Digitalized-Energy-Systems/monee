import pytest


@pytest.mark.pptest
def test_obtain_simbench_profile():
    from monee.io.from_simbench import obtain_simbench_profile

    # GIVEN WHEN
    td = obtain_simbench_profile("1-LV-rural3--1-no_sw")

    # THEN
    # Loads are now registered under the aggregated pandapower load name
    # produced by ``aggregated_pp_load_name`` so that the series actually
    # bind to the corresponding monee :class:`PowerLoad` on import.  Non-load
    # profile types (renewables, storage, powerplants) still pass through
    # under their raw simbench profile-column name.
    assert "LV3.101 Load 1" in td.child_name_data
    assert "PV1" in td.child_name_data
    attrs = td.child_name_data["LV3.101 Load 1"]
    assert "p_mw" in attrs and "q_mvar" in attrs
    # The registered series is the base p_mw scaled by the simbench
    # profile factor: it must be a positive load curve, not a raw 0..1
    # multiplier.
    assert max(attrs["p_mw"]) > 0

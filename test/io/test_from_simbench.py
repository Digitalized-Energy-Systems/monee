from monee.io.from_simbench import obtain_simbench_profile


def test_obtain_simbench_profile():
    # GIVEN WHEN
    td = obtain_simbench_profile("1-LV-rural3--1-no_sw")

    # THEN
    assert len(td.name_data) == 32
    assert "G1-C" in td.name_data
    assert "PV1" in td.name_data

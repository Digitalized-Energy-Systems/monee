from types import SimpleNamespace

import pandas as pd
import pytest


def _fake_simbench_net():
    load = pd.DataFrame(
        {
            "bus": [0, 0],
            "name": ["L1", "L2"],
            "profile": ["H0", "H0"],
            "p_mw": [2.0, 4.0],
            "q_mvar": [1.0, 0.5],
            "scaling": [0.5, 1.0],
        }
    )
    profiles = {
        "load": pd.DataFrame(
            {
                "time": [0, 1],
                "H0_pload": [0.1, 0.2],
                "H0_qload": [0.3, 0.4],
            }
        )
    }
    return SimpleNamespace(load=load, profiles=profiles)


def test_profile_includes_per_load_scaling():
    from monee.io.from_simbench import obtain_simbench_profile_by_pp_net

    td = obtain_simbench_profile_by_pp_net(_fake_simbench_net())

    attrs = td.child_name_data["L1+L2"]
    # p: 2.0*0.5*profile + 4.0*1.0*profile ; q: 1.0*0.5*profile + 0.5*1.0*profile
    assert attrs["p_mw"] == pytest.approx([0.5, 1.0])
    assert attrs["q_mvar"] == pytest.approx([0.3, 0.4])


def test_missing_profiles_raises_clear_error():
    from monee.io.from_simbench import obtain_simbench_profile_by_pp_net

    net = SimpleNamespace(load=pd.DataFrame())
    with pytest.raises(ValueError, match="profiles"):
        obtain_simbench_profile_by_pp_net(net)


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

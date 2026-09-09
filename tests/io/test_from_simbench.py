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


def _fake_simbench_net_with_sgen():
    net = _fake_simbench_net()
    net.sgen = pd.DataFrame(
        {
            "bus": [1, 2],
            "name": ["SG1", "SG2"],
            "profile": ["PV1", "PV_missing"],
            "p_mw": [0.5, 0.2],
            "q_mvar": [0.0, 0.0],
            "scaling": [2.0, 1.0],
            "in_service": [True, True],
        }
    )
    net.profiles["renewables"] = pd.DataFrame(
        {"time": [0, 1], "PV1": [0.0, 0.25], "PV_unused": [0.5, 0.5]}
    )
    net.profiles["storage"] = pd.DataFrame({"time": [0, 1], "SOC1": [0.4, 0.6]})
    return net


@pytest.mark.pptest
def test_profile_includes_per_load_scaling():
    from monee.io.from_simbench import obtain_simbench_profile_by_pp_net

    td = obtain_simbench_profile_by_pp_net(_fake_simbench_net())

    attrs = td.child_name_data["L1+L2"]
    # p: 2.0*0.5*profile + 4.0*1.0*profile ; q: 1.0*0.5*profile + 0.5*1.0*profile
    assert attrs["p_mw"] == pytest.approx([0.5, 1.0])
    assert attrs["q_mvar"] == pytest.approx([0.3, 0.4])


@pytest.mark.pptest
def test_sgen_profile_binds_by_element_name_with_generation_sign():
    from monee.io.from_simbench import obtain_simbench_profile_by_pp_net

    td = obtain_simbench_profile_by_pp_net(_fake_simbench_net_with_sgen())

    # -profile * p_mw * scaling, the internal sign of PowerGenerator
    assert td.child_name_data["SG1"]["p_mw"] == pytest.approx([0.0, -0.25])
    assert "PV1" not in td.child_name_data
    # a column no element claims still passes through by raw name
    assert "PV_unused" in td.child_name_data
    # storage columns are state of charge, never p_mw
    assert "SOC1" not in td.child_name_data


@pytest.mark.pptest
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
    # loads register under the aggregated pandapower load name, generation under
    # the element name that from_pandapower_net puts on the child
    assert "LV3.101 Load 1" in td.child_name_data
    assert "LV3.101 SGen 1" in td.child_name_data
    assert max(td.child_name_data["LV3.101 SGen 1"]["p_mw"]) <= 0

    attrs = td.child_name_data["LV3.101 Load 1"]
    assert "p_mw" in attrs and "q_mvar" in attrs

    # series is base p_mw scaled by the profile factor, not a raw 0..1 multiplier
    assert max(attrs["p_mw"]) > 0


@pytest.mark.pptest
def test_obtain_simbench_net_with_td_suppresses_simbench_future_warnings():
    import warnings

    from monee.io.from_simbench import obtain_simbench_net_with_td

    filters_before = list(warnings.filters)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        net, td = obtain_simbench_net_with_td("1-LV-rural1--0-no_sw")

    assert net is not None and td is not None
    assert not [w for w in caught if issubclass(w.category, FutureWarning)]
    assert list(warnings.filters) == filters_before


def test_obtain_functions_have_docstrings():
    import monee.io.from_simbench as fs

    for fn in (
        fs.obtain_simbench_net,
        fs.obtain_simbench_profile,
        fs.obtain_simbench_net_with_td,
        fs.obtain_simbench_profile_by_pp_net,
    ):
        doc = fn.__doc__
        assert doc and doc.strip(), fn.__name__
        assert doc.isascii(), fn.__name__

    assert "pp_line_to_branch" in fs.obtain_simbench_net.__doc__
    assert "TimeseriesData" in fs.obtain_simbench_net_with_td.__doc__

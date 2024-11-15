import pytest

import monee.model as md
from monee.simulation.timeseries import run


@pytest.mark.pptest
def test_timeseries_with_simbench():
    from monee.io.from_simbench import obtain_simbench_net_with_td

    # GIVEN
    steps = 3
    net, td = obtain_simbench_net_with_td("1-LV-rural3--1-no_sw")

    # WHEN
    result = run(net, td, steps)

    # THEN
    assert len(result.raw) == steps
    assert len(result.get_result_for(md.PowerLoad, "p_mw")) == steps

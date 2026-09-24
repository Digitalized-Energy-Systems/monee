"""Pure water/heat networks on the default (CasADi/IPOPT) backend: the
documented DHS patterns must solve without self-inflicted warnings, while an
explicitly requested discrete formulation keeps its relaxation warning."""

import warnings

import monee.express as mx
import monee.model as mm
from monee import run_energy_flow


def _documented_ring(n=4):
    net = mm.Network()
    dhs = mx.dhs_structure(net, diameter_m=0.15, length_m=100)
    seg = dhs.ring(n, heat_exchanger_q_mw=0.01)
    dhs.attach_heat_plant(seg.supply.first, seg.return_.first, name="plant")
    return net


def _unidirectional_loop(n=3):
    net = mm.Network()
    dhs = mx.dhs_structure(net, diameter_m=0.15, length_m=100)
    seg = dhs.line(n, heat_exchanger_q_mw=0.01)
    dhs.attach_heat_plant(seg.supply.first, seg.return_.last, name="plant")
    return net


def _solve(net, **kw):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = run_energy_flow(net, **kw)
    return result, [str(w.message) for w in caught]


def test_documented_ring_solves_warning_free_on_defaults():
    result, messages = _solve(_documented_ring())

    assert not [m for m in messages if "relaxes integer variables" in m]
    assert not [m for m in messages if "Result validation" in m]
    assert result.warnings == []


def test_default_ring_matches_heat_nlp_temperatures():
    default_result, _ = _solve(_documented_ring())
    nlp_result, _ = _solve(_documented_ring(), formulation="heat_nlp")

    t_default = default_result.dataframes["Junction"]["t_k"]
    t_nlp = nlp_result.dataframes["Junction"]["t_k"]
    assert (t_default - t_nlp).abs().max() < 0.1


def test_unidirectional_loop_emits_no_relaxation_warning():
    result, messages = _solve(_unidirectional_loop())

    assert not [m for m in messages if "relaxes integer variables" in m]
    assert result.warnings == []


def test_explicit_discrete_formulation_keeps_the_relaxation_warning():
    _, messages = _solve(_documented_ring(), formulation="heat_nonconvex_miqcqp")

    relax = [m for m in messages if "relaxes integer variables" in m]
    assert relax
    assert "heat_nlp" in relax[0]

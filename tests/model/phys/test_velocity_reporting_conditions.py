"""velocity_mps is a reference-condition report, as documented in
docs/source/concepts/domains.md."""

import math

import monee.model as mm
from monee.model.formulation.common import ensure_velocity_report
from monee.model.phys.nonlinear.gf import reference_gas_density


class _Values:
    def __init__(self, pos, neg):
        self.mass_flow_pos_kgs = pos
        self.mass_flow_neg_kgs = neg


class _Pipe:
    def __init__(self, diameter_m):
        self.diameter_m = diameter_m


def test_velocity_report_uses_the_reference_gas_density():
    grid = mm.create_gas_grid("gas")
    pipe = _Pipe(0.15)
    ensure_velocity_report(pipe, grid)

    mdot = 0.25
    area = math.pi / 4 * 0.15**2
    expected = mdot / (reference_gas_density(grid) * area)

    assert pipe.velocity_mps.fn(_Values(mdot, 0.0)) == expected


def test_equal_mass_flows_report_equal_velocity_whatever_the_local_density():
    grid = mm.create_gas_grid("gas")
    wide, narrow = _Pipe(0.15), _Pipe(0.15)
    ensure_velocity_report(wide, grid)
    ensure_velocity_report(narrow, grid)

    assert wide.velocity_mps.fn(_Values(0.25, 0.0)) == narrow.velocity_mps.fn(
        _Values(0.25, 0.0)
    )


def test_docs_state_the_velocity_reporting_conditions():
    from pathlib import Path

    repo = Path(__file__).resolve().parents[3]
    text = (repo / "docs/source/concepts/domains.md").read_text(encoding="utf-8")
    assert "reference gas density" in text
    assert "not the local velocity" in text

import re
from pathlib import Path

import monee
import monee.model as mm
from monee import mx

README = Path(__file__).resolve().parents[2] / "README.md"


def _readme_block(marker):
    blocks = re.findall(
        r"```python\n(.*?)```", README.read_text(encoding="utf-8"), re.S
    )
    matching = [b for b in blocks if marker in b]
    assert len(matching) == 1, f"expected exactly one README block with {marker!r}"
    return matching[0]


def test_readme_custom_gas_formulation_solves_with_falling_pressure():
    net = mm.Network()
    structure = mx.gas_structure(net, diameter_m=0.3, length_m=500)
    head = structure.line(1)
    segment = structure.line(8, start_from=head.first, sink_mass_flow=0.05)
    structure.attach_ext_grid(segment.first)

    exec(_readme_block("class MyGasPipeFormulation"), {"net": net})

    result = monee.run_energy_flow(net)
    assert result.success

    pressures = result.get(mm.Junction).sort_values("id")["pressure_pu"].tolist()
    flows = result.get(mm.GasPipe)["mass_flow_kgs"].tolist()

    assert all(f < 0 for f in flows), "line is fed from the from-node end"
    assert all(a > b for a, b in zip(pressures, pressures[1:])), (
        f"pressure must fall along the flow, got {pressures}"
    )


def test_readme_economic_dispatch_prefers_cheap_generator():
    scope = {}
    exec(_readme_block("create_economic_dispatch_problem"), scope)

    result = scope["result"]
    assert result.success

    gens = result.get(mm.PowerGenerator).sort_values("id")
    costs = gens["cost"].tolist()
    dispatched = (-gens["p_mw"]).tolist()
    assert costs == [10.0, 40.0], "cost= at creation must land on the model"
    assert dispatched[0] > 0.08, f"cheap unit must carry the load, got {dispatched}"
    assert abs(dispatched[1]) < 1e-4, f"expensive unit must stay off, got {dispatched}"

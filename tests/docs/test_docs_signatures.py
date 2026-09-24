"""Pin docs blocks that restate a python signature or an API claim, so the
pages cannot drift from the code they describe."""

import inspect
import re
from pathlib import Path

import monee.express as mx
import monee.model as mm
from monee import Stepper
from monee.network import mes
from monee.solver import SolverResult

DOCS = Path(__file__).resolve().parents[2] / "docs" / "source"


def _documented_kwargs(text: str, marker: str) -> set[str]:
    block = text[text.index(marker) :]
    block = block[: block.index("\n   )")]
    return set(re.findall(r"^\s{7}(\w+)=", block, flags=re.MULTILINE))


def test_stepper_signature_block_lists_every_kwarg():
    text = (DOCS / "how-to" / "stepper.rst").read_text(encoding="utf-8")
    documented = _documented_kwargs(text, "   Stepper(\n")
    real = {
        name
        for name, param in inspect.signature(Stepper.__init__).parameters.items()
        if param.kind is inspect.Parameter.KEYWORD_ONLY
    }
    assert documented == real


def test_stepper_page_documents_change_recording():
    text = (DOCS / "how-to" / "stepper.rst").read_text(encoding="utf-8")
    assert "record_changes" in text
    assert "max_changes" in text
    assert "changes_df" in text


def test_plot_kwargs_documented_in_pandapower_page():
    text = (DOCS / "how-to" / "convert_from_pandapower.rst").read_text(encoding="utf-8")
    documented = set(re.findall(r"^\s{8}(\w+)=", text, flags=re.MULTILINE))
    real = {
        name
        for name in inspect.signature(SolverResult.plot).parameters
        if name != "self"
    }
    assert real <= documented


def test_generate_mes_page_names_the_builders_cigre_actually_uses():
    text = (DOCS / "how-to" / "generate_mes.rst").read_text(encoding="utf-8")
    source = inspect.getsource(mes.create_mv_multi_cigre)
    called = {
        name
        for name in ("create_gas_net_for_power", "create_heat_net_for_power")
        if name in source
    }
    assert called
    assert all(name in text for name in called)
    assert "generate_supply_return_mes_based_on_power_net" not in source
    assert "runs this machinery" not in text


def test_multi_energy_network_is_an_alias_as_documented():
    net = mx.create_multi_energy_network()
    assert type(net) is mm.Network
    index = (DOCS / "index.md").read_text(encoding="utf-8")
    assert "alias for `mm.Network()`" in index

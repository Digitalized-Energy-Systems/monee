"""The component reference (docs/source/components/) restates every
constructor and express signature by hand. These checks pin it to the code so
a renamed parameter, a new component or a new create function fails here
instead of drifting silently."""

import inspect
import re
from pathlib import Path

import pytest

import monee
import monee.express as mx
import monee.model as mm
from monee.model.core import component_list

DOCS = Path(__file__).resolve().parents[2] / "docs" / "source"
PAGES = sorted((DOCS / "components").glob("*.rst"))
TEXT = "\n\n".join(page.read_text(encoding="utf-8") for page in PAGES)

MARKER = re.compile(
    r"^\.\. component: (?P<objects>[^|]+)\|(?P<express>.*)$", re.MULTILINE
)
MARKER_PREFIXES = (".. component:", ".. results:")


def _region_after(start, end):
    """Directive lines (code block, tables) that follow a marker, up to the
    next marker or the next paragraph at column 0."""
    lines = []
    for line in TEXT[start:end].splitlines():
        if line.startswith(MARKER_PREFIXES) or (
            line and not line.startswith((" ", ".."))
        ):
            break
        lines.append(line)
    return "\n".join(lines)


def _resolve(name):
    for namespace in (mm, mx, monee):
        if hasattr(namespace, name):
            return getattr(namespace, name)
    raise AssertionError(
        f"{name} is not exported from monee.model, monee.express or monee"
    )


def _params(obj):
    target = obj.__init__ if inspect.isclass(obj) else obj
    return {
        name
        for name, p in inspect.signature(target).parameters.items()
        if name != "self"
        and p.kind
        not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
    }


def _entries():
    matches = list(MARKER.finditer(TEXT))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(TEXT)
        objects = [n.strip() for n in m.group("objects").split(",") if n.strip()]
        express = [n.strip() for n in m.group("express").split(",") if n.strip()]
        yield pytest.param(objects, express, _region_after(m.end(), end), id=objects[0])


ENTRIES = list(_entries())


@pytest.mark.parametrize("objects, express, region", ENTRIES)
def test_signature_blocks_name_every_real_parameter(objects, express, region):
    tokens = set(re.findall(r"[A-Za-z_]\w*", region))
    for name in objects + express:
        missing = _params(_resolve(name)) - tokens
        assert not missing, (
            f"{name}: parameters missing from the page: {sorted(missing)}"
        )


def _code_blocks(region):
    return "\n".join(
        block.split(".. list-table::")[0]
        for block in region.split(".. code-block:: python")[1:]
    )


@pytest.mark.parametrize("objects, express, region", ENTRIES)
def test_keyword_tokens_are_real_parameters(objects, express, region):
    real = set().union(*(_params(_resolve(name)) for name in objects + express))
    # Compound placeholders (``net.compound(mm.CHP(...), gas_node_id=...)``)
    # name the create() endpoints, which the express function also exposes.
    written = set(re.findall(r"\b([A-Za-z_]\w*)=", _code_blocks(region)))
    assert written <= real, (
        f"unknown keyword arguments on the page: {sorted(written - real)}"
    )


@pytest.mark.parametrize("objects, express, region", ENTRIES)
def test_parameter_table_matches_the_constructor(objects, express, region):
    rows = set()
    for cell in re.findall(r"^   \* - (.+)$", region, flags=re.MULTILINE):
        rows.update(re.findall(r"``(\w+)``", cell))
    rows.discard("Parameter")
    if not rows:
        pytest.skip("no parameter table")
    constructor = set().union(*(_params(_resolve(name)) for name in objects))
    allowed = constructor | set().union(*(_params(_resolve(name)) for name in express))
    assert constructor <= rows, (
        f"constructor parameters without a row: {sorted(constructor - rows)}"
    )
    assert rows <= allowed, (
        f"table rows that are not parameters: {sorted(rows - allowed)}"
    )


RESULTS = re.compile(r"^\.\. results: (?P<tables>.+)$", re.MULTILINE)
CONTAINER_COLUMNS = {
    "active",
    "id",
    "independent",
    "ignored",
    "node_id",
    "regulation",
    "on_off",
}


def _results_entries():
    matches = list(RESULTS.finditer(TEXT))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(TEXT)
        region = _region_after(m.end(), end)
        columns = set(re.findall(r"^   \* - ``(\w+)``$", region, flags=re.MULTILINE))
        for table in m.group("tables").split(","):
            yield pytest.param(table.strip(), columns, id=table.strip())


def _networks():
    net = mx.create_multi_energy_network()
    b0 = mx.create_bus(net, base_kv=10)
    b1 = mx.create_bus(net, base_kv=10)
    b2 = mx.create_bus(net, base_kv=0.4)
    mx.create_line(net, b0, b1, length_m=500, r_ohm_per_m=1e-4, x_ohm_per_m=1e-4)
    mx.create_trafo(net, b2, b1, vk_percent=6, vkr_percent=1, sn_trafo_mva=0.63)
    mx.create_el_branch(
        net, b0, b1, mm.GenericPowerBranch(1, 0, 0.01, 0.05, 0, 0, 0, 0)
    )
    mx.create_ext_power_grid(net, b0, cost=1.0)
    mx.create_power_load(net, b1, p_mw=0.2, q_mvar=0.05)
    mx.create_power_generator(net, b1, p_mw=0.1, q_mvar=0.0, cost=10)
    mx.create_el_child(net, mm.PowerShunt(gs_mw=0.0, bs_mvar=0.05), b1)
    mx.create_el_child(net, mm.VoltageControlledGenerator(p_mw=0.05), b2)
    mx.create_el_child(
        net, mm.ElectricStorage(e_mwh_initial=1, e_mwh_max=2, p_max_mw=0.5), b1
    )
    js, jm, jr = (mx.create_water_junction(net) for _ in range(3))
    mx.create_water_ext_grid(net, js)
    mx.create_water_pipe(net, js, jm, diameter_m=0.15, length_m=200)
    mx.create_heat_exchanger(net, jm, jr, q_mw=0.5)
    mx.create_heat_exchanger(net, jm, jr, q_mw=-0.5)
    mx.create_passive_heat_exchanger(net, jm, jr, q_mw=0.1, diameter_m=0.1)
    mx.create_passive_heat_exchanger(net, jm, jr, q_mw=-0.1, diameter_m=0.1)
    mx.create_water_sink(net, jr, mass_flow_kgs=0.5)
    mx.create_water_source(net, jr, mass_flow_kgs=0.2, t_k=340)
    mx.create_heat_generator(net, jm, q_mw=0.05)
    mx.create_heat_load(net, jr, q_mw=0.05)
    mx.create_consume_hydr_grid(net, jr, mass_flow_kgs=0.1)
    mx.create_water_child(net, mm.ThermalStorage(1000, 5000, 1), jm)
    g0, g1, g2 = (mx.create_gas_junction(net) for _ in range(3))
    mx.create_gas_ext_grid(net, g0)
    mx.create_gas_pipe(net, g0, g1, diameter_m=0.2, length_m=1000)
    mx.create_compressor(net, g1, g2, compression_ratio=1.1, max_flow_kgs=5)
    mx.create_gas_sink(net, g2, mass_flow_kgs=0.3)
    mx.create_gas_child(net, mm.GasStorage(100, 500, 1), g1)
    mx.create_p2h(net, b1, jm, jr, heat_energy_mw=0.1, diameter_m=0.1, efficiency=0.9)
    mx.create_g2p(net, g1, b1, efficiency=0.4, p_mw_setpoint=0.05)
    mx.create_p2g(net, b1, g1, efficiency=0.6, mass_flow_setpoint_kgs=0.001)
    mx.create_chp(
        net,
        b1,
        jm,
        jr,
        g1,
        diameter_m=0.1,
        efficiency_power=0.3,
        efficiency_heat=0.5,
        mass_flow_setpoint_kgs=0.002,
    )
    mx.create_g2h(net, g1, jm, jr, heat_energy_mw=0.05, diameter_m=0.1, efficiency=0.9)
    mx.create_chp_hg(
        net,
        b1,
        jm,
        g1,
        efficiency_power=0.3,
        efficiency_heat=0.5,
        mass_flow_setpoint_kgs=0.001,
    )
    mx.create_p2h_hg(net, b1, jm, heat_energy_mw=0.02, efficiency=0.95)
    mx.create_g2h_hg(net, g1, jm, heat_energy_mw=0.02, efficiency=0.9)
    mx.create_grid_forming_generator(net, b2, p_mw_max=1, q_mvar_max=1)
    mx.create_gas_grid_forming_source(net, g2)
    return net


@pytest.fixture(scope="module")
def frame_columns():
    return {
        name: set(df.columns) - CONTAINER_COLUMNS
        for name, df in _networks().as_result_dataframe_dict().items()
    }


@pytest.mark.parametrize("table, documented", _results_entries())
def test_results_table_lists_exactly_the_frame_columns(
    table, documented, frame_columns
):
    assert table in frame_columns, f"no result table named {table}"
    actual = frame_columns[table]
    assert documented == actual, (
        f"{table}: undocumented columns {sorted(actual - documented)}, "
        f"documented but absent {sorted(documented - actual)}"
    )


def test_every_registered_model_class_is_on_the_page():
    # Test modules register throwaway @model classes into the same list at
    # collection time; only the library's own classes belong on the page.
    missing = [
        cls.__name__
        for cls in component_list
        if cls.__module__.startswith("monee.")
        and not cls.__name__.startswith("_")
        and cls.__name__ not in TEXT
    ]
    assert not missing, missing


def test_every_express_create_function_is_on_the_page():
    missing = [
        name
        for name in dir(mx)
        if name.startswith("create_")
        and callable(getattr(mx, name))
        and name not in TEXT
    ]
    assert not missing, missing


def test_pages_are_in_the_toctrees():
    index = (DOCS / "index.md").read_text(encoding="utf-8")
    assert "\ncomponents/index\n" in index
    landing = (DOCS / "components" / "index.rst").read_text(encoding="utf-8")
    for page in PAGES:
        if page.stem != "index":
            assert f"\n   {page.stem}\n" in landing, page.stem

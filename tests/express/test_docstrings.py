import inspect

import pytest

from monee import mm, mx

FACTORIES = [
    obj
    for name, obj in vars(mx).items()
    if name.startswith("create_") and inspect.isfunction(obj)
]
BUILDERS = [mx.gas_structure, mx.water_structure, mx.el_structure, mx.dhs_structure]


@pytest.mark.parametrize("func", FACTORIES + BUILDERS, ids=lambda f: f.__name__)
def test_factory_has_ascii_docstring(func):
    assert func.__doc__ and func.__doc__.strip()
    func.__doc__.encode("ascii")


@pytest.mark.parametrize(
    ("func", "params"),
    [
        (mx.create_ext_power_grid, ["max_import_mw", "max_export_mw", "regulate_vm"]),
        (
            mx.create_ext_hydr_grid,
            [
                "max_import_kgs",
                "max_export_kgs",
                "pin_temperature",
                "free_pressure_bounds",
            ],
        ),
        (
            mx.create_gas_ext_grid,
            [
                "max_import_kgs",
                "max_export_kgs",
                "pin_temperature",
                "free_pressure_bounds",
            ],
        ),
        (
            mx.create_water_ext_grid,
            [
                "max_import_kgs",
                "max_export_kgs",
                "pin_temperature",
                "free_pressure_bounds",
            ],
        ),
        (mx.create_source, ["t_k"]),
        (mx.create_gas_source, ["t_k"]),
        (mx.create_water_source, ["t_k"]),
        (mx.gas_structure, ["diameter_m", "length_m", "roughness_m", "grid"]),
        (mx.water_structure, ["diameter_m", "length_m", "unidirectional", "grid"]),
        (mx.el_structure, ["r_ohm_per_m", "x_ohm_per_m", "base_kv", "grid"]),
        (mx.dhs_structure, ["diameter_m", "length_m", "unidirectional", "grid"]),
    ],
    ids=lambda x: x.__name__ if callable(x) else ",".join(x),
)
def test_signature_lists_real_parameters(func, params):
    named = inspect.signature(func).parameters
    for param in params:
        assert param in named, f"{func.__name__} hides {param}"


def test_ext_grid_factories_document_sign_convention():
    assert "load convention" in mx.create_ext_power_grid.__doc__.lower()
    assert "load convention" in mx.create_gas_ext_grid.__doc__.lower()
    assert "load convention" in mx.create_water_ext_grid.__doc__.lower()
    assert "10 bar" in mx.create_gas_ext_grid.__doc__


def test_explicit_kwargs_still_forward():
    net = mm.Network()
    junc = mx.create_gas_junction(net)
    cid = mx.create_gas_ext_grid(
        net,
        junc,
        max_import_kgs=5,
        max_export_kgs=2,
        free_pressure_bounds=(0.9, 1.1),
        pin_temperature=False,
    )
    child = net.child_by_id(cid).model
    assert child.mass_flow_kgs.min == -5
    assert child.mass_flow_kgs.max == 2
    assert child.free_pressure_bounds == (0.9, 1.1)
    assert child.pin_temperature is False

    bus = mx.create_bus(net)
    gid = mx.create_ext_power_grid(
        net, bus, max_import_mw=3, max_export_mw=1, regulate_vm=False
    )
    grid_child = net.child_by_id(gid).model
    assert grid_child.p_mw.min == -3
    assert grid_child.p_mw.max == 1
    assert grid_child.regulate_vm is False

    sid = mx.create_gas_source(net, junc, mass_flow_kgs=1, t_k=310)
    assert net.child_by_id(sid).model.injection_t_k == 310

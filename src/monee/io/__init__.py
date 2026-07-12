"""Import/export entry points for monee networks.

Loaders that need optional heavy dependencies (pandapower, simbench) are
resolved lazily via module ``__getattr__`` so ``import monee.io`` stays cheap.
"""

from .matpower import (
    build_matpower_opf,
    read_matpower_case,
    read_matpower_opf_case,
    read_mpc,
)
from .native import (
    load_network,
    load_to_network,
    save_network,
    write_omef_network,
)

_LAZY_EXPORTS = {
    "from_pandapower_net": "monee.io.from_pandapower",
    "obtain_simbench_net": "monee.io.from_simbench",
    "obtain_simbench_net_with_td": "monee.io.from_simbench",
    "obtain_simbench_profile": "monee.io.from_simbench",
    "obtain_simbench_profile_by_pp_net": "monee.io.from_simbench",
}

__all__ = [
    "build_matpower_opf",
    "load_network",
    "load_to_network",
    "read_matpower_case",
    "read_matpower_opf_case",
    "read_mpc",
    "save_network",
    "write_omef_network",
    *sorted(_LAZY_EXPORTS),
]


def __getattr__(name):
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted({*globals(), *_LAZY_EXPORTS})

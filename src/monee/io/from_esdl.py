import logging
import warnings
from dataclasses import dataclass, field

import monee.express as mx
import monee.model as mm
from monee.model.branch import GasPipe, PowerLine, WaterPipe

logger = logging.getLogger(__name__)

_EXPERIMENTAL_NOTE = (
    "monee.io.from_esdl is an experimental sketch: topology mapping is partial "
    "and several asset families are skipped. Verify the result before using it."
)

# --- carrier classification ------------------------------------------------- #
EL, GAS, HEAT = "electricity", "gas", "heat"

# Placeholder electrical line parameters: ESDL ElectricityCable rarely carries
# r/x, so a flat per-metre value is used until impedance data is available.
# TODO: read from an ESDL extension / asset KPI when present.
_PLACEHOLDER_R_OHM_PER_M = 1e-4
_PLACEHOLDER_X_OHM_PER_M = 1e-4
# Default pipe geometry when an ESDL Pipe omits diameter.
_DEFAULT_PIPE_DIAMETER_M = 0.1


@dataclass
class EsdlImportReport:
    """Summary of what was mapped and what was skipped, for transparency."""

    nodes: int = 0
    el_branches: int = 0
    gas_pipes: int = 0
    heat_pipes: int = 0
    loads: int = 0
    sources: int = 0
    skipped: dict = field(default_factory=dict)

    def skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1

    def log(self) -> None:
        logger.info(
            "ESDL import: %d nodes, %d el-branches, %d gas-pipes, %d heat-pipes, "
            "%d loads, %d sources",
            self.nodes,
            self.el_branches,
            self.gas_pipes,
            self.heat_pipes,
            self.loads,
            self.sources,
        )
        for reason, count in sorted(self.skipped.items()):
            logger.warning("ESDL import: skipped %d x %s", count, reason)


# --------------------------------------------------------------------------- #
# pyESDL plumbing                                                              #
# --------------------------------------------------------------------------- #
def _lazy_esdl():
    try:
        from esdl import esdl
        from esdl.esdl_handler import EnergySystemHandler
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "ESDL import needs the optional 'pyESDL' package. "
            "Install it with `pip install pyESDL`."
        ) from exc
    return EnergySystemHandler, esdl


def _get(obj, attr, default=None):
    value = getattr(obj, attr, default)
    return default if value in ("", None) else value


def _num(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _class_name(obj) -> str:
    return type(obj).__name__


def _walk_assets(area):
    """Yield every asset in *area* and its nested sub-areas, depth-first."""
    for asset in _get(area, "asset", []) or []:
        yield from asset
    for sub_area in _get(area, "area", []) or []:
        yield from _walk_assets(sub_area)


def _ports(asset) -> list:
    ports = _get(asset, "port", []) or []
    return list(ports) if isinstance(ports, (list, tuple)) else [ports]


def _port_key(port):
    """Stable identity for a port (its mRID/id, else its object id)."""
    return _get(port, "id") or id(port)


def _carrier_kind(port) -> str | None:
    """Classify a port's carrier as 'electricity' | 'gas' | 'heat' | None."""
    carrier = _get(port, "carrier")
    if carrier is None:
        return None
    name = _class_name(carrier).lower()
    if "electric" in name:
        return EL
    if "gas" in name:
        return GAS
    if "heat" in name:
        return HEAT
    return None


# --------------------------------------------------------------------------- #
# Union-find over ports -> monee nodes                                         #
# --------------------------------------------------------------------------- #
class _DSU:
    def __init__(self):
        self._parent = {}

    def find(self, x):
        self._parent.setdefault(x, x)
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb


def _build_port_groups(assets):
    """Union connected ports. Returns (dsu, {port_key: carrier_kind})."""
    dsu = _DSU()
    kind_by_port = {}
    for asset in assets:
        for port in _ports(asset):
            key = _port_key(port)
            dsu.find(key)  # register
            kind = _carrier_kind(port)
            if kind is not None:
                kind_by_port[key] = kind
            for other in _get(port, "connectedTo", []) or []:
                dsu.union(key, _port_key(other))
    return dsu, kind_by_port


def _group_kind(dsu, kind_by_port):
    """Resolve one carrier kind per connected port-group (first non-None wins)."""
    kind_by_group = {}
    for key, kind in kind_by_port.items():
        kind_by_group.setdefault(dsu.find(key), kind)
    return kind_by_group


def _create_nodes(net, dsu, kind_by_group, report):
    """Create one monee node per port-group. Returns {group_root: node_id}."""
    node_of_group = {}
    for root, kind in kind_by_group.items():
        if kind == EL:
            node_of_group[root] = mx.create_bus(net, base_kv=1, grid=mm.EL)
        elif kind == GAS:
            node_of_group[root] = mx.create_gas_junction(net)
        elif kind == HEAT:
            node_of_group[root] = mx.create_water_junction(net)
        else:
            report.skip("port-group with unknown carrier")
            continue
        report.nodes += 1
    return node_of_group


def _node_for_port(port, dsu, node_of_group):
    return node_of_group.get(dsu.find(_port_key(port)))


# Asset class-name families (ESDL has many concrete subclasses; match loosely).
_EL_TRANSPORT = {"ElectricityCable", "Transformer"}
_PIPE = {"Pipe"}
_EL_DEMAND = {"ElectricityDemand", "EConnection"}
_EL_PRODUCER = {"ElectricityProducer", "PVInstallation", "PVPanel", "WindTurbine"}
_HEAT_DEMAND = {"HeatingDemand", "HeatDemand"}
_HEAT_PRODUCER = {"ResidualHeatSource", "GeothermalSource", "HeatProducer"}
_GAS_DEMAND = {"GasDemand"}
_GAS_PRODUCER = {"GasProducer"}
_CONVERSION = {"CHP", "HeatPump", "PowerToGas", "Electrolyzer", "GasHeater"}


def _power_mw(asset) -> float:
    """ESDL 'power' is in W; convert to MW. Falls back to 0."""
    return _num(_get(asset, "power")) / 1e6


def _add_transport(asset, kind, net, dsu, node_of_group, report):
    ports = _ports(asset)
    if len(ports) != 2:
        report.skip(f"{_class_name(asset)} without exactly 2 ports")
        return
    a = _node_for_port(ports[0], dsu, node_of_group)
    b = _node_for_port(ports[1], dsu, node_of_group)
    if a is None or b is None:
        report.skip(f"{_class_name(asset)} with unresolved endpoint")
        return
    length_m = _num(_get(asset, "length"), 1.0) or 1.0
    name = _get(asset, "name")

    if kind == EL:
        net.branch(
            # TODO: real impedance - ESDL cables seldom carry r/x.
            PowerLine(
                length_m=length_m,
                r_ohm_per_m=_PLACEHOLDER_R_OHM_PER_M,
                x_ohm_per_m=_PLACEHOLDER_X_OHM_PER_M,
                parallel=1,
            ),
            from_node_id=a,
            to_node_id=b,
            grid=mm.EL,
            name=name,
        )
        report.el_branches += 1
    elif kind == GAS:
        diameter = _num(_get(asset, "innerDiameter"), _DEFAULT_PIPE_DIAMETER_M)
        net.branch(
            GasPipe(diameter_m=diameter or _DEFAULT_PIPE_DIAMETER_M, length_m=length_m),
            from_node_id=a,
            to_node_id=b,
            grid=mm.GAS,
            name=name,
        )
        report.gas_pipes += 1
    elif kind == HEAT:
        diameter = _num(_get(asset, "innerDiameter"), _DEFAULT_PIPE_DIAMETER_M)
        net.branch(
            WaterPipe(
                diameter_m=diameter or _DEFAULT_PIPE_DIAMETER_M, length_m=length_m
            ),
            from_node_id=a,
            to_node_id=b,
            grid=mm.WATER,
            name=name,
        )
        report.heat_pipes += 1


def _add_child(asset, cls, net, dsu, node_of_group, report):
    ports = _ports(asset)
    node = _node_for_port(ports[0], dsu, node_of_group) if ports else None
    if node is None:
        report.skip(f"{cls} with unresolved bus/junction")
        return

    if cls in _EL_DEMAND:
        mx.create_power_load(net, node, p_mw=_power_mw(asset), q_mvar=0.0)
        report.loads += 1
    elif cls in _EL_PRODUCER:
        mx.create_power_generator(net, node, p_mw=_power_mw(asset), q_mvar=0.0)
        report.sources += 1
    elif cls in _HEAT_DEMAND:
        mx.create_heat_load(net, node, q_mw=_power_mw(asset))
        report.loads += 1
    elif cls in _HEAT_PRODUCER:
        mx.create_heat_generator(net, node, q_mw=_power_mw(asset))
        report.sources += 1
    else:
        # Gas demand/producer need a heating value to turn power into mass flow.
        # TODO: convert via the gas carrier's energyContent before mapping to
        # Sink/Source. Skipped for now to avoid inventing a number.
        report.skip(f"{cls} (mapping not implemented)")


# --------------------------------------------------------------------------- #
# Public entry points                                                         #
# --------------------------------------------------------------------------- #
def esdl_system_to_network(es):
    """Build a monee :class:`Network` from a loaded ESDL ``EnergySystem``.

    Returns ``(network, EsdlImportReport)``.
    """
    warnings.warn(_EXPERIMENTAL_NOTE, stacklevel=2)
    net = mx.create_multi_energy_network()
    report = EsdlImportReport()

    assets = [
        a for inst in (_get(es, "instance", []) or []) for a in _walk_assets(inst.area)
    ]

    dsu, kind_by_port = _build_port_groups(assets)
    kind_by_group = _group_kind(dsu, kind_by_port)
    node_of_group = _create_nodes(net, dsu, kind_by_group, report)

    for asset in assets:
        cls = _class_name(asset)
        if cls in _EL_TRANSPORT:
            _add_transport(asset, EL, net, dsu, node_of_group, report)
        elif cls in _PIPE:
            # A pipe's carrier (gas vs heat) comes from its port.
            kind = next(
                (
                    _carrier_kind(p)
                    for p in _ports(asset)
                    if _carrier_kind(p) in (GAS, HEAT)
                ),
                None,
            )
            if kind is None:
                report.skip("Pipe with unknown carrier")
            else:
                _add_transport(asset, kind, net, dsu, node_of_group, report)
        elif cls in _CONVERSION:
            # Multi-carrier coupling -> monee compound (CHP/PowerToHeat/...).
            report.skip(f"{cls} conversion asset (coupling TODO)")
        elif (
            cls in _EL_DEMAND
            or cls in _EL_PRODUCER
            or cls in _HEAT_DEMAND
            or cls in _HEAT_PRODUCER
            or cls in _GAS_DEMAND
            or cls in _GAS_PRODUCER
        ):
            _add_child(asset, cls, net, dsu, node_of_group, report)
        else:
            report.skip(f"{cls} (unmapped asset type)")

    report.log()
    return net, report


def import_esdl_file(path):
    """Import an ``.esdl`` file into a monee :class:`Network`.

    Returns ``(network, EsdlImportReport)``.
    """
    EnergySystemHandler, _ = _lazy_esdl()
    es = EnergySystemHandler().load_file(path)
    return esdl_system_to_network(es)

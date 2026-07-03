import logging
import warnings
from dataclasses import dataclass, field

import monee.express as mx
import monee.model as mm
from monee.io._utils import (
    _class_name,
    _get,
    _num,
    _resolve_single_node,
    _resolve_two_endpoints,
)
from monee.model.branch import GenericPowerBranch
from monee.model.child import PowerShunt

logger = logging.getLogger(__name__)

_EXPERIMENTAL_NOTE = (
    "monee.io.from_cim is experimental: the CGMES mapping is incomplete and "
    "unvalidated. Verify the result before relying on it."
)

# CGMES class names we treat as the electrical load family.
_LOAD_CLASSES = {"EnergyConsumer", "ConformLoad", "NonConformLoad", "StationSupply"}
# Default line current rating (kA) when no OperationalLimit is present.
_UNBOUNDED_I_KA = 9999.0
# Tolerance for treating physical parameters (impedance, admittance, sections) as zero.
_ZERO_TOL = 1e-12


@dataclass
class CimImportReport:
    """Summary of what was mapped and what was skipped, for transparency."""

    buses: int = 0
    lines: int = 0
    transformers: int = 0
    loads: int = 0
    generators: int = 0
    ext_grids: int = 0
    shunts: int = 0
    skipped: dict = field(default_factory=dict)

    def skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1

    def log(self) -> None:
        logger.info(
            "CIM import: %d buses, %d lines, %d trafos, %d loads, %d gens, "
            "%d ext-grids, %d shunts",
            self.buses,
            self.lines,
            self.transformers,
            self.loads,
            self.generators,
            self.ext_grids,
            self.shunts,
        )
        for reason, count in sorted(self.skipped.items()):
            logger.warning("CIM import: skipped %d x %s", count, reason)


def _terminals(equipment) -> list:
    """Return an equipment's Terminals, ordered by sequenceNumber when present."""
    terminals = _get(equipment, "Terminals", []) or []
    if not isinstance(terminals, (list, tuple)):
        terminals = [terminals]
    return sorted(terminals, key=lambda t: _num(_get(t, "sequenceNumber", 0)))


def _topo_node(terminal):
    """Resolve a Terminal to its TopologicalNode (bus-branch model)."""
    return _get(terminal, "TopologicalNode")


def _base_kv(topo_node, report=None) -> float:
    """Nominal voltage (kV) of a TopologicalNode via its BaseVoltage reference.

    Falls back to 1.0 kV when no usable BaseVoltage is present; with a *report*
    the fallback is recorded so mis-scaled per-unit branches are not silent.
    """
    base_voltage = _get(topo_node, "BaseVoltage")
    if base_voltage is not None:
        nominal = _num(_get(base_voltage, "nominalVoltage"), 0.0)
        if nominal > 0:
            return nominal
    if report is not None:
        report.skip("TopologicalNode without usable BaseVoltage (base_kv -> 1.0)")
    return 1.0


# --------------------------------------------------------------------------- #
# Topology -> monee buses                                                      #
# --------------------------------------------------------------------------- #
def _build_buses(objects, net, report):
    """Create one Bus per TopologicalNode. Returns ``{tn_mrid: node_id}``."""
    tn_to_node = {}
    for mrid, obj in objects.items():
        if _class_name(obj) != "TopologicalNode":
            continue
        node_id = mx.create_bus(
            net,
            base_kv=_base_kv(obj, report),
            grid=mm.EL,
            name=_get(obj, "name"),
        )
        tn_to_node[mrid] = node_id
        report.buses += 1
    return tn_to_node


def _node_of(terminal, tn_to_node):
    """monee node id behind a Terminal, or None if its TN was not imported."""
    tn = _topo_node(terminal)
    return tn_to_node.get(_get(tn, "mRID")) if tn is not None else None


# --------------------------------------------------------------------------- #
# Branches: lines and transformers                                            #
# --------------------------------------------------------------------------- #
def _add_line(obj, net, tn_to_node, base_kv_of, report):
    resolved = _resolve_two_endpoints(
        _terminals(obj),
        lambda terminal: _node_of(terminal, tn_to_node),
        report,
        "ACLineSegment without exactly 2 terminals",
        "ACLineSegment with unresolved endpoint",
    )
    if resolved is None:
        return
    from_node, to_node = resolved

    base_kv = base_kv_of(from_node)
    base_z = base_kv**2 / net_base_mva(net)  # ohms per p.u.
    r = _num(_get(obj, "r"))
    x = _num(_get(obj, "x"))
    bch = _num(_get(obj, "bch"))  # total line charging susceptance [S]
    gch = _num(_get(obj, "gch"))

    if abs(r) < _ZERO_TOL and abs(x) < _ZERO_TOL:
        report.skip("ACLineSegment without impedance data (r=x=0, singular)")
        return

    net.branch(
        GenericPowerBranch(
            tap=1.0,
            shift=0.0,
            br_r_pu=r / base_z,
            br_x_pu=x / base_z,
            # Split the total shunt admittance evenly across both ends (pi model).
            g_fr_pu=gch * base_z / 2.0,
            b_fr_pu=bch * base_z / 2.0,
            g_to_pu=gch * base_z / 2.0,
            b_to_pu=bch * base_z / 2.0,
            max_i_ka=_UNBOUNDED_I_KA,  # NOSONAR TODO: read CurrentLimit / OperationalLimit
        ),
        from_node_id=from_node,
        to_node_id=to_node,
        grid=mm.EL,
        name=_get(obj, "name"),
    )
    report.lines += 1


def _add_transformer(obj, ends, net, tn_to_node, base_kv_of, report):
    if len(ends) != 2:
        report.skip(f"{len(ends)}-winding PowerTransformer (only 2-winding supported)")
        return
    # Order ends by their winding terminal sequence.
    ends = sorted(ends, key=lambda e: _num(_get(e, "endNumber", 0)))
    from_node = _node_of(_get(ends[0], "Terminal"), tn_to_node)
    to_node = _node_of(_get(ends[1], "Terminal"), tn_to_node)
    if from_node is None or to_node is None:
        report.skip("PowerTransformer with unresolved endpoint")
        return

    # Lump both winding impedances onto the FROM side: Z = Σ z_e·(U_from/U_e)².
    u_from = _num(_get(ends[0], "ratedU"), base_kv_of(from_node)) or 1.0
    r_ohm = x_ohm = 0.0
    for end in ends:
        u_e = _num(_get(end, "ratedU"), u_from) or u_from
        ratio_sq = (u_from / u_e) ** 2
        r_ohm += _num(_get(end, "r")) * ratio_sq
        x_ohm += _num(_get(end, "x")) * ratio_sq

    if abs(r_ohm) < _ZERO_TOL and abs(x_ohm) < _ZERO_TOL:
        report.skip("PowerTransformer without impedance data (r=x=0, singular)")
        return

    base_z = base_kv_of(from_node) ** 2 / net_base_mva(net)
    net.branch(
        GenericPowerBranch(
            tap=1.0,  # NOSONAR TODO: RatioTapChanger.step -> off-nominal ratio
            shift=0.0,  # NOSONAR TODO: PhaseTapChanger -> phase shift
            br_r_pu=r_ohm / base_z,
            br_x_pu=x_ohm / base_z,
            g_fr_pu=0.0,
            b_fr_pu=0.0,
            g_to_pu=0.0,
            b_to_pu=0.0,
            max_i_ka=_UNBOUNDED_I_KA,
        ),
        from_node_id=from_node,
        to_node_id=to_node,
        grid=mm.EL,
        name=_get(obj, "name"),
    )
    report.transformers += 1


def _transformer_ends_by_owner(objects):
    """Group PowerTransformerEnd objects by their parent PowerTransformer mRID."""
    ends = {}
    for obj in objects.values():
        if _class_name(obj) != "PowerTransformerEnd":
            continue
        owner = _get(obj, "PowerTransformer")
        owner_mrid = _get(owner, "mRID")
        if owner_mrid is not None:
            ends.setdefault(owner_mrid, []).append(obj)
    return ends


# --------------------------------------------------------------------------- #
# Children: loads, generators, slack                                          #
# --------------------------------------------------------------------------- #
def _single_node(obj, tn_to_node, report, unresolved_msg):
    return _resolve_single_node(
        _terminals(obj),
        lambda terminal: _node_of(terminal, tn_to_node),
        report,
        unresolved_msg,
    )


def _add_load(obj, net, tn_to_node, report):
    node = _single_node(obj, tn_to_node, report, "load with unresolved bus")
    if node is None:
        return
    mx.create_power_load(
        net, node, p_mw=_num(_get(obj, "p")), q_mvar=_num(_get(obj, "q"))
    )
    report.loads += 1


def _add_generator(obj, net, tn_to_node, gen_sign, report):
    node = _single_node(obj, tn_to_node, report, "generator with unresolved bus")
    if node is None:
        return
    # CGMES SSH uses load reference at the terminal; *gen_sign* flips it to
    # generation convention (positive = injection). A machine that ends up
    # consuming (p < 0 after the flip) must stay a consumer, so it maps to a
    # PowerLoad (positive = consumption) instead of a generator.
    p = gen_sign * _num(_get(obj, "p"))
    q = gen_sign * _num(_get(obj, "q"))
    if p < 0:
        mx.create_power_load(net, node, p_mw=-p, q_mvar=-q)
        report.loads += 1
        return
    mx.create_power_generator(net, node, p_mw=p, q_mvar=q)
    report.generators += 1


def _add_equivalent_injection(obj, net, tn_to_node, report):
    node = _single_node(
        obj, tn_to_node, report, "EquivalentInjection with unresolved bus"
    )
    if node is None:
        return
    # SSH load reference: positive p = consumption.
    p = _num(_get(obj, "p"))
    q = _num(_get(obj, "q"))
    if p < 0:
        mx.create_power_generator(net, node, p_mw=-p, q_mvar=-q)
        report.generators += 1
        return
    mx.create_power_load(net, node, p_mw=p, q_mvar=q)
    report.loads += 1


def _add_shunt(obj, net, tn_to_node, report):
    node = _single_node(obj, tn_to_node, report, "ShuntCompensator with unresolved bus")
    if node is None:
        return
    sections = _num(_get(obj, "sections"), 1.0)
    if abs(sections) < _ZERO_TOL:
        report.skip("ShuntCompensator switched out (sections=0)")
        return
    b_siemens = _num(_get(obj, "bPerSection")) * sections
    g_siemens = _num(_get(obj, "gPerSection")) * sections
    if abs(b_siemens) < _ZERO_TOL and abs(g_siemens) < _ZERO_TOL:
        report.skip("ShuntCompensator without b/g data")
        return
    base_kv = mx_base_kv(net, node)
    # Admittance [S] at nominal voltage -> MW/MVAr drawn/injected at v=1 p.u.
    net.child_to(
        PowerShunt(gs_mw=g_siemens * base_kv**2, bs_mvar=b_siemens * base_kv**2),
        node_id=node,
        name=_get(obj, "name"),
    )
    report.shunts += 1


def _add_asynchronous_machine(obj, net, tn_to_node, report):
    node = _single_node(
        obj, tn_to_node, report, "AsynchronousMachine with unresolved bus"
    )
    if node is None:
        return
    # SSH load reference: an asynchronous machine (motor) consumes.
    mx.create_power_load(
        net, node, p_mw=_num(_get(obj, "p")), q_mvar=_num(_get(obj, "q"))
    )
    report.loads += 1


def _add_ext_grid(obj, net, tn_to_node, report):
    node = _single_node(
        obj, tn_to_node, report, "ExternalNetworkInjection with unresolved bus"
    )
    if node is None:
        return
    # Voltage setpoint from the regulating control target, if present.
    control = _get(obj, "RegulatingControl")
    terminals = _terminals(obj)
    base_kv = _base_kv(_topo_node(terminals[0])) if terminals else 1.0
    target_v = _num(_get(control, "targetValue"), 0.0) if control else 0.0
    vm_pu = (target_v / base_kv) if (target_v and base_kv) else 1.0
    mx.create_ext_power_grid(net, node, vm_pu=vm_pu, va_degree=0.0)
    report.ext_grids += 1


# --------------------------------------------------------------------------- #
# Public entry point                                                          #
# --------------------------------------------------------------------------- #
def net_base_mva(net) -> float:
    """System base MVA of the electrical grid (defaults to 1 like the model)."""
    grid = net._or_default(mm.EL_KEY)
    return _num(getattr(grid, "sn_mva", 1.0), 1.0) or 1.0


def cim_objects_to_network(objects, gen_sign=-1.0):
    """Build a monee :class:`Network` from a ``{mRID: cim_object}`` mapping.

    *objects* is CIMpy's merged topology dict. *gen_sign* maps the CGMES SSH
    terminal sign convention to generation magnitude (default ``-1`` treats a
    negative SSH ``p`` as generation). Returns ``(network, CimImportReport)``.
    """
    warnings.warn(_EXPERIMENTAL_NOTE, stacklevel=2)
    net = mx.create_multi_energy_network()
    report = CimImportReport()

    tn_to_node = _build_buses(objects, net, report)
    base_kv_of = {nid: mx_base_kv(net, nid) for nid in tn_to_node.values()}.get

    transformer_ends = _transformer_ends_by_owner(objects)

    for obj in objects.values():
        cls = _class_name(obj)
        if cls == "ACLineSegment":
            _add_line(obj, net, tn_to_node, base_kv_of, report)
        elif cls == "PowerTransformer":
            ends = transformer_ends.get(_get(obj, "mRID"), [])
            _add_transformer(obj, ends, net, tn_to_node, base_kv_of, report)
        elif cls in _LOAD_CLASSES:
            _add_load(obj, net, tn_to_node, report)
        elif cls == "SynchronousMachine":
            _add_generator(obj, net, tn_to_node, gen_sign, report)
        elif cls == "ExternalNetworkInjection":
            _add_ext_grid(obj, net, tn_to_node, report)
        elif cls == "EquivalentInjection":
            _add_equivalent_injection(obj, net, tn_to_node, report)
        elif cls in ("LinearShuntCompensator", "ShuntCompensator"):
            _add_shunt(obj, net, tn_to_node, report)
        elif cls == "NonlinearShuntCompensator":
            report.skip("NonlinearShuntCompensator (mapping not implemented)")
        elif cls == "AsynchronousMachine":
            _add_asynchronous_machine(obj, net, tn_to_node, report)

    if report.ext_grids == 0:
        logger.warning(
            "CIM import: no ExternalNetworkInjection found - the network has no "
            "slack. Designate one with create_ext_power_grid before solving."
        )
    report.log()
    return net, report


def mx_base_kv(net, node_id) -> float:
    """base_kv of an already-created Bus node."""
    return _num(getattr(net.node_by_id(node_id).model, "base_kv", 1.0), 1.0)


def import_cim_files(file_list, cgmes_version="cgmes_v2_4_15", gen_sign=-1.0):
    """Import CGMES XML/zip files into a monee :class:`Network`.

    Parameters
    ----------
    file_list:
        Path(s) accepted by ``cimpy.cim_import`` - the EQ/TP/SSH (and optionally
        SV) profile files of one CGMES model.
    cgmes_version:
        CIMpy version tag, e.g. ``"cgmes_v2_4_15"`` or ``"cgmes_v3_0_0"``.
    gen_sign:
        Sign applied to SSH machine setpoints (see :func:`cim_objects_to_network`).

    Returns
    -------
    (network, report):
        The electrical :class:`Network` and a :class:`CimImportReport`.
    """
    try:
        import cimpy
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "CIM/CGMES import needs the optional 'cimpy' package. "
            "Install it with `pip install cimpy`."
        ) from exc

    if isinstance(file_list, str):
        file_list = [file_list]
    imported = cimpy.cim_import(file_list, cgmes_version)
    objects = imported["topology"]
    return cim_objects_to_network(objects, gen_sign=gen_sign)

"""Bulk-construction helpers for common network topologies.

Each ``*_structure`` factory returns a builder that remembers carrier-level
defaults (diameter, length, grid, …) so the shape methods (``line``,
``ring``, ``star``) stay short.  Shape methods return :class:`Segment`,
:class:`StarSegment`, or :class:`DhsSegment` handles that can be passed
back via ``start_from=`` to compose larger structures.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import monee.express as _mx


@dataclass
class Segment:
    """Ordered handle to a line/ring of nodes and their branches."""

    nodes: list
    branches: list = field(default_factory=list)
    children: list = field(default_factory=list)

    @property
    def first(self):
        return self.nodes[0]

    @property
    def last(self):
        return self.nodes[-1]

    def __len__(self):
        return len(self.nodes)


@dataclass
class StarSegment:
    """Hub + arm-segments.  Each arm is itself a :class:`Segment` whose
    ``first`` is the shared hub."""

    hub: int
    arms: list
    hub_children: list = field(default_factory=list)

    @property
    def nodes(self):
        seen = {self.hub}
        out = [self.hub]
        for arm in self.arms:
            for nid in arm.nodes:
                if nid not in seen:
                    seen.add(nid)
                    out.append(nid)
        return out

    @property
    def branches(self):
        return [b for arm in self.arms for b in arm.branches]

    @property
    def children(self):
        return list(self.hub_children) + [c for arm in self.arms for c in arm.children]


@dataclass
class DhsSegment:
    """Paired supply/return segments bridged by heat exchangers."""

    supply: Segment
    return_: Segment
    heat_exchangers: list = field(default_factory=list)

    @property
    def nodes(self):
        return list(self.supply.nodes) + list(self.return_.nodes)

    @property
    def branches(self):
        return (
            list(self.supply.branches)
            + list(self.return_.branches)
            + list(self.heat_exchangers)
        )

    @property
    def children(self):
        return list(self.supply.children) + list(self.return_.children)


class _Structure:
    """Shared shape logic for single-carrier structures."""

    def __init__(self, network):
        self._net = network

    # --- carrier hooks: override these ---

    def _create_node(self):
        raise NotImplementedError

    def _create_branch(self, from_id, to_id):
        raise NotImplementedError

    def _attach_loads(self, node_id, kwargs):  # NOSONAR
        return []

    # --- shapes ---

    def line(self, n, *, start_from=None, **kwargs):
        if n < 1:
            raise ValueError("line requires n >= 1")
        nodes = [start_from] if start_from is not None else []
        for _ in range(n - len(nodes)):
            nodes.append(self._create_node())
        branches = [
            self._create_branch(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)
        ]
        load_targets = nodes if start_from is None else nodes[1:]
        children = []
        for nid in load_targets:
            children.extend(self._attach_loads(nid, kwargs))
        return Segment(nodes=nodes, branches=branches, children=children)

    def ring(self, n, *, start_from=None, **kwargs):
        if n < 3:
            raise ValueError("ring requires n >= 3")
        seg = self.line(n=n, start_from=start_from, **kwargs)
        seg.branches.append(self._create_branch(seg.last, seg.first))
        return seg

    def star(self, arms, *, start_from=None, **kwargs):
        if not arms:
            raise ValueError("star requires at least one arm")
        if any(a < 1 for a in arms):
            raise ValueError("each arm length must be >= 1")
        if start_from is None:
            hub = self._create_node()
            hub_children = self._attach_loads(hub, kwargs)
        else:
            hub = start_from
            hub_children = []
        arm_segments = [
            self.line(n=arm_len + 1, start_from=hub, **kwargs) for arm_len in arms
        ]
        return StarSegment(hub=hub, arms=arm_segments, hub_children=hub_children)


class GasStructure(_Structure):
    def __init__(
        self,
        network,
        *,
        diameter_m,
        length_m,
        temperature_ext_k=296.15,
        roughness_m=1e-5,
        grid=None,
    ):
        super().__init__(network)
        self._diameter_m = diameter_m
        self._length_m = length_m
        self._temperature_ext_k = temperature_ext_k
        self._roughness = roughness_m
        self._grid = grid

    def _create_node(self):
        kwargs = {"grid": self._grid} if self._grid is not None else {}
        return _mx.create_gas_junction(self._net, **kwargs)

    def _create_branch(self, from_id, to_id):
        return _mx.create_gas_pipe(
            self._net,
            from_id,
            to_id,
            diameter_m=self._diameter_m,
            length_m=self._length_m,
            temperature_ext_k=self._temperature_ext_k,
            roughness_m=self._roughness,
            grid=self._grid,
        )

    def _attach_loads(self, node_id, kwargs):
        out = []
        if kwargs.get("sink_mass_flow") is not None:
            out.append(
                _mx.create_gas_sink(
                    self._net, node_id, mass_flow_kgs=kwargs["sink_mass_flow"]
                )
            )
        if kwargs.get("source_mass_flow") is not None:
            out.append(
                _mx.create_gas_source(
                    self._net, node_id, mass_flow_kgs=kwargs["source_mass_flow"]
                )
            )
        return out

    def attach_ext_grid(self, node_id, **kwargs):
        return _mx.create_gas_ext_grid(self._net, node_id, **kwargs)


class WaterStructure(_Structure):
    def __init__(
        self,
        network,
        *,
        diameter_m,
        length_m,
        temperature_ext_k=296.15,
        roughness_m=0.001,
        lambda_insulation_w_per_m_k=0.025,
        insulation_thickness_m=0.2,
        unidirectional=False,
        grid=None,
    ):
        super().__init__(network)
        self._diameter_m = diameter_m
        self._length_m = length_m
        self._temperature_ext_k = temperature_ext_k
        self._roughness = roughness_m
        self._lambda = lambda_insulation_w_per_m_k
        self._ins_thickness = insulation_thickness_m
        self._unidirectional = unidirectional
        self._grid = grid

    def _create_node(self):
        kwargs = {"grid": self._grid} if self._grid is not None else {}
        return _mx.create_water_junction(self._net, **kwargs)

    def _create_branch(self, from_id, to_id):
        return _mx.create_water_pipe(
            self._net,
            from_id,
            to_id,
            diameter_m=self._diameter_m,
            length_m=self._length_m,
            temperature_ext_k=self._temperature_ext_k,
            roughness_m=self._roughness,
            lambda_insulation_w_per_m_k=self._lambda,
            insulation_thickness_m=self._ins_thickness,
            unidirectional=self._unidirectional,
            grid=self._grid,
        )

    def _attach_loads(self, node_id, kwargs):
        out = []
        if kwargs.get("sink_mass_flow") is not None:
            out.append(
                _mx.create_water_sink(
                    self._net, node_id, mass_flow_kgs=kwargs["sink_mass_flow"]
                )
            )
        if kwargs.get("source_mass_flow") is not None:
            out.append(
                _mx.create_water_source(
                    self._net, node_id, mass_flow_kgs=kwargs["source_mass_flow"]
                )
            )
        if kwargs.get("heat_load_q_mw") is not None:
            out.append(
                _mx.create_heat_load(self._net, node_id, q_mw=kwargs["heat_load_q_mw"])
            )
        if kwargs.get("heat_generator_q_mw") is not None:
            out.append(
                _mx.create_heat_generator(
                    self._net, node_id, q_mw=kwargs["heat_generator_q_mw"]
                )
            )
        return out

    def attach_ext_grid(self, node_id, **kwargs):
        return _mx.create_water_ext_grid(self._net, node_id, **kwargs)


class ElStructure(_Structure):
    def __init__(
        self,
        network,
        *,
        length_m,
        r_ohm_per_m,
        x_ohm_per_m,
        parallel=1,
        base_kv=1,
        grid=None,
    ):
        super().__init__(network)
        self._length_m = length_m
        self._r = r_ohm_per_m
        self._x = x_ohm_per_m
        self._parallel = parallel
        self._base_kv = base_kv
        self._grid = grid

    def _create_node(self):
        kwargs = {"base_kv": self._base_kv}
        if self._grid is not None:
            kwargs["grid"] = self._grid
        return _mx.create_bus(self._net, **kwargs)

    def _create_branch(self, from_id, to_id):
        return _mx.create_line(
            self._net,
            from_id,
            to_id,
            length_m=self._length_m,
            r_ohm_per_m=self._r,
            x_ohm_per_m=self._x,
            parallel=self._parallel,
            grid=self._grid,
        )

    def _attach_loads(self, node_id, kwargs):
        out = []
        if kwargs.get("load_p_mw") is not None:
            out.append(
                _mx.create_power_load(
                    self._net,
                    node_id,
                    p_mw=kwargs["load_p_mw"],
                    q_mvar=kwargs.get("load_q_mvar", 0),
                )
            )
        if kwargs.get("gen_p_mw") is not None:
            out.append(
                _mx.create_power_generator(
                    self._net,
                    node_id,
                    p_mw=kwargs["gen_p_mw"],
                    q_mvar=kwargs.get("gen_q_mvar", 0),
                )
            )
        return out

    def attach_ext_grid(self, node_id, **kwargs):
        return _mx.create_ext_power_grid(self._net, node_id, **kwargs)


def _resolve_q_list(q, n):
    if q is None:
        return [None] * n
    if isinstance(q, Iterable) and not isinstance(q, (str, bytes)):
        values = list(q)
        if len(values) != n:
            raise ValueError(
                f"heat_exchanger_q_mw list length {len(values)} "
                f"does not match node count {n}"
            )
        return values
    return [q] * n


class DhsStructure:
    """Paired supply/return district-heating builder.

    Mirrors every shape on both a supply line/ring/star and a return
    line/ring/star, then optionally bridges consumer nodes with heat
    exchangers.  Heat exchanger orientation is supply -> return (hot
    side into cold side), matching the conventional DHS wiring.
    """

    def __init__(
        self,
        network,
        *,
        diameter_m,
        length_m,
        temperature_ext_k=296.15,
        roughness_m=0.001,
        lambda_insulation_w_per_m_k=0.025,
        insulation_thickness_m=0.2,
        unidirectional=True,
        grid=None,
    ):
        self._net = network
        common = {
            "diameter_m": diameter_m,
            "length_m": length_m,
            "temperature_ext_k": temperature_ext_k,
            "roughness_m": roughness_m,
            "lambda_insulation_w_per_m_k": lambda_insulation_w_per_m_k,
            "insulation_thickness_m": insulation_thickness_m,
            "unidirectional": unidirectional,
            "grid": grid,
        }
        self.supply = WaterStructure(network, **common)
        self.return_ = WaterStructure(network, **common)

    def _bridge(self, supply_nodes, return_nodes, heat_exchanger_q_mw):
        if heat_exchanger_q_mw is None:
            return []
        if len(supply_nodes) != len(return_nodes):
            raise ValueError("supply and return node counts differ")
        values = _resolve_q_list(heat_exchanger_q_mw, len(supply_nodes))
        hxs = []
        for sn, rn, q in zip(supply_nodes, return_nodes, values):
            if q is None or q == 0:
                continue
            hxs.append(_mx.create_heat_exchanger(self._net, sn, rn, q_mw=q))
        return hxs

    def line(
        self,
        n,
        *,
        start_from_supply=None,
        start_from_return=None,
        heat_exchanger_q_mw=None,
        **kwargs,
    ):
        supply_seg = self.supply.line(n=n, start_from=start_from_supply, **kwargs)
        return_seg = self.return_.line(n=n, start_from=start_from_return, **kwargs)
        hxs = self._bridge(supply_seg.nodes, return_seg.nodes, heat_exchanger_q_mw)
        return DhsSegment(supply=supply_seg, return_=return_seg, heat_exchangers=hxs)

    def ring(
        self,
        n,
        *,
        start_from_supply=None,
        start_from_return=None,
        heat_exchanger_q_mw=None,
        **kwargs,
    ):
        supply_seg = self.supply.ring(n=n, start_from=start_from_supply, **kwargs)
        return_seg = self.return_.ring(n=n, start_from=start_from_return, **kwargs)
        hxs = self._bridge(supply_seg.nodes, return_seg.nodes, heat_exchanger_q_mw)
        return DhsSegment(supply=supply_seg, return_=return_seg, heat_exchangers=hxs)

    def star(
        self,
        arms,
        *,
        start_from_supply=None,
        start_from_return=None,
        heat_exchanger_q_mw=None,
        **kwargs,
    ):
        supply_star = self.supply.star(
            arms=arms, start_from=start_from_supply, **kwargs
        )
        return_star = self.return_.star(
            arms=arms, start_from=start_from_return, **kwargs
        )
        supply_seg = Segment(
            nodes=supply_star.nodes,
            branches=supply_star.branches,
            children=supply_star.children,
        )
        return_seg = Segment(
            nodes=return_star.nodes,
            branches=return_star.branches,
            children=return_star.children,
        )
        hxs = self._bridge(supply_seg.nodes, return_seg.nodes, heat_exchanger_q_mw)
        return DhsSegment(supply=supply_seg, return_=return_seg, heat_exchangers=hxs)

    def attach_heat_plant(self, supply_node, return_node, *, t_k=358.0, name=None):
        """Attach a heat plant: external hydr grid on supply, consumer on return."""
        supply_name = f"{name}_supply" if name else None
        return_name = f"{name}_return" if name else None
        sid = _mx.create_water_ext_grid(
            self._net, supply_node, t_k=t_k, name=supply_name
        )
        rid = _mx.create_consume_hydr_grid(self._net, return_node, name=return_name)
        return sid, rid


def gas_structure(network, **kwargs):
    """Builder for gas-pipe topologies (line/ring/star)."""
    return GasStructure(network, **kwargs)


def water_structure(network, **kwargs):
    """Builder for water/heat-pipe topologies (single chain, no return line)."""
    return WaterStructure(network, **kwargs)


def el_structure(network, **kwargs):
    """Builder for electrical-bus topologies."""
    return ElStructure(network, **kwargs)


def dhs_structure(network, **kwargs):
    """Builder for paired supply/return district-heating topologies."""
    return DhsStructure(network, **kwargs)

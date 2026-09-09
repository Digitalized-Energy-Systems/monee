"""Bulk-construction helpers for common network topologies.

Each ``*_structure`` factory returns a builder that remembers carrier-level
defaults (diameter, length, grid, and so on) so the shape methods (``line``,
``ring``, ``star``) stay short.  Shape methods return :class:`Segment`,
:class:`StarSegment`, or :class:`DhsSegment` handles that can be passed
back via ``start_from=`` to compose larger structures.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from contextlib import contextmanager
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


def _as_segment(star: StarSegment) -> Segment:
    return Segment(nodes=star.nodes, branches=star.branches, children=star.children)


class _Structure:
    """Shared shape logic for single-carrier structures."""

    LOAD_KWARGS: frozenset = frozenset()

    def __init__(self, network):
        self._net = network
        self._start_from_warned = set()

    @property
    def _grid_kwargs(self):
        return {"grid": self._grid} if self._grid is not None else {}

    def _check_load_kwargs(self, kwargs):
        unknown = sorted(set(kwargs) - self.LOAD_KWARGS)
        if unknown:
            raise TypeError(
                f"{type(self).__name__} got unknown load keyword argument(s) "
                f"{unknown}. Accepted: {sorted(self.LOAD_KWARGS)}"
            )

    @contextmanager
    def _closed_loop(self):
        yield

    # --- carrier hooks: override these ---

    def _create_node(self):
        raise NotImplementedError

    def _create_branch(self, from_id, to_id):
        raise NotImplementedError

    def _attach_loads(self, node_id, kwargs):  # NOSONAR
        return []

    def _check_start_from(self, node_id):  # NOSONAR
        pass

    # --- shapes ---

    def line(self, n, *, start_from=None, **kwargs):
        """Create a chain of *n* nodes joined by *n* - 1 branches.

        Args:
            n: number of nodes on the chain, including ``start_from``.
            start_from: existing node id to grow the chain from; it keeps its
                own loads, so only the newly created nodes get loads attached.
            **kwargs: load setpoints attached to every load target, one child
                per given key. The accepted keys are carrier specific and
                listed in :attr:`LOAD_KWARGS`; an unknown key raises
                ``TypeError`` rather than being silently dropped.

        Returns:
            Segment: the ordered nodes, their branches and the created loads.
        """
        self._check_load_kwargs(kwargs)
        if n < 1:
            raise ValueError("line requires n >= 1")
        if start_from is not None:
            self._check_start_from(start_from)
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
        """Create a :meth:`line` of *n* nodes and close it with one more branch.

        Args:
            n: number of nodes on the ring (at least 3).
            start_from: existing node id to include as the ring's first node.
            **kwargs: load setpoints, see :meth:`line` and :attr:`LOAD_KWARGS`.

        Returns:
            Segment: the ring nodes and its *n* branches, the last one closing
            the loop from ``last`` back to ``first``.
        """
        self._check_load_kwargs(kwargs)
        if n < 3:
            raise ValueError("ring requires n >= 3")
        with self._closed_loop():
            seg = self.line(n=n, start_from=start_from, **kwargs)
            seg.branches.append(self._create_branch(seg.last, seg.first))
        return seg

    def star(self, arms, *, start_from=None, hub_loads=False, **kwargs):
        """Create a hub node with one :meth:`line` arm per entry of *arms*.

        Args:
            arms: arm lengths in nodes, excluding the shared hub.
            start_from: existing node id to use as the hub.
            hub_loads: also attach the load setpoints to the hub itself;
                default False, the hub is treated as a pure branching or
                feed-in point and only the arm nodes get loads.
            **kwargs: load setpoints attached to every arm node (and to the
                hub if ``hub_loads=True``), see :meth:`line` and
                :attr:`LOAD_KWARGS`.

        Returns:
            StarSegment: the hub and one :class:`Segment` per arm, each arm
            starting at the hub.
        """
        self._check_load_kwargs(kwargs)
        if not arms:
            raise ValueError("star requires at least one arm")
        if any(a < 1 for a in arms):
            raise ValueError("each arm length must be >= 1")
        hub = self._create_node() if start_from is None else start_from
        hub_children = self._attach_loads(hub, kwargs) if hub_loads else []
        arm_segments = [
            self.line(n=arm_len + 1, start_from=hub, **kwargs) for arm_len in arms
        ]
        return StarSegment(hub=hub, arms=arm_segments, hub_children=hub_children)


class GasStructure(_Structure):
    """Gas-pipe topology builder; construct via :func:`gas_structure`.

    Every created pipe shares the builder's ``diameter_m`` / ``length_m`` /
    ``temperature_ext_k`` / ``roughness_m``; every created node is a gas
    junction on ``grid`` (default: the network's default gas grid, lgas with
    a 10 bar reference pressure and 300 K operating temperature).

    Shape methods accept the load keywords ``sink_mass_flow`` (consumption,
    positive kg/s) and ``source_mass_flow`` (injection, positive magnitude,
    sign handled internally), each attached to every load target node.
    """

    LOAD_KWARGS = frozenset({"sink_mass_flow", "source_mass_flow"})

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
        return _mx.create_gas_junction(self._net, **self._grid_kwargs)

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
        """Attach an external gas grid to *node_id*.

        Args:
            node_id: junction the slack is attached to.
            **kwargs: forwarded to
                :func:`monee.express.create_gas_ext_grid`, e.g.
                ``mass_flow_kgs``, ``pressure_pu``, ``t_k``, ``name``.

        Returns:
            The id of the created external-grid child.
        """
        return _mx.create_gas_ext_grid(self._net, node_id, **kwargs)


class WaterStructure(_Structure):
    """Water/heat-pipe topology builder; construct via :func:`water_structure`.

    Builds a single chain of water pipes (no return line; use
    :class:`DhsStructure` for a paired supply/return system). Every created
    pipe shares the builder's geometry and insulation parameters;
    ``unidirectional=True`` pins the flow direction of created pipes except
    inside :meth:`ring`, where a closed loop needs counter-flow.

    Shape methods accept the load keywords ``sink_mass_flow`` /
    ``source_mass_flow`` (kg/s, positive magnitudes, source sign handled
    internally) and ``heat_load_q_mw`` / ``heat_generator_q_mw`` (MW,
    positive magnitudes; load = consumption, generator sign handled
    internally), each attached to every load target node.
    """

    LOAD_KWARGS = frozenset(
        {
            "sink_mass_flow",
            "source_mass_flow",
            "heat_load_q_mw",
            "heat_generator_q_mw",
        }
    )

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

    @contextmanager
    def _closed_loop(self):
        # A closed loop generally splits the flow at the feed-in node, so at
        # least one pipe has to carry flow against the ring orientation.
        previous, self._unidirectional = self._unidirectional, False
        try:
            yield
        finally:
            self._unidirectional = previous

    def _create_node(self):
        return _mx.create_water_junction(self._net, **self._grid_kwargs)

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
        """Attach an external hydraulic grid to *node_id*.

        Args:
            node_id: water junction the slack is attached to.
            **kwargs: forwarded to
                :func:`monee.express.create_water_ext_grid`, e.g.
                ``mass_flow_kgs``, ``pressure_pu``, ``t_k``, ``name``.

        Returns:
            The id of the created external-grid child.
        """
        return _mx.create_water_ext_grid(self._net, node_id, **kwargs)


class ElStructure(_Structure):
    """Electrical-bus topology builder; construct via :func:`el_structure`.

    Every created line shares the builder's ``length_m`` / ``r_ohm_per_m`` /
    ``x_ohm_per_m`` / ``parallel``; every created bus gets ``base_kv``
    (default 1 kV; set the real voltage level). ``base_kv`` applies only to
    buses this builder creates, never to ``start_from`` buses; a
    ``start_from`` bus whose ``base_kv`` differs from the builder's raises
    a ``UserWarning``, because a mixed-base network skews per-unit results
    without failing the solve.

    Shape methods accept the load keywords ``load_p_mw`` / ``load_q_mvar``
    (positive = consumption) and ``gen_p_mw`` / ``gen_q_mvar`` (positive
    generation magnitudes, sign handled internally), each attached to every
    load target node.
    """

    LOAD_KWARGS = frozenset({"load_p_mw", "load_q_mvar", "gen_p_mw", "gen_q_mvar"})

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
        return _mx.create_bus(self._net, base_kv=self._base_kv, **self._grid_kwargs)

    def _check_start_from(self, node_id):
        if node_id in self._start_from_warned:
            return
        model = self._net.node_by_id(node_id).model
        base_kv = getattr(model, "base_kv", None)
        if base_kv is not None and base_kv != self._base_kv:
            self._start_from_warned.add(node_id)
            warnings.warn(
                f"start_from bus {node_id} has base_kv={base_kv} but this "
                f"builder creates buses with base_kv={self._base_kv}. The "
                "builder never changes an existing bus, so the connected "
                "network mixes base voltages and per-unit results are "
                "silently skewed. Create the bus with the matching base_kv "
                "or construct the builder with the bus's base_kv. See the "
                "how-to/express_structures docs page.",
                stacklevel=3,
            )

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
        """Attach an external power grid (slack) to *node_id*.

        Args:
            node_id: bus the slack is attached to.
            **kwargs: forwarded to
                :func:`monee.express.create_ext_power_grid`, e.g. ``p_mw``,
                ``q_mvar``, ``vm_pu``, ``va_degree``, ``max_import_mw``,
                ``max_export_mw``, ``name``.

        Returns:
            The id of the created external-grid child.
        """
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
        """Mirror a :meth:`_Structure.line` on the supply and the return side.

        Args:
            n: number of nodes per side.
            start_from_supply: existing supply node to grow the line from.
            start_from_return: existing return node to grow the line from.
            heat_exchanger_q_mw: one value, or one per node pair, bridging
                supply to return; ``None`` and 0 entries are skipped.
            **kwargs: load setpoints for both sides, see
                :attr:`WaterStructure.LOAD_KWARGS`.

        Returns:
            DhsSegment: the paired segments and their heat exchangers.
        """
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
        """Mirror a :meth:`_Structure.ring` on the supply and the return side.

        The pipes of a closed ring are always bidirectional, whatever
        ``unidirectional`` the builder was created with.

        Args:
            n: number of nodes per ring (at least 3).
            start_from_supply: existing supply node to include in the ring.
            start_from_return: existing return node to include in the ring.
            heat_exchanger_q_mw: one value, or one per node pair, bridging
                supply to return; ``None`` and 0 entries are skipped.
            **kwargs: load setpoints for both sides, see
                :attr:`WaterStructure.LOAD_KWARGS`.

        Returns:
            DhsSegment: the paired rings and their heat exchangers.
        """
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
        hub_loads=False,
        **kwargs,
    ):
        """Mirror a :meth:`_Structure.star` on the supply and the return side.

        Args:
            arms: arm lengths in nodes, excluding the shared hub.
            start_from_supply: existing supply node to use as the hub.
            start_from_return: existing return node to use as the hub.
            heat_exchanger_q_mw: one value, or one per node pair, bridging
                supply to return; ``None`` and 0 entries are skipped.
            hub_loads: also attach the load setpoints to the two hubs;
                default False, only arm nodes get loads.
            **kwargs: load setpoints for both sides, see
                :attr:`WaterStructure.LOAD_KWARGS`.

        Returns:
            DhsSegment: the paired stars and their heat exchangers.
        """
        supply_star = self.supply.star(
            arms=arms, start_from=start_from_supply, hub_loads=hub_loads, **kwargs
        )
        return_star = self.return_.star(
            arms=arms, start_from=start_from_return, hub_loads=hub_loads, **kwargs
        )
        supply_seg = _as_segment(supply_star)
        return_seg = _as_segment(return_star)
        hxs = self._bridge(supply_seg.nodes, return_seg.nodes, heat_exchanger_q_mw)
        return DhsSegment(supply=supply_seg, return_=return_seg, heat_exchangers=hxs)

    def attach_heat_plant(
        self, supply_node, return_node, *, t_k=358.0, pin_temperature=True, name=None
    ):
        """Attach a heat plant: an external hydraulic grid (slack) feeding the
        supply side and a :class:`~monee.model.child.ConsumeHydrGrid`
        absorbing the return side.

        Args:
            supply_node: supply junction the slack feeds (pinned pressure).
            return_node: return junction the consumer drains.
            t_k: feed temperature of the supply slack in K (default 358).
            pin_temperature: True (default) pins the supply junction
                temperature to ``t_k``; False leaves it free, e.g. when a
                timeseries or another component sets the feed temperature.
            name: base name; the children are named ``{name}_supply`` and
                ``{name}_return``.

        Returns:
            Tuple of (supply child id, return child id).
        """
        supply_name = f"{name}_supply" if name else None
        return_name = f"{name}_return" if name else None
        sid = _mx.create_water_ext_grid(
            self._net,
            supply_node,
            t_k=t_k,
            pin_temperature=pin_temperature,
            name=supply_name,
        )
        rid = _mx.create_consume_hydr_grid(self._net, return_node, name=return_name)
        return sid, rid


def gas_structure(
    network,
    *,
    diameter_m,
    length_m,
    temperature_ext_k=296.15,
    roughness_m=1e-5,
    grid=None,
):
    """Builder for gas-pipe topologies (line/ring/star).

    Args:
        network: the network to build into.
        diameter_m: pipe diameter in m, shared by every created pipe.
        length_m: pipe length in m, shared by every created pipe.
        temperature_ext_k: ambient temperature in K (default 296.15).
        roughness_m: pipe wall roughness in m (default 1e-5).
        grid: gas grid for created nodes and pipes; default None uses the
            network's default gas grid (lgas, 10 bar reference pressure,
            300 K operating temperature).

    Returns:
        GasStructure: the configured builder.
    """
    return GasStructure(
        network,
        diameter_m=diameter_m,
        length_m=length_m,
        temperature_ext_k=temperature_ext_k,
        roughness_m=roughness_m,
        grid=grid,
    )


def water_structure(
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
    """Builder for water/heat-pipe topologies (single chain, no return line).

    Args:
        network: the network to build into.
        diameter_m: pipe diameter in m, shared by every created pipe.
        length_m: pipe length in m, shared by every created pipe.
        temperature_ext_k: ambient temperature in K (default 296.15).
        roughness_m: pipe wall roughness in m (default 0.001).
        lambda_insulation_w_per_m_k: insulation heat conductivity
            (default 0.025).
        insulation_thickness_m: insulation thickness in m (default 0.2).
        unidirectional: True pins pipe flow direction (except inside a
            closed ring); default False keeps flow bidirectional.
        grid: water grid for created nodes and pipes; default None uses the
            network's default water grid.

    Returns:
        WaterStructure: the configured builder.
    """
    return WaterStructure(
        network,
        diameter_m=diameter_m,
        length_m=length_m,
        temperature_ext_k=temperature_ext_k,
        roughness_m=roughness_m,
        lambda_insulation_w_per_m_k=lambda_insulation_w_per_m_k,
        insulation_thickness_m=insulation_thickness_m,
        unidirectional=unidirectional,
        grid=grid,
    )


def el_structure(
    network,
    *,
    length_m,
    r_ohm_per_m,
    x_ohm_per_m,
    parallel=1,
    base_kv=1,
    grid=None,
):
    """Builder for electrical-bus topologies (line/ring/star).

    Args:
        network: the network to build into.
        length_m: line length in m, shared by every created line.
        r_ohm_per_m: line resistance per metre.
        x_ohm_per_m: line reactance per metre.
        parallel: number of parallel systems per line (default 1).
        base_kv: base voltage of created buses in kV; default 1, set the
            real voltage level. Applies only to buses the builder creates;
            growing from a ``start_from`` bus with a different ``base_kv``
            emits a ``UserWarning`` naming both values.
        grid: power grid for created nodes and lines; default None uses the
            network's default power grid.

    Returns:
        ElStructure: the configured builder.
    """
    return ElStructure(
        network,
        length_m=length_m,
        r_ohm_per_m=r_ohm_per_m,
        x_ohm_per_m=x_ohm_per_m,
        parallel=parallel,
        base_kv=base_kv,
        grid=grid,
    )


def dhs_structure(
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
    """Builder for paired supply/return district-heating topologies.

    Mirrors every shape on a supply and a return side and can bridge the
    node pairs with heat exchangers oriented supply to return.

    Args:
        network: the network to build into.
        diameter_m: pipe diameter in m, shared by every created pipe.
        length_m: pipe length in m, shared by every created pipe.
        temperature_ext_k: ambient temperature in K (default 296.15).
        roughness_m: pipe wall roughness in m (default 0.001).
        lambda_insulation_w_per_m_k: insulation heat conductivity
            (default 0.025).
        insulation_thickness_m: insulation thickness in m (default 0.2).
        unidirectional: True (default) pins pipe flow direction on both
            sides (except inside closed rings); False keeps flow
            bidirectional.
        grid: water grid for created nodes and pipes; default None uses the
            network's default water grid.

    Returns:
        DhsStructure: the configured builder.
    """
    return DhsStructure(
        network,
        diameter_m=diameter_m,
        length_m=length_m,
        temperature_ext_k=temperature_ext_k,
        roughness_m=roughness_m,
        lambda_insulation_w_per_m_k=lambda_insulation_w_per_m_k,
        insulation_thickness_m=insulation_thickness_m,
        unidirectional=unidirectional,
        grid=grid,
    )

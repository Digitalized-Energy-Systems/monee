r"""Epigraph-relaxed Weymouth gas formulation: a convex MIQCQP (MISOCP-shaped).

Weymouth is linear in pressure-squared space; the only quadratics are the
convex epigraph relaxations :math:`m \cdot m \le m_{sq}`, plus the
``direction``/``on_off`` binaries. A small :math:`\varepsilon` objective term
keeps the epigraph tight when nothing else is minimised (simulation); under an
economic objective it is negligible against the MIP gap, so
:func:`restore_exact_weymouth` restores the exact state after the solve.
"""

import math

import networkx as nx
import numpy as np
from scipy.linalg import null_space
from scipy.optimize import minimize

import monee.model.phys.core.hydraulics as hydraulicsmodel
import monee.model.phys.nonlinear.gf as ogfmodel
from monee.model.core import Const, Intermediate, Var, value
from monee.model.grid import GasGrid

from ...common import ensure_velocity_report
from ...core import BranchFormulation


class RelaxedWeymouthBranchFormulation(BranchFormulation):
    # The epigraph rows bind only through this objective term and through the
    # binaries, so this formulation needs a solver that enforces integrality. On
    # a continuous solver the term sits under the convergence tolerance (IPOPT
    # defaults to 1e-6), and next to a user objective it sits under any relative
    # MIP gap: the relaxation stays slack and the pressure drop comes out several
    # times too large - see :meth:`relaxation_gap` and restore_exact_weymouth.
    EPIGRAPH_TIGHTENING_EPS = 1e-5
    RELAXATION_GAP_TOL = 0.1

    def relaxation_gap(self, model):
        r"""Relative error of the signed squared flow the Weymouth row consumes:
        :math:`(m_{sq,pos} - m_{sq,neg})` against the exact :math:`m |m|` of the
        solved net flow. 0 when the epigraph relaxation came out tight; large
        when the solver kept slack (and the pressure drop is wrong by that much).
        """
        pos = value(model.mass_flow_pos_kgs)
        neg = value(model.mass_flow_neg_kgs)
        signed_sq = value(model.mass_flow_pos_kgs_squared) - value(
            model.mass_flow_neg_kgs_squared
        )
        net = pos - neg
        return abs(signed_sq - net * abs(net)) / max(abs(signed_sq), 1e-6)

    def ensure_var(self, model, simulation=False, grid=None):
        f_const = hydraulicsmodel.friction_at_high_re(
            model.diameter_m, model.roughness_m
        )

        model.friction = Const(f_const)
        model.reynolds_scaled = Const(0.0)
        ensure_velocity_report(model, grid)

    def minimize(self, branch, grid, from_node_model, to_node_model, **kwargs):
        return [
            self.EPIGRAPH_TIGHTENING_EPS
            * (branch.mass_flow_pos_kgs_squared + branch.mass_flow_neg_kgs_squared)
        ]

    def _epigraph_eqs(self, branch):
        r"""Convex epigraph relaxation :math:`m^2 \le m_{sq}`; the exact sibling
        overrides this with the equality form."""
        return [
            branch.mass_flow_pos_kgs * branch.mass_flow_pos_kgs
            <= branch.mass_flow_pos_kgs_squared,
            branch.mass_flow_neg_kgs * branch.mass_flow_neg_kgs
            <= branch.mass_flow_neg_kgs_squared,
        ]

    def equations(self, branch, grid, from_node_model, to_node_model, **kwargs):
        branch._pipe_area = hydraulicsmodel.calc_pipe_area(branch.diameter_m)

        # linearize sqrt(p) around nominal pressure
        p_from, p_to, p_avg = ogfmodel.linearized_pressure_pu(
            grid,
            from_node_model.vars["pressure_squared_pu"],
            to_node_model.vars["pressure_squared_pu"],
        )

        p_amb = getattr(grid, "pressure_ambient_pa", 0.0)
        gas_density_kg_per_m3 = ogfmodel.reference_gas_density(grid)
        f_max_local = hydraulicsmodel.calc_local_max_mass_flow(
            grid, branch, gas_density_kg_per_m3, getattr(grid, "v_max_mps", 20.0)
        )

        return self._epigraph_eqs(branch) + [
            branch.mass_flow_pos_kgs_squared <= f_max_local**2 * branch.direction,
            branch.mass_flow_neg_kgs_squared <= f_max_local**2 * (1 - branch.direction),
            branch.mass_flow_pos_kgs_squared <= f_max_local**2 * branch.on_off,
            branch.mass_flow_neg_kgs_squared <= f_max_local**2 * branch.on_off,
            ogfmodel.pipe_weymouth(
                # ABSOLUTE squared pressure (Pa^2). In gauge mode the offset uses
                # the LINEARIZED pressure p_from/p_to (affine in pressure_squared_pu,
                # the same linearization used for the density above) so the
                # constraint stays conic; a no-op when the grid's ambient is 0.
                p_squared_i=ogfmodel.abs_psq_pu(
                    from_node_model.vars["pressure_squared_pu"],
                    p_from,
                    p_amb,
                    grid.pressure_ref_pa,
                )
                * grid.pressure_ref_pa**2,
                p_squared_j=ogfmodel.abs_psq_pu(
                    to_node_model.vars["pressure_squared_pu"],
                    p_to,
                    p_amb,
                    grid.pressure_ref_pa,
                )
                * grid.pressure_ref_pa**2,
                f_a_pos_sq=branch.mass_flow_pos_kgs_squared,
                f_a_neg_sq=branch.mass_flow_neg_kgs_squared,
                diameter_m=branch.diameter_m,
                length_m=branch.length_m,
                t_k=grid.t_k,
                compressibility=grid.compressibility,
                on_off=branch.on_off,
                friction=branch.friction,
                r_specific=grid.universal_gas_constant / grid.molar_mass,
                **kwargs,
            ),
            ogfmodel.gas_density_eq(branch.gas_density_kg_per_m3, grid, p_avg, p_amb),
        ]


#: An island whose worst epigraph gap stays below this is already exact.
RESTORE_GAP_TOL = 1e-3


def _psq_drop_coefficient(branch):
    r""":math:`k` of the Weymouth row in node-variable space,
    :math:`p_{sq,from} - p_{sq,to} = -k \, m |m|` (``m = pos - neg``); the gauge
    offset is affine in ``pressure_squared_pu``, so its slope is read off the
    same helpers :meth:`RelaxedWeymouthBranchFormulation.equations` uses."""
    grid, model = branch.grid, branch.model
    p_amb = getattr(grid, "pressure_ambient_pa", 0.0)

    def abs_psq(x):
        p_lin = ogfmodel.linearized_pressure_pu(grid, x, x)[0]
        return ogfmodel.abs_psq_pu(x, p_lin, p_amb, grid.pressure_ref_pa)

    c_squared = ogfmodel.calc_c_squared(
        model.diameter_m,
        model.length_m,
        grid.t_k,
        grid.compressibility,
        grid.universal_gas_constant / grid.molar_mass,
    )
    slope = abs_psq(1.0) - abs_psq(0.0)
    return value(model.friction) / (c_squared * grid.pressure_ref_pa**2 * slope)


def _exact_island_state(network, island, pipes):
    """``(flows, {node id: pressure_squared_pu})`` of the exact Weymouth state
    for the island's solved injections, or ``None`` when it is not unique or
    leaves the pressure bounds / flow caps."""
    nodes = sorted(island)
    refs = [
        n
        for n in nodes
        if not isinstance(network.node_by_id(n).model.pressure_squared_pu, Var)
    ]
    if len(refs) != 1:
        return None
    row = {n: i for i, n in enumerate(nodes)}
    incidence = np.zeros((len(nodes), len(pipes)))
    for e, b in enumerate(pipes):
        incidence[row[b.from_node_id], e] = 1.0
        incidence[row[b.to_node_id], e] = -1.0
    m0 = np.array(
        [
            value(b.model.mass_flow_pos_kgs) - value(b.model.mass_flow_neg_kgs)
            for b in pipes
        ]
    )
    k = np.array([_psq_drop_coefficient(b) for b in pipes])
    m = m0
    if len(pipes) >= len(nodes):  # a connected island with a loop
        loops = null_space(incidence)

        # Exact flows minimise sum k|m|^3/3 on the injections' affine flow space.
        def flows(q):
            return m0 + loops @ q

        sol = minimize(
            lambda q: float(np.sum(k * np.abs(flows(q)) ** 3)) / 3.0,
            np.zeros(loops.shape[1]),
            jac=lambda q: loops.T @ (k * flows(q) * np.abs(flows(q))),
            hess=lambda q: loops.T @ ((2.0 * k * np.abs(flows(q)))[:, None] * loops),
            method="trust-exact",
            options={"gtol": 1e-14},
        )
        m = flows(sol.x)
    drop = -k * m * np.abs(m)
    ref = row[refs[0]]
    others = [i for i in range(len(nodes)) if i != ref]
    psq_ref = value(network.node_by_id(refs[0]).model.pressure_squared_pu)
    rhs = drop - incidence[ref] * psq_ref
    x, *_ = np.linalg.lstsq(incidence[others].T, rhs, rcond=None)
    # float: a plain int reference (GEKKO keeps pressure_pu=1 as int) would
    # make an int array that truncates the restored pressures
    psq = np.full(len(nodes), float(psq_ref))
    psq[others] = x
    scale = max(float(np.max(np.abs(drop))), 1e-12)
    if np.max(np.abs(incidence.T @ psq - drop)) > 1e-6 * scale:
        return None
    for n, val in zip(nodes, psq):
        var = network.node_by_id(n).model.pressure_squared_pu
        if isinstance(var, Var) and (
            (var.min is not None and val < var.min - 1e-9)
            or (var.max is not None and val > var.max + 1e-9)
        ):
            return None
    for b, flow in zip(pipes, m):
        cap = hydraulicsmodel.calc_local_max_mass_flow(
            b.grid,
            b.model,
            ogfmodel.reference_gas_density(b.grid),
            getattr(b.grid, "v_max_mps", 20.0),
        )
        if abs(flow) > cap * (1 + 1e-9):
            return None
    return m, dict(zip(nodes, psq))


def _set(attr, val):
    if isinstance(attr, (Var, Intermediate)):
        attr.value = val


def restore_exact_weymouth(network) -> int:
    r"""Replace a slack Weymouth epigraph by the exact state of the solved injections.

    In an optimisation nothing prices gas pressures, so the
    :math:`\varepsilon` term is the only pull on ``mass_flow_*_squared`` and it
    sits orders of magnitude below any relative MIP gap: the solver returns
    squared flows above :math:`m \cdot m`, mesh flows split arbitrarily and
    pressure drops come out too large. Pipe flows and junction pressures of a
    gas island enter no other equation, so for the solved nodal injections the
    exact state is the unique minimiser of :math:`\sum_e k_e |m_e|^3 / 3` under
    the mass balance, with the pressures from the Weymouth rows. It is written
    back for every slack island bounded by pipes only, with exactly one pinned
    pressure and no linepack, whose exact state keeps the pressure bounds and
    flow caps; the relaxed optimum is then optimal for the exact model too
    (custom constraints on gas pressures or pipe flows are not re-checked).
    Other islands keep their relaxed values, which the relaxation check of
    :func:`monee.solver.core.validate_result` reports. Returns the number of
    restored pipes."""
    from monee.model.extension.linepack import GasLinepack

    if any(isinstance(ext, GasLinepack) for ext in network.extensions):
        return 0
    pipes, coupled = [], set()
    for b in network.branches:
        if not b.active or b.ignored:
            continue
        ends = (b.from_node_id, b.to_node_id)
        if isinstance(b.formulation, RelaxedWeymouthBranchFormulation):
            if value(b.model.on_off) > 0.5:
                pipes.append(b)
        elif all(isinstance(network.node_by_id(n).grid, GasGrid) for n in ends):
            coupled.update(ends)
    graph = nx.MultiGraph((b.from_node_id, b.to_node_id) for b in pipes)
    restored = 0
    for island in nx.connected_components(graph):
        members = [b for b in pipes if b.from_node_id in island]
        if island & coupled or any(
            math.isnan(value(b.model.mass_flow_pos_kgs)) for b in members
        ):
            continue
        if (
            max(b.formulation.relaxation_gap(b.model) for b in members)
            <= RESTORE_GAP_TOL
        ):
            continue
        state = _exact_island_state(network, island, members)
        if state is None:
            continue
        flows, psq = state
        for b, m in zip(members, flows):
            model = b.model
            _set(model.mass_flow_pos_kgs, max(m, 0.0))
            _set(model.mass_flow_neg_kgs, max(-m, 0.0))
            _set(model.mass_flow_pos_kgs_squared, max(m, 0.0) ** 2)
            _set(model.mass_flow_neg_kgs_squared, max(-m, 0.0) ** 2)
            _set(model.mass_flow_kgs, m)
            if m != 0.0:
                _set(model.direction, int(m > 0.0))
            grid = b.grid
            p_avg = ogfmodel.linearized_pressure_pu(
                grid, psq[b.from_node_id], psq[b.to_node_id]
            )[2]
            _set(
                model.gas_density_kg_per_m3,
                ogfmodel.gas_density(
                    grid,
                    grid.pressure_ref_pa * p_avg
                    + getattr(grid, "pressure_ambient_pa", 0.0),
                ),
            )
        for n, val in psq.items():
            node_model = network.node_by_id(n).model
            if isinstance(node_model.pressure_squared_pu, Var):
                node_model.pressure_squared_pu.value = val
                _set(node_model.pressure_pu, math.sqrt(max(val, 0.0)))
        restored += len(members)
    return restored

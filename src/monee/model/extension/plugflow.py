r"""
Plug-flow temperature transport in water pipes (node method).

Each pipe holds its water as an ordered sequence of parcels ``(mass, T_entry,
age)``. In a step of length :math:`\Delta t` the outflow is the first
:math:`|\dot m| \Delta t` of the parcels at the downstream end, each cooled with
monee's steady attenuation law for its residence time :math:`a`,

.. math::

    T = T_{ext} + \frac{T_{entry} - T_{ext}}{1 + UA \cdot a / (M c_p)},

which equals the steady pipe law when :math:`a = M / \dot m`; the junctions mix
instantaneously. It is the first-order form of :math:`\exp(-UA \cdot a / (M c_p))`,
so water standing for a pipe's thermal time constant :math:`M c_p / UA` cools
too slowly (it keeps 1/2 of its excess, not 1/e). Fronts travel with the transport delay
:math:`M / \dot m` without the numerical diffusion of a tank cascade
(:class:`~monee.model.extension.ltc.LumpedThermalCapacitance`).
"""

import math

from monee.model.branch import WaterPipe
from monee.model.core import PostProcess, value
from monee.model.phys.nonlinear.hf import SPECIFIC_HEAT_CAP_WATER
from monee.model.phys.nonlinear.wf import pipe_insulation_ua

from .core import NetworkAspect
from .ltc import LumpedThermalCapacitance

_PROFILE = "_plug_profile"
_TINY = 1e-9
_MERGE_PU = 1e-5  # adjacent parcels this close in entry temperature merge (~4 mK)
_MAX_PARCELS = 32  # per pipe; merges stay below this fraction of its mass


class PlugFlowTransport(NetworkAspect):
    """Plug-flow transport delay of the water temperatures in every
    :class:`~monee.model.branch.WaterPipe` of a sequential timeseries
    (:func:`~monee.run_timeseries`, :class:`~monee.simulation.Stepper`).

    The first step is the steady energy flow of its inputs; every later step
    moves the pipe contents of the previous one by the step's solved flow (the
    outflow is a piecewise-linear function of it, its kinks smoothed over
    ``smoothing`` pipe volumes). A pipe fed by a single pipe receives that
    pipe's outflow parcels in their order, so a front keeps its shape across
    junctions; behind a mixing junction the inflow of a step is one parcel at
    the junction's temperature. Use it instead of
    :class:`~monee.model.extension.ltc.LumpedThermalCapacitance` (both would
    store the pipe water). Needs an NLP back-end (the smoothing takes a square
    root). Single-step solves are unaffected.
    """

    def __init__(self, smoothing: float = 1e-5):
        self.smoothing = float(smoothing)

    def prepare(self, network) -> None:
        if any(isinstance(e, LumpedThermalCapacitance) for e in network.extensions):
            raise ValueError(
                "PlugFlowTransport and LumpedThermalCapacitance both store the "
                "pipe water; attach only one of them."
            )

    def activate_timeseries(self, network, ignored_nodes: set, step_state=None) -> None:
        if step_state is None:
            raise NotImplementedError(
                "PlugFlowTransport needs the solved previous step: use the "
                "sequential run_timeseries loop or a Stepper (multi-period and "
                "build-once drivers are not supported)."
            )
        for branch in _pipes(network, ignored_nodes):
            if step_state.get(branch.id, _PROFILE) is not None:
                _require_outlet_law(branch)
                branch.model._plug_flow_active = True

    def inter_step_equations(self, network, ignored_nodes: set, step_state) -> list:
        dt_s = step_state.dt_h * 3600.0
        pipes = {
            b.id: _Pipe(b, step_state.get(b.id, _PROFILE))
            for b in _pipes(network, ignored_nodes)
        }
        eqs = []
        for pipe in pipes.values():
            setattr(
                pipe.model,
                _PROFILE,
                PostProcess(_advance_fn(pipe, pipes, network, dt_s)),
            )
            if pipe.profile is not None:
                eqs.append(pipe.outlet_equation(dt_s, self.smoothing))
        return eqs


class _Pipe:
    def __init__(self, branch, profile):
        self.branch, self.model, self.profile = branch, branch.model, profile
        bm = branch.model
        self.mass = (
            branch.grid.fluid_density_kg_per_m3
            * math.pi
            / 4
            * bm.diameter_m**2
            * bm.length_m
        )
        self.ua_c = pipe_insulation_ua(bm) / SPECIFIC_HEAT_CAP_WATER
        self.t_ext = bm.temperature_ext_k / branch.grid.t_ref_k

    def exit_t(self, theta, age, centre, mag):
        """Outlet temperature of a parcel whose centre is ``centre`` kg from the
        outlet: every element of it leaves at the age of the centre, ``age +
        centre / mag`` (exact under constant flow). Floats or solver terms."""
        held = self.mass * mag
        return self.t_ext + (theta - self.t_ext) * held / (
            held + self.ua_c * (age * mag + centre)
        )

    def outlet_equation(self, dt_s, eps):
        bm = self.model
        mag, mpos, mneg = (
            bm.mass_flow_mag_kgs,
            bm.mass_flow_pos_kgs,
            bm.mass_flow_neg_kgs,
        )
        x = mag * (dt_s / self.mass)  # outflow of the step in pipe volumes
        # the inflow of this step that also leaves in it took M / m: steady law
        t_flush = self.t_ext + bm.alpha * (bm.t_in_pu - self.t_ext)
        from_end, to_end = (
            self._outflow(parcels, x, mag, t_flush, eps)
            for parcels in (self.profile, self.profile[::-1])
        )
        # mass_flow_pos runs to -> from, so its outlet is the from end
        return mag * x * bm.t_out_pu == mpos * from_end + mneg * to_end

    def _outflow(self, parcels, x, mag, t_flush, eps):
        r""":math:`\int_0^x T` over the outflow ``x`` [pipe volumes] at the end
        ``parcels`` starts at: the parcels, then this step's inflow."""
        temps, edges, c = [], [], 0.0
        for mu, theta, age in parcels:
            temps.append(self.exit_t(theta, age, c + mu / 2, mag))
            c += mu
            edges.append(c / self.mass)
        out = temps[0] * x
        for j in range(len(temps) - 1):
            out += (temps[j + 1] - temps[j]) * _relu(x - edges[j], eps)
        return out + (t_flush - temps[-1]) * _relu(x - 1.0, eps)


def _pipes(network, ignored_nodes):
    for branch in network.branches:
        if not isinstance(branch.model, WaterPipe) or not branch.active:
            continue
        if branch.ignored or {branch.from_node_id, branch.to_node_id} & ignored_nodes:
            continue
        yield branch


def _require_outlet_law(branch):
    if not all(hasattr(branch.model, k) for k in ("t_out_pu", "alpha")):
        raise NotImplementedError(
            f"PlugFlowTransport needs a pipe formulation with the attenuation "
            f"law (t_out_pu, alpha); {type(branch.formulation).__name__} of pipe "
            f"{branch.id} has none."
        )


def _relu(y, eps):
    return 0.5 * (y + (y * y + eps * eps) ** 0.5)


def _signed_kgs(model):
    """Solved flow from the from end to the to end [kg/s]."""
    return value(model.mass_flow_neg_kgs) - value(model.mass_flow_pos_kgs)


def _inflow(pipe, node_id, m, pipes, network, dt_s):
    """The parcels entering ``pipe`` at ``node_id`` during the step, earliest
    first, ages at its end: the outflow of the single pipe feeding the node,
    or ``None`` when the node mixes (several feeds, an injection, heat added
    or a temperature pinned at the node, a feed that is no plug-flow pipe)."""
    node = network.node_by_id(node_id)
    for child in network.childs_by_ids(node.child_ids):
        cm = child.model
        if not child.active or getattr(child, "ignored", False):
            continue
        # the heat terms of Junction.calc_signed_heat_flow besides m * t
        if hasattr(cm, "q_mw_heat") or getattr(cm, "pin_temperature", False):
            return None
        if not hasattr(cm, "mass_flow_kgs"):
            continue
        if value(cm.mass_flow_kgs) * value(getattr(cm, "regulation", 1)) < -_TINY:
            return None
    feeds = []
    for bid in (*node.from_branch_ids, *node.to_branch_ids):
        branch = network.branch_by_id(bid)
        if not branch.active or branch.ignored:
            continue
        if branch.to_node_id == node_id and hasattr(branch.model, "q_mw_heat"):
            return None
        if not hasattr(branch.model, "mass_flow_pos_kgs"):
            continue
        s = _signed_kgs(branch.model) * (1 if branch.to_node_id == node_id else -1)
        if s > _TINY:
            feeds.append((branch, s))
    if len(feeds) != 1 or feeds[0][0].id not in pipes or m <= _TINY:
        return None
    feed, m_up = feeds[0]
    up = pipes[feed.id]
    if up.profile is None:
        return None
    x_up, scale = m_up * dt_s, m / m_up
    pieces, c = [], 0.0
    for mu, theta, age in (
        up.profile[::-1] if feed.to_node_id == node_id else up.profile
    ):
        if c >= x_up:
            break
        part = min(mu, x_up - c)
        t_out = up.exit_t(theta, age, c + mu / 2, m_up)
        pieces.append((part * scale, t_out, dt_s - (c + part / 2) / m_up))
        c += mu
    if x_up > up.mass:  # the feed's own inflow of the step passes through it
        rest = x_up - up.mass
        t_in = value(feed.model.t_in_pu)
        t_out = up.t_ext + (t_in - up.t_ext) * m_up / (m_up + up.ua_c)
        pieces.append((rest * scale, t_out, dt_s - (up.mass + rest / 2) / m_up))
    return pieces


def _advance_fn(pipe, pipes, network, dt_s):
    """Post-process of a solved step: the pipe contents at its end, from the
    from end to the to end, ages at the end of the step."""

    def advance(v):
        signed = v.mass_flow_neg_kgs - v.mass_flow_pos_kgs
        m = abs(signed)
        x = m * dt_s
        if pipe.profile is None:
            return ((pipe.mass, v.t_in_pu, pipe.mass / (2 * m) if m > 0 else 0.0),)
        inlet = pipe.branch.from_node_id if signed >= 0 else pipe.branch.to_node_id
        pieces = _inflow(pipe, inlet, m, pipes, network, dt_s) or (
            [(x, v.t_in_pu, dt_s / 2)] if x > _TINY else []
        )
        old = pipe.profile if signed >= 0 else pipe.profile[::-1]
        seq = pieces[::-1] + [
            (mu, theta, age + dt_s) for mu, theta, age in old
        ]  # inlet first
        need = x
        while need > _TINY and seq:
            mu, theta, age = seq[-1]
            if mu <= need:
                seq.pop()
                need -= mu
            else:
                # the part left behind entered later: its centre is younger
                seq[-1] = (mu - need, theta, age - need / (2 * m))
                need = 0.0
        seq = _merge(seq, pipe.mass / _MAX_PARCELS)
        return tuple(seq if signed >= 0 else seq[::-1])

    return advance


def _merge(seq, max_kg):
    """Merge neighbours of (almost) the same entry temperature up to ``max_kg``
    (mass-weighted temperature and centre age); beyond ``_MAX_PARCELS`` the
    neighbours whose merge loses the least heat contrast merge, which keeps
    fronts and bounds the size of the next outlet equation."""
    out = []
    for parcel in seq:
        if (
            out
            and abs(out[-1][1] - parcel[1]) < _MERGE_PU
            and out[-1][0] + parcel[0] <= max_kg
        ):
            out[-1] = _combined(out[-1], parcel)
        else:
            out.append(parcel)
    while len(out) > _MAX_PARCELS:
        j = min(range(len(out) - 1), key=lambda i: _contrast(out[i], out[i + 1]))
        out[j : j + 2] = [_combined(out[j], out[j + 1])]
    return out


def _combined(a, b):
    m = a[0] + b[0]
    return (m, (a[0] * a[1] + b[0] * b[1]) / m, (a[0] * a[2] + b[0] * b[2]) / m)


def _contrast(a, b):
    """Loss of temperature variance [kg pu^2] when ``a`` and ``b`` merge."""
    return a[0] * b[0] / (a[0] + b[0]) * (a[1] - b[1]) ** 2

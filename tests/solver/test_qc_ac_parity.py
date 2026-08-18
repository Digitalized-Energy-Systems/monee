"""QC-with-switch must reproduce the exact polar-AC NLP on radial and meshed grids.

Guards the four defects that made it diverge by orders of magnitude:

* no ``minimize`` hook, so nothing drove the relaxation tight and a feasibility
  solve returned an arbitrary point of the relaxed set (0.5 MW load -> 56 MW slack),
* ``current_soc_relax`` ignoring its ``v_sq_var`` argument in favour of a constant,
* ``on_off`` accepted but unused in the four flow equations, so an open branch
  still carried flow,
* ``i_*_ka`` / ``loading_*_pu`` left as free Vars under a one-sided ``>=``,
  reported at ~1e5 kA.
"""

import pytest

import monee.model as mm
from monee import mx, run_energy_flow, run_energy_flow_optimization
from monee.model.core import Var
from monee.model.formulation import EL_NLP_FORMULATION, EL_QC_FORMULATION
from monee.problem import create_min_load_shedding_problem
from monee.solver import PyomoSolver

# Both solves are exact here, so the tolerance only has to absorb the MIP gap.
TOL = 1e-4


def _radial(n=6, dg_mw=0.0):
    net = mx.create_multi_energy_network()
    buses = [mx.create_bus(net, base_kv=20.0)]
    mx.create_ext_power_grid(net, buses[0], vm_pu=1.0, va_degree=0.0)
    for _ in range(n):
        b = mx.create_bus(net, base_kv=20.0)
        buses.append(b)
        mx.create_line(
            net, buses[-2], b, length_m=1500, r_ohm_per_m=1.6e-4, x_ohm_per_m=1.2e-4
        )
        mx.create_power_load(net, b, p_mw=0.05, q_mvar=0.02)
    if dg_mw:
        mx.create_power_generator(net, buses[-1], p_mw=dg_mw, q_mvar=0.0)
    return net


def _mesh():
    net = mx.create_multi_energy_network()
    b = [mx.create_bus(net, base_kv=20.0) for _ in range(6)]
    mx.create_ext_power_grid(net, b[0], vm_pu=1.0, va_degree=0.0)
    for f, t in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (1, 4)]:
        mx.create_line(
            net, b[f], b[t], length_m=800, r_ohm_per_m=1.6e-4, x_ohm_per_m=1.2e-4
        )
    for i in range(1, 6):
        mx.create_power_load(net, b[i], p_mw=0.04, q_mvar=0.015)
    return net


def _summary(result, lifted):
    """*lifted* selects where the squared voltage comes from: the QC solve has it
    as a real variable, the polar-AC solve only reports ``vm_pu``.

    The comparison is deliberately on ``w = vm_pu^2`` rather than ``vm_pu``: in
    the QC model ``vm_pu`` is only the McCormick lifting variable for the
    ``v_i * v_j`` product and is tied to ``w`` by a single chord.
    """
    bus, line = result.get(mm.Bus), result.get(mm.PowerLine)
    return {
        "slack_p": -bus["p_mw"].iloc[0],
        "w_min": bus["vm_pu_squared"].min() if lifted else bus["vm_pu"].min() ** 2,
        "p_head": line["p_from_mw"].iloc[0],
        # every line, not just the head: the currents used to divide by the
        # vm_pu lifting variable and came out ~14% high from the second line on.
        "i_from_max_err_ref": tuple(round(v, 9) for v in line["i_from_ka"]),
        "i_to_max_err_ref": tuple(round(v, 9) for v in line["i_to_ka"]),
        "loss": line["p_from_mw"].sum() + line["p_to_mw"].sum(),
    }


def _solve(builder, formulation, problem=None, **kwargs):
    net = builder(**kwargs)
    net.apply_formulation(formulation)
    lifted = formulation is not EL_NLP_FORMULATION
    if formulation is EL_NLP_FORMULATION and problem is None:
        return _summary(run_energy_flow(net), lifted)
    return _summary(
        run_energy_flow_optimization(net, problem, solver=PyomoSolver()), lifted
    )


@pytest.mark.parametrize(
    "builder,kwargs", [(_radial, {}), (_radial, {"dg_mw": 1.5}), (_mesh, {})]
)
def test_qc_matches_ac_power_flow(builder, kwargs):
    ac = _solve(builder, EL_NLP_FORMULATION, **kwargs)
    qc = _solve(builder, EL_QC_FORMULATION, **kwargs)
    for key, ac_value in ac.items():
        assert qc[key] == pytest.approx(ac_value, abs=TOL), key
    # w must equal the AC vm_pu^2, not merely be self-consistent.
    assert qc["w_min"] > 0.9


def test_qc_matches_ac_under_load_shedding_opf():
    """The load-shedding objective does not price losses - without the
    formulation's own ``r * l`` term nothing pulls the relaxation tight and the
    slack ran to its bound (2.999 MW against a true 0.3 MW)."""
    ac = _solve(
        _radial,
        EL_NLP_FORMULATION,
        create_min_load_shedding_problem(bounds_ext_el=(-3, 3)),
    )
    qc = _solve(
        _radial,
        EL_QC_FORMULATION,
        create_min_load_shedding_problem(bounds_ext_el=(-3, 3)),
    )
    for key, ac_value in ac.items():
        assert qc[key] == pytest.approx(ac_value, abs=TOL), key


def test_qc_reported_current_is_pinned_not_free():
    """i_*_ka / loading_*_pu are equality-pinned intermediates, not free Vars,
    and are referenced to sqrt(w) rather than to the vm_pu lifting variable."""
    qc = _solve(_radial, EL_QC_FORMULATION)
    ac = _solve(_radial, EL_NLP_FORMULATION)
    for side in ("i_from_max_err_ref", "i_to_max_err_ref"):
        for qc_value, ac_value in zip(qc[side], ac[side], strict=True):
            assert qc_value == pytest.approx(ac_value, abs=TOL)
    assert max(qc["i_from_max_err_ref"]) < 1.0  # the free-Var regression gave ~1e5


@pytest.mark.parametrize("as_var", [False, True])
def test_qc_open_branch_carries_no_flow(as_var):
    """on_off = 0 must zero the flow - it was ignored by all four flow equations.

    Covers both a constant 0 and a Var pinned to 0. The constant-0 case used to
    be misclassified as non-switchable, pinning cs >= cos(theta_u) against
    ``cosine_relax``'s cs <= 0 — infeasible.
    """
    net = mx.create_multi_energy_network()
    b = [mx.create_bus(net, base_kv=20.0) for _ in range(3)]
    mx.create_ext_power_grid(net, b[0], vm_pu=1.0, va_degree=0.0)
    for f, t in [(0, 1), (1, 2), (0, 2)]:
        mx.create_line(
            net, b[f], b[t], length_m=500, r_ohm_per_m=1.6e-4, x_ohm_per_m=1.2e-4
        )
    mx.create_power_load(net, b[1], p_mw=0.05, q_mvar=0.02)
    mx.create_power_load(net, b[2], p_mw=0.05, q_mvar=0.02)

    open_branch = list(net.branches)[2]
    open_branch.model.on_off = Var(0, min=0, max=0, name="on_off") if as_var else 0
    net.apply_formulation(EL_QC_FORMULATION)

    result = run_energy_flow_optimization(net, None, solver=PyomoSolver())
    line = result.get(mm.PowerLine)
    row = line[line["id"] == open_branch.id].iloc[0]
    assert row["p_from_mw"] == pytest.approx(0.0, abs=TOL)
    assert row["q_from_mvar"] == pytest.approx(0.0, abs=TOL)
    assert row["p_to_mw"] == pytest.approx(0.0, abs=TOL)
    assert row["q_to_mvar"] == pytest.approx(0.0, abs=TOL)


def test_qc_voltage_band_is_opt_in_and_tightens_the_lifting_variable():
    """The default band comes from the grid, so it can never cut off a solution
    the AC NLP finds; tightening it is opt-in and only sharpens ``vm_pu``."""
    from monee.model.formulation.core import NetworkFormulation
    from monee.model.formulation.quadratic_convex.cq_with_switch import (
        QCElectricityBranchFormulation,
        QCElectricityNodeFormulation,
    )

    ac_net = _radial()
    ac_net.apply_formulation(EL_NLP_FORMULATION)
    ac_bus = run_energy_flow(ac_net).get(mm.Bus)

    tight = NetworkFormulation(
        branch_type_to_formulations={
            mm.GenericPowerBranch: QCElectricityBranchFormulation(v_min=0.9, v_max=1.1)
        },
        node_type_to_formulations={
            mm.Bus: QCElectricityNodeFormulation(v_min=0.9, v_max=1.1)
        },
    )
    for formulation, band_width in ((EL_QC_FORMULATION, 1.0), (tight, 0.2)):
        net = _radial()
        net.apply_formulation(formulation)
        bus = run_energy_flow_optimization(net, None, solver=PyomoSolver()).get(mm.Bus)
        # vm_pu_from_w is exact at either band ...
        assert bus["vm_pu_from_w"].min() == pytest.approx(
            ac_bus["vm_pu"].min(), abs=TOL
        )
        # ... while the raw lifting variable is only bounded by (v_max-v_min)^2/8.
        assert (bus["vm_pu"] - ac_bus["vm_pu"]).abs().max() <= band_width**2 / 8 + TOL


# ---------------------------------------------------------------------------
# Regressions found by adversarial review of the fix itself.
# ---------------------------------------------------------------------------


def _shunt_feeder(c_nf_per_km=120.0):
    """Feeder whose lines carry charging susceptance, so the branches are no
    longer pure series and |S_from|^2 = w_i * l_series does NOT hold."""
    import math

    net = mx.create_multi_energy_network()
    buses = [mx.create_bus(net, base_kv=20.0)]
    mx.create_ext_power_grid(net, buses[0], vm_pu=1.0, va_degree=0.0)
    for _ in range(3):
        b = mx.create_bus(net, base_kv=20.0)
        buses.append(b)
        bid = mx.create_line(
            net, buses[-2], b, length_m=1500, r_ohm_per_m=1.6e-4, x_ohm_per_m=2.4e-4
        )
        model = net.branch_by_id(bid).model
        b_sh = 2 * math.pi * 50 * c_nf_per_km * 1e-9 * 1500 / 2 * (20.0**2 / 1.0)
        model.b_fr_pu = b_sh
        model.b_to_pu = b_sh
        mx.create_power_load(net, b, p_mw=0.06, q_mvar=0.021)
    return net


@pytest.mark.parametrize("c_nf_per_km", [12.0, 120.0, 400.0])
def test_qc_does_not_cut_off_ac_points_on_shunt_branches(c_nf_per_km):
    """The naive cone ``p^2 + q^2 <= i*w`` EXCLUDES AC-feasible points here.

    ``current_flow_equation`` defines ``i_qc`` as the *series* current while
    ``p``/``q`` are the *terminal* flows, which include the shunt draw. The
    shunt-corrected cone subtracts it, and is then exact rather than merely
    valid — hence the tight tolerance.
    """
    net = _shunt_feeder(c_nf_per_km)
    net.apply_formulation(EL_NLP_FORMULATION)
    ac = _summary(run_energy_flow(net), lifted=False)

    net = _shunt_feeder(c_nf_per_km)
    net.apply_formulation(EL_QC_FORMULATION)
    qc = _summary(
        run_energy_flow_optimization(net, None, solver=PyomoSolver()), lifted=True
    )
    assert qc["slack_p"] == pytest.approx(ac["slack_p"], abs=TOL)


def test_qc_load_shedding_is_writable_as_an_lp():
    """``check_lp=True`` adds a line-loading limit. Routed through
    ``loading_*_pu`` it carries a sqrt and gurobi refuses the model; the
    formulation therefore exposes the MISOCP-style linear
    ``current_pu_squared`` path instead."""
    net = _radial()
    net.apply_formulation(EL_QC_FORMULATION)
    result = run_energy_flow_optimization(
        net,
        create_min_load_shedding_problem(bounds_ext_el=(-5, 5), check_lp=True),
        solver=PyomoSolver(),
    )
    assert str(result.termination_condition) == "optimal"


def test_qc_respects_the_upper_voltage_limit():
    """``bounds_vm`` has to bind the physics, not the lifting variable.

    ``square_relax`` gives ``vm_pu <= sqrt(w)``, so an upper bound on ``vm_pu``
    alone leaves the true voltage free to overshoot it — the problem bounds
    ``vm_pu_squared`` as well.
    """
    net = mx.create_multi_energy_network()
    b = [mx.create_bus(net, base_kv=20.0) for _ in range(4)]
    mx.create_ext_power_grid(net, b[0], vm_pu=1.0, va_degree=0.0)
    for i in range(3):
        mx.create_line(
            net, b[i], b[i + 1], length_m=6000, r_ohm_per_m=3.2e-4, x_ohm_per_m=3.2e-4
        )
        mx.create_power_load(net, b[i + 1], p_mw=0.02, q_mvar=0.008)
    mx.create_power_generator(net, b[-1], p_mw=20.0, q_mvar=0.0)
    net.apply_formulation(EL_QC_FORMULATION)

    result = run_energy_flow_optimization(
        net,
        create_min_load_shedding_problem(bounds_vm=(0.9, 1.1), bounds_ext_el=(-9, 9)),
        solver=PyomoSolver(),
    )
    bus = result.get(mm.Bus)
    assert bus["vm_pu_from_w"].max() <= 1.1 + 1e-4
    assert bus["vm_pu_from_w"].max() == pytest.approx(1.1, abs=1e-3)  # limit binds


def test_qc_switch_survives_a_problem_promoted_on_off():
    """``on_off`` promoted by a bare ``controllable()`` on a non-backup branch.

    The tight ``cs >= cos(theta_u)`` range must not be baked into the Var
    bounds at ``ensure_var`` time — combined with ``cosine_relax``'s
    ``cs <= on_off`` it would force the binary to 1 and silently disable
    switching.
    """
    from monee.problem.core import AttributeParameter

    net = mx.create_multi_energy_network()
    b = [mx.create_bus(net, base_kv=20.0) for _ in range(3)]
    mx.create_ext_power_grid(net, b[0], vm_pu=1.0, va_degree=0.0)
    for f, t in [(0, 1), (1, 2), (0, 2)]:
        mx.create_line(
            net, b[f], b[t], length_m=500, r_ohm_per_m=1.6e-4, x_ohm_per_m=1.2e-4
        )
    mx.create_power_load(net, b[1], p_mw=0.05, q_mvar=0.02)
    mx.create_power_load(net, b[2], p_mw=0.05, q_mvar=0.02)
    target = list(net.branches)[2].id
    net.apply_formulation(EL_QC_FORMULATION)

    problem = create_min_load_shedding_problem(bounds_ext_el=(-3, 3))
    problem.controllable(
        [
            (
                "on_off",
                AttributeParameter(
                    min=lambda a, v: 0,
                    max=lambda a, v: 1,
                    val=lambda a, v: 1,
                    integer=True,
                ),
            )
        ],
        component_condition=lambda c: getattr(c, "id", None) == target,
    )
    result = run_energy_flow_optimization(net, problem, solver=PyomoSolver())
    assert str(result.termination_condition) == "optimal"
    line = result.get(mm.PowerLine)
    row = line[line["id"] == target].iloc[0]
    # Whatever the solver picks, an open branch must carry nothing.
    if row["on_off"] < 0.5:
        assert row["p_from_mw"] == pytest.approx(0.0, abs=TOL)
        assert row["p_to_mw"] == pytest.approx(0.0, abs=TOL)


def test_qc_transformer_losses_are_not_over_charged():
    """``i_qc`` must use the full complex turns ratio.

    The paper's ``w_i + w_j - 2*wc`` is the tap = 1, shift = 0 special case; on
    a transformer it overstates the series current, and since ``minimize``
    charges ``r * i_qc`` that lands in the objective as a phantom loss.
    """

    def _with_trafo():
        net = mx.create_multi_energy_network()
        hv = mx.create_bus(net, base_kv=110.0)
        lv = mx.create_bus(net, base_kv=20.0)
        mx.create_ext_power_grid(net, hv, vm_pu=1.0, va_degree=0.0)
        mx.create_trafo(net, lv, hv, vk_percent=12.2, vkr_percent=0.25, sn_trafo_mva=40)
        prev = lv
        for _ in range(3):
            b = mx.create_bus(net, base_kv=20.0)
            mx.create_line(
                net, prev, b, length_m=1500, r_ohm_per_m=1.6e-4, x_ohm_per_m=1.2e-4
            )
            mx.create_power_load(net, b, p_mw=0.05, q_mvar=0.02)
            prev = b
        return net

    net = _with_trafo()
    net.apply_formulation(EL_NLP_FORMULATION)
    ac = _summary(run_energy_flow(net), lifted=False)

    net = _with_trafo()
    net.apply_formulation(EL_QC_FORMULATION)
    qc = _summary(
        run_energy_flow_optimization(net, None, solver=PyomoSolver()), lifted=True
    )
    assert qc["slack_p"] == pytest.approx(ac["slack_p"], abs=TOL)
    assert qc["loss"] == pytest.approx(ac["loss"], abs=TOL)

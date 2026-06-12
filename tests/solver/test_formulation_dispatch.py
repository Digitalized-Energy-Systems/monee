"""Solve-time formulation dispatch: registry resolution, per-component
resolution order (pinned > solver arg > apply_formulation > default) and the
end-to-end ``formulation=`` solver parameter."""

import pytest

import monee
import monee.express as mx
import monee.model as mm
import monee.solver as ms
from monee.model.formulation import (
    EL_MISOCP_FORMULATION,
    FORMULATIONS,
    NetworkFormulation,
    register_formulation,
    resolve_formulation,
)
from monee.model.formulation.miqcqp.convex.el import (
    MISOCPElectricityBranchFormulation,
)
from monee.model.formulation.nlp.el import AcPolarNlpBranchFormulation
from monee.model.formulation.registry import attach_formulations


def _small_power_net():
    net = mm.Network()
    b0 = mx.create_bus(net)
    b1 = mx.create_bus(net)
    line = mx.create_line(net, b0, b1, length_m=500, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)
    mx.create_ext_power_grid(net, b0)
    mx.create_power_load(net, b1, p_mw=0.2, q_mvar=0.05)
    return net, line


def _line_formulation(net, line_id):
    return net.branch_by_id(line_id).formulation


# Registry


def test_resolve_formulation_registry_key():
    assert isinstance(resolve_formulation("el_misocp"), NetworkFormulation)


def test_resolve_formulation_passthrough_and_none():
    assert resolve_formulation(EL_MISOCP_FORMULATION) is EL_MISOCP_FORMULATION
    assert resolve_formulation(None) is None


def test_resolve_formulation_sequence_merges_left_to_right():
    merged = resolve_formulation(["smooth_nlp", "el_misocp"])
    branch_form = merged.lookup(mm.GenericPowerBranch(1, 0, 1e-3, 1e-3, 0, 0, 0, 0), None)
    assert isinstance(branch_form, MISOCPElectricityBranchFormulation)


def test_resolve_formulation_unknown_key_lists_options():
    with pytest.raises(KeyError, match="convex_miqcqp"):
        resolve_formulation("does_not_exist")


def test_register_formulation_roundtrip():
    sentinel = NetworkFormulation()
    register_formulation("_test_sentinel", sentinel)
    try:
        assert resolve_formulation("_test_sentinel") is sentinel
    finally:
        del FORMULATIONS["_test_sentinel"]


# Resolution order (unit level, via attach_formulations)


def test_fallback_default_attaches_at_solve_time():
    net, line = _small_power_net()
    assert _line_formulation(net, line) is None

    attach_formulations(net)
    assert isinstance(_line_formulation(net, line), AcPolarNlpBranchFormulation)


def test_apply_formulation_choice_is_used_without_solver_arg():
    net, line = _small_power_net()
    net.apply_formulation(EL_MISOCP_FORMULATION)

    attach_formulations(net)
    assert isinstance(_line_formulation(net, line), MISOCPElectricityBranchFormulation)


def test_solver_arg_overrides_apply_formulation():
    net, line = _small_power_net()
    net.apply_formulation(EL_MISOCP_FORMULATION)

    attach_formulations(net, "el_nlp")
    assert isinstance(_line_formulation(net, line), AcPolarNlpBranchFormulation)


def test_builder_pinned_formulation_survives_solver_arg():
    net, _ = _small_power_net()
    pinned = MISOCPElectricityBranchFormulation()
    line2 = net.branch(
        mm.PowerLine(500, 7e-5, 7e-5, 1),
        from_node_id=0,
        to_node_id=1,
        formulation=pinned,
    )

    attach_formulations(net, "el_nlp")
    assert _line_formulation(net, line2) is pinned


def test_apply_formulation_has_no_var_side_effect():
    net, _ = _small_power_net()
    bus_model = net.node_by_id(0).model
    min_before = bus_model.vm_pu_squared.min

    # MISOCP's ensure_var would redeclare vm_pu_squared with min=0; the
    # declarative apply must not touch model variables.
    net.apply_formulation(EL_MISOCP_FORMULATION)
    assert net.node_by_id(0).model.vm_pu_squared.min == min_before


# End to end


def test_solve_with_formulation_key_pyomo():
    net, _ = _small_power_net()
    result = ms.PyomoSolver().solve(net, formulation="el_misocp")
    assert result.success
    # current_pu is MISOCP-only - proves the solver-arg formulation ran.
    assert "current_pu" in result.dataframes["PowerLine"].columns
    # The user's network stays pristine.
    assert all(c.formulation is None for c in net.all_components())


def test_solve_default_fallback_gekko_simulation():
    net, _ = _small_power_net()
    result = monee.run_energy_flow(net)
    assert result.success


def test_run_energy_flow_threads_formulation_kwarg():
    net, _ = _small_power_net()
    result = monee.run_energy_flow(
        net, solver="gurobi", formulation="el_misocp", simulation=False
    )
    assert result.success

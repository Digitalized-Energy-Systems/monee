import warnings
from types import SimpleNamespace

import pytest

import monee
import monee.model as mm
from monee.model import Bus, ExtPowerGrid, Network, PowerGrid, PowerLine, PowerLoad
from monee.model.core import Var
from monee.solver import PyomoSolver
from monee.solver.core import (
    ResultValidationSummary,
    ResultWarning,
    ValidationError,
    _carrier_balance_entries,
    _he_duty_shortfall,
    _integrality_entry,
    _model_bound_entries,
    _relaxation_entry,
    _served_delta_entries,
    _uncertified_relaxation_entry,
    inject_nans,
    reset_validation_summary_counts,
    validate_result,
    validation_summary_counts,
)


def _two_bus_net(load_mw=5.0):
    pn = Network(PowerGrid(name="power", sn_mva=1))
    n0 = pn.node(
        Bus(base_kv=1),
        child_ids=[pn.child(ExtPowerGrid(p_mw=0, q_mvar=0, vm_pu=1.0, va_degree=0))],
        grid=mm.EL,
    )
    n1 = pn.node(
        Bus(base_kv=1),
        child_ids=[pn.child(PowerLoad(p_mw=load_mw, q_mvar=0))],
        grid=mm.EL,
    )
    pn.branch(
        PowerLine(length_m=100, r_ohm_per_m=0.00007, x_ohm_per_m=0.00007, parallel=1),
        n0,
        n1,
    )
    return pn


def _capped_shedding_result(**kwargs):
    return monee.solve_load_shedding_problem(
        _two_bus_net(), bounds_ext_el=(-2, 0), check_lp=False, **kwargs
    )


def test_clean_net_has_no_warnings():
    result = monee.run_energy_flow(_two_bus_net())

    assert result.success
    assert result.warnings == []


def test_shedding_net_attaches_typed_warnings():
    result = _capped_shedding_result()

    assert result.success
    categories = {w.category for w in result.warnings}
    # Intentional shedding: regulation is an optimisation Var, so the optimal
    # curtailment is not reported as a served_delta finding.
    assert "served_delta" not in categories
    assert "bound_active" in categories
    bound = next(w for w in result.warnings if w.category == "bound_active")
    assert "ExtPowerGrid" in bound.component
    assert "min bound -2" in bound.message


def test_strict_mode_raises_listing_entries():
    with pytest.raises(ValidationError) as excinfo:
        _capped_shedding_result(strict=True)

    assert len(excinfo.value.warnings) >= 1
    assert "bound_active" in str(excinfo.value)
    assert "diagnose_infeasibility" in str(excinfo.value)


def test_strict_mode_passes_on_clean_net():
    result = monee.run_energy_flow(_two_bus_net(), strict=True)
    assert result.success
    assert result.warnings == []


def test_strict_mode_on_pyomo_backend():
    net = _two_bus_net()
    result = monee.run_energy_flow(net, solver=PyomoSolver(), simulation=False)
    assert result.success
    assert result.warnings == []
    with pytest.raises(ValidationError):
        monee.solve_load_shedding_problem(
            _two_bus_net(),
            bounds_ext_el=(-2, 0),
            check_lp=False,
            solver=PyomoSolver(),
            strict=True,
        )


def test_carrier_balance_entry_on_corrupted_solution():
    result = monee.run_energy_flow(_two_bus_net())
    network = result.network
    load = network.childs[1].model
    val = load.p_mw.value if isinstance(load.p_mw, mm.Var) else float(load.p_mw)
    load.p_mw = val + 1.0

    entries = _carrier_balance_entries(network, set())

    assert len(entries) == 1
    assert entries[0].category == "energy_balance"
    assert entries[0].value == pytest.approx(1.0, abs=1e-3)


def test_carrier_balance_closes_on_valid_solution():
    result = monee.run_energy_flow(_two_bus_net())
    assert _carrier_balance_entries(result.network, set()) == []


def test_validate_result_skips_failed_solves():
    result = monee.run_energy_flow(_two_bus_net())
    result.success = False
    result.warnings.clear()
    network = result.network
    load = network.childs[1].model
    val = load.p_mw.value if isinstance(load.p_mw, mm.Var) else float(load.p_mw)
    load.p_mw = val + 1.0

    validated = validate_result(network, result, strict=True)

    assert validated.warnings == []


def test_extra_warnings_are_kept_and_counted():
    result = monee.run_energy_flow(_two_bus_net())
    extra = ResultWarning("dof", "test entry")

    with pytest.raises(ValidationError) as excinfo:
        validate_result(result.network, result, strict=True, extra_warnings=[extra])

    assert extra in result.warnings
    assert extra in excinfo.value.warnings


def _solved_net(load_mw=0.5):
    return monee.run_energy_flow(_two_bus_net(load_mw=load_mw)).network


def test_served_delta_ignores_solver_noise_below_floor():
    network = _solved_net()
    network.childs[1].model.regulation = 1 - 1.556e-6

    assert _served_delta_entries(network, set(), {}) == []


def test_served_delta_fires_for_fixed_regulation_above_floor():
    network = _solved_net()
    network.childs[1].model.regulation = 0.9

    entries = _served_delta_entries(network, set(), {})

    assert len(entries) == 1
    assert entries[0].value == pytest.approx(0.05)


def test_served_delta_suppressed_for_controllable_regulation():
    network = _solved_net()
    network.childs[1].model.regulation = Var(0.0, min=0.0, max=1.0, name="regulation")

    assert _served_delta_entries(network, set(), {}) == []


def test_bound_active_suppressed_for_islanding_gated_var():
    load = PowerLoad(p_mw=0.15, q_mvar=0.0)
    load.p_mw = Var(0.15, min=0.0, max=0.15, name="islanding_gated_p_mw")
    load._islanding_gated_attrs = {"p_mw": 0.15}

    assert _model_bound_entries(load, "PowerLoad.0") == []


def test_bound_active_suppressed_for_setpoint_gate():
    load = PowerLoad(p_mw=0.15, q_mvar=0.0)
    load.p_mw = Var(0.15, min=0.0, max=0.15, name="p_mw")

    assert _model_bound_entries(load, "PowerLoad.1") == []


def test_bound_active_still_fires_for_grid_forming_child_at_cap():
    ext = ExtPowerGrid(p_mw=0, q_mvar=0, vm_pu=1.0, va_degree=0)
    ext.p_mw = Var(-2.0, min=-2.0, max=0.0, name="p_mw")

    entries = _model_bound_entries(ext, "ExtPowerGrid.0")

    assert len(entries) == 1
    assert "min bound -2" in entries[0].message


def test_bound_active_still_fires_for_two_nonzero_bounds():
    load = PowerLoad(p_mw=0.15, q_mvar=0.0)
    load.p_mw = Var(0.15, min=0.05, max=0.15, name="p_mw")

    entries = _model_bound_entries(load, "PowerLoad.2")

    assert len(entries) == 1


def test_carrier_balance_counts_linepack_as_storage():
    gas = mm.create_gas_grid("gas", type="lgas")
    pn = Network()
    g0 = pn.node(mm.Junction(), grid=gas, child_ids=[pn.child(mm.ExtHydrGrid())])
    g1 = pn.node(
        mm.Junction(),
        grid=gas,
        child_ids=[pn.child(mm.Sink(mass_flow_kgs=0.5))],
    )
    pn.branch(mm.GasPipe(diameter_m=0.15, length_m=400, temperature_ext_k=300), g0, g1)
    network = monee.run_energy_flow(pn).network
    assert _carrier_balance_entries(network, set()) == []

    network.branches[0].model.net_pack_kgs = Var(0.001, min=-1, max=1)
    network.childs[0].model.mass_flow_kgs.value -= 0.001

    assert _carrier_balance_entries(network, set()) == []

    network.childs[0].model.mass_flow_kgs.value -= 0.002
    entries = _carrier_balance_entries(network, set())
    assert len(entries) == 1
    assert entries[0].category == "energy_balance"


def test_he_duty_shortfall_has_absolute_noise_floor():
    tiny = SimpleNamespace(
        q_mw_delivered=Var(4.006e-6), q_mw=Var(6.049e-6), regulation=1, on_off=1
    )
    assert _he_duty_shortfall(tiny, 1e-6) is None

    real = SimpleNamespace(
        q_mw_delivered=Var(0.5), q_mw=Var(1.0), regulation=1, on_off=1
    )
    assert _he_duty_shortfall(real, 1e-6) == (0.5, 1.0)


def _fake_relaxed_branch(network, flow_kgs, gap=1.0):
    branch = network.branches[0]
    branch.formulation = SimpleNamespace(
        relaxation_gap=lambda m: gap, RELAXATION_GAP_TOL=0.1
    )
    branch.model.mass_flow_kgs.value = flow_kgs
    return branch


def _solved_gas_net(sink_kgs=0.5):
    gas = mm.create_gas_grid("gas", type="lgas")
    pn = Network()
    g0 = pn.node(mm.Junction(), grid=gas, child_ids=[pn.child(mm.ExtHydrGrid())])
    g1 = pn.node(
        mm.Junction(),
        grid=gas,
        child_ids=[pn.child(mm.Sink(mass_flow_kgs=sink_kgs))],
    )
    pn.branch(mm.GasPipe(diameter_m=0.15, length_m=400, temperature_ext_k=300), g0, g1)
    return monee.run_energy_flow(pn).network


def test_relaxation_entry_skips_negligible_flow():
    network = _solved_gas_net()
    _fake_relaxed_branch(network, 0.013)

    assert _relaxation_entry(network, "scip") is None


def test_relaxation_entry_never_recommends_the_running_solver():
    network = _solved_gas_net()
    _fake_relaxed_branch(network, 5.0)

    entry = _relaxation_entry(network, "scip")
    assert entry is not None
    assert "e.g. solver='scip'" not in entry.message
    assert "even under solver='scip'" in entry.message
    assert "gas_nlp" in entry.message

    entry_ipopt = _relaxation_entry(network, "ipopt")
    assert "e.g. solver='scip'" in entry_ipopt.message


def _misocp_branch(r_pu, ell, p_series_mw, q_series_mvar=0.0, w_from=1.0):
    from monee.model.formulation.miqcqp.convex.el import (
        MISOCPElectricityBranchFormulation,
    )

    branch = SimpleNamespace(
        id=(3, 6, 0),
        from_node_id=3,
        grid=SimpleNamespace(sn_mva=100.0),
        formulation=MISOCPElectricityBranchFormulation(),
        model=SimpleNamespace(
            br_r_pu=r_pu,
            tap=1.0,
            current_pu_squared=Var(ell),
            p_series_from_mw=Var(p_series_mw),
            q_series_from_mvar=Var(q_series_mvar),
        ),
    )
    node = SimpleNamespace(model=SimpleNamespace(vm_pu_squared=Var(w_from)))
    network = SimpleNamespace(branches=[branch], node_by_id=lambda _id: node)
    return branch, network


def test_misocp_cone_slack_is_uncertified_only_on_zero_resistance_branches():
    branch, network = _misocp_branch(r_pu=0.0, ell=3.5249, p_series_mw=28.8)
    slack = branch.formulation.uncertified_relaxation_slack(branch, network)
    assert slack == pytest.approx(0.9765, abs=1e-3)

    branch.model.br_r_pu = 0.017
    assert branch.formulation.uncertified_relaxation_slack(branch, network) == 0.0


def test_uncertified_relaxation_entry_fires_on_slack_zero_resistance_cone():
    _, network = _misocp_branch(r_pu=0.0, ell=3.5249, p_series_mw=28.8)

    entry = _uncertified_relaxation_entry(network)
    assert entry is not None
    assert entry.category == "relaxation"
    assert entry.component == "branch (3, 6, 0)"
    assert "limits/gap" in entry.message
    assert "concepts/formulations" in entry.message


def test_uncertified_relaxation_entry_silent_on_tight_cone():
    _, network = _misocp_branch(r_pu=0.0, ell=0.0829, p_series_mw=28.8)

    assert _uncertified_relaxation_entry(network) is None


def test_integrality_entry_softens_when_values_verify_integral():
    model = SimpleNamespace(direction=Var(1.0, integer=True))
    comp = SimpleNamespace(model=model, id=0)

    entry = _integrality_entry([comp], "ipopt")
    assert "physically meaningless" not in entry.message
    assert "self-consistent" in entry.message

    model.direction.value = 0.4
    entry = _integrality_entry([comp], "ipopt")
    assert "physically meaningless" in entry.message


def test_integrality_entry_solver_example_excludes_running_solver():
    model = SimpleNamespace(direction=Var(0.4, integer=True))
    comp = SimpleNamespace(model=model, id=0)

    assert "solver='gurobi'" in _integrality_entry([comp], "scip").message
    assert "solver='scip'" in _integrality_entry([comp], "ipopt").message


def test_inject_nans_covers_intermediates():
    junction = mm.Junction()
    inject_nans(junction)
    assert junction.mass_flow_kgs.value != junction.mass_flow_kgs.value

    pipe = mm.WaterPipe(diameter_m=0.1, length_m=100)
    inject_nans(pipe)
    assert pipe.mass_flow_kgs.value != pipe.mass_flow_kgs.value


def test_validation_summary_goes_through_warnings_module_and_dedups():
    reset_validation_summary_counts()
    result = monee.run_energy_flow(_two_bus_net())
    extra = ResultWarning("dof", "dedup test entry")

    with pytest.warns(ResultValidationSummary, match="Result validation"):
        result.warnings.clear()
        validate_result(result.network, result, extra_warnings=[extra])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for _ in range(2):
            result.warnings.clear()
            validate_result(result.network, result, extra_warnings=[extra])

    counts = validation_summary_counts()
    assert list(counts.values()) == [3]
    reset_validation_summary_counts()
    assert validation_summary_counts() == {}


def test_validation_summary_is_filterable():
    reset_validation_summary_counts()
    result = monee.run_energy_flow(_two_bus_net())
    extra = ResultWarning("dof", "filter test entry")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("ignore", ResultValidationSummary)
        result.warnings.clear()
        validate_result(result.network, result, extra_warnings=[extra])

    assert caught == []
    reset_validation_summary_counts()


def test_result_repr_and_str_are_cp1252_safe():
    result = monee.run_energy_flow(_two_bus_net())

    repr(result).encode("cp1252")
    str(result).encode("cp1252")

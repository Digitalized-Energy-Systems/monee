"""Temporal state lookups resolve an id shared by a node and a child to the
model of the requested type."""

import monee.express as mx
import monee.model as mm
from monee.problem.core import Constraints
from monee.solver.core import InterStepState, PeriodState, StepState, _find_model


def _colliding_gas_net():
    net = mm.Network(gas_model=mm.create_gas_grid("gas"))
    j0, j1 = mx.create_gas_junction(net), mx.create_gas_junction(net)
    mx.create_gas_pipe(net, j0, j1, diameter_m=0.15, length_m=100)
    mx.create_gas_ext_grid(net, j0)
    source_id = mx.create_gas_source(net, j1, mass_flow_kgs=0.1)
    assert source_id == j1 == 1
    return net


def test_find_model_legacy_resolution_unchanged():
    net = _colliding_gas_net()

    assert type(_find_model(net, 1, "mass_flow_kgs")) is mm.Junction


def test_find_model_type_scoped():
    net = _colliding_gas_net()

    assert type(_find_model(net, 1, "mass_flow_kgs", mm.Source)) is mm.Source
    assert type(_find_model(net, 1, "mass_flow_kgs", mm.Junction)) is mm.Junction


def test_find_model_never_substitutes_another_type():
    net = _colliding_gas_net()

    assert _find_model(net, 1, "pressure_pu", mm.Source) is None
    assert _find_model(net, 0, "mass_flow_kgs", mm.Source) is None


def test_step_and_period_state_pass_model_type():
    net = _colliding_gas_net()
    step_state = StepState()
    step_state.push(net)
    period_state = PeriodState([net], current_t=1)

    assert step_state.get(1, "mass_flow_kgs", model_type=mm.Source) == -0.1
    assert period_state.get(1, "mass_flow_kgs", model_type=mm.Source) == -0.1


class _RecordingState(InterStepState):
    def __init__(self):
        self.calls = []

    def get(self, component_id, attr, step=-1, model_type=None):
        self.calls.append((component_id, attr, model_type))


def test_regulation_ramp_passes_own_model_type():
    constraints = Constraints().regulation_ramp(0.1)
    state = _RecordingState()
    model = mm.PowerGenerator(1.0, 0.0)
    model.regulation = mm.Var(1, min=0, max=1)

    constraints._constraints[0]._temporal_equations[0](model, 7, state)

    assert state.calls == [(7, "regulation", mm.PowerGenerator)]

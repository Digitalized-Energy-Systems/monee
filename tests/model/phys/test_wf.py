from monee.model.phys.nonlinear.wf import darcy_friction


def test_darcy_friction():
    assert darcy_friction(0) == 64
    assert darcy_friction(1) == 32

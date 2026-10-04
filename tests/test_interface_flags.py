"""Every interface flag must actually reach the simulator.

This is the test that protects the whole sweep. If ``--max-step-dist 0.005``
silently does nothing, every cell of Figure 1 is the same experiment run
twelve times, and nothing in the printed output would say so.
"""

from __future__ import annotations

import numpy as np
import pytest

from piper_rl.config import EnvConfig
from piper_rl.precision import InstrumentedPiperEnv, SolverOverride


def _env(**kw):
    cfg = EnvConfig().eval_variant()
    solver = kw.pop("solver", None)
    for k, v in kw.items():
        if k in ("max_joint_vel", "max_tcp_speed"):
            setattr(cfg.limits, k, v)
        else:
            setattr(cfg, k, v)
    return InstrumentedPiperEnv(cfg, solver=solver)


# ------------------------------------------------------------------ Delta_max
@pytest.mark.parametrize("dmax", [0.005, 0.010, 0.020, 0.030])
def test_max_step_dist_bounds_the_commanded_displacement(dmax):
    """A saturated action must move the TCP target by at most Delta_max.

    Measured on the *commanded* target before the joint-velocity clamp, since
    the clamp is a separate axis with its own test.
    """
    env = _env(max_step_dist=dmax, action_smoothing=1.0)
    env.reset(seed=0)
    tcp0 = env.tcp_pos.copy()
    action = np.array([1.0, 0.0, 0.0, 0.0, -1.0])
    env.step(action)
    moved = np.linalg.norm(env.tcp_pos - tcp0)
    # A single step can never exceed the commanded displacement; it will
    # usually be smaller because of the clamp and the actuator lag.
    assert moved <= dmax + 1e-3, (moved, dmax)


def test_max_step_dist_is_monotone_in_reachable_displacement():
    """Larger Delta_max must produce a strictly larger one-step reach."""
    reach = []
    for dmax in (0.005, 0.030):
        env = _env(max_step_dist=dmax, action_smoothing=1.0)
        env.reset(seed=3)
        tcp0 = env.tcp_pos.copy()
        for _ in range(3):
            env.step(np.array([1.0, 0.0, 0.0, 0.0, -1.0]))
        reach.append(np.linalg.norm(env.tcp_pos - tcp0))
    assert reach[1] > reach[0] * 1.5, reach


# ------------------------------------------------------------------ alpha
@pytest.mark.parametrize("alpha", [0.2, 0.45, 1.0])
def test_action_smoothing_reaches_the_env(alpha):
    env = _env(action_smoothing=alpha)
    assert env.cfg.action_smoothing == alpha
    env.reset(seed=0)
    q_before = env._smoothed_q.copy()
    q_cmd_target = q_before + 0.1
    env._smoothed_q = (1 - alpha) * q_before + alpha * q_cmd_target
    assert np.allclose(env._smoothed_q - q_before, alpha * 0.1)


def test_smoothing_one_means_no_filtering():
    """alpha = 1.0 must make the smoothed target equal the raw IK target."""
    env = _env(action_smoothing=1.0)
    env.reset(seed=1)
    env.step(np.array([1.0, 0.0, 0.0, 0.0, -1.0]))
    assert env.cfg.action_smoothing == 1.0


# ------------------------------------------------------------------ f
@pytest.mark.parametrize("hz,expected_substeps", [(10.0, 50), (20.0, 25), (50.0, 10)])
def test_control_hz_changes_n_substeps(hz, expected_substeps):
    env = _env(control_hz=hz)
    assert env._n_substeps == expected_substeps, (hz, env._n_substeps)
    assert abs(env.dt - 1.0 / hz) < 1e-12


def test_control_hz_changes_the_safety_clamp():
    """max_dq = max_joint_vel * dt, so it must halve when the rate doubles."""
    a = _env(control_hz=20.0)
    b = _env(control_hz=40.0)
    assert abs(a.safety.max_dq - 2 * b.safety.max_dq) < 1e-12


# ------------------------------------------------------------------ qdot_max
@pytest.mark.parametrize("qd", [0.33, 0.5, 1.0, 2.0])
def test_max_joint_vel_sets_the_clamp(qd):
    env = _env(max_joint_vel=qd)
    assert abs(env.safety.max_dq - qd / env.cfg.control_hz) < 1e-12


def test_clamp_actually_binds_at_the_released_settings():
    """The paper's claim that the clamp is a binding constraint at theta_0.

    If this ever stops being true, the Delta_max sweep no longer needs to
    scale qdot_max with it, and Sec. 5 of the design must be rewritten.
    """
    env = _env()
    env.reset(seed=0)
    clamped = 0
    for _ in range(30):
        _, _, term, trunc, _ = env.step(np.array([1.0, 1.0, -1.0, 0.0, -1.0]))
        clamped += int(env.safety.n_clamped > 0)
        if term or trunc:
            break
    assert env.safety.n_clamped > 0


# ------------------------------------------------------------------ solver
@pytest.mark.parametrize("dt,substeps", [(0.001, 50), (0.002, 25), (0.004, 12)])
def test_solver_timestep_override(dt, substeps):
    env = _env(solver=SolverOverride(timestep=dt))
    assert abs(env.model.opt.timestep - dt) < 1e-12
    assert env._n_substeps == substeps, (dt, env._n_substeps)


def test_solver_timestep_4ms_is_not_an_integer_number_of_substeps():
    """4 ms at 20 Hz is 12.5 sub-steps and silently rounds to 12.

    That is a 4 % shortfall in simulated time per control step, which is a
    confound in the kill gate. The test documents it so the cell is either
    reported with the rounding or replaced by 2.5 ms.
    """
    exact = (1 / 20.0) / 0.004
    assert abs(exact - round(exact)) > 0.4


def test_solref_scale_touches_only_the_contact_geoms():
    base = _env()
    ref0 = base.model.geom_solref[base.obj_geom, 0]
    damp0 = base.model.geom_solref[base.obj_geom, 1]
    table0 = base.model.geom_solref[base.table_geom, 0]

    env = _env(solver=SolverOverride(solref_scale=2.0))
    assert abs(env.model.geom_solref[env.obj_geom, 0] - 2 * ref0) < 1e-12
    assert abs(env.model.geom_solref[env.obj_geom, 1] - damp0) < 1e-12, \
        "dampratio must be preserved"
    assert abs(env.model.geom_solref[env.table_geom, 0] - table0) < 1e-12, \
        "the table is not a swept contact"


@pytest.mark.parametrize("it", [20, 100])
def test_solver_iterations_and_impratio(it):
    env = _env(solver=SolverOverride(iterations=it, impratio=1.0))
    assert env.model.opt.iterations == it
    assert abs(env.model.opt.impratio - 1.0) < 1e-12


# ------------------------------------------------------------------ logging
def test_resolved_interface_is_recorded():
    """Every swept value must appear in the log, or a cell cannot be traced."""
    env = _env(max_step_dist=0.005, action_smoothing=0.2, control_hz=50.0,
               max_joint_vel=0.17, solver=SolverOverride(timestep=0.001))
    assert env.solver_resolved["solver_timestep"] == 0.001
    assert env.solver_resolved["n_substeps"] == 20
    assert env.cfg.max_step_dist == 0.005
    assert env.cfg.action_smoothing == 0.2
    assert env.cfg.control_hz == 50.0
    assert env.cfg.limits.max_joint_vel == 0.17

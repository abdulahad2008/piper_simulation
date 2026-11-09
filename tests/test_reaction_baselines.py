"""Regression tests for the scripted reactions (piper_rl/scripts/reaction_baselines.py).

    MUJOCO_GL=osmesa python -m pytest tests/test_reaction_baselines.py -q
"""
from __future__ import annotations

import numpy as np

from piper_rl.scripts import reaction_baselines as rb


def test_halted_hand_reports_zero_velocity():
    """9 Nov bug: a halted person kept its approach speed in the state, which
    inflated the ISO protective distance for the stop reaction."""
    env = rb.make_task_env("fixed_exact_h036_v1", "halts")
    env.reset(seed=31000)
    speeds_before, speeds_after = [], []
    for k in range(60):
        env.step(rb.Carry(env, 0.0, rb.TASK_CARRY_Z, rb.TASK_PLACE_Z, 0.5).action(k * env.dt)[0])
        v = float(np.linalg.norm(env.human_state.hand_velocity))
        (speeds_after if env.reacting else speeds_before).append(v)
    assert max(speeds_before) > 0.1          # the hand approached
    assert speeds_after and max(speeds_after) == 0.0


def test_pulling_back_hand_reverses_its_velocity():
    env = rb.make_task_env("fixed_exact_h036_v1", "pulls_back")
    env.reset(seed=31000)
    carry = rb.Carry(env, 0.0, rb.TASK_CARRY_Z, rb.TASK_PLACE_Z, 0.5)
    v_fwd = None
    for k in range(60):
        env.step(carry.action(k * env.dt)[0])
        v = env.human_state.hand_velocity.copy()
        if not env.reacting and np.linalg.norm(v) > 0.1:
            v_fwd = v
        elif env.reacting and v_fwd is not None and np.linalg.norm(v) > 0.05:
            assert float(np.dot(v, v_fwd)) < 0.0   # moving the other way
            return
    raise AssertionError("the person never pulled back")


def test_stop_holds_without_drifting():
    """9 Nov bug: the hold target was re-measured every step, so gravity sag
    ratcheted into a slow drift (about 3 cm over 2 s). A hold must keep the
    pan within the measured overshoot (S_s, 23 mm) and settle back."""
    env = rb.make_task_env("no_human", "keeps_going")
    env.reset(seed=0)
    carry = rb.Carry(env, 0.0, rb.TASK_CARRY_Z, rb.TASK_PLACE_Z, 0.5)
    for k in range(16):
        env.step(carry.action(k * env.dt)[0])
    env.hold_q = env.data.qpos[env.arm_qadr].copy()
    p0 = env.pan_centre().copy()
    hold = np.zeros(5); hold[4] = -1.0
    travel = []
    for _ in range(80):                     # 4 s
        env.step(hold)
        travel.append(np.linalg.norm(env.pan_centre() - p0))
    assert max(travel) <= rb.S_S + 0.002
    assert travel[-1] < 0.008


def test_run_episode_stop_keeps_one_hold_target():
    """run_episode must set the hold once at the trigger, not every step."""
    env = rb.make_task_env("fixed_exact_h036_v1", "keeps_going")
    targets = []
    orig = env.step

    def spy(act):
        if env.hold_q is not None:
            targets.append(env.hold_q.copy())
        return orig(act)
    env.step = spy
    rb.run_episode(env, "stop", 31000, task_env=True)
    assert len(targets) > 5
    assert all(np.array_equal(targets[0], t) for t in targets[:5])

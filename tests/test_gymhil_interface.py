"""The second environment must respond to the same flags as the first.

The point of adding gym-hil is that the interface sweep runs on an arm, a
controller and a task designed by other people. That only means anything if
the swept quantities actually reach it, so every assertion in
``test_interface_flags.py`` has a counterpart here.

These tests also pin gym-hil's released interface constants. If a future
version of the package changes them, the study's comparison of the two
environments silently changes with it, and this file is what catches that.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "osmesa")

gym_hil = pytest.importorskip("gym_hil")

from piper_rl.precision.gymhil_env import (  # noqa: E402
    ARRANGE, PICK, GymHilInterface, GymHilPrecisionEnv, characterise_interface)


# ------------------------------------------------------------ released values
def test_released_interface_constants_are_what_the_paper_states():
    c = characterise_interface()
    assert c["max_step_dist_mm"] == 25.0
    assert c["control_hz"] == 10.0
    assert c["physics_timestep_s"] == 0.002
    assert c["n_substeps"] == 50
    assert c["action_smoothing"] == 1.0          # no low-pass at all
    assert c["joint_velocity_clamp"] is None     # no safety clamp
    assert c["max_tcp_speed_m_s"] == pytest.approx(0.25)
    assert c["episode_budget_s"] == pytest.approx(10.0)
    assert c["tolerance_mm"]["PandaArrangeBoxes"] == 30.0


def test_tolerance_sits_just_above_the_step_size_in_both_environments():
    """The coincidence the interface hypothesis says is not a coincidence.

    gym-hil ArrangeBoxes: 30 mm tolerance on a 25 mm step  -> 1.2x
    gym-hil PickCube:     50 mm tolerance on a 25 mm step  -> 2.0x
    Piper place:          45 mm tolerance on a 30 mm step  -> 1.5x

    Three tasks, two environments, two authors, two arms: every success
    tolerance lands between one and two commanded step sizes. If the floor
    were set by anything other than the interface, there would be no reason
    for the people who wrote these environments to have converged on that.
    """
    from piper_rl.config import EnvConfig
    c = characterise_interface()
    dmax = c["max_step_dist_mm"]
    ratios = {
        "gymhil_arrange": c["tolerance_mm"]["PandaArrangeBoxes"] / dmax,
        "gymhil_pick": c["tolerance_mm"]["PandaPickCube_grasp_dist"] / dmax,
        "piper": (1000 * EnvConfig().reward.place_tol_xy)
                 / (1000 * EnvConfig().max_step_dist),
    }
    for name, r in ratios.items():
        assert 1.0 <= r <= 2.0, (name, r)


def test_arrange_boxes_is_not_learnable_from_state_as_released():
    """Documents why the sweep uses PickCube, not the placement task.

    Success needs five blocks within 30 mm of five targets; the state
    observation is three numbers, block 1's position. Four blocks and all five
    targets are unobservable. This is a defect in the released environment, and
    if a future gym-hil release fixes it this test will fail and the study
    should switch tasks.
    """
    env = GymHilPrecisionEnv(ARRANGE)
    obs, _ = env.reset(seed=0)
    blocks, targets = env._block_target_pairs()
    env.close()
    assert len(blocks) == 5
    assert np.asarray(obs["environment_state"]).size == 3, \
        "gym-hil now exposes more of the scene; re-check task choice"


# ------------------------------------------------------------------ Delta_max
@pytest.mark.parametrize("dmax", [0.005, 0.010, 0.025, 0.050])
def test_max_step_dist_bounds_the_realised_displacement(dmax):
    env = GymHilPrecisionEnv(PICK, GymHilInterface(max_step_dist=dmax))
    env.reset(seed=3)
    t0 = env.tcp_xy().copy()
    env.step(np.array([1.0, 0.0, 0.0, 0, 0, 0, -1.0]))
    moved = float(np.linalg.norm(env.tcp_xy() - t0))
    env.close()
    assert moved <= dmax + 1e-4, (moved, dmax)


def test_realised_step_is_a_fixed_fraction_of_the_commanded_cap():
    """Operational-space control does not reach the mocap target in one step.

    The realised displacement is about 40 % of the commanded cap, so gym-hil's
    nominal 25 mm step is an effective ~10 mm step. Any comparison of the two
    environments' step sizes has to use the realised number, and this test
    pins the ratio so the paper's figure cannot drift from it.
    """
    ratios = []
    for dmax in (0.005, 0.010, 0.025, 0.050):
        env = GymHilPrecisionEnv(PICK, GymHilInterface(max_step_dist=dmax))
        env.reset(seed=3)
        t0 = env.tcp_xy().copy()
        env.step(np.array([1.0, 0.0, 0.0, 0, 0, 0, -1.0]))
        ratios.append(float(np.linalg.norm(env.tcp_xy() - t0)) / dmax)
        env.close()
    assert max(ratios) - min(ratios) < 0.05, ratios      # linear in Delta_max
    assert 0.3 < float(np.mean(ratios)) < 0.5, ratios


# ------------------------------------------------------------------ alpha, f
@pytest.mark.parametrize("alpha", [0.2, 0.45, 1.0])
def test_smoothing_reaches_the_env_and_1_reproduces_the_default(alpha):
    env = GymHilPrecisionEnv(PICK, GymHilInterface(action_smoothing=alpha))
    assert env.iface.action_smoothing == alpha
    env.reset(seed=0)
    env.step(np.array([1.0, 0.0, 0.0, 0, 0, 0, -1.0]))
    # after one step the smoothed command is alpha * the raw command
    assert env._smoothed[0] == pytest.approx(alpha * env.iface.max_step_dist)
    env.close()


@pytest.mark.parametrize("hz,substeps", [(5.0, 100), (10.0, 50), (25.0, 20)])
def test_control_rate_changes_substeps_and_holds_the_budget_in_seconds(hz, substeps):
    env = GymHilPrecisionEnv(PICK, GymHilInterface(control_hz=hz))
    assert env._n_substeps == substeps, (hz, env._n_substeps)
    assert env.max_episode_steps == int(round(10.0 * hz))
    env.close()


# ------------------------------------------------------------------ contract
def test_episode_summary_matches_the_piper_contract():
    """The two environments must agree on what a results row means."""
    from piper_rl.precision.instrumented_env import EPISODE_FIELDS
    env = GymHilPrecisionEnv(PICK)
    env.reset(seed=0)
    for _ in range(3):
        env.step(np.zeros(7))
    s = env.episode_summary()
    env.close()
    for key in ("success", "grasp_success", "lift_success", "placement_error",
                "tcp_err_at_release", "scatter", "release_step", "obj_settled",
                "length", "collision_steps", "safety_clamped", "safety_vetoed",
                "n_substeps", "dr_vector", "solver"):
        assert key in s, key
    assert isinstance(s["success"], bool)
    assert np.isfinite(s["placement_error"])
    assert "place_err_mm" in EPISODE_FIELDS


def test_worst_block_drives_the_error_not_the_mean():
    """Success is a conjunction over blocks, so the curve must use the worst."""
    env = GymHilPrecisionEnv(ARRANGE)
    env.reset(seed=1)
    blocks, targets = env._block_target_pairs()
    dists = [float(np.linalg.norm(b - t)) for b, t in zip(blocks, targets)]
    assert env.terminal_error_m() == pytest.approx(max(dists))
    assert len(dists) > 1, "the arrangement task should have several blocks"
    env.close()

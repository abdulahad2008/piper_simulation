"""Eval-only human-cue overrides and the unchanged default observation path."""
import copy

import numpy as np
import pytest

from piper_rl.human_aware_env import PiperHumanAwarePickPlaceEnv
from piper_rl.human_config import HumanAwareEnvConfig
from piper_rl.scripts.evaluate_human_aware import _make_env


def assert_identical(left, right):
    if isinstance(left, np.ndarray):
        assert left.dtype == right.dtype
        assert left.shape == right.shape
        assert left.tobytes() == right.tobytes()
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_identical(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_identical(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("preset", ["no_human", "fixed_exact_h036_v1"])
@pytest.mark.parametrize("seed", [7, 30000, 31000])
def test_default_actual_observations_and_episode_outcomes_byte_identical(preset, seed):
    cfg = HumanAwareEnvConfig.from_preset(preset).eval_variant()
    cfg.include_human_state = True
    cfg.max_episode_steps = 25
    original = PiperHumanAwarePickPlaceEnv(copy.deepcopy(cfg))
    default = _make_env(copy.deepcopy(cfg))
    assert type(default) is PiperHumanAwarePickPlaceEnv
    try:
        obs, info = original.reset(seed=seed)
        actual_obs, actual_info = default.reset(seed=seed)
        assert_identical(obs, actual_obs)
        assert_identical(info, actual_info)
        for _ in range(cfg.max_episode_steps):
            # Observation-dependent, deterministic actions exercise both paths.
            action = np.tanh(obs[:5]).astype(np.float32)
            expected = original.step(action)
            actual = default.step(action)
            assert_identical(expected, actual)
            obs = expected[0]
            if expected[2] or expected[3]:
                assert_identical(expected[4]["episode_summary"],
                                 actual[4]["episode_summary"])
                break
        else:
            pytest.fail("episode did not finish")
    finally:
        original.close()
        default.close()


@pytest.mark.parametrize("source,preset", [
    ("phantom-exact-h036", "no_human"),
    ("frozen-parked", "fixed_exact_h036_v1"),
])
def test_probe_changes_only_human_observations_and_preserves_physics_rng(source, preset):
    cfg = HumanAwareEnvConfig.from_preset(preset).eval_variant()
    cfg.include_human_state = True
    cfg.max_episode_steps = 45
    # Nonzero noise and latency exercise the inherited human encoder too.
    cfg.human_position_noise = .002
    cfg.human_velocity_noise = .003
    cfg.human_observation_latency_steps = 2
    original = _make_env(copy.deepcopy(cfg))
    probe = _make_env(copy.deepcopy(cfg), source)
    try:
        a, _ = original.reset(seed=31000)
        b, _ = probe.reset(seed=31000)
        frozen = b[-13:].copy()
        assert b[-13] == 0
        for _ in range(45):
            assert_identical(a[:51], b[:51])
            assert_identical(original.data.qpos, probe.data.qpos)
            assert_identical(original.data.mocap_pos, probe.data.mocap_pos)
            assert_identical(original.np_random.bit_generator.state,
                             probe.np_random.bit_generator.state)
            if source == "frozen-parked":
                assert_identical(b[-13:], frozen)
            action = np.zeros(5, dtype=np.float32)
            a, reward_a, term_a, trunc_a, info_a = original.step(action)
            b, reward_b, term_b, trunc_b, info_b = probe.step(action)
            assert_identical((reward_a, term_a, trunc_a, info_a),
                             (reward_b, term_b, trunc_b, info_b))
            if term_a or trunc_a:
                break
        if source == "phantom-exact-h036":
            assert not probe._human_episode_active
            assert not info_b["human_collision"]
        else:
            assert probe.human_state.human_present
    finally:
        original.close()
        probe.close()


def test_phantom_uses_current_time_real_tcp_and_reproducible_exact_trajectory():
    cfg = HumanAwareEnvConfig.from_preset("no_human").eval_variant()
    cfg.include_human_state = True
    cfg.noise.enabled = False
    env = _make_env(cfg, "phantom-exact-h036")
    try:
        initial, _ = env.reset(seed=30000)
        waypoints = env._probe_trajectory.waypoints.copy()
        for _ in range(25):
            obs, _, _, _, _ = env.step(np.zeros(5))
            state = env._probe_trajectory.state(env.data.time - env._human_time_zero)
            expected = np.concatenate(([float(state.human_present)],
                                       state.hand_position, state.hand_velocity,
                                       state.elbow_position,
                                       state.hand_position - env.tcp_pos)).astype(np.float32)
            assert_identical(obs[-13:], expected)
        assert obs[-13] == 1
        again, _ = env.reset(seed=30000)
        assert_identical(again, initial)
        assert_identical(env._probe_trajectory.waypoints, waypoints)
        assert np.array_equal(waypoints[:, 2], [.36, .34, .36])
    finally:
        env.close()

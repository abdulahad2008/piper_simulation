import numpy as np
import pytest

from piper_rl.human_config import (FIXED_EXACT_H036_V1, HumanAwareEnvConfig,
                                   HumanMotionConfig, TRAJECTORY_TYPES)
from piper_rl.human_motion import (HumanTrajectoryGenerator,
                                   trajectory_fingerprint,
                                   trajectory_fingerprint_sha256)


OBJ = np.array([0.35, -0.12, 0.235])
TARGET = np.array([0.35, 0.14, 0.235])


@pytest.mark.parametrize("kind", TRAJECTORY_TYPES)
def test_all_trajectory_types_are_continuous_and_speed_limited(kind):
    cfg = HumanMotionConfig(
        appearance_time_range=(0.2, 0.2), speed_range=(0.25, 0.25),
        pause_probability=0.0, reverse_probability=0.0)
    trajectory = HumanTrajectoryGenerator(cfg).sample(
        np.random.default_rng(5), OBJ, TARGET, trajectory_type=kind)
    dt = 0.002
    times = np.arange(0, trajectory.end_time + dt, dt)
    states = [trajectory.state(t) for t in times]
    jumps = [np.linalg.norm(b.hand_position - a.hand_position)
             for a, b in zip(states[:-1], states[1:])]
    speeds = [np.linalg.norm(state.hand_velocity) for state in states]
    assert max(jumps, default=0.0) <= cfg.max_hand_speed * dt * 1.02
    assert max(speeds) <= cfg.max_hand_speed + 1e-7
    assert trajectory.state(0).phase == "parked"
    assert not trajectory.state(trajectory.end_time + 1).human_present


def test_sampling_is_seed_reproducible_and_uses_no_global_random_state():
    generator = HumanTrajectoryGenerator(HumanMotionConfig())
    np.random.seed(999)
    a = generator.sample(np.random.default_rng(42), OBJ, TARGET)
    np.random.seed(1)
    b = generator.sample(np.random.default_rng(42), OBJ, TARGET)
    assert a.trajectory_type == b.trajectory_type
    assert a.appearance_time == b.appearance_time
    assert np.array_equal(a.waypoints, b.waypoints)
    assert np.array_equal(a.durations, b.durations)


def test_different_seeds_vary_trajectory():
    generator = HumanTrajectoryGenerator(HumanMotionConfig())
    a = generator.sample(np.random.default_rng(10), OBJ, TARGET)
    b = generator.sample(np.random.default_rng(11), OBJ, TARGET)
    assert (a.trajectory_type != b.trajectory_type
            or a.appearance_time != b.appearance_time
            or not np.array_equal(a.waypoints, b.waypoints))


def test_exact_fixed_fingerprint_is_seed_invariant():
    cfg = HumanAwareEnvConfig.from_preset(FIXED_EXACT_H036_V1).human_motion
    generator = HumanTrajectoryGenerator(cfg)
    a = generator.sample(np.random.default_rng(3), OBJ, TARGET)
    b = generator.sample(np.random.default_rng(99), OBJ + 0.1, TARGET - 0.1)
    fingerprint_a = trajectory_fingerprint(a, cfg)
    fingerprint_b = trajectory_fingerprint(b, cfg)
    assert fingerprint_a == fingerprint_b
    assert trajectory_fingerprint_sha256(fingerprint_a) == trajectory_fingerprint_sha256(fingerprint_b)

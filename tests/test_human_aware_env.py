import mujoco
import numpy as np
import pytest

from piper_rl.human_aware_env import (HUMAN_OBSERVATION_DIM,
                                      PiperHumanAwarePickPlaceEnv)
from piper_rl.human_config import HUMAN_MODEL_PATH, HumanAwareEnvConfig
from piper_rl.human_motion import HumanArmState


def config(preset="fixed", include_state=True):
    cfg = HumanAwareEnvConfig.from_preset(preset)
    cfg.curriculum = False
    cfg.domain_rand.enabled = False
    cfg.noise.enabled = False
    cfg.include_human_state = include_state
    return cfg


def test_xml_loads_and_named_human_geometries_exist():
    model = mujoco.MjModel.from_xml_path(str(HUMAN_MODEL_PATH))
    assert model.nmocap == 3
    for name in ("human_upper_arm_geom", "human_forearm_geom", "human_hand_geom"):
        assert model.geom(name).id >= 0


def test_reset_step_and_human_observation_shape():
    env = PiperHumanAwarePickPlaceEnv(config())
    obs, info = env.reset(seed=7)
    assert obs.shape == (51 + HUMAN_OBSERVATION_DIM,)
    assert info["human_trajectory_type"] == "cross_workspace"
    obs, reward, terminated, truncated, info = env.step(np.zeros(5))
    assert obs.shape == env.observation_space.shape
    assert np.isfinite(obs).all() and np.isfinite(reward)
    assert "human_robot_distance" in info
    env.close()


def test_human_unaware_shape_exactly_matches_original():
    env = PiperHumanAwarePickPlaceEnv(config(include_state=False))
    obs, _ = env.reset(seed=2)
    assert obs.shape == (51,)
    assert env.observation_space.shape == (51,)
    env.close()


def test_seed_reproducibility_and_variation():
    env = PiperHumanAwarePickPlaceEnv(config("randomized"))
    env.reset(seed=123)
    a = np.array([env.human_trajectory.state(t).hand_position
                  for t in np.linspace(0, 5, 30)])
    type_a = env.human_trajectory.trajectory_type
    env.reset(seed=123)
    b = np.array([env.human_trajectory.state(t).hand_position
                  for t in np.linspace(0, 5, 30)])
    assert type_a == env.human_trajectory.trajectory_type
    assert np.array_equal(a, b)
    env.reset(seed=124)
    c = np.array([env.human_trajectory.state(t).hand_position
                  for t in np.linspace(0, 5, 30)])
    assert not np.array_equal(a, c)
    env.close()


def test_initial_pose_is_collision_free_and_distance_finite():
    env = PiperHumanAwarePickPlaceEnv(config("randomized"))
    for seed in range(10):
        env.reset(seed=seed)
        assert not env._robot_human_contacts()[0]
        assert np.isfinite(env._surface_distance())
        assert env._surface_distance() > 0
    env.close()


class _ForcedContactTrajectory:
    trajectory_type = "forced_contact"
    start_side = 1
    appearance_time = 0.0
    speed = 0.0

    def __init__(self, tcp):
        self.tcp = tcp

    def state(self, _time):
        return HumanArmState(
            self.tcp + [0, 0, 0.08], self.tcp + [0, 0, 0.03], self.tcp,
            np.zeros(3), np.zeros(3), True, "forced_contact")


def test_forced_human_contact_detection_distance_and_termination():
    env = PiperHumanAwarePickPlaceEnv(config())
    env.reset(seed=0)
    env.human_trajectory = _ForcedContactTrajectory(env.tcp_pos.copy())
    env._human_episode_active = True
    _, reward, terminated, truncated, info = env.step(np.zeros(5))
    assert terminated and not truncated
    assert info["termination"] == "human_collision"
    assert info["human_collision"]
    assert info["human_robot_distance"] <= 0
    assert reward <= -env.human_cfg.human_safety.human_collision_penalty
    env.close()


def test_no_human_preset_has_no_safety_events():
    env = PiperHumanAwarePickPlaceEnv(config("no_human"))
    env.reset(seed=1)
    for _ in range(20):
        _, reward, terminated, truncated, info = env.step(np.zeros(5))
        assert np.isfinite(reward)
        if terminated or truncated:
            break
    assert not info["human_present"]
    assert not info["human_collision"]
    assert info["cumulative_human_safety_cost"] == 0
    env.close()


def test_state_plus_rgb_extends_only_state_and_renders_headless():
    cfg = config()
    cfg.obs_mode = "state+rgb"
    cfg.camera_width = cfg.camera_height = 48
    env = PiperHumanAwarePickPlaceEnv(cfg)
    obs, _ = env.reset(seed=4)
    assert obs["state"].shape == (64,)
    assert obs["rgb"].shape == (48, 48, 3)
    frame = env.render("overview")
    assert frame.shape == (cfg.render_height, cfg.render_width, 3)
    assert frame.max() > frame.min()
    env.close()


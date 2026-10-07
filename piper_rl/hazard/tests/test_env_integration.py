"""Smoke tests: BladeHazardWrapper on the real PiperHumanAwarePickPlace-v0 env with the knife scene."""
try:  # same import-order guard as tests/conftest.py (OSMesa vs torch/triton LLVM)
    import torch  # noqa: F401
    import triton  # noqa: F401
except Exception:
    pass

import gymnasium as gym
import numpy as np
import pytest

import piper_rl  # noqa: F401  (registers PiperPickPlace-v0)
import piper_rl.human_aware_env  # noqa: F401  (registers PiperHumanAwarePickPlace-v0)
from piper_rl.hazard.scene import knife_human_config
from piper_rl.hazard.wrapper import BladeHazardWrapper

ENV_ID = "PiperHumanAwarePickPlace-v0"


def _make(no_human: bool):
    cfg = knife_human_config()
    cfg.curriculum = False
    cfg.domain_rand.enabled = False
    cfg.noise.enabled = False
    if no_human:
        cfg.human_enabled = False
        cfg.human_episode_probability = 0.0
    return BladeHazardWrapper(gym.make(ENV_ID, cfg=cfg), dt=0.05)


def test_random_steps_report_finite_hazard_and_cost():
    env = _make(no_human=False)
    env.reset(seed=0)
    env.action_space.seed(0)
    for _ in range(50):
        _, _, terminated, truncated, info = env.step(env.action_space.sample())
        assert "hazard" in info and "cost" in info
        assert np.isfinite(info["hazard"]) and np.isfinite(info["cost"])
        assert np.isfinite(info["edge_distance"])
        if terminated or truncated:
            assert "hazard_summary" in info
            env.reset()
    env.close()


def test_no_human_episode_has_zero_exposure():
    env = _make(no_human=True)
    _, reset_info = env.reset(seed=1)
    assert reset_info["human_episode"] is False
    env.action_space.seed(1)
    info = {}
    for _ in range(400):  # episode limit is 200 steps
        _, _, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated or truncated:
            break
    assert "hazard_summary" in info, "episode did not end within the step budget"
    assert info["hazard_summary"]["exposure"] == 0
    assert info["hazard_summary"]["cut"] is False
    env.close()

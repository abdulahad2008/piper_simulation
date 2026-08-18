import numpy as np

from piper_rl import EnvConfig, PiperPickPlaceEnv
from piper_rl.human_config import HumanAwareEnvConfig
from piper_rl.scripts.train import build_vec_env, default_hyperparams


def _base_config():
    cfg = EnvConfig()
    cfg.curriculum = False
    cfg.domain_rand.enabled = False
    cfg.noise.enabled = False
    return cfg


def test_original_environment_shape_and_seeded_rollout_regression():
    env = PiperPickPlaceEnv(_base_config())
    assert env.observation_space.shape == (51,)
    assert env.action_space.shape == (5,)
    actions = np.random.default_rng(8).uniform(-1, 1, (20, 5))

    def run():
        obs, _ = env.reset(seed=55)
        result = [obs.copy()]
        for action in actions:
            obs, reward, terminated, truncated, _ = env.step(action)
            result.append(np.r_[obs, reward])
            if terminated or truncated:
                break
        return result

    first, second = run(), run()
    assert len(first) == len(second)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    env.close()


def test_original_and_human_training_factories_start():
    original = build_vec_env(_base_config(), 1, seed=1, subproc=False)
    assert original.observation_space.shape == (51,)
    original.reset()
    original.step([original.action_space.sample()])
    original.close()

    human_cfg = HumanAwareEnvConfig.from_preset("fixed")
    human_cfg.domain_rand.enabled = False
    human_cfg.noise.enabled = False
    human = build_vec_env(human_cfg, 1, seed=1, subproc=False)
    assert human.observation_space.shape == (64,)
    human.reset()
    human.step([human.action_space.sample()])
    human.close()


def test_sac_learning_starts_default_and_smoke_override():
    assert default_hyperparams("sac", 4)["learning_starts"] == 5_000
    assert default_hyperparams("sac", 4, learning_starts=100)["learning_starts"] == 100

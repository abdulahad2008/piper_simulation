"""Tests for the hot-pan carry task (piper_rl/pan_env.py).

    MUJOCO_GL=osmesa python -m pytest tests/test_pan_env.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

from piper_rl.human_config import HumanAwareEnvConfig
from piper_rl.human_aware_env import PiperHumanAwarePickPlaceEnv
from piper_rl.pan_env import (CARRY_CLEARANCE, PAN_START_PHASES, PLACE_CLEARANCE,
                              PanEnvConfig, PiperPanPlaceEnv)
from piper_rl.piper_env import TABLE_TOP
from piper_rl.scripts.make_pan_scene import HEIGHT as PAN_HEIGHT, STOVE_XY, TRIVET_TOP, TRIVET_XY

CARRY_Z = TABLE_TOP + CARRY_CLEARANCE + PAN_HEIGHT
PLACE_Z = TRIVET_TOP + PAN_HEIGHT + PLACE_CLEARANCE


def make_env(preset="no_human", curriculum=False, human_collision_terminates=True):
    cfg = PanEnvConfig.from_preset(preset)
    if not curriculum:
        cfg = cfg.eval_variant()
    cfg.domain_rand.enabled = False
    cfg.noise.enabled = False
    cfg.human_safety.terminate_on_human_collision = human_collision_terminates
    return PiperPanPlaceEnv(cfg)


def carry_action(env, k, lift_steps=10):
    tcp, pan = env.tcp_pos, env.pan_centre()
    if k < lift_steps:
        goal_xy, goal_z = np.array(STOVE_XY), CARRY_Z
    else:
        goal_xy = np.array(TRIVET_XY)
        goal_z = CARRY_Z if np.linalg.norm(pan[:2] - goal_xy) > 0.02 else PLACE_Z
    delta = np.array([goal_xy[0] - pan[0], goal_xy[1] - pan[1], goal_z - tcp[2]])
    a = np.zeros(5)
    a[:3] = np.clip(delta / env.cfg.max_step_dist * 0.5, -1.0, 1.0)
    a[4] = -1.0
    return a


def rollout(env, policy, seed, max_steps=200):
    obs, info = env.reset(seed=seed)
    total = 0.0
    for k in range(max_steps):
        obs, r, te, tr, info = env.step(policy(env, k))
        total += r
        if te or tr:
            break
    return total, info


class TestInterface:
    def test_observation_layout_matches_the_cup_task(self):
        pan = make_env()
        cup = PiperHumanAwarePickPlaceEnv(HumanAwareEnvConfig.from_preset("no_human").eval_variant())
        assert pan.observation_space.shape == cup.observation_space.shape == (64,)
        assert pan.action_space.shape == cup.action_space.shape == (5,)
        obs, _ = pan.reset(seed=0)
        assert np.isfinite(obs).all()

    def test_config_points_at_the_pan_scene_with_the_pan_clearance(self):
        cfg = PanEnvConfig.from_preset("curriculum")
        assert cfg.model_path.endswith("piper_human_pan_task.xml")
        assert cfg.limits.tcp_table_clearance == pytest.approx(0.065)
        assert cfg.human_distribution == "curriculum_id"

    def test_start_phases_place_the_pan_where_they_say(self):
        env = make_env()
        for phase in PAN_START_PHASES:
            env.reset(seed=1, options={"pan_phase": phase})
            pan = env.pan_centre()
            if phase == "stove":
                assert np.linalg.norm(pan[:2] - STOVE_XY) < 0.01
                assert 0.005 < pan[2] - TRIVET_TOP < 0.02      # base just above the hob
            elif phase == "stove_lifted":
                assert np.linalg.norm(pan[:2] - STOVE_XY) < 0.01
                assert pan[2] - TABLE_TOP > CARRY_CLEARANCE - 0.005
            elif phase == "over_trivet":
                assert np.linalg.norm(pan[:2] - TRIVET_XY) < 0.03
            else:
                assert STOVE_XY[1] < pan[1] < TRIVET_XY[1]

    def test_eval_variant_always_starts_on_the_stove(self):
        env = make_env(preset="curriculum", curriculum=False)
        phases = {env.reset(seed=s)[1]["pan_phase"] for s in range(8)}
        assert phases == {"stove"}

    def test_curriculum_samples_every_phase(self):
        env = make_env(preset="curriculum", curriculum=True)
        phases = {env.reset(seed=s)[1]["pan_phase"] for s in range(60)}
        assert phases == set(PAN_START_PHASES)


class TestTask:
    def test_scripted_carry_succeeds_without_collisions(self):
        env = make_env()
        for seed in range(3):
            total, info = rollout(env, carry_action, seed)
            s = info["episode_summary"]
            assert info["termination"] == "success", s
            assert s["placement_error"] < 0.01
            assert s["collision_steps"] == 0
            assert s["hazard_free_success"]
            assert total > 150

    def test_holding_still_on_the_stove_never_succeeds(self):
        env = make_env()
        total, info = rollout(env, lambda env, k: np.array([0, 0, 0, 0, -1.0]), 0)
        assert info["termination"] == "timeout"
        assert not info["is_success"]
        assert total < 0

    def test_shaping_is_a_potential_and_cannot_be_farmed(self):
        """Sum of the shaping term over any trajectory equals phi(end) - phi(start)."""
        env = make_env()
        obs, _ = env.reset(seed=5)
        phi0 = env._potential()
        rng = np.random.default_rng(5)
        for _ in range(60):
            _, _, te, tr, info = env.step(rng.uniform(-1, 1, 5))
            if te or tr:
                break
        assert env._rew_terms["shaping"] == pytest.approx(env._potential() - phi0, abs=1e-6)

    def test_safety_layer_keeps_the_pan_off_the_hob(self):
        env = make_env()
        env.reset(seed=0)
        for _ in range(40):
            env.step(np.array([0, 0, -1.0, 0, -1.0]))      # push straight down
        assert env.pan_centre()[2] > TRIVET_TOP - 0.002


class TestPerson:
    def test_pan_presets_start_the_person_earlier_than_the_cup_presets(self):
        from piper_rl.pan_env import PAN_APPEARANCE_SHIFT_S
        for preset in ("evaluation_id", "fixed_exact_h036_v1", "evaluation_ood"):
            cup = HumanAwareEnvConfig.from_preset(preset).human_motion.appearance_time_range
            pan = PanEnvConfig.from_preset(preset).human_motion.appearance_time_range
            assert pan == pytest.approx((cup[0] + PAN_APPEARANCE_SHIFT_S, cup[1] + PAN_APPEARANCE_SHIFT_S))
        # the curriculum callback recomputes the window: the shift must survive it
        cfg = PanEnvConfig.from_preset("curriculum", 0.3)
        cfg.set_human_difficulty(1.0)
        full = HumanAwareEnvConfig.from_preset("curriculum", 1.0).human_motion.appearance_time_range
        assert cfg.human_motion.appearance_time_range == pytest.approx(
            (full[0] + PAN_APPEARANCE_SHIFT_S, full[1] + PAN_APPEARANCE_SHIFT_S))

    def test_direct_carry_meets_the_exact_crossing(self):
        """The point of the shift: on the exact crossing a direct carry now
        passes through the hand (hot contact) instead of finishing first."""
        env = make_env(preset="fixed_exact_h036_v1", human_collision_terminates=False)
        for seed in range(3):
            _, info = rollout(env, lambda e, k: carry_action(e, k, lift_steps=0), seed)
            assert info["episode_summary"]["hot_contact_steps"] > 0

    def test_person_pan_contact_is_classed_not_a_robot_collision(self):
        """On the exact crossing, carrying through the person touches the pan's
        hot surface; that counts as hot contact and exposure, not as a
        robot-body collision (the pan is not a robot link)."""
        env = make_env(preset="fixed_exact_h036_v1", human_collision_terminates=False)
        hot = 0
        for seed in range(4):
            total, info = rollout(env, lambda e, k: carry_action(e, k, lift_steps=0), seed)
            s = info["episode_summary"]
            hot += s["hot_contact_steps"]
            if s["hot_contact_steps"]:
                assert s["exposure_steps"] > 0
                assert s["minimum_hot_distance"] <= 0.0
                assert not s["hazard_free_success"]
        assert hot > 0

    def test_hazard_cost_is_a_per_step_flag(self):
        env = make_env(preset="fixed_exact_h036_v1", human_collision_terminates=False)
        env.reset(seed=0)
        seen = set()
        for k in range(120):
            _, _, te, tr, info = env.step(carry_action(env, k, lift_steps=0))
            seen.add(info["hazard_cost"])
            if te or tr:
                break
        assert seen == {0.0, 1.0}


class TestDeterminism:
    def test_same_seed_same_actions_same_observations(self):
        a, b = make_env(preset="curriculum", curriculum=True), make_env(preset="curriculum", curriculum=True)
        oa, _ = a.reset(seed=11)
        ob, _ = b.reset(seed=11)
        assert np.array_equal(oa, ob)
        rng = np.random.default_rng(11)
        for _ in range(50):
            act = rng.uniform(-1, 1, 5)
            oa, ra, tea, tra, _ = a.step(act)
            ob, rb, teb, trb, _ = b.step(act)
            assert np.array_equal(oa, ob) and ra == rb and (tea, tra) == (teb, trb)
            if tea or tra:
                break


class TestHazardBudget:
    def test_lambda_charges_the_exposed_steps_only(self):
        env = make_env(preset="fixed_exact_h036_v1", human_collision_terminates=False)
        env.set_hazard_lambda(10.0)
        _, info = rollout(env, lambda e, k: carry_action(e, k, lift_steps=0), 0)
        s = info["episode_summary"]
        assert s["exposure_steps"] > 0
        assert s["reward_terms"]["hazard"] == pytest.approx(-10.0 * s["exposure_steps"])
        env.set_hazard_lambda(0.0)
        _, info = rollout(env, lambda e, k: carry_action(e, k, lift_steps=0), 0)
        assert info["episode_summary"]["reward_terms"].get("hazard", 0.0) == 0.0

    def test_callback_moves_lambda_towards_the_budget(self):
        from piper_rl.callbacks import HazardBudgetCallback

        class FakeVecEnv:
            def __init__(self):
                self.lam = None

            def env_method(self, name, *a):
                assert name == "set_hazard_lambda"
                self.lam = a[0]

        cb = HazardBudgetCallback(budget_s=0.1, lr=2.0, every=4)

        class L:
            def record(self, *a, **k): pass

        class M:                      # stands in for the model the callback is attached to
            num_timesteps = 0
            logger = L()
        cb.model = M()
        cb.training_env_fake = FakeVecEnv()
        type(cb).training_env = property(lambda self: self.training_env_fake)

        class L:
            def record(self, *a, **k): pass
        cb._logger = L()
        cb._on_training_start()
        assert cb.training_env_fake.lam == 0.0
        cb.locals = {"infos": [{"episode_summary": {"exposure_time": 0.5}}] * 4}
        cb._on_step()
        assert cb.lam == pytest.approx(2.0 * (0.5 - 0.1))        # rose
        assert cb.training_env_fake.lam == cb.lam
        cb.locals = {"infos": [{"episode_summary": {"exposure_time": 0.0}}] * 4}
        cb._on_step()
        assert cb.lam == pytest.approx(0.8 - 0.2)                 # fell
        for _ in range(10):
            cb._on_step()
        assert cb.lam == 0.0                                       # clipped at zero

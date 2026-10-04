"""Evaluate existing or human-aware policies under controlled HRI distributions."""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO, SAC, TD3

from piper_rl.human_aware_env import PiperHumanAwarePickPlaceEnv
from piper_rl.human_config import HumanAwareEnvConfig
from piper_rl.human_motion import HumanTrajectoryGenerator


EXPERIMENTS = {
    "no-human": ("no_human", False, "no_human"),
    "human-unaware": ("randomized", False, "evaluation_id_unaware"),
    # Historical fixed-ID evaluations randomized hand height. Keep their
    # reports distinct from the new exact 0.36 m fixed crossing.
    "fixed": ("fixed", True, "constrained_fixed_height_randomized"),
    "fixed-exact-h036-v1": (
        "fixed_exact_h036_v1", True, "fixed_exact_h036_v1"),
    "fixed-exact-shifted-h036-v1": (
        "fixed_exact_shifted_h036_v1", True, "fixed_exact_shifted_h036_v1"),
    "randomized-id": ("evaluation_id", True, "evaluation_id"),
    "ood": ("evaluation_ood", True, "evaluation_ood"),
}


class _HumanObservationProbeEnv(PiperHumanAwarePickPlaceEnv):
    """Change policy-visible human channels without changing physical state."""

    def __init__(self, cfg, human_obs_source):
        self.human_obs_source = human_obs_source
        self._probe_motion_generator = HumanTrajectoryGenerator(
            HumanAwareEnvConfig.from_preset("fixed_exact_h036_v1").human_motion)
        super().__init__(cfg)

    def reset(self, *, seed=None, options=None):
        if seed is None:
            raise ValueError("human-observation probes require an episode seed")
        self._probe_rng = np.random.default_rng(seed)
        self._probe_obs_buf = collections.deque()
        self._frozen_human_obs = None
        return super().reset(seed=seed, options=options)

    def _after_reset(self, rng, options, randomization_info):
        info = super()._after_reset(rng, options, randomization_info)
        if self.human_obs_source == "phantom-exact-h036":
            self._probe_trajectory = self._probe_motion_generator.sample(
                self._probe_rng, self.obj_pos, self.target_pos, force_present=True)
        return info

    def _extra_state_obs(self):
        # Preserve the physical env's RNG stream and observation bookkeeping.
        # The discarded actual observation consumes exactly its usual noise draws.
        actual = super()._extra_state_obs()
        if not self.human_cfg.include_human_state:
            raise ValueError("human-observation probes require the 13 human channels")
        if self.human_obs_source == "frozen-parked":
            if self._frozen_human_obs is None:
                self._frozen_human_obs = actual.copy()
                self._frozen_human_obs[0] = 0.0
            return self._frozen_human_obs.copy()

        state = self._probe_trajectory.state(
            float(self.data.time - self._human_time_zero))
        # Reuse the existing 13-channel encoder, including its noise, latency,
        # and real-TCP relative position. No mocap/physics setters are called.
        saved = (self.human_state, self._human_episode_active,
                 self.np_random, self._human_obs_buf)
        try:
            self.human_state = state
            self._human_episode_active = True
            self.np_random = self._probe_rng
            self._human_obs_buf = self._probe_obs_buf
            return super()._extra_state_obs()
        finally:
            (self.human_state, self._human_episode_active,
             self.np_random, self._human_obs_buf) = saved


def _make_env(cfg, human_obs_source="actual"):
    """Keep the default on the original, unmodified environment code path."""
    if human_obs_source == "actual":
        return PiperHumanAwarePickPlaceEnv(cfg)
    if human_obs_source not in {"phantom-exact-h036", "frozen-parked"}:
        raise ValueError(f"unknown human observation source: {human_obs_source}")
    return _HumanObservationProbeEnv(cfg, human_obs_source)


def _load_model(path: str, algo: str, env):
    choices = {"sac": SAC, "td3": TD3, "ppo": PPO}
    if algo != "auto":
        return choices[algo].load(path, env=env)
    errors = []
    for cls in (SAC, TD3, PPO):
        try:
            return cls.load(path, env=env)
        except Exception as exc:
            errors.append(f"{cls.__name__}: {exc}")
    raise RuntimeError("could not load model\n" + "\n".join(errors))


def _wilson(successes: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = successes / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, centre - half), min(1.0, centre + half)]


def _mean(rows: list[dict], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--algo", choices=["auto", "sac", "td3", "ppo"], default="auto")
    p.add_argument("--experiment", choices=list(EXPERIMENTS), required=True)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", default="out/human_aware_evaluation")
    p.add_argument("--video", default=None)
    p.add_argument("--video-episodes", type=int, default=1)
    p.add_argument("--camera", default="overview")
    p.add_argument("--stochastic", action="store_true",
                   help="use stochastic prediction; deterministic is the default")
    p.add_argument("--no-human-state", action="store_true")
    p.add_argument("--human-obs-source",
                   choices=["actual", "phantom-exact-h036", "frozen-parked"],
                   default="actual",
                   help="evaluation-only source for the 13 human observation channels")
    args = p.parse_args(argv)

    preset, include_state, distribution_label = EXPERIMENTS[args.experiment]
    cfg = HumanAwareEnvConfig.from_preset(preset).eval_variant()
    cfg.include_human_state = include_state and not args.no_human_state
    cfg.human_distribution = distribution_label
    cfg.render_camera = args.camera
    policy_probe = _load_model(args.model, args.algo, None)
    env = _make_env(cfg, args.human_obs_source)
    if (args.experiment == "no-human" and not args.no_human_state
            and policy_probe.observation_space != env.observation_space):
        # A no-human episode can serve either a legacy 51-state policy or a
        # human-aware 64-state policy observing a parked, absent human. Infer
        # that choice from the saved policy while retaining the explicit
        # --no-human-state override.
        env.close()
        cfg.include_human_state = True
        env = _make_env(cfg, args.human_obs_source)
    if policy_probe.observation_space != env.observation_space:
        raise ValueError(
            f"policy observation space {policy_probe.observation_space} does not match "
            f"environment {env.observation_space}; use the appropriate experiment "
            "or --no-human-state")
    policy = _load_model(args.model, args.algo, env)

    rows: list[dict] = []
    frames = []
    for episode in range(args.episodes):
        episode_seed = args.seed + episode
        obs, reset_info = env.reset(seed=episode_seed)
        total_reward = 0.0
        done = False
        if args.video and episode < args.video_episodes:
            frames.append(env.render(args.camera))
        while not done:
            action, _ = policy.predict(obs, deterministic=not args.stochastic)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            done = terminated or truncated
            if args.video and episode < args.video_episodes:
                frames.append(env.render(args.camera))
        summary = info["episode_summary"]
        success = bool(summary["success"])
        episode_duration = float(summary["length"] / cfg.control_hz)
        safety_clamped = int(summary["safety_clamped"])
        rows.append({
            "experiment": args.experiment,
            "distribution": distribution_label,
            "episode": episode,
            "seed": episode_seed,
            "trajectory_type": reset_info["human_trajectory_type"],
            "task_success": int(success),
            "collision_free_success": int(summary["collision_free_success"]),
            "human_collision": int(summary["human_collision"]),
            "near_miss": int(summary["near_miss_events"] > 0),
            "near_miss_events": summary["near_miss_events"],
            "minimum_separation": summary["minimum_human_distance"],
            "completion_time": episode_duration if success else "",
            "episode_duration": episode_duration,
            "episode_length": summary["length"],
            "placement_error": summary["placement_error"],
            "grasp_success": int(summary["grasp_success"]),
            "lift_success": int(summary["lift_success"]),
            "safety_interventions": summary["safety_clamped"] + summary["safety_vetoed"],
            "safety_clamped_steps": safety_clamped,
            "safety_vetoed_steps": summary["safety_vetoed"],
            "safety_clamp_rate": safety_clamped / max(1, summary["length"]),
            "proximity_steps": summary["proximity_steps"],
            "waiting_steps": summary["waiting_steps"],
            "waiting_time": summary["waiting_steps"] / cfg.control_hz,
            "human_safety_cost": summary["human_safety_cost"],
            "reward": total_reward,
            "termination": info["termination"],
        })

    n = len(rows)
    success_n = sum(row["task_success"] for row in rows)
    collision_n = sum(row["human_collision"] for row in rows)
    success_rows = [row for row in rows if row["task_success"]]
    separations = np.asarray([row["minimum_separation"] for row in rows], dtype=float)
    placement_errors = np.asarray([row["placement_error"] for row in rows], dtype=float)
    summary = {
        "experiment": args.experiment,
        "distribution": distribution_label,
        "episodes": n,
        "deterministic_prediction": not args.stochastic,
        "task_success_rate": success_n / n,
        "task_success_95ci": _wilson(success_n, n),
        "collision_free_success_rate": _mean(rows, "collision_free_success"),
        "human_collision_rate": collision_n / n,
        "human_collision_95ci": _wilson(collision_n, n),
        "near_miss_rate": _mean(rows, "near_miss"),
        "mean_minimum_separation": _mean(rows, "minimum_separation"),
        "median_minimum_separation": float(np.median(separations)),
        "p05_minimum_separation": float(np.percentile(separations, 5)),
        "worst_minimum_separation": min(row["minimum_separation"] for row in rows),
        "mean_completion_time_successes": (
            _mean(success_rows, "completion_time") if success_rows else None),
        "mean_episode_duration": _mean(rows, "episode_duration"),
        "mean_episode_length": _mean(rows, "episode_length"),
        "mean_placement_error": _mean(rows, "placement_error"),
        "median_placement_error": float(np.median(placement_errors)),
        "p90_placement_error": float(np.percentile(placement_errors, 90)),
        "grasp_rate": _mean(rows, "grasp_success"),
        "lift_rate": _mean(rows, "lift_success"),
        "mean_safety_interventions": _mean(rows, "safety_interventions"),
        "mean_safety_clamped_steps": _mean(rows, "safety_clamped_steps"),
        "mean_safety_clamp_rate": _mean(rows, "safety_clamp_rate"),
        "mean_proximity_steps": _mean(rows, "proximity_steps"),
        "mean_waiting_steps": _mean(rows, "waiting_steps"),
        "mean_waiting_time": _mean(rows, "waiting_time"),
        "mean_episode_reward": _mean(rows, "reward"),
        "mean_human_safety_cost": _mean(rows, "human_safety_cost"),
    }

    prefix = Path(args.out)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = prefix.with_suffix(".csv")
    json_path = prefix.with_suffix(".json")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "config": cfg.to_dict(),
                   "arguments": vars(args)}, f, indent=2, default=str)
    if args.video and frames:
        import imageio.v2 as imageio
        imageio.mimsave(args.video, frames, fps=int(cfg.control_hz), quality=8)

    print(json.dumps(summary, indent=2))
    print(f"episode CSV: {csv_path}")
    print(f"summary JSON: {json_path}")
    if args.video:
        print(f"video: {args.video}")
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

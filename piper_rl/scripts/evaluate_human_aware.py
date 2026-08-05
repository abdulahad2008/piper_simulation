"""Evaluate existing or human-aware policies under controlled HRI distributions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO, SAC, TD3

from piper_rl.human_aware_env import PiperHumanAwarePickPlaceEnv
from piper_rl.human_config import HumanAwareEnvConfig


EXPERIMENTS = {
    "no-human": ("no_human", False, "no_human"),
    "human-unaware": ("randomized", False, "evaluation_id_unaware"),
    "fixed": ("fixed", True, "evaluation_fixed_id"),
    "randomized-id": ("evaluation_id", True, "evaluation_id"),
    "ood": ("evaluation_ood", True, "evaluation_ood"),
}


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
    args = p.parse_args(argv)

    preset, include_state, distribution_label = EXPERIMENTS[args.experiment]
    cfg = HumanAwareEnvConfig.from_preset(preset).eval_variant()
    cfg.include_human_state = include_state and not args.no_human_state
    cfg.human_distribution = distribution_label
    cfg.render_camera = args.camera
    env = PiperHumanAwarePickPlaceEnv(cfg)
    policy = _load_model(args.model, args.algo, env)
    if policy.observation_space != env.observation_space:
        raise ValueError(
            f"policy observation space {policy.observation_space} does not match "
            f"environment {env.observation_space}; use the appropriate experiment "
            "or --no-human-state")

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
        rows.append({
            "experiment": args.experiment,
            "distribution": distribution_label,
            "episode": episode,
            "seed": episode_seed,
            "trajectory_type": reset_info["human_trajectory_type"],
            "task_success": int(summary["success"]),
            "collision_free_success": int(summary["collision_free_success"]),
            "human_collision": int(summary["human_collision"]),
            "near_miss": int(summary["near_miss_events"] > 0),
            "near_miss_events": summary["near_miss_events"],
            "minimum_separation": summary["minimum_human_distance"],
            "completion_time": summary["completion_time"],
            "placement_error": summary["placement_error"],
            "grasp_success": int(summary["grasp_success"]),
            "lift_success": int(summary["lift_success"]),
            "safety_interventions": summary["safety_clamped"] + summary["safety_vetoed"],
            "human_safety_cost": summary["human_safety_cost"],
            "reward": total_reward,
            "termination": info["termination"],
        })

    n = len(rows)
    success_n = sum(row["task_success"] for row in rows)
    collision_n = sum(row["human_collision"] for row in rows)
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
        "worst_minimum_separation": min(row["minimum_separation"] for row in rows),
        "mean_completion_time": _mean(rows, "completion_time"),
        "mean_placement_error": _mean(rows, "placement_error"),
        "grasp_rate": _mean(rows, "grasp_success"),
        "lift_rate": _mean(rows, "lift_success"),
        "mean_safety_interventions": _mean(rows, "safety_interventions"),
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

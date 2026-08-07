"""Preflight the exact fixed human crossing used by the seed-0 baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from piper_rl.human_aware_env import PiperHumanAwarePickPlaceEnv
from piper_rl.human_config import FIXED_EXACT_H036_V1, HumanAwareEnvConfig
from piper_rl.human_motion import trajectory_fingerprint, trajectory_fingerprint_sha256


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resets", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    cfg = HumanAwareEnvConfig.from_preset(FIXED_EXACT_H036_V1)
    env = PiperHumanAwarePickPlaceEnv(cfg)
    action_rng = np.random.default_rng(9_001)
    records = []
    finite = True
    for offset in range(args.resets):
        seed = args.seed + offset
        obs, info = env.reset(seed=seed)
        fingerprint = trajectory_fingerprint(env.human_trajectory, cfg.human_motion)
        fingerprint["preset"] = FIXED_EXACT_H036_V1
        next_obs, reward, _, _, _ = env.step(
            action_rng.uniform(-1, 1, env.action_space.shape).astype(np.float32))
        finite &= bool(np.isfinite(obs).all() and np.isfinite(next_obs).all()
                       and np.isfinite(reward))
        records.append({
            "seed": seed,
            "fingerprint_sha256": trajectory_fingerprint_sha256(fingerprint),
            "human_episode": bool(info["human_episode"]),
            "trajectory_type": info["human_trajectory_type"],
            "difficulty": cfg.human_difficulty,
            "object_position_m": env.obj_pos.tolist(),
            "target_position_m": env.target_pos.tolist(),
        })

    def sampled(seed):
        obs, _ = env.reset(seed=seed)
        fp = trajectory_fingerprint(env.human_trajectory, cfg.human_motion)
        fp["preset"] = FIXED_EXACT_H036_V1
        return obs, fp, env.obj_pos.copy(), env.target_pos.copy()

    obs_a, fp_a, obj_a, target_a = sampled(args.seed + 777)
    obs_b, fp_b, obj_b, target_b = sampled(args.seed + 777)
    fp_hashes = {record["fingerprint_sha256"] for record in records}
    object_positions = {tuple(record["object_position_m"]) for record in records}
    target_positions = {tuple(record["target_position_m"]) for record in records}
    phases = set(fp_a["trajectory"]["phases"])
    checks = {
        "all_human_present": all(record["human_episode"] for record in records),
        "one_unique_fingerprint": len(fp_hashes) == 1,
        "human_state_observation": env.observation_space.shape == (64,),
        "difficulty_fixed_at_one": all(record["difficulty"] == 1.0 for record in records),
        "only_cross_workspace": {record["trajectory_type"] for record in records} == {"cross_workspace"},
        "no_pause_or_reverse": not ({"paused", "reversing"} & phases),
        "manipulation_curriculum_enabled": cfg.curriculum,
        "object_randomization_varies": len(object_positions) > 1,
        "target_randomization_varies": len(target_positions) > 1,
        "finite_observations_actions_rewards": finite,
        "same_seed_reset_deterministic": (
            np.array_equal(obs_a, obs_b) and fp_a == fp_b
            and np.array_equal(obj_a, obj_b) and np.array_equal(target_a, target_b)),
        "different_seeds_preserve_human_trajectory": len(fp_hashes) == 1,
        "no_ood_distribution": cfg.human_distribution == FIXED_EXACT_H036_V1,
    }
    report = {
        "preset": FIXED_EXACT_H036_V1,
        "resets": args.resets,
        "seed_range": [args.seed, args.seed + args.resets - 1],
        "fingerprint": fp_a,
        "fingerprint_sha256": trajectory_fingerprint_sha256(fp_a),
        "unique_fingerprint_count": len(fp_hashes),
        "unique_object_positions": len(object_positions),
        "unique_target_positions": len(target_positions),
        "observation_shape": list(env.observation_space.shape),
        "action_shape": list(env.action_space.shape),
        "checks": checks,
        "valid": all(checks.values()),
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    env.close()
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

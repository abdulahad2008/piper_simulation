"""Kinematic statistics of the simulated person's arm, per human preset.

Pure kinematics (no MuJoCo): samples trajectories exactly as the environment
does and evaluates them on the 20 Hz control grid, over human-present steps.
Reports segment lengths, shoulder-to-hand distance, hand height above the
table, hand speed (mean and peak per episode) and elbow angle, as percentiles.
Used by the Phase 1 size audit to compare the arm with anthropometric data.

Usage: python -m piper_rl.scripts.human_arm_stats --episodes 1000 --out out/arm_stats.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from piper_rl.piper_env import TABLE_TOP
from piper_rl.human_config import HumanAwareEnvConfig
from piper_rl.human_motion import HumanTrajectoryGenerator

PRESETS = {
    "training_id": ("randomized", 1.0),
    "curriculum_d0": ("curriculum", 0.0),
    "curriculum_d05": ("curriculum", 0.5),
    "curriculum_d1": ("curriculum", 1.0),
    "fixed_height_randomized": ("fixed", 1.0),
    "fixed_exact_h036_v1": ("fixed_exact_h036_v1", 1.0),
    "fixed_exact_shifted_h036_v1": ("fixed_exact_shifted_h036_v1", 1.0),
    "evaluation_id": ("evaluation_id", 1.0),
    "evaluation_ood": ("evaluation_ood", 1.0),
}
PCTS = (5, 50, 95, 100)


def _pct(x: np.ndarray) -> dict:
    x = np.asarray(x, float)
    if x.size == 0:
        return {f"p{p}": None for p in PCTS}
    return {f"p{p}": float(np.percentile(x, p)) for p in PCTS}


def stats_for(preset: str, difficulty: float, episodes: int, seed0: int = 100_000,
              control_hz: int = 20, horizon_s: float = 10.0) -> dict:
    cfg = HumanAwareEnvConfig.from_preset(preset, difficulty)
    gen = HumanTrajectoryGenerator(cfg.human_motion)
    times = np.arange(0, int(horizon_s * control_hz) + 1) / control_hz
    upper, fore, reach, hand_z, speed, elbow_angle, elbow_z = [], [], [], [], [], [], []
    peak_speed, mean_speed = [], []
    obj = np.array([0.36, -0.13, 0.235])
    tgt = np.array([0.36, 0.15, 0.235])
    for ep in range(episodes):
        rng = np.random.default_rng(seed0 + ep)
        # object/target positions only shape reach_* trajectories; use the
        # scene defaults with the same jitter the environment applies.
        o = obj + np.array([*rng.uniform(-0.03, 0.03, 2), 0.0])
        t = tgt + np.array([*rng.uniform(-0.03, 0.03, 2), 0.0])
        traj = gen.sample(rng, o, t, force_present=True)
        ep_speeds = []
        for tt in times:
            st = traj.state(float(tt))
            if not st.human_present:
                continue
            u = st.elbow_position - st.shoulder_position
            f = st.hand_position - st.elbow_position
            upper.append(np.linalg.norm(u))
            fore.append(np.linalg.norm(f))
            reach.append(np.linalg.norm(st.hand_position - st.shoulder_position))
            hand_z.append(st.hand_position[2] - TABLE_TOP)
            elbow_z.append(st.elbow_position[2] - TABLE_TOP)
            v = float(np.linalg.norm(st.hand_velocity))
            speed.append(v)
            ep_speeds.append(v)
            c = float(np.dot(-u, f) / (np.linalg.norm(u) * np.linalg.norm(f) + 1e-12))
            elbow_angle.append(np.degrees(np.arccos(np.clip(c, -1, 1))))
        if ep_speeds:
            peak_speed.append(max(ep_speeds))
            mean_speed.append(float(np.mean([s for s in ep_speeds if s > 1e-6] or [0.0])))
    return {
        "preset": preset, "difficulty": difficulty, "episodes": episodes,
        "human_present_samples": len(upper),
        "upper_arm_length_m": _pct(upper),
        "forearm_length_m": _pct(fore),
        "shoulder_to_hand_m": _pct(reach),
        "hand_height_above_table_m": _pct(hand_z),
        "elbow_height_above_table_m": _pct(elbow_z),
        "hand_speed_m_s": _pct(speed),
        "episode_peak_hand_speed_m_s": _pct(peak_speed),
        "episode_mean_moving_speed_m_s": _pct(mean_speed),
        "elbow_angle_deg": _pct(elbow_angle),
        "shoulder_height_above_table_m": 0.54 - TABLE_TOP,
        "capsule_radii_m": {"upper_arm": 0.035, "forearm": 0.032, "hand": 0.045},
        "hand_capsule_half_length_m": 0.055,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--out", default="out/arm_stats.json")
    args = p.parse_args(argv)
    rows = {name: stats_for(pre, d, args.episodes) for name, (pre, d) in PRESETS.items()}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    keys = ["upper_arm_length_m", "forearm_length_m", "shoulder_to_hand_m",
            "hand_height_above_table_m", "hand_speed_m_s", "episode_peak_hand_speed_m_s", "elbow_angle_deg"]
    print(f"{'preset':<28}" + "".join(f"{k[:22]:>24}" for k in keys))
    print(f"{'(p5 / p50 / p95 / max)':<28}")
    for name, r in rows.items():
        cells = []
        for k in keys:
            v = r[k]
            cells.append(f"{v['p5']:.2f}/{v['p50']:.2f}/{v['p95']:.2f}/{v['p100']:.2f}")
        print(f"{name:<28}" + "".join(f"{c:>24}" for c in cells))
    print(f"written {out}")


if __name__ == "__main__":
    main()

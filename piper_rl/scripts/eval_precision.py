"""Per-episode evaluation -- the raw data of the precision-floor paper.

Unlike ``evaluate.py``, which prints aggregates and stores nothing, this writes
one CSV row per episode with the terminal error, the release-step
decomposition, the resolved interface and solver settings, and the full
domain-randomisation vector. Every table and figure in the paper is then
regenerated from those CSVs by ``reproduce.py`` -- no number in the paper comes
from a print statement.

Examples
--------
    # the archived baseline: v2, n=2000, nominal solver
    python -m piper_rl.scripts.eval_precision \
        --model release/piper_sac_v2_96pct.zip --episodes 2000 \
        --tag v2_n2000 --workers 2

    # a kill-gate cell
    python -m piper_rl.scripts.eval_precision --model ... --episodes 1000 \
        --timestep 0.001 --solref-scale 0.5 --tag v2_dt1ms_solref0.5
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from piper_rl.config import EnvConfig
from piper_rl.precision.instrumented_env import (InstrumentedPiperEnv,
                                                 EPISODE_FIELDS)
from piper_rl.precision.solver import SolverOverride

PROJECT_DIR = Path(__file__).resolve().parents[2]


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_DIR,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


# --------------------------------------------------------------------- #
def build_cfg(args) -> EnvConfig:
    """Resolve the interface from the CLI, holding latency and budget in seconds.

    Changing ``control_hz`` silently changes four other things: the number of
    physics sub-steps, the safety clamp ``max_dq = max_joint_vel/hz``, the
    observation latency in milliseconds, and the episode budget in seconds.
    Latency and budget are therefore re-derived from seconds here so the
    control-rate axis is not confounded three ways.
    """
    cfg = EnvConfig().eval_variant()

    base_hz = cfg.control_hz
    latency_s = cfg.noise.obs_latency_steps / base_hz
    budget_s = cfg.max_episode_steps / base_hz

    if args.control_hz is not None:
        cfg.control_hz = float(args.control_hz)
        cfg.noise.obs_latency_steps = int(round(latency_s * cfg.control_hz))
        cfg.max_episode_steps = int(round(budget_s * cfg.control_hz))
    if args.max_step_dist is not None:
        cfg.max_step_dist = float(args.max_step_dist)
    if args.action_smoothing is not None:
        cfg.action_smoothing = float(args.action_smoothing)
    if args.max_joint_vel is not None:
        cfg.limits.max_joint_vel = float(args.max_joint_vel)
    if args.max_tcp_speed is not None:
        cfg.limits.max_tcp_speed = float(args.max_tcp_speed)
    if args.place_tol_xy is not None:
        cfg.reward.place_tol_xy = float(args.place_tol_xy)
    if args.hard_corner_frac is not None:
        cfg.domain_rand.hard_corner_frac = float(args.hard_corner_frac)
    cfg.domain_rand.enabled = not args.no_domain_rand
    cfg.noise.enabled = not args.no_noise
    return cfg


def build_solver(args) -> SolverOverride:
    return SolverOverride(timestep=args.timestep, iterations=args.solver_iters,
                          impratio=args.impratio,
                          solref_scale=args.solref_scale)


# --------------------------------------------------------------------- #
def _worker(payload):
    """Run a contiguous block of episodes in one process."""
    args, lo, hi = payload
    import torch
    torch.set_num_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    cfg = build_cfg(args)
    if args.env == "piper":
        env = InstrumentedPiperEnv(cfg, solver=build_solver(args))
    else:
        # Second environment: gym-hil's MuJoCo Franka, the open simulation
        # companion to the HIL-SERL line of work. Same delta-action interface
        # family, different arm, different controller, different author.
        from piper_rl.precision.gymhil_env import (GymHilPrecisionEnv,
                                                   GymHilInterface, ARRANGE, PICK)
        iface = GymHilInterface(
            max_step_dist=(args.max_step_dist if args.max_step_dist is not None
                           else 0.025),
            action_smoothing=(args.action_smoothing
                              if args.action_smoothing is not None else 1.0),
            control_hz=(args.control_hz if args.control_hz is not None else 10.0),
            timestep=args.timestep)
        env = GymHilPrecisionEnv(ARRANGE if args.env == "gymhil-arrange" else PICK,
                                 interface=iface)
        cfg.max_step_dist = iface.max_step_dist
        cfg.action_smoothing = iface.action_smoothing
        cfg.control_hz = iface.control_hz
        cfg.max_episode_steps = env.max_episode_steps
        cfg.noise.obs_latency_steps = 0
        cfg.limits.max_joint_vel = float("nan")

    policy = None
    if args.model:
        import stable_baselines3 as sb3
        policy = sb3.SAC.load(args.model, env=None, device="cpu")

    gh = git_hash()
    rows = []
    for ep in range(lo, hi):
        seed = args.seed + ep
        if args.nested_noise > 1:
            # Nested design: `nested_noise` independent noise replays of each
            # scene. The intra-class correlation of the outcome within a scene
            # is what separates environment-determined failures from
            # irreducible ones.
            dr_seed = args.seed + ep // args.nested_noise
            noise_seed = 900_000 + ep % args.nested_noise
            obs, _ = env.reset(seed=seed, options={"dr_seed": dr_seed,
                                                   "noise_seed": noise_seed})
        else:
            dr_seed, noise_seed = seed, seed
            obs, _ = env.reset(seed=seed)
        total_r, steps, info, done = 0.0, 0, {}, False
        while not done and steps < cfg.max_episode_steps:
            if policy is not None:
                action, _ = policy.predict(obs, deterministic=True)
            else:                                   # zero-action reference
                action = np.zeros(env.action_space.shape, dtype=np.float32)
            obs, r, term, trunc, info = env.step(action)
            total_r += r
            steps += 1
            done = term or trunc

        s = info.get("episode_summary") or env._episode_summary(
            {"success": info.get("is_success", False)})
        dr = s.get("dr_vector", {})
        solver = s.get("solver", {})
        rows.append({
            "episode": ep, "seed": seed, "dr_seed": dr_seed,
            "noise_seed": noise_seed, "git_hash": gh, "cell": args.tag,
            "success_at_default_tol": int(bool(s["success"])),
            "grasp": int(bool(s["grasp_success"])),
            "lift": int(bool(s["lift_success"])),
            "termination": info.get("termination", ""),
            "steps": int(s["length"]),
            "place_err_mm": 1000.0 * float(s["placement_error"]),
            "tcp_err_at_release_mm": 1000.0 * float(s["tcp_err_at_release"]),
            "scatter_mm": 1000.0 * float(s["scatter"]),
            "release_step": int(s["release_step"]),
            "obj_settled": int(bool(s["obj_settled"])),
            "max_step_dist_mm": 1000.0 * cfg.max_step_dist,
            "action_smoothing": cfg.action_smoothing,
            "control_hz": cfg.control_hz,
            "max_joint_vel": cfg.limits.max_joint_vel,
            "n_substeps": int(s["n_substeps"]),
            "obs_latency_steps": cfg.noise.obs_latency_steps,
            "max_episode_steps": cfg.max_episode_steps,
            "solver_timestep": solver.get("solver_timestep"),
            "solver_iterations": solver.get("solver_iterations"),
            "solver_impratio": solver.get("solver_impratio"),
            "solver_solref_scale": solver.get("solver_solref_scale"),
            "collision_steps": int(s["collision_steps"]),
            "safety_clamped": int(s["safety_clamped"]),
            "safety_vetoed": int(s["safety_vetoed"]),
            "clamp_rate": (float(s["safety_clamped"]) / max(1, int(s["length"]))),
            **{k: dr.get(k, float("nan")) for k in
               ("obj_r", "obj_th", "obj_yaw", "obj_half_w", "obj_half_h",
                "obj_mass", "mu_obj", "mu_table", "tgt_r", "tgt_th",
                "kp_scale_mean", "kv_scale_mean", "frictionloss_scale_mean")},
            "reward": total_r,
        })
    env.close()
    return rows


# --------------------------------------------------------------------- #
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--seed", type=int, default=10_000)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)))
    p.add_argument("--tag", type=str, required=True)
    p.add_argument("--env", choices=["piper", "gymhil-arrange", "gymhil-pick"],
                   default="piper",
                   help="which environment to evaluate in. The gym-hil options "
                        "are the second environment of the study: a Franka "
                        "under operational-space control rather than a Piper "
                        "under differential IK.")
    p.add_argument("--out-dir", type=str, default="results/precision")
    # --- interface ---------------------------------------------------- #
    p.add_argument("--max-step-dist", type=float, default=None, help="m")
    p.add_argument("--action-smoothing", type=float, default=None)
    p.add_argument("--control-hz", type=float, default=None)
    p.add_argument("--max-joint-vel", type=float, default=None, help="rad/s")
    p.add_argument("--max-tcp-speed", type=float, default=None, help="m/s")
    # --- solver (kill gate) -------------------------------------------- #
    p.add_argument("--timestep", type=float, default=None, help="s")
    p.add_argument("--solver-iters", type=int, default=None)
    p.add_argument("--impratio", type=float, default=None)
    p.add_argument("--solref-scale", type=float, default=1.0)
    # --- task ----------------------------------------------------------- #
    p.add_argument("--place-tol-xy", type=float, default=None, help="m")
    p.add_argument("--hard-corner-frac", type=float, default=None)
    p.add_argument("--no-domain-rand", action="store_true")
    p.add_argument("--no-noise", action="store_true")
    p.add_argument("--nested-noise", type=int, default=1,
                   help="if > 1, replay every scene this many times under "
                        "independent noise; --episodes must be a multiple of it")
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{args.tag}.csv"
    meta_path = out_dir / f"{args.tag}.meta.json"

    n_workers = max(1, min(args.workers, args.episodes))
    bounds = np.linspace(0, args.episodes, n_workers + 1).astype(int)
    payloads = [(args, int(bounds[i]), int(bounds[i + 1]))
                for i in range(n_workers) if bounds[i + 1] > bounds[i]]

    t0 = time.time()
    if len(payloads) == 1:
        chunks = [_worker(payloads[0])]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(len(payloads)) as pool:
            chunks = pool.map(_worker, payloads)
    rows = [r for c in chunks for r in c]
    rows.sort(key=lambda r: r["episode"])
    dt = time.time() - t0

    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=EPISODE_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    cfg = build_cfg(args)
    if args.env != "piper":
        # build_cfg resolves the Piper interface; the gym-hil cells resolve a
        # different one in the worker, and the meta must record what actually
        # ran, not the default it never used.
        from piper_rl.precision.gymhil_env import GymHilInterface
        gi = GymHilInterface(
            max_step_dist=(args.max_step_dist if args.max_step_dist is not None
                           else 0.025),
            action_smoothing=(args.action_smoothing
                              if args.action_smoothing is not None else 1.0),
            control_hz=(args.control_hz if args.control_hz is not None else 10.0),
            timestep=args.timestep)
        cfg.max_step_dist = gi.max_step_dist
        cfg.action_smoothing = gi.action_smoothing
        cfg.control_hz = gi.control_hz
        cfg.max_episode_steps = int(round(gi.max_episode_s * gi.control_hz))
        cfg.noise.obs_latency_steps = 0
        cfg.limits.max_joint_vel = float("nan")
        cfg.limits.max_tcp_speed = gi.max_step_dist * gi.control_hz
    meta = {
        "tag": args.tag, "env": args.env, "model": args.model,
        "episodes": args.episodes,
        "seed": args.seed, "git_hash": git_hash(),
        "wall_clock_s": round(dt, 1), "workers": len(payloads),
        "interface": {"max_step_dist_m": cfg.max_step_dist,
                      "action_smoothing": cfg.action_smoothing,
                      "control_hz": cfg.control_hz,
                      "max_joint_vel": cfg.limits.max_joint_vel,
                      "max_tcp_speed": cfg.limits.max_tcp_speed,
                      "obs_latency_steps": cfg.noise.obs_latency_steps,
                      "max_episode_steps": cfg.max_episode_steps},
        "solver": build_solver(args).as_dict(),
        "domain_rand": cfg.domain_rand.enabled, "noise": cfg.noise.enabled,
        "versions": {},
    }
    for mod in ("mujoco", "stable_baselines3", "torch", "gymnasium", "numpy"):
        try:
            meta["versions"][mod] = __import__(mod).__version__
        except Exception:
            meta["versions"][mod] = "unavailable"
    meta_path.write_text(json.dumps(meta, indent=2))

    err = np.array([r["place_err_mm"] for r in rows])
    succ = np.array([r["success_at_default_tol"] for r in rows], dtype=bool)
    print(f"[{args.tag}] {len(rows)} episodes in {dt/60:.1f} min "
          f"({dt/max(1,len(rows)):.2f} s/ep)  ->  {csv_path}")
    print(f"  success@default {succ.mean():.1%}   "
          f"median err {np.median(err):.1f} mm   p90 {np.percentile(err,90):.1f} mm")
    return 0


if __name__ == "__main__":
    sys.exit(main())

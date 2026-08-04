"""Deterministic evaluation + visual demonstration of a trained PIPER policy.

Every episode starts from the *true* initial state (arm at home, object on the
table) -- the training curriculum is always disabled here, so the number this
prints is the number that matters.

Reported metrics
----------------
    success rate         object placed on the target, released, and settled
    grasp success rate   a two-sided grasp was achieved at least once
    lift success rate    the object cleared the table by ``lift_height``
    placement accuracy   mean / median / p90 object-to-target distance [mm]
    episode reward       mean +- std
    episode length       mean, and mean over successful episodes only
    collision steps      steps in which an arm link touched table/object/itself
    safety events        commands the safety layer clamped or vetoed

Examples
--------
    python -m piper_rl.scripts.evaluate --model runs/sac_v1/best/best_model.zip
    python -m piper_rl.scripts.evaluate --model ... --episodes 50 --video eval.mp4
    python -m piper_rl.scripts.evaluate --model ... --viewer      # interactive
    python -m piper_rl.scripts.evaluate --scripted                # baseline
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from piper_rl.config import EnvConfig
from piper_rl.piper_env import PiperPickPlaceEnv

ALGOS = {"sac": "SAC", "td3": "TD3", "ppo": "PPO"}


def load_model(path: str, algo: str | None, env):
    import stable_baselines3 as sb3
    if algo is None:
        name = Path(path).as_posix().lower()
        algo = next((a for a in ALGOS if a in name), "sac")
    return getattr(sb3, ALGOS[algo]).load(path, env=None, device="cpu"), algo


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--algo", choices=list(ALGOS), default=None)
    p.add_argument("--scripted", action="store_true",
                   help="evaluate the scripted IK controller instead of a policy")
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--seed", type=int, default=10_000)
    p.add_argument("--deterministic", type=int, default=1)
    p.add_argument("--video", type=str, default=None,
                   help="write an mp4 of the first N episodes")
    p.add_argument("--video-episodes", type=int, default=3)
    p.add_argument("--camera", type=str, default="forehead",
                   help="forehead | wrist | overview | front | side | top")
    p.add_argument("--viewer", action="store_true",
                   help="open the interactive MuJoCo viewer (needs a display; "
                        "on macOS run with `mjpython`)")
    p.add_argument("--no-domain-rand", action="store_true")
    p.add_argument("--no-noise", action="store_true")
    p.add_argument("--action-mode", choices=["cartesian", "joint"], default=None)
    p.add_argument("--realtime", action="store_true",
                   help="throttle the viewer to wall-clock speed")
    p.add_argument("--place-tol-xy", type=float, default=None,
                   help="m, score success against this tolerance instead of the "
                        "default 0.045. Use it to compare runs on equal terms.")
    p.add_argument("--hard-corner-frac", type=float, default=None,
                   help="fraction of episodes spawned in the hard outer corner; "
                        "use 1.0 to evaluate the worst case only")
    args = p.parse_args(argv)

    cfg = EnvConfig().eval_variant()
    if args.action_mode:
        cfg.action_mode = args.action_mode
    if args.place_tol_xy is not None:
        cfg.reward.place_tol_xy = args.place_tol_xy
    if args.hard_corner_frac is not None:
        cfg.domain_rand.hard_corner_frac = args.hard_corner_frac
    cfg.domain_rand.enabled = not args.no_domain_rand
    cfg.noise.enabled = not args.no_noise
    cfg.render_camera = args.camera
    cfg.render_width, cfg.render_height = 640, 480

    env = PiperPickPlaceEnv(cfg, render_mode="human" if args.viewer else None)

    policy = None
    label = "scripted IK controller"
    if not args.scripted:
        if not args.model:
            p.error("--model is required unless --scripted is given")
        policy, algo = load_model(args.model, args.algo, env)
        label = f"{algo.upper()} policy  ({args.model})"
        if policy.observation_space.shape != env.observation_space.shape:
            print(f"WARNING: model obs {policy.observation_space.shape} != "
                  f"env obs {env.observation_space.shape}")

    if args.scripted:
        from piper_rl.scripts.scripted_demo import build_plan

    frames = []
    rows = []
    dt = 1.0 / cfg.control_hz

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        record = args.video and ep < args.video_episodes
        total_r, steps, info = 0.0, 0, {}
        done = False

        if args.scripted:
            plan = build_plan(env)
            stage = 0
            stage_steps = 0

        while not done and steps < cfg.max_episode_steps:
            if policy is not None:
                action, _ = policy.predict(obs, deterministic=bool(args.deterministic))
            else:
                name, target, grip, ms, tol, gain = plan[min(stage, len(plan) - 1)]
                err = target - env.tcp_pos
                a = np.clip(gain * err / cfg.max_step_dist, -1, 1)
                action = np.array([a[0], a[1], a[2], 0.0, float(grip)])
                stage_steps += 1
                if stage_steps >= ms or (tol > 0 and np.linalg.norm(err) < tol):
                    stage += 1
                    stage_steps = 0
                    if stage >= len(plan):
                        action = np.array([0, 0, 0.3, 0, 1.0])
                        stage = len(plan) - 1

            t0 = time.time()
            obs, r, terminated, truncated, info = env.step(action)
            total_r += r
            steps += 1
            done = terminated or truncated
            if record:
                frames.append(env.render(args.camera))
            if args.viewer:
                env.render()
                if args.realtime:
                    time.sleep(max(0.0, dt - (time.time() - t0)))

        s = info.get("episode_summary") or env._episode_summary(
            {"success": info.get("is_success", False)})
        rows.append(dict(reward=total_r, **s))
        print(f"ep {ep:3d}  R {total_r:8.2f}  success {int(s['success'])}"
              f"  grasp {int(s['grasp_success'])}  lift {int(s['lift_success'])}"
              f"  place_err {s['placement_error']*1000:6.1f} mm"
              f"  len {s['length']:3d}  collisions {s['collision_steps']:3d}")

    # ------------------------------------------------------------- summary
    def col(k):
        return np.array([r[k] for r in rows], dtype=float)

    succ = col("success").astype(bool)
    errs = col("placement_error") * 1000
    print("\n" + "=" * 72)
    print(f"EVALUATION  |  {label}")
    print(f"  episodes            {args.episodes}   deterministic="
          f"{bool(args.deterministic)}  domain_rand={cfg.domain_rand.enabled}"
          f"  noise={cfg.noise.enabled}")
    print("-" * 72)
    print(f"  SUCCESS RATE        {succ.mean():6.1%}"
          f"   ({int(succ.sum())}/{args.episodes})")
    print(f"  grasp rate          {col('grasp_success').mean():6.1%}")
    print(f"  lift rate           {col('lift_success').mean():6.1%}")
    print(f"  episode reward      {col('reward').mean():8.2f} +- {col('reward').std():.2f}")
    print(f"  episode length      {col('length').mean():8.1f} steps"
          + (f"   (successful only: {col('length')[succ].mean():.1f})"
             if succ.any() else ""))
    print("-" * 72)
    print(f"  placement error     mean {errs.mean():6.1f} mm |"
          f" median {np.median(errs):6.1f} mm | p90 {np.percentile(errs, 90):6.1f} mm")
    if succ.any():
        e = errs[succ]
        print(f"    on successes      mean {e.mean():6.1f} mm |"
              f" median {np.median(e):6.1f} mm | max {e.max():6.1f} mm")
    print(f"  collision steps     {col('collision_steps').mean():8.1f} / episode")
    print(f"  safety clamped      {col('safety_clamped').mean():8.1f} cmds cumulative")
    print(f"  safety vetoed       {col('safety_vetoed').mean():8.1f} cmds cumulative")
    print("=" * 72)

    if args.video and frames:
        import imageio
        imageio.mimsave(args.video, frames, fps=int(cfg.control_hz), quality=8)
        print(f"wrote {args.video}  ({len(frames)} frames, camera='{args.camera}')")

    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Scripted pick-and-place, executed through the RL environment itself.

Purpose: prove the task is *physically solvable* by this arm, this gripper and
this object, and that the env's grasp/lift/place detection and reward agree with
what the eye sees. If this does not pass, no amount of RL will help.

The controller is an open-loop Cartesian waypoint follower driving the same
5-D action space the policy uses -- it is not privileged. It therefore also
serves as a sanity check on the action scaling: if 3 cm/step cannot get the arm
anywhere useful, the policy will not manage either.

Usage
-----
    python -m piper_rl.scripts.scripted_demo                  # 20 episodes, headless
    python -m piper_rl.scripts.scripted_demo --episodes 5 --video demo.mp4
    python -m piper_rl.scripts.scripted_demo --no-domain-rand --no-noise
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from piper_rl.config import EnvConfig
from piper_rl.piper_env import PiperPickPlaceEnv, GRIP_OPEN, TABLE_TOP


# Waypoint plan. Each entry: (name, target, gripper, max_steps, tol, gain)
#   gripper: +1 = open, -1 = closed
#   gain   : proportional gain on the Cartesian error, in units of
#            ``max_step_dist``. 1.0 saturates the action whenever the error
#            exceeds one step. Stages that move NEAR THE OBJECT use a lower
#            gain so the gripper decelerates instead of swatting the cup off
#            the table -- with gain 1.0 everywhere, ~40% of episodes ended with
#            the object 0.4 m away.
CARRY_Z = 0.305          # top of the reachable straight-down envelope (~0.32)


def build_plan(env: PiperPickPlaceEnv):
    obj = env.obj_pos.copy()
    tgt = env.target_pos.copy()
    grasp_z = max(obj[2], TABLE_TOP + 0.045)
    return [
        ("approach", np.array([obj[0], obj[1], CARRY_Z]), +1, 70, 0.015, 1.0),
        ("descend",  np.array([obj[0], obj[1], grasp_z]), +1, 70, 0.010, 0.7),
        ("close",    np.array([obj[0], obj[1], grasp_z]), -1, 30, 0.0,   0.5),
        ("lift",     np.array([obj[0], obj[1], CARRY_Z]), -1, 60, 0.02,  1.0),
        ("carry",    np.array([tgt[0], tgt[1], CARRY_Z]), -1, 90, 0.015, 1.0),
        ("lower",    np.array([tgt[0], tgt[1], grasp_z + 0.004]), -1, 70, 0.010, 0.7),
        ("release",  np.array([tgt[0], tgt[1], grasp_z + 0.004]), +1, 30, 0.0, 0.4),
        ("retreat",  np.array([tgt[0], tgt[1], CARRY_Z]), +1, 45, 0.03,  1.0),
    ]


def run_episode(env: PiperPickPlaceEnv, seed: int, frames: list | None = None,
                verbose: bool = False):
    obs, info = env.reset(seed=seed)
    plan = build_plan(env)
    total_r = 0.0
    stage_log = []
    last_info = {}

    for name, target, grip, max_steps, tol, gain in plan:
        for _ in range(max_steps):
            err = target - env.tcp_pos
            # P-controller in Cartesian space, saturated to the action range.
            a_xyz = np.clip(gain * err / env.cfg.max_step_dist, -1.0, 1.0)
            action = np.array([a_xyz[0], a_xyz[1], a_xyz[2], 0.0, float(grip)])
            obs, r, terminated, truncated, last_info = env.step(action)
            total_r += r
            if frames is not None:
                frames.append(env.render())
            if terminated or truncated:
                stage_log.append((name, float(np.linalg.norm(err))))
                return total_r, last_info, stage_log
            if tol > 0 and np.linalg.norm(err) < tol:
                break
        stage_log.append((name, float(np.linalg.norm(target - env.tcp_pos))))
        if verbose:
            print(f"    {name:9s} err={np.linalg.norm(target-env.tcp_pos)*1000:5.1f}mm "
                  f"grasped={last_info.get('grasped')} lift={last_info.get('lift_height',0):.3f}")

    # Let the scene settle so the success detector sees a static object.
    for _ in range(20):
        obs, r, terminated, truncated, last_info = env.step(
            np.array([0, 0, 0.3, 0, 1.0]))
        total_r += r
        if frames is not None:
            frames.append(env.render())
        if terminated or truncated:
            break
    return total_r, last_info, stage_log


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--video", type=str, default=None,
                   help="write an mp4/gif of the first episode")
    p.add_argument("--camera", type=str, default="forehead")
    p.add_argument("--no-domain-rand", action="store_true")
    p.add_argument("--no-noise", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    cfg = EnvConfig()
    cfg.curriculum = False                 # always start from the true state
    cfg.domain_rand.enabled = not args.no_domain_rand
    cfg.noise.enabled = not args.no_noise
    cfg.render_camera = args.camera
    env = PiperPickPlaceEnv(cfg)

    n_succ = n_grasp = n_lift = 0
    place_errs, rewards, lengths = [], [], []

    for ep in range(args.episodes):
        frames = [] if (args.video and ep == 0) else None
        if args.verbose:
            print(f"episode {ep}")
        r, info, _ = run_episode(env, args.seed + ep, frames, args.verbose)
        s = info.get("episode_summary", {})
        n_succ += bool(s.get("success", info.get("success", False)))
        n_grasp += bool(s.get("grasp_success", False))
        n_lift += bool(s.get("lift_success", False))
        place_errs.append(s.get("placement_error", info.get("d_place", np.nan)))
        rewards.append(r)
        lengths.append(s.get("length", cfg.max_episode_steps))
        print(f"ep {ep:3d}  reward {r:8.2f}  grasp {int(bool(s.get('grasp_success')))}"
              f"  lift {int(bool(s.get('lift_success')))}"
              f"  success {int(bool(s.get('success')))}"
              f"  place_err {place_errs[-1]*1000:6.1f} mm  len {lengths[-1]}")

        if frames:
            import imageio
            imageio.mimsave(args.video, frames, fps=int(cfg.control_hz))
            print(f"  wrote {args.video} ({len(frames)} frames)")

    n = args.episodes
    print("\n" + "=" * 62)
    print(f"scripted controller over {n} episodes"
          f"  (domain_rand={cfg.domain_rand.enabled}, noise={cfg.noise.enabled})")
    print(f"  grasp rate      {n_grasp/n:6.1%}")
    print(f"  lift  rate      {n_lift/n:6.1%}")
    print(f"  SUCCESS rate    {n_succ/n:6.1%}")
    print(f"  mean reward     {np.mean(rewards):8.2f}")
    print(f"  mean place err  {np.nanmean(place_errs)*1000:6.1f} mm")
    print(f"  mean length     {np.mean(lengths):6.1f} steps")
    print("=" * 62)
    env.close()
    return 0 if n_succ / n > 0.5 else 1


if __name__ == "__main__":
    sys.exit(main())

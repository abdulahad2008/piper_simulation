"""Train the second environment's policy: SAC on gym-hil's MuJoCo Franka.

Same algorithm, same hyperparameters and the same final-checkpoint selection
rule as the Piper cells, so that a difference between the two environments is
a difference between the environments and not between two training recipes.
The interface flags are the same flags.

    python -m piper_rl.scripts.train_gymhil --tag gymhil_theta0_s0 --seed 0 \
        --timesteps 300000 --max-step-dist 0.025 --control-hz 10

Note on the interface constants: gym-hil ships Delta_max = 25 mm at 10 Hz with
no smoothing and no joint-velocity clamp. Those are theta_0 for this
environment; the sweep moves them exactly as it moves the Piper's.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]


def _git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=str(PROJECT_DIR),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def make_env_fn(args, seed: int):
    import gymnasium as gym
    from piper_rl.precision.gymhil_env import (GymHilPrecisionEnv,
                                               GymHilInterface, ARRANGE, PICK)

    class _Gym(gym.Env):
        """Thin Gymnasium adapter so SB3 sees a standard env."""

        def __init__(self):
            self.inner = GymHilPrecisionEnv(
                ARRANGE if args.task == "arrange" else PICK,
                GymHilInterface(max_step_dist=args.max_step_dist,
                                action_smoothing=args.action_smoothing,
                                control_hz=args.control_hz,
                                timestep=args.timestep))
            self.observation_space = self.inner.env.observation_space
            self.action_space = self.inner.action_space

        def reset(self, *, seed=None, options=None):
            return self.inner.reset(seed=seed)

        def step(self, a):
            return self.inner.step(a)

        def close(self):
            self.inner.close()

    def _init():
        e = _Gym()
        e.reset(seed=seed)
        e.action_space.seed(seed)
        return e
    return _init


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", required=True)
    p.add_argument("--task", choices=["pick", "arrange"], default="pick")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timesteps", type=int, default=300_000)
    p.add_argument("--n-envs", type=int, default=2)
    p.add_argument("--out", type=str, default="runs/gymhil")
    p.add_argument("--checkpoint-freq", type=int, default=25_000)
    # the swept interface -- gym-hil's released values are the defaults
    p.add_argument("--max-step-dist", type=float, default=0.025)
    p.add_argument("--action-smoothing", type=float, default=1.0)
    p.add_argument("--control-hz", type=float, default=10.0)
    p.add_argument("--timestep", type=float, default=None)
    args = p.parse_args(argv)

    import torch, triton  # noqa: F401  -- before any GL context; see gymhil_env
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    out = Path(args.out) / args.tag
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)

    fns = [lambda i=i: Monitor(make_env_fn(args, args.seed + i)(),
                               filename=str(out / f"m{i}"))
           for i in range(args.n_envs)]
    venv = SubprocVecEnv(fns, start_method="fork") if args.n_envs > 1 \
        else DummyVecEnv(fns)

    model = SAC("MultiInputPolicy", venv, seed=args.seed, device="cpu",
                learning_rate=3e-4, buffer_size=400_000, batch_size=256,
                tau=0.01, gamma=0.98, train_freq=(1, "step"),
                gradient_steps=args.n_envs, learning_starts=5_000,
                policy_kwargs=dict(net_arch=[256, 256]), verbose=1)

    (out / "config.json").write_text(json.dumps(
        {"args": vars(args), "git_hash": _git_hash(),
         "env": "gym_hil", "algo": "sac"}, indent=2, default=str))

    t0 = time.time()
    model.learn(total_timesteps=args.timesteps,
                callback=CheckpointCallback(
                    save_freq=max(1, args.checkpoint_freq // args.n_envs),
                    save_path=str(out / "checkpoints"), name_prefix="gymhil"),
                progress_bar=False)
    # The FINAL model is the reported one, never the best -- same rule as the
    # Piper cells, for the same reason.
    model.save(str(out / "final_model"))
    print(f"[{args.tag}] {args.timesteps} steps in {(time.time()-t0)/60:.1f} min "
          f"-> {out/'final_model.zip'}")
    venv.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Micro-benchmark: cost of one gradient update and one environment step.

Answers, on the machine it runs on, the two numbers that set training speed:
ms per update for each learner at each torch thread count, and environment
steps per second for 1, 4 and 8 environments. Run it on an otherwise idle
machine, then again while the real runs are going.

    python -m piper_rl.scripts.bench_update_cost --threads 1 2 4 --out out/bench_update_cost.json

Learners: sac (SB3), tqc and crossq (sb3-contrib, if installed), sbx_sac and
sbx_crossq (SBX / JAX, if installed). Each learner is trained for a few
hundred updates on a 64-dim observation, 5-dim action, batch 256, 256x256
network, from a random buffer; timing excludes the first 50 warm-up updates.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np


def _dummy_env():
    import gymnasium as gym
    from gymnasium import spaces

    class Dummy(gym.Env):
        observation_space = spaces.Box(-np.inf, np.inf, (64,), np.float32)
        action_space = spaces.Box(-1, 1, (5,), np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return np.zeros(64, np.float32), {}

        def step(self, a):
            return np.random.randn(64).astype(np.float32), 0.0, False, False, {}
    return Dummy()


def bench_learner(name: str, threads: int, n_updates: int = 300) -> dict | None:
    import torch
    torch.set_num_threads(threads)
    os.environ["OMP_NUM_THREADS"] = str(threads)
    kw = dict(policy="MlpPolicy", env=_dummy_env(), learning_starts=1000, batch_size=256,
              policy_kwargs=dict(net_arch=[256, 256]), verbose=0, device="cpu", seed=0)
    try:
        if name == "sac":
            from stable_baselines3 import SAC as Algo
        elif name == "tqc":
            from sb3_contrib import TQC as Algo
        elif name == "crossq":
            from sb3_contrib import CrossQ as Algo
        elif name == "sbx_sac":
            from sbx import SAC as Algo
            kw.pop("device")
        elif name == "sbx_crossq":
            from sbx import CrossQ as Algo
            kw.pop("device")
        else:
            return None
    except Exception as e:  # noqa: BLE001
        return {"learner": name, "threads": threads, "error": f"{type(e).__name__}: {e}"}
    model = Algo(**kw)
    # fill the buffer with random transitions
    model.learn(total_timesteps=1000, log_interval=None)
    model.learn(total_timesteps=50, log_interval=None, reset_num_timesteps=False)   # warm-up updates
    t0 = time.perf_counter()
    model.learn(total_timesteps=n_updates, log_interval=None, reset_num_timesteps=False)
    dt = (time.perf_counter() - t0) / n_updates
    return {"learner": name, "threads": threads, "ms_per_update": dt * 1e3, "n_updates": n_updates}


def bench_env(n_envs: int, steps: int = 400) -> dict:
    from piper_rl.human_config import HumanAwareEnvConfig
    from piper_rl.human_aware_env import PiperHumanAwarePickPlaceEnv
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    def make(i):
        def f():
            cfg = HumanAwareEnvConfig.from_preset("curriculum", 1.0)
            return PiperHumanAwarePickPlaceEnv(cfg)
        return f
    fns = [make(i) for i in range(n_envs)]
    venv = SubprocVecEnv(fns, start_method="spawn") if n_envs > 1 else DummyVecEnv(fns)
    venv.reset()
    acts = np.zeros((n_envs, 5), np.float32)
    for _ in range(20):
        venv.step(acts)
    t0 = time.perf_counter()
    for _ in range(steps):
        venv.step(acts)
    dt = time.perf_counter() - t0
    venv.close()
    return {"n_envs": n_envs, "env_steps_per_s": n_envs * steps / dt, "ms_per_vec_step": dt / steps * 1e3}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--learners", nargs="+", default=["sac", "tqc", "crossq", "sbx_sac", "sbx_crossq"])
    p.add_argument("--envs", type=int, nargs="+", default=[1, 4, 8])
    p.add_argument("--out", default="out/bench_update_cost.json")
    args = p.parse_args(argv)
    import torch
    info = {"platform": platform.platform(), "python": sys.version.split()[0], "torch": torch.__version__,
            "cpu_count": os.cpu_count(), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    rows = []
    print(f"{'learner':<12}{'threads':>8}{'ms/update':>12}")
    for name in args.learners:
        for th in args.threads:
            r = bench_learner(name, th)
            if r is None:
                continue
            rows.append(r)
            if "error" in r:
                print(f"{name:<12}{th:>8}   unavailable: {r['error'][:70]}")
                break
            print(f"{name:<12}{th:>8}{r['ms_per_update']:>12.2f}")
    env_rows = []
    print(f"\n{'n_envs':>6}{'env steps/s':>14}{'ms/vec step':>14}")
    for n in args.envs:
        r = bench_env(n)
        env_rows.append(r)
        print(f"{n:>6}{r['env_steps_per_s']:>14.1f}{r['ms_per_vec_step']:>14.2f}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"info": info, "learners": rows, "envs": env_rows}, indent=2))
    print("written", out)


if __name__ == "__main__":
    main()

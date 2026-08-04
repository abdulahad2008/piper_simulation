"""Find the fastest training settings for THIS machine.

Training throughput here is a tug-of-war between two things that both want your
cores:

  * the **simulators** -- one MuJoCo process per parallel environment, each
    single-threaded, each costing ~2.2 ms per env step;
  * the **gradient update** -- one PyTorch process doing small dense matmuls,
    which scales with threads but with rapidly diminishing returns.

Give torch every core and the simulators starve. Give the simulators every core
and the update starves. The optimum depends on your core count, so measure it
instead of guessing.

    python -m piper_rl.scripts.benchmark
    python -m piper_rl.scripts.benchmark --quick
    python -m piper_rl.scripts.benchmark --n-envs 4 8 12 --threads 4 8 16

Prints a table of end-to-end training throughput (env steps/s, i.e. including
the gradient updates) and a recommended command line.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

from piper_rl.config import EnvConfig
from piper_rl.piper_env import PiperPickPlaceEnv


def cpu_info():
    logical = os.cpu_count() or 1
    physical = logical
    try:
        import psutil
        physical = psutil.cpu_count(logical=False) or logical
    except Exception:
        pass
    return physical, logical


def bench_env_only(n: int = 400) -> float:
    """Pure simulator throughput, one environment, no learning."""
    env = PiperPickPlaceEnv(EnvConfig())
    env.reset(seed=0)
    a = env.action_space.sample()
    for _ in range(50):
        env.step(a)
    t = time.time()
    for _ in range(n):
        _, _, te, tr, _ = env.step(a)
        if te or tr:
            env.reset()
    el = time.time() - t
    env.close()
    return n / el


def bench_train(n_envs: int, threads: int, steps: int, net_arch, batch_size):
    """End-to-end training throughput: simulators + gradient updates."""
    from stable_baselines3 import SAC
    from piper_rl.scripts.train import build_vec_env

    torch.set_num_threads(threads)
    cfg = EnvConfig()
    venv = build_vec_env(cfg, n_envs, seed=0, subproc=n_envs > 1)
    try:
        model = SAC("MlpPolicy", venv, verbose=0, device="cpu",
                    buffer_size=60_000, batch_size=batch_size,
                    learning_starts=200, gamma=0.98, tau=0.01,
                    train_freq=(1, "step"), gradient_steps=n_envs,
                    policy_kwargs=dict(net_arch=list(net_arch)))
        model.learn(max(600, n_envs * 100))              # warm up
        t = time.time()
        model.learn(steps, reset_num_timesteps=False)
        el = time.time() - t
        return steps / el
    finally:
        venv.close()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-envs", type=int, nargs="+", default=None)
    p.add_argument("--threads", type=int, nargs="+", default=None)
    p.add_argument("--steps", type=int, default=1500,
                   help="measured steps per configuration")
    p.add_argument("--quick", action="store_true", help="fewer configurations")
    p.add_argument("--net-arch", type=int, nargs="+", default=[256, 256])
    p.add_argument("--batch-size", type=int, default=256)
    args = p.parse_args(argv)

    physical, logical = cpu_info()
    print("=" * 68)
    print(f"CPU: {physical} physical cores / {logical} logical")
    print(f"torch default threads: {torch.get_num_threads()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print("=" * 68)

    sim = bench_env_only()
    print(f"\nsimulator alone, 1 env, no learning: {sim:6.0f} steps/s "
          f"({1000/sim:.2f} ms per step)")
    print(f"theoretical ceiling with N simulators: ~{sim:.0f} x N\n")

    if args.n_envs is None:
        cands = [1, 2, 4, 8, 12, 16, 24]
        args.n_envs = [n for n in cands if n <= max(2, physical)]
        if args.quick:
            args.n_envs = args.n_envs[:4]
    if args.threads is None:
        cands = [1, 2, 4, 8, 16]
        args.threads = [t for t in cands if t <= max(1, physical)]
        if args.quick:
            args.threads = args.threads[:3]

    print(f"measuring {len(args.n_envs) * len(args.threads)} configurations "
          f"x {args.steps} steps  (this takes a few minutes)\n")
    print("  end-to-end training throughput [env steps/s]")
    print("  n_envs \\ torch threads " + "".join(f"{t:>9d}" for t in args.threads))

    best = (0.0, None)
    for n in args.n_envs:
        row = []
        for t in args.threads:
            try:
                r = bench_train(n, t, args.steps, args.net_arch, args.batch_size)
            except Exception as e:                       # pragma: no cover
                print(f"\n  n_envs={n} threads={t} failed: {type(e).__name__}: {e}")
                r = float("nan")
            row.append(r)
            if r == r and r > best[0]:
                best = (r, (n, t))
        print(f"  {n:>6d}                " + "".join(f"{v:>9.0f}" for v in row))

    rate, (n, t) = best[0], best[1]
    print("\n" + "=" * 68)
    print(f"FASTEST: --n-envs {n} --torch-threads {t}   ->  {rate:.0f} env steps/s")
    for total in (500_000, 1_000_000, 2_000_000):
        h = total / rate / 3600
        print(f"    {total:>9,} steps  ~ {h:5.1f} h")
    print("=" * 68)
    print(f"""
Recommended command:

    python -m piper_rl.scripts.train --algo sac --timesteps 2000000 \\
           --run-name sac_full --n-envs {n} --torch-threads {t} \\
           --gradient-steps {n} --batch-size {args.batch_size} \\
           --eval-freq 25000 --n-eval-episodes 20 --checkpoint-freq 50000

`--gradient-steps {n}` keeps one gradient update per environment step, which is
the standard SAC ratio. Halving it roughly doubles wall-clock throughput at some
cost in sample efficiency -- worth trying if you are impatient.
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())

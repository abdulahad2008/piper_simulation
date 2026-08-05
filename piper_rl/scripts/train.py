"""Train a pick-and-place policy for the AgileX PIPER.

Algorithm: **SAC** by default.

Why SAC for this task
---------------------
* The action space is continuous (Cartesian TCP deltas + gripper). That rules
  out DQN-family methods immediately.
* Each environment step costs a 25-substep MuJoCo rollout (~2 ms). Sample
  efficiency therefore dominates wall-clock. SAC is off-policy with a replay
  buffer and does ~1 gradient update per environment step, so it reaches a
  working policy in ~10^5-10^6 steps. PPO is on-policy and throws every sample
  away after one epoch; on manipulation benchmarks it typically needs 10-50x
  more environment interaction for the same result.
* Contact-rich manipulation has a nasty exploration problem: the reward is flat
  until the fingers happen to close on the object. SAC's maximum-entropy
  objective with an automatically tuned temperature keeps the policy stochastic
  exactly as long as it needs to be, then anneals. TD3 relies on a fixed
  Gaussian exploration noise that you have to hand-schedule.
* TD3 is available here (``--algo td3``) as the closest alternative and is worth
  trying if SAC's entropy term makes the final policy too jittery for hardware;
  PPO (``--algo ppo``) is included for completeness and because it parallelises
  better if you have many cores and few GPUs.

Examples
--------
    # quick smoke test
    python -m piper_rl.scripts.train --timesteps 20000 --run-name smoke

    # the real thing
    python -m piper_rl.scripts.train --timesteps 1500000 --n-envs 4 \
        --run-name sac_v1

    # resume
    python -m piper_rl.scripts.train --timesteps 500000 \
        --load runs/sac_v1/checkpoints/piper_600000_steps.zip --run-name sac_v1b
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import multiprocessing as mp

import numpy as np
import torch

from stable_baselines3 import SAC, TD3, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (DummyVecEnv, SubprocVecEnv,
                                              VecNormalize)
from stable_baselines3.common.callbacks import (CheckpointCallback, EvalCallback,
                                                CallbackList)
from stable_baselines3.common.noise import NormalActionNoise

from piper_rl.config import EnvConfig
from piper_rl.piper_env import PiperPickPlaceEnv
from piper_rl.human_config import HumanAwareEnvConfig
from piper_rl.human_aware_env import PiperHumanAwarePickPlaceEnv
from piper_rl.callbacks import (TaskMetricsCallback, PeriodicDumpCallback,
                                HumanCurriculumCallback)

ALGOS = {"sac": SAC, "td3": TD3, "ppo": PPO}


# --------------------------------------------------------------------------- #
def make_env(cfg: EnvConfig, seed: int, rank: int = 0, monitor_dir=None):
    def _init():
        c = copy.deepcopy(cfg)
        env_cls = (PiperHumanAwarePickPlaceEnv
                   if isinstance(c, HumanAwareEnvConfig) else PiperPickPlaceEnv)
        env = env_cls(c)
        env.reset(seed=seed + rank)
        env.action_space.seed(seed + rank)
        path = None if monitor_dir is None else str(Path(monitor_dir) / f"m{rank}")
        keywords = (("is_success", "human_collision", "collision_free_success")
                    if isinstance(c, HumanAwareEnvConfig) else ("is_success",))
        return Monitor(env, filename=path, info_keywords=keywords)
    return _init


def build_vec_env(cfg: EnvConfig, n_envs: int, seed: int, monitor_dir=None,
                  subproc: bool = True):
    fns = [make_env(cfg, seed, i, monitor_dir) for i in range(n_envs)]
    if n_envs > 1 and subproc:
        # "fork" does not exist on Windows -- hard-coding it made --n-envs > 1
        # crash there with `ValueError: cannot find context for 'fork'`.
        # Let SB3 pick the platform default: fork on Linux/macOS, spawn on
        # Windows. Spawn re-imports this module in each worker, which is why
        # train.py keeps everything behind `if __name__ == "__main__"`.
        method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
        return SubprocVecEnv(fns, start_method=method)
    return DummyVecEnv(fns)


# --------------------------------------------------------------------------- #
def default_hyperparams(algo: str, n_envs: int, net_arch=None,
                        batch_size: int | None = None,
                        gradient_steps: int | None = None) -> dict:
    """Hyper-parameters, with the reasoning for the non-default ones.

    The defaults are sized for a CPU box. On CPU the gradient update, not the
    simulator, is the bottleneck: [256,256] with batch 256 runs at ~55 env
    steps/s on 2 cores, while [512,512,256] with batch 512 drops to ~16.
    If you have a GPU, ``--net-arch 512 512 256 --batch-size 512`` is the
    better setting and costs almost nothing there.
    """
    net_arch = list(net_arch) if net_arch else [256, 256]
    if algo == "sac":
        return dict(
            learning_rate=3e-4,
            buffer_size=400_000,            # ~1600 episodes of 250 steps
            batch_size=batch_size or 256,
            tau=0.01,
            gamma=0.98,                     # 250-step episodes: an effective
                                            # horizon of ~50 steps is plenty and
                                            # a lower gamma sharply reduces
                                            # critic variance
            train_freq=(1, "step"),
            gradient_steps=gradient_steps or max(1, n_envs),
            learning_starts=5_000,          # random play fills the buffer first
            ent_coef="auto_0.1",            # start hot, let it anneal itself
            target_entropy="auto",
            use_sde=False,
            policy_kwargs=dict(net_arch=net_arch),
        )
    if algo == "td3":
        return dict(
            learning_rate=3e-4, buffer_size=400_000,
            batch_size=batch_size or 256, tau=0.01, gamma=0.98,
            train_freq=(1, "step"),
            gradient_steps=gradient_steps or max(1, n_envs),
            learning_starts=5_000, policy_delay=2,
            policy_kwargs=dict(net_arch=net_arch),
        )
    return dict(                            # ppo
        learning_rate=3e-4, n_steps=1024, batch_size=batch_size or 256,
        n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        ent_coef=0.001,
        policy_kwargs=dict(net_arch=dict(pi=net_arch, vf=net_arch)),
    )


# --------------------------------------------------------------------------- #
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algo", choices=list(ALGOS), default="sac")
    p.add_argument("--task", choices=["pick-place", "human-aware"],
                   default="pick-place", help="environment task; default unchanged")
    p.add_argument("--human-preset",
                   choices=["fixed", "randomized", "curriculum"],
                   default="randomized")
    p.add_argument("--human-difficulty", type=float, default=1.0)
    p.add_argument("--no-human-state", action="store_true",
                   help="keep the original policy observation shape")
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--n-envs", type=int, default=1,
                   help="parallel environments; SAC/TD3 like 1-4, PPO likes 8+")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--out", type=str, default="runs")
    p.add_argument("--load", type=str, default=None,
                   help="checkpoint .zip to resume from")
    p.add_argument("--resume", type=str, default=None,
                   help="resume a run directory, e.g. runs/sac_full. Picks up "
                        "final_model.zip AND its replay buffer, and continues "
                        "logging into the same run so the curves stay one line.")
    p.add_argument("--no-save-buffer", action="store_true",
                   help="do not save the replay buffer on exit (saves ~200 MB "
                        "of disk, but makes resuming much less effective)")
    p.add_argument("--eval-freq", type=int, default=20_000,
                   help="env steps between deterministic evaluations")
    p.add_argument("--n-eval-episodes", type=int, default=15)
    p.add_argument("--checkpoint-freq", type=int, default=50_000)
    p.add_argument("--action-mode", choices=["cartesian", "joint"],
                   default="cartesian")
    p.add_argument("--no-curriculum", action="store_true",
                   help="always start episodes from the true initial state")
    p.add_argument("--no-domain-rand", action="store_true")
    p.add_argument("--no-noise", action="store_true")
    p.add_argument("--max-episode-steps", type=int, default=None)
    # --- precision / hard-corner fine-tuning (all no-ops if unset) --------- #
    p.add_argument("--place-tol-xy", type=float, default=None,
                   help="m, radius that counts as 'placed'. Default 0.045. "
                        "Tighten it to trade success rate for precision. NOTE: "
                        "this changes the success DEFINITION, so evaluate "
                        "against the old value too before comparing runs.")
    p.add_argument("--w-precision", type=float, default=None,
                   help="one-off bonus at success, scaled by how far inside the "
                        "tolerance the object landed. Default 0 (off). 60 is a "
                        "reasonable value next to bonus_success=200.")
    p.add_argument("--hard-corner-frac", type=float, default=None,
                   help="fraction of episodes spawned in the outer-radius / "
                        "far-azimuth corner where the failures live. Default 0.")
    p.add_argument("--device", default="auto")
    p.add_argument("--net-arch", type=int, nargs="+", default=None,
                   help="hidden layer sizes, e.g. --net-arch 512 512 256")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--gradient-steps", type=int, default=None,
                   help="gradient updates per vec-env step (SAC/TD3)")
    p.add_argument("--torch-threads", type=int, default=None,
                   help="threads for the gradient update. On a many-core box "
                        "leave ~n_envs cores free for the simulators; see "
                        "scripts/benchmark.py")
    args = p.parse_args(argv)

    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)

    # ---- --resume: continue an existing run in place --------------------- #
    buffer_path = None
    if args.resume:
        rdir = Path(args.resume)
        if not rdir.exists():
            p.error(f"--resume: {rdir} does not exist")
        ckpt = rdir / "final_model.zip"
        if not ckpt.exists():
            snaps = sorted(rdir.glob("checkpoints/*_steps.zip"),
                           key=lambda q: int(q.stem.split("_")[-2]))
            if not snaps:
                p.error(f"--resume: no final_model.zip or checkpoint in {rdir}")
            ckpt = snaps[-1]
        args.load = str(ckpt)
        args.run_name = rdir.name
        args.out = str(rdir.parent)
        cand = rdir / "replay_buffer.pkl"
        buffer_path = cand if cand.exists() else None
        print(f"resuming from {ckpt}")
        if buffer_path:
            print(f"replay buffer: {buffer_path}")
        else:
            print("replay buffer: NOT FOUND -- the networks carry over but the "
                  "experience does not; expect a dip while it refills")

    run = args.run_name or f"{args.algo}_{time.strftime('%Y%m%d_%H%M%S')}"
    out = Path(args.out) / run
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "best").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ cfg
    if args.task == "human-aware":
        cfg = HumanAwareEnvConfig.from_preset(
            args.human_preset, difficulty=args.human_difficulty)
        cfg.include_human_state = not args.no_human_state
    else:
        cfg = EnvConfig()
    cfg.action_mode = args.action_mode
    cfg.curriculum = not args.no_curriculum
    cfg.domain_rand.enabled = not args.no_domain_rand
    cfg.noise.enabled = not args.no_noise
    if args.max_episode_steps:
        cfg.max_episode_steps = args.max_episode_steps
    if args.place_tol_xy is not None:
        cfg.reward.place_tol_xy = args.place_tol_xy
        print(f"place tolerance: {args.place_tol_xy*1000:.0f} mm "
              f"(success is now scored against this, not 45 mm)")
    if args.w_precision is not None:
        cfg.reward.w_precision = args.w_precision
    if args.hard_corner_frac is not None:
        cfg.domain_rand.hard_corner_frac = args.hard_corner_frac

    with open(out / "config.json", "w") as f:
        json.dump({"args": vars(args), "env": cfg.to_dict()}, f, indent=2,
                  default=str)

    # ------------------------------------------------------------------ envs
    train_env = build_vec_env(cfg, args.n_envs, args.seed,
                              monitor_dir=str(out), subproc=args.n_envs > 1)
    # Evaluation env: curriculum OFF (always the true initial state), same
    # dynamics randomisation, different seeds.
    eval_cfg = cfg.eval_variant()
    eval_env = build_vec_env(eval_cfg, 1, args.seed + 10_000, subproc=False)

    # ------------------------------------------------------------------ model
    Algo = ALGOS[args.algo]
    hp = default_hyperparams(args.algo, args.n_envs, args.net_arch,
                             args.batch_size, args.gradient_steps)
    if args.algo == "td3":
        n_act = train_env.action_space.shape[0]
        hp["action_noise"] = NormalActionNoise(np.zeros(n_act),
                                               0.1 * np.ones(n_act))

    if args.load:
        model = Algo.load(args.load, env=train_env, device=args.device,
                          tensorboard_log=str(out / "tb"))
        # SAC/TD3 keep their experience in a replay buffer that is NOT inside
        # the .zip (it is ~200 MB). Without it the networks resume but the
        # buffer starts empty, the first `learning_starts` steps are random
        # again, and performance dips for tens of thousands of steps. Reload it.
        if buffer_path is not None and hasattr(model, "load_replay_buffer"):
            model.load_replay_buffer(str(buffer_path))
            print(f"replay buffer restored: {model.replay_buffer.size():,} "
                  f"transitions")
        # Algo.load() restores EVERY hyper-parameter from the checkpoint, so
        # flags passed on a resume are otherwise silently ignored. Re-apply the
        # ones the user asked for explicitly, and say so.
        for attr, val in (("gradient_steps", args.gradient_steps),
                          ("batch_size", args.batch_size),
                          ("learning_rate", args.learning_rate)):
            if val is not None and getattr(model, attr, None) != val:
                print(f"  override: {attr} {getattr(model, attr, '?')} -> {val}")
                setattr(model, attr, val)
        if args.learning_rate is not None:
            model.lr_schedule = lambda _: args.learning_rate
            model._setup_lr_schedule()
        if model.n_envs != args.n_envs:
            print(f"  note: checkpoint was trained with {model.n_envs} env(s), "
                  f"now running {args.n_envs}. This is fine -- the replay "
                  f"buffer is shared -- but the effective gradient-to-env-step "
                  f"ratio changes unless you scale --gradient-steps with it.")
    else:
        model = Algo("MlpPolicy", train_env, seed=args.seed, verbose=1,
                     device=args.device, tensorboard_log=str(out / "tb"), **hp)

    # ------------------------------------------------------------- callbacks
    metrics_cb = TaskMetricsCallback(window=50)
    callback_items = [
        metrics_cb,
        PeriodicDumpCallback(every=2000, metrics=metrics_cb),
        CheckpointCallback(
            save_freq=max(1, args.checkpoint_freq // args.n_envs),
            save_path=str(out / "checkpoints"), name_prefix="piper",
            save_replay_buffer=False, save_vecnormalize=False),
        EvalCallback(
            eval_env,
            best_model_save_path=str(out / "best"),
            log_path=str(out / "eval"),
            eval_freq=max(1, args.eval_freq // args.n_envs),
            n_eval_episodes=args.n_eval_episodes,
            deterministic=True, render=False, verbose=1),
    ]
    if args.task == "human-aware" and args.human_preset == "curriculum":
        callback_items.append(HumanCurriculumCallback(
            total_timesteps=args.timesteps,
            start=args.human_difficulty, end=1.0))
    cbs = CallbackList(callback_items)

    print(f"\n{'='*70}\n{args.algo.upper()}  |  {args.timesteps:,} steps  |  "
          f"{args.n_envs} env(s)  |  obs {train_env.observation_space.shape}  |  "
          f"act {train_env.action_space.shape}\n"
          f"run dir: {out}\n{'='*70}\n")

    t0 = time.time()
    try:
        model.learn(total_timesteps=args.timesteps, callback=cbs,
                    log_interval=10, progress_bar=False,
                    reset_num_timesteps=args.load is None)
    except KeyboardInterrupt:
        print("\ninterrupted -- saving current model")
    finally:
        model.save(str(out / "final_model"))
        if not args.no_save_buffer and hasattr(model, "save_replay_buffer"):
            try:
                model.save_replay_buffer(str(out / "replay_buffer"))
                print(f"replay buffer saved: {out/'replay_buffer.pkl'} "
                      f"({model.replay_buffer.size():,} transitions)")
            except Exception as e:                       # pragma: no cover
                print(f"could not save the replay buffer: {e}")
        print(f"\nsaved {out/'final_model'}.zip  "
              f"({time.time()-t0:.0f} s, {args.timesteps} steps)")
        print(f"best model (by eval reward): {out/'best'/'best_model.zip'}")
        train_env.close()
        eval_env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

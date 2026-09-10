"""The 51-run training campaign of the precision-floor paper, as one command.

Every cell is a dict of ``train.py`` flags plus a seed. The runner is
resumable (a cell whose ``final_model.zip`` exists is skipped), it writes a
manifest so a half-finished campaign can be reported honestly, and it
evaluates each finished cell immediately so the CSV exists even if the machine
dies later.

    # what would run, and how long it should take
    python -m piper_rl.scripts.run_sweep --dry-run

    # phase 1 only, seeds 0-2, 6 h per run
    python -m piper_rl.scripts.run_sweep --phase 1 --seeds 0 1 2

Design notes that are easy to get wrong and are therefore encoded here rather
than left to the operator:

* The Delta_max axis scales ``max_joint_vel`` proportionally, because at the
  released settings the joint-velocity clamp binds on most steps; a step-size
  sweep with a fixed clamp measures the clamp.
* A separate qdot_max axis at Delta_max = 30 mm separates the two.
* The 30 mm cell is retrained fresh rather than reusing the released policy,
  so every cell of the axis shares one protocol (same seeds, same budget, same
  hard_corner_frac, same selection rule).
* ``hard_corner_frac`` is 0.35 from the start in every cell, so there is no
  fine-tune stage and no cell differs from another by training history.
* The FINAL checkpoint is the reported one, never the best: best-checkpoint
  selection on a small eval callback is noise and biases the number upward.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]

#: theta_0, the released interface.
THETA0 = {"max_step_dist": 0.030, "action_smoothing": 0.45,
          "control_hz": 20.0, "max_joint_vel": 1.0}

COMMON = ["--algo", "sac", "--timesteps", "1200000", "--n-envs", "12",
          "--gradient-steps", "12", "--hard-corner-frac", "0.35",
          "--eval-freq", "20000", "--n-eval-episodes", "50",
          "--checkpoint-freq", "50000"]


def cells(phase: int | None) -> list[dict]:
    out: list[dict] = []

    def add(name, phase_, flags, purpose):
        out.append({"name": name, "phase": phase_, "flags": flags,
                    "purpose": purpose})

    # ---- phase 0: retrained solver extremes (only if the eval-only gate moves)
    add("solver_extremeA", 0,
        {**THETA0, "timestep": 0.001, "solref_scale": 0.5},
        "H0d retrain, stiff/fine extreme")
    add("solver_extremeB", 0,
        {**THETA0, "timestep": 0.0025, "solref_scale": 2.0},
        "H0d retrain, soft/coarse extreme (2.5 ms, not 4 ms: 4 ms is 12.5 "
        "sub-steps at 20 Hz and silently rounds to 12)")

    # ---- phase 1: the interface sweep -------------------------------------
    add("theta0", 1, dict(THETA0), "shared baseline for every axis")
    for dmax, qd in [(0.020, 0.67), (0.010, 0.33), (0.005, 0.17)]:
        add(f"dmax{int(dmax*1000)}mm", 1,
            {**THETA0, "max_step_dist": dmax, "max_joint_vel": qd},
            "step axis, clamp scaled so its relative bite is constant")
    for a in (0.2, 1.0):
        add(f"alpha{a}", 1, {**THETA0, "action_smoothing": a}, "smoothing axis")
    for qd in (0.5, 2.0):
        add(f"qdmax{qd}", 1, {**THETA0, "max_joint_vel": qd},
            "clamp axis at Delta_max = 30 mm")
    add("f10hz", 1, {**THETA0, "control_hz": 10.0}, "rate axis")
    # At 50 Hz, 30 mm/step is 1.5 m/s > max_tcp_speed 0.6, so the TCP rate
    # limiter would fire on every command. Raise the cap for this cell and say
    # so in the paper rather than letting a second limiter enter silently.
    add("f50hz", 1, {**THETA0, "control_hz": 50.0, "max_tcp_speed": 1.6},
        "rate axis; max_tcp_speed raised to 1.6 m/s so the Cartesian rate "
        "limiter does not bind (reported)")

    # ---- phase 2: the control arms ----------------------------------------
    add("reward_wprec60", 2, {**THETA0, "w_precision": 60.0},
        "H0a, from scratch so reward and budget are not confounded")
    add("budget3x", 2, {**THETA0, "timesteps": 3_600_000}, "H0b")
    add("fourier", 2, {**THETA0, "fourier_features": 64, "fourier_sigma": 1.0},
        "H0c")

    return [c for c in out if phase is None or c["phase"] == phase]


#: The second environment. Same axes, same seeds, same selection rule; a much
#: cheaper cell (~63 fps on two CPU cores against the Piper's 26-62 fps with
#: 12 gradient steps, and a 300 k-step budget rather than 1.2 M), so the
#: cross-environment check costs a fraction of the main sweep.
GYMHIL_THETA0 = {"max_step_dist": 0.025, "action_smoothing": 1.0,
                 "control_hz": 10.0}


def gymhil_cells() -> list[dict]:
    out = []

    def add(name, flags, purpose):
        out.append({"name": name, "phase": 1, "flags": flags,
                    "purpose": purpose, "env": "gymhil"})

    add("gh_theta0", dict(GYMHIL_THETA0), "gym-hil baseline at its released interface")
    for dmax in (0.005, 0.010, 0.050):
        add(f"gh_dmax{int(dmax*1000)}mm", {**GYMHIL_THETA0, "max_step_dist": dmax},
            "step axis in the second environment")
    for a in (0.2, 0.45):
        add(f"gh_alpha{a}", {**GYMHIL_THETA0, "action_smoothing": a},
            "smoothing axis; gym-hil ships none, so this axis is added by us")
    for hz in (5.0, 25.0):
        add(f"gh_f{int(hz)}hz", {**GYMHIL_THETA0, "control_hz": hz}, "rate axis")
    return out


def build_cmd(cell: dict, seed: int, out_root: str) -> list[str]:
    flags = list(COMMON)
    for k, v in cell["flags"].items():
        if k == "timesteps":
            flags[flags.index("--timesteps") + 1] = str(int(v))
            continue
        flags += [f"--{k.replace('_', '-')}", str(v)]
    tag = f"{cell['name']}_s{seed}"
    return ([sys.executable, "-m", "piper_rl.scripts.train",
             "--seed", str(seed), "--run-name", tag, "--out", out_root]
            + flags), tag


def _report(todo, args):
    print(f"{len(todo)} training runs")
    for t in todo:
        root = "runs/gymhil" if t.get("env") == "gymhil" else args.out
        done = (Path(root) / t["tag"] / "final_model.zip").exists()
        print(f"  [{'done' if done else '    '}] {t['tag']:24s} {t['purpose']}")


def _run(todo, args):
    import json as _json
    manifest = Path("runs/gymhil") / "manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    log = []
    for t in todo:
        final = Path("runs/gymhil") / t["tag"] / "final_model.zip"
        if not final.exists():
            print(f"\n=== training {t['tag']} ===", flush=True)
            t["returncode"] = subprocess.call(t["cmd"], cwd=str(PROJECT_DIR))
        csv = Path(args.results) / f"{t['tag']}.csv"
        if not csv.exists() and final.exists():
            subprocess.call([sys.executable, "-m",
                             "piper_rl.scripts.eval_precision",
                             "--env", "gymhil-pick", "--model", str(final),
                             "--episodes", str(args.eval_episodes),
                             "--workers", str(args.eval_workers),
                             "--tag", t["tag"], "--out-dir", args.results],
                            cwd=str(PROJECT_DIR))
        log.append(t)
        manifest.write_text(_json.dumps(log, indent=2))
    print(f"\nmanifest: {manifest}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", type=int, default=None, choices=[0, 1, 2])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--out", type=str, default="runs/sweep")
    p.add_argument("--results", type=str, default="results/precision")
    p.add_argument("--eval-episodes", type=int, default=2000)
    p.add_argument("--eval-workers", type=int, default=2)
    p.add_argument("--env", choices=["piper", "gymhil", "both"], default="piper",
                   help="which environment's cells to run. 'gymhil' is the "
                        "second environment (Franka, operational-space "
                        "control) and is far cheaper per run.")
    p.add_argument("--gymhil-timesteps", type=int, default=300_000)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--hours-per-run", type=float, default=6.0)
    args = p.parse_args(argv)

    todo = []
    if args.env in ("gymhil", "both"):
        for cell in gymhil_cells():
            for seed in args.seeds:
                tag = f"{cell['name']}_s{seed}"
                cmd = [sys.executable, "-m", "piper_rl.scripts.train_gymhil",
                       "--tag", tag, "--seed", str(seed),
                       "--timesteps", str(args.gymhil_timesteps),
                       "--out", "runs/gymhil"]
                for k, v in cell["flags"].items():
                    cmd += [f"--{k.replace('_', '-')}", str(v)]
                todo.append({"tag": tag, "cell": cell["name"], "seed": seed,
                             "purpose": cell["purpose"], "cmd": cmd,
                             "env": "gymhil"})
    if args.env == "gymhil":
        _report(todo, args)
        if args.dry_run:
            return 0
        return _run(todo, args)

    for cell in cells(args.phase):
        seeds = [0] if cell["name"] == "budget3x" else args.seeds
        for seed in seeds:
            cmd, tag = build_cmd(cell, seed, args.out)
            todo.append({"tag": tag, "cell": cell["name"], "seed": seed,
                         "purpose": cell["purpose"], "cmd": cmd})

    print(f"{len(todo)} training runs "
          f"(~{len(todo) * args.hours_per_run:.0f} h sequential at "
          f"{args.hours_per_run:g} h/run)")
    for t in todo:
        done = (Path(args.out) / t["tag"] / "final_model.zip").exists()
        print(f"  [{'done' if done else '    '}] {t['tag']:24s} {t['purpose']}")
    if args.dry_run:
        return 0

    manifest = Path(args.out) / "manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    log = []
    for t in todo:
        final = Path(args.out) / t["tag"] / "final_model.zip"
        if not final.exists():
            print(f"\n=== training {t['tag']} ===", flush=True)
            t0 = time.time()
            rc = subprocess.call(t["cmd"], cwd=str(PROJECT_DIR))
            t["train_seconds"] = round(time.time() - t0, 1)
            t["returncode"] = rc
            if rc != 0:
                print(f"!! {t['tag']} failed (rc={rc}); continuing")
                log.append(t)
                manifest.write_text(json.dumps(log, indent=2))
                continue
        # evaluate the FINAL model, never the best -- see the module docstring.
        csv = Path(args.results) / f"{t['tag']}.csv"
        if not csv.exists() and final.exists():
            print(f"=== evaluating {t['tag']} ===", flush=True)
            subprocess.call([sys.executable, "-m",
                             "piper_rl.scripts.eval_precision",
                             "--model", str(final),
                             "--episodes", str(args.eval_episodes),
                             "--workers", str(args.eval_workers),
                             "--tag", t["tag"], "--out-dir", args.results],
                            cwd=str(PROJECT_DIR))
        log.append(t)
        manifest.write_text(json.dumps(log, indent=2))
    print(f"\nmanifest: {manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

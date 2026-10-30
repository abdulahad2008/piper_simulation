"""Learning curves at equal wall clock for the Stage A / Stage B benchmark.

Reads <runs>/<config>/eval/wallclock.json for every config, draws validation
success and mean reward against wall-clock hours (and against steps), and
prints a table: steps reached in the budget, env steps per second, first
time the validation success reached 50 % and 80 %, best success.

    python -m piper_rl.scripts.plot_stage_a --runs runs/stage_a --out runs/stage_a/stage_a_curves.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(runs: Path) -> dict[str, list[dict]]:
    out = {}
    for d in sorted(runs.iterdir()):
        f = d / "eval" / "wallclock.json"
        if f.exists():
            rows = json.loads(f.read_text())
            if rows:
                out[d.name] = rows
    return out


def first_time(rows, key, thr):
    for r in rows:
        if r.get(key, 0.0) >= thr:
            return r["wall_s"] / 3600, r["timesteps"]
    return None, None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", default="runs/stage_a")
    p.add_argument("--out", default=None)
    p.add_argument("--marks", type=float, nargs="+", default=[0.5, 0.8])
    args = p.parse_args(argv)
    runs = Path(args.runs)
    data = load(runs)
    if not data:
        raise SystemExit(f"no eval/wallclock.json under {runs}")
    print(f"{'config':<18}{'hours':>7}{'steps':>10}{'steps/s':>9}" + "".join(f"{'t@'+str(int(m*100))+'%':>9}" for m in args.marks) + f"{'best':>7}")
    summary = {}
    for name, rows in data.items():
        hours = rows[-1]["wall_s"] / 3600
        steps = rows[-1]["timesteps"]
        sps = steps / rows[-1]["wall_s"]
        best = max(r.get("success_rate", 0.0) for r in rows)
        marks = [first_time(rows, "success_rate", m)[0] for m in args.marks]
        summary[name] = dict(hours=hours, steps=steps, steps_per_s=sps, best_success=best,
                             **{f"hours_to_{int(m*100)}": t for m, t in zip(args.marks, marks)})
        print(f"{name:<18}{hours:>7.2f}{steps:>10,}{sps:>9.1f}" + "".join(f"{(f'{t:.2f}' if t is not None else '-'):>9}" for t in marks) + f"{best*100:>6.0f}%")
    out = Path(args.out) if args.out else runs / "curves.png"
    (out.with_suffix(".json")).write_text(json.dumps(summary, indent=1))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing; table only")
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for name, rows in data.items():
        h = np.array([r["wall_s"] for r in rows]) / 3600
        st = np.array([r["timesteps"] for r in rows])
        sr = np.array([r.get("success_rate", np.nan) for r in rows]) * 100
        rw = np.array([r["mean_reward"] for r in rows])
        axes[0].plot(h, sr, marker=".", label=name)
        axes[1].plot(st / 1e6, sr, marker=".", label=name)
        axes[2].plot(h, rw, marker=".", label=name)
    axes[0].set(xlabel="wall clock (h)", ylabel="validation success (%)", title="success vs wall clock (the benchmark)")
    axes[1].set(xlabel="environment steps (M)", ylabel="validation success (%)", title="success vs steps (sample efficiency)")
    axes[2].set(xlabel="wall clock (h)", ylabel="mean validation reward", title="reward vs wall clock")
    for ax in axes:
        ax.grid(alpha=0.3)
    for m in args.marks:
        axes[0].axhline(m * 100, color="k", lw=0.8, ls=":")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print("written", out, "and", out.with_suffix(".json"))


if __name__ == "__main__":
    main()

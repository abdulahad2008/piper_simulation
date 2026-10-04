"""Precision as a function of training steps -- the RL analogue of Xu et al.

Every stored checkpoint of a run is evaluated with the same protocol, giving a
precision-vs-environment-steps curve at zero training cost. It is the RL
counterpart of the data-scaling curve of Xu et al. (ICRA 2026) with
environment steps in place of demonstrations, and it is also the cheapest
possible test of H0b (budget): if the curve is flat over the last 1 M steps,
training longer does not move the floor.

    python -m piper_rl.scripts.eval_checkpoints \
        --run runs/sac_full --episodes 400 --workers 2 --out-dir results/precision
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from piper_rl.scripts import eval_precision

STEP_RE = re.compile(r"(\d+)_steps\.zip$")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=str, required=True,
                   help="run directory containing checkpoints/")
    p.add_argument("--episodes", type=int, default=400)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=30_000)
    p.add_argument("--every", type=int, default=1,
                   help="evaluate every Nth checkpoint (thinning)")
    p.add_argument("--out-dir", type=str, default="results/precision")
    args = p.parse_args(argv)

    run = Path(args.run)
    ckpts = sorted(run.glob("checkpoints/*_steps.zip"),
                   key=lambda q: int(STEP_RE.search(q.name).group(1)))
    ckpts = ckpts[::args.every]
    print(f"{len(ckpts)} checkpoints from {run}")

    for ck in ckpts:
        steps = int(STEP_RE.search(ck.name).group(1))
        tag = f"ckpt_{run.name}_{steps:08d}"
        if (Path(args.out_dir) / f"{tag}.csv").exists():
            print(f"  skip {tag} (already done)")
            continue
        eval_precision.main([
            "--model", str(ck), "--episodes", str(args.episodes),
            "--workers", str(args.workers), "--seed", str(args.seed),
            "--tag", tag, "--out-dir", args.out_dir])
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Markdown tables (with 95 % Wilson intervals) from reaction_baselines summaries.

    python -m piper_rl.scripts.reaction_table docs/sharp_tools/reactions > docs/sharp_tools/reactions/TABLES.md
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def wilson(p_pct: float, n: int) -> str:
    p = p_pct / 100.0
    z = 1.96
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return f"{p_pct:.0f} ({100*(centre-half):.0f}–{100*(centre+half):.0f})"


def main(d: Path) -> None:
    for s in ("exact", "random", "ood"):
        f = d / f"reactions_{s}_summary.json"
        if not f.exists():
            continue
        j = json.loads(f.read_text())
        n = j["episodes_per_cell"]
        print(f"\n### {s} set (preset `{j['preset']}`, {n} episodes per cell, seeds from {j['seed0']})\n")
        print("| reaction | person | success % (95 % CI) | done any time % | intrusion before done % | hot contact % (95 % CI) | handle % | robot body % | exposure 1 s after intrusion (s) | p5 hot distance (mm) | time to done (s) | reaction time (s) |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for k, c in j["cells"].items():
            r, p = k.split("/")
            rt = c["mean_reaction_time_s"]
            rt = f"{rt:.2f}" if isinstance(rt, (int, float)) else "–"
            print(f"| {r} | {p} | {wilson(c['success_pct'], n)} | {c['completed_any_time_pct']:.0f} | {c['intrusion_before_completion_pct']:.0f} | "
                  f"{wilson(c['hot_contact_pct'], n)} | {c['handle_contact_pct']:.0f} | {c['robot_contact_pct']:.0f} | "
                  f"{c['mean_exposure_1s_after_intrusion_s']:.2f} | {c['p5_min_hot_distance_mm']:.0f} | {c['mean_completion_time_s']:.2f} | {rt} |")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "out/reactions"))

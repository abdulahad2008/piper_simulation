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


def num(x) -> str:
    """A mean that may be undefined (no episode completed / no reaction)."""
    return f"{x:.2f}" if isinstance(x, (int, float)) else "–"


def main(d: Path) -> None:
    for s in ("exact", "random", "ood"):
        f = d / f"reactions_{s}_summary.json"
        if not f.exists():
            continue
        j = json.loads(f.read_text())
        n = j["episodes_per_cell"]
        print(f"\n### {s} set (preset `{j['preset']}`, {n} episodes per cell, seeds from {j['seed0']})\n")
        task = j.get("env") == "task"
        if task:
            print(f"Training env (`--env task`), release rule {'on' if j.get('release_when_done') else 'off'}, "
                  f"person window {[round(x, 2) for x in j.get('appearance_window_s')]} s. "
                  "Success = the env's own criterion (which ends a learner's episode); "
                  "'until env done' columns count contacts only up to that end, the others over the full 10 s.\n")
            print("| reaction | person | success % (95 % CI) | intrusion before done % | hot contact % until env done (95 % CI) | hot contact % full episode | robot body % until env done | robot body % full | handle % full | exposure until env done (s) | exposure full (s) | p5 hot distance (mm) | env done at (s) | reaction time (s) |")
            print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
            for k, c in j["cells"].items():
                r, p = k.split("/")
                print(f"| {r} | {p} | {wilson(c['env_success_pct'], n)} | {c['intrusion_before_completion_pct']:.0f} | "
                      f"{wilson(c['hot_contact_pre_done_pct'], n)} | {c['hot_contact_pct']:.0f} | "
                      f"{c['robot_contact_pre_done_pct']:.0f} | {c['robot_contact_pct']:.0f} | {c['handle_contact_pct']:.0f} | "
                      f"{num(c['mean_exposure_pre_done_s'])} | {c['mean_exposure_total_s']:.2f} | {c['p5_min_hot_distance_mm']:.0f} | "
                      f"{num(c['mean_env_done_time_s'])} | {num(c['mean_reaction_time_s'])} |")
            continue
        print("| reaction | person | success % (95 % CI) | done any time % | intrusion before done % | hot contact % (95 % CI) | handle % | robot body % | exposure 1 s after intrusion (s) | exposure total (s) | p5 hot distance (mm) | time to done (s) | reaction time (s) |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for k, c in j["cells"].items():
            r, p = k.split("/")
            print(f"| {r} | {p} | {wilson(c['success_pct'], n)} | {c['completed_any_time_pct']:.0f} | {c['intrusion_before_completion_pct']:.0f} | "
                  f"{wilson(c['hot_contact_pct'], n)} | {c['handle_contact_pct']:.0f} | {c['robot_contact_pct']:.0f} | "
                  f"{c['mean_exposure_1s_after_intrusion_s']:.2f} | {c['mean_exposure_total_s']:.2f} | {c['p5_min_hot_distance_mm']:.0f} | "
                  f"{num(c['mean_completion_time_s'])} | {num(c['mean_reaction_time_s'])} |")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "out/reactions"))

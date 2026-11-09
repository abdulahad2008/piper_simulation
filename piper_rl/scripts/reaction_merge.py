"""Merge reaction_baselines outputs run in parts into one directory.

    python -m piper_rl.scripts.reaction_merge OUT PART [PART ...]

Each PART is a directory of `reactions_<set>.csv` + `reactions_<set>_summary.json`.
Cells (reaction/person) from later parts replace those from earlier ones; the
CSV rows follow the cells. The merged summary records where each cell came from.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


def merge(out: Path, parts: list[Path]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for s in ("exact", "random", "ood"):
        summary = None
        cells: dict[str, dict] = {}
        origin: dict[str, str] = {}
        rows: dict[str, list[dict]] = {}
        header = None
        for part in parts:
            f = part / f"reactions_{s}_summary.json"
            if not f.exists():
                continue
            j = json.loads(f.read_text())
            if summary is None:
                summary = {k: v for k, v in j.items() if k != "cells"}
            else:
                for k in ("set", "preset", "episodes_per_cell", "seed0", "env"):
                    assert summary[k] == j[k], (k, summary[k], j[k], part)
            with open(part / f"reactions_{s}.csv", newline="") as fh:
                r = csv.DictReader(fh)
                header = header or r.fieldnames
                part_rows: dict[str, list[dict]] = {}
                for row in r:
                    part_rows.setdefault(f"{row['reaction']}/{row['person_type']}", []).append(row)
            for k, c in j["cells"].items():
                cells[k] = c
                origin[k] = str(part)
                rows[k] = part_rows[k]
                summary["parameters"] = j["parameters"]
        if summary is None:
            continue
        summary["cells"] = dict(sorted(cells.items()))
        summary["cell_origin"] = dict(sorted(origin.items()))
        (out / f"reactions_{s}_summary.json").write_text(json.dumps(summary, indent=1))
        with open(out / f"reactions_{s}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=header)
            w.writeheader()
            for k in sorted(rows):
                for row in rows[k]:
                    w.writerow(row)
        print(f"{s}: {len(cells)} cells from {sorted(set(origin.values()))}")


if __name__ == "__main__":
    merge(Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]])

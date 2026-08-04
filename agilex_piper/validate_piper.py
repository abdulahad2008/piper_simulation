"""SUPERSEDED -- see piper_rl/scripts/validate_model.py.

The original version of this file solved IK for a *full 6-DoF pose* (position
plus a completely specified orientation). The PIPER's joint limits make that
infeasible over most of the workspace, so the solver stalled 50-90 mm and
20-40 degrees from the target and the script used the answer anyway; it always
reported "NEEDS TUNING" because the object never left the table.

The replacement solves a 5-DoF task (position + approach *direction*, roll free)
and runs a full battery of model, API, reward and throughput checks:

    python -m piper_rl.scripts.validate_model
    python -m piper_rl.scripts.validate_model --map

and the scripted pick-and-place that this file was trying to be lives in

    python -m piper_rl.scripts.scripted_demo --episodes 20

Both must be run from the project root (the directory containing piper_rl/).
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    print(__doc__)
    print("running piper_rl.scripts.validate_model ...\n")
    sys.exit(subprocess.call(
        [sys.executable, "-m", "piper_rl.scripts.validate_model", *sys.argv[1:]],
        cwd=str(ROOT)))

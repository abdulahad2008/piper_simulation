#!/usr/bin/env bash
# Phase 0 of the precision-floor paper: everything that needs no training.
set -u
cd "$(dirname "$0")"
E=python3
LOG=results/precision/phase0.log
mkdir -p results/precision
run () { echo "=== $(date -u +%H:%M:%S) $* ===" >> $LOG; $E -m piper_rl.scripts.eval_precision "$@" >> $LOG 2>&1; }

# --- A. the archived baselines (Tables 2-3): v1 and v2 at n=2000 ----------
run --model release/piper_sac_v1_95pct.zip --episodes 2000 --workers 2 --seed 10000 --tag v1_n2000
run --model release/piper_sac_v2_96pct.zip --episodes 2000 --workers 2 --seed 10000 --tag v2_n2000

# --- B. solver kill gate, eval-only, on v2 (n=1000 each) ------------------
run --model release/piper_sac_v2_96pct.zip --episodes 1000 --workers 2 --seed 20000 --tag v2_solver_nominal
run --model release/piper_sac_v2_96pct.zip --episodes 1000 --workers 2 --seed 20000 --timestep 0.001 --tag v2_solver_dt1ms
run --model release/piper_sac_v2_96pct.zip --episodes 1000 --workers 2 --seed 20000 --timestep 0.004 --tag v2_solver_dt4ms
run --model release/piper_sac_v2_96pct.zip --episodes 1000 --workers 2 --seed 20000 --solref-scale 0.5 --tag v2_solver_solref0.5
run --model release/piper_sac_v2_96pct.zip --episodes 1000 --workers 2 --seed 20000 --solref-scale 2.0 --tag v2_solver_solref2
run --model release/piper_sac_v2_96pct.zip --episodes 1000 --workers 2 --seed 20000 --solver-iters 20 --tag v2_solver_iter20
run --model release/piper_sac_v2_96pct.zip --episodes 1000 --workers 2 --seed 20000 --impratio 1.0 --tag v2_solver_imp1
run --model release/piper_sac_v2_96pct.zip --episodes 1000 --workers 2 --seed 20000 --timestep 0.001 --solref-scale 0.5 --tag v2_solver_extremeA
run --model release/piper_sac_v2_96pct.zip --episodes 1000 --workers 2 --seed 20000 --timestep 0.004 --solref-scale 2.0 --tag v2_solver_extremeB

echo "=== $(date -u +%H:%M:%S) PHASE0-AB-DONE ===" >> $LOG

#!/usr/bin/env bash
# Phase 0 part C: precision vs training steps, all stored checkpoints.
set -u
cd "$(dirname "$0")"
LOG=results/precision/phase0.log
while ! grep -q "PHASE0-AB-DONE" $LOG 2>/dev/null; do sleep 20; done
echo "=== $(date -u +%H:%M:%S) checkpoint curve ===" >> $LOG
python3 -m piper_rl.scripts.eval_checkpoints --run runs/sac_full --episodes 1000 \
        --workers 2 --seed 30000 --out-dir results/precision >> $LOG 2>&1
python3 -m piper_rl.scripts.eval_checkpoints --run runs/corner_ft --episodes 1000 \
        --workers 2 --seed 30000 --out-dir results/precision >> $LOG 2>&1
echo "=== $(date -u +%H:%M:%S) PHASE0-C-DONE ===" >> $LOG

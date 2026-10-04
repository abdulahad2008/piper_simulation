#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
LOG=results/precision/phase0.log
while ! grep -q "PHASE0-C-DONE" $LOG 2>/dev/null; do sleep 30; done
echo "=== $(date -u +%H:%M:%S) nested ICC design ===" >> $LOG
python3 -m piper_rl.scripts.eval_precision --model release/piper_sac_v2_96pct.zip \
  --episodes 2000 --nested-noise 10 --workers 2 --seed 40000 \
  --tag v2_nested_200x10 >> $LOG 2>&1
echo "=== $(date -u +%H:%M:%S) PHASE0-D-DONE ===" >> $LOG

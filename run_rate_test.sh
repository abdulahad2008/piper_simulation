#!/usr/bin/env bash
# 20-minute rate test (no learning claim): does SBX SAC with 8 environments and
# 4 updates per vec step (0.5 updates per env step) reach ~90 env steps/s on this
# machine, as the micro-benchmark model predicts? Also runs the 4-env version of
# the same ratio alongside for a same-time comparison.
#
#   bash run_rate_test.sh            # both, 20 min each, then prints steps/s
set -u
OUT="${OUT:-runs/rate_test}"; SECS="${SECS:-1200}"; PY="${PY:-python}"
mkdir -p "$OUT"
COMMON="--task human-aware --human-preset curriculum --timesteps 2000000 --curriculum-ramp-timesteps 1200000 \
  --eval-freq 40000 --n-eval-episodes 20 --checkpoint-freq 1000000 --torch-threads 1 \
  --max-wall-seconds $SECS --seed 0 --no-save-buffer --out $OUT"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 XLA_FLAGS=--xla_cpu_multi_thread_eigen=false \
  nohup $PY -m piper_rl.scripts.train --algo sbx_sac --n-envs 8 --gradient-steps 4 $COMMON --run-name sbx8_g4 > "$OUT/sbx8_g4.log" 2>&1 &
sleep 5
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 XLA_FLAGS=--xla_cpu_multi_thread_eigen=false \
  nohup $PY -m piper_rl.scripts.train --algo sbx_sac --n-envs 4 --gradient-steps 2 $COMMON --run-name sbx4_g2 > "$OUT/sbx4_g2.log" 2>&1 &
wait
$PY - <<'PY'
import json, pathlib
for n in ("sbx8_g4", "sbx4_g2"):
    f = pathlib.Path("runs/rate_test") / n / "eval" / "wallclock.json"
    rows = json.loads(f.read_text()) if f.exists() else []
    if rows:
        r = rows[-1]; print(f"{n}: {r['timesteps']:,} steps in {r['wall_s']/60:.1f} min = {r['timesteps']/r['wall_s']:.1f} steps/s")
    else:
        print(n, "no eval row yet; see the log")
PY

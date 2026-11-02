#!/usr/bin/env bash
# Stage B: the reported benchmark. Three configurations x three seeds at EQUAL
# WALL CLOCK (10 h) on the human-aware cup task, curriculum ramp 1.2 M steps as
# in the October campaign, validation every 40 k steps with 100 episodes.
#
#   bash run_stage_b.sh                       # 9 runs, seeds 0 1 2, 10 h each
#   SBX_ENVS=8 SBX_G=4 bash run_stage_b.sh    # if the rate test favours 8 envs
#   SEEDS="0 1 2" HOURS=10 bash run_stage_b.sh
#
# Configs:
#   sbx_sac_half   SBX SAC, 0.5 updates per env step (Stage A's best per hour);
#                  4 envs x 2 updates by default, or SBX_ENVS x SBX_G
#   sbx_sac_utd1   SBX SAC, 1 update per env step, 4 envs
#   sb3_sac_ctrl   SB3 SAC, 1 update per env step, 4 envs, torch threads
#                  unmanaged = the October setting (the control)
set -u
HOURS="${HOURS:-10}"; SEEDS="${SEEDS:-0 1 2}"; OUT="${OUT:-runs/stage_b}"; PY="${PY:-python}"
SBX_ENVS="${SBX_ENVS:-4}"; SBX_G="${SBX_G:-2}"
SECS=$(python -c "print(int($HOURS*3600))")
mkdir -p "$OUT"
COMMON="--task human-aware --human-preset curriculum --timesteps 2000000 --curriculum-ramp-timesteps 1200000 \
  --eval-freq 40000 --n-eval-episodes 100 --checkpoint-freq 100000 \
  --max-wall-seconds $SECS --no-save-buffer --out $OUT"
for seed in $SEEDS; do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 XLA_FLAGS=--xla_cpu_multi_thread_eigen=false \
    nohup $PY -m piper_rl.scripts.train --algo sbx_sac --n-envs $SBX_ENVS --gradient-steps $SBX_G --torch-threads 1 \
    $COMMON --seed $seed --run-name "sbx_sac_half_s$seed" > "$OUT/sbx_sac_half_s$seed.log" 2>&1 &
  sleep 5
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 XLA_FLAGS=--xla_cpu_multi_thread_eigen=false \
    nohup $PY -m piper_rl.scripts.train --algo sbx_sac --n-envs 4 --gradient-steps 4 --torch-threads 1 \
    $COMMON --seed $seed --run-name "sbx_sac_utd1_s$seed" > "$OUT/sbx_sac_utd1_s$seed.log" 2>&1 &
  sleep 5
  nohup $PY -m piper_rl.scripts.train --algo sac --n-envs 4 --gradient-steps 4 \
    $COMMON --seed $seed --run-name "sb3_sac_ctrl_s$seed" > "$OUT/sb3_sac_ctrl_s$seed.log" 2>&1 &
  sleep 5
done
echo "started 3 configs x seeds [$SEEDS], $HOURS h each; logs in $OUT/*.log"
echo "check after 10 min:  tail -n 3 $OUT/*.log"
echo "curves when done:    $PY -m piper_rl.scripts.plot_stage_a --runs $OUT --out $OUT/stage_b_curves.png"

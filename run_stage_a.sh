#!/usr/bin/env bash
# Stage A of the Phase 2 benchmark: screen learners at EQUAL WALL CLOCK on the
# cup task (human-aware, curriculum preset, corrected arm), one seed each.
#
#   bash run_stage_a.sh                 # all configs, 6 h each, seed 0
#   HOURS=6 SEED=0 bash run_stage_a.sh sbx_sac_utd05 sb3_sac_utd1
#
# Each config runs in the background with one torch/JAX thread and 4
# environments (about 5 threads per run). Logs and results go to
# $OUT/<config>/ ; the learning curve is $OUT/<config>/eval/wallclock.json.
# Works in Git Bash on Windows and in bash on Linux.
#
# Configs (name: learner, gradient steps per 4-env step = updates per env step):
#   sb3_sac_utd1      SB3 SAC, 4 updates per vec step  (today's setting, the control)
#   sb3_sac_utd025    SB3 SAC, 1 update per vec step
#   sbx_sac_utd1      SBX SAC (JAX), 4
#   sbx_sac_utd05     SBX SAC, 2
#   sbx_sac_utd025    SBX SAC, 1
#   sbx_crossq_utd1   SBX CrossQ, 4
#   ppo16             SB3 PPO, 16 environments (the on-policy reference)
set -u
HOURS="${HOURS:-6}"
SEED="${SEED:-0}"
OUT="${OUT:-runs/stage_a}"
EVAL_FREQ="${EVAL_FREQ:-40000}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
PY="${PY:-python}"
SECS=$(python -c "print(int($HOURS*3600))")
mkdir -p "$OUT"

declare -A CFG
CFG[sb3_sac_utd1]="--algo sac --n-envs 4 --gradient-steps 4"
CFG[sb3_sac_utd025]="--algo sac --n-envs 4 --gradient-steps 1"
CFG[sbx_sac_utd1]="--algo sbx_sac --n-envs 4 --gradient-steps 4"
CFG[sbx_sac_utd05]="--algo sbx_sac --n-envs 4 --gradient-steps 2"
CFG[sbx_sac_utd025]="--algo sbx_sac --n-envs 4 --gradient-steps 1"
CFG[sbx_crossq_utd1]="--algo sbx_crossq --n-envs 4 --gradient-steps 4"
CFG[ppo16]="--algo ppo --n-envs 16"

if [ $# -gt 0 ]; then NAMES=("$@"); else NAMES=(sb3_sac_utd1 sb3_sac_utd025 sbx_sac_utd1 sbx_sac_utd05 sbx_sac_utd025 sbx_crossq_utd1 ppo16); fi

COMMON="--task human-aware --human-preset curriculum --timesteps 2000000 \
  --eval-freq $EVAL_FREQ --n-eval-episodes $EVAL_EPISODES --checkpoint-freq 100000 \
  --torch-threads 1 --max-wall-seconds $SECS --seed $SEED --no-save-buffer --out $OUT"

for name in "${NAMES[@]}"; do
  if [ -z "${CFG[$name]:-}" ]; then echo "unknown config $name"; continue; fi
  echo "launching $name  (${HOURS} h, seed $SEED)"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 XLA_FLAGS=--xla_cpu_multi_thread_eigen=false \
  nohup $PY -m piper_rl.scripts.train ${CFG[$name]} $COMMON --run-name "$name" \
    > "$OUT/$name.log" 2>&1 &
  sleep 5
done
echo "started ${#NAMES[@]} run(s); logs in $OUT/*.log"
echo "check:   tail -n 3 $OUT/*.log"
echo "curves:  $PY -m piper_rl.scripts.plot_stage_a --runs $OUT --out $OUT/stage_a_curves.png"

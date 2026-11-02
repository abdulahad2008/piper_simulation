#!/usr/bin/env bash
# One-shot setup of a rented Linux CPU box (Ubuntu 22.04/24.04) for the benchmark.
# Tested layout: Hetzner CCX/CPX or any x86-64 VM with >= 32 vCPU.
#
#   curl -fsSL https://raw.githubusercontent.com/abdulahad2008/piper_simulation/kitchen-hazard-phase1/cloud_bootstrap.sh | bash
#   cd piper_simulation && HOURS=6 bash run_stage_a.sh
#
# Results come back with: scp -r user@host:piper_simulation/runs/stage_a ./runs/stage_a_cloud
set -eu
sudo apt-get update -q
sudo apt-get install -y -q git python3.11 python3.11-venv python3-pip libgl1 libosmesa6 libglfw3 build-essential >/dev/null
git clone --depth 1 --branch kitchen-hazard-phase1 https://github.com/abdulahad2008/piper_simulation.git || true
cd piper_simulation
python3.11 -m venv .venv
. .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt
pip install -q sb3-contrib==2.9.0 "jax[cpu]" sbx-rl==0.28.0
python -m piper_rl.scripts.bench_update_cost --threads 1 --out out/bench_update_cost.json
echo
echo "ready. nproc=$(nproc). launch with:  HOURS=6 bash run_stage_a.sh"
echo "the python inside run_stage_a.sh is this venv's: export PY=$PWD/.venv/bin/python"

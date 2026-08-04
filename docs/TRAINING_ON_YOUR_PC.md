# Training on your machine — step by step

Your hardware:

| | | verdict |
|---|---|---|
| CPU | 2 × Intel Xeon E5-2650 v4 @ 2.2 GHz | **24 physical cores / 48 threads.** This is the resource that matters, and you have a lot of it. |
| RAM | 32 GB | Ample. The replay buffer is ~200 MB; 12 simulator processes are ~4 GB. |
| GPU | AMD Radeon RX 5700 XT (8 GB) | **Will not be used.** PyTorch's CUDA path is NVIDIA-only, and ROCm has no Windows build. See §0. |
| OS | Windows 10 Pro 22H2 | Fine. One thing had to be fixed for it — see §0. |

Net: you have a genuinely good machine for this. Many cores beat one weak GPU
for a task like this, because the network is tiny (two 256-unit layers) and the
simulator is the thing you want to run 12 copies of.

---

## 0. Two things to know before you start

**Your GPU won't help, and that's fine.** `torch-directml` exists and can target
Radeon cards on Windows, but this workload is thousands of *small* matrix
multiplies, which is the case DirectML handles worst — it is usually slower than
a good CPU here, and it silently falls back to CPU for unsupported ops. Don't
bother. Everything below runs `--device cpu` deliberately.

**A Windows bug is fixed.** `train.py` hard-coded the `fork` process start
method, which doesn't exist on Windows, so `--n-envs 2` or higher would have
crashed with `ValueError: cannot find context for 'fork'`. It now picks the
platform default (`spawn` on Windows). Make sure you have the copy I just wrote
to `D:\Games\piper_sim\piper_rl\scripts\train.py`.

---

## 1. Set up (once)

```powershell
cd D:\Games\piper_sim
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install psutil          # lets the benchmark see physical vs logical cores
```

Verify:

```powershell
python -c "import mujoco, gymnasium, stable_baselines3, torch; print(mujoco.__version__, torch.__version__)"
```

**Every command below assumes you are in `D:\Games\piper_sim` with `.venv`
activated.** The scripts are Python modules; the working directory matters.

---

## 2. Confirm the environment is sane (2 minutes)

```powershell
python -m piper_rl.scripts.validate_model
```

You want `RESULT: all checks passed`. Then confirm the task is solvable:

```powershell
python -m piper_rl.scripts.scripted_demo --episodes 20
```

Expect ~100 % success, ~10 mm placement error. **If either of these fails, stop
and fix it — training a broken environment just wastes hours.**

---

## 3. Find your fastest settings (5–10 minutes)

Don't guess how many parallel environments to use. Measure it:

```powershell
python -m piper_rl.scripts.benchmark
```

It sweeps `n_envs` × torch thread counts and prints a table of end-to-end
throughput plus a ready-to-paste training command. On a 24-core box expect it to
land somewhere around `--n-envs 8..16` and `--torch-threads 8..16`.

Why both knobs: each parallel environment is a separate single-threaded MuJoCo
process, and the gradient update is a separate multi-threaded PyTorch job. They
compete for the same cores. Too many threads for torch and the simulators
starve; too few and the update becomes the bottleneck.

Quicker sweep if you're impatient:

```powershell
python -m piper_rl.scripts.benchmark --quick
```

For reference: a 2-core VM manages ~67 env steps/s end to end. You should see
several times that.

---

## 4. Start training

Use whatever the benchmark recommended. A reasonable starting point for your box:

```powershell
python -m piper_rl.scripts.train --algo sac --timesteps 2000000 ^
       --run-name sac_full ^
       --n-envs 12 --torch-threads 12 --gradient-steps 12 ^
       --batch-size 256 --net-arch 256 256 ^
       --eval-freq 25000 --n-eval-episodes 20 --checkpoint-freq 50000 ^
       --device cpu
```

What each flag is doing:

| flag | why |
|---|---|
| `--n-envs 12` | 12 MuJoCo processes filling the replay buffer in parallel |
| `--torch-threads 12` | leaves 12 cores for the simulators |
| `--gradient-steps 12` | one gradient update per environment step — the standard SAC ratio. Drop to 6 to roughly double wall-clock speed at some cost in sample efficiency |
| `--timesteps 2000000` | what this task realistically needs |
| `--checkpoint-freq 50000` | snapshots you can resume from |
| `--device cpu` | explicit, since there's no usable GPU |

Leave the terminal open. **Ctrl+C is safe** — the model is saved on the way out,
and you can resume:

```powershell
python -m piper_rl.scripts.train --timesteps 1000000 --run-name sac_full2 ^
       --load runs\sac_full\checkpoints\piper_800000_steps.zip ^
       --n-envs 12 --torch-threads 12 --gradient-steps 12 --device cpu
```

---

## 5. Watch it

Second terminal, same venv:

```powershell
cd D:\Games\piper_sim
.venv\Scripts\activate
tensorboard --logdir runs\sac_full\tb
```

Open <http://localhost:6006>. The curves that matter, in order:

| curve | what it means |
|---|---|
| **`eval/mean_reward`** | **the honest one.** Deterministic, always starts from the true initial state. This is what should climb. |
| `rollout/grasp_rate` | fraction of episodes with a two-sided grasp |
| `rollout/lift_rate` | fraction that got the cup off the table |
| `rollout/place_err_mm` | should fall from ~280 (never moved it) toward ~10 |
| `rollout/success_rate` | **inflated — do not quote it.** ~50 % of training episodes start part-way through the task by design |
| `train/ent_coef` | SAC's exploration temperature; should decay as it commits to a strategy |

In the console you'll see the same numbers as text blocks every 2000 steps.

---

## 6. Checkpoints

```
runs\sac_full\
  best\best_model.zip              <- best by DETERMINISTIC eval. Use this one.
  checkpoints\piper_*_steps.zip    <- periodic snapshots, for resuming
  final_model.zip                  <- last state (also written on Ctrl+C)
  tb\                              <- TensorBoard
  config.json                      <- the exact config this run used
```

---

## 6b. Pausing and resuming

**Yes, you can stop and pick up later — but resume with `--resume`, not `--load`.**

To pause: **Ctrl+C** in the training terminal. It saves three things before
exiting:

```
runs\sac_full\final_model.zip      the networks
runs\sac_full\replay_buffer.pkl    ~176 MB of collected experience
runs\sac_full\best\best_model.zip  best-by-eval so far
```

To continue:

```powershell
python -m piper_rl.scripts.train --resume runs\sac_full --timesteps 1000000 ^
       --n-envs 12 --torch-threads 12 --gradient-steps 12 --device cpu
```

`--timesteps` is how many **additional** steps to run. The step counter, the
TensorBoard curves and the checkpoint numbering all continue where they left
off, in the same run directory, so you get one continuous graph rather than two
disconnected ones.

### Why `--resume` and not `--load`

SAC learns from a **replay buffer** — a rolling store of the last 400 000
transitions. It is not inside the `.zip` (it is ~176 MB on its own). `--load`
restores the networks but starts with an empty buffer, so the first
`learning_starts` steps are random again and the critic retrains against a
tiny, freshly-collected sample. In practice that costs a visible dip and tens of
thousands of steps of recovery.

`--resume` reloads the buffer too, and picks up essentially where it stopped.
It prints what it found:

```
resuming from runs\sac_full\final_model.zip
replay buffer: runs\sac_full\replay_buffer.pkl
replay buffer restored: 400,000 transitions
```

If it says `replay buffer: NOT FOUND`, you either used `--no-save-buffer` or the
process was killed rather than interrupted — the networks still carry over,
you just lose the experience.

### Notes

- **Ctrl+C once**, then wait. Saving 176 MB takes a few seconds. Hitting it
  repeatedly or closing the window kills the process before it writes.
- If the machine crashes or loses power, fall back to the newest file in
  `checkpoints\` — those are written every `--checkpoint-freq` steps and have
  no buffer, so expect the dip.
- The buffer file is a fixed ~176 MB regardless of how full it is (the arrays
  are pre-allocated). `--no-save-buffer` skips it if disk is tight.
- Evaluating a paused run is completely safe and doesn't disturb anything:
  ```powershell
  python -m piper_rl.scripts.evaluate --model runs\sac_full\best\best_model.zip --episodes 30
  ```

---

## 7. Evaluate

Don't judge before ~500k steps.

```powershell
:: the number that counts
python -m piper_rl.scripts.evaluate --model runs\sac_full\best\best_model.zip --episodes 50

:: watch it live
python -m piper_rl.scripts.evaluate --model runs\sac_full\best\best_model.zip ^
       --viewer --realtime --episodes 5 --camera forehead

:: record it
python -m piper_rl.scripts.evaluate --model runs\sac_full\best\best_model.zip ^
       --episodes 5 --video eval.mp4 --camera forehead
```

**Watch the video before believing any number.**

Rough milestones on the honest (`eval`) metrics:

| steps | what a healthy run looks like |
|---|---|
| 50k | eval reward climbing off the floor (≈ −35 → −15). Grasps starting to appear. |
| 200k | occasional real successes from the true initial state. Lift rate meaningfully above zero. |
| 500k | 20–50 % success, placement error falling |
| 1–2M | 70–90 % success, error inside the 45 mm tolerance |

If you're flat at 500k, §8.

---

## 8. If it stalls

In order of how cheap they are to try:

1. **Learn the easy version first.** Turn randomisation off, confirm it can
   solve the clean task, then re-enable and expect a dip:
   ```powershell
   python -m piper_rl.scripts.train --timesteps 500000 --run-name sac_clean ^
          --no-domain-rand --no-noise --n-envs 12 --torch-threads 12 --device cpu
   ```
   If it can't solve the *clean* task, randomisation isn't the problem.
2. **Halve the update cost**: `--gradient-steps 6` with the same `--n-envs 12`.
   Twice the environment steps per hour, slightly less efficient per step.
   Usually a net win when you're compute-bound.
3. **Try TD3**: `--algo td3`. SAC's entropy bonus can keep the gripper action
   too random to ever hold a stable grasp.
4. **Bigger network**, now that you have cores: `--net-arch 400 300 --batch-size 512`.
5. **HER / goal-conditioned refactor.** The biggest untried win, and not yet
   implemented: make the observation a `Dict` with `achieved_goal` = object
   position and `desired_goal` = target, then use SB3's `HerReplayBuffer`. On
   pick-and-place this usually beats any amount of reward shaping.
6. **Behaviour-cloning warm start.** `scripted_demo.py` already produces
   successful trajectories through the exact policy action space — pre-train the
   actor on a few thousand of them, then fine-tune with SAC.

---

## 9. Troubleshooting

| symptom | fix |
|---|---|
| `ValueError: cannot find context for 'fork'` | you have the old `train.py`; re-copy the one delivered above |
| `ModuleNotFoundError: piper_rl` | not in `D:\Games\piper_sim`, or venv not activated |
| training slows to a crawl after a while | replay buffer growth — 400k transitions is ~200 MB, should be fine on 32 GB; check Task Manager for swapping |
| CPU sits at ~10 % | `--n-envs 1`; run the benchmark and raise it |
| CPU pinned at 100 % but throughput is poor | torch threads + n_envs exceed your core count; lower `--torch-threads` |
| `eval/mean_reward` flat and equal to about −12 | that's the do-nothing score — the policy has learned to sit still. Go to §8. |
| viewer window won't open over RDP | RDP's GL support is limited; use `--video` instead |

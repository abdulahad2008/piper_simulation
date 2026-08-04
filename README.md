# AgileX PIPER — pick & place in MuJoCo, with a path to hardware

Reinforcement learning for the AgileX PIPER 6-DoF arm: pick a cup off the table
and place it on a target pad. Trained in MuJoCo, watched through a fixed
"forehead" camera on the robot base, and structured so the policy can later be
moved onto the physical arm behind a safety layer.

* **`piper_rl/README.md`** — full reference: observation/action/reward spec,
  algorithm choice, every command.
* **`docs/ARCHITECTURE.md`** — what the project looked like before, what changed
  and why.
* **`docs/SIM2REAL.md`** — hardware bring-up plan, and an honest list of what is
  *not* solved.

---

## Quick start

```bash
cd piper_sim
pip install -r requirements.txt
# headless Linux only:  sudo apt-get install libosmesa6 libgl1-mesa-dri
#                       export MUJOCO_GL=osmesa

python -m piper_rl.scripts.validate_model                     # health check
python -m piper_rl.scripts.scripted_demo --episodes 20        # is it solvable?
python -m piper_rl.scripts.train --algo sac --timesteps 1500000 --run-name sac_v1
python -m piper_rl.scripts.evaluate --model runs/sac_v1/best/best_model.zip \
       --episodes 50 --video eval.mp4
```

Interactive viewer (needs a display; on macOS use `mjpython`):

```bash
python agilex_piper/view_piper.py --camera forehead
python -m piper_rl.scripts.evaluate --model ... --viewer --realtime
```

---

## The scene

| | |
|---|---|
| robot | AgileX PIPER, 6 revolute joints + parallel gripper, bolted to the table at the world origin (base flange z = 0.20) |
| object | cup — cylinder ~48 mm across, ~70 mm tall, 30–120 g, randomised size/mass/friction/colour |
| pick region | r 0.30–0.39 m, azimuth −42° … −10° |
| place region | r 0.30–0.39 m, azimuth +10° … +42° |
| **forehead camera** | fixed bracket on `base_link`: 25 cm behind, 30 cm above the base flange, 25° down, 65° vertical FoV — the primary view, shown during training and evaluation |
| other cameras | `wrist` (eye-in-hand), `front`, `side`, `top`, `overview` |
| control | 20 Hz policy over 500 Hz physics |

---

## Status

| stage | state |
|---|---|
| MuJoCo model loads, renders headless, all named handles present | ✅ |
| Straight-down IK converges < 1 mm across the task workspace | ✅ |
| Scripted controller picks and places | ✅ **100 %** clean, **60 %** with full domain randomisation + sensor noise (open-loop) |
| Gymnasium env, `check_env` clean, deterministic, reward-exploit tested | ✅ |
| SAC / TD3 / PPO training, checkpointing, TensorBoard, task metrics | ✅ |
| Deterministic evaluation + forehead-camera video | ✅ |
| Safety layer, shared by sim and the hardware interface | ✅ |
| Hardware interface | 🟡 abstract API + sim backend done; the `piper_sdk` adapter is a **deliberate stub** |
| Trained policy | see the training report — do not treat any number here as final without the evaluation output and video |

---

## Repository layout

```
piper_sim/
├─ agilex_piper/     MuJoCo model (Menagerie PIPER) + task scene + viewer
├─ piper_rl/         env, IK, safety, domain randomisation, training, eval
├─ docs/             architecture + sim-to-real plan
├─ out/              rendered proof frames and videos
└─ requirements.txt
```

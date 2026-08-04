# Quick start — running and testing in simulation

Everything here is run from **`D:\Games\piper_sim`** (the folder that contains
`piper_rl\`). That matters: the scripts are Python *modules*, so the working
directory has to be the project root.

---

## 0. Install (once)

```powershell
cd D:\Games\piper_sim
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.10–3.12. On Windows no graphics setup is needed — MuJoCo uses the
system OpenGL. (Only headless Linux needs `MUJOCO_GL=osmesa`.)

Sanity check:

```powershell
python -c "import mujoco, gymnasium, stable_baselines3; print(mujoco.__version__)"
```

---

## 1. Health check — run this first

```powershell
python -m piper_rl.scripts.validate_model
```

Checks the MJCF compiles, every named handle exists, the gripper can actually
fit around the cup, the workspace is reachable, the Gymnasium API is clean, the
env is deterministic, and — importantly — that the reward cannot be farmed.

Expected tail:

```
       do-nothing          -34.80
       hover-and-bob        -1.76   (shaping exploit)
       scripted solution   284.69
[  ok  ] solving beats hovering by a wide margin   margin 286.4
[  ok  ] hovering is not profitable   -1.8
       2.29 ms per env step -> 437 steps/s
RESULT: all checks passed -- the environment is ready to train
```

Add `--map` for the full straight-down IK reachability table (slower). Run this
after **any** change to the XML or to `config.py`.

---

## 2. Look at the scene

```powershell
python agilex_piper\view_piper.py                      # free camera, orbit with the mouse
python agilex_piper\view_piper.py --camera forehead    # the fixed first-person camera
python agilex_piper\view_piper.py --camera wrist       # eye-in-hand
```

`[` and `]` cycle cameras inside the window, Tab toggles the panels.
The arm just holds its home pose here — this is for inspecting the model.

---

## 3. Watch the task actually being solved

A scripted IK controller drives the **same 5-D action space the policy uses**, so
this proves the task is physically solvable and that the grasp/lift/place
detection agrees with what you see.

```powershell
:: 20 episodes, headless, prints per-episode metrics
python -m piper_rl.scripts.scripted_demo --episodes 20

:: clean physics, no randomisation or sensor noise
python -m piper_rl.scripts.scripted_demo --episodes 15 --no-domain-rand --no-noise

:: record the forehead camera to an mp4
python -m piper_rl.scripts.scripted_demo --episodes 3 --video demo.mp4 --camera forehead
```

Expected (with full domain randomisation **and** sensor noise on):

```
  grasp rate      100.0%
  lift  rate       96.0%
  SUCCESS rate    100.0%
  mean place err     8.7 mm
  mean length       98.5 steps
```

**If this does not pass, stop and fix it before training anything.** No policy
will learn a task the arm cannot physically do.

You can also run it through the evaluation harness for the full metric table:

```powershell
python -m piper_rl.scripts.evaluate --scripted --episodes 50
```

---

## 4. Poke the environment by hand

```python
# save as try_env.py in D:\Games\piper_sim, then:  python try_env.py
import numpy as np
from piper_rl import PiperPickPlaceEnv, EnvConfig

cfg = EnvConfig()
cfg.curriculum = False          # always start from the true initial state
env = PiperPickPlaceEnv(cfg)

obs, info = env.reset(seed=0)
print("obs", obs.shape, "action", env.action_space.shape)
print("cup at", np.round(env.obj_pos, 3), " target at", np.round(env.target_pos, 3))

for i in range(60):
    #  [dx, dy, dz, dyaw, gripper]   each in [-1, 1]
    obs, r, terminated, truncated, info = env.step(np.array([0, 0, -1, 0, 1.0]))
print("TCP now", np.round(env.tcp_pos, 3), "grasped:", info["grasped"])

frame = env.render("forehead")   # (480, 640, 3) uint8
env.close()
```

Useful config switches while experimenting:

```python
cfg.domain_rand.enabled = False   # nominal physics
cfg.noise.enabled       = False   # perfect sensors
cfg.action_mode         = "joint" # 7-D joint deltas instead of Cartesian
cfg.obs_mode            = "rgb"   # 84x84 forehead camera as the observation
cfg.render_camera       = "overview"
```

---

## 5. Train

```powershell
:: quick smoke test -- proves the loop runs end to end, ~30 s
python -m piper_rl.scripts.train --timesteps 2000 --run-name smoke

:: the real run
python -m piper_rl.scripts.train --algo sac --timesteps 2000000 --run-name sac_full

:: on a GPU, use the bigger network -- it costs almost nothing there
python -m piper_rl.scripts.train --algo sac --timesteps 2000000 --run-name sac_full ^
       --net-arch 512 512 256 --batch-size 512 --n-envs 4 --gradient-steps 4
```

Watch it:

```powershell
tensorboard --logdir runs\sac_full\tb
```

Outputs land in `runs\<name>\`:

| path | what |
|---|---|
| `best\best_model.zip` | best by **deterministic evaluation** reward — use this one |
| `checkpoints\piper_*_steps.zip` | periodic snapshots |
| `final_model.zip` | last state, also written if you Ctrl-C |
| `tb\` | TensorBoard logs |
| `config.json` | the exact config the run used |

Resume: `--load runs\sac_full\checkpoints\piper_500000_steps.zip`.
Ctrl-C is safe — the model is saved on the way out.

**Expect this to take a while.** On CPU the gradient update, not the simulator,
is the bottleneck: ~44 env steps/s on 2 cores, so 2 M steps is ~12 h. On a GPU
with `--n-envs 4` it is a few hours.

---

## 6. Evaluate and watch the policy

Every evaluation episode starts from the **true** initial state (arm at home,
cup on the table) — the training curriculum is always disabled here.

```powershell
:: the number that matters
python -m piper_rl.scripts.evaluate --model runs\sac_full\best\best_model.zip --episodes 50

:: record it
python -m piper_rl.scripts.evaluate --model runs\sac_full\best\best_model.zip ^
       --episodes 5 --video eval.mp4 --camera forehead

:: watch it live in the MuJoCo window, at real-time speed
python -m piper_rl.scripts.evaluate --model runs\sac_full\best\best_model.zip ^
       --viewer --realtime --episodes 5

:: nominal physics, perfect sensors -- an upper bound, not the headline number
python -m piper_rl.scripts.evaluate --model ... --no-domain-rand --no-noise
```

Output:

```
SUCCESS RATE         xx.x%   (n/50)
grasp rate           xx.x%
lift  rate           xx.x%
episode reward       ...
episode length       ...  (successful only: ...)
placement error      mean / median / p90  in mm
collision steps      per episode
safety clamped/vetoed
```

There is a partially-trained checkpoint at `models\sac_220k_partial.zip` you can
point this at to see the harness work. **It does not solve the task** — it was
trained before the differential-IK fix and scores 0 %. It is there to exercise
the tooling, not as a result.

---

## 7. How to know whether it worked

Two rules, both worth being strict about:

1. **`rollout/success_rate` in TensorBoard is not the answer.** It is measured on
   training episodes, ~50 % of which start part-way through the task (the
   reverse curriculum), so it is inflated. Only `eval/` and `evaluate.py` start
   from the true initial state.
2. **Watch the video.** A success rate without a recording is not a result. Run
   `--video` and look at it.

A policy worth taking to hardware should reach **≥ 80 % over 50 episodes with
domain randomisation and noise on**, with placement error comfortably inside the
45 mm success tolerance. Then read `docs\SIM2REAL.md`.

---

## Troubleshooting

| symptom | cause |
|---|---|
| `ModuleNotFoundError: piper_rl` | you are not in `D:\Games\piper_sim`, or the venv is not activated |
| `FileNotFoundError: ...piper_task.xml` | same — `config.py` resolves the model relative to the package |
| viewer window never opens | headless/remote machine; use `--video` instead of `--viewer` |
| `check_env` complains about shapes | you changed the observation without updating `state_dim` in `_build_spaces` |
| training crashes with a torch/GL segfault on Linux | import-order issue; `piper_rl/__init__.py` already works around it — make sure you `import piper_rl` before creating an optimiser |
| high `safety clamped` count | normal if the policy saturates its actions; the joint-velocity limit is doing its job. Only worrying if `safety vetoed` is non-zero |

# PIPER pick-and-place — RL stack

Reinforcement learning for the AgileX PIPER 6-DoF arm: pick a cup off the table
and place it on a target pad, learned in MuJoCo, structured so the policy can
later be moved onto the physical robot.

---

## 1. Layout

```
piper_sim/
├─ agilex_piper/              # MuJoCo model (Menagerie PIPER + task scene)
│  ├─ piper.xml               #   robot: + forehead camera, tcp site, named pads
│  ├─ piper_task.xml          #   scene: table, cup, target, cameras, keyframe
│  ├─ scene.xml               #   bare robot + ground (unchanged)
│  ├─ assets/                 #   meshes (unchanged)
│  ├─ view_piper.py           #   interactive viewer (unchanged)
│  └─ validate_piper.py       #   original scripted test (superseded, kept)
│
├─ piper_rl/                  # everything new
│  ├─ config.py               #   ALL tunable parameters, documented
│  ├─ ik.py                   #   damped-least-squares IK (5-DoF task)
│  ├─ piper_env.py            #   Gymnasium environment
│  ├─ domain_rand.py          #   reset-time randomisation
│  ├─ safety.py               #   command validator (shared sim + hardware)
│  ├─ callbacks.py            #   task-metric logging for SB3
│  ├─ hardware/
│  │  ├─ interface.py         #   RobotInterface ABC + RobotState
│  │  ├─ sim_backend.py       #   MuJoCo implementation
│  │  └─ piper_real.py        #   piper_sdk adapter — DELIBERATE STUB
│  └─ scripts/
│     ├─ validate_model.py    #   health checks, reachability map
│     ├─ scripted_demo.py     #   IK waypoint baseline (proves solvability)
│     ├─ train.py             #   SAC / TD3 / PPO training
│     └─ evaluate.py          #   deterministic eval + video
│
├─ docs/SIM2REAL.md           # step-by-step hardware bring-up plan
└─ requirements.txt
```

---

## 2. Install and run

```bash
cd piper_sim
python -m venv .venv && .venv\Scripts\activate      # Windows
# source .venv/bin/activate                          # Linux / macOS
pip install -r requirements.txt

# Headless Linux only:
#   sudo apt-get install libosmesa6 libgl1-mesa-dri
#   export MUJOCO_GL=osmesa
```

All commands below are run from `piper_sim/`.

```bash
# 0. health check -- run this first, and after any XML edit
python -m piper_rl.scripts.validate_model
python -m piper_rl.scripts.validate_model --map      # + full reachability map

# 1. prove the task is solvable with a scripted controller
python -m piper_rl.scripts.scripted_demo --episodes 20
python -m piper_rl.scripts.scripted_demo --episodes 3 --video demo.mp4

# 2. TRAIN
python -m piper_rl.scripts.train --algo sac --timesteps 1500000 --run-name sac_v1
#    on a GPU box, use the bigger network:
#      ... --net-arch 512 512 256 --batch-size 512
#    watch it:  tensorboard --logdir runs/sac_v1/tb

# 3. EVALUATE + VISUALISE
python -m piper_rl.scripts.evaluate --model runs/sac_v1/best/best_model.zip \
       --episodes 50
python -m piper_rl.scripts.evaluate --model runs/sac_v1/best/best_model.zip \
       --episodes 5 --video eval_forehead.mp4 --camera forehead
python -m piper_rl.scripts.evaluate --model runs/sac_v1/best/best_model.zip \
       --viewer --realtime                    # live MuJoCo window
#    macOS: prefix viewer commands with `mjpython` instead of `python`

# baseline for comparison
python -m piper_rl.scripts.evaluate --scripted --episodes 50
```

Checkpoints land in `runs/<name>/checkpoints/`, the best-by-eval-reward model in
`runs/<name>/best/best_model.zip`, the last one in `runs/<name>/final_model.zip`.
Resume with `--load <checkpoint.zip>`.

---

## 3. The task

The arm is bolted to the table at the world origin, base flange at `z = 0.20`.
A cup (cylinder, ~48 mm across, ~70 mm tall, 30–120 g) starts on a blue pad in
the right half of the workspace; a green pad marks the target in the left half.
Both are re-sampled every reset:

| | radius from base | azimuth |
|---|---|---|
| cup | 0.30 – 0.39 m | −42° … −10° |
| target | 0.30 – 0.39 m | +10° … +42° |

### Cameras

| name | mount | role |
|---|---|---|
| **`forehead`** | fixed bracket on `base_link`, 25 cm behind / 30 cm above the base flange, 25° down, 65° vertical FoV | **primary** — the view shown during training and evaluation, and the one a vision policy would consume |
| `wrist` | eye-in-hand on `link6` | secondary, sees the grasp close up |
| `front` `side` `top` `overview` | world-fixed | third-person renders |

The forehead camera is attached to the robot base, not the world, so it moves
with the robot but not with the arm — the sim twin of a camera bracketed to the
PIPER's mounting plate.

---

## 4. Observation space

`obs_mode="state"` (default): `Box(-inf, inf, (51,), float32)`

| slice | size | contents |
|---|---|---|
| 0:6 | 6 | arm joint positions, normalised to [−1, 1] by joint range |
| 6:12 | 6 | arm joint velocities / `max_joint_vel` |
| 12:14 | 2 | gripper opening (normalised), gripper velocity |
| 14:17 | 3 | TCP position, base frame |
| 17:20 | 3 | TCP approach axis (unit vector; −z when pointing down) |
| 20:23 | 3 | TCP pinch axis (encodes wrist yaw) |
| 23:26 | 3 | object position |
| 26:29 | 3 | object up-axis (reveals a tipped-over cup) |
| 29:32 | 3 | object linear velocity |
| 32:35 | 3 | target position |
| 35:38 | 3 | object − TCP |
| 38:41 | 3 | target − object |
| 41:43 | 2 | `grasped`, `lifted` flags |
| 43:44 | 1 | last gripper command |
| 44:46 | 2 | object half-width, half-height |
| 46:51 | 5 | previous action |

Every proprioceptive and object term is corrupted with Gaussian noise
(`NoiseConfig`) and the whole vector is delayed by `obs_latency_steps`
(1 step = 50 ms) before the policy sees it.

Other modes, both fully implemented:
`obs_mode="rgb"` → `uint8 (84, 84, 3)` from the forehead camera, with brightness
jitter, Gaussian noise and its own latency;
`obs_mode="state+rgb"` → `Dict` of both.

**Rotations are given as axis vectors, never as Euler angles or raw
quaternions** — no wrap-around discontinuities for the network to learn around.

---

## 5. Action space

`action_mode="cartesian"` (default): `Box(-1, 1, (5,), float32)`

| index | meaning | scale |
|---|---|---|
| 0:3 | TCP displacement in the base frame | ±`max_step_dist` = 3 cm/step |
| 3 | wrist yaw rate (joint 6) | ±`max_step_yaw` = 0.2 rad/step |
| 4 | gripper command | maps [−1, 1] → [closed, open] |

The gripper's approach axis is servoed to point straight down by the
differential IK; the policy never controls roll or pitch. This is a deliberate
restriction: a top-down pick needs nothing more, it removes the large
kinematically-infeasible region of the PIPER's workspace from the search
problem, and it maps onto the SDK's end-pose command mode.

`action_mode="joint"`: `Box(-1, 1, (7,))` = six joint-position deltas
(±`max_step_joint` = 0.06 rad) + gripper. The most literal match to the
hardware's joint-command interface; use it if you would rather run IK on the
robot side.

**Every action passes through, in order:** differential IK → exponential
smoothing (α = 0.45) → actuation noise + 1 % command dropout → the safety layer
→ MuJoCo. The identical filter and the identical safety layer run in the
hardware interface.

---

## 6. Reward

Five phases. The full definition, with weights, is in
`config.py :: RewardConfig`.

| phase | term | weight |
|---|---|---|
| reach | `Δ‖tcp − grasp point‖` | +12 /m |
| | gripper pointing down | +0.6 |
| | closing the gripper *while at the object* | +0.4 |
| grasp | first two-sided pad contact | +1.5 once |
| | first stable grasp (3 consecutive steps) | +12 once |
| lift | height gained, capped at 6 cm | +20 /m |
| | first time clear of the table | +12 once |
| transport | `Δ‖object − target‖`, **only while held and airborne** | +18 /m |
| | first time over the target | +6 once |
| place | placed + released + settled | **+120 once** |

| penalty | weight |
|---|---|
| per step (time) | −0.05 |
| ‖a‖² (effort) | −0.10 |
| ‖a − a_prev‖² (smoothness) | −0.30 |
| ‖q̇‖² (excessive joint motion) | −0.02 |
| arm-link collision with table / object / itself | −2.0 per step |
| dropping the object away from the target | −12, and the grasp/lift bonuses are revoked |
| object off the table | −40, episode ends |
| safety-layer veto / severe contact | −30 |
| moving *away* from the current sub-goal | −4 /m, on top of the negative shaping |

### Why this cannot be farmed

1. **All distance shaping is difference-based** (`prev_d − d`). Over an episode
   these telescope to `d_start − d_end`; oscillating in and out earns exactly
   zero, and hovering earns zero.
2. **Bonuses are one-shot per episode**, and the grasp/lift bonuses are
   *revoked* on a drop. "Tap the cup and let go, repeatedly" nets negative.
3. **Shaping is phase-gated.** Transport reward only pays while the object is
   genuinely held *and* more than 2 cm off the table, so shoving the cup along
   the surface with the gripper earns nothing.
4. **The success bonus (+120) dominates** the entire achievable shaping budget
   (~45 for a perfect run), so finishing is always optimal.

`validate_model.py` check 5 tests exactly this: it scores a do-nothing policy, a
"hover over the object and bob up and down" exploit, and the scripted solution,
and fails if the exploit is anywhere near the solution.

### Termination

| condition | flag |
|---|---|
| object placed on target, released, settled for 3 steps | `terminated` (success) |
| object falls off the table (`z < 0.15` or outside the table) | `terminated` |
| safety layer vetoes / contact force > 40 N | `terminated` |
| 250 control steps (12.5 s) elapsed | `truncated` |

Dropping the object does *not* end the episode by default — the agent is given
the chance to pick it up again — but it loses the bonuses.

---

## 7. Reset distribution (the curriculum)

**Training** episodes start from one of four phases:

| phase | probability | initial state |
|---|---|---|
| `reach` | 0.55 | arm at home, cup on the table |
| `near` | 0.20 | gripper already hovering 4–9 cm above the cup |
| `grasped` | 0.15 | cup already held, 3–7.5 cm above the table |
| `over_target` | 0.10 | cup held above the target |

This changes only the *initial-state distribution*, not the task or the reward.
It is the standard reverse-curriculum trick, and it is the difference between
learning pick-and-place in ~10⁵ steps and ~10⁷: without it the agent has to
stumble onto a grasp by chance before any of the later reward terms are ever
observed.

**Evaluation always uses `curriculum=False`** — every eval episode starts from
the true initial state, arm at home, cup on the table. `EnvConfig.eval_variant()`
enforces this, and both `EvalCallback` during training and `evaluate.py` use it.

Turn it off entirely with `--no-curriculum` if you want the unaided result.

---

## 8. Why SAC

* The action space is continuous — value-based discrete methods are out.
* Each env step costs a 25-substep MuJoCo rollout (~2 ms), so **sample
  efficiency dominates wall-clock**. SAC is off-policy with a replay buffer and
  does ~1 gradient update per env step; PPO discards every sample after one
  epoch and typically needs 10–50× more interaction on manipulation tasks.
* The exploration problem here is severe: the reward is nearly flat until the
  fingers happen to close on the cup. SAC's maximum-entropy objective with an
  auto-tuned temperature stays stochastic exactly as long as it needs to, then
  anneals — no hand-scheduled noise.
* **TD3** (`--algo td3`) is the closest alternative and worth trying if SAC's
  entropy term leaves the final policy too jittery for hardware.
  **PPO** (`--algo ppo`) is included for completeness and parallelises better if
  you have many cores and no GPU.

Defaults: `lr 3e-4`, `buffer 400 k`, `batch 256`, `τ 0.01`, `γ 0.98`,
`net [256, 256]`, `ent_coef auto_0.1`, `learning_starts 5 000`.
`γ = 0.98` (not 0.99) because episodes are 250 steps and a ~50-step effective
horizon is plenty — it cuts critic variance noticeably.
On CPU the gradient update, not MuJoCo, is the bottleneck: `[256,256]`/batch 256
runs ~55 env steps/s on 2 cores; `[512,512,256]`/batch 512 drops to ~16. On a
GPU use the larger network.

---

## 9. Sim-to-real features already in the environment

| gap | how it is handled | where |
|---|---|---|
| joint position limits | from the official URDF, enforced with a 2 % margin | `safety.py` |
| joint velocity / acceleration | 1.0 rad/s, 20 rad/s² — clamped per control tick | `RobotLimits` |
| control frequency | 20 Hz policy over 500 Hz physics | `EnvConfig.control_hz` |
| gripper travel + speed | 0–35 mm joint (0–70 mm opening), 0.15 m/s | `RobotLimits` |
| action smoothing | EMA α = 0.45 on the command, identical in the hardware interface | `piper_env`, `hardware/interface` |
| sensor noise | joint 3 mrad, object pose 6 mm / 50 mrad, target 4 mm | `NoiseConfig` |
| observation latency | 1 control step (50 ms) | `NoiseConfig` |
| camera noise + latency | Gaussian + brightness jitter + 1-step delay | `NoiseConfig` |
| command dropout | 1 % of commands are lost, previous one held | `NoiseConfig` |
| object randomisation | position, yaw, size ±15 %, mass 30–120 g, friction 0.8–2.0 | `DomainRandConfig` |
| lighting / appearance | key-light position and intensity, object and table colour | `DomainRandConfig` |
| actuator model error | kp, kv ±25 %; joint damping ±40 %; friction loss ±60 % | `DomainRandConfig` |
| conservative hardware limits | separate profile, ~⅓ of sim speeds | `hardware/piper_real.conservative_limits()` |
| command validation | same `SafetyLayer` object in sim and on hardware | `safety.py` |

See `docs/SIM2REAL.md` for the bring-up procedure and the honest list of what is
**not** solved (object pose estimation, above all).

---

## 10. Metrics

`evaluate.py` and the training logs report:

```
success_rate      object placed on the target, released, settled
grasp_rate        a two-sided grasp achieved at least once
lift_rate         object cleared the table by 6 cm
place_err_mm      final object-to-target distance: mean / median / p90
ep_rew_mean       episode reward
ep_len            episode length (and length over successes only)
collision_steps   steps with an arm-link collision
safety_clamped    commands the safety layer had to clamp, per episode
```

During training these appear in TensorBoard under `rollout/`.

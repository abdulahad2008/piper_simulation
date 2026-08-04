# Architecture: what existed, what changed, and why

## 1. The project as found

`D:\Games\piper_sim` contained two near-identical copies of the same work — a
top-level `agilex_piper/` plus a nested `piper_sim/` that was ~40 minutes newer
and slightly further along. The newer copy is the one this work builds on.

### What already worked

| piece | state |
|---|---|
| `agilex_piper/piper.xml` | MuJoCo Menagerie AgileX PIPER: 6 revolute joints, mimic-constrained parallel gripper, position actuators, `impratio=10` + elliptic cone, hand-built box collision geoms on the fingers. Solid, and reused as-is apart from the additions below. |
| `agilex_piper/assets/` | 84 meshes, all referenced, all present. Untouched. |
| `agilex_piper/piper_task.xml` | Task scene: table, free-joint object, source/destination pads, four external cameras, a `home` keyframe. Good bones. |
| `agilex_piper/scene.xml` | Bare robot + ground plane. Untouched. |
| `agilex_piper/view_piper.py` | Passive-viewer launcher. Worked. |
| `make_views_page.py` + `piper_views.html` | Renders five POVs into a self-contained HTML page. Worked (paths hard-coded to a macOS home directory). |
| local edits already made | The arm had been raised onto the table (`base_link pos="0 0 0.2"`), the `tcp` site moved from z=0.17 to z=0.09, gripper `kp` raised 40→300, and the object made taller and grippier. All were steps in the right direction. |

### What was missing or broken

1. **The scripted pick-and-place did not work.** `validate_piper.py` printed
   `RESULT: NEEDS TUNING` — the object never left the table. Three independent
   causes:
   * **IK was over-constrained.** It solved for a *full 6-DoF pose* (position +
     complete orientation). The PIPER's joint limits (`joint2 ∈ [0, π]`,
     `joint3 ∈ [-2.697, 0]`, `joint5 ∈ ±1.22`) make that infeasible over most of
     the workspace; the solver stalled 50–90 mm and 20–40° away and the script
     used the answer anyway.
   * **Waypoints were outside the reachable set.** The lift/carry height of
     z = 0.42 is simply not reachable with the gripper pointing down: sampling
     200 000 random configurations shows the straight-down TCP tops out at
     z ≈ 0.376, and only up to z ≈ 0.34 at usable radii.
   * **The object was too tall for the gripper.** `link6`'s collision capsule —
     the gripper "palm" — sits ~46 mm above the TCP. The 90 mm-tall object was
     struck by the palm before the fingers could close on it (confirmed by
     dumping the contact pairs: four `link6_col ↔ obj_geom` contacts during
     every descent).
2. **No "forehead" camera.** The only robot-mounted camera was `wrist`, on
   `link6` — eye-in-hand, not the fixed first-person view asked for.
3. **No RL anything**: no Gymnasium env, no reward, no observation/action space,
   no training, evaluation, checkpointing or logging; no `requirements.txt`.
4. **No sim-to-real structure**: no safety layer, no domain randomisation, no
   sensor noise, no action smoothing, no hardware abstraction.
5. **Unnamed geoms.** The finger pads and arm collision geoms had no names, so
   nothing could reason about contacts (grasp detection, collision penalties).

---

## 2. Proposed architecture

```
                    ┌──────────────────────────────────────────┐
                    │  scripts/  train · evaluate · validate   │
                    │            scripted_demo                 │
                    └───────────────┬──────────────────────────┘
                                    │ Gymnasium API
                    ┌───────────────▼──────────────────────────┐
                    │        PiperPickPlaceEnv                 │
                    │  obs: state | rgb(forehead) | both       │
                    │  act: cartesian(5) | joint(7)            │
                    │  reward: 5-phase, difference-shaped      │
                    └──┬──────────┬──────────┬─────────────┬───┘
                       │          │          │             │
              ┌────────▼──┐  ┌────▼─────┐  ┌─▼──────────┐ ┌▼────────────┐
              │  PiperIK  │  │  Safety  │  │  Domain    │  │  MuJoCo    │
              │  DLS,5-DoF│  │  Layer   │  │ Randomizer │  │  MjModel   │
              └───────────┘  └────┬─────┘  └────────────┘  └────────────┘
                                  │  same object, same thresholds
                    ┌─────────────▼────────────────────────────┐
                    │           RobotInterface (ABC)           │
                    ├──────────────────┬───────────────────────┤
                    │   SimBackend     │  PiperHardware (STUB) │
                    │   (MuJoCo)       │  (piper_sdk)          │
                    └──────────────────┴───────────────────────┘
```

Three principles:

1. **The safety layer is not a wrapper, it is the write path.** The same
   `SafetyLayer` instance type validates every command in simulation and in the
   hardware interface, with the same limits object. By the time hardware is
   connected, the layer has vetoed millions of commands and its thresholds are
   known to be neither too tight (the policy still learns) nor useless.
2. **Everything the policy touches is configurable and documented in one file.**
   `config.py` holds every number, each with a comment saying what it is, why
   it has that value, and whether it matters for sim-to-real.
3. **The sim/hardware boundary is exactly one file.** A policy runner written
   against `RobotInterface` does not change between the two.

---

## 3. What changed, concretely

### MJCF (`agilex_piper/piper.xml`)

| change | reason |
|---|---|
| **added the `forehead` camera** on `base_link` at (−0.25, 0, 0.30), 65° FoV, aimed at (0.36, 0, 0.21) | the requested fixed first-person view. Mounted closer in (≤ 15 cm behind the base) the arm's own `link2` fills the frame at the home pose and the object is never visible — this position was chosen by rendering candidates. |
| added cosmetic `cam_mast` / `cam_head` geoms, **behind** the lens | so the camera is visible in third-person renders rather than being a floating viewpoint. Placing the housing in *front* of the lens fills the entire forehead image with a black blob — it was there first, and it did. |
| moved the `tcp` site to z = 0.105 in the `link6` frame | measured midpoint of the volume between the finger pads (they span 0.075–0.135). The previous 0.09 sat at the inner pad, the original 0.17 was past the fingertips entirely. Both made IK targets drive the fingers into the object. |
| **named** the four finger-pad geoms and the seven arm collision geoms | grasp detection and collision penalties need to reason about contact pairs by name. |
| gripper actuator `kp` 300 → 200, force 25 → 20 N | at `kp=300` the squeeze stored enough elastic energy that *opening* the fingers launched the cup across the table. 200/20 N still holds a 120 g cup with a wide margin. |
| finger pads given softer, better-damped contacts (`solref="0.012 1.4"`) and `condim=4` | same reason, plus more realistic pad compliance. |

### MJCF (`agilex_piper/piper_task.xml`)

| change | reason |
|---|---|
| object: 42 × 90 mm block → **48 × 70 mm cylinder, 60 g** | two hard constraints. Width ≤ 55 mm because the gripper's inner opening is 70 mm and an exploring policy needs clearance. Height ≤ ~90 mm because the palm sits 46 mm above the TCP — the old 90 mm object was struck by the palm before the fingers could close. |
| `target` site + `pad_dst`/`pad_src` geoms addressed by name, moved every reset | the target is randomised; the reward reads the site. |
| named `key_light` / `fill_light`, textured table | lighting and appearance randomisation. |
| `offwidth`/`offheight` raised to 1280×960 | headless rendering of eval videos at 640×480 was silently capped. |

### New: `piper_rl/`

| module | contents |
|---|---|
| `config.py` | `EnvConfig`, `RewardConfig`, `DomainRandConfig`, `NoiseConfig`, `RobotLimits` — every parameter, documented. |
| `ik.py` | Damped-least-squares IK on a **5-DoF task** (position + approach *direction*, roll free), with adaptive damping and a faded null-space term. Converges < 1 mm across the whole task workspace where the original full-pose solver stalled at 50–90 mm. Plus `diff_ik()`, a 2-iteration velocity-level solver used once per control step. |
| `piper_env.py` | The Gymnasium environment. |
| `domain_rand.py` | Reset-time randomisation of object, scene, appearance and actuator dynamics, always re-derived from a pristine copy of the model so it never compounds. |
| `safety.py` | Command validator: finite → joint position → joint velocity → joint acceleration → gripper → Cartesian envelope → Cartesian speed. Clamps in sim, vetoes on hardware. |
| `callbacks.py` | SB3 callbacks that log success / grasp / lift / placement-error, not just `ep_rew_mean`. |
| `hardware/` | `RobotInterface` ABC, `SimBackend`, and the `PiperHardware` stub. |
| `scripts/` | `validate_model`, `scripted_demo`, `train`, `evaluate`. |

### Kept as-is

`assets/`, `scene.xml`, `README.md`, `CHANGELOG.md`, `LICENSE`,
`make_views_page.py`, `piper_views.html`. `view_piper.py` gained a `--camera`
argument. `validate_piper.py` is superseded by
`piper_rl/scripts/validate_model.py` but left in place.

---

## 4. Design decisions worth arguing about

**Cartesian actions rather than joint actions (default).** Learning a top-down
pick in joint space means the policy must discover the arm's kinematics before
it can discover the task, and it must do so inside a joint-limit box where large
parts of the natural approach are infeasible. Constraining the gripper to point
down and handing the policy TCP deltas removes both problems and still maps onto
the SDK's end-pose command mode. `action_mode="joint"` is implemented for anyone
who disagrees.

**A reverse curriculum on the initial state.** 45 % of training episodes start
from a later phase of the task (near the object, holding it, or holding it over
the target). This changes the initial-state distribution only — not the task,
not the reward — and it is the difference between ~10⁵ and ~10⁷ steps to a
working policy. **Every evaluation episode starts from the true initial state**,
enforced by `EnvConfig.eval_variant()`.

**State observations for now, pixels wired but untrained.** A vision policy is
what you actually want on hardware, and the env supports it end to end
(`obs_mode="rgb"`, forehead camera, brightness jitter, Gaussian noise, latency).
It needs a GPU and ~5–20 M steps. The state-based policy is the staging step,
and it comes with a real cost that `docs/SIM2REAL.md` is blunt about: it needs an
object-pose estimator on hardware, which does not exist yet.

**`gravcomp="1"` left on.** The Menagerie model perfectly cancels gravity. Real
servos do not. This makes sim easier than reality; the actuator-gain
randomisation covers part of the gap, and the bring-up plan measures the rest.

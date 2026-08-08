# Piper human-aware pick-and-place

## Research question

`PiperHumanAwarePickPlaceEnv` studies whether an AgileX PiPER policy can finish
pick-and-place while a human arm enters a shared workspace, including when it is
safer to continue, alter motion, wait, or resume. This repository supplies the
simulation and measurement infrastructure; it does not claim learned HRI
performance without a separately reported training and evaluation run.

Gymnasium ID: `PiperHumanAwarePickPlace-v0`.

## Environment structure

The environment subclasses `PiperPickPlaceEnv`. Small no-op hooks in the base
class extend reset, physics substeps, state observations, reward, termination,
`info`, and episode summaries. The existing task, action/IK pipeline, domain
randomization, manipulation reward, safety layer, and rendering are shared.
`PiperPickPlace-v0` and all original commands remain unchanged.

The active scene is `agilex_piper/piper_human_task.xml`. It includes the
unchanged `piper_task.xml` and adds three independently controlled mocap bodies:

| Body | Geometry | Representation |
|---|---|---|
| `human_upper_arm` | `human_upper_arm_geom` | orange capsule |
| `human_forearm` | `human_forearm_geom` | orange capsule |
| `human_hand` | `human_hand_geom` | orange capsule |

The environment moves and reorients the capsules at every 2 ms physics substep.
Soft contacts reduce impulses; identified robot-human contact is handled before
the next policy step. Inactive arms are parked beyond the side of the table.

## Human trajectory generator

`piper_rl/human_motion.py` supports `cross_workspace`, `reach_object`,
`reach_target`, `pause_and_continue`, and `reverse`. Piecewise minimum-jerk
interpolation has zero velocity at phase boundaries. Each state supplies
shoulder, elbow and hand positions, elbow and hand velocities, presence, and a
phase label. Appearance time, peak speed, start side, end jitter, closest task
approach, pauses, reversals, and trajectory type are sampled only from the
Gymnasium-seeded NumPy generator. Resetting with the same seed reproduces the
same human trajectory.

## Action space

Actions are identical to `PiperPickPlaceEnv`: Cartesian mode is 5 values
`[dx, dy, dz, dyaw, gripper]`; joint mode is 7 values
`[dq1..dq6, gripper]`. All values are normalized to `[-1, 1]` and use the same
IK, smoothing, actuation noise, command dropout, and safety checking.

## Observation layout

Privileged simulator state is used for the first version. With Cartesian
actions, the original state is 51 float32 values. With joint actions it is 53
because the previous-action suffix has 7 rather than 5 values. See
`piper_rl/README.md` for the exact original slices.

When `include_human_state=True`, these 13 values are appended programmatically:

| Relative slice | Size | Value | Unit |
|---|---:|---|---|
| `0:1` | 1 | human present flag | binary |
| `1:4` | 3 | hand world position | m |
| `4:7` | 3 | hand world velocity | m/s |
| `7:10` | 3 | elbow world position | m |
| `10:13` | 3 | hand minus TCP position | m |

Thus Cartesian human-aware state is `(64,)`; joint human-aware state is `(66,)`.
Position and velocity noise use `human_position_noise` and
`human_velocity_noise`. The entire human suffix is delayed by
`human_observation_latency_steps` when global observation noise is enabled.
The shape is fixed at construction.

With `include_human_state=False`, state is exactly the original shape, allowing
existing SAC policies to run while the human is visible and moving. In
`state+rgb`, only `state` is extended. In `rgb`, the array shape is unchanged;
the orange human arm is rendered by the forehead and overview cameras.

## Reward and safety cost

The manipulation reward is unchanged. Human reward additions are:

```text
x = max(0, (safe_distance - surface_distance) / safe_distance)
r_human = -proximity_penalty_weight * x^2
          -human_collision_penalty * first_robot_human_collision
          -human_time_penalty
r_total = r_pick_place + r_human
```

No positive reward is paid for staying far from the human. The optional time
penalty discourages permanent freezing and defaults to zero. `human_safety_cost`
is reported separately from total RL reward as the positive accumulated
proximity and collision cost.

Distance is the minimum MuJoCo geometry-to-geometry surface distance across
robot collision/finger geoms and all human geoms, using `mj_geomDistance`.
Robot-human contact, object-human contact, robot-table contact, and robot
self-contact remain separately identifiable. Episode metrics include current
and minimum distance, human collision and collision steps, near-miss entry
events, safe-distance steps, waiting steps, cumulative safety cost, and
collision-free success.

## Termination

The original success, object-lost, unsafe-state, optional drop, and time-limit
conditions remain. A robot-human collision additionally terminates when
`terminate_on_human_collision=True`. Task success retains its original
definition. `collision_free_success` means task success with no human collision.

## Training and evaluation distributions

Presets are `no_human`, `fixed`, `fixed_exact_h036_v1`, `randomized`,
`curriculum`, `evaluation_id`, and `evaluation_ood`. `fixed` is the historical
constrained fixed-type crossing: it fixes type, side, timing, speed, path, and
approach but samples hand height uniformly in `[0.30, 0.42]` m. It must be
labelled `constrained_fixed_height_randomized` in new reports and must not be
described as an exact fixed trajectory. `fixed_exact_h036_v1` is the separate,
authoritative exact crossing: it preserves the historical crossing parameters
and sets hand height to exactly `0.36` m on every human-present reset. Its
trajectory fingerprint is stable across seeds; object, target, manipulation,
domain-randomization, and sensor-noise factors remain independently seeded.
`fixed_exact_shifted_h036_v1` is a predeclared holdout with a 0.2 s earlier
appearance, 0.02 m/s higher speed, and 0.01 m closer approach, all at 0.36 m.

Curriculum changes only the ID episode probability,
speed, appearance window, proximity, pause rate, and reversal rate from
difficulty 0 to 1. It never selects the OOD preset.

OOD evaluation is explicitly separate and contains higher speeds, the opposite
approach side, earlier or later starts, closer approaches, longer pauses, more
reversals, and three control steps of human-state latency. Results should always
retain the `distribution` label in the generated CSV/JSON.

## Validation

```powershell
python -m piper_rl.scripts.validate_model
python -m piper_rl.scripts.validate_human_env
python -m pytest -q tests
```

The human validator checks every trajectory, seeds, continuity, peak speed,
initial contacts, 100 random actions, finite outputs, Gymnasium conformance,
both requested cameras, and writes `out/human_trajectory_validation.mp4` with
trajectory labels.

## Training

Original default, unchanged:

```powershell
python -m piper_rl.scripts.train --task pick-place --algo sac --timesteps 1500000 --run-name sac_pick_place
```

Human-aware fixed and randomized examples:

```powershell
python -m piper_rl.scripts.train --task human-aware --human-preset fixed --algo sac --timesteps 1500000 --run-name human_fixed
python -m piper_rl.scripts.train --task human-aware --human-preset fixed-exact-h036-v1 --human-difficulty 1.0 --algo sac --timesteps 2000000 --n-envs 4 --seed 0 --run-name human_aware_fixed_s0
python -m piper_rl.scripts.train --task human-aware --human-preset randomized --algo sac --timesteps 2000000 --n-envs 4 --run-name human_randomized
python -m piper_rl.scripts.train --task human-aware --human-preset curriculum --human-difficulty 0 --algo sac --timesteps 2000000 --n-envs 4 --run-name human_curriculum
```

Use `--no-human-state` to train or evaluate a state-compatible unaware baseline.

## Evaluation

Each command requires an explicit seed and exports `<out>.csv` and `<out>.json`:

```powershell
python -m piper_rl.scripts.evaluate_human_aware --model runs/sac_full/best/best_model.zip --experiment no-human --episodes 50 --seed 100 --out out/existing_no_human
python -m piper_rl.scripts.evaluate_human_aware --model runs/sac_full/best/best_model.zip --experiment human-unaware --episodes 50 --seed 100 --out out/existing_human_unaware --video out/existing_human_unaware.mp4
python -m piper_rl.scripts.evaluate_human_aware --model runs/human_randomized/best/best_model.zip --experiment fixed --episodes 50 --seed 100 --out out/human_fixed
python -m piper_rl.scripts.evaluate_human_aware --model runs/human_aware_fixed_s0/best/best_model.zip --experiment fixed-exact-h036-v1 --episodes 200 --seed 31000 --out out/human_exact_h036
python -m piper_rl.scripts.evaluate_human_aware --model runs/human_aware_fixed_s0/best/best_model.zip --experiment fixed-exact-shifted-h036-v1 --episodes 200 --seed 32000 --out out/human_shifted_exact_h036
python -m piper_rl.scripts.evaluate_human_aware --model runs/human_randomized/best/best_model.zip --experiment randomized-id --episodes 100 --seed 200 --out out/human_id
python -m piper_rl.scripts.evaluate_human_aware --model runs/human_randomized/best/best_model.zip --experiment ood --episodes 100 --seed 300 --out out/human_ood
```

Prediction is deterministic unless `--stochastic` is supplied. Reports include
task success, collision-free success, human collision and near-miss rates,
Wilson binomial confidence intervals, separation percentiles, successful-only
completion time, episode duration, placement-error percentiles, grasp/lift
rates, safety-clamp rate, waiting/proximity metrics, and safety cost. For the
`no-human` experiment, the evaluator infers whether the saved policy expects 51
or 64 state values; `--no-human-state` explicitly forces the legacy layout.

## Known limitations

- Human poses are privileged simulator state, not camera estimates.
- The arm is kinematic and does not model compliant human reactions or injury.
- Three capsules are not a full biomechanical or whole-body human model.
- Waiting is measured behavior, not a dedicated discrete action.
- The task has no communication, gaze, intent inference, or uncertainty model.
- Human-object contact is measured but the human does not grasp the object.

Future work should add camera-based human pose estimation with uncertainty and
latency, then a distinct human-to-robot handover task with compliant contact and
explicit intent cues.

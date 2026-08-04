# Sim-to-real: moving the PIPER policy onto hardware

**Nothing in this repository commands a real robot.** `piper_rl/hardware/
piper_real.py` is a stub whose every motion method raises `NotImplementedError`.
This document is the plan for finishing it, in the order the steps must happen.

Read section 0 before anything else.

---

## 0. What is *not* solved

Be blunt with yourself about these before you plug anything in.

### 0.1 Object pose estimation — the blocker

The trained policy consumes the **object's 3-D position and orientation** and
the **target's position** (obs indices 23–34). Simulation hands these over for
free. On hardware they do not exist. Until you supply them, a state-based policy
cannot run at all.

Three ways out, cheapest first:

| approach | effort | accuracy | notes |
|---|---|---|---|
| **Fiducial marker** (AprilTag / ArUco on the cup and the target) | low | 2–5 mm | Calibrate the forehead camera intrinsics + the camera→base extrinsic. This is the recommended first deployment. |
| **Fixed fixtures** — cup and target always at surveyed positions | lowest | exact | Removes the perception problem entirely but also removes the point of the policy. Useful for the very first hardware run. |
| **Vision policy** — retrain with `obs_mode="rgb"` | high | n/a | The env already supports it end to end (forehead camera, noise, latency). Needs a GPU and roughly 5–20 M steps. Removes the perception module but adds a sim-to-real *appearance* gap that the current visual randomisation is probably not wide enough to cover on its own. |

Whichever you choose, the pose must arrive **in the robot base frame**, matching
the sim's convention (base flange at the origin of the arm, +x forward,
+z up), and with a latency you have measured.

### 0.2 Other known gaps

| gap | why it matters | mitigation already in the env |
|---|---|---|
| **Gravity compensation.** The MJCF sets `gravcomp="1"` on every link, so sim gravity is perfectly cancelled. The real servos compensate in firmware, but imperfectly and temperature-dependently. | Expect a small static droop, worst at full extension. | Actuator gain randomisation (kp ±25 %) covers part of it. Measure the real droop and, if it exceeds ~5 mm, either set `gravcomp="0"` and add the real payload, or add a static feed-forward offset in the hardware backend. |
| **Contact model.** MuJoCo's soft contacts with `solref`/`solimp` are a fit, not physics. Grasp stability in sim is optimistic. | The grasp may slip on hardware where it holds in sim. | Object friction is randomised 0.8–2.0 and the pads are compliant. Start with a high-friction cup (rubber-based mug) on hardware. |
| **Gripper force.** The sim gripper is a position servo, kp 200, ±20 N. The real PIPER gripper takes an angle *and an effort*. | Over-squeezing crushes a paper cup; under-squeezing drops it. | Map the sim's commanded joint angle to the SDK angle, and pick the effort empirically starting low. |
| **Encoder zeroing / joint sign.** The MJCF joints come from the official URDF, so ordering and signs *should* match the SDK — but "should" is not "do". | A single flipped sign drives the arm into the table at full speed. | Step 3 below tests exactly this, one joint at a time, with the arm parked in free space. |
| **Camera extrinsics.** The sim camera is at exactly (−0.25, 0, 0.30) in the base frame with a 65° vertical FoV. | If the physical bracket differs, a vision policy sees a different world. | Either build the bracket to match, or measure the real transform and edit `piper.xml` to match *it*, then retrain. |
| **Latency budget.** The env models 1 control step (50 ms) of observation latency. | Real perception pipelines are often 80–150 ms. | Measure yours. If it exceeds 50 ms, raise `NoiseConfig.obs_latency_steps` and retrain — do not deploy a policy trained on optimistic latency. |
| **Table height.** The sim table top is at z = 0.20 with the arm bolted to it at the origin. | A different mounting height changes the whole workspace. | Edit `piper_task.xml` (`table` geom, `obj` z, `keyframe`) and re-run `validate_model.py --map` before retraining. |

---

## 1. Prerequisites

- [ ] Physical **e-stop in series with the arm's supply**, tested by pressing it
      while the arm is holding position. A software `DisableArm` is a backup,
      not the primary.
- [ ] Arm bolted down, workspace cleared, nothing fragile within reach.
- [ ] `piper_sdk` installed and CAN up (`ip link show can0`).
- [ ] The forehead camera bracket built to the sim geometry, or the sim edited
      to the real geometry.
- [ ] A **sim result you actually believe**: ≥ 80 % success over 50 evaluation
      episodes with domain randomisation and noise **on**, and the videos
      inspected. A number without a video is not a result.

---

## 2. Finish the hardware backend

Implement `piper_rl/hardware/piper_real.py`. Every method carries a docstring
naming the SDK call it needs. Rules:

1. **Keep `dry_run=True` until step 5.** In dry-run every command is filtered,
   validated and logged but never put on the wire.
2. **Use `conservative_limits()`** (already the default): 0.30 rad/s joint
   speed, 0.12 m/s TCP, 15 N contact, 3 cm table clearance. Roughly a third of
   the sim limits.
3. **Set `veto_on_violation=True`** (already the default for hardware). In sim
   the safety layer clamps; on hardware it must refuse and hold.
4. **Never bypass `send_joint_command()`.** It is the only write path and it
   applies the same EMA filter and the same `SafetyLayer` the policy trained
   with.

---

## 3. Bring-up, in order

Do not skip a step because the previous one looked fine.

### Step 1 — Read-only

Arm **disabled**. Connect, stream `get_state()` at 20 Hz for a few minutes.

- [ ] Six joint values, plausible units (radians), stable when the arm is still.
- [ ] Move each joint **by hand** and confirm the reported value moves in the
      direction and with the sign the MJCF uses. Compare against
      `mj_forward` on the same joint vector.
- [ ] Gripper reads 0 closed, ~0.035 open.
- [ ] Log the timestamp jitter — you need it for the latency budget.

### Step 2 — Forward-kinematics agreement

Still disabled, still by hand.

- [ ] For ≥ 10 hand-set poses, compare the SDK's reported TCP (if it has one)
      and `PiperIK.forward(q)` against a **tape measure**. Agreement to within
      5 mm across the workspace. Disagreement here means a frame convention
      mismatch, and everything downstream is meaningless until it is fixed.

### Step 3 — Single-joint motion, arm in free space

Enable. Arm posed so **no joint can reach the table or itself**.

- [ ] One joint at a time, ±0.05 rad, at `conservative_limits()`. Hand on the
      e-stop.
- [ ] Confirm each joint moves the direction you commanded, and that the
      measured position converges to the command.
- [ ] Measure the steady-state tracking error per joint. If any joint droops
      more than ~0.02 rad, that is the gravity-compensation gap (§0.2).

### Step 4 — Scripted trajectory, dry run then live

- [ ] Run `scripted_demo`'s waypoint plan through the hardware interface with
      `dry_run=True`. Plot the logged joint trajectory against the same plan run
      in sim. They should overlay.
- [ ] Repeat live, **with no object on the table**, at conservative limits.
      Watch for: overshoot, oscillation at the waypoints, the palm approaching
      the table.
- [ ] Add the object. Confirm the scripted controller picks and places it. If
      the *scripted* controller cannot do it on hardware, a learned policy
      certainly will not, and the problem is mechanical or calibration — not RL.

### Step 5 — Policy, dry run

- [ ] Load the trained policy. Feed it real observations (including real object
      poses from your perception module). Keep `dry_run=True`.
- [ ] Log the action stream for ≥ 50 episodes' worth of real states. Check:
      actions are inside [−1, 1] and not saturated every step; the resulting
      joint targets are smooth; the safety layer's veto count is near zero.
      **A high veto rate here means the policy is asking for something the
      hardware will not do — fix that before going live.**

### Step 6 — Policy, live, no object

- [ ] `dry_run=False`, conservative limits, no object, hand on the e-stop.
- [ ] Confirm the arm reaches towards where the (absent) object is reported and
      does not thrash.

### Step 7 — Policy, live, with the object

- [ ] Light, soft, cheap object first. Same nominal position every time.
- [ ] 10 episodes. Record video from the forehead camera. Log everything.
- [ ] Only then start randomising the object position, and only within the
      region you validated in sim (r 0.30–0.39 m, ±42°).

### Step 8 — Raise the limits

Only after ≥ 20 consecutive successful hardware episodes:

- [ ] Raise `max_joint_vel` in ~25 % increments toward the sim value (1.0 rad/s),
      re-running step 7 at each level.
- [ ] Never raise `max_contact_force` above what the object tolerates.

---

## 4. When hardware performance is worse than sim (it will be)

Diagnose in this order — the cheapest fix that could explain it, first.

1. **Is the scripted controller also worse?** If yes, the problem is
   calibration, mounting, or the contact model — not the policy. Fix that.
2. **Is the object pose wrong?** Compare your perception output against a tape
   measure at 10 positions. A 2 cm bias is invisible in the logs and fatal to
   the grasp.
3. **Is the latency worse than you trained with?** Measure it end to end
   (photodiode or a timestamped LED works). If it is > 50 ms, raise
   `NoiseConfig.obs_latency_steps` and retrain.
4. **Is the action stream saturating?** If the policy commands ±1 every step,
   the hardware's rate limits are truncating the intended motion. Either raise
   the limits (carefully) or retrain with matching limits.
5. **Only then**: widen domain randomisation and retrain. The usual first
   candidates are `actuator_kp_scale`, `joint_frictionloss_scale` and
   `obj_friction`. Widening randomisation costs sim performance, so widen the
   *specific* parameter you have evidence about, not everything.

---

## 5. Recommended deployment order

```
scripted controller in sim         <- already passing (see the README)
        |
policy in sim, no randomisation
        |
policy in sim, full randomisation  <- the number you report
        |
policy on hardware, dry run
        |
policy on hardware, fixed object position, conservative limits
        |
policy on hardware, randomised object position
        |
policy on hardware, full speed
```

Each arrow is a gate. Do not pass one because you are in a hurry.

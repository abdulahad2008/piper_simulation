# Training report — what actually happened

**Bottom line: the RL policy does not solve the task yet.** The environment,
the reward, the safety layer and the evaluation harness are all validated and
working, and a scripted controller solves the task through the same action space
100 % of the time. But the SAC policy trained here — 220 k steps on 2 CPU cores —
places the cup on the target in **0 of 30** deterministic evaluation episodes.

This document reports that honestly, explains why, and gives you the command to
finish the job on hardware that can actually do it.

---

## 1. What is verified to work

### Environment health — `python -m piper_rl.scripts.validate_model`

```
[ ok ] MJCF compiles                       nq=15 nv=14 nu=7 ngeom=95
[ ok ] every named handle present          sites, bodies, geoms, cameras, actuators, joints, lights
[ ok ] action scale consistent with the TCP speed limit   0.03 m/step * 20 Hz = 0.60 m/s vs cap 0.60
[ ok ] control timestep divides cleanly    2.0 ms physics, 50 ms control
[ ok ] home keyframe collision-free
[ ok ] object fits the open gripper        48 mm object vs 70 mm opening (11 mm/side)
[ ok ] object clears the gripper palm      35 mm half-height vs ~46 mm palm clearance
[ ok ] task workspace fully reachable      worst straight-down IK residual 0.9 mm
[ ok ] stable_baselines3 env_checker       obs (51,), act (5,)
[ ok ] deterministic given a seed          max reward delta 0.0
[ ok ] observations and rewards finite
[ ok ] solving beats hovering              margin 286.2
[ ok ] hovering is not profitable          -2.7
        throughput                         2.25 ms/step -> 445 steps/s, single env
```

### Reward-exploit test

| behaviour | episode reward |
|---|---|
| do nothing | **−12.0** |
| hover over the object and bob up and down (the classic shaping exploit) | **−2.7** |
| scripted solution | **+283.5** |

Neither degenerate strategy is profitable, and the solution beats the best
exploit by 286. This test is part of `validate_model.py` and runs every time.

### Scripted controller — the task *is* solvable

Same 5-D action space the policy uses; open-loop Cartesian waypoints.

| | clean sim | full domain randomisation + sensor noise |
|---|---|---|
| grasp rate | **100 %** (15/15) | 60 % (15/25) |
| lift rate | **100 %** | 60 % |
| **success rate** | **100 %** | **60 %** |
| placement error | **6.9 mm** mean | 117 mm mean |
| episode length | 108 steps (5.4 s) | 143 steps |

The 60 % under randomisation is expected: this controller is **open loop** — it
computes its waypoints once at reset from the true object pose and never reacts.
A closed-loop policy should beat it comfortably. It is a solvability proof and a
lower bound, not a target.

Video: `out/scripted_forehead.mp4`, stage-by-stage sheet: `out/scripted_stages.png`.

---

## 2. The RL run

```
algorithm      SAC (MlpPolicy, net [256,256], batch 256, lr 3e-4, gamma 0.98,
               tau 0.01, ent_coef auto_0.1, buffer 400k, 1 grad step / env step)
steps          220 000
wall clock     ~1 h 55 m on 2 CPU cores, no GPU  (~44 env steps/s, update-bound)
env            cartesian actions, state observations, domain randomisation ON,
               sensor noise ON, reverse curriculum ON
```

### Deterministic evaluation — 30 episodes, true initial state, DR + noise on

```
SUCCESS RATE          0.0%   (0/30)
grasp rate            0.0%
lift  rate            0.0%
episode reward      -11.51 +- 4.31
episode length       200.0 steps   (i.e. every episode timed out)
placement error     mean 294.2 mm | median 290.9 mm | p90 404.3 mm
collision steps       0.0 / episode
safety vetoed         0
```

The mean placement error of 294 mm is essentially the *initial* object-to-target
distance (~280 mm): the policy does not move the cup. Episode reward −11.5 is
indistinguishable from the do-nothing baseline of −12.0.

### The training curve did improve

Deterministic evaluation reward, measured every 10 k steps:

```
 10k  -30.6      100k  -13.6
 20k  -23.8      130k  -14.3
 40k  -20.0      160k  -30.9
 60k  -23.7      190k  -10.4
 80k  -23.7      220k  -19.5
```

Rising from −31 to about −13, but it plateaus at roughly the do-nothing level
and never gets near the +283 a solution scores. Training-rollout `grasp_rate`
reached 0.44, **but that number is inflated and should not be quoted**: 28 % of
training episodes start with the cup already in the gripper (the curriculum), and
those count as a grasp at reset. The honest grasp number is the evaluation one:
**0 %**.

### Why it did not get there

1. **Compute.** 220 k steps is one to two orders of magnitude short. SAC on
   comparable MuJoCo pick-and-place tasks typically needs 1–3 M steps. On 2 CPU
   cores that is 6–20 hours; the gradient update, not the simulator, is the
   bottleneck (445 env steps/s standalone, 44 with learning attached).
2. **One run, one seed, no tuning.** No hyper-parameter search, no seed sweep.
3. **A handicap for most of the run.** `max_step_dist` (0.03 m/step = 0.6 m/s at
   20 Hz) exceeded `max_tcp_speed` (0.4 m/s), so the safety layer rate-limited
   198 of every 200 commands and the policy's effective action scale was a third
   smaller than nominal. Found while auditing this evaluation; now fixed, and
   `validate_model.py` asserts the relationship. **The reported run predates the
   fix**, so it is a pessimistic data point.

### Bugs found and fixed along the way

Each of these was found by measurement, not inspection, and each would have
capped final performance on its own:

| bug | symptom | fix |
|---|---|---|
| IK solved a full 6-DoF pose | stalled 50–90 mm / 20–40° out; the original `validate_piper.py` never picked the object up | 5-DoF task (position + approach direction), adaptive damping |
| waypoints outside the reachable set | lift height z = 0.42 unreachable straight-down (real limit ≈ 0.376) | reachability map; carry height 0.305 |
| object taller than the palm clearance | `link6` collision capsule struck the cup before the fingers could close | object 90 mm → 70 mm tall |
| gripper too stiff | opening the fingers launched the cup across the table | kp 300 → 200, force 25 → 20 N, compliant pads |
| safety envelope excluded the home pose | 100 % of commands clamped, including at reset | separated the *safety* envelope from the *task* workspace |
| TCP-speed check froze the arm | hard block on a limit exceeded by the nominal action scale | rate-limit (scale the delta) instead of blocking |
| gripper command seeded from the *measured* finger position | the grasp went slack and the cup dropped on step 1 of every curriculum episode that started holding it | seed from the commanded value |
| **alignment reward sign flipped** | paid ~+16/episode for tilting the gripper *up*; hovering scored ≈ 0 and beat attempting the task | negated; now a penalty only, with an assertion |
| dense shaping smaller than the regularisers | the whole reach was worth +3.6 while action + time penalties over the same window cost −6, so the gradient pointed at "sit still" | shaping weights raised 3–4× |
| episode budget flat at 250 steps | only ~200 episodes in the first 40 k steps, so very few terminal events for an off-policy learner | phase-dependent budgets (200/120/100/70) |

---

## 3. What to run next

On a machine with a GPU (or just more time), from `piper_sim/`:

```bash
# the full run -- this is the command that should actually produce a policy
python -m piper_rl.scripts.train --algo sac --timesteps 2000000 \
       --run-name sac_full --net-arch 512 512 256 --batch-size 512 \
       --eval-freq 25000 --n-eval-episodes 20

tensorboard --logdir runs/sac_full/tb        # watch rollout/success_rate

python -m piper_rl.scripts.evaluate --model runs/sac_full/best/best_model.zip \
       --episodes 50 --video eval.mp4 --camera forehead
```

**Do not believe the result until `evaluate.py` reports it and you have watched
the video.** `rollout/success_rate` in TensorBoard is measured on training
episodes and is inflated by the curriculum; `eval/` and `evaluate.py` are not.

### If it is still not learning at ~1 M steps

In rough order of expected value:

1. **Turn domain randomisation off** (`--no-domain-rand --no-noise`) and confirm
   it learns the clean task first. If it cannot solve the clean task, the
   randomisation is not the problem. Re-enable afterwards and expect a dip.
2. **More parallel envs** (`--n-envs 8`) with `--gradient-steps 8`. SAC scales
   fine and this is nearly free on a GPU.
3. **Try TD3** (`--algo td3`). SAC's entropy bonus can keep the gripper action
   too stochastic to ever hold a stable grasp.
4. **Goal-conditioned + HER.** This task is a natural fit: refactor the
   observation into a `Dict` with `achieved_goal` = object position and
   `desired_goal` = target position, and use SB3's `HerReplayBuffer`. On sparse-
   reward pick-and-place HER is usually worth more than any amount of shaping.
   This is the single biggest available win and it is not implemented here.
5. **Behaviour cloning warm start.** `scripted_demo.py` already produces
   successful trajectories through the exact policy action space. Pre-train the
   actor on a few thousand of them, then fine-tune with SAC.
6. **Widen the curriculum** (`p_start_near_object` up, `p_start_reaching` down)
   for the first few hundred thousand steps, then anneal it back.

---

## 4. Reproducing the numbers in this report

```bash
python -m piper_rl.scripts.validate_model                      # section 1
python -m piper_rl.scripts.scripted_demo --episodes 15 --no-domain-rand --no-noise
python -m piper_rl.scripts.scripted_demo --episodes 25         # with DR + noise
python -m piper_rl.scripts.evaluate --scripted --episodes 50   # baseline
python -m piper_rl.scripts.evaluate --model <ckpt> --episodes 30
```

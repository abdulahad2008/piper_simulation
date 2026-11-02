# Stage A: learner screening at equal wall clock (cup task, 1 seed)

Run on the PC (Windows 10, 48 threads, torch 2.14.0+cpu, sbx-rl/JAX CPU) from 31 Oct 00:45 to 06:45, seven configurations at once, 6 h budget each, `run_stage_a.sh` at commit e29747a. Human-aware cup task, `curriculum` preset, corrected arm. Validation: 50 episodes on the full task (curriculum off) every 40 k steps. Curves: `<config>.json` (one row per validation), figure `stage_a_figure.png`, table `stage_a_curves.json`.

| config | learner | updates per env step | steps/s | steps in 6 h | best success | at steps | h to 2 M at this rate |
|---|---|---:|---:|---:|---:|---:|---:|
| sb3_sac_utd1 | SB3 SAC (the October setting) | 1 | 25.5 | 520 k | 2 % | 160 k | 21.8 |
| sb3_sac_utd025 | SB3 SAC | 0.25 | 57.4 | 1.24 M | 2 % | 720 k | 9.7 |
| sbx_sac_utd1 | SBX SAC | 1 | 33.6 | 720 k | 14 % | 600 k | 16.6 |
| sbx_sac_utd05 | SBX SAC | 0.5 | 54.4 | 1.16 M | 38 % | 1.12 M | 10.2 |
| sbx_sac_utd025 | SBX SAC | 0.25 | 75.6 | 1.64 M | 12 % | 1.60 M | 7.3 |
| sbx_crossq_utd1 | SBX CrossQ | 1 | 22.7 | 480 k | 0 % | – | 24.4 |
| ppo16 | SB3 PPO, 16 envs | – | 164.4 | 2.0 M (3.4 h) | 4 % | 1.44 M | 3.4 |

October campaign for comparison (`results/human_aware_curriculum_hold_s{3,4,5,7}`, SB3 SAC, 1 update per env step, 4 envs, 15-episode validations, curriculum ramp 1.2 M): first validation ≥ 50 % at 0.90 / 1.04 / 1.24 / 0.96 M steps; ≥ 80 % at 0.98 / 1.20 / 1.70 / 1.18 M; 15.4 h per run at 34–38 steps/s with about six runs at once.

## What it shows

1. **No configuration reaches 50 % in 6 h.** [Certain] The task needs about 1–1.5 M steps to cross 50 % (October seeds), and nothing here gets 1.5 M useful steps in 6 h. The 6 h window in the plan was too short for this task; a Stage B budget must be 10–12 h or the pass mark is never met.
2. **Best at 6 h: SBX SAC with 0.5 updates per env step (38 %, 1.12 M steps).** Per step it tracks the October curves' lower seeds; per hour it is the only one that climbs. Projection to 2 M steps: 10.2 h, i.e. 1.5× the October run, not the 3× target. [Certain for the measurement; Guessing for the projection]
3. **0.25 updates per env step loses learning per step.** [Likely] SBX at 0.25 reached 1.6 M steps but only 12 %; the October runs were at 73–100 % by 1.6 M. SB3 at 0.25 reached 1.24 M at 2 %. Cheaper steps are not worth it at this ratio.
4. **Pinning SB3 SAC to one thread made it slower than October** (25.5 vs 34–38 steps/s). [Certain] The October runs used unmanaged torch threads with six runs at once; one thread per run with seven runs at once is worse for torch. Lever 1 of the plan is a loss for SB3 and a gain only for JAX.
5. **SBX SAC is 1.3× SB3 SAC end to end at equal ratio** (33.6 vs 25.5 steps/s), below the 1.65× per-update gain because the environment step (17 ms on the PC) is now a large share of each step.
6. **PPO with 16 environments: 2 M steps in 3.4 h and 0–4 % success.** [Certain] The "why not PPO" answer for this task, one seed.
7. **SBX CrossQ is the slowest run and learned nothing in 480 k steps.** Not enough steps to judge its sample efficiency; dropped on wall clock.

## Caveats that limit the ranking

- One seed. The October seeds at 1.2 M steps spread from 27 % to 93 %, so a single run's position at 6 h is inside seed noise. The ordering by rate (steps/s) is firm; the ordering by learning is not.
- Curriculum ramp confound: Stage A used the default ramp (full 2 M steps); the October campaign ramped to full difficulty by 1.2 M. At any step count Stage A trained on an easier distribution, which does not favour any Stage A configuration over another but makes the grey curves an optimistic comparison for per-step learning.
- Validation with 50 episodes is ±7–14 points at the rates seen.
- All seven ran at once (45 % CPU); rates with fewer runs will differ.

## What follows (needs a decision)

- Stage B budget 10 h, 3 seeds, on the cup task (pan env not built yet): sbx_sac_utd05, sbx_sac_utd1, sb3_sac_utd1 as the control with the October thread setting, curriculum ramp 1.2 M as in October. 9 runs at once fit the machine.
- A 20-minute rate test of 8 environments with 4 updates per vec step (same 0.5 ratio): the micro-benchmark model predicts about 90 steps/s, 2 M steps in 6 h; if it holds, it replaces sbx_sac_utd05 in Stage B.
- Lever for the environment step itself (17 ms on the PC vs 6.5 ms here): profile before Stage B.

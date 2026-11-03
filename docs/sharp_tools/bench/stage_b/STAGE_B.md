# Stage B, attempt 1 (2–3 Nov 2026): no result, thread oversubscription

Nine runs launched 2 Nov about 21:00 on the PC with `run_stage_b.sh` at commit 5c897d9: `sbx_sac_half` (4 envs × 2 updates), `sbx_sac_utd1` (4 × 4) and `sb3_sac_ctrl` (SB3 SAC, 4 × 4, torch threads unmanaged = the October setting), seeds 0/1/2, 10 h each, cup task, curriculum ramp 1.2 M, 100-episode validation every 40 k steps. Curves in `<run>.json`; table `stage_b_curves.json`.

| run | hours | steps | steps/s | Stage A rate for the same config | best success |
|---|---:|---:|---:|---:|---:|
| sb3_sac_ctrl s0/s1/s2 | 8.5 / 5.1 / 8.7 | 200 k / 40 k / 200 k | 6.5 / 2.2 / 6.4 | 25.5 (pinned, 1 thread) | 0 / 0 / 1 % |
| sbx_sac_half s0/s1/s2 | 9.9 / 10.1 / 9.7 | 720 k / 680 k / 600 k | 20.2 / 18.8 / 17.1 | 54.4 | 5 / 0 / 1 % |
| sbx_sac_utd1 s0/s1/s2 | 10.1 / 9.7 / 10.0 | 640 k / 560 k / 520 k | 17.7 / 16.0 / 14.4 | 33.6 | 2 / 3 / 1 % |

("hours" is the wall clock of the last validation, not the run length; every run used its full 10 h budget.)

## What happened

Every run was 2.7–4× slower than the same configuration in Stage A, from the first validation interval to the last, and the machine sat at 100 % CPU from the 10-minute check onward (Stage A: 45 % with seven pinned runs). No run got past 720 k steps, where this task is still at 0–20 % success, so the ten hours produced no learning information. [Certain]

Cause [Likely]: the three control runs with unmanaged torch threads. Each spawns a thread pool the size of the machine (48) and those threads spin-wait between operations, so three of them occupy every core and starve the six single-threaded JAX learners and the 36 environment subprocesses. In October six such runs coexisted at 34–38 steps/s because there was nothing else to starve; mixed with pinned runs the setting is destructive. The one run that got 2.2 steps/s (`sb3_sac_ctrl_s1`) was the worst-placed victim of its own siblings.

My mistake: I kept the "October setting" for the control to make it faithful and did not test the mix for even ten minutes before launch. The 10-minute check reported 100 % CPU and I read it as acceptable.

## Rate test (same night, `sbx8_g4.log`, `sbx4_g2.log`)

SBX SAC, 0.5 updates per env step, two runs alone on the machine: 8 envs × 4 updates = 77 steps/s; 4 envs × 2 updates = 70 steps/s. The 8-env gain is 1.1×, not the 1.65× the micro-benchmark model predicted (the model assumed the 8-env vec step costs the same as 4; it does not when two runs share the box). 4 envs stay the Stage B configuration; 8 envs are a small bonus for the campaign, not a lever.

Note also 70 steps/s alone versus 54 in Stage A with seven runs: rates depend on what else is running, so every "at equal wall clock" claim must state the load.

## Fix (in `run_stage_b.sh` from this commit)

The control runs with torch pinned to 2 threads (`--torch-threads 2`, 21.8 ms per update in the micro-benchmark, versus 27.9 at 1 thread). This is no longer byte-for-byte the October setting, but the October runs were on the buggy arm anyway, so the control has to be rerun on the corrected arm regardless; a pinned control is the fairer one. Expected load: 9 runs × (4 env processes + 1–2 learner threads) on 48 threads, about the Stage A load plus a third.

## Attempt 2

Same nine runs, same 10 h, with the fix. Nothing else changes. Needs a launch decision (10 h of PC).

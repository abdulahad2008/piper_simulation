# Scripted reaction baselines on the training env (9 Nov 2026)

What was run: the three hand-written reactions (keep / ISO stop / retract) × three person types × the three evaluation sets of the pan task, on the pan training env itself (`PiperPanPlaceEnv`, `reaction_baselines.py --env task`), twice: once as written and once with a "task done, let go" release rule for the reactions (`--release-when-done`). No learning. One machine (Linux, 2 cores, MuJoCo 3.3.7, `MUJOCO_GL=osmesa`), one run per cell, about 1.1 s per episode.

Code: `piper_rl/scripts/reaction_baselines.py` and `reaction_table.py` at the commit of this file. Raw rows: `reactions_task/reactions_{exact,random,ood}.csv` and `reactions_task_release/...`, per-cell summaries `*_summary.json`, full tables with Wilson 95 % intervals `TABLES.md` in each directory. Every number below is copied from those tables.

This replaces the 30 Oct study (`REACTION_BASELINES.md`) as the reference rows for the pan campaign. The 30 Oct `stop` rows were produced with two bugs (correction note at the top of that file).

## What changed since 30 Oct

1. **Environment.** The reactions run on the pan training env, not the preview scene: same robot, controller, 20 Hz, 64-number observation, safety layer, person windows (cup windows shifted by −2.4 s, `PAN_TASK.md`), and the env's own success criterion. The scripted carry starts at t = 0 with no random hold (a learner's episode has none either). Consequence: the **exact-crossing set is one deterministic episode repeated 200 times** (fixed person timing, no start delay). Its rows are a single case study; the intervals printed for it mean nothing and are omitted below.
2. **Two bugs in the stop reaction, fixed.** (a) The hold target was re-measured every control step, so the PD controller's gravity sag ratcheted into a drift of about 3 cm over 2 s; a "stopped" pan crept onto the halted hand. Now the joint position is measured once at the trigger. (b) A halted or pulling-back person kept the trajectory's hand velocity (about 0.22 m/s) in the state the robot saw, which inflated the protective distance. Now the velocity is scaled by the person's clock rate (0 when halted, −1 when pulling back). Regression tests: `tests/test_reaction_baselines.py`.
3. **Stop inputs re-measured on the training env** (no person, hold commanded mid-carry, 3 seeds × 6 instants): pan speed during the carry 0.33–0.51 m/s → v_r = 0.50 m/s (was 0.30); after a hold the pan overshoots by 18–23 mm, peaking 0.10 s after the command and settling to 4 mm → S_s = 23 mm, the peak, since that is what reaches the hand (was 6 mm, which I had measured at 0.5 s only — my error, corrected the same day). T_r = 0.05 s, T_s = 0.30 s unchanged. S = v_h·0.35 s + 0.025 m + 0.023 m.
4. **Two contact counts.** The pan env ends a learner's episode at success. So every contact metric is reported twice: **until env done** (what a learned policy is scored on) and over the **full 10 s** (what the placed pan and the parked arm do to the person afterwards, with the scripted carry holding its last pose). Success is the env's own criterion (`env_success`), not the 30 Oct "at rest for the last 0.5 s" rule, because the scripted carry dithers at 2–4 cm/s on a pan that is 4 mm off and fails the old rule while the env counts it done.
5. **Release rule** (`--release-when-done`): once the env has counted the pan placed, the reaction no longer fires. Run as a second variant, not as a replacement, because it changes the retract rows a lot (30 Oct finding 3).

Person types (unchanged): hand clock rate +1 / 0 / −1 with a 0.25 s delay; `halts` freezes and `pulls_back` retreats when the hand is within 25 cm of the pan's hot disc, both resume when it is beyond 30 cm. Exposure = any person segment within 3 cm of a hot surface. Robot-body contact does not end the episode in this harness (the training env ends it; see decision 4 below).

## Headline tables (release rule off)

Hot contact and robot-body contact as "until env done / full episode" (% of episodes). Success is the env's criterion.

**Random set** (`evaluation_id`, 500 episodes per cell, person window −1.8 to −0.2 s; the hand comes within 25 cm of the pan before placement in 75–76 % of episodes):

| reaction | person | success % | hot contact % | robot body % | handle % (full) | exposure s (until done / full) | done at s |
|---|---|---|---|---|---|---|---|
| keep | keeps going | 100 | 41 / 73 | 0 / 34 | 51 | 0.36 / 2.38 | 2.05 |
| keep | halts | 100 | 19 / 19 | 0 / 0 | 3 | 0.24 / 1.77 | 2.05 |
| keep | pulls back | 100 | 12 / 12 | 0 / 0 | 0 | 0.09 / 0.09 | 2.05 |
| stop | keeps going | 100 | 44 / 76 | 14 / 34 | 50 | 1.55 / 2.72 | 3.74 |
| stop | halts | 84 | 16 / 16 | 0 / 0 | 0 | 1.35 / 1.76 | 2.09 |
| stop | pulls back | 100 | 2 / 2 | 0 / 0 | 0 | 0.03 / 0.03 | 2.18 |
| retract | keeps going | 85 | 8 / 9 | 26 / 34 | 42 | 0.17 / 0.17 | 5.44 |
| retract | halts | 77 | 8 / 9 | 26 / 34 | 42 | 0.17 / 0.18 | 5.60 |
| retract | pulls back | 64 | 8 / 9 | 25 / 33 | 41 | 0.18 / 0.18 | 5.53 |

Wilson 95 % intervals at n = 500: ±3–4 points for rates near 10–45 %, ±1 point near 0 or 100 (TABLES.md).

**OOD set** (`evaluation_ood`, 500 per cell, window −2.4 to +1.6 s; intrusion before placement 54 %):

| reaction | person | success % | hot contact % | robot body % | handle % (full) | exposure s (until done / full) | done at s |
|---|---|---|---|---|---|---|---|
| keep | keeps going | 100 | 30 / 54 | 20 / 47 | 58 | 0.42 / 1.38 | 2.05 |
| keep | halts | 100 | 23 / 23 | 6 / 6 | 20 | 0.22 / 0.24 | 2.05 |
| keep | pulls back | 100 | 23 / 23 | 6 / 6 | 20 | 0.15 / 0.15 | 2.05 |
| stop | keeps going | 100 | 37 / 58 | 14 / 40 | 62 | 1.30 / 1.99 | 3.45 |
| stop | halts | 77 | 22 / 22 | 5 / 5 | 15 | 2.31 / 2.31 | 2.06 |
| stop | pulls back | 100 | 22 / 22 | 5 / 5 | 19 | 0.19 / 0.19 | 2.22 |
| retract | keeps going | 97 | 20 / 20 | 28 / 44 | 45 | 0.08 / 0.08 | 3.98 |
| retract | halts | 89 | 21 / 21 | 25 / 41 | 45 | 0.08 / 0.08 | 4.45 |
| retract | pulls back | 78 | 21 / 21 | 24 / 39 | 45 | 0.08 / 0.08 | 4.78 |

**Exact crossing** (`fixed_exact_h036_v1`, one deterministic episode; the hand sweeps along the counter at about 13 cm above the trivet top, through the robot's path):

| reaction | person | success | hot contact (until done / full) | robot body (until done / full) | exposure s (until done / full) | p5 hot distance mm | done at s |
|---|---|---|---|---|---|---|---|
| keep | keeps going | yes | yes / yes | no / yes | 0.95 / 3.55 | −43 | 2.05 |
| keep | halts | yes | no / no | no / no | 0.95 / 8.90 | +5 | 2.05 |
| keep | pulls back | yes | no / no | no / no | 0.10 / 0.10 | +26 | 2.05 |
| stop | keeps going | yes | yes / yes | yes / yes | 3.90 / 3.90 | −49 | 6.50 |
| stop | halts | yes | yes (0.10 s) / yes | no / no | 0.90 / 8.65 | −4 | 2.25 |
| stop | pulls back | yes | no / no | no / no | 0.00 / 0.00 | +34 | 2.40 |
| retract | keeps going | yes | no / no | yes / yes | 0.00 / 0.00 | +52 | 6.60 |
| retract | halts | yes | no / no | yes / yes | 0.00 / 0.00 | +58 | 7.75 |
| retract | pulls back | yes | no / no | yes / yes | 0.00 / 0.00 | +61 | 8.55 |

Negative p5 distance = penetration of the hand into the hot surface (contacts are force-free, so the hand passes through).

## Release rule on

Only the retract rows change by design (keep has no reaction; the stop's hold ends when the hand leaves S + 5 cm whether or not the pan is placed, and in practice its rows are unchanged to within one episode). Retract, random set, release rule off → on (env success %, hot contact % until done / full, robot body % until done / full):

| person | success | hot contact | robot body |
|---|---|---|---|
| keeps going | 85 → 93 | 8 / 9 → 8 / 20 | 26 / 34 → 26 / 34 |
| halts | 77 → 86 | 8 / 9 → 8 / 8 | 26 / 34 → 26 / 26 |
| pulls back | 64 → 76 | 8 / 9 → 8 / 8 | 25 / 33 → 25 / 26 |

OOD, retract, off → on: success 97 / 89 / 78 → 100 / 100 / 96 (keeps going / halts / pulls back); hot contact until done unchanged at 20–21 %; full-episode hot contact against the person who keeps going 20 → 36 %; robot body full episode 44 / 41 / 39 → 47 / 25 / 24 %. Exact crossing: identical tables with the rule on and off (the intrusion precedes placement, so the rule never fires). Full tables: `reactions_task_release/TABLES.md`.

## Findings

1. **No scripted reaction is dominant, and against a person who keeps coming none of them helps.** [Certain for this simulation] Random set, until env done: keep 41 % hot contact, stop 44 %, retract 8 % but 26 % robot-body contact and 42 % handle contact; full episode: keep 73 %, stop 76 %, retract 9 % hot / 34 % body. The stop costs 1.7 s and buys nothing against this person: the hand is committed to the pan's path before S is reached, and a held pan is still a hot pan (30 Oct finding 1 stands). Retract swaps the hot surface for the lifted forearm and the pan handle, which the sweeping arm then hits.
2. **The ISO stop works against a person who moves away, barely against one who freezes.** [Certain] Random set: against `pulls_back` the stop cuts hot contact from 12 % to 2 %; against `halts` from 19 % to 16 %, and 16 % of those episodes (79 of 500) never complete because the hand freezes inside the resume band (30 Oct finding 2 stands). On OOD the stop changes nothing against either reacting person (23 → 22 %), see finding 4.
3. **Stopping or carrying on next to a frozen hand means long exposure without contact.** [Certain] Exact crossing, `halts`: keep places the pan 5 mm from the hand and leaves it there for the remaining 8.9 s; the fixed stop holds, resumes, brushes the hand for 0.10 s (−4 mm) and parks 4 mm away for 8.65 s. OOD, stop/halts: 2.31 s of exposure per episode while the robot holds next to the frozen hand. The pass mark counts contact; exposure time is the metric that sees this, and it is the one the Lagrangian budget constrains.
4. **About a fifth of the OOD set, and some of the random set, start with the hand already at the pan on the stove.** [Certain] All hot contacts against the halting and pulling-back persons on OOD (115 of 500 episodes) have intrusion at t = 0.00 s; replaying 40 of them with a robot that never moves gives hot contact in 38. The OOD window (−2.4 to +1.6 s after the −2.4 s shift) puts the person mid-motion at the stove when the episode begins. No policy can avoid these; they set a floor of roughly 23 % hot contact on OOD for every reaction, learned or scripted, and the pass mark cannot be applied to that set as it stands. On the random set (the validation distribution) the same replay gives 3 of the 97 hot-contact episodes against the halting person, i.e. under 1 % of episodes: there the pass mark is meaningful. Against the person who keeps going, a pan left motionless on the stove is touched in 248 of 500 random episodes within 10 s (the arm's sweep passes over the stove), so the full-episode hot-contact count has a floor of about 50 % for any policy that does not move the pan, and the until-done count has none.
5. **A learner scored at the env's own episode end does not see about half of the hot contacts and all of the body contacts.** [Certain] Random set, keep / keeps going: 41 % hot until done vs 73 % over 10 s; robot body 0 % vs 34 %. Stop / keeps going: 44 vs 76 %, 14 vs 34 %. The pan is placed in 2.05 s and the hand arrives afterwards. A policy that races and parks is rewarded for it by the current task. Whether that is the task we mean is decision 5 below.
6. **The release rule is a trade, not a fix.** [Certain] Random set, retract: success rises by 8–12 points (85 → 93, 77 → 86, 64 → 76) and the post-placement body contact against reacting persons disappears (34 → 26 %, i.e. only the pre-placement contacts remain); but against the person who keeps going the full-episode hot-contact rate rises from 9 % to 20 %, because the placed pan is now left where the hand sweeps. Until the env's own end nothing changes (8 % hot, 26 % body), so a learner scored there sees no difference between the two rules. On the exact crossing the rule never fires (the intrusion precedes placement) and the tables are identical.
7. **Reaction times.** The stop fires 0.19–0.21 s after the 25 cm mark against reacting persons on the random set and 0.82 s against the one who keeps going (the hand is faster, S is larger, and the trigger comes later relative to the intrusion mark); on OOD 0.04 s, because the hand is already inside S when the episode starts. Retract's 0.00 s is still the definition of its trigger, not a measurement (30 Oct finding 4).

## What this does and does not show

- It shows what the three simplest reactions do on the exact env, sets and scoring that the learned policies will be judged on. These are the reference rows for `PREREGISTRATION_PAN.md`.
- One scripted carry, one kinematic person model (force-free contacts: the hand passes through the pan), one machine, intervals over episodes only. The exact crossing is one case.
- The person types are a property of the harness, not of the training env: the kinematic arm the learner trains against never reacts. It is the `keeps_going` row that a learned policy should be compared with first; the reacting rows say what the same policy would face from a person who does react.
- Not run: fast-hand set, knife, preview-env stop rows with the fixes, a learned policy against the three person types.

## Decisions this raises (for the author)

1. Pass mark (proposed: success ≥ 90 %, robot-body contact ≤ 2 %, hot contact ≤ 5 % on validation). With finding 4 the hot-contact mark cannot hold on OOD; it can on validation/random only if the unavoidable fraction there is small (see the sentence in finding 4).
2. Retract with or without the release rule as the reference row (both are reported here).
3. Robot-body contact with the kinematic sweeping arm: the training env ends the episode and the proposed pass mark allows 2 %, yet the scripted carry itself hits 34 % (random, full episode) and the retract 26 % before placement. A policy that places in 2 s is only "safe" by this count because the episode ended before the arm arrived.
4. Terminate at success (as built) versus running every episode to 10 s and charging exposure after placement. (b) is closer to the question in the title but changes the task and every number; not done.
5. The OOD set: keep as a stress test with its floor stated, or move its window so that the hand is not already on the stove at t = 0.

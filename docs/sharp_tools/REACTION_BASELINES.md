# Scripted reaction baselines, hot pan (first results, 30 Oct 2026)

What was run: three hand-written reactions × three person types × three test sets, no learning. Commit `ab9ab89` of `piper_rl/scripts/reaction_baselines.py`; raw rows in `reactions/reactions_{exact,random,ood}.csv`, summaries in `reactions/*_summary.json`, tables with 95 % Wilson intervals in `reactions/TABLES.md`. One machine (Linux, 2 cores, MuJoCo 3.3.7, `MUJOCO_GL=osmesa`), one run per cell; the numbers are machine-specific at the few-percent level (see PHASE1_AUDIT §1.5).

## Set-up

- Task: scripted carry of the pan from the stove to the trivet (same controller as preview d), random start delay U[0, 3] s, episode 15 s at 20 Hz.
- Person: corrected interpolated arm, `fixed_exact_h036_v1` (exact crossing, 100 episodes per cell), `evaluation_id` (random, 200), `evaluation_ood` (out-of-distribution, 200).
- Person types (hand "clock" rate +1 / 0 / −1, 0.25 s delay): `keeps_going`, `halts` (freezes when the pan is within 25 cm), `pulls_back` (retreats when within 25 cm, resumes when beyond 30 cm).
- Reactions: `keep` (ignore the person); `stop` (ISO/TS 15066 protective distance S = v_h(T_r+T_s) + v_r T_r + S_s with the measured T_r 0.05 s, T_s 0.30 s, v_r 0.30 m/s, S_s 6 mm; hold the measured joint position; resume at S + 5 cm); `retract` (lift first, half-speed horizontal away from the hand; triggered at 25 cm, released at 30 cm).
- Scoring: success = pan centre within 4.5 cm of the trivet and at rest for the last 0.5 s of the episode; "done any time" = it was placed at some point; contact classes hot surface / handle / robot body from MuJoCo contacts; exposure = time the hand is within 3 cm of the hot surface in the second after the first intrusion.

## Headline table (success % / hot-surface contact %)

| person → | keeps going | halts | pulls back |
|---|---|---|---|
| **exact** keep | 100 / 100 | 100 / 25 | 100 / 18 |
| **exact** stop | 100 / 100 | 77 / 25 | 100 / 18 |
| **exact** retract | 83 / 0 | 0 / 0 | 0 / 0 |
| **random** keep | 100 / 67 | 100 / 12 | 100 / 9 |
| **random** stop | 99 / 67 | 89 / 13 | 100 / 8 |
| **random** retract | 36 / 0 | 21 / 0 | 17 / 0 |
| **OOD** keep | 100 / 44 | 100 / 0 | 100 / 0 |
| **OOD** stop | 99 / 47 | 98 / 0 | 100 / 0 |
| **OOD** retract | 39 / 0 | 26 / 0 | 16 / 0 |

Intervals: ±4–9 points at n = 100, ±3–7 at n = 200 (TABLES.md). Intrusion before completion happened in 70 % (exact), 46 % (random) and 34 % (OOD) of episodes; the remaining episodes are the same for all reactions.

## Findings

1. **The ISO stop does not reduce hot-surface contact against a person who keeps coming.** [Certain for this simulation] Stop and keep have the same hot-contact rate in every set (exact 100 vs 100, random 67 vs 67, OOD 47 vs 44 %). The stop only changes when the contact happens and adds 0.6–1.1 s of reaction. The protective distance (S = 0.021 m + 0.35 s × hand speed, about 0.2 m for a 0.5 m/s hand) is reached after the hand is already committed to the pan's path, and a held pan is still a hot pan.
2. **Stop and a halting person deadlock.** [Certain] exact: 77 % success, 23 % never completed (timeouts); random 89 %; OOD 98 %. The robot holds at S, the person freezes inside S + 5 cm, neither moves until the clock runs out. The rate scales with how often the hand's rest position is inside the resume band.
3. **Retract removes hot-surface contact entirely but trades it for blunt contact and lost completions.** [Certain] 0 % hot contact in all nine retract cells (p5 of the hot-surface distance +44 to +61 mm, versus −6 to −42 mm for keep/stop in every cell where contact occurred, i.e. penetration). Handle or robot-body contact in 100 % of exact-set episodes with an intrusion (the scripted upper arm follows the hand over the table and meets the lifted robot) and 20–39 % in the random and OOD sets. Success is low (0–39 %) for a scoring reason that matters: most of the shortfall is pans that were placed and then **lifted off the trivet again** by a later intrusion (random set: 162 of 200 halts-episodes completed, 121 of those ended off the trivet). The stop reaction is not penalised the same way because holding after placement leaves the pan where it is. The retract baseline as written has no "task done, let go" rule. Adding one is a one-line change but it changes the table; your call before I do it.
4. **Reaction times.** Stop reacts 0.31–0.35 s after the 25 cm intrusion mark against halting and pulling-back persons (the hand is slower, S is smaller; 0.92 s for OOD/halts) and 0.63–1.09 s against the keeps-going person. Retract shows 0.00 s because its trigger *is* the 25 cm mark that defines intrusion time; the column is not comparable across reactions and will be re-based on a common event.
5. **Person type matters more than reaction on the exact crossing.** The pulling-back person gets 18 % hot contact with *any* non-retracting reaction, the halting person 25 %, the keeps-going person 100 %. The 18–25 % is the 0.25 s delay: the hand reaches the pan before the person has reacted.
6. **OOD is the easiest set here, not the hardest.** Intrusion before completion drops to 34 % and the halting/pulling-back persons never touch the hot surface. The OOD preset moves the person's start and reach outside the training range, which on this geometry mostly moves them away from the trivet path. The "fast hand" case that defeats the ISO stop (0.72 m reach at 2 m/s) is not in any of these sets yet.

## What this does and does not show

- It shows what the simplest version of each reaction does, with the measured stop parameters, on the corrected arm, and why a learned reaction with a hazard budget is worth training: no scripted rule avoids both hot contact and blunt contact, and the sensible ones deadlock.
- It is one scripted policy against one scripted person model, one machine, no learning, no seeds to average over beyond the episode seeds. The intervals are over episodes only.
- Not yet run: fast-hand set, knife, learned policy, retract with a release rule, reaction time on a common event.

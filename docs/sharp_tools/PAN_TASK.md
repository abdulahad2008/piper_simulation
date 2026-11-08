# Hot-pan carry task: definition and decisions (4 Nov 2026)

Code: `piper_rl/pan_env.py` (`PanEnvConfig`, `PiperPanPlaceEnv`), tests `tests/test_pan_env.py`, training `python -m piper_rl.scripts.train --task pan --human-preset curriculum ...`.

## Task
Carry the frying pan (welded by its handle in the gripper, `make_pan_scene.py`) from the stove to the trivet while the kinematic person shares the counter. Robot, table, control rate (20 Hz), action interface (5-D Cartesian) and the 64-number observation layout are those of the cup task; the pan fills the cup's observation slots. Safety-layer table clearance 6.5 cm (pan base 5.4 cm below the TCP).

Success: pan centre within 4.5 cm of the trivet centre, base between 5 mm below and 20 mm above the trivet top, speed below 0.05 m/s, for 3 consecutive control steps. No release is modelled. The scripted carry succeeds in 41 steps (2.05 s) with 3 mm error.

Start phases (reverse curriculum, same probabilities as the cup task's phases): on the stove / lifted at the stove / mid-carry / over the trivet. Evaluation always starts on the stove.

## Reward
One potential plus one-shot bonuses and the cup task's per-step penalties:

    phi(s) = -w_transport * d_xy + w_lift * [ min(h, H) s(d_xy) + (H - min(h, H)) (1 - s(d_xy)) ]

d_xy = pan centre to trivet (horizontal), h = pan base above the table, H = 6 cm, s = smoothstep from 0 at d_xy = 4.5 cm to 1 at 9 cm. Shaping reward = phi(s_t) - phi(s_{t-1}); over any trajectory it sums to phi(end) - phi(start) (tested), so cycling up/down or near/far earns nothing. Retreat penalty on increasing d_xy, success bonus 200 once, time/action/action-rate/joint-velocity penalties, collision penalty for arm links against counter or fixtures and for the pan dragging on the counter away from the trivet. Person terms are inherited from the cup task (proximity cost, robot-body collision penalty and termination).

## Person-pan contact classes
Per control step: hot surface (base and walls), handle, robot body; `exposure_steps` = steps with any person segment within 3 cm of a hot surface; `hazard_cost` = 1 on such a step, else 0 (the Lagrangian cost signal). Contacts are detected but exert no force (`gap="1"`, Phase 1 audit §1.7): the person is a sensor, not an obstacle.

## Decision: the person starts 2.4 s earlier than in the cup task
Measured 4 Nov (`pan_task/direct_carry_meeting.json`): with the cup task's appearance window (0.6–2.2 s after the start) a direct carry finishes before the hand reaches the path: on the random set only 2 % of episodes bring a hot surface within 25 cm of the person and 0 % touch; on the exact crossing the pan passes 25 cm ahead of the hand. A policy trained on that would learn to race, and the stop-versus-retract question would never arise.

Options put to the author: (1) shift the appearance window earlier, (2) a random "pan not ready" hold at the start, (3) keep and report racing. Decision (4 Nov, author): option 1.

Implementation: `PAN_APPEARANCE_SHIFT_S = -2.4` applied to every pan preset after the preset is built and again after every curriculum difficulty update (the callback recomputes the window). A negative appearance time means the person is already mid-motion at the start of the episode. Resulting windows: random/curriculum-full (−1.8, −0.2) s; exact crossing −1.4 s; shifted exact −1.6 s; OOD (−2.4, 1.6) s; curriculum at difficulty 0 (−0.6, 0.6) s.

Direct scripted carry under the shifted windows (no reaction, no hold):

| set | episodes | hot surface within 25 cm of the person | hot contact | robot-body contact | success |
|---|---:|---:|---:|---:|---:|
| exact crossing | 10 | 100 % | 100 % | 0 % | 100 % |
| shifted exact | 10 | 100 % | 100 % | 0 % | 100 % |
| random (evaluation_id) | 100 | 73 % | 22 % | 0 % | 100 % |
| OOD | 100 | 48 % | 14 % | 8 % | 100 % |

The exact crossing is now a guaranteed meeting (the hard case the reactions are judged on); the random set meets the person in about three quarters of episodes and touches in a fifth. The paper must state that the pan person-distribution is the cup distribution shifted by −2.4 s, and the scripted reaction baselines of 30 Oct (which used the cup windows plus a 0–3 s hold) are not directly comparable to pan-task policy numbers; they will be rerun on the pan env's own sets before the campaign.

## Not yet done
- Pass mark for the pan validation set (proposed: success ≥ 90 %, robot-body contact ≤ 2 %, hot contact 0 %; random policy and scripted carry evaluated first).
- Reaction baselines rerun on this env (keep / stop / retract × person types) as the comparison rows.
- Lagrangian callback on `hazard_cost`; 200 k pilot.
- Float64 numpy actor export.

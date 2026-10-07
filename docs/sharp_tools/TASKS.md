# Tasks T1–T5 (to be pre-registered before any multi-seed run)

All tasks use the `PiperHumanAwarePickPlace-v0` scene: 6-DoF PiPER, 20 Hz
control over 500 Hz physics, 5-D Cartesian action, the simulated forearm +
hand (capsules), minimum-jerk human trajectories drawn from the same
five-type distribution (crossing, reach-to-object, reach-to-target, crossing with pause, enter-and-reverse). The knife is `hazard/assets/knife.xml`
welded to the tool frame once grasped (`<weld>` equality or a child body of
the gripper); the policy observes the blade frame (heel, tip, cutting
direction in base frame, 9 numbers) in addition to the existing 51 + 13
channels.

Episode length 200 steps (10 s). Collision with a robot link or a cut (edge
penetrates a human capsule) terminates the episode.

| id | task | success condition | human motion | why it is in the set |
|---|---|---|---|---|
| T1 Carry | pick the knife from a rack pose, carry it across the board, place it on a target pad | knife at rest on the pad within 45 mm, edge down | workspace crossings | plain pick-and-place with a tool; shows body-safe but blade-unsafe behaviour |
| T2 Put-down | knife already in hand; a hand approaches the board; place the knife safely and withdraw | knife on the board, edge away from the human side, gripper retracted ≥ 0.2 m | reaching toward the board, variable timing | tests whether the policy chooses an orientation, not just a position |
| T3 Interrupted cut | repeated chop strokes on a block (kinematic slice event when edge passes through the block's z-plane) while a hand enters the board region | ≥ N slices and no cut/collision | entering and reversing, pauses | the HRIBench "intruder" case with a blade; stop vs retract behaviour |
| T4 Handover | present the knife to the human hand handle-first, release when the hand closes on the handle | handle within the hand capsule, edge alignment to the hand ≤ 0 at release | hand reaching to a handover point | HRI literature on handle visibility; first learned version with a hazard metric |
| T5 Pass-by | carry the knife from A to B while the human arm rests or moves across the path | at B, no cut, no collision | all five types | measures hazard exposure on a route choice problem |

## Held-out suites

| suite | episodes | seed | note |
|---|---|---|---|
| No human | 200 | 30000 | hazard must be zero by construction; checks the wrapper |
| Exact | 200 | 31000 | training crossing |
| Shifted | 200 | 32000 | 0.2 s earlier, 0.02 m/s faster, 1 cm closer |
| Rand-ID | 500 | 40000 | training distribution, disjoint seeds |
| OOD | 500 | 50000 | from −y, faster, closer, 150 ms latency |
| Counterfactual pair | 200 each | 31000 | person shown to the policy but absent from the scene; person present but hidden from the policy |

## Primary outcomes (pre-register)

1. Task success S and body-contact rate C.
2. Cut rate K (edge penetration), hazard exposure E, p05 of min edge distance
   given success, fraction of successful episodes with max hazard > 0.5.
3. Pre-registered hypothesis H1: policies trained with the body-contact reward
   reach high S and low C on T1 and T5 while K + E are not smaller than for a
   random-orientation control. H2: adding the hazard cost
   (shaped, weight 2) or a Lagrangian constraint (E ≤ 0.1 s) reduces E by at
   least 50 % at a success cost below 10 points on Rand-ID.

## Training regimes (3 seeds each, SAC 2 M steps)

- R0 baseline: body-contact reward (task + proximity + link-collision penalty).
- R1 shaped: R0 − 2 · h_t.
- R2 constrained: SAC-Lagrangian on E with budget 0.1 s per episode (OmniSafe).
- R3 body shield: R0 policy under a SaRA-style stop shield on robot links
  only, to show a provably body-safe controller can still have E > 0.

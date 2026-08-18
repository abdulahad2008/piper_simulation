# Human-aware SAC v1 results

This is the measured baseline for the 2,000,000-step SAC curriculum run. The
run used four environments, seed 0, privileged human state, domain
randomization, and observation/action noise. Training ran from 2026-08-04
18:31 to 2026-08-05 08:56 local time (about 14 hours 25 minutes).

The committed policy is `models/human_aware_sac_v1.zip`, copied from the best
checkpoint after the final evaluation. SHA-256:

```text
4124B81633AC7E4BEEBBC145BE65D83C8411C50835957DC31E72CE4BD10187D7
```

## Deterministic evaluation

Each row is a fresh 30-episode evaluation with a disjoint explicit seed range.
Intervals are Wilson 95% binomial confidence intervals. CSV episode records and
JSON configuration/summary files are stored beside this report.

| Distribution | Task success | Collision-free success | Human collision | Near miss | Mean min separation | Worst separation |
|---|---:|---:|---:|---:|---:|---:|
| Fixed ID | 90.0% (74.4-96.5%) | 90.0% | 10.0% (3.5-25.6%) | 13.3% | 169 mm | -0.6 mm |
| Randomized ID | 80.0% (62.7-90.5%) | 80.0% | 10.0% (3.5-25.6%) | 16.7% | 192 mm | -24.9 mm |
| OOD | 80.0% (62.7-90.5%) | 80.0% | 6.7% (1.8-21.3%) | 13.3% | 228 mm | -35.1 mm |

The result demonstrates task learning but does **not** establish a safe policy.
Robot-human collisions remain in every evaluated distribution, and 30 episodes
leave wide confidence intervals. This model must not be deployed on hardware.

The built-in training evaluator reached 100% success over its final 15 episodes,
but that evaluator retained the initial easy curriculum distribution. It is not
used as the headline result above.

## Reproduction

```powershell
python -m piper_rl.scripts.evaluate_human_aware --model models/human_aware_sac_v1.zip --experiment fixed --episodes 30 --seed 1000 --out results/human_aware_sac_v1/fixed
python -m piper_rl.scripts.evaluate_human_aware --model models/human_aware_sac_v1.zip --experiment randomized-id --episodes 30 --seed 2000 --out results/human_aware_sac_v1/randomized_id
python -m piper_rl.scripts.evaluate_human_aware --model models/human_aware_sac_v1.zip --experiment ood --episodes 30 --seed 3000 --out results/human_aware_sac_v1/ood
```


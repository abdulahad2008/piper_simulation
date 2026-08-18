# Full-random human-motion ablation (seed 0)

## Outcome

- Status: complete, 2,000,000 environment steps.
- Training commit: `73b5612a2131f6bb605432816d63b92c38c74376`.
- Wall clock: 16.00 h (57600 s from run artifact timestamps); final logged FPS: 34.
- Best callback checkpoint: 1,540,000 steps, mean reward 275.84, success 100.0% over 15 episodes.
- Final callback: reward 201.71, success 80.0%, mean length 62.3.
- Best model SHA-256: `4064b2a8ac9bc7571bb8dbbbb6f76fe39c26c9ff620fbc64ece9fa6efd0e9a5b`.

## Experimental controls

- SAC seed 0, 4 environments, Cartesian action, 64-value state observation, `[256, 256]` network, batch 256, replay capacity 400,000, learning rate 3e-4, gamma 0.98, tau 0.01, 4 gradient steps, and automatic entropy (`auto_0.1`).
- Manipulation reverse curriculum remained enabled with phase probabilities `0.50/0.22/0.16/0.12`.
- Human-motion difficulty was exactly 1.0 for all 3,587 logged rollout summaries (min=max=1.0).
- Preflight sampled 200 resets from `training_id`; every OOD-only range check passed. See `distribution_preflight.json`.
- The normalized configuration diff has 11 intended differences and zero invalid differences. See `config_diff.json`.

## Validation checkpoint comparison

Both best policies used deterministic actions on seeds 20,000-20,049 in the full-difficulty randomized ID environment.

| Policy | Task success | Collision-free success | Human collision | Mean reward |
|---|---:|---:|---:|---:|
| Curriculum reference | 82.0% | 82.0% | 8.0% | 215.46 |
| Full random | 88.0% | 86.0% | 6.0% | 234.81 |

## Held-out selected-checkpoint results

| Distribution | Episodes | Task success | Collision-free | Collision | Near miss | Completion time (successes) |
|---|---:|---:|---:|---:|---:|---:|
| No human | 200 | 85.0% | 85.0% | 0.0% | 0.0% | 3.063 s |
| Fixed ID | 200 | 74.5% | 74.5% | 24.5% | 26.5% | 2.684 s |
| Randomized ID | 500 | 81.2% | 81.2% | 8.6% | 16.0% | 2.870 s |
| OOD | 500 | 87.2% | 87.2% | 4.2% | 11.6% | 2.945 s |

The fixed crossing is a clear safety failure mode: 24.5% collision rate (95% CI 19.1%-30.9%). Do not interpret the aggregate OOD result as evidence that the policy is safe.

## Learning efficiency

Offline evaluations used identical seeds 21,000-21,049 and exact checkpoints.

| Steps | Curriculum success | Full-random success |
|---:|---:|---:|
| 100,000 | 0.0% | 0.0% |
| 200,000 | 4.0% | 0.0% |
| 400,000 | 0.0% | 2.0% |
| 600,000 | 0.0% | 0.0% |
| 1,000,000 | 42.0% | 6.0% |
| 1,500,000 | 82.0% | 82.0% |
| 2,000,000 | 82.0% | 94.0% |


- First nonzero success: curriculum 200,000; full-random 400,000 steps.
- 50%, 70%, and 80% thresholds: both first reached at 1,500,000 steps on this sparse offline grid.
- Normalized success AUC (100k-2M): curriculum 0.426; full-random 0.356.
- At 2M: curriculum 82.0% collision-free success / 4.0% collision; full-random 94.0% / 6.0%.

## Limitations and next experiment

This is a single-seed comparison, callback selection uses only 15 episodes, and exact threshold timing is bounded by the sparse checkpoint grid. The full-random policy is not safe in the fixed crossing. A fixed-human ablation is therefore scientifically useful as the next controlled run, but it should not be deployed and was not started automatically.

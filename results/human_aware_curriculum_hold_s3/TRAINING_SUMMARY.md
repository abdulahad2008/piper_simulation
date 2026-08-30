# human_aware_curriculum_hold_s3

- Method/seed: `curriculum_hold` / `3`
- Steps: `2,000,000`
- Duration: `15.38 h`
- Common-ID selected checkpoint: `callback_best`
- Model SHA-256: `f98f00e9d56fa6e61f1dab87e6ed128a46f2ed9d765a78f198428997bc2c8b8c`

## Held-out summaries

| Distribution | Success | Collision | Collision-free success |
|---|---:|---:|---:|
| no_human | 0.910 | 0.000 | 0.910 |
| exact_h036 | 0.770 | 0.220 | 0.765 |
| constrained_fixed_height_randomized | 0.760 | 0.230 | 0.760 |
| shifted_exact_h036 | 0.410 | 0.540 | 0.410 |
| randomized_id | 0.804 | 0.124 | 0.804 |
| ood | 0.804 | 0.120 | 0.802 |

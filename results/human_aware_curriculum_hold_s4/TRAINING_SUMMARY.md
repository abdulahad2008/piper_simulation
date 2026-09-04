# human_aware_curriculum_hold_s4

- Method/seed: `curriculum_hold` / `4`
- Steps: `2,000,000`
- Duration: `15.06 h`
- Common-ID selected checkpoint: `step_2000000`
- Model SHA-256: `8c1d7353708555aaa75347450f2f54a3b49aa3a1b25ddd631d5c5f3bdec1051a`

## Held-out summaries

| Distribution | Success | Collision | Collision-free success |
|---|---:|---:|---:|
| no_human | 0.855 | 0.000 | 0.855 |
| exact_h036 | 0.590 | 0.375 | 0.590 |
| constrained_fixed_height_randomized | 0.610 | 0.365 | 0.610 |
| shifted_exact_h036 | 0.490 | 0.500 | 0.490 |
| randomized_id | 0.780 | 0.104 | 0.780 |
| ood | 0.824 | 0.068 | 0.822 |

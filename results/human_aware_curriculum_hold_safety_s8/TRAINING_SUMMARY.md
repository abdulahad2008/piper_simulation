# human_aware_curriculum_hold_safety_s8

- Method/seed: `curriculum_hold_safety` / `8`
- Steps: `2,000,000`
- Duration: `14.47 h`
- Common-ID selected checkpoint: `step_1000000`
- Model SHA-256: `0721ac0dada8e2239f24f386eef6c90cdbda3c091c0317d21794d279d0e8bee5`

## Held-out summaries

| Distribution | Success | Collision | Collision-free success |
|---|---:|---:|---:|
| no_human | 0.030 | 0.000 | 0.030 |
| exact_h036 | 0.000 | 0.895 | 0.000 |
| constrained_fixed_height_randomized | 0.000 | 0.920 | 0.000 |
| shifted_exact_h036 | 0.000 | 0.920 | 0.000 |
| randomized_id | 0.002 | 0.254 | 0.002 |
| ood | 0.010 | 0.244 | 0.010 |

# human_aware_curriculum_hold_s5

- Method/seed: `curriculum_hold` / `5`
- Steps: `2,000,000`
- Duration: `14.15 h`
- Common-ID selected checkpoint: `step_2000000`
- Model SHA-256: `e7d4f6615c4a3c35c6d51c222d152cc6d307f084ce6be9f88f22a79b88fbcebb`

## Held-out summaries

| Distribution | Success | Collision | Collision-free success |
|---|---:|---:|---:|
| no_human | 0.815 | 0.000 | 0.815 |
| exact_h036 | 0.840 | 0.115 | 0.840 |
| constrained_fixed_height_randomized | 0.835 | 0.125 | 0.835 |
| shifted_exact_h036 | 0.750 | 0.230 | 0.750 |
| randomized_id | 0.840 | 0.098 | 0.838 |
| ood | 0.768 | 0.132 | 0.768 |

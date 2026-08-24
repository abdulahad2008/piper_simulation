# human_aware_fixed_s1

- Method/seed: `fixed` / `1`
- Steps: `2,000,000`
- Duration: `15.83 h`
- Common-ID selected checkpoint: `step_2000000`
- Model SHA-256: `a58158ef4ddfe91db948b9bdae59e88f01189c5ce95f800a2cca3626bec31d1b`

## Held-out summaries

| Distribution | Success | Collision | Collision-free success |
|---|---:|---:|---:|
| no_human | 0.485 | 0.000 | 0.485 |
| exact_h036 | 0.960 | 0.040 | 0.960 |
| constrained_fixed_height_randomized | 0.950 | 0.040 | 0.950 |
| shifted_exact_h036 | 0.965 | 0.035 | 0.965 |
| randomized_id | 0.464 | 0.168 | 0.458 |
| ood | 0.002 | 0.276 | 0.002 |

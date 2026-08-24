# human_aware_random_full_s1

- Training: valid final policy at `2,000,000` steps.
- Required held-out suite: `final_model.zip` (not used for model selection).
- Supplemental callback-best suite: preserved in `heldout/` for the `1,760,000`-step policy.
- Required final-model SHA-256: `292e2e84e3746fd9ba2f182d82aadabbfb6ea13e43cf73f9750f397de6e98738`

## Required held-out final-policy summaries

| Distribution | Success | Collision | Collision-free success |
|---|---:|---:|---:|
| no_human | 0.905 | 0.000 | 0.905 |
| exact_h036 | 0.855 | 0.145 | 0.855 |
| constrained_fixed_height_randomized | 0.825 | 0.160 | 0.825 |
| shifted_exact_h036 | 0.670 | 0.325 | 0.670 |
| randomized_id | 0.842 | 0.070 | 0.842 |
| ood | 0.902 | 0.042 | 0.900 |

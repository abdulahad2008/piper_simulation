# human_aware_fixed_s0

Controlled seed-0 SAC experiment with `fixed_exact_h036_v1`. This is the exact 0.36 m fixed crossing, distinct from the historical constrained fixed-type preset whose hand height is randomized over [0.30, 0.42] m.

## Training

- Commit: `70827c3cee4bcf5961f205618b501c69ed7f7444`
- Steps: `2,000,000`
- Wall clock: `16.01 h`
- Final reported FPS: `34.0`
- All TensorBoard scalar values finite: `True`
- Human difficulty scalar: min/max `1.0` / `1.0`

## Exact trajectory

- Preset: `fixed_exact_h036_v1`.
- Hand height: exactly `0.36 m`.
- Canonical fingerprint SHA-256: `499cf653ba1edd4f5f83c4cfebaf4c23ca63e20fb1ce4e45eca8f1886d5355f1`.
- 200-reset preflight: one unique human trajectory fingerprint; object and target factors varied independently.

## Common-ID checkpoint selection

- Validation: 50 deterministic randomized-ID episodes, seeds 21000-21049.
- Selection metric: highest task-success rate, held-out tests excluded.
- Callback best: `0.360` success; selected common-ID model is the callback-best policy.
- The highest 7-budget grid checkpoint was 1.5M steps at 0.340 success; the callback-best was also evaluated and scored 0.360.

## Held-out comparison

| Method | Distribution | Episodes | Success | Collision-free success | Human collision | Near miss | Mean min separation (m) |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed_exact_h036 | no_human | 200 | 0.465 | 0.465 | 0.000 | 0.000 | 0.401 |
| fixed_exact_h036 | exact_h036 | 200 | 0.955 | 0.955 | 0.040 | 0.105 | 0.155 |
| fixed_exact_h036 | constrained_fixed_height_randomized | 200 | 0.945 | 0.945 | 0.045 | 0.085 | 0.154 |
| fixed_exact_h036 | shifted_exact_h036 | 200 | 0.860 | 0.860 | 0.135 | 0.280 | 0.103 |
| fixed_exact_h036 | randomized_id | 500 | 0.424 | 0.422 | 0.384 | 0.596 | 0.083 |
| fixed_exact_h036 | ood | 500 | 0.000 | 0.000 | 0.826 | 0.982 | 0.002 |
| curriculum | no_human | 200 | 0.945 | 0.945 | 0.000 | 0.000 | 0.331 |
| curriculum | exact_h036 | 200 | 0.730 | 0.730 | 0.255 | 0.260 | 0.144 |
| curriculum | constrained_fixed_height_randomized | 200 | 0.740 | 0.740 | 0.230 | 0.255 | 0.148 |
| curriculum | shifted_exact_h036 | 200 | 0.570 | 0.570 | 0.400 | 0.440 | 0.111 |
| curriculum | randomized_id | 500 | 0.760 | 0.756 | 0.148 | 0.204 | 0.206 |
| curriculum | ood | 500 | 0.822 | 0.822 | 0.092 | 0.162 | 0.225 |
| full_random | no_human | 200 | 0.850 | 0.850 | 0.000 | 0.000 | 0.347 |
| full_random | exact_h036 | 200 | 0.760 | 0.755 | 0.225 | 0.245 | 0.148 |
| full_random | constrained_fixed_height_randomized | 200 | 0.745 | 0.745 | 0.245 | 0.265 | 0.145 |
| full_random | shifted_exact_h036 | 200 | 0.505 | 0.500 | 0.490 | 0.540 | 0.078 |
| full_random | randomized_id | 500 | 0.812 | 0.812 | 0.086 | 0.160 | 0.239 |
| full_random | ood | 500 | 0.872 | 0.872 | 0.042 | 0.116 | 0.262 |

Success is the original task-success definition. Collision-free success additionally requires no robot-human collision. Completion time is averaged only over successful episodes in the JSON summaries.

## Collision audit

The historical constrained fixed-type audit used the full-random best policy for seeds 31000-31019. It recorded four genuine robot-human contacts: `link2_col`-`human_upper_arm_geom` and `link6_col`-`human_hand_geom`; minimum geometry distances were negative at the contact steps. This experiment did not modify collision detection or geometry.

## Limits

This is one seed and does not establish a statistically significant method ranking. The shifted crossing was predeclared; no held-out results were used to modify the exact trajectory or select the model.

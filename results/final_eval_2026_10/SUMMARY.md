# Final HRI evaluation - October 2026

| Run | Condition | n | Task success % | Collision % | Collision-free success % | Near-miss % | Timeout % | Model SHA-256 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| human_aware_curriculum_hold_safety_s8 | no_human | 200 | 3.0 | 0.0 | 3.0 | 0.0 | 93.5 | 0721ac0dada8 |
| human_aware_curriculum_hold_safety_s8 | exact_h036 | 200 | 0.0 | 89.5 | 0.0 | 93.0 | 0.0 | 0721ac0dada8 |
| human_aware_curriculum_hold_safety_s8 | constrained_fixed_height_randomized | 200 | 0.0 | 92.0 | 0.0 | 95.0 | 0.0 | 0721ac0dada8 |
| human_aware_curriculum_hold_safety_s8 | shifted_exact_h036 | 200 | 0.0 | 92.0 | 0.0 | 95.5 | 0.0 | 0721ac0dada8 |
| human_aware_curriculum_hold_safety_s8 | randomized_id | 500 | 0.2 | 25.4 | 0.2 | 28.8 | 46.4 | 0721ac0dada8 |
| human_aware_curriculum_hold_safety_s8 | ood | 500 | 1.0 | 24.4 | 1.0 | 29.0 | 54.6 | 0721ac0dada8 |
| human_aware_sac_v1 | common-ID candidate: step_100000 | 50 | 0.0 | 42.0 | 0.0 | 58.0 | 36.0 | 89d021c62f3a |
| human_aware_sac_v1 | common-ID candidate: step_200000 | 50 | 4.0 | 60.0 | 4.0 | 72.0 | 22.0 | de6451c48ab5 |
| human_aware_sac_v1 | common-ID candidate: step_400000 | 50 | 0.0 | 40.0 | 0.0 | 64.0 | 44.0 | ceb64deebf83 |
| human_aware_sac_v1 | common-ID candidate: step_600000 | 50 | 0.0 | 56.0 | 0.0 | 82.0 | 36.0 | e7b9ca71df1b |
| human_aware_sac_v1 | common-ID candidate: step_1000000 | 50 | 42.0 | 22.0 | 42.0 | 42.0 | 24.0 | 5937fecf27f1 |
| human_aware_sac_v1 | common-ID candidate: step_1500000 | 50 | 82.0 | 8.0 | 82.0 | 20.0 | 6.0 | 2ce59173b233 |
| human_aware_sac_v1 | common-ID candidate: step_2000000 | 50 | 78.0 | 10.0 | 78.0 | 24.0 | 10.0 | 0e0032dfd219 |
| human_aware_sac_v1 | common-ID candidate: callback_best | 50 | 82.0 | 4.0 | 82.0 | 18.0 | 6.0 | 4124b81633ac |
| human_aware_sac_v1 | common-ID selected: callback_best | 50 | 82.0 | 4.0 | 82.0 | 18.0 | 6.0 | 4124b81633ac |
| human_aware_random_full_s0 | common-ID candidate: step_100000 | 50 | 0.0 | 30.0 | 0.0 | 64.0 | 50.0 | 8bd7b1199d1f |
| human_aware_random_full_s0 | common-ID candidate: step_200000 | 50 | 0.0 | 22.0 | 0.0 | 24.0 | 56.0 | 5c15085d945a |
| human_aware_random_full_s0 | common-ID candidate: step_400000 | 50 | 2.0 | 16.0 | 2.0 | 42.0 | 48.0 | 37d190dc62b2 |
| human_aware_random_full_s0 | common-ID candidate: step_600000 | 50 | 0.0 | 26.0 | 0.0 | 58.0 | 56.0 | 9d584a04cb7f |
| human_aware_random_full_s0 | common-ID candidate: step_1000000 | 50 | 6.0 | 26.0 | 6.0 | 42.0 | 40.0 | b5283a9a7145 |
| human_aware_random_full_s0 | common-ID candidate: step_1500000 | 50 | 82.0 | 8.0 | 82.0 | 8.0 | 6.0 | ebe91bae2400 |
| human_aware_random_full_s0 | common-ID candidate: step_2000000 | 50 | 88.0 | 10.0 | 88.0 | 14.0 | 2.0 | 5ed60cb286b0 |
| human_aware_random_full_s0 | common-ID candidate: callback_best | 50 | 74.0 | 12.0 | 74.0 | 18.0 | 10.0 | 4064b2a8ac9b |
| human_aware_random_full_s0 | common-ID selected: step_2000000 | 50 | 88.0 | 10.0 | 88.0 | 14.0 | 2.0 | 5ed60cb286b0 |
| human_aware_random_full_s0 | heldout_common_id/no_human | 200 | 87.5 | 0.0 | 87.5 | 0.0 | 10.0 | 5ed60cb286b0 |
| human_aware_random_full_s0 | heldout_common_id/exact_h036 | 200 | 90.0 | 9.5 | 90.0 | 13.5 | 0.0 | 5ed60cb286b0 |
| human_aware_random_full_s0 | heldout_common_id/constrained_fixed_height_randomized | 200 | 89.5 | 10.0 | 89.5 | 13.5 | 0.0 | 5ed60cb286b0 |
| human_aware_random_full_s0 | heldout_common_id/shifted_exact_h036 | 200 | 88.0 | 12.0 | 88.0 | 18.5 | 0.0 | 5ed60cb286b0 |
| human_aware_random_full_s0 | heldout_common_id/randomized_id | 500 | 82.8 | 9.6 | 82.6 | 16.0 | 6.2 | 5ed60cb286b0 |
| human_aware_random_full_s0 | heldout_common_id/ood | 500 | 81.6 | 10.2 | 81.6 | 20.2 | 5.2 | 5ed60cb286b0 |
| human_aware_fixed_s0 | actual no-human determinism check | 20 | 45.0 | 0.0 | 45.0 | 0.0 | 55.0 | 888b0c6c22ce |
| human_aware_fixed_s0 | P1 | 200 | 94.0 | 0.0 | 94.0 | 0.0 | 5.0 | 888b0c6c22ce |
| human_aware_fixed_s0 | P2 | 200 | 17.0 | 81.0 | 17.0 | 87.0 | 0.0 | 888b0c6c22ce |
| human_aware_fixed_s1 | P1 | 200 | 97.0 | 0.0 | 97.0 | 0.0 | 3.0 | a58158ef4ddf |
| human_aware_fixed_s1 | P2 | 200 | 68.5 | 31.0 | 68.0 | 33.5 | 0.0 | a58158ef4ddf |
| human_aware_fixed_s2 | P1 | 200 | 97.0 | 0.0 | 97.0 | 0.0 | 3.0 | b97b391147fc |
| human_aware_fixed_s2 | P2 | 200 | 33.5 | 66.0 | 33.5 | 70.5 | 0.0 | b97b391147fc |
| human_aware_curriculum_s1 | P1 | 200 | 91.5 | 0.0 | 91.5 | 0.0 | 7.5 | ae3cb0ae4c31 |
| human_aware_curriculum_s1 | P2 | 200 | 83.5 | 16.0 | 83.0 | 34.5 | 0.0 | ae3cb0ae4c31 |
| human_aware_curriculum_s2 | P1 | 200 | 90.0 | 0.0 | 90.0 | 0.0 | 10.0 | d43140cb5a94 |
| human_aware_curriculum_s2 | P2 | 200 | 83.5 | 17.5 | 82.5 | 34.0 | 0.0 | d43140cb5a94 |

P1: physical no-human scene, phantom exact crossing observations. P2: physical exact crossing, frozen parked observations.
Selection rows show the existing 50-episode score of the winning candidate.

## Seed-0 selection outcomes

- `human_aware_sac_v1`: `callback_best`; model SHA-256 `4124b81633ac`. Selection unchanged; existing held-out results stand.
- `human_aware_random_full_s0`: `step_2000000`; model SHA-256 `5ed60cb286b0`. Held-out results use `heldout_common_id/`.

## Unexpected

- S8 common-ID scores and the complete no-human suite were reused under the mandated skip rule. No fresh no-human determinism rerun was performed; the stored result remains 3.0% task success and 0.0% collision.

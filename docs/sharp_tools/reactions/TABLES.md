
### exact set (preset `fixed_exact_h036_v1`, 100 episodes per cell, seeds from 31000)

| reaction | person | success % (95 % CI) | done any time % | intrusion before done % | hot contact % (95 % CI) | handle % | robot body % | exposure 1 s after intrusion (s) | p5 hot distance (mm) | time to done (s) | reaction time (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| keep | halts | 100 (96–100) | 100 | 70 | 25 (18–34) | 0 | 0 | 0.34 | -34 | 3.37 | – |
| keep | keeps_going | 100 (96–100) | 100 | 70 | 100 (96–100) | 100 | 100 | 0.66 | -35 | 3.37 | – |
| keep | pulls_back | 100 (96–100) | 100 | 70 | 18 (12–27) | 0 | 0 | 0.15 | -21 | 3.37 | – |
| retract | halts | 0 (0–4) | 46 | 70 | 0 (0–4) | 100 | 100 | 0.00 | 60 | 4.89 | 0.00 |
| retract | keeps_going | 83 (74–89) | 100 | 70 | 0 (0–4) | 100 | 100 | 0.00 | 55 | 7.07 | 0.00 |
| retract | pulls_back | 0 (0–4) | 30 | 70 | 0 (0–4) | 100 | 100 | 0.00 | 60 | 2.21 | 0.00 |
| stop | halts | 77 (68–84) | 77 | 70 | 25 (18–34) | 0 | 0 | 0.34 | -35 | 2.99 | 0.31 |
| stop | keeps_going | 100 (96–100) | 100 | 70 | 100 (96–100) | 100 | 100 | 0.66 | -36 | 4.26 | 0.63 |
| stop | pulls_back | 100 (96–100) | 100 | 70 | 18 (12–27) | 0 | 0 | 0.15 | -14 | 3.45 | 0.35 |

### random set (preset `evaluation_id`, 200 episodes per cell, seeds from 40000)

| reaction | person | success % (95 % CI) | done any time % | intrusion before done % | hot contact % (95 % CI) | handle % | robot body % | exposure 1 s after intrusion (s) | p5 hot distance (mm) | time to done (s) | reaction time (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| keep | halts | 100 (98–100) | 100 | 46 | 12 (8–17) | 3 | 0 | 0.25 | -36 | 3.54 | – |
| keep | keeps_going | 100 (98–100) | 100 | 46 | 66 (60–73) | 50 | 42 | 0.51 | -39 | 3.54 | – |
| keep | pulls_back | 100 (98–100) | 100 | 46 | 9 (6–14) | 0 | 0 | 0.11 | -22 | 3.54 | – |
| retract | halts | 20 (15–27) | 81 | 46 | 0 (0–2) | 36 | 36 | 0.00 | 46 | 4.64 | 0.00 |
| retract | keeps_going | 36 (30–43) | 91 | 46 | 0 (0–2) | 38 | 38 | 0.00 | 46 | 5.10 | 0.00 |
| retract | pulls_back | 16 (12–22) | 74 | 46 | 0 (0–2) | 30 | 31 | 0.00 | 44 | 4.14 | 0.00 |
| stop | halts | 89 (84–93) | 89 | 46 | 12 (9–18) | 0 | 0 | 0.25 | -32 | 3.45 | 0.32 |
| stop | keeps_going | 99 (96–100) | 100 | 46 | 66 (60–73) | 50 | 42 | 0.51 | -42 | 3.96 | 1.06 |
| stop | pulls_back | 100 (98–100) | 100 | 46 | 8 (5–13) | 0 | 0 | 0.10 | -6 | 3.58 | 0.31 |

### ood set (preset `evaluation_ood`, 200 episodes per cell, seeds from 50000)

| reaction | person | success % (95 % CI) | done any time % | intrusion before done % | hot contact % (95 % CI) | handle % | robot body % | exposure 1 s after intrusion (s) | p5 hot distance (mm) | time to done (s) | reaction time (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| keep | halts | 100 (98–100) | 100 | 34 | 0 (0–2) | 0 | 0 | 0.00 | 70 | 3.41 | – |
| keep | keeps_going | 100 (98–100) | 100 | 34 | 44 (37–50) | 40 | 36 | 0.36 | -40 | 3.41 | – |
| keep | pulls_back | 100 (98–100) | 100 | 34 | 0 (0–2) | 0 | 0 | 0.00 | 76 | 3.41 | – |
| retract | halts | 26 (20–32) | 92 | 34 | 0 (0–2) | 30 | 30 | 0.00 | 48 | 4.49 | 0.00 |
| retract | keeps_going | 39 (33–46) | 97 | 34 | 0 (0–2) | 32 | 33 | 0.00 | 61 | 4.50 | 0.00 |
| retract | pulls_back | 16 (12–22) | 84 | 34 | 0 (0–2) | 20 | 21 | 0.00 | 56 | 4.14 | 0.00 |
| stop | halts | 98 (94–99) | 98 | 34 | 0 (0–2) | 0 | 0 | 0.00 | 70 | 3.39 | 0.92 |
| stop | keeps_going | 98 (96–99) | 99 | 34 | 46 (40–53) | 46 | 36 | 0.42 | -39 | 3.94 | 1.09 |
| stop | pulls_back | 100 (98–100) | 100 | 34 | 0 (0–2) | 0 | 0 | 0.00 | 76 | 3.41 | – |

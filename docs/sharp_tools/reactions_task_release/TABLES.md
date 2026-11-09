
### exact set (preset `fixed_exact_h036_v1`, 200 episodes per cell, seeds from 31000)

Training env (`--env task`), release rule on, person window [-1.4, -1.4] s. Success = the env's own criterion (which ends a learner's episode); 'until env done' columns count contacts only up to that end, the others over the full 10 s.

| reaction | person | success % (95 % CI) | intrusion before done % | hot contact % until env done (95 % CI) | hot contact % full episode | robot body % until env done | robot body % full | handle % full | exposure until env done (s) | exposure full (s) | p5 hot distance (mm) | env done at (s) | reaction time (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| keep | halts | 100 (98–100) | 100 | 0 (0–2) | 0 | 0 | 0 | 0 | 0.95 | 8.90 | 5 | 2.05 | – |
| keep | keeps_going | 100 (98–100) | 100 | 100 (98–100) | 100 | 0 | 100 | 100 | 0.95 | 3.55 | -43 | 2.05 | – |
| keep | pulls_back | 100 (98–100) | 100 | 0 (0–2) | 0 | 0 | 0 | 0 | 0.10 | 0.10 | 26 | 2.05 | – |
| retract | halts | 100 (98–100) | 100 | 0 (0–2) | 0 | 100 | 100 | 100 | 0.00 | 0.00 | 58 | 7.75 | 0.00 |
| retract | keeps_going | 100 (98–100) | 100 | 0 (0–2) | 0 | 100 | 100 | 100 | 0.00 | 0.00 | 52 | 6.60 | 0.00 |
| retract | pulls_back | 100 (98–100) | 100 | 0 (0–2) | 0 | 100 | 100 | 100 | 0.00 | 0.00 | 61 | 8.55 | 0.00 |
| stop | halts | 100 (98–100) | 100 | 100 (98–100) | 100 | 0 | 0 | 0 | 0.90 | 8.65 | -4 | 2.25 | 0.20 |
| stop | keeps_going | 100 (98–100) | 100 | 100 (98–100) | 100 | 100 | 100 | 100 | 3.90 | 3.90 | -49 | 6.50 | 0.20 |
| stop | pulls_back | 100 (98–100) | 100 | 0 (0–2) | 0 | 0 | 0 | 0 | 0.00 | 0.00 | 34 | 2.40 | 0.20 |

### random set (preset `evaluation_id`, 500 episodes per cell, seeds from 40000)

Training env (`--env task`), release rule on, person window [-1.8, -0.2] s. Success = the env's own criterion (which ends a learner's episode); 'until env done' columns count contacts only up to that end, the others over the full 10 s.

| reaction | person | success % (95 % CI) | intrusion before done % | hot contact % until env done (95 % CI) | hot contact % full episode | robot body % until env done | robot body % full | handle % full | exposure until env done (s) | exposure full (s) | p5 hot distance (mm) | env done at (s) | reaction time (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| keep | halts | 100 (99–100) | 76 | 19 (16–23) | 19 | 0 | 0 | 3 | 0.24 | 1.77 | -46 | 2.05 | – |
| keep | keeps_going | 100 (99–100) | 75 | 41 (36–45) | 73 | 0 | 34 | 51 | 0.36 | 2.38 | -49 | 2.05 | – |
| keep | pulls_back | 100 (99–100) | 76 | 12 (10–15) | 12 | 0 | 0 | 0 | 0.09 | 0.09 | -27 | 2.05 | – |
| retract | halts | 86 (83–89) | 76 | 8 (6–11) | 8 | 26 | 26 | 34 | 0.17 | 0.17 | -25 | 5.58 | 0.00 |
| retract | keeps_going | 93 (90–95) | 75 | 8 (6–11) | 20 | 26 | 34 | 42 | 0.17 | 0.52 | -27 | 5.42 | 0.00 |
| retract | pulls_back | 76 (72–80) | 76 | 8 (6–11) | 8 | 25 | 26 | 33 | 0.18 | 0.18 | -24 | 5.53 | 0.00 |
| stop | halts | 84 (81–87) | 76 | 16 (13–20) | 16 | 0 | 0 | 0 | 1.35 | 1.76 | -33 | 2.09 | 0.21 |
| stop | keeps_going | 100 (99–100) | 75 | 44 (40–49) | 76 | 14 | 34 | 50 | 1.55 | 2.73 | -56 | 3.74 | 0.29 |
| stop | pulls_back | 100 (99–100) | 76 | 2 (1–4) | 2 | 0 | 0 | 0 | 0.03 | 0.03 | 10 | 2.18 | 0.19 |

### ood set (preset `evaluation_ood`, 500 episodes per cell, seeds from 50000)

Training env (`--env task`), release rule on, person window [-2.4, 1.6] s. Success = the env's own criterion (which ends a learner's episode); 'until env done' columns count contacts only up to that end, the others over the full 10 s.

| reaction | person | success % (95 % CI) | intrusion before done % | hot contact % until env done (95 % CI) | hot contact % full episode | robot body % until env done | robot body % full | handle % full | exposure until env done (s) | exposure full (s) | p5 hot distance (mm) | env done at (s) | reaction time (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| keep | halts | 100 (99–100) | 54 | 23 (20–27) | 23 | 6 | 6 | 20 | 0.22 | 0.24 | -46 | 2.05 | – |
| keep | keeps_going | 100 (99–100) | 54 | 30 (27–35) | 54 | 20 | 47 | 58 | 0.42 | 1.38 | -46 | 2.05 | – |
| keep | pulls_back | 100 (99–100) | 54 | 23 (20–27) | 23 | 6 | 6 | 20 | 0.15 | 0.15 | -45 | 2.05 | – |
| retract | halts | 100 (99–100) | 54 | 21 (17–24) | 21 | 25 | 25 | 30 | 0.08 | 0.08 | -42 | 4.45 | 0.00 |
| retract | keeps_going | 100 (99–100) | 54 | 20 (17–24) | 36 | 28 | 47 | 50 | 0.08 | 0.60 | -42 | 3.98 | 0.00 |
| retract | pulls_back | 96 (94–98) | 54 | 21 (17–24) | 21 | 24 | 24 | 30 | 0.08 | 0.08 | -42 | 4.78 | 0.00 |
| stop | halts | 77 (73–80) | 54 | 22 (19–26) | 22 | 5 | 5 | 15 | 2.31 | 2.31 | -21 | 2.06 | 0.04 |
| stop | keeps_going | 100 (99–100) | 54 | 37 (32–41) | 58 | 14 | 40 | 62 | 1.30 | 1.99 | -41 | 3.45 | 0.07 |
| stop | pulls_back | 100 (99–100) | 54 | 22 (18–26) | 22 | 5 | 5 | 19 | 0.19 | 0.19 | -23 | 2.22 | 0.04 |

# Open-source code and data to add to the repo (verified 7 Oct 2026)

Add as git submodules under `third_party/` or as pip dependencies. Licences
checked on the repository pages; re-check before redistributing assets.

## Pull in now

| repo | licence | what to take | how |
|---|---|---|---|
| `TUMcps/human-robot-gym` | not stated on the page, check LICENSE in the repo before vendoring | the human **motion-capture animations** (8–15 per task) and the idle/time-scaling animation logic; the task list for comparison. robosuite 1.5 + `mujoco` package, Python up to 3.13 | `git submodule add https://github.com/TUMcps/human-robot-gym third_party/human-robot-gym`; use the `icra2024` branch if you want the exact paper version |
| `TUMcps/sara-shield` | MIT | the provably safe shield (SSM and PFL modes) as a **baseline** that is tool-blind; Python bindings `safety_shield_py` | `pip install .` from the submodule; needs gcc, C++17, Eigen 3.4 |
| `google-deepmind/mujoco_menagerie` | MIT for AgileX PiPER; Leap (MIT), Allegro (BSD-2), Shadow (Apache-2) | the clean **PiPER MJCF** (compare with `agilex_piper/`), and the **Leap/Allegro hand** for the later dexterous-hand follow-up | `git submodule add https://github.com/google-deepmind/mujoco_menagerie third_party/menagerie` |
| `PKU-Alignment/safety-gymnasium` | Apache-2.0 | the **cost-API convention** `(obs, reward, cost, terminated, truncated, info)` and its Gymnasium wrappers, plus `ShadowHandCatchOver2UnderarmSafeFinger/SafeJoint` as reference safe-dexterous tasks | `pip install safety-gymnasium` (Python 3.8–3.10) |
| `PKU-Alignment/omnisafe` | Apache-2.0 | **Lagrangian SAC / PPO-Lag / CPO** implementations that consume `info["cost"]` from the hazard wrapper | `pip install omnisafe` |
| `DLR-RM/stable-baselines3` | MIT | already in use; keep SAC as the unconstrained baseline | already a dependency |

## Already copied into this module (figure rendering only)

`hazard/assets/hrg_human/` — the mesh humanoid from `TUMcps/human-robot-gym`
(`models/assets/human/human.xml`, 24 STL segments, three textures downscaled to 256 px).
Human-Robot Gym declares MIT in its package metadata and ships no LICENSE file; its README
says the model comes from `KlabCMU/kin-poly` (BSD-3-Clause), whose meshes are segments of
the SMPL neutral body (SMPL licence is non-commercial). Used only by `scripts/render_tasks.py`.
Full note in `hazard/assets/hrg_human/ATTRIBUTION.md`. For a commercial release replace the meshes.

## Pull in later (follow-up papers)

| repo | licence | why |
|---|---|---|
| `PKU-MARL/ReDMan` | check | safe RL for dexterous hands (Jenga), no human; reference for the hand follow-up |
| `PKU-MARL/DexterousHands` (Bi-DexHands) | Apache-2.0 | bimanual dexterous tasks in Isaac Gym; hand reorientation baselines |
| `NVlabs/DiSECt` | NVIDIA Source Code licence (non-commercial) | differentiable cutting, if you ever want real cutting physics instead of the kinematic slice event |
| `omron-sinicx/sliceit` | check | dual-simulator food slicing, same reason |
| `apple/ml-egodex` (EgoDex) | check per-dataset terms | 829 h of egocentric hand trajectories including kitchen tasks; source for human-hand motion beyond minimum-jerk |
| HOI4D, OpenEgo | per-dataset terms | same purpose |

## What is NOT taken from anywhere (your contribution)

- `hazard/` metrics and wrapper: no existing repo computes edge distance,
  alignment, closing speed or exposure for a held tool against a human.
- the knife asset and the five tasks in `docs/TASKS.md`.
- the counterfactual suites (person shown-but-absent, present-but-hidden).

## Dependencies for the hazard module alone

`mujoco>=3.0`, `gymnasium>=0.29`, `numpy`, `pytest` (tests). Nothing else.

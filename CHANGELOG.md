# Changelog

Reviewers respect a paper that documents its own corrections. This file records
every claim that changed between the August 2026 status documents and the
archived numbers, and why.

## [unreleased] — precision-floor instrumentation

### Corrected claims

| Claim in the August status docs | What the repository actually contains | Evidence |
|---|---|---|
| "v2 = 96.3 % (385/400)" | No stored evaluation of v2 existed. "96pct" appeared only in a release filename. | `results/precision/v2_n2000.csv` now supersedes it. |
| "v1 = 96.0 % (n=200)" | Reproduced at n=2000 on the same protocol; the point estimate moved. The old figure was inside its own Wilson interval, so this is a precision gain, not a contradiction. | `results/precision/v1_n2000.csv`, `docs/baseline_eval.txt` |
| "Tolerance sweep 45→10 mm exists for v1 and v2" | It was computed by `failure_analysis.py` and printed, never written to a file. | Every tolerance curve is now regenerated from archived CSVs by `reproduce.py`. |
| "v1 and v2 differ by their reward" | They do not. Both have `w_precision = 0.0`. `runs/corner_ft/config.json` differs from `runs/sac_full/config.json` only by `hard_corner_frac 0.35` and `lr 1e-4`. The release zip is byte-identical to `runs/corner_ft/final_model.zip`. | `md5sum release/piper_sac_v2_96pct.zip runs/corner_ft/final_model.zip` |
| (not previously stated) | **v1 is a *best*-checkpoint selection** (byte-identical to `runs/sac_full/best/best_model.zip`) while v2 is a *final*-model selection. Any v1-vs-v2 comparison therefore confounds the fine-tune, the extra steps, **and the selection rule**. | `md5sum release/piper_sac_v1_95pct.zip runs/sac_full/best/best_model.zip` |
| "±40 % joint-damping randomisation" | `agilex_piper/piper.xml` sets no joint damping, so `joint_damping_scale` multiplies zero. It is a no-op. The randomiser now reports `joint_damping_is_noop` in its info dict so the log cannot hide it. | `piper_rl/domain_rand.py`; asserted at runtime |
| "The 30 mm max step is the action limit" | The safety layer's joint-velocity clamp (`max_joint_vel · dt` = 0.05 rad/step at 20 Hz) is active on most steps of a typical episode and is at least as binding. It is now a swept variable with its own axis, and the clamp rate is logged per episode. | `clamp_rate` column of every results CSV |
| "Smoothing is applied to actions" | `piper_env.py` smooths the **joint target after IK**. It is a controller-side low-pass filter, which changes how the α axis should be read. | `piper_rl/piper_env.py`, step §2 |
| `docs/TRAINING_REPORT.md` describing a 220 k-step, 0/30 run | Superseded; it predated and contradicted the releases and was the first document a visitor read. Archived as `docs/TRAINING_REPORT_220k.md`. | — |
| Env docstring: 52-D observation | It is 51-D. | `env.observation_space.shape` |
| `piper_rl/README.md`: success bonus +120 | Config says 200. | `piper_rl/config.py:237` |

### Added

- `piper_rl/precision/` — instrumented env (TCP error at release, post-release
  scatter, full domain-randomisation vector), contact-solver overrides, the
  statistics module (Wilson, two-proportion z, c50 + bootstrap, IQM), and the
  Fourier-feature observation wrapper.
- `piper_rl/scripts/eval_precision.py` — per-episode CSV evaluation, parallel.
  **The paper's raw data.** No number in the paper comes from a print statement.
- `piper_rl/scripts/eval_checkpoints.py` — precision vs. training steps over
  every stored checkpoint, at zero training cost.
- `piper_rl/scripts/run_sweep.py` — the training campaign as one resumable
  command, with the Δmax/q̇max coupling and the control-rate hygiene encoded.
- `reproduce.py` — regenerates every table and figure from the archived CSVs.
- `tests/test_interface_flags.py` — every swept flag is asserted to reach the
  simulator. This is what protects the sweep from silent confounds.
- `tests/test_tolerance_semantics.py` — the success flag flips at exactly τ,
  the error is planar, and a post-hoc curve equals direct thresholding.
- `LICENSE` (MIT), `environment.yml`, pinned `requirements.txt`.

### Changed

- `requirements.txt` is pinned, not lower-bounded. A precision floor is a
  sub-millimetre statement about a contact simulation and cannot be reproduced
  against a floating MuJoCo version.
- Every run config and every results CSV records the git hash and the resolved
  package versions.

### Phase 0 measurements now archived (63,000 evaluation episodes, no training)

| Cell group | Files | What it settles |
|---|---|---|
| Baselines, n=2000 each | `v1_n2000`, `v2_n2000` | Tables 2-3 of the paper. c50 = 18.7 [18.1, 19.4] and 16.1 [15.7, 16.6] mm. |
| Contact-solver kill gate, n=1000 x 9 | `v2_solver_*` | H0d. c50 moves by at most 1.9 mm; two cells move beyond their interval, so the gate escalates to retraining. |
| Checkpoint curve, n=1000 x 38 | `ckpt_sac_full_*`, `ckpt_corner_ft_*` | H0b, within-run. The floor stops improving after ~1.2 M steps and then oscillates 16.8-24.3 mm. Xu et al.'s scaling form does not fit (r^2 = 0.17). |
| Nested scene x noise, 200 x 10 | `v2_nested_200x10` | ICC = 0.175 at the floor: five parts noise to one part scene. |
| Release decomposition | from the baseline CSVs | H0e rejected. Terminal error is the TCP error at release (r = 0.35-0.53), not post-release scatter (r^2 ~ 0.004). |

Two further facts that emerged and are recorded so they are not rediscovered:

- `model.opt.iterations` anywhere in [20, 100] is bit-identical for this task:
  the Newton solver converges in under 20. Below 5 the simulation stops solving
  the contacts and the arm fails outright. See
  `results/precision/solver_iterations_probe.md`.
- `release/piper_sac_v2_96pct.zip` is not the best checkpoint of its own run:
  `runs/corner_ft/checkpoints/*_1059408_steps.zip` reaches c50 = 14.6 mm against
  the release's 16.1 mm.

# Excluded warm-start evaluation protocol

- Run: `human_aware_curriculum_s2_warmstart_from1450k`
- Initialization: `runs/human_aware_curriculum_s2/checkpoints/piper_1450000_steps.zip`
- Initialization checkpoint SHA-256: `BD1E7D9B5FB1115323234C4F869C06D07580F226AECC439FC59285A4E0A31CA3`
- Continuation: 550,000 additional steps to global timestep 2,000,000.
- Recovery classification: warm start only; no compatible replay buffer, worker RNG, or callback state was available at initialization.
- Campaign status: excluded from `human_aware_multiseed_s0_s2`; it must not replace the interrupted seed-2 curriculum run or authorize `human_aware_random_full_s2`.
- Held-out model selection: final 2,000,000-step model only, explicitly chosen instead of the campaign's pre-registered checkpoint-grid selection because this warm-start does not contain the complete grid.
- Held-out protocol: the campaign's unchanged six distributions, episode counts, and seed ranges.

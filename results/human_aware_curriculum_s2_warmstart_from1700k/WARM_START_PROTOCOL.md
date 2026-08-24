# Excluded warm-start continuation protocol

- Run: `human_aware_curriculum_s2_warmstart_from1700k`
- Initialization: `runs/human_aware_curriculum_s2_interrupted_powerloss_20260817T210413Z/checkpoints/piper_1700000_steps.zip`
- Initialization checkpoint SHA-256: `ABC19D539DAAEE170FE0F8B463614AD59BA54E32CDA6F1537B59B56C0B79F654`
- Continuation: 300,000 additional steps to global timestep 2,000,000.
- Recovery classification: warm start only; no compatible replay buffer, worker RNG, or callback state was available at initialization.
- Campaign status: excluded from `human_aware_multiseed_s0_s2`; it must not replace the interrupted seed-2 curriculum run or authorize `human_aware_random_full_s2`.
- Curriculum schedule: global 2,000,000-step horizon retained.
- Held-out model selection: final 2,000,000-step model only, explicitly chosen because this warm-start does not contain the campaign's complete pre-registered checkpoint grid.
- Held-out protocol: the campaign's unchanged six distributions, episode counts, and seed ranges.

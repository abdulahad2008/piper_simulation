# Training report — superseded

**This file previously described a 220 k-step run that placed the object in
0 of 30 evaluation episodes.** That run is real (`models/sac_220k_partial.zip`)
and it is an honest record of an early state of the project, but it predates
and contradicts the released policies, and it was the first document a visitor
to this repository read. It has therefore been replaced by this pointer.

## What the repository actually contains, as of this commit

| Artefact | What it is | Where the number comes from |
|---|---|---|
| `release/piper_sac_v1_95pct.zip` | SAC, `runs/sac_full`, **best-checkpoint** selection (byte-identical to `runs/sac_full/best/best_model.zip`) | `results/precision/v1_n2000.csv` |
| `release/piper_sac_v2_96pct.zip` | the above fine-tuned in `runs/corner_ft` (hard-corner oversampling 0.35, lr 1e-4), **final-model** selection (byte-identical to `runs/corner_ft/final_model.zip`) | `results/precision/v2_n2000.csv` |
| `models/sac_220k_partial.zip` | the failed 220 k-step run this document used to describe | `docs/TRAINING_REPORT_220k.md` (archived) |

The percentages in the release filenames were measured at *n* = 200 on a
single evaluation seed block and are **not** the numbers to quote. Every
current number is regenerated from the archived per-episode CSVs by

```
python reproduce.py --results results/precision --out paper/generated
```

and `paper/generated/summary.md` is the file to read.

## Selection caveat that matters for any comparison of v1 and v2

v1 is a *best*-checkpoint selection and v2 is a *final*-model selection. Best-
checkpoint selection on a 15–50-episode evaluation callback is noise-driven and
biases the reported number upward. Any v1-vs-v2 difference therefore confounds
three things — the fine-tune, the extra 0.3 M steps, and the selection rule —
and none of them is the reward, which is identical (`w_precision = 0.0`) in
both. New runs should select the final checkpoint and say so.

## Reproducing a training run

See `docs/TRAINING_ON_YOUR_PC.md`. The interface flags added for the
precision study (`--max-step-dist`, `--action-smoothing`, `--control-hz`,
`--max-joint-vel`, `--timestep`, `--solref-scale`, `--solver-iters`,
`--impratio`, `--fourier-features`) are documented in
`python -m piper_rl.scripts.train --help`, and every one of them is covered by
`tests/test_interface_flags.py`.

# Is the Newton solver iteration count a live knob?

`v2_solver_iter20.csv` is **bit-identical** to `v2_solver_nominal.csv` in every
one of 1000 episodes, while `solver_iterations` in the log reads 20 and 100
respectively. That could mean the flag is mis-wired, so it was probed directly
(12 episodes, seeds 20000+, `release/piper_sac_v2_96pct.zip`, everything else
at nominal):

| `model.opt.iterations` | max abs. difference vs. nominal (100) | median terminal error |
|---|---|---|
| 20 | 0.0000 mm | 12.6 mm |
| 5  | 18.15 mm | 16.4 mm |
| 2  | 618.3 mm | 459.5 mm |
| 1  | 651.0 mm | 520.9 mm |

**Conclusion.** The knob is live; the Newton solver simply converges in fewer
than 20 iterations for this contact configuration, so anything in [20, 100] is
inert. Below about 5 iterations the solver no longer resolves the contacts at
all and the arm fails the task outright (a half-metre median error is the
object never being placed), so those settings are not a meaningful fidelity
comparison — they are a broken simulation.

This is why the kill-gate table reports iteration count as *exactly* zero
movement rather than "small movement": it is an identity, not a measurement,
and reporting it as a tight null would overstate the evidence.

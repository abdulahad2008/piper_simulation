# What is left for you, and nothing else

Everything in this file needs either your GitHub credentials, your compute, or a
judgement only you can make. Everything not in this file is done and committed
on the branch `precision-floor-instrumentation`.

---

## 1. Push the branch (5 minutes, needs your credentials)

The work is four commits on the branch `precision-floor-instrumentation`.

A push was attempted and refused, and the reason is specific rather than a
missing token: this session runs behind a git proxy that only injects a
credential for repositories in the session's authorized set, and
`abdulahad2008/piper_simulation` is not in it:

```
remote: access denied by the git proxy: abdulahad2008/piper_simulation is not
in this session's authorized repository set, so the proxy will not inject a
credential for it. To fix, add the repository to the session's sources.
```

So there are two ways to land it. Either **add the repository to this session's
sources** and ask me to push — that is one click on your side and nothing else
changes — or apply the bundle yourself:

**From the bundle** (`piper-precision-floor.bundle`, delivered in the chat):

```bash
cd /path/to/piper_simulation
git fetch /path/to/piper-precision-floor.bundle \
    precision-floor-instrumentation:precision-floor-instrumentation
git checkout precision-floor-instrumentation
git push -u origin precision-floor-instrumentation
```

Then open a PR against `main`, or fast-forward `main` if you prefer.

**Do not** merge before reading §2 — one commit changes files you may care
about.

## 2. Review these three changes before merging

1. `docs/TRAINING_REPORT.md` was replaced by a pointer; the original is
   preserved verbatim as `docs/TRAINING_REPORT_220k.md`. If you would rather
   delete it outright, that is your call.
2. `.gitignore` now ignores `__pycache__` and the compiled `.pyc` files were
   untracked (`git rm --cached`). Files on disk are untouched.
3. `runs/` (177 MB) and `out/` are **still committed**. They were left alone on
   purpose: they are what makes Phase 0 reproducible without a GPU. Before the
   paper release they should move to a GitHub Release or a Zenodo DOI and come
   out of the tree with `git rm -r --cached runs out`. That is a decision about
   your repository's history, not mine to make.

## 3. Run the training campaign (the only real compute cost)

Nothing below can be faked, borrowed from an open dataset, or substituted. It is
the paper.

```bash
# what would run, and the estimate
python -m piper_rl.scripts.run_sweep --dry-run

# the gate first -- six runs, and the study may stop here
python -m piper_rl.scripts.run_sweep --phase 0 --seeds 0 1 2

# then the sweep, then the controls
python -m piper_rl.scripts.run_sweep --phase 1 --seeds 0 1 2
python -m piper_rl.scripts.run_sweep --phase 2 --seeds 0 1 2
```

The runner is resumable, writes a manifest, and evaluates each cell at n=2000 as
soon as it finishes, so a crash costs one cell and never the archive. It selects
the **final** checkpoint, not the best.

Two things to do before you start:

- **Measure your throughput.** The sweep estimate assumes 6 h per run. On the
  development machine training is CPU-bound at 26--62 env-steps/s with 12
  gradient steps per env step. Move the SAC update to the GPU and re-measure
  before committing to 43 runs; the answer changes the plan.
- **Decide the gate.** Phase 0 says two solver cells move c50 beyond their
  interval (§4 below). The pre-registered rule sends you to the retrained
  extremes. Do that before Phase 1, not after.

## 4. The decision the data now forces

The eval-only kill gate bounds the contact solver's effect on c50 at **1.9 mm**
(2.7 mm total span across nine cells, nominal 16.4 mm). Two cells --
`solref x0.5` and the stiff/fine extreme -- move it by more than their interval.

So: **the interface sweep must move c50 by comfortably more than 2 mm or there
is no paper.** If the 30 -> 5 mm step axis moves it by, say, 2 mm, the honest
output is a short negative note, not a TMLR submission. Decide now that you will
write that note if the numbers say so; deciding later is how papers get
massaged.

The gate is also a genuine finding on its own: contact-solver settings change
how often the policy fails outright (S(45) ranges 88.6--95.8%) far more than
they change how precisely it places when it succeeds. That belongs in the paper
whichever way the sweep goes.

## 5. Fix the one claim that did not survive

The draft's motivating observation -- that the two released policies are
indistinguishable below 15 mm -- was an n=200 artefact. At n=2000 the difference
at 10 mm is +3.3 pp (p = 0.008) and the c50 intervals do not overlap. The
rewritten draft already says so. What you have to decide is whether the paper's
opening still works with the weaker, true version ("the effect of training
history decays as the tolerance tightens"). I think it does, and it is a better
motivation because it is a gradient rather than a coincidence, but that is an
authorial call.

## 6. Administrative, unblocked, and none of it needs me

- OpenReview profile (TMLR requires one). Do it now; it takes ten minutes and it
  is on the critical path for week 15.
- arXiv endorsement for cs.RO -- only after someone has replied to you once.
- Zenodo DOI for the checkpoints, at submission.
- Author name and email in `paper/main.tex` (currently `\todo`).
- **Do not send any cold email yet.** Every one of them is built around a number
  from Figure 1, and Figure 1 does not exist. `coldemails.md` says this itself
  and it is right.

## 7. Optional, and I would skip it for now

The physical-tolerance task (a pedestal target, so tau is a property of the
scene rather than a post-hoc ruler) is the single strongest answer to the
softest objection in the paper. It is about a week. Do it only if Phases 1--2
come in on schedule. The two-servo encoder rig is not on the critical path and
should not start before week 10.

---

## Known non-issue

`tests/test_human_aware_env.py::test_state_plus_rgb_extends_only_state_and_renders_headless`
fails in this sandbox with `GLFWError: The GLFW library is not initialized`. It
needs an OpenGL context to render the forehead camera and the container has
none; it is unrelated to anything in this branch and will pass on your machine.
The other 58 tests pass, including all 38 new ones.

---

## 8. The second environment (added after the HIL-SERL reframing)

`gym-hil` is installed, characterised, tested and wired into the same
evaluator, trainer and sweep runner. Nothing here is blocked on you except the
compute.

```bash
export MUJOCO_GL=osmesa            # needed once per shell; osmesa is installed
python -m piper_rl.scripts.run_sweep --env gymhil --dry-run
python -m piper_rl.scripts.run_sweep --env gymhil --seeds 0 1 2
```

24 runs at roughly an hour each — the whole cross-environment check is about a
day, against 258 h for the Piper sweep. Do this **before** the Piper Phase 1 if
you want an early read on whether H1 is about interfaces or about your arm: it
is by far the cheapest information in the plan.

Two things it already produced without any training:

- **gym-hil's released interface**, which its documentation does not state:
  25 mm per step, 10 Hz, no smoothing, no joint-velocity clamp, 100-step
  episodes. The base env ids apply *no* scaling at all — an action of 1.0
  commands a one-metre mocap displacement. The realised displacement of a
  saturated step is only ~40 % of the commanded cap (~10 mm), because
  operational-space control does not reach the mocap target within one 100 ms
  step.
- **`PandaArrangeBoxes` is not learnable from state as released**: success is a
  conjunction over five blocks, the state observation is block 1's position
  alone. Four blocks and five targets are unobservable. That is why the sweep
  uses `PandaPickCube`. This is worth reporting to the maintainers as an issue
  regardless of the paper.

Both facts answer, as measurements, the questions your cold-email drafts to
Michel Aractingi / Adil Zouitine and Stone Tao were going to *ask*. Rewrite
those two emails to lead with the answer rather than the question — an email
that tells a maintainer something true about their own package gets replied to.

Also fixed in passing: `libosmesa6` is now installed, so the headless-render
test that used to fail now passes. The suite is **75 passed, 0 failed**.

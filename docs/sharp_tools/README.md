# sharp_hrc — blade-hazard metrics and tasks for RL manipulation near people

Drop-in module for `piper_simulation`. It adds what Human-Robot Gym, HRIBench
and the current `PiperHumanAwarePickPlace-v0` env all lack: a safety signal
that depends on **what the gripper holds**, not only on where the robot's
links are.

```
hazard/
  geometry.py        segment–capsule distance (NumPy, no simulator needed)
  metrics.py         edge_distance, alignment, closing_speed, hazard, exposure
  mujoco_bridge.py   read blade sites + human capsules from a MuJoCo model
  wrapper.py         Gymnasium wrapper: info["cost"], info["hazard"], shaping, cut termination
  assets/knife.xml   rigid chef's knife MJCF with blade_heel / blade_tip / blade_frame sites
  assets/demo_scene.xml
  assets/hrg_human/  mesh humanoid for figures (from Human-Robot Gym / kin-poly, see ATTRIBUTION.md)
  tests/             11 unit tests (pytest)
scripts/demo_hazard.py   knife sweeps past a forearm, edge-in vs spine-in, prints the trace
scripts/render_tasks.py  renders Figure 1 (needs MUJOCO_GL=egl or osmesa, pillow, scipy)
docs/TASKS.md            the five pre-registered tasks T1–T5 and held-out suites
docs/THIRD_PARTY.md      open-source repos to pull in, licences, what to take from each
docs/INTEGRATION.md      step-by-step: add to piper_simulation today
paper/                   LaTeX draft (Overleaf-ready) + refs.bib
```

Quick check:

```
pip install mujoco gymnasium pytest
python -m pytest hazard/tests -q
python scripts/demo_hazard.py
```

The demo moves a knife along the same path twice at 0.3 m/s, never touching
the person. Edge facing the forearm: exposure 1.15 s, max hazard 1.06.
Spine facing the forearm: exposure 0. A link-collision metric sees no
difference between the two runs. That difference is the paper.

## Metric definitions (pre-registered defaults: d0 = 0.15 m, v0 = 0.25 m/s)

| symbol | meaning |
|---|---|
| d | signed distance from the edge segment to the nearest human capsule surface; d ≤ 0 is a cut |
| a | clip(cos ∠(cutting direction, edge→human), 0, 1); 1 = sharp side faces the person |
| v | relative approach speed of the edge toward the nearest human point, clipped at 0, only while a > 0 |
| h | a · (1 + v / v0) · clip(1 − d / d0, 0, 1) |
| E | Σ h · Δt over an episode (seconds of hazard-weighted exposure) |

Per episode the wrapper also reports min d, max h, max v, fraction of steps
with the edge facing within 15 cm, and the cut flag.

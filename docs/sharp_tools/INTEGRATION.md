# Adding this to piper_simulation today

1. Copy `hazard/` into `piper_rl/hazard/` and `scripts/demo_hazard.py` into
   `piper_rl/scripts/`. Run `python -m pytest piper_rl/hazard/tests -q`.

2. Knife in the scene. In `agilex_piper/` task XML, add the knife as a body
   with a free joint on the rack pose, and a weld equality to the gripper
   tool site that you enable when the grasp succeeds (the existing grasp
   detection can toggle `model.eq_active`). Simplest first version: make the
   knife a child body of the tool frame from the start (T2–T5), and add the
   grasp later for T1.

   ```xml
   <include file="hazard/assets/knife.xml"/>   <!-- or paste the body -->
   <equality><weld name="knife_weld" body1="tool_frame" body2="knife" active="false"/></equality>
   ```

3. Human geoms. Rename the forearm and hand geoms so they start with
   `human_` (or pass `human_prefix=` to the wrapper).

4. Wrap the env:

   ```python
   from piper_rl.hazard.wrapper import BladeHazardWrapper
   env = BladeHazardWrapper(gym.make("PiperHumanAwarePickPlace-v0"),
                            penalty_weight=0.0, dt=0.05)   # R0 / R2
   env = BladeHazardWrapper(..., penalty_weight=2.0)        # R1 shaped
   ```

   The eval scripts already log per-episode CSVs; add the keys from
   `info["hazard_summary"]` (exposure, min_edge_distance, max_hazard,
   max_closing_speed, frac_edge_facing, cut) to the row.

5. Observation. Append the 9 blade numbers (heel, tip, cutting direction in
   the base frame) to the observation vector. The counterfactual suites act on the
   13 human channels; the blade channels are left alone.

6. Add the repos in `docs/THIRD_PARTY.md` as submodules, and pin
   `omnisafe` for R2.

7. Commit the manifest for the held-out suites and the primary outcomes
   (`docs/TASKS.md`) before the first multi-seed run.

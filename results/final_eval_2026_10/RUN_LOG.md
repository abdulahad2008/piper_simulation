# Final evaluation run log

## Initial inspection
Start: 2026-10-05T01:30:39.1350324+05:00 (initial read-only inspection immediately before log creation)
Command: Get-Location; git status --short; git remote -v; rg --files (repository instructions and evaluation files)
End: 2026-10-05T01:30:39.1350324+05:00
Exit code: 0
Checkout: D:\Games\piper_sim; clean working tree; origin https://github.com/abdulahad2008/piper_simulation.git

Logger bootstrap
Start: 2026-10-05T01:30:39.1350324+05:00
Command: Create new final-evaluation log and temporary PowerShell command logger
End: 2026-10-05T01:30:39.1570197+05:00
Exit code: 0

### Command
Start: 2026-10-05T01:30:39.1770360+05:00
Command: Inspect repository instructions and evaluation code
End: 2026-10-05T01:30:39.3350477+05:00
Exit code: 1

### Command
Start: 2026-10-05T01:30:47.3651930+05:00
Command: git fetch origin
End: 2026-10-05T01:30:48.6327252+05:00
Exit code: 0

### Command
Start: 2026-10-05T01:30:48.6437547+05:00
Command: git checkout main
End: 2026-10-05T01:30:49.3083398+05:00
Exit code: 0

### Command
Start: 2026-10-05T01:30:49.3093321+05:00
Command: git pull
End: 2026-10-05T01:30:53.0196222+05:00
Exit code: 0

### Command
Start: 2026-10-05T01:30:53.0206106+05:00
Command: git merge-base --is-ancestor 972b88f HEAD
End: 2026-10-05T01:30:53.0605171+05:00
Exit code: 0

### Command
Start: 2026-10-05T01:30:53.0615189+05:00
Command: git checkout -b final-eval-2026-10
End: 2026-10-05T01:30:53.1062494+05:00
Exit code: 0

### Command
Start: 2026-10-05T01:31:06.7722572+05:00
Command: Record system versions and CPU metadata

HEAD: 972b88f4e101e2d1950b4e0fd59b3775e68be47e



Name                      : Intel(R) Xeon(R) CPU E5-2650 v4 @ 2.20GHz
NumberOfCores             : 12
NumberOfLogicalProcessors : 24

Name                      : Intel(R) Xeon(R) CPU E5-2650 v4 @ 2.20GHz
NumberOfCores             : 12
NumberOfLogicalProcessors : 24




End: 2026-10-05T01:31:07.8108647+05:00
Exit code: 0

### Command
Start: 2026-10-05T01:31:07.8188650+05:00
Command: Read evaluator, observation model, protocol and test configuration
End: 2026-10-05T01:31:07.9308559+05:00
Exit code: 0

### Command
Start: 2026-10-05T01:31:22.8271307+05:00
Command: Record Python/package versions (retry using stdin to preserve quoting)
Python: 3.11.9 (tags/v3.11.9:de54cf5, Apr  2 2024, 10:12:12) [MSC v.1938 64 bit (AMD64)]
torch: 2.13.0+cpu
mujoco: 3.11.0
stable-baselines3: 2.9.0
Logical CPU count: 48
End: 2026-10-05T01:31:40.3342390+05:00
Exit code: 0

### Command
Start: 2026-10-05T01:31:40.3432415+05:00
Command: .venv/Scripts/python.exe -B -m pytest tests/test_interface_flags.py -q

### Command
Start: 2026-10-05T01:31:48.7782014+05:00
Command: Inspect checkpoint inventory and observation implementation
End: 2026-10-05T01:31:49.6525304+05:00
Exit code: 0
End: 2026-10-05T01:32:00.4892675+05:00
Exit code: 0

### Command
Start: 2026-10-05T01:32:00.4912741+05:00
Command: .venv/Scripts/python.exe -B -m pytest tests/test_tolerance_semantics.py -q
End: 2026-10-05T01:32:08.7040155+05:00
Exit code: 0

### Command
Start: 2026-10-05T01:32:08.7050233+05:00
Command: .venv/Scripts/python.exe -B -m pytest tests/test_pick_place_regression.py -q

### Command
Start: 2026-10-05T01:32:13.3394803+05:00
Command: Inspect subprocess helper and evaluation configuration
End: 2026-10-05T01:32:13.5173873+05:00
Exit code: 0
End: 2026-10-05T01:32:18.4812327+05:00
Exit code: 0

### Command
Start: 2026-10-05T01:32:18.4832288+05:00
Command: .venv/Scripts/python.exe -B -m pytest tests/test_human_motion.py -q
End: 2026-10-05T01:32:26.0149846+05:00
Exit code: 0

### Command
Start: 2026-10-05T01:32:26.0159538+05:00
Command: .venv/Scripts/python.exe -B -m pytest tests/test_human_aware_env.py -q
End: 2026-10-05T01:32:38.2066237+05:00
Exit code: 0

### Command
Start: 2026-10-05T01:33:20.9358746+05:00
Command: Create temporary guarded Step-1 launcher (evaluation only, resumable, command logging)
End: 2026-10-05T01:33:21.0099043+05:00
Exit code: 0

### Command
Start: 2026-10-05T01:33:34.6045765+05:00
Command: Step 1: .venv/Scripts/python.exe -B -m piper_rl.scripts.run_curriculum_hold_campaign --seeds 8 --safety-pilot --run human_aware_curriculum_hold_safety_s8 (temporary launcher verifies already_trained, blocks training/runs writes, skips complete JSON outputs, logs every subprocess)

Skipped complete evaluation (50 episodes): results\human_aware_curriculum_hold_safety_s8\checkpoint_validation\step_100000.json

Skipped complete evaluation (50 episodes): results\human_aware_curriculum_hold_safety_s8\checkpoint_validation\step_200000.json

Skipped complete evaluation (50 episodes): results\human_aware_curriculum_hold_safety_s8\checkpoint_validation\step_400000.json

Skipped complete evaluation (50 episodes): results\human_aware_curriculum_hold_safety_s8\checkpoint_validation\step_600000.json

Skipped complete evaluation (50 episodes): results\human_aware_curriculum_hold_safety_s8\checkpoint_validation\step_1000000.json

Skipped complete evaluation (50 episodes): results\human_aware_curriculum_hold_safety_s8\checkpoint_validation\step_1500000.json

Skipped complete evaluation (50 episodes): results\human_aware_curriculum_hold_safety_s8\checkpoint_validation\step_2000000.json

Skipped complete evaluation (50 episodes): results\human_aware_curriculum_hold_safety_s8\checkpoint_validation\callback_best.json

Preserved existing model with matching SHA-256: runs\human_aware_curriculum_hold_safety_s8\selected_common_id_model.zip

Skipped complete evaluation (200 episodes): results\human_aware_curriculum_hold_safety_s8\heldout\no_human.json

### Evaluation subprocess
Start: 2026-10-04T20:33:40.249941+00:00
Command: D:\Games\piper_sim\.venv\Scripts\python.exe -B -m piper_rl.scripts.evaluate_human_aware --model runs\human_aware_curriculum_hold_safety_s8\selected_common_id_model.zip --algo sac --experiment fixed-exact-h036-v1 --episodes 200 --seed 31000 --out results\human_aware_curriculum_hold_safety_s8\heldout\exact_h036 --video results\human_aware_curriculum_hold_safety_s8\videos\exact_h036.mp4 --video-episodes 1

### Command
Start: 2026-10-05T01:35:11.4921457+05:00
Command: Create seed0_common_selection.py using preregistered campaign helpers
End: 2026-10-05T01:35:11.5657050+05:00
Exit code: 0

### Command
Start: 2026-10-05T01:35:25.1081661+05:00
Command: Step-1 first-minute monitoring: confirm only evaluation subprocesses are active
End: 2026-10-05T01:35:26.2377827+05:00
Exit code: 0

### Command
Start: 2026-10-05T01:36:38.9964890+05:00
Command: Read Step-1 progress and active evaluator resource use
End: 2026-10-05T01:36:39.1156547+05:00
Exit code: 0

### Command
Start: 2026-10-05T01:37:25.6271635+05:00
Command: Record setup results and inspect earlier evaluation timings without changing protocol

Setup tests: test_interface_flags.py 26 passed; test_tolerance_semantics.py 12 passed; test_pick_place_regression.py 3 passed; test_human_motion.py 8 passed; test_human_aware_env.py 10 passed (rendering included). Each ran in its own process.
Tooling note: initial no-match rg search returned 1; initial version-query Python subcommand returned 1 due to PowerShell quoting although surrounding command returned 0. The stdin-based retry succeeded.
End: 2026-10-05T01:37:25.7461345+05:00
Exit code: 0

### Command
Start: 2026-10-05T01:38:02.6637564+05:00
Command: Inspect current s8 evaluation progress and immutable-result status
End: 2026-10-05T01:38:03.1091247+05:00
Exit code: 0

### Command
Start: 2026-10-05T01:38:32.8682790+05:00
Command: Prepare evaluation-only human-observation override in temporary file (not yet installed)
End: 2026-10-05T01:38:32.9432790+05:00
Exit code: 0

### Command
Start: 2026-10-05T01:39:49.8001907+05:00
Command: Prepare human-cue regression tests in temporary file
End: 2026-10-05T01:39:49.8791904+05:00
Exit code: 0

### Command
Start: 2026-10-05T01:40:32.6753032+05:00
Command: Check GitHub CLI availability and repository identity for final PR
End: 2026-10-05T01:40:32.9503035+05:00
Exit code: 0

### Command
Start: 2026-10-05T01:41:06.4127417+05:00
Command: Read Git credential-helper configuration and record s8 baseline provenance

S8 preflight: final model, all seven GRID checkpoint files, and callback-best exist. run_is_trained=True.
S8 selection reused: step_1000000; selected SHA-256: 0721ac0dada8e2239f24f386eef6c90cdbda3c091c0317d21794d279d0e8bee5.
S8 committed no-human baseline: n=200, task_success_rate=0.03, human_collision_rate=0.0. Completed output reused under hard rule 5; no fresh no-human rerun was performed.
End: 2026-10-05T01:41:12.6200320+05:00
Exit code: 0

End: 2026-10-04T20:41:21.687870+00:00
Exit code: 0

### Evaluation subprocess
Start: 2026-10-04T20:41:21.698355+00:00
Command: D:\Games\piper_sim\.venv\Scripts\python.exe -B -m piper_rl.scripts.evaluate_human_aware --model runs\human_aware_curriculum_hold_safety_s8\selected_common_id_model.zip --algo sac --experiment fixed --episodes 200 --seed 31000 --out results\human_aware_curriculum_hold_safety_s8\heldout\constrained_fixed_height_randomized

### Command
Start: 2026-10-05T01:41:46.5224657+05:00
Command: Check GitHub API authentication through Git credential manager (credentials captured, never printed)
End: 2026-10-05T01:41:48.4271820+05:00
Exit code: 0

### Command
Start: 2026-10-05T01:43:19.2268830+05:00
Command: Inspect completed s8 exact-crossing result and current sequential evaluator

Completed s8 exact_h036: {"experiment": "fixed-exact-h036-v1", "distribution": "fixed_exact_h036_v1", "episodes": 200, "deterministic_prediction": true, "task_success_rate": 0.0, "task_success_95ci": [0.0, 0.018846005918320894], "collision_free_success_rate": 0.0, "human_collision_rate": 0.895, "human_collision_95ci": [0.8448186177864638, 0.9302930375380627], "near_miss_rate": 0.93, "mean_minimum_separation": 0.014455880760610324, "median_minimum_separation": -0.0001420603505934842, "p05_minimum_separation": -0.026178326460983486, "worst_minimum_separation": -0.07044417821634405, "mean_completion_time_successes": null, "mean_episode_duration": 6.467249999999999, "mean_episode_length": 129.345, "mean_placement_error": 0.3335601024687324, "median_placement_error": 0.32899985199996007, "p90_placement_error": 0.45055390530792294, "grasp_rate": 0.21, "lift_rate": 0.02, "mean_safety_interventions": 109.895, "mean_safety_clamped_steps": 109.895, "mean_safety_clamp_rate": 0.8435682956397383, "mean_proximity_steps": 42.435, "mean_waiting_steps": 32.245, "mean_waiting_time": 1.61225, "mean_episode_reward": -148.9176931792005, "mean_human_safety_cost": 121.00421045351642}
End: 2026-10-05T01:43:20.2550403+05:00
Exit code: 0

### Command
Start: 2026-10-05T01:43:57.7367176+05:00
Command: Prepare default-actual determinism comparison helper (outcomes, first 20 committed rows)
End: 2026-10-05T01:43:57.8107175+05:00
Exit code: 0

### Command
Start: 2026-10-05T01:44:25.4536648+05:00
Command: Tighten prepared determinism comparison to every evaluator CSV field
End: 2026-10-05T01:44:26.0554085+05:00
Exit code: 0

### Command
Start: 2026-10-05T01:48:23.2053105+05:00
Command: Prepare sequential probe launcher with prerequisite determinism gate and watchdog
End: 2026-10-05T01:48:23.2913106+05:00
Exit code: 0

End: 2026-10-04T20:49:00.161397+00:00
Exit code: 0

### Evaluation subprocess
Start: 2026-10-04T20:49:00.171397+00:00
Command: D:\Games\piper_sim\.venv\Scripts\python.exe -B -m piper_rl.scripts.evaluate_human_aware --model runs\human_aware_curriculum_hold_safety_s8\selected_common_id_model.zip --algo sac --experiment fixed-exact-shifted-h036-v1 --episodes 200 --seed 32000 --out results\human_aware_curriculum_hold_safety_s8\heldout\shifted_exact_h036

### Command
Start: 2026-10-05T01:49:20.3050173+05:00
Command: Apply a shared four-hour watchdog budget to the entire seed-0 step
End: 2026-10-05T01:49:20.8936818+05:00
Exit code: 0

### Command
Start: 2026-10-05T01:51:37.0753742+05:00
Command: Prepare final summary generator and record completed constrained-crossing metrics

Completed s8 constrained_fixed_height_randomized: {"experiment": "fixed", "distribution": "constrained_fixed_height_randomized", "episodes": 200, "deterministic_prediction": true, "task_success_rate": 0.0, "task_success_95ci": [0.0, 0.018846005918320894], "collision_free_success_rate": 0.0, "human_collision_rate": 0.92, "human_collision_95ci": [0.8740095102048769, 0.9501598448237337], "near_miss_rate": 0.95, "mean_minimum_separation": 0.011985467994953668, "median_minimum_separation": -0.0001520033811653193, "p05_minimum_separation": -0.024585488156155515, "worst_minimum_separation": -0.07039385410522853, "mean_completion_time_successes": null, "mean_episode_duration": 6.54875, "mean_episode_length": 130.975, "mean_placement_error": 0.3219591990483018, "median_placement_error": 0.3213670822175375, "p90_placement_error": 0.4290807778835445, "grasp_rate": 0.25, "lift_rate": 0.02, "mean_safety_interventions": 111.315, "mean_safety_clamped_steps": 111.315, "mean_safety_clamp_rate": 0.8448970029735255, "mean_proximity_steps": 43.445, "mean_waiting_steps": 33.44, "mean_waiting_time": 1.6720000000000002, "mean_episode_reward": -147.39274520287222, "mean_human_safety_cost": 123.94051005996998}
End: 2026-10-05T01:51:37.6988992+05:00
Exit code: 0

End: 2026-10-04T20:55:50.587560+00:00
Exit code: 0

### Evaluation subprocess
Start: 2026-10-04T20:55:50.598558+00:00
Command: D:\Games\piper_sim\.venv\Scripts\python.exe -B -m piper_rl.scripts.evaluate_human_aware --model runs\human_aware_curriculum_hold_safety_s8\selected_common_id_model.zip --algo sac --experiment randomized-id --episodes 500 --seed 40000 --out results\human_aware_curriculum_hold_safety_s8\heldout\randomized_id --video results\human_aware_curriculum_hold_safety_s8\videos\randomized_id.mp4 --video-episodes 1

### Command
Start: 2026-10-05T01:57:39.0022404+05:00
Command: Verify protocol constants against the preregistration and record shifted-crossing result

Protocol constants verified against committed analysis_manifest.json.
Completed s8 shifted_exact_h036: {"experiment": "fixed-exact-shifted-h036-v1", "distribution": "fixed_exact_shifted_h036_v1", "episodes": 200, "deterministic_prediction": true, "task_success_rate": 0.0, "task_success_95ci": [0.0, 0.018846005918320894], "collision_free_success_rate": 0.0, "human_collision_rate": 0.92, "human_collision_95ci": [0.8740095102048769, 0.9501598448237337], "near_miss_rate": 0.955, "mean_minimum_separation": 0.012759896149161985, "median_minimum_separation": -0.00014726728002814604, "p05_minimum_separation": -0.025795018686336737, "worst_minimum_separation": -0.06617182197583313, "mean_completion_time_successes": null, "mean_episode_duration": 5.844000000000001, "mean_episode_length": 116.88, "mean_placement_error": 0.31756754893269756, "median_placement_error": 0.3202899840409539, "p90_placement_error": 0.42601721354667216, "grasp_rate": 0.135, "lift_rate": 0.03, "mean_safety_interventions": 97.785, "mean_safety_clamped_steps": 97.785, "mean_safety_clamp_rate": 0.8310963630971179, "mean_proximity_steps": 39.945, "mean_waiting_steps": 29.66, "mean_waiting_time": 1.483, "mean_episode_reward": -144.30718822261224, "mean_human_safety_cost": 121.58251036676057}
End: 2026-10-05T01:57:45.0999401+05:00
Exit code: 0

### Command
Start: 2026-10-05T01:58:42.0580340+05:00
Command: Make new seed-0 selected-model publication atomic for power-loss recovery
End: 2026-10-05T01:58:42.6705589+05:00
Exit code: 0

### Command
Start: 2026-10-05T02:00:10.7967630+05:00
Command: Inspect resumable CSV metadata augmentation and total CPU core count

CPU totals: 24 physical cores; 48 logical processors across 2 sockets
End: 2026-10-05T02:00:11.5081158+05:00
Exit code: 0

### Command
Start: 2026-10-05T02:06:28.8817291+05:00
Command: Record the guarded campaign launcher and the no-human reuse exception for review/resumption

## Step 1 guard source for review and resumption

The launcher calls the requested campaign main with the exact arguments, verifies already_trained, blocks training and runs writes, preserves the existing matching selected model, and skips completed JSON outputs. It does not change experiments, episode counts, seeds, or selection keys.

```python
from pathlib import Path
from datetime import datetime, timezone
import json, subprocess, sys, time, shutil

ROOT = Path(r'D:\Games\piper_sim')
sys.path.insert(0, str(ROOT))
LOG = ROOT / 'results/final_eval_2026_10/RUN_LOG.md'
ALLOWED = ROOT / 'results/human_aware_curriculum_hold_safety_s8'
started = time.monotonic()

def note(text):
    with LOG.open('a', encoding='utf-8') as f:
        f.write('\n' + text + '\n')

def now():
    return datetime.now(timezone.utc).isoformat()

from piper_rl.scripts import run_multiseed_campaign as campaign
from piper_rl.scripts import run_curriculum_hold_campaign as hold

original_copy = shutil.copy2

def safe_copy(source, destination, *args, **kwargs):
    destination = Path(destination)
    if destination.exists():
        if campaign.sha256(Path(source)) != campaign.sha256(destination):
            raise RuntimeError(f'Existing model cannot be overwritten: {destination}')
        note(f'Preserved existing model with matching SHA-256: {destination}')
        return destination
    raise RuntimeError(f'Step 1 unexpectedly needs to create a model: {destination}')

shutil.copy2 = safe_copy

def forbidden(*args, **kwargs):
    raise RuntimeError('Training is forbidden in final evaluation')

hold.training_command = forbidden
hold.run_training_logged = forbidden
campaign.training_command = forbidden
campaign.run_training_logged = forbidden

def audit(event, args):
    if event == 'open':
        name, mode, flags = args
        if isinstance(name, (str, bytes)):
            p = Path(name).resolve()
            writing = (mode and any(c in mode for c in 'wax+')) or (flags & (1 | 2 | 64 | 512))
            if writing and (ROOT / 'runs') in p.parents:
                raise RuntimeError(f'Write under runs is forbidden: {p}')
    if event == 'subprocess.Popen':
        command = args[1]
        if isinstance(command, (list, tuple)) and 'piper_rl.scripts.train' in command:
            raise RuntimeError('Training subprocess is forbidden')

sys.addaudithook(audit)

def guarded_run(command, log_path):
    if 'piper_rl.scripts.evaluate_human_aware' not in command or 'piper_rl.scripts.train' in command:
        raise RuntimeError(f'Non-evaluation subprocess rejected: {command}')
    prefix = Path(command[command.index('--out') + 1])
    if ALLOWED not in prefix.resolve().parents:
        raise RuntimeError(f'Unexpected result destination: {prefix}')
    episodes = int(command[command.index('--episodes') + 1])
    output = prefix.with_suffix('.json')
    if output.exists():
        count = campaign.load_json(output)['summary']['episodes']
        if count != episodes:
            raise RuntimeError(f'Existing output has {count} episodes, expected {episodes}: {output}')
        note(f'Skipped complete evaluation ({episodes} episodes): {output}')
        print(f'SKIP {prefix} ({episodes} episodes)', flush=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    begin = now()
    note(f'### Evaluation subprocess\nStart: {begin}\nCommand: {subprocess.list2cmdline(command)}')
    print('EVALUATE:', subprocess.list2cmdline(command), flush=True)
    try:
        with log_path.open('w', encoding='utf-8') as f:
            process = subprocess.Popen(command, stdout=f, stderr=subprocess.STDOUT, text=True)
            while process.poll() is None:
                if time.monotonic() - started > 7200:
                    subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], check=False)
                    process.wait()
                    raise RuntimeError('Step 1 exceeded twice its one-hour estimate; process tree stopped')
                time.sleep(1)
        code = process.returncode
        if code:
            raise RuntimeError(f'Evaluation failed ({code}); see {log_path}')
        if campaign.load_json(output)['summary']['episodes'] != episodes:
            raise RuntimeError(f'Wrong episode count: {output}')
    finally:
        note(f'End: {now()}\nExit code: {process.returncode if "process" in locals() else 1}')

campaign.run_logged = guarded_run
hold.configure_campaign((8,), safety_pilot=True)
spec = hold.SPECS[0]
assert hold.run_is_trained(spec), 'run_is_trained is false: refusing campaign command'
assert all((campaign.run_dir(spec) / 'checkpoints' / f'piper_{step}_steps.zip').exists() for step in campaign.GRID)
print('VERIFIED already_trained=True; training hooks and writes under runs are blocked', flush=True)
code = hold.main(['--seeds', '8', '--safety-pilot', '--run', spec.run_name])
value = campaign.load_json(ALLOWED / 'heldout/no_human.json')['summary']
assert value['task_success_rate'] == .03 and value['human_collision_rate'] == 0.0, value
for name, _, n, _ in campaign.EVALUATIONS:
    assert campaign.load_json(ALLOWED / 'heldout' / f'{name}.json')['summary']['episodes'] == n
assert hold.state()['runs'][spec.run_name]['status'] == 'completed'
print('STEP 1 COMPLETE; determinism and six episode counts verified', flush=True)
raise SystemExit(code)

```
End: 2026-10-05T02:06:29.4691648+05:00
Exit code: 0

### Command
Start: 2026-10-05T02:12:30.9730666+05:00
Command: Monitor active randomized-ID process, elapsed time, and completed suite count
End: 2026-10-05T02:12:31.5146181+05:00
Exit code: 0

End: 2026-10-04T21:18:19.618518+00:00
Exit code: 0

### Evaluation subprocess
Start: 2026-10-04T21:18:19.635517+00:00
Command: D:\Games\piper_sim\.venv\Scripts\python.exe -B -m piper_rl.scripts.evaluate_human_aware --model runs\human_aware_curriculum_hold_safety_s8\selected_common_id_model.zip --algo sac --experiment ood --episodes 500 --seed 50000 --out results\human_aware_curriculum_hold_safety_s8\heldout\ood --video results\human_aware_curriculum_hold_safety_s8\videos\ood.mp4 --video-episodes 1

### Command
Start: 2026-10-05T02:18:49.2581049+05:00
Command: Record the completed 500-episode s8 randomized-ID result

Completed s8 randomized_id: {"experiment": "randomized-id", "distribution": "evaluation_id", "episodes": 500, "deterministic_prediction": true, "task_success_rate": 0.002, "task_success_95ci": [0.000353127309584542, 0.011240992747195207], "collision_free_success_rate": 0.002, "human_collision_rate": 0.254, "human_collision_95ci": [0.2178196276777558, 0.2939316846394487], "near_miss_rate": 0.288, "mean_minimum_separation": 0.1481443225049958, "median_minimum_separation": 0.16645353327515328, "p05_minimum_separation": -0.0002469004488422313, "worst_minimum_separation": -0.045423080180884035, "mean_completion_time_successes": 3.1, "mean_episode_duration": 7.970299999999999, "mean_episode_length": 159.406, "mean_placement_error": 0.3307476718051237, "median_placement_error": 0.32770776497136456, "p90_placement_error": 0.44296833092096305, "grasp_rate": 0.184, "lift_rate": 0.014, "mean_safety_interventions": 137.814, "mean_safety_clamped_steps": 137.814, "mean_safety_clamp_rate": 0.8493175008527286, "mean_proximity_steps": 16.47, "mean_waiting_steps": 11.868, "mean_waiting_time": 0.5934, "mean_episode_reward": -76.05579882684295, "mean_human_safety_cost": 34.71093568633164}
End: 2026-10-05T02:18:49.8718460+05:00
Exit code: 0

### Command
Start: 2026-10-05T02:29:10.7758470+05:00
Command: Monitor final s8 OOD process and the two-hour campaign watchdog
End: 2026-10-05T02:29:11.2943661+05:00
Exit code: 0

End: 2026-10-04T21:39:58.591130+00:00
Exit code: 0
End: 2026-10-05T02:39:59.4194365+05:00
Exit code: 0

### Command
Start: 2026-10-05T02:40:18.4224227+05:00
Command: Verify completed s8 state and record OOD summary

Step 1 completed; six requested episode counts verified. No training executed; existing selected model preserved.
Completed s8 ood: {"experiment": "ood", "distribution": "evaluation_ood", "episodes": 500, "deterministic_prediction": true, "task_success_rate": 0.01, "task_success_95ci": [0.004278690781520357, 0.023193435378764938], "collision_free_success_rate": 0.01, "human_collision_rate": 0.244, "human_collision_95ci": [0.20839823999994375, 0.2835055646878788], "near_miss_rate": 0.29, "mean_minimum_separation": 0.12151894020343762, "median_minimum_separation": 0.13443950747366598, "p05_minimum_separation": -0.0003481759379440841, "worst_minimum_separation": -0.03208549703214965, "mean_completion_time_successes": 4.42, "mean_episode_duration": 7.7128000000000005, "mean_episode_length": 154.256, "mean_placement_error": 0.297145314140226, "median_placement_error": 0.3000659634006173, "p90_placement_error": 0.40030628195160145, "grasp_rate": 0.18, "lift_rate": 0.006, "mean_safety_interventions": 132.526, "mean_safety_clamped_steps": 132.526, "mean_safety_clamp_rate": 0.8372336243118544, "mean_proximity_steps": 10.052, "mean_waiting_steps": 4.678, "mean_waiting_time": 0.23390000000000002, "mean_episode_reward": -64.35597595203419, "mean_human_safety_cost": 31.208003832391906}
End: 2026-10-05T02:40:18.8057406+05:00
Exit code: 0

### Command
Start: 2026-10-05T02:40:18.8157419+05:00
Command: .venv/Scripts/python.exe -B -m piper_rl.scripts.seed0_common_selection

Candidate human_aware_sac_v1 / step_100000: runs\human_aware_sac_v1\checkpoints\piper_100000_steps.zip; exists=True

Candidate human_aware_sac_v1 / step_200000: runs\human_aware_sac_v1\checkpoints\piper_200000_steps.zip; exists=True

Candidate human_aware_sac_v1 / step_400000: runs\human_aware_sac_v1\checkpoints\piper_400000_steps.zip; exists=True

Candidate human_aware_sac_v1 / step_600000: runs\human_aware_sac_v1\checkpoints\piper_600000_steps.zip; exists=True

Candidate human_aware_sac_v1 / step_1000000: runs\human_aware_sac_v1\checkpoints\piper_1000000_steps.zip; exists=True

Candidate human_aware_sac_v1 / step_1500000: runs\human_aware_sac_v1\checkpoints\piper_1500000_steps.zip; exists=True

Candidate human_aware_sac_v1 / step_2000000: runs\human_aware_sac_v1\final_model.zip; exists=True

Candidate human_aware_sac_v1 / callback_best: runs\human_aware_sac_v1\best\best_model.zip; exists=True

### Evaluation subprocess
Start: 2026-10-04T21:40:25.508438+00:00
Command: D:\Games\piper_sim\.venv\Scripts\python.exe -B -m piper_rl.scripts.evaluate_human_aware --model runs\human_aware_sac_v1\checkpoints\piper_100000_steps.zip --algo sac --experiment randomized-id --episodes 50 --seed 21000 --out results\human_aware_sac_v1\checkpoint_validation_common_id\step_100000
Estimate: 30 min; watchdog: 3600 seconds
Output log: results\human_aware_sac_v1\checkpoint_validation_common_id\logs\step_100000_20261004T214025507430Z_1.log

End: 2026-10-04T21:42:29.622036+00:00
Exit code: 0

### Evaluation subprocess
Start: 2026-10-04T21:42:29.638916+00:00
Command: D:\Games\piper_sim\.venv\Scripts\python.exe -B -m piper_rl.scripts.evaluate_human_aware --model runs\human_aware_sac_v1\checkpoints\piper_200000_steps.zip --algo sac --experiment randomized-id --episodes 50 --seed 21000 --out results\human_aware_sac_v1\checkpoint_validation_common_id\step_200000
Estimate: 30 min; watchdog: 3600 seconds
Output log: results\human_aware_sac_v1\checkpoint_validation_common_id\logs\step_200000_20261004T214229638382Z_1.log

End: 2026-10-04T21:44:30.811965+00:00
Exit code: 0

### Evaluation subprocess
Start: 2026-10-04T21:44:30.827953+00:00
Command: D:\Games\piper_sim\.venv\Scripts\python.exe -B -m piper_rl.scripts.evaluate_human_aware --model runs\human_aware_sac_v1\checkpoints\piper_400000_steps.zip --algo sac --experiment randomized-id --episodes 50 --seed 21000 --out results\human_aware_sac_v1\checkpoint_validation_common_id\step_400000
Estimate: 30 min; watchdog: 3600 seconds
Output log: results\human_aware_sac_v1\checkpoint_validation_common_id\logs\step_400000_20261004T214430827953Z_1.log

End: 2026-10-04T21:46:55.997369+00:00
Exit code: 0

### Evaluation subprocess
Start: 2026-10-04T21:46:56.014081+00:00
Command: D:\Games\piper_sim\.venv\Scripts\python.exe -B -m piper_rl.scripts.evaluate_human_aware --model runs\human_aware_sac_v1\checkpoints\piper_600000_steps.zip --algo sac --experiment randomized-id --episodes 50 --seed 21000 --out results\human_aware_sac_v1\checkpoint_validation_common_id\step_600000
Estimate: 30 min; watchdog: 3600 seconds
Output log: results\human_aware_sac_v1\checkpoint_validation_common_id\logs\step_600000_20261004T214656014081Z_1.log

End: 2026-10-04T21:49:05.432212+00:00
Exit code: 0

### Evaluation subprocess
Start: 2026-10-04T21:49:05.448210+00:00
Command: D:\Games\piper_sim\.venv\Scripts\python.exe -B -m piper_rl.scripts.evaluate_human_aware --model runs\human_aware_sac_v1\checkpoints\piper_1000000_steps.zip --algo sac --experiment randomized-id --episodes 50 --seed 21000 --out results\human_aware_sac_v1\checkpoint_validation_common_id\step_1000000
Estimate: 30 min; watchdog: 3600 seconds
Output log: results\human_aware_sac_v1\checkpoint_validation_common_id\logs\step_1000000_20261004T214905448210Z_1.log

End: 2026-10-04T21:50:48.262733+00:00
Exit code: 0

### Evaluation subprocess
Start: 2026-10-04T21:50:48.278731+00:00
Command: D:\Games\piper_sim\.venv\Scripts\python.exe -B -m piper_rl.scripts.evaluate_human_aware --model runs\human_aware_sac_v1\checkpoints\piper_1500000_steps.zip --algo sac --experiment randomized-id --episodes 50 --seed 21000 --out results\human_aware_sac_v1\checkpoint_validation_common_id\step_1500000
Estimate: 30 min; watchdog: 3600 seconds
Output log: results\human_aware_sac_v1\checkpoint_validation_common_id\logs\step_1500000_20261004T215048277731Z_1.log

End: 2026-10-04T21:52:00.796883+00:00
Exit code: 0

### Evaluation subprocess
Start: 2026-10-04T21:52:00.813877+00:00
Command: D:\Games\piper_sim\.venv\Scripts\python.exe -B -m piper_rl.scripts.evaluate_human_aware --model runs\human_aware_sac_v1\final_model.zip --algo sac --experiment randomized-id --episodes 50 --seed 21000 --out results\human_aware_sac_v1\checkpoint_validation_common_id\step_2000000
Estimate: 30 min; watchdog: 3600 seconds
Output log: results\human_aware_sac_v1\checkpoint_validation_common_id\logs\step_2000000_20261004T215200812877Z_1.log

End: 2026-10-04T21:53:16.475582+00:00
Exit code: 0

### Evaluation subprocess
Start: 2026-10-04T21:53:16.491618+00:00
Command: D:\Games\piper_sim\.venv\Scripts\python.exe -B -m piper_rl.scripts.evaluate_human_aware --model runs\human_aware_sac_v1\best\best_model.zip --algo sac --experiment randomized-id --episodes 50 --seed 21000 --out results\human_aware_sac_v1\checkpoint_validation_common_id\callback_best
Estimate: 30 min; watchdog: 3600 seconds
Output log: results\human_aware_sac_v1\checkpoint_validation_common_id\logs\callback_best_20261004T215316491618Z_1.log

End: 2026-10-04T21:54:29.960454+00:00
Exit code: 0

Selected human_aware_sac_v1: callback_best; SHA-256: 4124b81633ac7e4beebbc145be65d83c8411c50835957dc31e72ce4bd10187d7

human_aware_sac_v1: selection unchanged; existing held-out results stand

Candidate human_aware_random_full_s0 / step_100000: runs\human_aware_random_full_s0\checkpoints\piper_100000_steps.zip; exists=True

Candidate human_aware_random_full_s0 / step_200000: runs\human_aware_random_full_s0\checkpoints\piper_200000_steps.zip; exists=True

Candidate human_aware_random_full_s0 / step_400000: runs\human_aware_random_full_s0\checkpoints\piper_400000_steps.zip; exists=True

Candidate human_aware_random_full_s0 / step_600000: runs\human_aware_random_full_s0\checkpoints\piper_600000_steps.zip; exists=True

Candidate human_aware_random_full_s0 / step_1000000: runs\human_aware_random_full_s0\checkpoints\piper_1000000_steps.zip; exists=True

Candidate human_aware_random_full_s0 / step_1500000: runs\human_aware_random_full_s0\checkpoints\piper_1500000_steps.zip; exists=True

Candidate human_aware_random_full_s0 / step_2000000: runs\human_aware_random_full_s0\final_model.zip; exists=True

Candidate human_aware_random_full_s0 / callback_best: runs\human_aware_random_full_s0\best\best_model.zip; exists=True

### Evaluation subprocess
Start: 2026-10-04T21:54:30.211461+00:00
Command: D:\Games\piper_sim\.venv\Scripts\python.exe -B -m piper_rl.scripts.evaluate_human_aware --model runs\human_aware_random_full_s0\checkpoints\piper_100000_steps.zip --algo sac --experiment randomized-id --episodes 50 --seed 21000 --out results\human_aware_random_full_s0\checkpoint_validation_common_id\step_100000
Estimate: 30 min; watchdog: 3600 seconds
Output log: results\human_aware_random_full_s0\checkpoint_validation_common_id\logs\step_100000_20261004T215430211461Z_1.log

End: 2026-10-04T21:56:41.528296+00:00
Exit code: 0

### Evaluation subprocess
Start: 2026-10-04T21:56:41.545026+00:00
Command: D:\Games\piper_sim\.venv\Scripts\python.exe -B -m piper_rl.scripts.evaluate_human_aware --model runs\human_aware_random_full_s0\checkpoints\piper_200000_steps.zip --algo sac --experiment randomized-id --episodes 50 --seed 21000 --out results\human_aware_random_full_s0\checkpoint_validation_common_id\step_200000
Estimate: 30 min; watchdog: 3600 seconds
Output log: results\human_aware_random_full_s0\checkpoint_validation_common_id\logs\step_200000_20261004T215641545026Z_1.log

End: 2026-10-04T21:58:58.347625+00:00
Exit code: 0

### Evaluation subprocess
Start: 2026-10-04T21:58:58.363621+00:00
Command: D:\Games\piper_sim\.venv\Scripts\python.exe -B -m piper_rl.scripts.evaluate_human_aware --model runs\human_aware_random_full_s0\checkpoints\piper_400000_steps.zip --algo sac --experiment randomized-id --episodes 50 --seed 21000 --out results\human_aware_random_full_s0\checkpoint_validation_common_id\step_400000
Estimate: 30 min; watchdog: 3600 seconds
Output log: results\human_aware_random_full_s0\checkpoint_validation_common_id\logs\step_400000_20261004T215858363621Z_1.log

End: 2026-10-04T22:01:14.363932+00:00
Exit code: 0

### Evaluation subprocess
Start: 2026-10-04T22:01:14.379931+00:00
Command: D:\Games\piper_sim\.venv\Scripts\python.exe -B -m piper_rl.scripts.evaluate_human_aware --model runs\human_aware_random_full_s0\checkpoints\piper_600000_steps.zip --algo sac --experiment randomized-id --episodes 50 --seed 21000 --out results\human_aware_random_full_s0\checkpoint_validation_common_id\step_600000
Estimate: 30 min; watchdog: 3600 seconds
Output log: results\human_aware_random_full_s0\checkpoint_validation_common_id\logs\step_600000_20261004T220114379931Z_1.log

End: 2026-10-04T22:03:38.598015+00:00
Exit code: 0

### Evaluation subprocess
Start: 2026-10-04T22:03:38.615007+00:00
Command: D:\Games\piper_sim\.venv\Scripts\python.exe -B -m piper_rl.scripts.evaluate_human_aware --model runs\human_aware_random_full_s0\checkpoints\piper_1000000_steps.zip --algo sac --experiment randomized-id --episodes 50 --seed 21000 --out results\human_aware_random_full_s0\checkpoint_validation_common_id\step_1000000
Estimate: 30 min; watchdog: 3600 seconds
Output log: results\human_aware_random_full_s0\checkpoint_validation_common_id\logs\step_1000000_20261004T220338615007Z_1.log

End: 2026-10-04T22:05:46.160249+00:00
Exit code: 0

### Evaluation subprocess
Start: 2026-10-04T22:05:46.176247+00:00
Command: D:\Games\piper_sim\.venv\Scripts\python.exe -B -m piper_rl.scripts.evaluate_human_aware --model runs\human_aware_random_full_s0\checkpoints\piper_1500000_steps.zip --algo sac --experiment randomized-id --episodes 50 --seed 21000 --out results\human_aware_random_full_s0\checkpoint_validation_common_id\step_1500000
Estimate: 30 min; watchdog: 3600 seconds
Output log: results\human_aware_random_full_s0\checkpoint_validation_common_id\logs\step_1500000_20261004T220546176247Z_1.log

End: 2026-10-04T22:06:57.162091+00:00
Exit code: 0

### Evaluation subprocess
Start: 2026-10-04T22:06:57.179092+00:00
Command: D:\Games\piper_sim\.venv\Scripts\python.exe -B -m piper_rl.scripts.evaluate_human_aware --model runs\human_aware_random_full_s0\final_model.zip --algo sac --experiment randomized-id --episodes 50 --seed 21000 --out results\human_aware_random_full_s0\checkpoint_validation_common_id\step_2000000
Estimate: 30 min; watchdog: 3600 seconds
Output log: results\human_aware_random_full_s0\checkpoint_validation_common_id\logs\step_2000000_20261004T220657179092Z_1.log

End: 2026-10-04T22:08:01.277358+00:00
Exit code: 0

### Evaluation subprocess
Start: 2026-10-04T22:08:01.293348+00:00
Command: D:\Games\piper_sim\.venv\Scripts\python.exe -B -m piper_rl.scripts.evaluate_human_aware --model runs\human_aware_random_full_s0\best\best_model.zip --algo sac --experiment randomized-id --episodes 50 --seed 21000 --out results\human_aware_random_full_s0\checkpoint_validation_common_id\callback_best
Estimate: 30 min; watchdog: 3600 seconds
Output log: results\human_aware_random_full_s0\checkpoint_validation_common_id\logs\callback_best_20261004T220801293348Z_1.log

End: 2026-10-04T22:09:22.607570+00:00
Exit code: 0

Selected human_aware_random_full_s0: step_2000000; SHA-256: 5ed60cb286b07c88abd214942ee153d789ef9a70794cd8a18f6eb9319cba4133

### Evaluation subprocess
Start: 2026-10-04T22:09:22.832420+00:00
Command: D:\Games\piper_sim\.venv\Scripts\python.exe -B -m piper_rl.scripts.evaluate_human_aware --model runs\human_aware_random_full_s0\selected_common_id_model.zip --algo sac --experiment no-human --episodes 200 --seed 30000 --out results\human_aware_random_full_s0\heldout_common_id\no_human
Estimate: 30 min; watchdog: 3600 seconds
Output log: results\human_aware_random_full_s0\heldout_common_id\logs\no_human_20261004T220922832420Z_1.log

### Command
Start: 2026-10-05T03:09:52.4382609+05:00
Command: Record both seed-0 common-ID selection outcomes

Seed-0 selection outcome: {"run": "human_aware_sac_v1", "selected": "callback_best", "sha256": "4124b81633ac7e4beebbc145be65d83c8411c50835957dc31e72ce4bd10187d7", "episodes": 50, "task_success_rate": 0.82, "human_collision_rate": 0.04}

Seed-0 selection outcome: {"run": "human_aware_random_full_s0", "selected": "step_2000000", "sha256": "5ed60cb286b07c88abd214942ee153d789ef9a70794cd8a18f6eb9319cba4133", "episodes": 50, "task_success_rate": 0.88, "human_collision_rate": 0.1}
End: 2026-10-05T03:09:52.9578897+05:00
Exit code: 0

End: 2026-10-04T22:13:29.970275+00:00
Exit code: 0

### Evaluation subprocess
Start: 2026-10-04T22:13:29.971276+00:00
Command: D:\Games\piper_sim\.venv\Scripts\python.exe -B -m piper_rl.scripts.evaluate_human_aware --model runs\human_aware_random_full_s0\selected_common_id_model.zip --algo sac --experiment fixed-exact-h036-v1 --episodes 200 --seed 31000 --out results\human_aware_random_full_s0\heldout_common_id\exact_h036
Estimate: 30 min; watchdog: 3600 seconds
Output log: results\human_aware_random_full_s0\heldout_common_id\logs\exact_h036_20261004T221329971276Z_1.log

End: 2026-10-04T22:17:34.873982+00:00
Exit code: 0

### Evaluation subprocess
Start: 2026-10-04T22:17:34.874967+00:00
Command: D:\Games\piper_sim\.venv\Scripts\python.exe -B -m piper_rl.scripts.evaluate_human_aware --model runs\human_aware_random_full_s0\selected_common_id_model.zip --algo sac --experiment fixed --episodes 200 --seed 31000 --out results\human_aware_random_full_s0\heldout_common_id\constrained_fixed_height_randomized
Estimate: 30 min; watchdog: 3600 seconds
Output log: results\human_aware_random_full_s0\heldout_common_id\logs\constrained_fixed_height_randomized_20261004T221734874967Z_1.log

End: 2026-10-04T22:21:32.687942+00:00
Exit code: 0

### Evaluation subprocess
Start: 2026-10-04T22:21:32.688942+00:00
Command: D:\Games\piper_sim\.venv\Scripts\python.exe -B -m piper_rl.scripts.evaluate_human_aware --model runs\human_aware_random_full_s0\selected_common_id_model.zip --algo sac --experiment fixed-exact-shifted-h036-v1 --episodes 200 --seed 32000 --out results\human_aware_random_full_s0\heldout_common_id\shifted_exact_h036
Estimate: 30 min; watchdog: 3600 seconds
Output log: results\human_aware_random_full_s0\heldout_common_id\logs\shifted_exact_h036_20261004T222132688942Z_1.log

End: 2026-10-04T22:25:11.405238+00:00
Exit code: 0

### Evaluation subprocess
Start: 2026-10-04T22:25:11.406232+00:00
Command: D:\Games\piper_sim\.venv\Scripts\python.exe -B -m piper_rl.scripts.evaluate_human_aware --model runs\human_aware_random_full_s0\selected_common_id_model.zip --algo sac --experiment randomized-id --episodes 500 --seed 40000 --out results\human_aware_random_full_s0\heldout_common_id\randomized_id
Estimate: 30 min; watchdog: 3600 seconds
Output log: results\human_aware_random_full_s0\heldout_common_id\logs\randomized_id_20261004T222511406232Z_1.log

### Command
Start: 2026-10-05T03:32:31.4336177+05:00
Command: Prepare allowlisted commit, push, and GitHub PR publication helper
End: 2026-10-05T03:32:31.5096176+05:00
Exit code: 0

End: 2026-10-04T22:35:36.713257+00:00
Exit code: 0

### Evaluation subprocess
Start: 2026-10-04T22:35:36.714261+00:00
Command: D:\Games\piper_sim\.venv\Scripts\python.exe -B -m piper_rl.scripts.evaluate_human_aware --model runs\human_aware_random_full_s0\selected_common_id_model.zip --algo sac --experiment ood --episodes 500 --seed 50000 --out results\human_aware_random_full_s0\heldout_common_id\ood
Estimate: 30 min; watchdog: 3600 seconds
Output log: results\human_aware_random_full_s0\heldout_common_id\logs\ood_20261004T223536714261Z_1.log

End: 2026-10-04T22:46:55.321596+00:00
Exit code: 0
End: 2026-10-05T03:46:56.1276175+05:00
Exit code: 0

### Command
Start: 2026-10-05T04:03:00.3359183+05:00
Command: Install evaluation-only human-observation flag and regression tests
End: 2026-10-05T04:03:00.7269464+05:00
Exit code: 0

### Command
Start: 2026-10-05T04:03:00.7369387+05:00
Command: .venv/Scripts/python.exe -B -m pytest tests/test_human_cue_probe.py -q
End: 2026-10-05T04:03:21.1685341+05:00
Exit code: 0

### Command
Start: 2026-10-05T04:03:25.8392239+05:00
Command: Review evaluation flag diff and verify Step-2 held-out counts and model hashes
End: 2026-10-05T04:03:31.8648830+05:00
Exit code: 0

### Command
Start: 2026-10-05T04:03:57.0687986+05:00
Command: Record new tests and tidy the no-human observation comment

New test file: tests/test_human_cue_probe.py ? 9 passed in 14.12s. Default actual observations/outcomes byte-identical to original env path; overrides preserve physical state and real RNG, and phantom time/TCP/trajectory reproducibility checked.
Step 2 complete: both eight-candidate manifests and all model SHA-256 hashes verified; full-random six held-out episode counts verified.
End: 2026-10-05T04:03:57.4505241+05:00
Exit code: 0

### Command
Start: 2026-10-05T04:03:57.4605204+05:00
Command: Step 3: sequential actual determinism gate, then 10 prescribed human-cue probes (.venv/Scripts/python.exe -B temporary guarded launcher)

### Evaluation subprocess
Start: 2026-10-04T23:04:04.201973+00:00
Command: D:\Games\piper_sim\.venv\Scripts\python.exe -B -m piper_rl.scripts.evaluate_human_aware --model runs\human_aware_fixed_s0\selected_common_id_model.zip --algo sac --experiment no-human --episodes 20 --seed 30000 --out results\final_eval_2026_10\determinism\fixed_s0_actual_no_human --human-obs-source actual
Estimate: 30 min; watchdog: 3600 seconds
Output log: results\final_eval_2026_10\determinism\logs\fixed_s0_actual_no_human_20261004T230404200974Z_1.log

End: 2026-10-04T23:04:55.886903+00:00
Exit code: 0

### Determinism gate
Start: 2026-10-04T23:04:55.887891+00:00
Command: D:\Games\piper_sim\.venv\Scripts\python.exe -B "C:\Users\Rent Market\AppData\Local\Temp\piper_probe_determinism_compare.py"

Step 3 default-actual determinism comparison: {"run": "human_aware_fixed_s0", "episodes": 20, "compared_fields": ["experiment", "distribution", "episode", "seed", "trajectory_type", "task_success", "collision_free_success", "human_collision", "near_miss", "near_miss_events", "minimum_separation", "completion_time", "episode_duration", "episode_length", "placement_error", "grasp_success", "lift_success", "safety_interventions", "safety_clamped_steps", "safety_vetoed_steps", "safety_clamp_rate", "proximity_steps", "waiting_steps", "waiting_time", "human_safety_cost", "reward", "termination"], "reference": "results\\human_aware_fixed_s0\\heldout\\no_human.csv", "actual": "results\\final_eval_2026_10\\determinism\\fixed_s0_actual_no_human.csv", "identical": true, "differences": []}

End: 2026-10-04T23:04:58.013970+00:00
Exit code: 0

### Evaluation subprocess
Start: 2026-10-04T23:04:58.014958+00:00
Command: D:\Games\piper_sim\.venv\Scripts\python.exe -B -m piper_rl.scripts.evaluate_human_aware --model runs\human_aware_fixed_s0\selected_common_id_model.zip --algo sac --experiment no-human --episodes 200 --seed 30000 --out results\human_aware_probe_human_cue\human_aware_fixed_s0\P1 --human-obs-source phantom-exact-h036
Estimate: 30 min; watchdog: 3600 seconds
Output log: results\human_aware_probe_human_cue\human_aware_fixed_s0\logs\P1_20261004T230458014958Z_1.log

End: 2026-10-04T23:08:37.366994+00:00
Exit code: 0

Completed probe human_aware_fixed_s0/P1; model SHA-256: 888b0c6c22ce6add8f1eef870c6dd8932cb389d33ca82a58b85d4501db0161c7; summary: {"experiment": "no-human", "distribution": "no_human", "episodes": 200, "deterministic_prediction": true, "task_success_rate": 0.94, "task_success_95ci": [0.8980673646931491, 0.9653481500987285], "collision_free_success_rate": 0.94, "human_collision_rate": 0.0, "human_collision_95ci": [0.0, 0.018846005918320894], "near_miss_rate": 0.0, "mean_minimum_separation": 0.3483694766717502, "median_minimum_separation": 0.3456440720271971, "p05_minimum_separation": 0.24682881407850216, "worst_minimum_separation": 0.20702120913925567, "mean_completion_time_successes": 2.822074468085106, "mean_episode_duration": 3.2015, "mean_episode_length": 64.03, "mean_placement_error": 0.03660814870554435, "median_placement_error": 0.022485806857254854, "p90_placement_error": 0.038429681319331804, "grasp_rate": 0.975, "lift_rate": 0.925, "mean_safety_interventions": 55.7, "mean_safety_clamped_steps": 55.7, "mean_safety_clamp_rate": 0.8605087023857746, "mean_proximity_steps": 0.0, "mean_waiting_steps": 0.0, "mean_waiting_time": 0.0, "mean_episode_reward": 267.82102233321586, "mean_human_safety_cost": 0.0}

### Evaluation subprocess
Start: 2026-10-04T23:08:37.383987+00:00
Command: D:\Games\piper_sim\.venv\Scripts\python.exe -B -m piper_rl.scripts.evaluate_human_aware --model runs\human_aware_fixed_s0\selected_common_id_model.zip --algo sac --experiment fixed-exact-h036-v1 --episodes 200 --seed 31000 --out results\human_aware_probe_human_cue\human_aware_fixed_s0\P2 --human-obs-source frozen-parked
Estimate: 30 min; watchdog: 3600 seconds
Output log: results\human_aware_probe_human_cue\human_aware_fixed_s0\logs\P2_20261004T230837382985Z_1.log

### Command
Start: 2026-10-05T04:09:25.1273173+05:00
Command: Record historical determinism pass and first completed cue-probe metrics

Step 3 historical determinism gate passed: all 27 CSV fields exactly equal the first 20 committed no-human rows for fixed seed 0.
End: 2026-10-05T04:09:25.5498270+05:00
Exit code: 0

End: 2026-10-04T23:16:47.324569+00:00
Exit code: 0

Completed probe human_aware_fixed_s0/P2; model SHA-256: 888b0c6c22ce6add8f1eef870c6dd8932cb389d33ca82a58b85d4501db0161c7; summary: {"experiment": "fixed-exact-h036-v1", "distribution": "fixed_exact_h036_v1", "episodes": 200, "deterministic_prediction": true, "task_success_rate": 0.17, "task_success_95ci": [0.12427835351105657, 0.22816001039503528], "collision_free_success_rate": 0.17, "human_collision_rate": 0.81, "human_collision_95ci": [0.7499864142552062, 0.8583290620754351], "near_miss_rate": 0.87, "mean_minimum_separation": 0.02004045673806287, "median_minimum_separation": -0.00010709875282950004, "p05_minimum_separation": -0.0003853338596031229, "worst_minimum_separation": -0.07037386591029972, "mean_completion_time_successes": 3.086764705882353, "mean_episode_duration": 5.93225, "mean_episode_length": 118.645, "mean_placement_error": 0.23192451990136317, "median_placement_error": 0.26166545614203474, "p90_placement_error": 0.39618208409597266, "grasp_rate": 0.365, "lift_rate": 0.25, "mean_safety_interventions": 108.365, "mean_safety_clamped_steps": 108.365, "mean_safety_clamp_rate": 0.9026511483857204, "mean_proximity_steps": 35.7, "mean_waiting_steps": 33.81, "mean_waiting_time": 1.6905000000000001, "mean_episode_reward": -49.125695402092795, "mean_human_safety_cost": 103.51878393157351}

### Evaluation subprocess
Start: 2026-10-04T23:16:47.341569+00:00
Command: D:\Games\piper_sim\.venv\Scripts\python.exe -B -m piper_rl.scripts.evaluate_human_aware --model runs\human_aware_fixed_s1\selected_common_id_model.zip --algo sac --experiment no-human --episodes 200 --seed 30000 --out results\human_aware_probe_human_cue\human_aware_fixed_s1\P1 --human-obs-source phantom-exact-h036
Estimate: 30 min; watchdog: 3600 seconds
Output log: results\human_aware_probe_human_cue\human_aware_fixed_s1\logs\P1_20261004T231647341569Z_1.log

End: 2026-10-04T23:20:15.612383+00:00
Exit code: 0

Completed probe human_aware_fixed_s1/P1; model SHA-256: a58158ef4ddfe91db948b9bdae59e88f01189c5ce95f800a2cca3626bec31d1b; summary: {"experiment": "no-human", "distribution": "no_human", "episodes": 200, "deterministic_prediction": true, "task_success_rate": 0.97, "task_success_95ci": [0.9361048802675267, 0.9861798741692518], "collision_free_success_rate": 0.97, "human_collision_rate": 0.0, "human_collision_95ci": [0.0, 0.018846005918320894], "near_miss_rate": 0.0, "mean_minimum_separation": 0.3527914534258039, "median_minimum_separation": 0.35606446573612777, "p05_minimum_separation": 0.2557576364511494, "worst_minimum_separation": 0.21119282321924432, "mean_completion_time_successes": 2.718814432989691, "mean_episode_duration": 2.9372499999999997, "mean_episode_length": 58.745, "mean_placement_error": 0.02792950831570422, "median_placement_error": 0.022786119026081034, "p90_placement_error": 0.03501969603848329, "grasp_rate": 0.995, "lift_rate": 0.965, "mean_safety_interventions": 39.985, "mean_safety_clamped_steps": 39.985, "mean_safety_clamp_rate": 0.6622205516308497, "mean_proximity_steps": 0.0, "mean_waiting_steps": 0.0, "mean_waiting_time": 0.0, "mean_episode_reward": 276.84782409427515, "mean_human_safety_cost": 0.0}

### Evaluation subprocess
Start: 2026-10-04T23:20:15.629267+00:00
Command: D:\Games\piper_sim\.venv\Scripts\python.exe -B -m piper_rl.scripts.evaluate_human_aware --model runs\human_aware_fixed_s1\selected_common_id_model.zip --algo sac --experiment fixed-exact-h036-v1 --episodes 200 --seed 31000 --out results\human_aware_probe_human_cue\human_aware_fixed_s1\P2 --human-obs-source frozen-parked
Estimate: 30 min; watchdog: 3600 seconds
Output log: results\human_aware_probe_human_cue\human_aware_fixed_s1\logs\P2_20261004T232015629267Z_1.log

End: 2026-10-04T23:25:03.023271+00:00
Exit code: 0

Completed probe human_aware_fixed_s1/P2; model SHA-256: a58158ef4ddfe91db948b9bdae59e88f01189c5ce95f800a2cca3626bec31d1b; summary: {"experiment": "fixed-exact-h036-v1", "distribution": "fixed_exact_h036_v1", "episodes": 200, "deterministic_prediction": true, "task_success_rate": 0.685, "task_success_95ci": [0.617649158589664, 0.7453778192205573], "collision_free_success_rate": 0.68, "human_collision_rate": 0.31, "human_collision_95ci": [0.24998842733633997, 0.377173054912622], "near_miss_rate": 0.335, "mean_minimum_separation": 0.11732577341177325, "median_minimum_separation": 0.13472813698238634, "p05_minimum_separation": -0.00021990705513329212, "worst_minimum_separation": -0.004710880595404605, "mean_completion_time_successes": 2.8536496350364966, "mean_episode_duration": 3.7597500000000004, "mean_episode_length": 75.195, "mean_placement_error": 0.08801832627593549, "median_placement_error": 0.03520002601929244, "p90_placement_error": 0.2913567166946862, "grasp_rate": 0.835, "lift_rate": 0.37, "mean_safety_interventions": 57.67, "mean_safety_clamped_steps": 57.67, "mean_safety_clamp_rate": 0.7362405181177888, "mean_proximity_steps": 11.51, "mean_waiting_steps": 10.285, "mean_waiting_time": 0.51425, "mean_episode_reward": 141.47814245528974, "mean_human_safety_cost": 39.09055383491796}

### Evaluation subprocess
Start: 2026-10-04T23:25:03.040257+00:00
Command: D:\Games\piper_sim\.venv\Scripts\python.exe -B -m piper_rl.scripts.evaluate_human_aware --model runs\human_aware_fixed_s2\selected_common_id_model.zip --algo sac --experiment no-human --episodes 200 --seed 30000 --out results\human_aware_probe_human_cue\human_aware_fixed_s2\P1 --human-obs-source phantom-exact-h036
Estimate: 30 min; watchdog: 3600 seconds
Output log: results\human_aware_probe_human_cue\human_aware_fixed_s2\logs\P1_20261004T232503040257Z_1.log

End: 2026-10-04T23:28:29.203833+00:00
Exit code: 0

Completed probe human_aware_fixed_s2/P1; model SHA-256: b97b391147fcb118b084451131121ed18023eb4cd434167da54b8ff3582022ed; summary: {"experiment": "no-human", "distribution": "no_human", "episodes": 200, "deterministic_prediction": true, "task_success_rate": 0.97, "task_success_95ci": [0.9361048802675267, 0.9861798741692518], "collision_free_success_rate": 0.97, "human_collision_rate": 0.0, "human_collision_95ci": [0.0, 0.018846005918320894], "near_miss_rate": 0.0, "mean_minimum_separation": 0.33722453850369066, "median_minimum_separation": 0.33510294993659, "p05_minimum_separation": 0.24339539372974986, "worst_minimum_separation": 0.20888831592543738, "mean_completion_time_successes": 2.7268041237113403, "mean_episode_duration": 2.945, "mean_episode_length": 58.9, "mean_placement_error": 0.022954660574863835, "median_placement_error": 0.013538278702815423, "p90_placement_error": 0.029326431233586782, "grasp_rate": 1.0, "lift_rate": 0.965, "mean_safety_interventions": 42.705, "mean_safety_clamped_steps": 42.705, "mean_safety_clamp_rate": 0.7097100348093213, "mean_proximity_steps": 0.0, "mean_waiting_steps": 0.0, "mean_waiting_time": 0.0, "mean_episode_reward": 275.3734688297289, "mean_human_safety_cost": 0.0}

### Evaluation subprocess
Start: 2026-10-04T23:28:29.219827+00:00
Command: D:\Games\piper_sim\.venv\Scripts\python.exe -B -m piper_rl.scripts.evaluate_human_aware --model runs\human_aware_fixed_s2\selected_common_id_model.zip --algo sac --experiment fixed-exact-h036-v1 --episodes 200 --seed 31000 --out results\human_aware_probe_human_cue\human_aware_fixed_s2\P2 --human-obs-source frozen-parked
Estimate: 30 min; watchdog: 3600 seconds
Output log: results\human_aware_probe_human_cue\human_aware_fixed_s2\logs\P2_20261004T232829219827Z_1.log

End: 2026-10-04T23:35:14.043090+00:00
Exit code: 0

Completed probe human_aware_fixed_s2/P2; model SHA-256: b97b391147fcb118b084451131121ed18023eb4cd434167da54b8ff3582022ed; summary: {"experiment": "fixed-exact-h036-v1", "distribution": "fixed_exact_h036_v1", "episodes": 200, "deterministic_prediction": true, "task_success_rate": 0.335, "task_success_95ci": [0.2732398096874265, 0.4029793722656195], "collision_free_success_rate": 0.335, "human_collision_rate": 0.66, "human_collision_95ci": [0.5918836703220929, 0.7220856077840445], "near_miss_rate": 0.705, "mean_minimum_separation": 0.046070612332539077, "median_minimum_separation": -5.929172983169806e-05, "p05_minimum_separation": -0.0002621713866102035, "worst_minimum_separation": -0.00046971686918074324, "mean_completion_time_successes": 2.9604477611940303, "mean_episode_duration": 5.3115, "mean_episode_length": 106.23, "mean_placement_error": 0.19325895968469728, "median_placement_error": 0.22936999539191089, "p90_placement_error": 0.3894856045857418, "grasp_rate": 0.465, "lift_rate": 0.38, "mean_safety_interventions": 87.585, "mean_safety_clamped_steps": 87.585, "mean_safety_clamp_rate": 0.7878799016073704, "mean_proximity_steps": 28.66, "mean_waiting_steps": 27.165, "mean_waiting_time": 1.3582500000000002, "mean_episode_reward": 20.27870597007062, "mean_human_safety_cost": 82.23117928108572}

### Evaluation subprocess
Start: 2026-10-04T23:35:14.060075+00:00
Command: D:\Games\piper_sim\.venv\Scripts\python.exe -B -m piper_rl.scripts.evaluate_human_aware --model runs\human_aware_curriculum_s1\selected_common_id_model.zip --algo sac --experiment no-human --episodes 200 --seed 30000 --out results\human_aware_probe_human_cue\human_aware_curriculum_s1\P1 --human-obs-source phantom-exact-h036
Estimate: 30 min; watchdog: 3600 seconds
Output log: results\human_aware_probe_human_cue\human_aware_curriculum_s1\logs\P1_20261004T233514060075Z_1.log

End: 2026-10-04T23:39:01.631584+00:00
Exit code: 0

Completed probe human_aware_curriculum_s1/P1; model SHA-256: ae3cb0ae4c31f4b191f5601245adfa99d308d60e4597349ca7376d15a539ff31; summary: {"experiment": "no-human", "distribution": "no_human", "episodes": 200, "deterministic_prediction": true, "task_success_rate": 0.915, "task_success_95ci": [0.8681031145576462, 0.9462547005301477], "collision_free_success_rate": 0.915, "human_collision_rate": 0.0, "human_collision_95ci": [0.0, 0.018846005918320894], "near_miss_rate": 0.0, "mean_minimum_separation": 0.3469450681176528, "median_minimum_separation": 0.34471213107488846, "p05_minimum_separation": 0.25603430885001666, "worst_minimum_separation": 0.21549369320303757, "mean_completion_time_successes": 2.6453551912568307, "mean_episode_duration": 3.214, "mean_episode_length": 64.28, "mean_placement_error": 0.040765360531902654, "median_placement_error": 0.01890961155438849, "p90_placement_error": 0.04256042237504393, "grasp_rate": 0.985, "lift_rate": 0.92, "mean_safety_interventions": 57.3, "mean_safety_clamped_steps": 57.3, "mean_safety_clamp_rate": 0.8810887136394889, "mean_proximity_steps": 0.0, "mean_waiting_steps": 0.0, "mean_waiting_time": 0.0, "mean_episode_reward": 255.570178020065, "mean_human_safety_cost": 0.0}

### Evaluation subprocess
Start: 2026-10-04T23:39:01.648561+00:00
Command: D:\Games\piper_sim\.venv\Scripts\python.exe -B -m piper_rl.scripts.evaluate_human_aware --model runs\human_aware_curriculum_s1\selected_common_id_model.zip --algo sac --experiment fixed-exact-h036-v1 --episodes 200 --seed 31000 --out results\human_aware_probe_human_cue\human_aware_curriculum_s1\P2 --human-obs-source frozen-parked
Estimate: 30 min; watchdog: 3600 seconds
Output log: results\human_aware_probe_human_cue\human_aware_curriculum_s1\logs\P2_20261004T233901647563Z_1.log

End: 2026-10-04T23:43:47.580505+00:00
Exit code: 0

Completed probe human_aware_curriculum_s1/P2; model SHA-256: ae3cb0ae4c31f4b191f5601245adfa99d308d60e4597349ca7376d15a539ff31; summary: {"experiment": "fixed-exact-h036-v1", "distribution": "fixed_exact_h036_v1", "episodes": 200, "deterministic_prediction": true, "task_success_rate": 0.835, "task_success_95ci": [0.7773410172613647, 0.8800321587733603], "collision_free_success_rate": 0.83, "human_collision_rate": 0.16, "human_collision_95ci": [0.11567342208544587, 0.2171418619390124], "near_miss_rate": 0.345, "mean_minimum_separation": 0.10905058544149795, "median_minimum_separation": 0.10195862675661102, "p05_minimum_separation": -0.00014809764854917078, "worst_minimum_separation": -0.0007546675280983781, "mean_completion_time_successes": 3.0230538922155694, "mean_episode_duration": 3.3114999999999997, "mean_episode_length": 66.23, "mean_placement_error": 0.033770624193499736, "median_placement_error": 0.017123792802088145, "p90_placement_error": 0.04141690565262907, "grasp_rate": 0.955, "lift_rate": 0.915, "mean_safety_interventions": 54.94, "mean_safety_clamped_steps": 54.94, "mean_safety_clamp_rate": 0.8222895746940233, "mean_proximity_steps": 7.03, "mean_waiting_steps": 4.695, "mean_waiting_time": 0.23475, "mean_episode_reward": 222.7159754017247, "mean_human_safety_cost": 20.667623616240316}

### Evaluation subprocess
Start: 2026-10-04T23:43:47.597510+00:00
Command: D:\Games\piper_sim\.venv\Scripts\python.exe -B -m piper_rl.scripts.evaluate_human_aware --model runs\human_aware_curriculum_s2\selected_common_id_model.zip --algo sac --experiment no-human --episodes 200 --seed 30000 --out results\human_aware_probe_human_cue\human_aware_curriculum_s2\P1 --human-obs-source phantom-exact-h036
Estimate: 30 min; watchdog: 3600 seconds
Output log: results\human_aware_probe_human_cue\human_aware_curriculum_s2\logs\P1_20261004T234347597510Z_1.log

End: 2026-10-04T23:47:46.009312+00:00
Exit code: 0

Completed probe human_aware_curriculum_s2/P1; model SHA-256: d43140cb5a9442351790ff083d85562c2a4972956052f59c942274a25eea4a3f; summary: {"experiment": "no-human", "distribution": "no_human", "episodes": 200, "deterministic_prediction": true, "task_success_rate": 0.9, "task_success_95ci": [0.8505931364369146, 0.9343300588284289], "collision_free_success_rate": 0.9, "human_collision_rate": 0.0, "human_collision_95ci": [0.0, 0.018846005918320894], "near_miss_rate": 0.0, "mean_minimum_separation": 0.3497669003752475, "median_minimum_separation": 0.35068076608366333, "p05_minimum_separation": 0.2464604930120794, "worst_minimum_separation": 0.20432820829836693, "mean_completion_time_successes": 2.679444444444444, "mean_episode_duration": 3.4114999999999998, "mean_episode_length": 68.23, "mean_placement_error": 0.04129182261105232, "median_placement_error": 0.0208292973018524, "p90_placement_error": 0.049994797036059115, "grasp_rate": 0.975, "lift_rate": 0.88, "mean_safety_interventions": 60.145, "mean_safety_clamped_steps": 60.145, "mean_safety_clamp_rate": 0.8599908705478265, "mean_proximity_steps": 0.0, "mean_waiting_steps": 0.0, "mean_waiting_time": 0.0, "mean_episode_reward": 252.64125053174376, "mean_human_safety_cost": 0.0}

### Evaluation subprocess
Start: 2026-10-04T23:47:46.026007+00:00
Command: D:\Games\piper_sim\.venv\Scripts\python.exe -B -m piper_rl.scripts.evaluate_human_aware --model runs\human_aware_curriculum_s2\selected_common_id_model.zip --algo sac --experiment fixed-exact-h036-v1 --episodes 200 --seed 31000 --out results\human_aware_probe_human_cue\human_aware_curriculum_s2\P2 --human-obs-source frozen-parked
Estimate: 30 min; watchdog: 3600 seconds
Output log: results\human_aware_probe_human_cue\human_aware_curriculum_s2\logs\P2_20261004T234746025006Z_1.log

### Command
Start: 2026-10-05T04:49:55.1813151+05:00
Command: Prepare final summary table formatting and selection-row clarification
End: 2026-10-05T04:49:55.6588430+05:00
Exit code: 0

End: 2026-10-04T23:52:18.611852+00:00
Exit code: 0

Completed probe human_aware_curriculum_s2/P2; model SHA-256: d43140cb5a9442351790ff083d85562c2a4972956052f59c942274a25eea4a3f; summary: {"experiment": "fixed-exact-h036-v1", "distribution": "fixed_exact_h036_v1", "episodes": 200, "deterministic_prediction": true, "task_success_rate": 0.835, "task_success_95ci": [0.7773410172613647, 0.8800321587733603], "collision_free_success_rate": 0.825, "human_collision_rate": 0.175, "human_collision_95ci": [0.12860441174608936, 0.23364549210081922], "near_miss_rate": 0.34, "mean_minimum_separation": 0.11623138725575838, "median_minimum_separation": 0.11945723050821527, "p05_minimum_separation": -0.0001971685153151329, "worst_minimum_separation": -0.00920532792362281, "mean_completion_time_successes": 3.0116766467065865, "mean_episode_duration": 3.31375, "mean_episode_length": 66.275, "mean_placement_error": 0.034332325185547984, "median_placement_error": 0.02268649579200705, "p90_placement_error": 0.04001270575163766, "grasp_rate": 0.985, "lift_rate": 0.92, "mean_safety_interventions": 52.75, "mean_safety_clamped_steps": 52.75, "mean_safety_clamp_rate": 0.7892526198465316, "mean_proximity_steps": 6.435, "mean_waiting_steps": 4.03, "mean_waiting_time": 0.20149999999999998, "mean_episode_reward": 223.95706176760206, "mean_human_safety_cost": 21.87115244712265}
End: 2026-10-05T04:52:19.3655355+05:00
Exit code: 0

### Command
Start: 2026-10-05T04:52:54.7031284+05:00
Command: Verify all probe protocols and CSV counts, then generate the final summary

Step 3 complete: all 10 jobs verified at n=200, requested seeds/experiments/sources, deterministic prediction, no videos, and correct selected model. All P1 collision and near-miss rates are zero. No policy was trained.
End: 2026-10-05T04:53:08.7729063+05:00
Exit code: 0

### Command
Start: 2026-10-05T04:53:08.7749048+05:00
Command: git diff --check and final changed-file scope inspection
End: 2026-10-05T04:53:08.9142345+05:00
Exit code: 0

### Command
Start: 2026-10-05T04:58:32.1517724+05:00
Command: Finalize summary display order and prepare the concrete PR description
End: 2026-10-05T04:58:40.9284636+05:00
Exit code: 0

### Command
Start: 2026-10-05T04:58:40.9304900+05:00
Command: Publish final evaluation: allowlisted git commit/push and open PR against main (no merge)

### Publication command
Start: 2026-10-04T23:58:43.107849+00:00
Command: git diff --cached --name-only -z

End: 2026-10-04T23:58:43.142863+00:00
Exit code: 0

### Publication command
Start: 2026-10-04T23:58:43.142863+00:00
Command: git diff --name-only -z

End: 2026-10-04T23:58:43.182851+00:00
Exit code: 0

### Publication command
Start: 2026-10-04T23:58:43.182851+00:00
Command: git ls-files --others --exclude-standard -z

End: 2026-10-04T23:58:43.225853+00:00
Exit code: 0

### Publication command
Start: 2026-10-04T23:58:43.226852+00:00
Command: git add -- piper_rl/scripts/evaluate_human_aware.py piper_rl/scripts/seed0_common_selection.py results/final_eval_2026_10/RUN_LOG.md results/final_eval_2026_10/SUMMARY.md results/final_eval_2026_10/UNEXPECTED.md results/final_eval_2026_10/determinism/comparison.json results/final_eval_2026_10/determinism/fixed_s0_actual_no_human.csv results/final_eval_2026_10/determinism/fixed_s0_actual_no_human.json results/final_eval_2026_10/determinism/logs/fixed_s0_actual_no_human_20261004T230404200974Z_1.log results/human_aware_curriculum_hold_safety_s8/TRAINING_SUMMARY.md results/human_aware_curriculum_hold_safety_s8/campaign_state.json results/human_aware_curriculum_hold_safety_s8/evaluations.npz results/human_aware_curriculum_hold_safety_s8/heldout/constrained_fixed_height_randomized.csv results/human_aware_curriculum_hold_safety_s8/heldout/constrained_fixed_height_randomized.json results/human_aware_curriculum_hold_safety_s8/heldout/exact_h036.csv results/human_aware_curriculum_hold_safety_s8/heldout/exact_h036.json results/human_aware_curriculum_hold_safety_s8/heldout/ood.csv results/human_aware_curriculum_hold_safety_s8/heldout/ood.json results/human_aware_curriculum_hold_safety_s8/heldout/randomized_id.csv results/human_aware_curriculum_hold_safety_s8/heldout/randomized_id.json results/human_aware_curriculum_hold_safety_s8/heldout/shifted_exact_h036.csv results/human_aware_curriculum_hold_safety_s8/heldout/shifted_exact_h036.json results/human_aware_curriculum_hold_safety_s8/logs/heldout_constrained_fixed_height_randomized.log results/human_aware_curriculum_hold_safety_s8/logs/heldout_exact_h036.log results/human_aware_curriculum_hold_safety_s8/logs/heldout_ood.log results/human_aware_curriculum_hold_safety_s8/logs/heldout_randomized_id.log results/human_aware_curriculum_hold_safety_s8/logs/heldout_shifted_exact_h036.log results/human_aware_curriculum_hold_safety_s8/model_hashes.json results/human_aware_curriculum_hold_safety_s8/training_summary.json results/human_aware_probe_human_cue/human_aware_curriculum_s1/P1.csv

End: 2026-10-04T23:58:44.472432+00:00
Exit code: 0

### Publication command
Start: 2026-10-04T23:58:44.472432+00:00
Command: git add -- results/human_aware_probe_human_cue/human_aware_curriculum_s1/P1.json results/human_aware_probe_human_cue/human_aware_curriculum_s1/P2.csv results/human_aware_probe_human_cue/human_aware_curriculum_s1/P2.json results/human_aware_probe_human_cue/human_aware_curriculum_s1/logs/P1_20261004T233514060075Z_1.log results/human_aware_probe_human_cue/human_aware_curriculum_s1/logs/P2_20261004T233901647563Z_1.log results/human_aware_probe_human_cue/human_aware_curriculum_s2/P1.csv results/human_aware_probe_human_cue/human_aware_curriculum_s2/P1.json results/human_aware_probe_human_cue/human_aware_curriculum_s2/P2.csv results/human_aware_probe_human_cue/human_aware_curriculum_s2/P2.json results/human_aware_probe_human_cue/human_aware_curriculum_s2/logs/P1_20261004T234347597510Z_1.log results/human_aware_probe_human_cue/human_aware_curriculum_s2/logs/P2_20261004T234746025006Z_1.log results/human_aware_probe_human_cue/human_aware_fixed_s0/P1.csv results/human_aware_probe_human_cue/human_aware_fixed_s0/P1.json results/human_aware_probe_human_cue/human_aware_fixed_s0/P2.csv results/human_aware_probe_human_cue/human_aware_fixed_s0/P2.json results/human_aware_probe_human_cue/human_aware_fixed_s0/logs/P1_20261004T230458014958Z_1.log results/human_aware_probe_human_cue/human_aware_fixed_s0/logs/P2_20261004T230837382985Z_1.log results/human_aware_probe_human_cue/human_aware_fixed_s1/P1.csv results/human_aware_probe_human_cue/human_aware_fixed_s1/P1.json results/human_aware_probe_human_cue/human_aware_fixed_s1/P2.csv results/human_aware_probe_human_cue/human_aware_fixed_s1/P2.json results/human_aware_probe_human_cue/human_aware_fixed_s1/logs/P1_20261004T231647341569Z_1.log results/human_aware_probe_human_cue/human_aware_fixed_s1/logs/P2_20261004T232015629267Z_1.log results/human_aware_probe_human_cue/human_aware_fixed_s2/P1.csv results/human_aware_probe_human_cue/human_aware_fixed_s2/P1.json results/human_aware_probe_human_cue/human_aware_fixed_s2/P2.csv results/human_aware_probe_human_cue/human_aware_fixed_s2/P2.json results/human_aware_probe_human_cue/human_aware_fixed_s2/logs/P1_20261004T232503040257Z_1.log results/human_aware_probe_human_cue/human_aware_fixed_s2/logs/P2_20261004T232829219827Z_1.log results/human_aware_random_full_s0/checkpoint_validation_common_id.json

End: 2026-10-04T23:58:45.717984+00:00
Exit code: 0

### Publication command
Start: 2026-10-04T23:58:45.717984+00:00
Command: git add -- results/human_aware_random_full_s0/checkpoint_validation_common_id/callback_best.csv results/human_aware_random_full_s0/checkpoint_validation_common_id/callback_best.json results/human_aware_random_full_s0/checkpoint_validation_common_id/logs/callback_best_20261004T220801293348Z_1.log results/human_aware_random_full_s0/checkpoint_validation_common_id/logs/step_1000000_20261004T220338615007Z_1.log results/human_aware_random_full_s0/checkpoint_validation_common_id/logs/step_100000_20261004T215430211461Z_1.log results/human_aware_random_full_s0/checkpoint_validation_common_id/logs/step_1500000_20261004T220546176247Z_1.log results/human_aware_random_full_s0/checkpoint_validation_common_id/logs/step_2000000_20261004T220657179092Z_1.log results/human_aware_random_full_s0/checkpoint_validation_common_id/logs/step_200000_20261004T215641545026Z_1.log results/human_aware_random_full_s0/checkpoint_validation_common_id/logs/step_400000_20261004T215858363621Z_1.log results/human_aware_random_full_s0/checkpoint_validation_common_id/logs/step_600000_20261004T220114379931Z_1.log results/human_aware_random_full_s0/checkpoint_validation_common_id/step_100000.csv results/human_aware_random_full_s0/checkpoint_validation_common_id/step_100000.json results/human_aware_random_full_s0/checkpoint_validation_common_id/step_1000000.csv results/human_aware_random_full_s0/checkpoint_validation_common_id/step_1000000.json results/human_aware_random_full_s0/checkpoint_validation_common_id/step_1500000.csv results/human_aware_random_full_s0/checkpoint_validation_common_id/step_1500000.json results/human_aware_random_full_s0/checkpoint_validation_common_id/step_200000.csv results/human_aware_random_full_s0/checkpoint_validation_common_id/step_200000.json results/human_aware_random_full_s0/checkpoint_validation_common_id/step_2000000.csv results/human_aware_random_full_s0/checkpoint_validation_common_id/step_2000000.json results/human_aware_random_full_s0/checkpoint_validation_common_id/step_400000.csv results/human_aware_random_full_s0/checkpoint_validation_common_id/step_400000.json results/human_aware_random_full_s0/checkpoint_validation_common_id/step_600000.csv results/human_aware_random_full_s0/checkpoint_validation_common_id/step_600000.json results/human_aware_random_full_s0/heldout_common_id/constrained_fixed_height_randomized.csv results/human_aware_random_full_s0/heldout_common_id/constrained_fixed_height_randomized.json results/human_aware_random_full_s0/heldout_common_id/exact_h036.csv results/human_aware_random_full_s0/heldout_common_id/exact_h036.json results/human_aware_random_full_s0/heldout_common_id/logs/constrained_fixed_height_randomized_20261004T221734874967Z_1.log results/human_aware_random_full_s0/heldout_common_id/logs/exact_h036_20261004T221329971276Z_1.log

End: 2026-10-04T23:58:46.721517+00:00
Exit code: 0

### Publication command
Start: 2026-10-04T23:58:46.721517+00:00
Command: git add -- results/human_aware_random_full_s0/heldout_common_id/logs/no_human_20261004T220922832420Z_1.log results/human_aware_random_full_s0/heldout_common_id/logs/ood_20261004T223536714261Z_1.log results/human_aware_random_full_s0/heldout_common_id/logs/randomized_id_20261004T222511406232Z_1.log results/human_aware_random_full_s0/heldout_common_id/logs/shifted_exact_h036_20261004T222132688942Z_1.log results/human_aware_random_full_s0/heldout_common_id/no_human.csv results/human_aware_random_full_s0/heldout_common_id/no_human.json results/human_aware_random_full_s0/heldout_common_id/ood.csv results/human_aware_random_full_s0/heldout_common_id/ood.json results/human_aware_random_full_s0/heldout_common_id/randomized_id.csv results/human_aware_random_full_s0/heldout_common_id/randomized_id.json results/human_aware_random_full_s0/heldout_common_id/shifted_exact_h036.csv results/human_aware_random_full_s0/heldout_common_id/shifted_exact_h036.json results/human_aware_sac_v1/checkpoint_validation_common_id.json results/human_aware_sac_v1/checkpoint_validation_common_id/callback_best.csv results/human_aware_sac_v1/checkpoint_validation_common_id/callback_best.json results/human_aware_sac_v1/checkpoint_validation_common_id/logs/callback_best_20261004T215316491618Z_1.log results/human_aware_sac_v1/checkpoint_validation_common_id/logs/step_1000000_20261004T214905448210Z_1.log results/human_aware_sac_v1/checkpoint_validation_common_id/logs/step_100000_20261004T214025507430Z_1.log results/human_aware_sac_v1/checkpoint_validation_common_id/logs/step_1500000_20261004T215048277731Z_1.log results/human_aware_sac_v1/checkpoint_validation_common_id/logs/step_2000000_20261004T215200812877Z_1.log results/human_aware_sac_v1/checkpoint_validation_common_id/logs/step_200000_20261004T214229638382Z_1.log results/human_aware_sac_v1/checkpoint_validation_common_id/logs/step_400000_20261004T214430827953Z_1.log results/human_aware_sac_v1/checkpoint_validation_common_id/logs/step_600000_20261004T214656014081Z_1.log results/human_aware_sac_v1/checkpoint_validation_common_id/step_100000.csv results/human_aware_sac_v1/checkpoint_validation_common_id/step_100000.json results/human_aware_sac_v1/checkpoint_validation_common_id/step_1000000.csv results/human_aware_sac_v1/checkpoint_validation_common_id/step_1000000.json results/human_aware_sac_v1/checkpoint_validation_common_id/step_1500000.csv results/human_aware_sac_v1/checkpoint_validation_common_id/step_1500000.json results/human_aware_sac_v1/checkpoint_validation_common_id/step_200000.csv

End: 2026-10-04T23:58:47.910973+00:00
Exit code: 0

### Publication command
Start: 2026-10-04T23:58:47.910973+00:00
Command: git add -- results/human_aware_sac_v1/checkpoint_validation_common_id/step_200000.json results/human_aware_sac_v1/checkpoint_validation_common_id/step_2000000.csv results/human_aware_sac_v1/checkpoint_validation_common_id/step_2000000.json results/human_aware_sac_v1/checkpoint_validation_common_id/step_400000.csv results/human_aware_sac_v1/checkpoint_validation_common_id/step_400000.json results/human_aware_sac_v1/checkpoint_validation_common_id/step_600000.csv results/human_aware_sac_v1/checkpoint_validation_common_id/step_600000.json tests/test_human_cue_probe.py

End: 2026-10-04T23:58:48.254801+00:00
Exit code: 0

### Publication command
Start: 2026-10-04T23:58:48.254801+00:00
Command: git diff --cached --name-only -z

End: 2026-10-04T23:58:48.297807+00:00
Exit code: 0

### Publication command
Start: 2026-10-04T23:58:48.298776+00:00
Command: git diff --cached --check

End: 2026-10-04T23:58:48.374810+00:00
Exit code: 0

### Publication command
Start: 2026-10-04T23:58:48.375811+00:00
Command: git commit -m "Add final HRI evaluation results and eval-only human-cue probes"

End: 2026-10-04T23:58:49.712957+00:00
Exit code: 0

### Publication command
Start: 2026-10-04T23:58:49.712957+00:00
Command: git push -u origin final-eval-2026-10

End: 2026-10-04T23:58:53.517065+00:00
Exit code: 0

### Credential lookup (secret stdout captured and never logged)
Start: 2026-10-04T23:58:53.517065+00:00
Command: git credential fill (protocol=https, host=github.com)

End: 2026-10-04T23:58:53.988080+00:00
Exit code: 0

### GitHub API command
Start: 2026-10-04T23:58:53.989087+00:00
Command: GET https://api.github.com/repos/abdulahad2008/piper_simulation/pulls?state=open&head=abdulahad2008%3Afinal-eval-2026-10&base=main

End: 2026-10-04T23:58:54.744618+00:00
Exit code: 0

### GitHub API command
Start: 2026-10-04T23:58:54.745604+00:00
Command: POST https://api.github.com/repos/abdulahad2008/piper_simulation/pulls

End: 2026-10-04T23:58:56.931087+00:00
Exit code: 0

Pull request: https://github.com/abdulahad2008/piper_simulation/pull/7; not merged.

### Publication command
Start: 2026-10-04T23:58:56.932075+00:00
Command: git add -- results/final_eval_2026_10/RUN_LOG.md results/final_eval_2026_10/PULL_REQUEST.md

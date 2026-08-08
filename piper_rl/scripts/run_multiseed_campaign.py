"""Run and aggregate the preregistered human-aware SAC seed-1/2 campaign.

The script deliberately orchestrates existing train/evaluate entry points rather
than changing their dynamics.  It records every state transition atomically and
never resumes a partial SAC run: exact restoration of all worker RNG/callback
state is not available, so a partial run remains preserved and the campaign
stops for review.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from piper_rl.human_aware_env import PiperHumanAwarePickPlaceEnv
from piper_rl.human_config import HumanAwareEnvConfig
from piper_rl.human_motion import trajectory_fingerprint, trajectory_fingerprint_sha256


ROOT = Path("results/human_aware_multiseed_s0_s2")
STATE_PATH = ROOT / "campaign_state.json"
ANALYSIS_PATH = ROOT / "analysis_manifest.json"
GRID = (100_000, 200_000, 400_000, 600_000, 1_000_000, 1_500_000, 2_000_000)
COMMON_ID_SEED = 21_000
COMMON_ID_EPISODES = 50
EVALUATIONS = (
    ("no_human", "no-human", 200, 30_000),
    ("exact_h036", "fixed-exact-h036-v1", 200, 31_000),
    ("constrained_fixed_height_randomized", "fixed", 200, 31_000),
    ("shifted_exact_h036", "fixed-exact-shifted-h036-v1", 200, 32_000),
    ("randomized_id", "randomized-id", 500, 40_000),
    ("ood", "ood", 500, 50_000),
)


@dataclass(frozen=True)
class RunSpec:
    method: str
    seed: int
    run_name: str
    preset: str
    difficulty: float
    reference_run: str


SPECS = (
    RunSpec("fixed", 1, "human_aware_fixed_s1", "fixed-exact-h036-v1", 1.0,
            "human_aware_fixed_s0"),
    RunSpec("curriculum", 1, "human_aware_curriculum_s1", "curriculum", 0.0,
            "human_aware_sac_v1"),
    RunSpec("full_random", 1, "human_aware_random_full_s1", "randomized", 1.0,
            "human_aware_random_full_s0"),
    RunSpec("fixed", 2, "human_aware_fixed_s2", "fixed-exact-h036-v1", 1.0,
            "human_aware_fixed_s0"),
    RunSpec("curriculum", 2, "human_aware_curriculum_s2", "curriculum", 0.0,
            "human_aware_sac_v1"),
    RunSpec("full_random", 2, "human_aware_random_full_s2", "randomized", 1.0,
            "human_aware_random_full_s0"),
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False,
                                     dir=path.parent, suffix=".tmp") as stream:
        json.dump(value, stream, indent=2, default=str)
        stream.write("\n")
        temp = Path(stream.name)
    os.replace(temp, path)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def result_dir(spec: RunSpec) -> Path:
    return Path("results") / spec.run_name


def run_dir(spec: RunSpec) -> Path:
    return Path("runs") / spec.run_name


def state() -> dict:
    if STATE_PATH.exists():
        return load_json(STATE_PATH)
    return {
        "schema": "piper_human_aware_multiseed_campaign_v1",
        "created_at": now(),
        "seed0_mapping": {
            "fixed": "human_aware_fixed_s0",
            "curriculum": "human_aware_sac_v1",
            "full_random": "human_aware_random_full_s0",
        },
        "runs": {spec.run_name: {"method": spec.method, "seed": spec.seed,
                                  "status": "pending"} for spec in SPECS},
    }


def save_state(current: dict) -> None:
    current["updated_at"] = now()
    atomic_json(STATE_PATH, current)


def set_status(current: dict, spec: RunSpec, status: str, **details: Any) -> None:
    record = current["runs"][spec.run_name]
    record["status"] = status
    record.update(details)
    save_state(current)


def manifest() -> dict:
    return {
        "schema": "piper_human_aware_multiseed_analysis_v1",
        "created_at": now(),
        "training_seeds": [0, 1, 2],
        "new_training_seeds": [1, 2],
        "evaluation_seed_protocol": {
            "common_validation": {"start": COMMON_ID_SEED, "episodes": COMMON_ID_EPISODES,
                                  "distribution": "evaluation_id"},
            "heldout": [{"name": name, "experiment": experiment, "episodes": episodes,
                         "start_seed": seed} for name, experiment, episodes, seed in EVALUATIONS],
        },
        "primary_outcomes": [
            "randomized_id.task_success_rate", "randomized_id.human_collision_rate",
            "randomized_id.collision_free_success_rate", "ood.task_success_rate",
            "ood.human_collision_rate", "ood.collision_free_success_rate",
            "normalized_success_auc", "steps_to_50pct", "steps_to_70pct", "steps_to_80pct",
        ],
        "secondary_outcomes": [
            "exact_h036 task success and collision", "constrained-height crossing performance",
            "shifted-exact crossing performance", "no-human performance", "near-miss rate",
            "minimum human separation", "human safety cost", "completion time", "waiting steps",
            "placement error", "grasp rate", "lift rate", "mean episode reward",
        ],
        "hypotheses": [
            "Fixed training will specialize strongly to the exact or similar fixed crossing.",
            "Fixed training will generalize poorly to randomized-ID and OOD motion.",
            "Curriculum training may improve early sample efficiency.",
            "Full-random training may provide stronger final ID/OOD robustness.",
            "High task success does not necessarily imply low human-collision risk.",
        ],
        "selection": {
            "candidates": [*GRID, "native_callback_best"],
            "metric": "highest randomized-ID task_success_rate",
            "ties": "earlier checkpoint wins; callback best uses its saved step only if available",
            "heldout_test_results_used": False,
        },
        "smoothing": "none; all learning curves use the seven saved checkpoint evaluations",
        "replication_unit": "trained seed, not individual evaluation episode",
    }


def normalize_args(args: dict) -> dict:
    value = copy.deepcopy(args)
    value.setdefault("learning_starts", None)  # Older seed-0 curriculum config predates this CLI field.
    return value


def deep_diff(before: Any, after: Any, path: str = "") -> list[dict]:
    if isinstance(before, dict) and isinstance(after, dict):
        result = []
        for key in sorted(set(before) | set(after)):
            child = f"{path}.{key}" if path else key
            result.extend(deep_diff(before.get(key), after.get(key), child))
        return result
    if isinstance(before, list) and isinstance(after, list):
        if before == after:
            return []
    elif before == after:
        return []
    return [{"path": path, "before": before, "after": after}]


def expected_config(spec: RunSpec) -> dict:
    ref = load_json(Path("runs") / spec.reference_run / "config.json")
    args = normalize_args(ref["args"])
    args.update({"seed": spec.seed, "run_name": spec.run_name, "human_preset": spec.preset,
                 "human_difficulty": spec.difficulty, "load": None, "resume": None})
    cfg = HumanAwareEnvConfig.from_preset(spec.preset, difficulty=spec.difficulty)
    cfg.include_human_state = not args["no_human_state"]
    cfg.action_mode = args["action_mode"]
    cfg.curriculum = not args["no_curriculum"]
    cfg.domain_rand.enabled = not args["no_domain_rand"]
    cfg.noise.enabled = not args["no_noise"]
    return {"args": args, "env": cfg.to_dict()}


def validate_config(spec: RunSpec, actual: dict) -> dict:
    expected = expected_config(spec)
    actual = {"args": normalize_args(actual["args"]), "env": actual["env"]}
    differences = deep_diff(expected, actual)
    permitted = {"args.seed", "args.run_name"}
    invalid = [item for item in differences if item["path"] not in permitted]
    return {"reference_run": spec.reference_run, "run_name": spec.run_name,
            "differences": differences, "invalid_differences": invalid,
            "valid": not invalid}


def write_config_audit(spec: RunSpec, actual: dict | None = None) -> dict:
    proposed = expected_config(spec)
    result = result_dir(spec)
    atomic_json(result / "config.json", actual or proposed)
    audit = validate_config(spec, actual or proposed)
    atomic_json(result / "config_diff.json", audit)
    if not audit["valid"]:
        raise RuntimeError(f"invalid controlled diff for {spec.run_name}: {audit['invalid_differences']}")
    return audit


def preflight(spec: RunSpec) -> dict:
    """Exercise reset/action paths without changing training code or global dynamics."""
    random.seed(spec.seed)
    np.random.seed(spec.seed)
    torch.manual_seed(spec.seed)
    cfg = HumanAwareEnvConfig.from_preset(spec.preset, difficulty=spec.difficulty)
    env = PiperHumanAwarePickPlaceEnv(cfg)
    action_rng = np.random.default_rng(spec.seed + 90_001)
    resets = 200 if spec.method == "fixed" else 80
    records, finite, collision_free = [], True, True
    for index in range(resets):
        reset_seed = spec.seed * 100_000 + index
        obs, info = env.reset(seed=reset_seed)
        trajectory = env.human_trajectory
        fingerprint = trajectory_fingerprint(trajectory, cfg.human_motion)
        fingerprint["preset"] = spec.preset
        next_obs, reward, _, _, _ = env.step(action_rng.uniform(
            -1, 1, env.action_space.shape).astype(np.float32))
        finite &= bool(np.isfinite(obs).all() and np.isfinite(next_obs).all() and np.isfinite(reward))
        collision_free &= not env._robot_human_contacts()[0]
        records.append({
            "reset_seed": reset_seed,
            "human_episode": bool(info["human_episode"]),
            "trajectory_type": info["human_trajectory_type"],
            "appearance_time": info["human_appearance_time"],
            "speed": info["human_motion_speed"],
            "start_side": info["human_start_side"],
            "fingerprint": trajectory_fingerprint_sha256(fingerprint),
            "object_position": env.obj_pos.tolist(), "target_position": env.target_pos.tolist(),
        })
    active = [row for row in records if row["human_episode"]]
    fps = {row["fingerprint"] for row in active}
    object_positions = {tuple(row["object_position"]) for row in records}
    target_positions = {tuple(row["target_position"]) for row in records}
    sampled_types = sorted({row["trajectory_type"] for row in active})
    checks = {
        "finite_observations_actions_rewards": finite,
        "initial_robot_human_contact_free": collision_free,
        "human_state_included": env.observation_space.shape == (64,),
        "manipulation_curriculum_enabled": cfg.curriculum,
        "object_randomization_varies": len(object_positions) > 1,
        "target_randomization_varies": len(target_positions) > 1,
        "no_ood_distribution": cfg.human_distribution not in {"evaluation_ood"},
    }
    if spec.method == "fixed":
        checks.update({
            "all_human_present": len(active) == resets,
            "one_unique_trajectory_fingerprint": len(fps) == 1,
            "hand_height_exactly_036": tuple(cfg.human_motion.hand_height) == (0.36, 0.36),
            "human_difficulty_fixed_one": cfg.human_difficulty == 1.0,
            "no_human_randomization": (cfg.human_motion.pause_probability == 0.0
                                        and cfg.human_motion.reverse_probability == 0.0
                                        and cfg.human_motion.end_position_jitter == 0.0),
        })
    elif spec.method == "full_random":
        checks.update({
            "human_difficulty_fixed_one": cfg.human_difficulty == 1.0,
            "multiple_trajectory_types_sampled": len(sampled_types) > 1,
            "training_id_distribution": cfg.human_distribution == "training_id",
        })
    else:
        checks.update({
            "curriculum_starts_at_reference_difficulty": cfg.human_difficulty == 0.0,
            "curriculum_distribution": cfg.human_distribution == "curriculum_id",
            "training_types_match_reference": tuple(cfg.human_motion.trajectory_types)
                                        == tuple(HumanAwareEnvConfig().human_motion.trajectory_types),
        })
    report = {
        "run_name": spec.run_name, "method": spec.method, "seed": spec.seed,
        "resets": resets, "seed_propagation": {
            "python_random": spec.seed, "numpy_global": spec.seed, "torch": spec.seed,
            "gymnasium_reset_seed_pattern": f"{spec.seed}*100000+offset",
            "training_entry_point": "SAC(seed=seed), build_vec_env(seed=seed), Monitor/action-space seeded per rank",
            "evaluation_entry_point": "deterministic fixed seed protocol in evaluate_human_aware.py",
        },
        "sampled_trajectory_types": sampled_types,
        "unique_active_fingerprints": len(fps),
        "unique_object_positions": len(object_positions), "unique_target_positions": len(target_positions),
        "checks": checks, "valid": all(checks.values()),
    }
    env.close()
    atomic_json(result_dir(spec) / "preflight_report.json", report)
    if not report["valid"]:
        raise RuntimeError(f"preflight failed for {spec.run_name}: {checks}")
    return report


def run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND: " + subprocess.list2cmdline(command) + "\n")
        log.flush()
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    if completed.returncode:
        raise RuntimeError(f"command failed ({completed.returncode}); see {log_path}")


def run_is_trained(spec: RunSpec) -> bool:
    run = run_dir(spec)
    required = [run / "best" / "best_model.zip", run / "final_model.zip",
                run / "replay_buffer.pkl", run / "config.json", run / "eval" / "evaluations.npz"]
    if not all(path.exists() for path in required):
        return False
    values = np.load(run / "eval" / "evaluations.npz")
    return bool(values["timesteps"].size and int(values["timesteps"][-1]) == 2_000_000
                and np.isfinite(values["results"]).all())


def assert_safe_run_directory(spec: RunSpec) -> None:
    run = run_dir(spec)
    if not run.exists():
        return
    if run_is_trained(spec):
        return
    if any(run.iterdir()):
        raise RuntimeError(f"partial run exists at {run}; preserving it and refusing inexact resume")


def training_command(spec: RunSpec) -> list[str]:
    return [sys.executable, "-B", "-m", "piper_rl.scripts.train", "--task", "human-aware",
            "--human-preset", spec.preset, "--human-difficulty", str(spec.difficulty),
            "--algo", "sac", "--timesteps", "2000000", "--n-envs", "4", "--seed", str(spec.seed),
            "--run-name", spec.run_name, "--out", "runs", "--eval-freq", "20000",
            "--n-eval-episodes", "15", "--checkpoint-freq", "50000", "--action-mode", "cartesian",
            "--device", "auto"]


def verify_trained(spec: RunSpec) -> dict:
    if not run_is_trained(spec):
        raise RuntimeError(f"{spec.run_name} did not reach an intact 2,000,000-step state")
    run = run_dir(spec)
    actual = load_json(run / "config.json")
    audit = write_config_audit(spec, actual)
    values = np.load(run / "eval" / "evaluations.npz")
    models = {name: path for name, path in {
        "callback_best": run / "best" / "best_model.zip", "final": run / "final_model.zip"}.items()}
    report = {"final_timestep": int(values["timesteps"][-1]), "eval_values_finite": bool(np.isfinite(values["results"]).all()),
              "model_hashes": {name: sha256(path) for name, path in models.items()}, "config_valid": audit["valid"]}
    atomic_json(result_dir(spec) / "post_training_validation.json", report)
    return report


def augment_episode_csv(path: Path, spec: RunSpec, model_hash: str, selected: str) -> None:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
        fields = list(rows[0]) if rows else []
    additions = ["training_method", "training_seed", "model_sha256", "selected_checkpoint"]
    fields.extend(name for name in additions if name not in fields)
    for row in rows:
        row.update({"training_method": spec.method, "training_seed": spec.seed,
                    "model_sha256": model_hash, "selected_checkpoint": selected})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def evaluate_command(model: Path, experiment: str, episodes: int, seed: int, out: Path,
                     video: Path | None = None) -> list[str]:
    command = [sys.executable, "-B", "-m", "piper_rl.scripts.evaluate_human_aware",
               "--model", str(model), "--algo", "sac", "--experiment", experiment,
               "--episodes", str(episodes), "--seed", str(seed), "--out", str(out)]
    if video is not None:
        command.extend(["--video", str(video), "--video-episodes", "1"])
    return command


def common_selection(spec: RunSpec) -> dict:
    result = result_dir(spec)
    run = run_dir(spec)
    candidates: list[tuple[str, Path, int | None]] = []
    for step in GRID:
        path = run / "final_model.zip" if step == 2_000_000 else run / "checkpoints" / f"piper_{step}_steps.zip"
        if not path.exists():
            raise RuntimeError(f"missing checkpoint {path}")
        candidates.append((f"step_{step}", path, step))
    candidates.append(("callback_best", run / "best" / "best_model.zip", None))
    evaluations = []
    for label, model, step in candidates:
        prefix = result / "checkpoint_validation" / label
        run_logged(evaluate_command(model, "randomized-id", COMMON_ID_EPISODES, COMMON_ID_SEED, prefix),
                   result / "logs" / f"common_{label}.log")
        summary = load_json(prefix.with_suffix(".json"))["summary"]
        evaluations.append({"label": label, "step": step, "path": str(model),
                            "sha256": sha256(model), "summary": summary})
    # Stable max implements preregistered earlier-grid tie breaking; callback-best sorts after grid.
    selected = max(enumerate(evaluations), key=lambda item: (item[1]["summary"]["task_success_rate"], -item[0]))[1]
    selected_path = Path(selected["path"])
    output = run / "selected_common_id_model.zip"
    shutil.copy2(selected_path, output)
    selection = {"distribution": "evaluation_id", "seed_start": COMMON_ID_SEED,
                 "episodes": COMMON_ID_EPISODES, "metric": "task_success_rate",
                 "candidates": evaluations, "selected": {**selected, "path": str(output),
                 "sha256": sha256(output)}}
    atomic_json(result / "checkpoint_validation.json", selection)
    return selection


def heldout_evaluations(spec: RunSpec, selected: dict) -> dict:
    result = result_dir(spec)
    model = run_dir(spec) / "selected_common_id_model.zip"
    model_hash = sha256(model)
    summaries = {}
    for name, experiment, episodes, seed in EVALUATIONS:
        prefix = result / "heldout" / name
        video = result / "videos" / f"{name}.mp4" if name in {"exact_h036", "randomized_id", "ood"} else None
        if video is not None:
            video.parent.mkdir(parents=True, exist_ok=True)
        run_logged(evaluate_command(model, experiment, episodes, seed, prefix, video),
                   result / "logs" / f"heldout_{name}.log")
        augment_episode_csv(prefix.with_suffix(".csv"), spec, model_hash, selected["selected"]["label"])
        summaries[name] = load_json(prefix.with_suffix(".json"))["summary"]
    return summaries


def curate_run(spec: RunSpec, validation: dict, selection: dict, heldout: dict,
               duration: float) -> None:
    run = run_dir(spec); result = result_dir(spec)
    values = np.load(run / "eval" / "evaluations.npz")
    np.savez_compressed(result / "evaluations.npz", **{key: values[key] for key in values.files})
    models = {"callback_best": run / "best" / "best_model.zip", "final": run / "final_model.zip",
              "selected_common_id": run / "selected_common_id_model.zip"}
    hashes = {name: {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
              for name, path in models.items()}
    atomic_json(result / "model_hashes.json", hashes)
    summary = {"run_name": spec.run_name, "method": spec.method, "seed": spec.seed,
               "duration_seconds": duration, "duration_hours": duration / 3600,
               "final_timestep": validation["final_timestep"], "models": hashes,
               "common_selection": selection["selected"], "heldout": heldout}
    atomic_json(result / "training_summary.json", summary)
    lines = [f"# {spec.run_name}", "", f"- Method/seed: `{spec.method}` / `{spec.seed}`",
             f"- Steps: `{validation['final_timestep']:,}`", f"- Duration: `{duration / 3600:.2f} h`",
             f"- Common-ID selected checkpoint: `{selection['selected']['label']}`",
             f"- Model SHA-256: `{hashes['selected_common_id']['sha256']}`", "",
             "## Held-out summaries", "", "| Distribution | Success | Collision | Collision-free success |",
             "|---|---:|---:|---:|"]
    for name, values in heldout.items():
        lines.append(f"| {name} | {values['task_success_rate']:.3f} | {values['human_collision_rate']:.3f} | {values['collision_free_success_rate']:.3f} |")
    (result / "TRAINING_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_one(spec: RunSpec, current: dict) -> None:
    assert_safe_run_directory(spec)
    if current["runs"][spec.run_name]["status"] == "completed" and run_is_trained(spec):
        return
    set_status(current, spec, "preflight", started_at=now())
    write_config_audit(spec)
    preflight(spec)
    run = run_dir(spec)
    if not run_is_trained(spec):
        set_status(current, spec, "training", training_started_at=now(), command=training_command(spec))
        t0 = time.monotonic()
        run_logged(training_command(spec), result_dir(spec) / "logs" / "training.log")
        duration = time.monotonic() - t0
    else:
        duration = current["runs"][spec.run_name].get("duration_seconds", 0.0)
    validation = verify_trained(spec)
    set_status(current, spec, "trained", training_finished_at=now(), duration_seconds=duration,
               final_timestep=validation["final_timestep"], model_hashes=validation["model_hashes"])
    set_status(current, spec, "evaluating", evaluation_started_at=now())
    selection = common_selection(spec)
    heldout = heldout_evaluations(spec, selection)
    curate_run(spec, validation, selection, heldout, duration)
    set_status(current, spec, "completed", completed_at=now(),
               selected_model_sha256=selection["selected"]["sha256"],
               selected_checkpoint=selection["selected"]["label"])


def seed0_sources() -> dict[str, dict[str, Path]]:
    fixed = Path("results/human_aware_fixed_s0")
    full = Path("results/human_aware_random_full_s0")
    return {
        "fixed": {name: fixed / "heldout" / f"{name}.json" for name, *_ in EVALUATIONS},
        "curriculum": {name: fixed / "comparison_evaluations" / "curriculum" / f"{name}.json"
                       for name, *_ in EVALUATIONS},
        "full_random": {
            "no_human": full / "evaluation" / "no_human.json",
            "exact_h036": fixed / "pretraining_reference_evaluation" / "full_random_exact_h036.json",
            "constrained_fixed_height_randomized": full / "evaluation" / "fixed.json",
            "shifted_exact_h036": fixed / "comparison_evaluations" / "full_random" / "shifted_exact_h036.json",
            "randomized_id": full / "evaluation" / "randomized_id.json",
            "ood": full / "evaluation" / "ood.json",
        },
    }


def curve_rows(method: str, seed: int) -> list[dict]:
    if seed == 0:
        if method == "fixed": root = Path("results/human_aware_fixed_s0/checkpoint_validation")
        else:
            root = Path("results/human_aware_random_full_s0/learning_efficiency")
        rows = []
        for step in GRID:
            path = (root / f"step_{step}.json" if method == "fixed"
                    else root / f"{'reference' if method == 'curriculum' else 'full_random'}_{step}.json")
            summary = load_json(path)["summary"]
            rows.append({"method": method, "training_seed": seed, "step": step,
                         "task_success_rate": summary["task_success_rate"],
                         "collision_free_success_rate": summary["collision_free_success_rate"],
                         "human_collision_rate": summary["human_collision_rate"]})
        return rows
    root = Path("results") / f"human_aware_{'random_full' if method == 'full_random' else method}_s{seed}" / "checkpoint_validation"
    rows = []
    for step in GRID:
        summary = load_json(root / f"step_{step}.json")["summary"]
        rows.append({"method": method, "training_seed": seed, "step": step,
                     "task_success_rate": summary["task_success_rate"],
                     "collision_free_success_rate": summary["collision_free_success_rate"],
                     "human_collision_rate": summary["human_collision_rate"]})
    return rows


def milestones(rows: list[dict]) -> dict:
    values = np.asarray([row["task_success_rate"] for row in rows], dtype=float)
    steps = np.asarray([row["step"] for row in rows], dtype=float)
    result = {"first_success_checkpoint": None, "steps_to_50pct": None,
              "steps_to_70pct": None, "steps_to_80pct": None}
    for key, threshold in (("first_success_checkpoint", 1e-12), ("steps_to_50pct", .5),
                           ("steps_to_70pct", .7), ("steps_to_80pct", .8)):
        idx = np.flatnonzero(values >= threshold)
        if len(idx): result[key] = int(steps[idx[0]])
    span = max(1.0, steps[-1] - steps[0])
    result["normalized_success_auc"] = float(np.trapezoid(values, steps) / span)
    result["normalized_collision_free_success_auc"] = float(np.trapezoid(
        np.asarray([row["collision_free_success_rate"] for row in rows]), steps) / span)
    return result


def aggregate() -> None:
    current = state()
    if any(current["runs"][spec.run_name]["status"] != "completed" for spec in SPECS):
        raise RuntimeError("cannot aggregate before every seed-1/2 run is completed")
    source0 = seed0_sources()
    records, learning = [], []
    for method in ("fixed", "curriculum", "full_random"):
        for seed in (0, 1, 2):
            paths = source0[method] if seed == 0 else {
                name: Path("results") / f"human_aware_{'random_full' if method == 'full_random' else method}_s{seed}"
                      / "heldout" / f"{name}.json" for name, *_ in EVALUATIONS}
            for name, path in paths.items():
                summary = load_json(path)["summary"]
                records.append({"method": method, "training_seed": seed, "distribution": name, **summary})
            rows = curve_rows(method, seed)
            learning.extend(rows)
            records.append({"method": method, "training_seed": seed, "distribution": "learning_efficiency", **milestones(rows)})
    metrics = ["task_success_rate", "collision_free_success_rate", "human_collision_rate",
               "near_miss_rate", "mean_minimum_separation", "mean_human_safety_cost",
               "mean_completion_time_successes", "mean_waiting_steps", "mean_placement_error",
               "grasp_rate", "lift_rate", "mean_episode_reward"]
    aggregate_rows = []
    for method in ("fixed", "curriculum", "full_random"):
        for name, *_ in EVALUATIONS:
            values = [row for row in records if row["method"] == method and row["distribution"] == name]
            for metric in metrics:
                x = np.asarray([row[metric] for row in values if row.get(metric) is not None], dtype=float)
                mean, sd = float(x.mean()), float(x.std(ddof=1))
                half = 4.30265272975 * sd / np.sqrt(len(x))
                aggregate_rows.append({"method": method, "distribution": name, "metric": metric,
                                       "n_training_seeds": len(x), "mean": mean, "sample_sd": sd,
                                       "min": float(x.min()), "max": float(x.max()),
                                       "t95_low_unstable_n3": mean-half, "t95_high_unstable_n3": mean+half})
    paired = []
    for left, right in (("fixed", "curriculum"), ("fixed", "full_random"), ("curriculum", "full_random")):
        for name, *_ in EVALUATIONS:
            for metric in ("task_success_rate", "collision_free_success_rate", "human_collision_rate"):
                for seed in (0, 1, 2):
                    a = next(row for row in records if row["method"] == left and row["training_seed"] == seed and row["distribution"] == name)
                    b = next(row for row in records if row["method"] == right and row["training_seed"] == seed and row["distribution"] == name)
                    paired.append({"left": left, "right": right, "distribution": name,
                                   "metric": metric, "training_seed": seed, "left_minus_right": a[metric] - b[metric]})
    def write_csv(path: Path, rows: list[dict]) -> None:
        fields = sorted({key for row in rows for key in row})
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    ROOT.mkdir(parents=True, exist_ok=True)
    write_csv(ROOT / "per_seed_metrics.csv", records)
    write_csv(ROOT / "aggregate_metrics.csv", aggregate_rows)
    write_csv(ROOT / "paired_method_differences.csv", paired)
    write_csv(ROOT / "learning_efficiency.csv", learning)
    models = {}
    for spec in SPECS:
        models[spec.run_name] = load_json(result_dir(spec) / "model_hashes.json")
    models["seed0"] = {
        "curriculum": sha256(Path("runs/human_aware_sac_v1/best/best_model.zip")),
        "full_random": sha256(Path("runs/human_aware_random_full_s0/best/best_model.zip")),
        "fixed": sha256(Path("runs/human_aware_fixed_s0/selected_common_id_model.zip")),
    }
    atomic_json(ROOT / "model_manifest.json", models)
    atomic_json(ROOT / "multiseed_summary.json", {"records": records, "aggregate": aggregate_rows,
                                                   "paired_differences": paired, "learning": learning})
    lines = ["# Human-aware SAC multi-seed replication (seeds 0-2)", "",
             "The unit of replication is the trained seed (`n=3`), not the number of evaluation episodes.",
             "The t intervals are descriptive and unstable with three seeds; no superiority claim is warranted.", "",
             "## Primary outcomes", "", "See `aggregate_metrics.csv`, `paired_method_differences.csv`, and `learning_efficiency.csv` for complete values.",
             "", "## Interpretation", "",
             "Fixed exact-motion training should be assessed for specialization versus randomized-ID/OOD robustness across all three seeds. Curriculum and full-random comparisons are simulation evidence only and are not deployment-safety claims."]
    (ROOT / "MULTISEED_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    if not ANALYSIS_PATH.exists(): atomic_json(ANALYSIS_PATH, manifest())
    current = state(); save_state(current)
    reference_sizes = sum(path.stat().st_size for name in ("human_aware_sac_v1", "human_aware_random_full_s0", "human_aware_fixed_s0")
                          for path in (Path("runs") / name).rglob("*") if path.is_file())
    free = shutil.disk_usage(Path.cwd()).free
    estimate = int(reference_sizes * 2.2)  # six new runs relative to three completed representative runs.
    storage = {"reference_runs_bytes": reference_sizes, "estimated_campaign_bytes": estimate,
               "free_bytes": free, "sufficient": free > estimate}
    atomic_json(ROOT / "storage_preflight.json", storage)
    if not storage["sufficient"]:
        raise RuntimeError(f"insufficient storage: need approximately {estimate} bytes, have {free}")
    for spec in SPECS:
        assert_safe_run_directory(spec)
        write_config_audit(spec)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="write preregistration, state, storage and config audits")
    parser.add_argument("--run", choices=[spec.run_name for spec in SPECS], help="run one campaign member")
    parser.add_argument("--all", action="store_true", help="run every pending member sequentially in preregistered order")
    parser.add_argument("--aggregate", action="store_true", help="write the final three-seed aggregate after completion")
    args = parser.parse_args(argv)
    if not any((args.prepare, args.run, args.all, args.aggregate)):
        parser.error("choose --prepare, --run, --all, or --aggregate")
    if args.prepare: prepare()
    current = state()
    selected = [spec for spec in SPECS if spec.run_name == args.run] if args.run else SPECS if args.all else ()
    for spec in selected:
        try:
            run_one(spec, current)
        except Exception as exc:
            set_status(current, spec, "failed", failed_at=now(), error=str(exc))
            raise
    if args.aggregate: aggregate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Explicit, provenance-preserving recovery actions for the multiseed campaign."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import tempfile
import re
from datetime import datetime, timezone
from pathlib import Path

from piper_rl.scripts import run_multiseed_campaign as campaign


ROOT = campaign.ROOT
STATE = campaign.STATE_PATH


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False,
                                     dir=path.parent, suffix=".tmp") as stream:
        json.dump(value, stream, indent=2, default=str)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def save_state(value: dict) -> None:
    value["updated_at"] = datetime.now(timezone.utc).isoformat()
    atomic_json(STATE, value)


def inventory(directory: Path) -> list[dict]:
    rows = []
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            rows.append({"path": str(path.relative_to(directory)), "bytes": path.stat().st_size,
                         "sha256": campaign.sha256(path)})
    return rows


def last_durable_checkpoint(directory: Path) -> int:
    """Return the largest fully named periodic checkpoint, or zero if absent."""
    steps = []
    for path in (directory / "checkpoints").glob("piper_*_steps.zip"):
        match = re.fullmatch(r"piper_(\d+)_steps\.zip", path.name)
        if match:
            steps.append(int(match.group(1)))
    return max(steps, default=0)


def last_tensorboard_step(directory: Path) -> int:
    """Read the greatest scalar step from the saved event files without training."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as exc:  # A preservation record must never guess this value.
        raise RuntimeError("TensorBoard parser is required to preserve the interrupted run") from exc
    maximum = 0
    for event in (directory / "tb").glob("**/events.out.tfevents.*"):
        accumulator = EventAccumulator(str(event))
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", []):
            values = accumulator.Scalars(tag)
            if values:
                maximum = max(maximum, max(value.step for value in values))
    return maximum


def archive_curriculum_s2() -> None:
    """Move the incomplete run and its curated partial outputs without deleting data."""
    spec = next(item for item in campaign.SPECS if item.run_name == "human_aware_curriculum_s2")
    source_run, source_result = campaign.run_dir(spec), campaign.result_dir(spec)
    if not source_run.exists():
        raise RuntimeError(f"missing interrupted run directory: {source_run}")
    if (source_run / "final_model.zip").exists():
        raise RuntimeError("refusing to archive a run that has a final model")
    if (source_run / "replay_buffer.pkl").exists():
        raise RuntimeError("refusing this clean-restart recovery: replay buffer unexpectedly exists")
    durable_step = last_durable_checkpoint(source_run)
    tensorboard_step = last_tensorboard_step(source_run)
    if durable_step <= 0 or tensorboard_step < durable_step:
        raise RuntimeError("interrupted run has no coherent durable checkpoint/TensorBoard evidence")
    stamp = timestamp()
    archived_run = source_run.with_name(f"{source_run.name}_interrupted_powerloss_{stamp}")
    archived_result = source_result.with_name(f"{source_result.name}_interrupted_powerloss_{stamp}")
    if archived_run.exists() or archived_result.exists():
        raise RuntimeError("timestamp collision while preserving interrupted artifacts")
    before = {"run": inventory(source_run), "result": inventory(source_result) if source_result.exists() else []}
    shutil.move(str(source_run), str(archived_run))
    if source_result.exists():
        shutil.move(str(source_result), str(archived_result))
    preserved = {
        "schema": "piper_human_aware_interrupted_run_preservation_v1",
        "reason": "power loss; no replay buffer or complete deterministic worker/callback state",
        "archived_at": datetime.now(timezone.utc).isoformat(),
        "classification": "archival evidence only; not a research result and not resumable",
        "last_confirmed_tensorboard_timestep": tensorboard_step,
        "last_durable_checkpoint_timestep": durable_step,
        "original_inventory": before,
    }
    atomic_json(archived_result / "PRESERVATION_MANIFEST.json", preserved)
    state = campaign.state()
    record = state["runs"][spec.run_name]
    history = record.setdefault("interrupted_history", [])
    history.append({
        "preserved_at": preserved["archived_at"], "status": "interrupted_preserved",
        "run_archive": str(archived_run), "result_archive": str(archived_result),
        "last_confirmed_tensorboard_timestep": tensorboard_step,
        "last_durable_checkpoint_timestep": durable_step,
        "reason": preserved["reason"],
        "classification": preserved["classification"],
    })
    record["status"] = "awaiting_clean_restart"
    record["preserved_interrupted_artifacts"] = history[-1]
    state.setdefault("recovery", {})["audit_reconciled_at"] = datetime.now(timezone.utc).isoformat()
    state["recovery"]["audit_summary"] = {
        "verified_completed": [
            "human_aware_fixed_s1", "human_aware_curriculum_s1",
            "human_aware_random_full_s1", "human_aware_fixed_s2",
        ],
        "curriculum_s2": {
            "status": "awaiting_clean_restart",
            "last_tensorboard_timestep": tensorboard_step,
            "last_durable_checkpoint_timestep": durable_step,
            "action": "clean retrain; no replay buffer or complete deterministic recovery state",
        },
        "random_full_s2": {"last_confirmed_timestep": 0, "action": "pending after curriculum_s2"},
    }
    save_state(state)
    print(json.dumps(history[-1], indent=2))


def archive_inexact_random_full_s2() -> None:
    """Preserve a completed but protocol-ineligible resumed seed-2 run.

    The campaign requires clean runs because it cannot restore worker/callback
    RNG state exactly.  A model completed through ``--resume`` is useful
    supplemental evidence, but must not occupy the preregistered seed-2 slot.
    """
    spec = next(item for item in campaign.SPECS if item.run_name == "human_aware_random_full_s2")
    source_run, source_result = campaign.run_dir(spec), campaign.result_dir(spec)
    if not campaign.run_is_trained(spec):
        raise RuntimeError("random_full_s2 is not an intact completed run")
    actual = campaign.load_json(source_run / "config.json").get("args", {})
    if not actual.get("resume"):
        raise RuntimeError("refusing to archive random_full_s2: it is not a resumed run")
    stamp = timestamp()
    archived_run = source_run.with_name(f"{source_run.name}_inexact_resume_{stamp}")
    archived_result = source_result.with_name(f"{source_result.name}_inexact_resume_{stamp}")
    if archived_run.exists() or archived_result.exists():
        raise RuntimeError("timestamp collision while preserving inexact resumed artifacts")
    before = {
        "run": inventory(source_run),
        "result": inventory(source_result) if source_result.exists() else [],
    }
    shutil.move(str(source_run), str(archived_run))
    if source_result.exists():
        shutil.move(str(source_result), str(archived_result))
    preserved = {
        "schema": "piper_human_aware_inexact_resume_preservation_v1",
        "reason": "completed via checkpoint/replay-buffer continuation; worker and callback state were not exactly restorable",
        "archived_at": datetime.now(timezone.utc).isoformat(),
        "classification": "supplemental evidence only; excluded from the preregistered multiseed campaign",
        "saved_args": actual,
        "original_inventory": before,
    }
    atomic_json(archived_result / "PRESERVATION_MANIFEST.json", preserved)
    state = campaign.state()
    record = state["runs"][spec.run_name]
    history = record.setdefault("inexact_resume_history", [])
    history.append({
        "preserved_at": preserved["archived_at"],
        "status": "inexact_resume_preserved",
        "run_archive": str(archived_run),
        "result_archive": str(archived_result),
        "reason": preserved["reason"],
        "classification": preserved["classification"],
    })
    record["status"] = "awaiting_clean_restart"
    record["preserved_inexact_resume_artifacts"] = history[-1]
    state.setdefault("recovery", {}).setdefault("audit_summary", {})["random_full_s2"] = {
        "status": "awaiting_clean_restart",
        "action": "archive inexact continuation and clean retrain",
    }
    save_state(state)
    print(json.dumps(history[-1], indent=2))


def reeval_random_full_s1_final() -> None:
    """Run locked held-out tests on the final 2M policy without touching callback-best outputs."""
    spec = next(item for item in campaign.SPECS if item.run_name == "human_aware_random_full_s1")
    run, result = campaign.run_dir(spec), campaign.result_dir(spec)
    final = run / "final_model.zip"
    final_output = result / "heldout_final_2m"
    if final_output.exists():
        raise RuntimeError(f"refusing to overwrite final-policy held-out outputs: {final_output}")
    validation = campaign.verify_trained(spec)
    if validation["final_timestep"] != 2_000_000:
        raise RuntimeError("final model is not exactly 2,000,000 timesteps")
    final_hash = campaign.sha256(final)
    selections = {
        "required_policy": "final_model.zip at exactly 2,000,000 steps",
        "model_path": str(final), "model_sha256": final_hash,
        "selected_checkpoint": "step_2000000_final_required_heldout",
        "heldout_test_results_used_for_selection": False,
        "locked_protocol": [{"name": n, "experiment": e, "episodes": ep, "seed": seed}
                            for n, e, ep, seed in campaign.EVALUATIONS],
    }
    atomic_json(result / "required_final_2m_heldout_protocol.json", selections)
    summaries = {}
    for name, experiment, episodes, seed in campaign.EVALUATIONS:
        prefix = final_output / name
        video = result / "videos_final_2m" / f"{name}.mp4" if name in {"exact_h036", "randomized_id", "ood"} else None
        if video is not None:
            video.parent.mkdir(parents=True, exist_ok=True)
        campaign.run_logged(campaign.evaluate_command(final, experiment, episodes, seed, prefix, video),
                            result / "logs" / f"heldout_final_2m_{name}.log")
        campaign.augment_episode_csv(prefix.with_suffix(".csv"), spec, final_hash,
                                     "step_2000000_final_required_heldout")
        summary = campaign.load_json(prefix.with_suffix(".json"))["summary"]
        if not summary.get("deterministic_prediction", False):
            raise RuntimeError(f"non-deterministic held-out evaluator result: {name}")
        summaries[name] = summary
    # Keep both provenance sets.  The original heldout/ directory remains unchanged.
    prior_summary_path = result / "training_summary.json"
    prior_summary = campaign.load_json(prior_summary_path)
    atomic_json(result / "training_summary_callback_best_1760000.json", prior_summary)
    prior_report = result / "TRAINING_SUMMARY.md"
    if prior_report.exists():
        shutil.copy2(prior_report, result / "TRAINING_SUMMARY_callback_best_1760000.md")
    model_hashes = campaign.load_json(result / "model_hashes.json")
    model_hashes["required_heldout_final_2m"] = {
        "path": str(final), "sha256": final_hash, "bytes": final.stat().st_size,
        "purpose": "locked held-out suite; not used for model selection",
    }
    atomic_json(result / "model_hashes.json", model_hashes)
    summary = copy.deepcopy(prior_summary)
    summary["heldout"] = summaries
    summary["required_heldout_dir"] = "heldout_final_2m"
    summary["required_heldout_policy"] = selections
    summary["supplemental_callback_best_1760000"] = {
        "model_path": str(run / "best" / "best_model.zip"),
        "model_sha256": campaign.sha256(run / "best" / "best_model.zip"),
        "output_directory": "heldout",
        "preserved_report": "training_summary_callback_best_1760000.json",
        "not_campaign_acceptance_evaluation": True,
    }
    atomic_json(prior_summary_path, summary)
    lines = [f"# {spec.run_name}", "", "- Training: valid final policy at `2,000,000` steps.",
             "- Required held-out suite: `final_model.zip` (not used for model selection).",
             "- Supplemental callback-best suite: preserved in `heldout/` for the `1,760,000`-step policy.",
             f"- Required final-model SHA-256: `{final_hash}`", "",
             "## Required held-out final-policy summaries", "",
             "| Distribution | Success | Collision | Collision-free success |",
             "|---|---:|---:|---:|"]
    for name, values in summaries.items():
        lines.append(f"| {name} | {values['task_success_rate']:.3f} | {values['human_collision_rate']:.3f} | {values['collision_free_success_rate']:.3f} |")
    (result / "TRAINING_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    state = campaign.state()
    record = state["runs"][spec.run_name]
    record.update({"status": "completed", "reevaluated_completed_at": datetime.now(timezone.utc).isoformat(),
                   "required_heldout_model": "final_model.zip", "required_heldout_model_sha256": final_hash,
                   "required_heldout_timestep": 2_000_000, "required_heldout_dir": "heldout_final_2m",
                   "supplemental_callback_best_heldout_dir": "heldout"})
    save_state(state)


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--reevaluate-random-full-s1-final", action="store_true")
    group.add_argument("--archive-curriculum-s2", action="store_true")
    group.add_argument("--archive-inexact-random-full-s2", action="store_true")
    args = parser.parse_args()
    if args.reevaluate_random_full_s1_final:
        reeval_random_full_s1_final()
    elif args.archive_curriculum_s2:
        archive_curriculum_s2()
    else:
        archive_inexact_random_full_s2()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

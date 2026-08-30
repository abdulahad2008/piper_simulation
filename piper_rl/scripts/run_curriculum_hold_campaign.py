"""Run clean curriculum ramp-and-hold SAC replications.

Difficulty increases linearly from 0 to 1 over the first 1.2M environment
steps, then remains at the full randomized-ID distribution for the final
0.8M steps.  Runs are deliberately sequential and each uses the existing
common-ID model-selection and held-out evaluation protocol.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from piper_rl.human_config import HumanAwareEnvConfig
from piper_rl.scripts import run_multiseed_campaign as campaign


RAMP_TIMESTEPS = 1_200_000
TOTAL_TIMESTEPS = 2_000_000
DEFAULT_SEEDS = (3, 4, 5)
ROOT: Path
STATE_PATH: Path
SPECS: tuple[campaign.RunSpec, ...]


def configure_campaign(seeds: tuple[int, ...]) -> None:
    """Set the campaign identity before reading or writing its state."""
    global ROOT, STATE_PATH, SPECS
    ordered = tuple(sorted(set(seeds)))
    if not ordered:
        raise ValueError("at least one seed is required")
    label = f"s{ordered[0]}_s{ordered[-1]}" if len(ordered) > 1 else f"s{ordered[0]}"
    ROOT = Path("results") / f"human_aware_curriculum_hold_{label}"
    STATE_PATH = ROOT / "campaign_state.json"
    SPECS = tuple(
        campaign.RunSpec("curriculum_hold", seed, f"human_aware_curriculum_hold_s{seed}",
                         "curriculum", 0.0, "human_aware_sac_v1")
        for seed in ordered
    )


configure_campaign(DEFAULT_SEEDS)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def state() -> dict[str, Any]:
    if STATE_PATH.exists():
        return campaign.load_json(STATE_PATH)
    return {
        "schema": "piper_human_aware_curriculum_hold_campaign_v1",
        "created_at": now(),
        "schedule": {
            "start_difficulty": 0.0,
            "end_difficulty": 1.0,
            "ramp_timesteps": RAMP_TIMESTEPS,
            "hold_timesteps": TOTAL_TIMESTEPS - RAMP_TIMESTEPS,
            "total_timesteps": TOTAL_TIMESTEPS,
        },
        "reference_run": "human_aware_sac_v1",
        "runs": {
            spec.run_name: {"method": spec.method, "seed": spec.seed, "status": "pending"}
            for spec in SPECS
        },
    }


def save(current: dict[str, Any]) -> None:
    current["updated_at"] = now()
    campaign.atomic_json(STATE_PATH, current)


def set_status(current: dict[str, Any], spec: campaign.RunSpec, status: str,
               **details: Any) -> None:
    current["runs"][spec.run_name].update({"status": status, **details})
    save(current)


def expected_config(spec: campaign.RunSpec) -> dict[str, Any]:
    reference = campaign.load_json(Path("runs") / spec.reference_run / "config.json")
    args = campaign.normalize_args(copy.deepcopy(reference["args"]))
    args.setdefault("curriculum_total_timesteps", None)
    args.setdefault("curriculum_ramp_timesteps", None)
    args.update({
        "seed": spec.seed,
        "run_name": spec.run_name,
        "human_preset": "curriculum",
        "human_difficulty": 0.0,
        "timesteps": TOTAL_TIMESTEPS,
        "curriculum_total_timesteps": None,
        "curriculum_ramp_timesteps": RAMP_TIMESTEPS,
        "load": None,
        "resume": None,
    })
    cfg = HumanAwareEnvConfig.from_preset("curriculum", difficulty=0.0)
    cfg.include_human_state = not args["no_human_state"]
    cfg.action_mode = args["action_mode"]
    cfg.curriculum = not args["no_curriculum"]
    cfg.domain_rand.enabled = not args["no_domain_rand"]
    cfg.noise.enabled = not args["no_noise"]
    return {"args": args, "env": cfg.to_dict()}


def audit_config(spec: campaign.RunSpec, actual: dict[str, Any] | None = None) -> dict[str, Any]:
    expected = expected_config(spec)
    value = actual or expected
    compared = {
        "args": campaign.normalize_args(value["args"]),
        "env": value["env"],
    }
    differences = campaign.deep_diff(expected, compared)
    report = {
        "reference_run": spec.reference_run,
        "run_name": spec.run_name,
        "schedule": {"ramp_timesteps": RAMP_TIMESTEPS, "total_timesteps": TOTAL_TIMESTEPS},
        "differences": differences,
        "invalid_differences": differences,
        "valid": not differences,
    }
    result = campaign.result_dir(spec)
    campaign.atomic_json(result / "config.json", value)
    campaign.atomic_json(result / "config_diff.json", report)
    if differences:
        raise RuntimeError(f"invalid configuration for {spec.run_name}: {differences}")
    return report


def training_command(spec: campaign.RunSpec) -> list[str]:
    return [
        sys.executable, "-B", "-m", "piper_rl.scripts.train",
        "--task", "human-aware", "--human-preset", "curriculum", "--human-difficulty", "0.0",
        "--curriculum-ramp-timesteps", str(RAMP_TIMESTEPS),
        "--algo", "sac", "--timesteps", str(TOTAL_TIMESTEPS), "--n-envs", "4",
        "--seed", str(spec.seed), "--run-name", spec.run_name, "--out", "runs",
        "--eval-freq", "20000", "--n-eval-episodes", "15", "--checkpoint-freq", "50000",
        "--action-mode", "cartesian", "--device", "auto",
    ]


def run_training_logged(spec: campaign.RunSpec, current: dict[str, Any], command: list[str]) -> float:
    log_path = campaign.result_dir(spec) / "logs" / "training.log"
    status_path = log_path.with_name("training_status.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND: " + subprocess.list2cmdline(command) + "\n")
        log.flush()
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, text=True)
        while process.poll() is None:
            snapshot = campaign.training_status(spec, started)
            with status_path.open("a", encoding="utf-8") as status_log:
                status_log.write(json.dumps(snapshot, sort_keys=True) + "\n")
            set_status(current, spec, "training", last_training_status=snapshot)
            if not snapshot["finite_values"] and snapshot["observed_tensorboard_timestep"]:
                process.terminate()
                process.wait(timeout=60)
                raise RuntimeError("non-finite training values; training process terminated")
            for _ in range(30 * 60):
                if process.poll() is not None:
                    break
                time.sleep(1)
        if process.returncode:
            raise RuntimeError(f"command failed ({process.returncode}); see {log_path}")
    return time.monotonic() - started


def verify_trained(spec: campaign.RunSpec) -> dict[str, Any]:
    if not campaign.run_is_trained(spec):
        raise RuntimeError(f"{spec.run_name} did not reach an intact {TOTAL_TIMESTEPS:,}-step state")
    run = campaign.run_dir(spec)
    actual = campaign.load_json(run / "config.json")
    audit = audit_config(spec, actual)
    values = np.load(run / "eval" / "evaluations.npz")
    models = {
        "callback_best": run / "best" / "best_model.zip",
        "final": run / "final_model.zip",
    }
    report = {
        "final_timestep": int(values["timesteps"][-1]),
        "eval_values_finite": bool(np.isfinite(values["results"]).all()),
        "model_hashes": {name: campaign.sha256(path) for name, path in models.items()},
        "config_valid": audit["valid"],
    }
    campaign.atomic_json(campaign.result_dir(spec) / "post_training_validation.json", report)
    return report


def run_one(spec: campaign.RunSpec, current: dict[str, Any]) -> None:
    campaign.assert_safe_run_directory(spec)
    if current["runs"][spec.run_name]["status"] == "completed" and campaign.run_is_trained(spec):
        return
    set_status(current, spec, "preflight", started_at=now())
    audit_config(spec)
    campaign.preflight(spec)
    set_status(current, spec, "training", training_started_at=now(), command=training_command(spec))
    duration = run_training_logged(spec, current, training_command(spec))
    validation = verify_trained(spec)
    set_status(current, spec, "trained", training_finished_at=now(), duration_seconds=duration,
               final_timestep=validation["final_timestep"], model_hashes=validation["model_hashes"])
    set_status(current, spec, "evaluating", evaluation_started_at=now())
    selection = campaign.common_selection(spec)
    heldout = campaign.heldout_evaluations(spec, selection)
    campaign.curate_run(spec, validation, selection, heldout, duration)
    set_status(current, spec, "completed", completed_at=now(),
               selected_model_sha256=selection["selected"]["sha256"],
               selected_checkpoint=selection["selected"]["label"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", help="run name to execute; must match --seeds")
    parser.add_argument("--all", action="store_true", help="run every pending seed sequentially")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS),
                        help="training seeds for this campaign (default: 3 4 5)")
    args = parser.parse_args(argv)
    configure_campaign(tuple(args.seeds))
    run_names = {spec.run_name for spec in SPECS}
    if args.run and args.run not in run_names:
        parser.error("--run must name a run implied by --seeds")
    if bool(args.run) == args.all:
        parser.error("choose exactly one of --run or --all")
    current = state()
    save(current)
    selected = [spec for spec in SPECS if spec.run_name == args.run] if args.run else SPECS
    for spec in selected:
        try:
            run_one(spec, current)
        except Exception as exc:
            set_status(current, spec, "failed", failed_at=now(), error=str(exc))
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

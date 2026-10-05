"""Eval-only common-ID selection for the two historical seed-0 policies.

Run one policy at a time. Existing completed results are reused, and existing
models/results are never overwritten. Training entry points are not invoked.
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import time
import uuid

from piper_rl.scripts.run_multiseed_campaign import (
    COMMON_ID_EPISODES, COMMON_ID_SEED, EVALUATIONS, GRID,
    atomic_json, evaluate_command, load_json, sha256,
)

RUNS = ("human_aware_sac_v1", "human_aware_random_full_s0")
RUN_LOG = Path("results/final_eval_2026_10/RUN_LOG.md")


def log(message: str) -> None:
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with RUN_LOG.open("a", encoding="utf-8") as stream:
        stream.write("\n" + message + "\n")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def evaluate_resumable(command: list[str], *, timeout_seconds: float = 3600) -> dict:
    """Execute one evaluation, with durable completion checks and two attempts.

    The default watchdog permits twice a conservative 30-minute job estimate.
    Callers can use a smaller remaining step-wide watchdog budget.
    """
    if "piper_rl.scripts.evaluate_human_aware" not in command:
        raise RuntimeError("only the evaluation entry point is allowed")
    if "piper_rl.scripts.train" in command:
        raise RuntimeError("training is forbidden")
    prefix = Path(command[command.index("--out") + 1])
    requested = int(command[command.index("--episodes") + 1])
    result_path = prefix.with_suffix(".json")
    if result_path.exists():
        result = load_json(result_path)
        if result["summary"]["episodes"] != requested:
            raise RuntimeError(f"wrong episode count in {result_path}: "
                               f"{result['summary']['episodes']} != {requested}")
        log(f"Skipped complete output ({requested} episodes): {result_path}")
        print(f"SKIP {prefix}: {requested} episodes", flush=True)
        return result
    if prefix.with_suffix(".csv").exists():
        raise RuntimeError(f"CSV exists without completed JSON; preserved: {prefix}")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for attempt in (1, 2):
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output_log = prefix.parent / "logs" / f"{prefix.name}_{timestamp}_{attempt}.log"
        output_log.parent.mkdir(parents=True, exist_ok=True)
        log(f"### Evaluation subprocess\nStart: {now()}\n"
            f"Command: {subprocess.list2cmdline(command)}\n"
            f"Estimate: 30 min; watchdog: {timeout_seconds:.0f} seconds\n"
            f"Output log: {output_log}")
        print("EVALUATE:", subprocess.list2cmdline(command), flush=True)
        code = 1
        try:
            with output_log.open("x", encoding="utf-8") as stream:
                process = subprocess.Popen(command, stdout=stream,
                                           stderr=subprocess.STDOUT, text=True)
                try:
                    code = process.wait(timeout=timeout_seconds)
                except subprocess.TimeoutExpired:
                    # Kill descendants as well as Windows' venv launcher.
                    kill_command = ["taskkill", "/PID", str(process.pid), "/T", "/F"]
                    log(f"### Watchdog stop\nStart: {now()}\n"
                        f"Command: {subprocess.list2cmdline(kill_command)}")
                    killed = subprocess.run(kill_command, check=False, capture_output=True)
                    log(f"End: {now()}\nExit code: {killed.returncode}")
                    process.wait()
                    code = process.returncode
                    raise RuntimeError(f"evaluation exceeded watchdog: {prefix}")
        finally:
            log(f"End: {now()}\nExit code: {code}")
        if code == 0:
            result = load_json(result_path)
            if result["summary"]["episodes"] != requested:
                raise RuntimeError(f"wrong episode count after evaluation: {result_path}")
            return result
        if result_path.exists() or prefix.with_suffix(".csv").exists():
            raise RuntimeError(f"failed evaluation left results; preserved: {prefix}; see {output_log}")
        if attempt == 2:
            raise RuntimeError(f"evaluation failed twice ({code}); see {output_log}")
        log(f"Retrying failed evaluation once (exit {code}): {prefix}")
    raise AssertionError("unreachable")


def select_run(run_name: str, *, deadline: float | None = None) -> dict:
    if deadline is None:
        deadline = time.monotonic() + 14_400
    run = Path("runs") / run_name
    result = Path("results") / run_name
    candidates = [(f"step_{step}",
                   run / "final_model.zip" if step == 2_000_000 else
                   run / "checkpoints" / f"piper_{step}_steps.zip", step)
                  for step in GRID]
    candidates.append(("callback_best", run / "best/best_model.zip", None))
    missing = []
    for label, path, _ in candidates:
        present = path.is_file()
        log(f"Candidate {run_name} / {label}: {path}; exists={present}")
        print(f"CANDIDATE {label}: {path}; exists={present}", flush=True)
        if not present:
            missing.append(str(path))
    if missing:
        raise RuntimeError("missing candidates; run skipped: " + ", ".join(missing))

    def evaluate(command: list[str]) -> dict:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError("seed-0 step exceeded twice its 120-minute estimate")
        return evaluate_resumable(command, timeout_seconds=min(3600, remaining))

    evaluations = []
    for label, model, step in candidates:
        prefix = result / "checkpoint_validation_common_id" / label
        values = evaluate(evaluate_command(model, "randomized-id", COMMON_ID_EPISODES,
                                           COMMON_ID_SEED, prefix))["summary"]
        evaluations.append({"label": label, "step": step, "path": str(model),
                            "sha256": sha256(model), "summary": values})
    selected = max(enumerate(evaluations), key=lambda item: (
        item[1]["summary"]["task_success_rate"],
        -item[1]["summary"]["human_collision_rate"], -item[0]))[1]
    destination = run / "selected_common_id_model.zip"
    if destination.exists():
        if sha256(destination) != selected["sha256"]:
            raise RuntimeError(f"existing selected model differs; preserved: {destination}")
    else:
        # Publish only a complete, hash-verified model. A power loss leaves a
        # uniquely named temporary file, so the final path remains resumable.
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        with Path(selected["path"]).open("rb") as source, temporary.open("xb") as target:
            shutil.copyfileobj(source, target)
            target.flush()
            os.fsync(target.fileno())
        if sha256(temporary) != selected["sha256"]:
            raise RuntimeError(f"selected-model copy hash mismatch: {temporary}")
        if destination.exists():
            raise RuntimeError(f"selected model appeared during copy; preserved: {destination}")
        # Windows rename rejects an existing destination (no replacement).
        temporary.rename(destination)
    selection = {"distribution": "evaluation_id", "seed_start": COMMON_ID_SEED,
                 "episodes": COMMON_ID_EPISODES, "metric": "task_success_rate",
                 "candidates": evaluations,
                 "selected": {**selected, "path": str(destination),
                              "sha256": sha256(destination)}}
    manifest = result / "checkpoint_validation_common_id.json"
    if manifest.exists():
        if load_json(manifest) != selection:
            raise RuntimeError(f"existing selection manifest differs; preserved: {manifest}")
    else:
        atomic_json(manifest, selection)
    log(f"Selected {run_name}: {selected['label']}; SHA-256: {selected['sha256']}")
    if selected["label"] == "callback_best":
        log(f"{run_name}: selection unchanged; existing held-out results stand")
        print("selection unchanged; existing held-out results stand", flush=True)
        return selection
    for suite, experiment, episodes, seed in EVALUATIONS:
        evaluate(evaluate_command(destination, experiment, episodes, seed,
                                  result / "heldout_common_id" / suite))
    return selection


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", choices=RUNS)
    args = parser.parse_args(argv)
    failed = False
    deadline = time.monotonic() + 14_400
    for run in (args.run,) if args.run else RUNS:
        try:
            select_run(run, deadline=deadline)
        except Exception as exc:
            failed = True
            log(f"Unexpected — Step 2, {run}: {type(exc).__name__}: {exc}; run stopped")
            print(f"STOP {run}: {exc}", flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())

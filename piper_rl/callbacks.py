"""Logging callbacks: the task metrics that actually tell you if it works.

``ep_rew_mean`` going up is not evidence that the arm places the cup. These
callbacks pull the per-episode summary the env emits and log the numbers a
manipulation result is judged on:

    rollout/success_rate        placed on the target, released, settled
    rollout/grasp_rate          got a two-sided grasp at least once
    rollout/lift_rate           lifted it clear of the table
    rollout/place_err_mm        final object-to-target distance
    rollout/collision_steps     steps with an arm-link collision
    rollout/safety_clamped      commands the safety layer had to clamp
"""

from __future__ import annotations

import collections
from typing import Deque

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class TaskMetricsCallback(BaseCallback):
    """Aggregates ``info['episode_summary']`` into a rolling window."""

    KEYS = ("success", "grasp_success", "lift_success", "placement_error",
            "max_lift", "collision_steps", "length", "safety_clamped",
            "safety_vetoed", "collision_free_success", "human_collision",
            "minimum_human_distance", "near_miss_events", "proximity_steps",
            "human_safety_cost", "completion_time", "waiting_steps",
            "human_difficulty")

    def __init__(self, window: int = 50, verbose: int = 0):
        super().__init__(verbose)
        self.window = window
        self.buf: dict[str, Deque] = {
            k: collections.deque(maxlen=window) for k in self.KEYS}
        self.n_episodes = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            s = info.get("episode_summary")
            if s is None:
                continue
            self.n_episodes += 1
            for k in self.KEYS:
                if k in s:
                    self.buf[k].append(float(s[k]))
        return True

    def _on_rollout_end(self) -> None:
        self._dump()

    def _dump(self) -> None:
        if not self.buf["success"]:
            return
        L = self.logger
        L.record("rollout/success_rate", float(np.mean(self.buf["success"])))
        L.record("rollout/grasp_rate", float(np.mean(self.buf["grasp_success"])))
        L.record("rollout/lift_rate", float(np.mean(self.buf["lift_success"])))
        L.record("rollout/place_err_mm",
                 float(np.mean(self.buf["placement_error"]) * 1000))
        L.record("rollout/ep_len", float(np.mean(self.buf["length"])))
        L.record("rollout/collision_steps",
                 float(np.mean(self.buf["collision_steps"])))
        L.record("rollout/safety_clamped",
                 float(np.mean(self.buf["safety_clamped"])))
        optional = {
            "collision_free_success": "collision_free_success_rate",
            "human_collision": "human_collision_rate",
            "minimum_human_distance": "minimum_human_distance",
            "near_miss_events": "near_misses",
            "proximity_steps": "proximity_steps",
            "human_safety_cost": "human_safety_cost",
            "completion_time": "completion_time",
            "waiting_steps": "waiting_steps",
            "human_difficulty": "human_difficulty",
        }
        for key, label in optional.items():
            if self.buf[key]:
                L.record(f"rollout/{label}", float(np.mean(self.buf[key])))
        L.record("rollout/episodes", self.n_episodes)

    # off-policy algorithms do not emit rollout_end often enough
    def _on_training_end(self) -> None:
        self._dump()


class PeriodicDumpCallback(BaseCallback):
    """Force a logger dump every ``every`` steps (SAC/TD3 log sparsely)."""

    def __init__(self, every: int = 2000, metrics: TaskMetricsCallback | None = None):
        super().__init__()
        self.every = every
        self.metrics = metrics

    def _on_step(self) -> bool:
        if self.num_timesteps % self.every == 0:
            if self.metrics is not None:
                self.metrics._dump()
            self.logger.dump(self.num_timesteps)
        return True


class HumanCurriculumCallback(BaseCallback):
    """Linearly raise difficulty, optionally holding the final stage afterwards."""

    def __init__(self, total_timesteps: int, start: float = 0.0,
                 end: float = 1.0, ramp_timesteps: int | None = None,
                 update_every: int = 5000, verbose: int = 0):
        super().__init__(verbose)
        self.total_timesteps = max(1, int(total_timesteps))
        self.ramp_timesteps = max(1, int(ramp_timesteps or total_timesteps))
        self.start = float(start)
        self.end = float(end)
        self.update_every = max(1, int(update_every))
        self._last_update = -self.update_every

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_update >= self.update_every:
            progress = min(1.0, self.num_timesteps / self.ramp_timesteps)
            difficulty = self.start + progress * (self.end - self.start)
            self.training_env.env_method("set_human_difficulty", difficulty)
            self.logger.record("rollout/human_difficulty", difficulty)
            self._last_update = self.num_timesteps
        return True


class HazardBudgetCallback(BaseCallback):
    """Lagrangian hazard budget for the pan task.

    Every `every` finished training episodes, reads the mean exposure time
    (seconds the person spent within 3 cm of a hot surface) from the episode
    summaries and moves the multiplier:

        lambda <- max(0, lambda + lr * (mean_exposure_s - budget_s))

    then pushes it into every training env (`set_hazard_lambda`), where the
    reward charges lambda per exposed step. Evaluation envs are untouched.
    Approximation, stated in the paper: transitions already in the replay
    buffer keep the lambda they were collected under.
    """

    def __init__(self, budget_s: float, lr: float = 2.0, every: int = 50,
                 lambda_init: float = 0.0, lambda_max: float = 200.0, verbose: int = 0,
                 path=None):
        super().__init__(verbose)
        self.path = path
        self.budget_s = float(budget_s)
        self.lr = float(lr)
        self.every = int(every)
        self.lam = float(lambda_init)
        self.lambda_max = float(lambda_max)
        self._exposures: list[float] = []
        self._n_updates = 0
        self.history: list[dict] = []

    def _on_training_start(self) -> None:
        self.training_env.env_method("set_hazard_lambda", self.lam)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            summary = info.get("episode_summary")
            if summary is not None and "exposure_time" in summary:
                self._exposures.append(float(summary["exposure_time"]))
        if len(self._exposures) >= self.every:
            mean_exp = float(np.mean(self._exposures))
            self.lam = float(np.clip(self.lam + self.lr * (mean_exp - self.budget_s), 0.0, self.lambda_max))
            self.training_env.env_method("set_hazard_lambda", self.lam)
            self._n_updates += 1
            self.history.append({"timesteps": int(self.num_timesteps), "lambda": self.lam,
                                 "mean_exposure_s": mean_exp, "episodes": len(self._exposures)})
            self.logger.record("hazard/lambda", self.lam)
            self.logger.record("hazard/mean_exposure_s", mean_exp)
            self.logger.record("hazard/budget_s", self.budget_s)
            if self.verbose:
                print(f"hazard budget: mean exposure {mean_exp:.3f} s vs budget {self.budget_s:.3f} s"
                      f" -> lambda {self.lam:.2f} (update {self._n_updates})")
            self._exposures.clear()
            self._dump()
        return True

    def _dump(self) -> None:
        if self.path is None:
            return
        import json
        from pathlib import Path
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.path).write_text(json.dumps(
            {"budget_s": self.budget_s, "lr": self.lr, "every": self.every, "history": self.history}, indent=1))

    def _on_training_end(self) -> None:
        self._dump()

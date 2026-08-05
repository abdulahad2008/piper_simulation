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
    """Linearly raise human difficulty through vector-env ``env_method`` calls."""

    def __init__(self, total_timesteps: int, start: float = 0.0,
                 end: float = 1.0, update_every: int = 5000, verbose: int = 0):
        super().__init__(verbose)
        self.total_timesteps = max(1, int(total_timesteps))
        self.start = float(start)
        self.end = float(end)
        self.update_every = max(1, int(update_every))
        self._last_update = -self.update_every

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_update >= self.update_every:
            progress = min(1.0, self.num_timesteps / self.total_timesteps)
            difficulty = self.start + progress * (self.end - self.start)
            self.training_env.env_method("set_human_difficulty", difficulty)
            self.logger.record("rollout/human_difficulty", difficulty)
            self._last_update = self.num_timesteps
        return True

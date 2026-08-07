"""Configuration for the human-aware PIPER pick-and-place task.

Units are metres, seconds, and metres/second.  The presets intentionally keep
the training (ID) and out-of-distribution evaluation distributions separate.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

from .config import EnvConfig, PROJECT_DIR


HUMAN_MODEL_PATH = PROJECT_DIR / "agilex_piper" / "piper_human_task.xml"
TRAJECTORY_TYPES = (
    "cross_workspace", "reach_object", "reach_target",
    "pause_and_continue", "reverse",
)

# The historical ``fixed`` preset deliberately remains height-randomized for
# reproducibility of already-published results.  New exact-fixed experiments
# must use this named profile rather than changing the historical preset.
FIXED_EXACT_H036_V1 = "fixed_exact_h036_v1"
FIXED_EXACT_SHIFTED_H036_V1 = "fixed_exact_shifted_h036_v1"


@dataclass
class HumanMotionConfig:
    """Distribution and physical limits for one sampled human-arm motion."""

    trajectory_types: Tuple[str, ...] = TRAJECTORY_TYPES
    appearance_time_range: Tuple[float, float] = (0.6, 2.2)
    speed_range: Tuple[float, float] = (0.16, 0.32)
    start_sides: Tuple[int, ...] = (-1, 1)
    end_position_jitter: float = 0.05
    closest_task_approach: Tuple[float, float] = (0.04, 0.14)
    pause_probability: float = 0.25
    pause_duration_range: Tuple[float, float] = (0.35, 1.1)
    reverse_probability: float = 0.15
    max_hand_speed: float = 0.40
    parked_y: float = 0.62
    hand_height: Tuple[float, float] = (0.30, 0.42)


@dataclass
class HumanSafetyConfig:
    """Reward, termination, and event thresholds for human interaction."""

    safe_separation_distance: float = 0.12
    near_miss_distance: float = 0.06
    proximity_penalty_weight: float = 2.0
    human_collision_penalty: float = 100.0
    terminate_on_human_collision: bool = True
    time_penalty: float = 0.0


@dataclass
class HumanAwareEnvConfig(EnvConfig):
    """Top-level config for :class:`PiperHumanAwarePickPlaceEnv`."""

    model_path: str = str(HUMAN_MODEL_PATH)
    human_enabled: bool = True
    human_episode_probability: float = 1.0
    include_human_state: bool = True
    human_position_noise: float = 0.0
    human_velocity_noise: float = 0.0
    human_observation_latency_steps: int = 0
    human_difficulty: float = 1.0
    human_distribution: str = "training_id"
    human_motion: HumanMotionConfig = field(default_factory=HumanMotionConfig)
    human_safety: HumanSafetyConfig = field(default_factory=HumanSafetyConfig)

    @staticmethod
    def _apply_historical_fixed_crossing(cfg: "HumanAwareEnvConfig") -> None:
        """Apply the exact historical Fixed ID configuration in one place."""
        cfg.human_motion.trajectory_types = ("cross_workspace",)
        cfg.human_motion.appearance_time_range = (1.0, 1.0)
        cfg.human_motion.speed_range = (0.22, 0.22)
        cfg.human_motion.start_sides = (1,)
        cfg.human_motion.end_position_jitter = 0.0
        cfg.human_motion.closest_task_approach = (0.08, 0.08)
        cfg.human_motion.pause_probability = 0.0
        cfg.human_motion.reverse_probability = 0.0

    @classmethod
    def from_preset(cls, name: str, difficulty: float = 1.0) -> "HumanAwareEnvConfig":
        """Construct a documented human distribution preset."""
        name = name.lower().replace("-", "_")
        cfg = cls()
        if name == "no_human":
            cfg.human_enabled = False
            cfg.human_episode_probability = 0.0
            cfg.human_distribution = "no_human"
        elif name == "fixed":
            cfg._apply_historical_fixed_crossing(cfg)
            cfg.human_distribution = "fixed_id"
        elif name == FIXED_EXACT_H036_V1:
            cfg._apply_historical_fixed_crossing(cfg)
            cfg.human_motion.hand_height = (0.36, 0.36)
            cfg.human_distribution = FIXED_EXACT_H036_V1
        elif name == FIXED_EXACT_SHIFTED_H036_V1:
            cfg._apply_historical_fixed_crossing(cfg)
            # Predeclared holdout: 0.2 s earlier, 0.02 m/s faster, and 1 cm
            # closer than the exact training crossing. It is never tuned.
            cfg.human_motion.appearance_time_range = (0.8, 0.8)
            cfg.human_motion.speed_range = (0.24, 0.24)
            cfg.human_motion.closest_task_approach = (0.07, 0.07)
            cfg.human_motion.hand_height = (0.36, 0.36)
            cfg.human_distribution = FIXED_EXACT_SHIFTED_H036_V1
        elif name in ("randomized", "evaluation_id"):
            cfg.human_distribution = "evaluation_id" if name == "evaluation_id" else "training_id"
        elif name == "curriculum":
            cfg.human_distribution = "curriculum_id"
            cfg.set_human_difficulty(difficulty)
        elif name == "evaluation_ood":
            cfg.human_motion.trajectory_types = (
                "cross_workspace", "pause_and_continue", "reverse")
            cfg.human_motion.appearance_time_range = (0.0, 4.0)
            cfg.human_motion.speed_range = (0.28, 0.40)
            cfg.human_motion.start_sides = (-1,)
            cfg.human_motion.closest_task_approach = (0.01, 0.08)
            cfg.human_motion.pause_probability = 0.55
            cfg.human_motion.pause_duration_range = (1.0, 2.5)
            cfg.human_motion.reverse_probability = 0.50
            cfg.human_observation_latency_steps = 3
            cfg.human_distribution = "evaluation_ood"
        else:
            raise ValueError(f"unknown human preset {name!r}")
        return cfg

    def set_human_difficulty(self, difficulty: float) -> None:
        """Update only the ID curriculum distribution, with difficulty in [0, 1]."""
        d = float(max(0.0, min(1.0, difficulty)))
        self.human_difficulty = d
        self.human_episode_probability = 0.25 + 0.75 * d
        self.human_motion.speed_range = (0.12 + 0.04 * d, 0.18 + 0.14 * d)
        self.human_motion.appearance_time_range = (1.8 - 1.2 * d, 3.0 - 0.8 * d)
        self.human_motion.closest_task_approach = (0.12 - 0.08 * d, 0.18 - 0.04 * d)
        self.human_motion.pause_probability = 0.05 + 0.20 * d
        self.human_motion.reverse_probability = 0.02 + 0.13 * d

    def eval_variant(self) -> "HumanAwareEnvConfig":
        cfg = copy.deepcopy(self)
        cfg.curriculum = False
        cfg.p_start_reaching = 1.0
        cfg.p_start_near_object = 0.0
        cfg.p_start_grasped = 0.0
        cfg.p_start_over_target = 0.0
        return cfg

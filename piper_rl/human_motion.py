"""Seeded, minimum-jerk trajectories for the simplified human arm."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Sequence

import numpy as np

from .human_config import HumanMotionConfig, TRAJECTORY_TYPES


@dataclass(frozen=True)
class HumanArmState:
    shoulder_position: np.ndarray
    elbow_position: np.ndarray
    hand_position: np.ndarray
    elbow_velocity: np.ndarray
    hand_velocity: np.ndarray
    human_present: bool
    phase: str


def _minimum_jerk(u: float) -> tuple[float, float]:
    u = float(np.clip(u, 0.0, 1.0))
    return 10 * u**3 - 15 * u**4 + 6 * u**5, 30 * u**2 - 60 * u**3 + 30 * u**4


class HumanArmTrajectory:
    """One fully sampled arm motion, queryable at arbitrary simulation time."""

    def __init__(self, trajectory_type: str, appearance_time: float,
                 waypoints: Sequence[np.ndarray], durations: Sequence[float],
                 phases: Sequence[str], start_side: int, speed: float,
                 moving_shoulder: bool = False):
        self.trajectory_type = trajectory_type
        self.appearance_time = float(appearance_time)
        self.waypoints = np.asarray(waypoints, dtype=np.float64)
        self.durations = np.asarray(durations, dtype=np.float64)
        self.phases = tuple(phases)
        self.start_side = int(start_side)
        self.speed = float(speed)
        self.moving_shoulder = bool(moving_shoulder)
        self.segment_ends = self.appearance_time + np.cumsum(self.durations)
        self.end_time = float(self.segment_ends[-1])

    def _shoulder(self, hand: np.ndarray, hand_vel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.moving_shoulder:
            shoulder = np.array([0.08, np.clip(1.35 * hand[1], -0.68, 0.68), 0.54])
            active = abs(1.35 * hand[1]) < 0.68
            velocity = np.array([0.0, 1.35 * hand_vel[1] if active else 0.0, 0.0])
        else:
            shoulder = np.array([0.08, self.start_side * 0.62, 0.54])
            velocity = np.zeros(3)
        return shoulder, velocity

    def state(self, simulation_time: float) -> HumanArmState:
        t = float(simulation_time)
        if t < self.appearance_time:
            hand, velocity, phase, present = self.waypoints[0], np.zeros(3), "parked", False
        elif t >= self.end_time:
            hand, velocity, phase, present = self.waypoints[-1], np.zeros(3), "complete", False
        else:
            i = int(np.searchsorted(self.segment_ends, t, side="right"))
            start_t = self.appearance_time if i == 0 else self.segment_ends[i - 1]
            duration = self.durations[i]
            s, ds_du = _minimum_jerk((t - start_t) / duration)
            delta = self.waypoints[i + 1] - self.waypoints[i]
            hand = self.waypoints[i] + s * delta
            velocity = ds_du * delta / duration
            phase, present = self.phases[i], True

        shoulder, shoulder_vel = self._shoulder(hand, velocity)
        bend = np.array([0.03, self.start_side * 0.055, 0.09])
        elbow = shoulder + 0.52 * (hand - shoulder) + bend
        elbow_vel = shoulder_vel + 0.52 * (velocity - shoulder_vel)
        return HumanArmState(
            shoulder.copy(), elbow, hand.copy(), elbow_vel, velocity.copy(),
            bool(present), phase)


class HumanTrajectoryGenerator:
    """Samples deterministic trajectories exclusively from a supplied RNG."""

    def __init__(self, config: HumanMotionConfig):
        self.config = config

    def sample(self, rng: np.random.Generator, object_position: np.ndarray,
               target_position: np.ndarray, trajectory_type: str | None = None,
               force_present: bool = True) -> HumanArmTrajectory:
        c = self.config
        kind = trajectory_type or str(rng.choice(c.trajectory_types))
        if kind not in TRAJECTORY_TYPES:
            raise ValueError(f"unknown human trajectory type {kind!r}")
        if trajectory_type is None and kind != "reverse" and rng.random() < c.reverse_probability:
            kind = "reverse"
        side = int(rng.choice(c.start_sides))
        appearance = float(rng.uniform(*c.appearance_time_range)) if force_present else 1e9
        speed = float(rng.uniform(*c.speed_range))
        z = float(rng.uniform(*c.hand_height))
        jitter = lambda: rng.uniform(-c.end_position_jitter, c.end_position_jitter, 2)
        start = np.array([0.34, side * c.parked_y, z])
        start[:2] += jitter()
        closest = float(rng.uniform(*c.closest_task_approach))
        moving_shoulder = kind in ("cross_workspace", "pause_and_continue")

        if kind == "cross_workspace":
            middle = np.array([0.36, side * closest, 0.34])
            end = np.array([0.36, -side * c.parked_y, z])
            end[:2] += jitter()
            points, phases = [start, middle, end], ["entering", "crossing"]
        elif kind in ("reach_object", "reach_target"):
            task = np.asarray(object_position if kind == "reach_object" else target_position)
            near = np.array([task[0], task[1] + side * closest, max(task[2] + 0.12, 0.31)])
            reach = np.array([task[0], task[1], max(task[2] + 0.055, 0.285)])
            points = [start, near, reach, near.copy(), start.copy()]
            phases = ["entering", "reaching", "withdrawing", "exiting"]
        elif kind == "pause_and_continue":
            mid = np.array([0.36, side * closest, 0.34])
            end = np.array([0.36, -side * c.parked_y, z])
            points = [start, mid, mid.copy(), end]
            phases = ["entering", "paused", "continuing"]
        else:  # reverse
            inner = np.array([0.36, side * closest, 0.33])
            points = [start, inner, start.copy()]
            phases = ["entering", "reversing"]

        # Optional pause in ordinary trajectories, inserted at the midpoint.
        pause_duration = None
        if kind not in ("pause_and_continue",) and rng.random() < c.pause_probability:
            insert = max(1, len(points) // 2)
            points.insert(insert + 1, points[insert].copy())
            phases.insert(insert, "paused")
            pause_duration = float(rng.uniform(*c.pause_duration_range))

        durations = []
        for i, (a, b) in enumerate(zip(points[:-1], points[1:])):
            dist = float(np.linalg.norm(b - a))
            if dist < 1e-9:
                duration = pause_duration or float(rng.uniform(*c.pause_duration_range))
            else:
                # Minimum jerk peaks at 1.875 times average speed.
                duration = max(0.2, 1.875 * dist / max(speed, 1e-6))
            durations.append(duration)
        return HumanArmTrajectory(kind, appearance, points, durations, phases,
                                  side, speed, moving_shoulder)


def trajectory_fingerprint(trajectory: HumanArmTrajectory,
                           config: HumanMotionConfig) -> dict:
    """Serialize every trajectory and motion-profile parameter reproducibly."""
    motion_config = {
        "trajectory_types": list(config.trajectory_types),
        "appearance_time_range_s": list(config.appearance_time_range),
        "speed_range_m_s": list(config.speed_range),
        "start_sides": list(config.start_sides),
        "end_position_jitter_m": config.end_position_jitter,
        "closest_task_approach_m": list(config.closest_task_approach),
        "pause_probability": config.pause_probability,
        "pause_duration_range_s": list(config.pause_duration_range),
        "reverse_probability": config.reverse_probability,
        "max_hand_speed_m_s": config.max_hand_speed,
        "parked_y_m": config.parked_y,
        "hand_height_m": list(config.hand_height),
    }
    return {
        "schema": "piper_human_trajectory_fingerprint_v1",
        "motion_config": motion_config,
        "trajectory": {
            "type": trajectory.trajectory_type,
            "appearance_time_s": trajectory.appearance_time,
            "start_side": trajectory.start_side,
            "speed_m_s": trajectory.speed,
            "moving_shoulder": trajectory.moving_shoulder,
            "waypoints_m": np.asarray(trajectory.waypoints, dtype=float).tolist(),
            "durations_s": np.asarray(trajectory.durations, dtype=float).tolist(),
            "phases": list(trajectory.phases),
        },
    }


def trajectory_fingerprint_sha256(fingerprint: dict) -> str:
    """Return the stable SHA-256 identifier for a serialized fingerprint."""
    payload = json.dumps(fingerprint, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

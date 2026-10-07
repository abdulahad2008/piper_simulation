"""Blade-hazard metrics for a rigid sharp tool near a human.

The tool is described by a *blade frame*: an edge segment from heel to tip
and a unit cutting direction ``c`` that points out of the sharp edge, away
from the spine. The human is a list of capsules (segment + radius), which is
how the piper_simulation human arm and the Human-Robot Gym animations are
both already represented.

Four per-step quantities are defined (all in SI units):

``edge_distance``     signed distance from the edge segment to the nearest
                      human capsule surface (m). Negative means the edge is
                      inside the capsule, i.e. a cut.
``alignment``         cos of the angle between the cutting direction and the
                      edge-to-human direction, clipped to [0, 1]. 1 means the
                      sharp side faces the person squarely, 0 means the spine
                      or flat faces them.
``closing_speed``     component of the edge's velocity relative to the nearest
                      human point, taken along the edge-to-human direction and
                      clipped at 0 (m/s). Only approach counts, and only while
                      the sharp side faces the person (alignment > 0).
``hazard``            dimensionless instantaneous hazard
                      h = alignment * (1 + closing_speed / v0) * g(edge_distance)
                      with g(d) = clip(1 - d / d0, 0, 1). It is zero when the
                      edge is farther than d0 or points away, and grows with
                      proximity, alignment and closing speed.

``exposure`` is the running integral of ``hazard`` over an episode (seconds).
A robot-link collision rate cannot see any of this: a gripper can keep
every link 10 cm from the person while the knife it holds points at their
forearm at 0.3 m/s.

Defaults d0 = 0.15 m and v0 = 0.25 m/s are stated in the paper and should
be pre-registered, not tuned on test data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .geometry import segment_capsule_distance, unit


@dataclass
class Capsule:
    a: np.ndarray            # world position of one end of the axis
    b: np.ndarray            # world position of the other end
    radius: float
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))  # linear velocity of the axis (m/s)
    name: str = ""


@dataclass
class BladeFrame:
    heel: np.ndarray         # world position of the edge start
    tip: np.ndarray          # world position of the edge end
    cut_dir: np.ndarray      # unit vector out of the sharp edge
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))  # linear velocity of the edge midpoint (m/s)


@dataclass
class HazardSample:
    edge_distance: float
    alignment: float
    closing_speed: float
    hazard: float
    nearest: str
    cp_edge: np.ndarray
    cp_human: np.ndarray


def blade_hazard(blade: BladeFrame, humans: Sequence[Capsule],
                 d0: float = 0.15, v0: float = 0.25) -> HazardSample:
    """Instantaneous hazard of one blade against the nearest of several capsules."""
    if not humans:
        return HazardSample(np.inf, 0.0, 0.0, 0.0, "", blade.heel.copy(), blade.heel.copy())

    best = None
    for cap in humans:
        d, s, t, cp, cq = segment_capsule_distance(blade.heel, blade.tip, cap.a, cap.b, cap.radius)
        if best is None or d < best[0]:
            best = (d, cp, cq, cap)
    d, cp, cq, cap = best

    to_human = unit(cq - cp)
    c = unit(np.asarray(blade.cut_dir, float))
    alignment = float(np.clip(c @ to_human, 0.0, 1.0)) if np.linalg.norm(cq - cp) > 1e-9 else 1.0

    v_rel = np.asarray(blade.vel, float) - np.asarray(cap.vel, float)
    closing = float(max(0.0, v_rel @ to_human)) if alignment > 0.0 else 0.0

    g = float(np.clip(1.0 - d / d0, 0.0, 1.0))
    h = alignment * (1.0 + closing / v0) * g
    return HazardSample(float(d), alignment, closing, float(h), cap.name, cp, cq)


class ExposureAccumulator:
    """Integrates hazard over an episode and tracks the worst step."""

    def __init__(self, dt: float):
        self.dt = float(dt)
        self.reset()

    def reset(self):
        self.exposure = 0.0
        self.min_edge_distance = np.inf
        self.max_hazard = 0.0
        self.max_closing_speed = 0.0
        self.steps_edge_facing = 0
        self.steps = 0
        self.cut = False

    def update(self, s: HazardSample):
        self.steps += 1
        self.exposure += s.hazard * self.dt
        self.min_edge_distance = min(self.min_edge_distance, s.edge_distance)
        self.max_hazard = max(self.max_hazard, s.hazard)
        self.max_closing_speed = max(self.max_closing_speed, s.closing_speed)
        if s.alignment > 0.5 and s.edge_distance < 0.15:
            self.steps_edge_facing += 1
        if s.edge_distance <= 0.0:
            self.cut = True

    def summary(self) -> dict:
        return dict(
            exposure=self.exposure,
            min_edge_distance=self.min_edge_distance,
            max_hazard=self.max_hazard,
            max_closing_speed=self.max_closing_speed,
            frac_edge_facing=(self.steps_edge_facing / self.steps) if self.steps else 0.0,
            cut=self.cut,
        )

"""Segment geometry used by the blade-hazard metrics.

Everything is plain NumPy so it can be unit-tested without a simulator and
called from a MuJoCo step at 20 Hz without noticeable cost.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-12


def closest_points_segments(p0: np.ndarray, p1: np.ndarray,
                            q0: np.ndarray, q1: np.ndarray):
    """Closest points between segments P=[p0,p1] and Q=[q0,q1].

    Returns (s, t, cp, cq): parameters in [0,1] along each segment and the
    two closest points. Standard Ericson (Real-Time Collision Detection,
    section 5.1.9) formulation, robust to degenerate (point) segments.
    """
    d1 = p1 - p0
    d2 = q1 - q0
    r = p0 - q0
    a = float(d1 @ d1)
    e = float(d2 @ d2)
    f = float(d2 @ r)

    if a <= _EPS and e <= _EPS:
        s = t = 0.0
    elif a <= _EPS:
        s = 0.0
        t = float(np.clip(f / e, 0.0, 1.0))
    else:
        c = float(d1 @ r)
        if e <= _EPS:
            t = 0.0
            s = float(np.clip(-c / a, 0.0, 1.0))
        else:
            b = float(d1 @ d2)
            denom = a * e - b * b
            s = float(np.clip((b * f - c * e) / denom, 0.0, 1.0)) if denom > _EPS else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = float(np.clip(-c / a, 0.0, 1.0))
            elif t > 1.0:
                t = 1.0
                s = float(np.clip((b - c) / a, 0.0, 1.0))
    cp = p0 + s * d1
    cq = q0 + t * d2
    return s, t, cp, cq


def segment_capsule_distance(p0, p1, q0, q1, radius: float):
    """Signed distance from segment [p0,p1] to a capsule with axis [q0,q1].

    Negative values mean penetration. Also returns the closest points.
    """
    s, t, cp, cq = closest_points_segments(np.asarray(p0, float), np.asarray(p1, float),
                                           np.asarray(q0, float), np.asarray(q1, float))
    return float(np.linalg.norm(cp - cq)) - radius, s, t, cp, cq


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > _EPS else np.zeros_like(v)

"""Read blade frames and human capsules straight out of a MuJoCo model.

Conventions (see assets/knife.xml):
  * a site named ``<prefix>blade_heel`` and ``<prefix>blade_tip`` mark the edge,
  * a site named ``<prefix>blade_frame`` whose local +Z axis is the cutting
    direction (out of the sharp edge),
  * human capsules are MuJoCo capsule geoms whose names start with
    ``human_prefix`` (default ``human_``). Spheres are accepted as zero-length
    capsules.
"""
from __future__ import annotations

from typing import List

import mujoco
import numpy as np

from .metrics import BladeFrame, Capsule


def _site_id(model, name):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if sid < 0:
        raise KeyError(f"site '{name}' not found in model")
    return sid


def _site_linvel(model, data, sid) -> np.ndarray:
    v = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, sid, v, 0)
    return v[3:6].copy()  # mj_objectVelocity returns [ang, lin]


def read_blade(model, data, prefix: str = "") -> BladeFrame:
    heel = _site_id(model, prefix + "blade_heel")
    tip = _site_id(model, prefix + "blade_tip")
    frame = _site_id(model, prefix + "blade_frame")
    cut_dir = data.site_xmat[frame].reshape(3, 3)[:, 2]
    vel = 0.5 * (_site_linvel(model, data, heel) + _site_linvel(model, data, tip))
    return BladeFrame(data.site_xpos[heel].copy(), data.site_xpos[tip].copy(), cut_dir.copy(), vel)


class FiniteDifferenceVelocity:
    """Replace MuJoCo velocities with (x_t - x_{t-1}) / dt.

    MuJoCo reports zero velocity for mocap-driven bodies, and kinematically
    driven humans (as in piper_simulation and Human-Robot Gym) are mocap
    or qpos-set, so the default in the wrapper is to difference positions.
    """

    def __init__(self, dt: float):
        self.dt = float(dt)
        self.reset()

    def reset(self):
        self._prev_blade = None
        self._prev_caps = {}

    def apply(self, blade: BladeFrame, caps: List[Capsule]):
        mid = 0.5 * (blade.heel + blade.tip)
        if self._prev_blade is not None:
            blade.vel = (mid - self._prev_blade) / self.dt
        else:
            blade.vel = np.zeros(3)
        self._prev_blade = mid
        for c in caps:
            centre = 0.5 * (c.a + c.b)
            prev = self._prev_caps.get(c.name)
            c.vel = (centre - prev) / self.dt if prev is not None else np.zeros(3)
            self._prev_caps[c.name] = centre
        return blade, caps


def read_human_capsules(model, data, human_prefix: str = "human_") -> List[Capsule]:
    caps: List[Capsule] = []
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if not name.startswith(human_prefix):
            continue
        gtype = model.geom_type[gid]
        pos = data.geom_xpos[gid]
        mat = data.geom_xmat[gid].reshape(3, 3)
        r = float(model.geom_size[gid][0])
        if gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
            half = float(model.geom_size[gid][1])
            axis = mat[:, 2]
            a, b = pos - half * axis, pos + half * axis
        elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
            a = b = pos.copy()
        else:
            continue
        v = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_GEOM, gid, v, 0)
        caps.append(Capsule(np.array(a), np.array(b), r, v[3:6].copy(), name))
    return caps

"""Gymnasium wrapper that adds blade-hazard metrics and a cost signal.

Works with any env whose ``unwrapped`` exposes ``model`` and ``data`` (MuJoCo)
and a ``dt`` attribute or ``control_dt``. Per step it writes

    info["hazard"]      instantaneous hazard h_t
    info["cost"]        cost used by constrained RL (default: h_t, or 1.0 on a cut)
    info["edge_distance"], info["alignment"], info["closing_speed"]

and on episode end ``info["hazard_summary"]`` with the exposure integral and
extremes. With ``penalty_weight > 0`` the hazard is also subtracted from the
reward, which gives the "hazard-shaped reward" baseline. Leave it at 0 and
read ``info["cost"]`` for Lagrangian / CPO style training (Safety-Gymnasium
and OmniSafe both consume this key).

``terminate_on_cut`` ends the episode when the edge penetrates a human
capsule, mirroring the collision termination already used in piper_simulation.
"""
from __future__ import annotations

import gymnasium as gym

from .metrics import ExposureAccumulator, blade_hazard
from .mujoco_bridge import FiniteDifferenceVelocity, read_blade, read_human_capsules


class BladeHazardWrapper(gym.Wrapper):
    def __init__(self, env, blade_prefix: str = "", human_prefix: str = "human_",
                 d0: float = 0.15, v0: float = 0.25,
                 penalty_weight: float = 0.0, cut_cost: float = 1.0,
                 terminate_on_cut: bool = True, dt: float | None = None,
                 fd_velocity: bool = True):
        super().__init__(env)
        self.blade_prefix = blade_prefix
        self.human_prefix = human_prefix
        self.d0, self.v0 = d0, v0
        self.penalty_weight = penalty_weight
        self.cut_cost = cut_cost
        self.terminate_on_cut = terminate_on_cut
        u = env.unwrapped
        if dt is None:
            dt = getattr(u, "control_dt", None) or getattr(u, "dt", None)
            if dt is None:
                raise ValueError("pass dt explicitly; env exposes neither control_dt nor dt")
        self.acc = ExposureAccumulator(dt)
        self.fd = FiniteDifferenceVelocity(dt) if fd_velocity else None

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self.acc.reset()
        if self.fd is not None:
            self.fd.reset()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        u = self.env.unwrapped
        blade = read_blade(u.model, u.data, self.blade_prefix)
        humans = read_human_capsules(u.model, u.data, self.human_prefix)
        if self.fd is not None:
            blade, humans = self.fd.apply(blade, humans)
        s = blade_hazard(blade, humans, self.d0, self.v0)
        self.acc.update(s)

        cost = self.cut_cost if s.edge_distance <= 0.0 else s.hazard
        info.update(hazard=s.hazard, cost=cost, edge_distance=s.edge_distance,
                    alignment=s.alignment, closing_speed=s.closing_speed,
                    nearest_human_part=s.nearest)
        if self.penalty_weight > 0.0:
            reward = reward - self.penalty_weight * cost
        if self.terminate_on_cut and s.edge_distance <= 0.0:
            terminated = True
            info["cut"] = True
        if terminated or truncated:
            info["hazard_summary"] = self.acc.summary()
        return obs, reward, terminated, truncated, info

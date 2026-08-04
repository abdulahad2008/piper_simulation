"""``RobotInterface`` backed by the MuJoCo environment.

This exists so the *policy runner* you will eventually point at hardware can be
written and debugged now, against exactly the same API. If a runner works with
``SimBackend`` and fails with ``PiperHardware``, the difference is hardware, not
your code.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import mujoco

from ..config import EnvConfig, RobotLimits
from ..piper_env import PiperPickPlaceEnv
from .interface import RobotInterface, RobotState


class SimBackend(RobotInterface):
    """Drives a :class:`PiperPickPlaceEnv` through the hardware API."""

    def __init__(self, cfg: Optional[EnvConfig] = None, **kw):
        cfg = cfg or EnvConfig()
        super().__init__(limits=cfg.limits, control_hz=cfg.control_hz,
                         action_smoothing=cfg.action_smoothing,
                         veto_on_violation=False, **kw)
        self.cfg = cfg
        self.env: PiperPickPlaceEnv | None = None
        self._enabled = False

    # ------------------------------------------------------------ lifecycle
    def connect(self) -> None:
        self.env = PiperPickPlaceEnv(self.cfg)
        self.env.reset(seed=0)
        self._init_safety(self.env.arm_qlim)

    def disconnect(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    def emergency_stop(self) -> None:
        self._enabled = False
        if self.env is not None:
            self.env.data.ctrl[:6] = self.env.data.qpos[self.env.arm_qadr]

    # -------------------------------------------------------------- sensing
    def get_state(self) -> RobotState:
        e = self.env
        d = e.data
        R = d.site_xmat[e.tcp_site].reshape(3, 3).copy()
        return RobotState(
            joint_pos=d.qpos[e.arm_qadr].copy(),
            joint_vel=d.qvel[e.arm_dofadr].copy(),
            gripper_pos=float(d.qpos[e.grip_qadr]),
            gripper_vel=float(d.qvel[e.grip_dofadr]),
            tcp_pos=d.site_xpos[e.tcp_site].copy(),
            tcp_rot=R,
            joint_torque=d.qfrc_actuator[e.arm_dofadr].copy(),
            timestamp=float(d.time),
            object_pos=e.obj_pos,
            object_rot=d.xmat[e.obj_body].reshape(3, 3).copy(),
            target_pos=e.target_pos.copy(),
        )

    def get_camera_frame(self, name: str = "forehead"):
        return self.env.render(name)

    # ------------------------------------------------------------- commands
    def _send_joint_command(self, q: np.ndarray, gripper: float) -> None:
        if not self._enabled:
            return
        e = self.env
        e.data.ctrl[:6] = q
        e.data.ctrl[e.grip_act] = gripper
        for _ in range(e._n_substeps):
            mujoco.mj_step(e.model, e.data)

    def move_to_home(self, duration: float = 4.0) -> None:
        e = self.env
        key_q = e.model.key_qpos[0][e.arm_qadr].copy()
        n = int(duration * self.control_hz)
        q0 = e.data.qpos[e.arm_qadr].copy()
        for i in range(n):
            a = (i + 1) / n
            self.send_joint_command(q0 * (1 - a) + key_q * a, 0.035)

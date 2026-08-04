"""Hardware abstraction layer for the AgileX PIPER.

**Nothing in this package commands a real robot.** ``PiperHardware`` below is a
deliberately unfinished adapter: every method that would move a physical joint
raises ``NotImplementedError`` with a note describing exactly what has to be
filled in from the AgileX ``piper_sdk``. That is the whole point -- the boundary
between "validated in simulation" and "touches a real arm" is one file, and it
is this one.

The abstraction
---------------
::

    RobotInterface                 what the policy runner talks to
      |-- SimBackend               wraps PiperPickPlaceEnv (validated here)
      `-- PiperHardware            wraps piper_sdk               (STUB)

A policy runner written against ``RobotInterface`` therefore does not change at
all between sim and hardware; only the backend swaps. The safety layer sits
*inside* the interface, so it is impossible to send a command that has not been
validated -- in sim or on hardware.

Sim-to-real gaps this interface is responsible for
--------------------------------------------------
===============================  =========================================
Gap                              Where it is handled
===============================  =========================================
control frequency                ``RobotInterface.control_hz``; the backend
                                 must interpolate to the servo's rate
joint + gripper limits           ``SafetyLayer`` (shared, identical values)
action smoothing                 ``RobotInterface.send_action`` (EMA filter,
                                 identical coefficient to the sim env)
observation latency              backend must timestamp and align samples
camera latency / noise           backend's ``get_camera_frame``
object pose estimation           NOT SOLVED -- see ``get_observation``; on
                                 hardware this needs a perception module
                                 (AprilTag, pose net, or a hand-labelled
                                 fixture) that the sim gets for free
gravity compensation             the MJCF sets ``gravcomp=1``; the real arm's
                                 servos do this in firmware, but the residual
                                 differs -- expect a static offset
===============================  =========================================
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..config import RobotLimits
from ..safety import SafetyLayer


@dataclass
class RobotState:
    """One synchronised snapshot of the robot."""

    joint_pos: np.ndarray               # (6,) rad
    joint_vel: np.ndarray               # (6,) rad/s
    gripper_pos: float                  # m, commanded finger joint
    gripper_vel: float = 0.0
    tcp_pos: Optional[np.ndarray] = None        # (3,) m, base frame
    tcp_rot: Optional[np.ndarray] = None        # (3,3)
    joint_torque: Optional[np.ndarray] = None   # (6,) Nm
    timestamp: float = 0.0
    #: Object / target pose in the base frame. In sim these are ground truth;
    #: on hardware they must come from a perception module.
    object_pos: Optional[np.ndarray] = None
    object_rot: Optional[np.ndarray] = None
    target_pos: Optional[np.ndarray] = None


class RobotInterface(abc.ABC):
    """Everything a policy runner is allowed to do to a PIPER."""

    def __init__(self, limits: Optional[RobotLimits] = None,
                 control_hz: float = 20.0,
                 action_smoothing: float = 0.45,
                 veto_on_violation: bool = True):
        self.limits = limits or RobotLimits()
        self.control_hz = control_hz
        self.dt = 1.0 / control_hz
        self.action_smoothing = action_smoothing
        self.veto_on_violation = veto_on_violation
        self._safety: SafetyLayer | None = None
        self._filtered_q: np.ndarray | None = None
        self._filtered_grip: float | None = None

    # ------------------------------------------------------------- lifecycle
    @abc.abstractmethod
    def connect(self) -> None: ...

    @abc.abstractmethod
    def disconnect(self) -> None: ...

    @abc.abstractmethod
    def enable(self) -> None:
        """Energise the joints. Must be a separate, explicit step."""

    @abc.abstractmethod
    def disable(self) -> None:
        """De-energise / brake. Must be safe to call at any time."""

    @abc.abstractmethod
    def emergency_stop(self) -> None:
        """Halt immediately. Must not depend on the control loop running."""

    # ------------------------------------------------------------- sensing
    @abc.abstractmethod
    def get_state(self) -> RobotState: ...

    def get_camera_frame(self, name: str = "forehead") -> Optional[np.ndarray]:
        """RGB frame from a named camera, or None if unavailable."""
        return None

    # ------------------------------------------------------------- commands
    @abc.abstractmethod
    def _send_joint_command(self, q: np.ndarray, gripper: float) -> None:
        """Backend-specific: put the validated command on the wire."""

    def send_joint_command(self, q_target, gripper_target):
        """Filter -> validate -> send. The only public write path.

        Applies the identical EMA smoothing and the identical
        :class:`SafetyLayer` used during training, so the command stream the
        hardware sees has the same spectral content the policy was trained with.
        """
        state = self.get_state()
        if self._safety is None:
            raise RuntimeError("call connect() before sending commands")
        if self._filtered_q is None:
            self._filtered_q = state.joint_pos.copy()
            self._filtered_grip = float(state.gripper_pos)

        a = self.action_smoothing
        self._filtered_q = (1 - a) * self._filtered_q + a * np.asarray(q_target, float)
        self._filtered_grip = ((1 - a) * self._filtered_grip
                               + a * float(gripper_target))

        q_safe, g_safe, report = self._safety.check(
            self._filtered_q, self._filtered_grip, state.joint_pos)

        state_report = self._safety.check_state(
            joint_torques=state.joint_torque)
        if state_report.vetoed:
            self.emergency_stop()
            raise RuntimeError(f"SAFETY STOP: {state_report.violations}")

        self._send_joint_command(q_safe, g_safe)
        return q_safe, g_safe, report

    # ------------------------------------------------------------- helpers
    def _init_safety(self, joint_limits) -> None:
        self._safety = SafetyLayer(
            joint_limits=joint_limits,
            limits=self.limits,
            control_dt=self.dt,
            veto_on_violation=self.veto_on_violation,
        )
        st = self.get_state()
        self._safety.reset(st.joint_pos, st.gripper_pos)
        self._filtered_q = st.joint_pos.copy()
        self._filtered_grip = float(st.gripper_pos)

    @abc.abstractmethod
    def move_to_home(self, duration: float = 4.0) -> None:
        """Slow, interpolated move to the home pose. Blocking."""

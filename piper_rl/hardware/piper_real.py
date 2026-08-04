"""AgileX PIPER hardware backend -- **DELIBERATE STUB, DOES NOT MOVE A ROBOT**.

Every method that would put current through a motor raises
``NotImplementedError`` carrying the exact ``piper_sdk`` call it needs. Nothing
in this repository imports this module by default; you have to reach for it.

Before you finish it, read ``docs/SIM2REAL.md``. In particular:

  * The policy trained here consumes an **object pose**. Simulation hands that
    over for free. On hardware it does not exist until you build a perception
    module -- an AprilTag on the cup, a fixture at a known offset, or a learned
    pose estimator. **A state-based policy cannot be deployed without it.** If
    you would rather not build one, retrain with ``obs_mode="rgb"`` (the env
    already supports it) so the forehead camera *is* the observation.

  * The PIPER's joint frames in this MJCF come from the official URDF, so joint
    ordering and sign conventions should match the SDK -- but VERIFY THAT
    JOINT-BY-JOINT with the arm disabled before enabling anything.

  * ``RobotLimits.hardware_profile()`` (see below) exists so first contact runs
    at roughly a third of the sim speed.

Reference (verify against the version you install):
    pip install piper_sdk
    from piper_sdk import C_PiperInterface
    piper = C_PiperInterface("can0"); piper.ConnectPort()
    piper.EnableArm(7)
    piper.JointCtrl(j1, ..., j6)         # units: 0.001 deg
    piper.GripperCtrl(angle, effort, code, set_zero)
    piper.GetArmJointMsgs() / GetArmGripperMsgs() / GetArmHighSpdInfoMsgs()
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..config import RobotLimits
from .interface import RobotInterface, RobotState

_NOT_WIRED = (
    "PiperHardware is a stub. Implement this against piper_sdk, then remove "
    "this guard. Do not skip the bring-up checklist in docs/SIM2REAL.md."
)


def conservative_limits() -> RobotLimits:
    """Motion limits for first contact with the physical arm.

    Roughly a third of the simulation limits everywhere. Slow enough that a
    human can reach the e-stop, and small enough that a mis-signed joint moves a
    few centimetres instead of slamming into the table. Raise these only after
    the bring-up checklist passes end to end.
    """
    lim = RobotLimits()
    lim.max_joint_vel = 0.30            # rad/s  (~17 deg/s)
    lim.max_joint_accel = 3.0           # rad/s^2
    lim.max_gripper_vel = 0.05          # m/s
    lim.max_tcp_speed = 0.12            # m/s
    lim.max_contact_force = 15.0        # N
    lim.max_joint_torque = 25.0         # Nm
    lim.tcp_table_clearance = 0.03      # keep 3 cm off the table until trusted
    return lim


class PiperHardware(RobotInterface):
    """Adapter to the real arm over CAN. **Not implemented on purpose.**"""

    def __init__(self, can_port: str = "can0",
                 limits: Optional[RobotLimits] = None,
                 control_hz: float = 20.0,
                 action_smoothing: float = 0.45,
                 dry_run: bool = True):
        super().__init__(limits=limits or conservative_limits(),
                         control_hz=control_hz,
                         action_smoothing=action_smoothing,
                         veto_on_violation=True)   # hardware: refuse, never clamp
        self.can_port = can_port
        #: When True, every command is validated and logged but never sent.
        #: Run the entire policy this way first and check the logged trajectory.
        self.dry_run = dry_run
        self._piper = None

    # ------------------------------------------------------------ lifecycle
    def connect(self) -> None:
        raise NotImplementedError(
            _NOT_WIRED + "\n"
            "    from piper_sdk import C_PiperInterface\n"
            "    self._piper = C_PiperInterface(self.can_port)\n"
            "    self._piper.ConnectPort()\n"
            "    self._init_safety(<(6,2) joint limits, radians, from the URDF>)")

    def disconnect(self) -> None:
        raise NotImplementedError(_NOT_WIRED + "  self._piper.DisconnectPort()")

    def enable(self) -> None:
        raise NotImplementedError(
            _NOT_WIRED + "\n"
            "    self._piper.EnableArm(7)   # then POLL until every joint reports\n"
            "    # enabled, with a timeout -- do NOT command motion before that.")

    def disable(self) -> None:
        raise NotImplementedError(_NOT_WIRED + "  self._piper.DisableArm(7)")

    def emergency_stop(self) -> None:
        raise NotImplementedError(
            _NOT_WIRED + "\n"
            "    Must work even if the control loop is wedged. Prefer a\n"
            "    hardware e-stop in series with the arm's supply; the software\n"
            "    path (DisableArm) is a backup, not the primary.")

    # -------------------------------------------------------------- sensing
    def get_state(self) -> RobotState:
        raise NotImplementedError(
            _NOT_WIRED + "\n"
            "    j = self._piper.GetArmJointMsgs()      # 0.001 deg -> rad\n"
            "    g = self._piper.GetArmGripperMsgs()\n"
            "    Differentiate joint positions for velocity (the SDK's velocity\n"
            "    field is noisy); filter it the same way you filtered the sim's.\n"
            "    object_pos/target_pos MUST come from a perception module --\n"
            "    they are not available from the arm.")

    def get_camera_frame(self, name: str = "forehead"):
        raise NotImplementedError(
            _NOT_WIRED + "\n"
            "    Grab from the physical forehead camera. Match the sim: 65 deg\n"
            "    vertical FoV, mounted 25 cm behind and 30 cm above the base\n"
            "    flange, tilted ~25 deg down. Resize to the training resolution\n"
            "    and reproduce the same latency budget you trained with.")

    # ------------------------------------------------------------- commands
    def _send_joint_command(self, q: np.ndarray, gripper: float) -> None:
        if self.dry_run:
            return
        raise NotImplementedError(
            _NOT_WIRED + "\n"
            "    self._piper.JointCtrl(*(np.degrees(q) * 1000).astype(int))\n"
            "    self._piper.GripperCtrl(int(gripper * 1e6), effort, 0x01, 0)\n"
            "    Check the SDK's units and gripper scaling on YOUR version.")

    def move_to_home(self, duration: float = 8.0) -> None:
        raise NotImplementedError(
            _NOT_WIRED + "\n"
            "    Interpolate from the measured pose to home over `duration`\n"
            "    seconds at self.control_hz, pushing every waypoint through\n"
            "    send_joint_command() so the safety layer sees it too.")

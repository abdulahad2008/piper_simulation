"""Safety layer: validates every command *before* it reaches an actuator.

The exact same object runs in simulation and (later) in front of the real PIPER.
Running it in sim from day one is the point: by the time you plug in hardware,
the layer has already vetoed millions of commands and you know its thresholds
are not so tight that they break the policy, nor so loose that they are useless.

Checks, in order:
  1. finite / NaN
  2. joint position limits (with a configurable safety margin)
  3. joint velocity limit  (max change per control tick)
  4. joint acceleration limit
  5. gripper travel + speed
  6. Cartesian workspace envelope of the resulting TCP pose
  7. Cartesian speed of the TCP

Each check either **clamps** the command (soft violation) or **vetoes** it
(hard violation -> hold position and raise). Which behaviour applies is set by
``veto_on_violation``; in simulation we clamp and report, on hardware you almost
certainly want to veto and stop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class SafetyReport:
    """What the safety layer did to one command."""

    ok: bool = True
    clamped: bool = False
    vetoed: bool = False
    violations: List[str] = field(default_factory=list)

    def add(self, msg: str, veto: bool = False) -> None:
        self.violations.append(msg)
        self.ok = False
        if veto:
            self.vetoed = True
        else:
            self.clamped = True


class SafetyLayer:
    """Command validator for the PIPER arm.

    Parameters
    ----------
    joint_limits:
        ``(6, 2)`` array of (low, high) joint position limits [rad].
    limits:
        A :class:`piper_rl.config.RobotLimits`.
    control_dt:
        Seconds between control ticks; converts velocity/accel limits into
        per-tick position deltas.
    margin:
        Fraction of each joint's range kept clear of the hard limit.
    veto_on_violation:
        ``False`` (sim default): clamp the command into the safe set.
        ``True``  (hardware default): refuse the command entirely.
    """

    def __init__(self, joint_limits, limits, control_dt: float,
                 margin: float = 0.02, veto_on_violation: bool = False,
                 forward_kinematics=None):
        self.qlim = np.asarray(joint_limits, dtype=float)
        span = self.qlim[:, 1] - self.qlim[:, 0]
        self.qlim_soft = np.stack(
            [self.qlim[:, 0] + margin * span, self.qlim[:, 1] - margin * span],
            axis=1)
        self.lim = limits
        self.dt = float(control_dt)
        self.veto = veto_on_violation
        self.fk = forward_kinematics       # callable(q) -> tcp position, optional

        self.max_dq = self.lim.max_joint_vel * self.dt
        self.max_ddq = self.lim.max_joint_accel * self.dt ** 2
        self.max_dgrip = self.lim.max_gripper_vel * self.dt

        self._prev_cmd: np.ndarray | None = None
        self._prev_dq: np.ndarray | None = None
        self._prev_grip: float | None = None
        self.n_clamped = 0
        self.n_vetoed = 0
        self.violation_counts: dict[str, int] = {}

    # ------------------------------------------------------------------ #
    def reset(self, q_now: np.ndarray, grip_now: float) -> None:
        self._prev_cmd = np.asarray(q_now, float).copy()
        self._prev_dq = np.zeros_like(self._prev_cmd)
        self._prev_grip = float(grip_now)
        # Per-episode counters, so `safety_clamped` in the episode summary means
        # "commands clamped in THIS episode" and not "since the process started".
        self.n_clamped = 0
        self.n_vetoed = 0
        self.violation_counts: dict[str, int] = {}

    # ------------------------------------------------------------------ #
    def check(self, q_cmd: np.ndarray, grip_cmd: float, q_now: np.ndarray):
        """Validate a joint-position + gripper command.

        Returns ``(q_safe, grip_safe, report)``.
        """
        rep = SafetyReport()
        q_cmd = np.asarray(q_cmd, dtype=float).copy()
        grip_cmd = float(grip_cmd)

        if self._prev_cmd is None:
            self.reset(q_now, grip_cmd)

        # 1 -- finite ---------------------------------------------------- #
        if not np.all(np.isfinite(q_cmd)) or not np.isfinite(grip_cmd):
            rep.add("non-finite command", veto=True)
            self.n_vetoed += 1
            return self._prev_cmd.copy(), self._prev_grip, rep

        # 2 -- joint position limits ------------------------------------- #
        clipped = np.clip(q_cmd, self.qlim_soft[:, 0], self.qlim_soft[:, 1])
        if np.any(np.abs(clipped - q_cmd) > 1e-9):
            bad = np.where(np.abs(clipped - q_cmd) > 1e-9)[0]
            rep.add(f"joint position limit on {bad.tolist()}", veto=self.veto)
            q_cmd = clipped

        # 3 -- joint velocity -------------------------------------------- #
        dq = q_cmd - self._prev_cmd
        over = np.abs(dq) > self.max_dq
        if np.any(over):
            rep.add(f"joint velocity limit on {np.where(over)[0].tolist()}",
                    veto=self.veto)
            dq = np.clip(dq, -self.max_dq, self.max_dq)

        # 4 -- joint acceleration ---------------------------------------- #
        ddq = dq - self._prev_dq
        over = np.abs(ddq) > self.max_ddq
        if np.any(over):
            rep.add(f"joint acceleration limit on {np.where(over)[0].tolist()}",
                    veto=self.veto)
            ddq = np.clip(ddq, -self.max_ddq, self.max_ddq)
            dq = self._prev_dq + ddq
        q_safe = self._prev_cmd + dq

        # 5 -- gripper ---------------------------------------------------- #
        g_lo, g_hi = self.lim.gripper_range
        g = float(np.clip(grip_cmd, g_lo, g_hi))
        if abs(g - grip_cmd) > 1e-9:
            rep.add("gripper travel limit", veto=self.veto)
        dg = np.clip(g - self._prev_grip, -self.max_dgrip, self.max_dgrip)
        if abs((g - self._prev_grip) - dg) > 1e-9:
            rep.add("gripper speed limit", veto=self.veto)
        g_safe = self._prev_grip + dg

        # 6/7 -- Cartesian envelope + speed ------------------------------- #
        # A Cartesian violation cannot be "clamped" component-wise without
        # re-solving IK, and it is exactly the class of error that damages
        # hardware (driving the TCP into the table, over-extending the arm).
        # So it always HOLDS POSITION: the previous command is re-issued.
        cartesian_block = False
        if self.fk is not None:
            p = np.asarray(self.fk(q_safe), float)
            radius = float(np.hypot(p[0], p[1]))
            azim = float(np.arctan2(p[1], p[0]))
            r_lo, r_hi = self.lim.ws_radius
            a_lo, a_hi = self.lim.ws_azimuth
            z_lo, z_hi = self.lim.ws_height
            z_floor = max(z_lo, self.lim.table_top + self.lim.tcp_table_clearance)
            if not (r_lo - 1e-6 <= radius <= r_hi + 1e-6):
                rep.add(f"TCP radius outside [{r_lo},{r_hi}] ({radius:.3f})",
                        veto=self.veto)
                cartesian_block = True
            if not (a_lo - 1e-6 <= azim <= a_hi + 1e-6):
                rep.add(f"TCP azimuth outside [{a_lo},{a_hi}] ({azim:.3f})",
                        veto=self.veto)
                cartesian_block = True
            if not (z_floor - 1e-6 <= p[2] <= z_hi + 1e-6):
                rep.add(f"TCP height outside [{z_floor:.3f},{z_hi}] ({p[2]:.3f})",
                        veto=self.veto)
                cartesian_block = True
            # TCP speed is RATE-LIMITED, not blocked. Freezing the arm whenever
            # it asks to move fast is both unphysical and catastrophic for the
            # policy: with max_step_dist = 0.03 m at 20 Hz the nominal command
            # speed is 0.6 m/s, so a hard block on a 0.4 m/s cap stops the arm
            # dead on almost every step. Scaling the joint delta back towards
            # the previous command is what a real motion controller does.
            p_prev = np.asarray(self.fk(self._prev_cmd), float)
            speed = float(np.linalg.norm(p - p_prev) / self.dt)
            if speed > self.lim.max_tcp_speed:
                rep.add(f"TCP speed > {self.lim.max_tcp_speed} m/s ({speed:.2f})",
                        veto=self.veto)
                scale = self.lim.max_tcp_speed / speed
                q_safe = self._prev_cmd + scale * (q_safe - self._prev_cmd)

        for v in rep.violations:                      # diagnostics, no numbers
            key = " ".join(v.split(" outside")[0].split(" on ")[0].split(" >")[0]
                           .split()[:3])
            self.violation_counts[key] = self.violation_counts.get(key, 0) + 1

        if rep.vetoed:
            self.n_vetoed += 1
            return self._prev_cmd.copy(), self._prev_grip, rep

        if cartesian_block:
            self.n_clamped += 1
            self._prev_dq = np.zeros_like(self._prev_cmd)
            self._prev_grip = g_safe
            return self._prev_cmd.copy(), g_safe, rep

        if rep.clamped:
            self.n_clamped += 1

        self._prev_dq = q_safe - self._prev_cmd
        self._prev_cmd = q_safe
        self._prev_grip = g_safe
        return q_safe, g_safe, rep

    # ------------------------------------------------------------------ #
    def check_state(self, contact_forces=None, joint_torques=None) -> SafetyReport:
        """Post-step state check (contact force / torque envelope)."""
        rep = SafetyReport()
        if contact_forces is not None and len(contact_forces):
            f = float(np.max(np.abs(contact_forces)))
            if f > self.lim.max_contact_force:
                rep.add(f"contact force {f:.1f} N > {self.lim.max_contact_force}",
                        veto=True)
        if joint_torques is not None and len(joint_torques):
            t = float(np.max(np.abs(joint_torques)))
            if t > self.lim.max_joint_torque:
                rep.add(f"joint torque {t:.1f} Nm > {self.lim.max_joint_torque}",
                        veto=True)
        return rep

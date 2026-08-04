"""Damped-least-squares inverse kinematics for the PIPER arm.

Used by:
  * ``scripts/scripted_demo.py`` -- to prove the task is physically solvable and
    to generate demonstration trajectories,
  * ``piper_env.PiperPickPlaceEnv`` when ``action_mode="cartesian"`` -- to turn a
    Cartesian TCP delta into joint targets.

Why a *5-DoF* task instead of a full 6-DoF pose
-----------------------------------------------
The PIPER has 6 revolute joints but restrictive limits (joint2 in [0, pi],
joint3 in [-2.697, 0], joint5 in [-1.22, 1.22]). Asking for a fully specified
orientation (position + roll + pitch + yaw) is frequently infeasible and the
solver stalls in a local minimum tens of millimetres away -- this is exactly
what made the original ``validate_piper.py`` fail.

Instead we constrain:
  * TCP position                       (3 equations)
  * the gripper *approach axis*        (2 equations -- direction only)
leaving the roll about the approach axis free (1 null-space DoF). That is all a
top-down pick actually needs, and it converges reliably over the whole
workspace. An optional secondary objective aligns the finger-opening axis with
a preferred direction using the leftover null space.
"""

from __future__ import annotations

import numpy as np
import mujoco

# Joint names of the 6 arm DoF, in order.
ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 7))

# Gripper approach axis expressed in the ``tcp`` site frame. The site inherits
# link6's orientation and the fingers extend along link6 +z.
APPROACH_AXIS_LOCAL = np.array([0.0, 0.0, 1.0])
# Finger-opening ("pinch") axis in the same frame.
PINCH_AXIS_LOCAL = np.array([0.0, 1.0, 0.0])


class PiperIK:
    """Damped-least-squares IK on a scratch :class:`mujoco.MjData`.

    Parameters
    ----------
    model:
        Compiled PIPER model (must contain the ``tcp`` site and joint1..joint6).
    damping:
        Levenberg-Marquardt damping. Larger = slower but more stable near
        singularities.
    step:
        Fraction of the Newton step applied per iteration.
    """

    def __init__(self, model: mujoco.MjModel, damping: float = 0.06,
                 step: float = 0.6):
        self.m = model
        self._d = mujoco.MjData(model)
        self.damping = damping
        self.step = step

        self.tcp_id = model.site("tcp").id
        self.qadr = np.array(
            [model.jnt_qposadr[model.joint(n).id] for n in ARM_JOINTS])
        self.dofadr = np.array(
            [model.jnt_dofadr[model.joint(n).id] for n in ARM_JOINTS])
        self.qlim = np.array(
            [model.jnt_range[model.joint(n).id] for n in ARM_JOINTS])
        self.q_mid = self.qlim.mean(axis=1)
        self.q_span = np.maximum(self.qlim[:, 1] - self.qlim[:, 0], 1e-6)

    # ------------------------------------------------------------------ utils
    def forward(self, q: np.ndarray, qpos_full: np.ndarray | None = None):
        """Return (tcp_pos, tcp_rotmat 3x3) for arm configuration ``q``."""
        if qpos_full is not None:
            self._d.qpos[:] = qpos_full
        self._d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self._d)
        mujoco.mj_comPos(self.m, self._d)
        return (self._d.site_xpos[self.tcp_id].copy(),
                self._d.site_xmat[self.tcp_id].reshape(3, 3).copy())

    # --------------------------------------------------------------- solving
    def solve(self,
              target_pos,
              approach_dir=(0.0, 0.0, -1.0),
              q_seed: np.ndarray | None = None,
              qpos_full: np.ndarray | None = None,
              iters: int = 220,
              pos_tol: float = 1e-3,
              ang_tol: float = np.deg2rad(2.0),
              pinch_dir=None):
        """Solve for joint angles placing the TCP at ``target_pos`` with the
        gripper approach axis pointing along ``approach_dir`` (world frame).

        Returns
        -------
        q : (6,) joint angles, clipped to limits
        pos_err : float, final Cartesian error [m]
        ang_err : float, final approach-axis error [rad]
        """
        target_pos = np.asarray(target_pos, dtype=float)
        a_des = np.asarray(approach_dir, dtype=float)
        a_des = a_des / (np.linalg.norm(a_des) + 1e-12)

        q = (self.q_mid.copy() if q_seed is None
             else np.clip(np.asarray(q_seed, float), self.qlim[:, 0], self.qlim[:, 1]))

        jacp = np.zeros((3, self.m.nv))
        jacr = np.zeros((3, self.m.nv))
        pos_err = ang_err = np.inf

        for _ in range(iters):
            if qpos_full is not None:
                self._d.qpos[:] = qpos_full
            self._d.qpos[self.qadr] = q
            mujoco.mj_kinematics(self.m, self._d)
            mujoco.mj_comPos(self.m, self._d)

            p = self._d.site_xpos[self.tcp_id]
            R = self._d.site_xmat[self.tcp_id].reshape(3, 3)
            a_cur = R @ APPROACH_AXIS_LOCAL

            e_pos = target_pos - p
            # Rotation that takes a_cur onto a_des (axis-angle, magnitude = angle).
            cross = np.cross(a_cur, a_des)
            s = np.linalg.norm(cross)
            c = float(np.clip(np.dot(a_cur, a_des), -1.0, 1.0))
            angle = np.arctan2(s, c)
            e_rot = cross / s * angle if s > 1e-9 else np.zeros(3)

            pos_err = float(np.linalg.norm(e_pos))
            ang_err = float(abs(angle))
            if pos_err < pos_tol and ang_err < ang_tol:
                break

            mujoco.mj_jacSite(self.m, self._d, jacp, jacr, self.tcp_id)
            J = np.vstack([jacp[:, self.dofadr], jacr[:, self.dofadr]])
            err = np.concatenate([e_pos, e_rot])

            # Adaptive damping: heavy while far away (stable, no wild steps),
            # light once close (so the residual actually goes to zero instead of
            # parking a few millimetres out -- the bug in the original solver).
            lam = self.damping * max(0.05, min(1.0, 10.0 * pos_err + ang_err))
            JJt = J @ J.T + lam ** 2 * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, err)

            # Null-space terms, projected so they never fight the main task, and
            # faded out as the task error shrinks.
            null_gain = min(1.0, 20.0 * pos_err + 2.0 * ang_err)
            if null_gain > 1e-3:
                N = np.eye(6) - J.T @ np.linalg.solve(JJt, J)
                dq_null = -0.3 * (q - self.q_mid) / self.q_span
                if pinch_dir is not None:
                    pd = np.asarray(pinch_dir, float)
                    pd = pd / (np.linalg.norm(pd) + 1e-12)
                    p_cur = R @ PINCH_AXIS_LOCAL
                    sgn = float(np.dot(np.cross(p_cur, pd), a_cur))
                    dq_null = dq_null + 1.0 * sgn * (J[3:].T @ a_cur)
                dq = dq + null_gain * (N @ dq_null)

            q = np.clip(q + self.step * dq, self.qlim[:, 0], self.qlim[:, 1])

        return q, pos_err, ang_err

    # --------------------------------------------------------------- helpers
    def reachable(self, target_pos, approach_dir=(0, 0, -1), **kw) -> bool:
        """True if :meth:`solve` converges within tolerance for this target."""
        _, pe, ae = self.solve(target_pos, approach_dir, **kw)
        return pe < 5e-3 and ae < np.deg2rad(5.0)

    def diff_ik(self,
                q: np.ndarray,
                dpos: np.ndarray,
                approach_dir=(0.0, 0.0, -1.0),
                qpos_full: np.ndarray | None = None,
                iters: int = 3,
                ori_gain: float = 0.6,
                max_ori_step: float = 0.10,
                ori_weight: float = 1.0):
        """Velocity-level ("differential") IK -- the cheap per-control-step path.

        Given the current arm configuration ``q`` and a desired *Cartesian TCP
        displacement* ``dpos``, return the joint configuration that realises it
        while pulling the approach axis back towards ``approach_dir``.

        This is what ``PiperPickPlaceEnv`` uses in ``action_mode="cartesian"``:
        2 Jacobian iterations per control step (~0.1 ms) instead of a full
        200-iteration solve, which keeps the env fast enough for RL.
        """
        a_des = np.asarray(approach_dir, float)
        a_des = a_des / (np.linalg.norm(a_des) + 1e-12)
        q = np.asarray(q, float).copy()

        if qpos_full is not None:
            self._d.qpos[:] = qpos_full
        self._d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self._d)
        mujoco.mj_comPos(self.m, self._d)
        p0 = self._d.site_xpos[self.tcp_id].copy()
        goal = p0 + np.asarray(dpos, float)

        jacp = np.zeros((3, self.m.nv))
        jacr = np.zeros((3, self.m.nv))
        for _ in range(iters):
            self._d.qpos[self.qadr] = q
            mujoco.mj_kinematics(self.m, self._d)
            mujoco.mj_comPos(self.m, self._d)
            p = self._d.site_xpos[self.tcp_id]
            R = self._d.site_xmat[self.tcp_id].reshape(3, 3)
            a_cur = R @ APPROACH_AXIS_LOCAL

            cross = np.cross(a_cur, a_des)
            s = np.linalg.norm(cross)
            angle = np.arctan2(s, float(np.clip(np.dot(a_cur, a_des), -1, 1)))
            e_rot = (cross / s * angle * ori_gain) if s > 1e-9 else np.zeros(3)
            # CRITICAL: bound the orientation correction per control step.
            #
            # The translation task is at most `max_step_dist` = 0.03 m, while the
            # raw orientation error can be 0.5 rad (the home pose is ~26 deg off
            # straight-down). Fed into the same least-squares, the rotational
            # rows demand joint deltas an order of magnitude larger than the
            # translational ones, saturate the safety layer's joint-velocity
            # limit every step, and leave the policy with essentially NO
            # translational authority -- a full +x and a full -x action produced
            # the same trajectory. Clipping to ~0.04 rad/step turns this into a
            # gentle servo that levels the gripper over ~10 steps while the
            # policy's translation command dominates.
            n_rot = float(np.linalg.norm(e_rot))
            if n_rot > max_ori_step:
                e_rot *= max_ori_step / n_rot

            mujoco.mj_jacSite(self.m, self._d, jacp, jacr, self.tcp_id)
            Jp = jacp[:, self.dofadr]            # 3 x 6, position
            Jr = jacr[:, self.dofadr]            # 3 x 6, orientation
            lam2 = self.damping ** 2

            # STRICT TASK PRIORITY: position first, orientation in its null space.
            #
            # Stacking the two tasks into one least-squares lets the orientation
            # servo trade position error for angular error. With a 6-joint arm, a
            # 3-DoF position task leaves a 3-DoF null space -- more than enough
            # for the 2-DoF approach-axis task -- so there is no reason to accept
            # that trade. Solving them jointly made a zero-translation action
            # drift the TCP ~11 cm over 3 seconds while the wrist levelled
            # itself, i.e. "do nothing" was not a reachable action.
            dq = Jp.T @ np.linalg.solve(Jp @ Jp.T + lam2 * np.eye(3), goal - p)

            if np.any(e_rot):
                # Damped null-space projector of the position task.
                N = np.eye(6) - Jp.T @ np.linalg.solve(
                    Jp @ Jp.T + lam2 * np.eye(3), Jp)
                dq_ori = Jr.T @ np.linalg.solve(
                    Jr @ Jr.T + lam2 * np.eye(3), e_rot)
                dq = dq + ori_weight * (N @ dq_ori)

            q = np.clip(q + dq, self.qlim[:, 0], self.qlim[:, 1])
        return q

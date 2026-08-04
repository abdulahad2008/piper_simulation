"""Scripted Piper pick-and-place using offline IK per waypoint.
Headless. If this places the cube on the green pad, the arm can do the task and
the RL env has a solvable problem. Renders the key frames to views/ too."""
import numpy as np
import mujoco

m = mujoco.MjModel.from_xml_path("piper_task.xml")
d = mujoco.MjData(m)
d_ik = mujoco.MjData(m)                      # scratch for IK, never disturbs d

tcp = m.site("tcp").id
gi = m.actuator("gripper").id
arm_dof = [m.jnt_dofadr[m.joint(f"joint{j}").id] for j in range(1, 7)]
arm_qadr = [m.jnt_qposadr[m.joint(f"joint{j}").id] for j in range(1, 7)]
arm_rng = np.array([m.jnt_range[m.joint(f"joint{j}").id] for j in range(1, 7)])
DOWN = np.array([1, 0, 0, 0, -1, 0, 0, 0, -1], float)


def ori_err(cur, tgt):
    cq = np.zeros(4); mujoco.mju_mat2Quat(cq, np.asarray(cur, float))
    tq = np.zeros(4); mujoco.mju_mat2Quat(tq, np.asarray(tgt, float))
    dq = np.zeros(4)
    mujoco.mju_mulQuat(dq, tq, np.array([cq[0], -cq[1], -cq[2], -cq[3]]))
    if dq[0] < 0:
        dq = -dq
    a = 2 * np.arccos(np.clip(dq[0], -1, 1))
    s = np.sqrt(max(0.0, 1 - dq[0] ** 2))
    return dq[1:] / s * a if s > 1e-6 else np.zeros(3)


def solve_ik(target_pos, seed, iters=300):
    """Joint angles that put the grasp point at target_pos, gripper pointing
    down. Seeded from `seed` for continuity between waypoints."""
    d_ik.qpos[:] = d.qpos
    q = seed.copy()
    for _ in range(iters):
        d_ik.qpos[arm_qadr] = q
        mujoco.mj_kinematics(m, d_ik)
        mujoco.mj_comPos(m, d_ik)
        pe = target_pos - d_ik.site_xpos[tcp]
        re = ori_err(d_ik.site_xmat[tcp], DOWN)
        err = np.concatenate([pe, re])
        jacp = np.zeros((3, m.nv)); jacr = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d_ik, jacp, jacr, tcp)
        J = np.vstack([jacp[:, arm_dof], jacr[:, arm_dof]])
        dq = J.T @ np.linalg.solve(J @ J.T + 0.04 ** 2 * np.eye(6), err)
        q = np.clip(q + 0.5 * dq, arm_rng[:, 0], arm_rng[:, 1])
    return q


def move_to(target_pos, grip, seed, settle=240):
    """Solve IK for the waypoint, then RAMP the joint command from the current
    pose to it, so the arm moves smoothly and doesn't fling the object."""
    q_target = solve_ik(target_pos, seed)
    q_start = d.qpos[arm_qadr].copy()
    for i in range(settle):
        a = min(1.0, (i + 1) / (settle * 0.6))   # ease in over the first 60%
        d.ctrl[:6] = q_start * (1 - a) + q_target * a
        d.ctrl[gi] = grip
        mujoco.mj_step(m, d)
    return q_target


mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d)
obj = d.body("obj").xpos[:2].copy()
dest = d.site("dest").xpos[:2].copy()
GRASP = 0.27           # grip the upper body, fingertips clear the table
OPEN, GRIP = 0.035, 0.004
seed = d.qpos[arm_qadr].copy()

seed = move_to([obj[0], obj[1], 0.40], OPEN, seed)      # above object
seed = move_to([obj[0], obj[1], GRASP], OPEN, seed)     # descend around it
seed = move_to([obj[0], obj[1], GRASP], GRIP, seed, 250)  # close
seed = move_to([obj[0], obj[1], 0.42], GRIP, seed)      # lift
z_lift = d.body("obj").xpos[2]
seed = move_to([dest[0], dest[1], 0.42], GRIP, seed)    # carry
seed = move_to([dest[0], dest[1], GRASP], GRIP, seed)   # lower
seed = move_to([dest[0], dest[1], GRASP], OPEN, seed)   # release
seed = move_to([dest[0], dest[1], 0.42], OPEN, seed)    # retreat
final = d.body("obj").xpos.copy()

print("object lifted to z = %.3f  (rest ~0.246)" % z_lift)
print("object final xy %s   dest %s   err %.3f"
      % (np.round(final[:2], 3), np.round(dest, 3), np.linalg.norm(final[:2] - dest)))
ok = z_lift > 0.33 and np.linalg.norm(final[:2] - dest) < 0.06
print("RESULT:", "PASS -- Piper picks the object and places it on the pad"
      if ok else "NEEDS TUNING")

"""Render the five task scenes as a figure panel, in the style of benchmark papers.

Composes, with mujoco.MjSpec: the AgileX PiPER arm (MuJoCo Menagerie, MIT) on
a table, the knife from hazard/assets/knife.xml attached to the gripper, a
cutting board, a target pad, and a stylised person made of capsules whose
arm reaches over the table.

    MUJOCO_GL=osmesa python scripts/render_tasks.py --piper third_party/menagerie/agilex_piper --out paper/fig_tasks.png

Poses are hand-set for illustration; they are not policy rollouts.
"""
import argparse
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

TABLE_Z = 0.0
SKIN = (0.80, 0.55, 0.38, 1)
SHIRT = (0.25, 0.62, 0.30, 1)
JEANS = (0.35, 0.45, 0.65, 1)


def add_person(spec, base_xy, facing_deg, reach, arm_name="human"):
    """Standing person at base_xy facing the table; `reach` is a dict with the
    hand position (world) the right arm should reach toward."""
    wb = spec.worldbody
    cx, cy = base_xy
    yaw = np.deg2rad(facing_deg)
    fwd = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    side = np.array([-np.sin(yaw), np.cos(yaw), 0.0])

    # legs and torso (stand on the floor at z = -0.75)
    floor = -0.75
    for s in (-1, 1):
        p = np.array([cx, cy, 0]) + s * 0.09 * side
        wb.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.06, 0, 0],
                    fromto=[p[0], p[1], floor + 0.05, p[0], p[1], floor + 0.80], rgba=JEANS)
    wb.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.14, 0, 0],
                fromto=[cx, cy, floor + 0.80, cx, cy, floor + 1.30], rgba=SHIRT)
    wb.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.10, 0, 0],
                pos=[cx, cy, floor + 1.50], rgba=SKIN)
    shoulder_z = floor + 1.28
    # right arm reaches toward `reach`; left arm hangs
    sh_r = np.array([cx, cy, shoulder_z]) + 0.17 * side * (-1)
    sh_l = np.array([cx, cy, shoulder_z]) + 0.17 * side
    wb.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.05, 0, 0],
                fromto=[*sh_l, sh_l[0], sh_l[1], sh_l[2] - 0.30], rgba=SHIRT)
    wb.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.045, 0, 0],
                fromto=[sh_l[0], sh_l[1], sh_l[2] - 0.30, sh_l[0], sh_l[1], sh_l[2] - 0.56], rgba=SKIN)
    hand = np.array(reach["hand"])
    elbow = sh_r + 0.5 * (hand - sh_r) + np.array([0, 0, reach.get("elbow_lift", 0.05)])
    wb.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.05, 0, 0], fromto=[*sh_r, *elbow], rgba=SHIRT)
    wb.add_geom(name=f"{arm_name}_forearm", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.04, 0, 0],
                fromto=[*elbow, *hand], rgba=SKIN)
    wb.add_geom(name=f"{arm_name}_hand", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.05, 0, 0],
                pos=hand, rgba=SKIN)


HRG_HUMAN = os.path.join(os.path.dirname(__file__), "..", "hazard", "assets", "hrg_human", "human.xml")
PELVIS_ABOVE_FLOOR = 1.14   # from the model's toe site


def add_person_mesh(spec, base_xy, facing_deg, reach, prefix="human_"):
    """Attach the mesh humanoid (Human-Robot Gym / kin-poly, see ATTRIBUTION.md)
    standing at base_xy, turned by facing_deg, with the right arm to be posed
    toward reach["hand"] by `pose_arm` after compilation.

    The model faces its local -x; we rotate about z so that -x points along facing_deg.
    """
    cx, cy = base_xy
    yaw = np.deg2rad(facing_deg) + np.pi   # local -x must point along `facing`
    quat = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    floor = -0.75
    human = mujoco.MjSpec.from_file(HRG_HUMAN)
    fr = spec.worldbody.add_frame(pos=[cx, cy, floor + PELVIS_ABOVE_FLOOR], quat=quat.tolist())
    fr.attach_body(human.body("object"), prefix, "")


def pose_arm(model, data, target, prefix="human_", side="R", seed=0, lean=True):
    """Numerical IK on the 3 shoulder + 3 elbow hinges so the hand site reaches `target`.
    Also bends the torso slightly forward. Pure visual posing for the figure."""
    from scipy.optimize import minimize
    names = ([f"{prefix}{side}_Shoulder_{a}" for a in "zyx"] + [f"{prefix}{side}_Elbow_{a}" for a in "zyx"]
             + [f"{prefix}Spine_x", f"{prefix}Torso_x"])
    qadr = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names]
    hand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{prefix}{side}_Hand")
    elbow = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{prefix}{side}_Elbow")
    target = np.asarray(target, float)

    lean_now = [data.qpos[a] for a in qadr[6:]]

    def cost(q):
        q = q.copy()
        q[6:] = np.clip(q[6:], 0.0, 0.45) if lean else lean_now   # lean forward only, modestly
        for adr, v in zip(qadr, q):
            data.qpos[adr] = v
        mujoco.mj_kinematics(model, data)
        err = np.linalg.norm(data.site_xpos[hand] - target)
        elbow_up = max(0.0, data.site_xpos[elbow][2] - target[2] - 0.35)   # keep the elbow from flying up
        return err + 0.01 * np.sum(np.square(q[:6])) + 0.5 * elbow_up

    rng = np.random.default_rng(seed)
    best = None
    for _ in range(16):
        q0 = np.concatenate([rng.uniform(-1.5, 1.5, 6), rng.uniform(0, 0.4, 2) if lean else np.array(lean_now)])
        r = minimize(cost, q0, method="Nelder-Mead", options=dict(maxiter=600, xatol=1e-3, fatol=1e-4))
        if best is None or r.fun < best.fun:
            best = r
    cost(best.x)
    mujoco.mj_forward(model, data)
    return float(np.linalg.norm(data.site_xpos[hand] - target))


def add_knife(parent, pos, quat):
    k = parent.add_body(name="knife", pos=pos, quat=quat)
    k.add_geom(name="knife_handle", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.055, 0.011, 0.013],
               pos=[-0.055, 0, 0], rgba=[0.12, 0.08, 0.05, 1])
    k.add_geom(name="knife_blade", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.100, 0.004, 0.022],
               pos=[0.100, 0, 0], rgba=[0.85, 0.87, 0.92, 1])  # blade drawn 8 mm thick for legibility
    k.add_site(name="blade_heel", pos=[0.0, 0, 0.02], size=[0.003, 0, 0], rgba=[1, 0, 0, 1])
    k.add_site(name="blade_tip", pos=[0.2, 0, 0.02], size=[0.003, 0, 0], rgba=[1, 0, 0, 1])
    k.add_site(name="blade_frame", pos=[0.1, 0, 0.02], size=[0.002, 0, 0], rgba=[0, 1, 0, 0.3])
    return k


def build(piper_dir, panel):
    piper_dir = os.path.abspath(piper_dir)  # meshdir is resolved relative to the XML otherwise
    spec = mujoco.MjSpec.from_file(os.path.join(piper_dir, "piper.xml"))
    spec.modelname = "sharp_hrc_scene"
    # the Menagerie file has a compiler setting that needs meshdir to resolve
    spec.meshdir = os.path.join(piper_dir, "assets")
    wb = spec.worldbody
    # visuals
    spec.visual.headlight.diffuse[:] = [0.7, 0.7, 0.7]
    spec.visual.headlight.ambient[:] = [0.35, 0.35, 0.35]
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 960, 720
    wb.add_light(pos=[0.5, -0.5, 2.0], dir=[-0.3, 0.3, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    # table
    wb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.75, 0.45, 0.02], pos=[0.25, 0, TABLE_Z - 0.02],
                rgba=[0.76, 0.62, 0.45, 1])
    for sx, sy in [(-0.45, -0.4), (0.95, -0.4), (-0.45, 0.4), (0.95, 0.4)]:
        wb.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.02, 0.36, 0],
                    pos=[sx, sy, TABLE_Z - 0.40], rgba=[0.5, 0.5, 0.5, 1])
    # cutting board and props
    if panel.get("board", True):
        wb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.17, 0.12, 0.008], pos=[0.42, 0.0, TABLE_Z + 0.008],
                    rgba=[0.85, 0.75, 0.55, 1])
    if panel.get("block"):
        wb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.03, 0.03, 0.03], pos=[0.42, 0.0, TABLE_Z + 0.046],
                    rgba=[0.95, 0.45, 0.2, 1])
    if panel.get("pad"):
        wb.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.06, 0.004, 0], pos=[*panel["pad"], TABLE_Z + 0.004],
                    rgba=[0.3, 0.8, 0.9, 1])
    if panel.get("rack"):
        wb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.12, 0.03, 0.03], pos=[*panel["rack"], TABLE_Z + 0.03],
                    rgba=[0.3, 0.3, 0.3, 1])

    # knife
    if panel.get("knife_in_gripper", True):
        link6 = spec.body("link6")
        q = np.array(panel.get("knife_quat", Q_DOWN), float)
        # put the handle centre (knife-frame x = -0.055) at the fingertips, whatever the orientation
        off = np.zeros(3)
        mujoco.mju_rotVecQuat(off, np.array([-0.055, 0.0, 0.0]), q)
        grip = np.array([0, 0, 0.105]) - off
        add_knife(link6, grip.tolist(), q.tolist())
    elif panel.get("knife_world"):
        kw = panel["knife_world"]
        add_knife(wb, kw["pos"], kw["quat"])

    add_person_mesh(spec, panel["person_xy"], panel["person_yaw"], panel["reach"])

    look = wb.add_body(name="look_at", pos=panel.get("look_at", [0.45, 0.15, 0.25]))
    cam = wb.add_camera(name="view", pos=panel.get("cam_pos", [1.45, -1.05, 0.95]),
                          mode=mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY, targetbody=panel.get("target", "look_at"), fovy=panel.get("fovy", 42))
    model = spec.compile()
    data = mujoco.MjData(model)
    data.qpos[:6] = panel["qpos"]
    data.qpos[6:8] = [0.02, -0.02]
    mujoco.mj_forward(model, data)
    err = pose_arm(model, data, panel["reach"]["hand"])
    # left arm hangs beside the left hip
    hip = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "human_L_Hip")].copy()
    shoulder = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "human_L_Shoulder")].copy()
    out = shoulder - hip; out[2] = 0; out = out / (np.linalg.norm(out) + 1e-9)
    pose_arm(model, data, hip + 0.12 * out + np.array([0, 0, -0.30]), side="L", lean=False)
    print(f"  arm IK residual {err*100:.1f} cm")
    return model, data


Q_DOWN = [0.7071, 0, -0.7071, 0]     # blade along the gripper axis (chop / put-down)
Q_FLAT = [0.5, 0.5, 0.5, 0.5]       # blade horizontal, edge down (carry / pass-by)
Q_HAND = [0.5, -0.5, 0.5, -0.5]     # blade horizontal, handle toward the person

PANELS = {
    "T1 Carry": dict(qpos=[0.6, 1.1, -0.9, 0.0, 0.8, 0.0], knife_quat=Q_FLAT, rack=[-0.05, 0.35], pad=[0.55, -0.25],
                     person_xy=(0.55, 0.62), person_yaw=-100, reach=dict(hand=[0.45, 0.22, 0.12])),
    "T2 Put-down": dict(qpos=[0.0, 1.5, -1.3, 0.0, 1.0, 0.0], knife_quat=Q_DOWN,
                        person_xy=(0.70, 0.62), person_yaw=-110, reach=dict(hand=[0.50, 0.18, 0.08])),
    "T3 Interrupted chop": dict(qpos=[0.0, 1.6, -1.1, 0.0, 1.2, 1.57], block=True, knife_quat=Q_DOWN,
                                person_xy=(0.60, 0.62), person_yaw=-100, reach=dict(hand=[0.44, 0.16, 0.07])),
    "T4 Handover": dict(qpos=[0.3, 1.0, -0.6, 0.0, 0.6, 3.14], board=False, knife_quat=Q_HAND,
                        person_xy=(1.05, 0.30), person_yaw=-170, reach=dict(hand=[0.66, 0.12, 0.22])),
    "T5 Pass-by": dict(qpos=[-0.5, 1.2, -0.9, 0.0, 0.9, 0.0], pad=[0.45, -0.35], board=False, knife_quat=Q_FLAT,
                       person_xy=(0.55, 0.62), person_yaw=-95, reach=dict(hand=[0.40, 0.12, 0.04])),
    "Hazard metric: edge vs spine": dict(qpos=[0.3, 1.3, -1.0, 0.0, 1.0, 0.0], board=False, knife_quat=Q_FLAT,
                       person_xy=(0.60, 0.62), person_yaw=-100, reach=dict(hand=[0.44, 0.12, 0.07]),
                       cam_pos=[1.0, -0.35, 0.35], fovy=32, target="knife"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--piper", required=True, help="path to menagerie/agilex_piper")
    ap.add_argument("--out", default="paper/fig_tasks.png")
    ap.add_argument("--w", type=int, default=640)
    ap.add_argument("--h", type=int, default=480)
    args = ap.parse_args()

    from PIL import Image, ImageDraw, ImageFont

    tiles = []
    for name, panel in PANELS.items():
        model, data = build(args.piper, panel)
        r = mujoco.Renderer(model, args.h, args.w)
        r.update_scene(data, camera="view")
        img = Image.fromarray(r.render())
        d = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 26)
        except OSError:
            font = ImageFont.load_default()
        d.rectangle([0, args.h - 44, args.w, args.h], fill=(255, 255, 255))
        d.text((12, args.h - 38), name, fill=(0, 0, 0), font=font)
        tiles.append(img)
        r.close()

    gap = 8
    cols = 3
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * args.w + (cols - 1) * gap, rows * args.h + (rows - 1) * gap), (255, 255, 255))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * (args.w + gap), (i // cols) * (args.h + gap)))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    sheet.save(args.out)
    print("wrote", args.out, sheet.size)


if __name__ == "__main__":
    main()

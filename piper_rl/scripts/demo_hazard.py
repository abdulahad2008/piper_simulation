"""Drive a knife past a human forearm with mocap and print the hazard trace.

Shows the point of the metric: in both passes the knife follows the same
path at the same speed and never touches the person, so a link-collision
metric reports nothing. Only the pass with the edge facing the person
accumulates hazard.

    python piper_rl/scripts/demo_hazard.py
"""
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from piper_rl.hazard.metrics import ExposureAccumulator, blade_hazard  # noqa: E402
from piper_rl.hazard.mujoco_bridge import FiniteDifferenceVelocity, read_blade, read_human_capsules  # noqa: E402

ASSETS = os.path.join(os.path.dirname(__file__), "..", "hazard", "assets")


def run(edge_toward_human: bool, speed: float = 0.3, dt: float = 0.05):
    model = mujoco.MjModel.from_xml_path(os.path.join(ASSETS, "demo_scene.xml"))
    data = mujoco.MjData(model)
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "knife_mocap")
    hid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "human_mocap")
    k_m = model.body_mocapid[kid]
    h_m = model.body_mocapid[hid]

    # human forearm vertical-ish along y at x=0.35; knife sweeps along +x at y=0.0
    data.mocap_pos[h_m] = [0.35, 0.05, 0.15]
    # knife: blade along +x; roll about x so the edge (+Z local) points +y (toward human side) or -y
    angle = -np.pi / 2 if edge_toward_human else np.pi / 2
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, np.array([1.0, 0, 0]), angle)
    data.mocap_quat[k_m] = q

    acc = ExposureAccumulator(dt)
    fd = FiniteDifferenceVelocity(dt)
    n_sub = int(round(dt / model.opt.timestep))
    x = -0.1
    trace = []
    for _ in range(int(0.6 / (speed * dt))):
        x += speed * dt
        data.mocap_pos[k_m] = [x, -0.08, 0.15]
        for _ in range(n_sub):
            mujoco.mj_step(model, data)
        blade = read_blade(model, data)
        caps = read_human_capsules(model, data)
        blade, caps = fd.apply(blade, caps)
        s = blade_hazard(blade, caps)
        acc.update(s)
        trace.append((x, s.edge_distance, s.alignment, s.closing_speed, s.hazard))
    return acc.summary(), trace


if __name__ == "__main__":
    for facing in (True, False):
        summ, trace = run(facing)
        print(f"\nedge facing human: {facing}")
        print("   x      d_edge  align  v_close  hazard")
        for row in trace[::3]:
            print("  {:5.2f}  {:6.3f}  {:5.2f}  {:6.3f}  {:6.3f}".format(*row))
        print("summary:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in summ.items()})

"""Open the PIPER pick-and-place scene in the MuJoCo viewer.

Two modes:

    python agilex_piper\\view_piper.py --camera forehead
        STATIC INSPECTION. Physics runs, but nothing commands the arm, so it
        just holds its home pose. This is for looking at the model -- it is
        *not* the task running, and the arm not moving is expected.

    python agilex_piper\\view_piper.py --demo --camera forehead
        RUNS THE TASK. Drives the scripted pick-and-place through the RL
        environment so you can watch the arm actually pick the cup up and put
        it on the green pad, live, at real-time speed. Loops forever with a new
        randomised cup and target each episode.

macOS: run with `mjpython` instead of `python`.

Camera controls inside the window
---------------------------------
    [  and  ]        cycle cameras: Free -> forehead -> wrist -> front ->
                     side -> top -> overview
    left panel       "Rendering" section -> "Camera" selector (same thing,
                     with names, if you prefer clicking)
    Tab              show/hide the side panels
    Esc              back to the Free camera
    Free camera:     left-drag rotate | right-drag pan | scroll zoom |
                     double-click a body to orbit it
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

HERE = Path(__file__).resolve().parent

p = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--model", default=str(HERE / "piper_task.xml"))
p.add_argument("--camera", default=None,
               help="forehead | wrist | front | side | top | overview")
p.add_argument("--demo", action="store_true",
               help="run the scripted pick-and-place instead of standing still")
p.add_argument("--episodes", type=int, default=0,
               help="with --demo: how many episodes (0 = loop forever)")
p.add_argument("--speed", type=float, default=1.0,
               help="with --demo: playback speed multiplier")
args = p.parse_args()


def set_camera(viewer, model):
    if args.camera is None:
        return
    try:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = model.camera(args.camera).id
    except (KeyError, ValueError):
        names = ", ".join(model.camera(i).name for i in range(model.ncam))
        print(f"unknown camera {args.camera!r}; available: {names}")


# --------------------------------------------------------------------------- #
if args.demo:
    # Import the env lazily so plain viewing does not need gymnasium/SB3.
    sys.path.insert(0, str(HERE.parent))
    from piper_rl.config import EnvConfig
    from piper_rl.piper_env import PiperPickPlaceEnv
    from piper_rl.scripts.scripted_demo import build_plan

    cfg = EnvConfig()
    cfg.curriculum = False                     # always the true initial state
    env = PiperPickPlaceEnv(cfg)
    m, d = env.model, env.data
    env.reset(seed=0)

    dt = 1.0 / cfg.control_hz / max(args.speed, 1e-3)
    print("cameras:", ", ".join(m.camera(i).name for i in range(m.ncam)))
    print("running the scripted pick-and-place -- close the window to stop")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        set_camera(viewer, m)
        ep = 0
        while viewer.is_running() and (args.episodes == 0 or ep < args.episodes):
            env.reset(seed=ep)
            plan = build_plan(env)
            done = False
            for name, target, grip, max_steps, tol, gain in plan:
                for _ in range(max_steps):
                    if not viewer.is_running():
                        break
                    t0 = time.time()
                    err = target - env.tcp_pos
                    a = np.clip(gain * err / cfg.max_step_dist, -1.0, 1.0)
                    _, _, terminated, truncated, info = env.step(
                        np.array([a[0], a[1], a[2], 0.0, float(grip)]))
                    viewer.sync()
                    time.sleep(max(0.0, dt - (time.time() - t0)))
                    if terminated or truncated:
                        done = True
                        break
                    if tol > 0 and np.linalg.norm(err) < tol:
                        break
                if done or not viewer.is_running():
                    break
            # let the object settle so the success detector sees it at rest
            for _ in range(25):
                if not viewer.is_running() or done:
                    break
                t0 = time.time()
                _, _, terminated, truncated, info = env.step(
                    np.array([0, 0, 0.3, 0, 1.0]))
                viewer.sync()
                time.sleep(max(0.0, dt - (time.time() - t0)))
                done = terminated or truncated
            print(f"episode {ep}: success={bool(info.get('is_success'))}  "
                  f"placement error={info.get('d_place', float('nan'))*1000:.0f} mm")
            ep += 1
    env.close()

# --------------------------------------------------------------------------- #
else:
    m = mujoco.MjModel.from_xml_path(args.model)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)       # home pose, gripper open
    mujoco.mj_forward(m, d)
    d.ctrl[:] = m.key_ctrl[0]                  # hold the pose instead of sagging

    print("cameras:", ", ".join(m.camera(i).name for i in range(m.ncam)))
    print("static view -- the arm holds its home pose. "
          "Add --demo to watch the pick-and-place.")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        set_camera(viewer, m)
        while viewer.is_running():
            t0 = time.time()
            mujoco.mj_step(m, d)
            viewer.sync()
            dt = m.opt.timestep - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)

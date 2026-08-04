"""Open the Piper pick-and-place scene in the MuJoCo passive viewer.
Run with:  mjpython view_piper.py
Orbit with the mouse; switch cameras with the [ and ] keys."""
import time
import mujoco
import mujoco.viewer

m = mujoco.MjModel.from_xml_path("piper_task.xml")
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, 0)          # home pose, gripper open

with mujoco.viewer.launch_passive(m, d) as viewer:
    while viewer.is_running():
        step_start = time.time()
        mujoco.mj_step(m, d)
        viewer.sync()
        dt = m.opt.timestep - (time.time() - step_start)
        if dt > 0:
            time.sleep(dt)

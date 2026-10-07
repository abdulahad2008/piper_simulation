"""The knife scene must equal the base human scene plus the knife, and the generated XML must be current."""
import mujoco
import numpy as np

from piper_rl.hazard.scene import KNIFE_HUMAN_MODEL_PATH
from piper_rl.human_config import HUMAN_MODEL_PATH
from piper_rl.scripts.make_knife_scene import SCENE_DIR, generate


def _names(model, obj, n):
    return {mujoco.mj_id2name(model, obj, i) for i in range(n)}


def test_generated_files_are_current():
    for name, text in generate().items():
        assert (SCENE_DIR / name).read_text() == text, f"{name} is stale: run python -m piper_rl.scripts.make_knife_scene"


def test_knife_scene_is_base_plus_knife():
    base = mujoco.MjModel.from_xml_path(str(HUMAN_MODEL_PATH))
    knife = mujoco.MjModel.from_xml_path(str(KNIFE_HUMAN_MODEL_PATH))
    assert (base.nq, base.nv, base.nu, base.nmocap) == (knife.nq, knife.nv, knife.nu, knife.nmocap)
    assert _names(knife, mujoco.mjtObj.mjOBJ_BODY, knife.nbody) - _names(base, mujoco.mjtObj.mjOBJ_BODY, base.nbody) == {"knife"}
    assert _names(knife, mujoco.mjtObj.mjOBJ_GEOM, knife.ngeom) - _names(base, mujoco.mjtObj.mjOBJ_GEOM, base.ngeom) == {
        "knife_handle", "knife_blade"}
    assert _names(knife, mujoco.mjtObj.mjOBJ_SITE, knife.nsite) - _names(base, mujoco.mjtObj.mjOBJ_SITE, base.nsite) == {
        "blade_heel", "blade_tip", "blade_frame"}
    for gid in (knife.geom("knife_handle").id, knife.geom("knife_blade").id):
        assert knife.geom_contype[gid] == 0 and knife.geom_conaffinity[gid] == 0


def test_handle_centre_on_tcp_and_blade_along_gripper_axis():
    m = mujoco.MjModel.from_xml_path(str(KNIFE_HUMAN_MODEL_PATH))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    assert np.allclose(d.geom_xpos[m.geom("knife_handle").id], d.site_xpos[m.site("tcp").id], atol=1e-6)
    approach = d.xmat[m.body("link6").id].reshape(3, 3)[:, 2]
    blade = d.site_xpos[m.site("blade_tip").id] - d.site_xpos[m.site("blade_heel").id]
    assert np.allclose(blade / np.linalg.norm(blade), approach, atol=1e-6)

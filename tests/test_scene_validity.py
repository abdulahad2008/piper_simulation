"""Scene-validity tests for the human-aware PiPER scenes (kitchen-hazard study, Phase 1).

Two groups of tests live here.

BUG DETECTORS. These fail on the code before the arm-length fix
(``_quat_from_z_axis`` normalised its argument in place, so every segment
capsule was set to half-length 0.5 m) and on scenes whose held object cannot
touch the person. They read the geometry MuJoCo actually draws and collides
(``geom_xpos``, ``geom_xmat``, ``geom_size``, ``geom_rbound``) rather than the
kinematic model that produced it, so they catch any future disagreement
between the two.

REGRESSION GUARDS. Finite values, same-seed determinism and the invariance of
no-human episodes to the human geometry cannot fail on the old code by
construction. They are here so that a later change cannot break them silently.
They are labelled as guards in the class names.

Episode budget: ``SCENE_VALIDITY_EPISODES`` (default 1000) kinematic episodes
per preset for the geometry sweep, sampled at the 20 Hz control grid over the
10 s horizon (201 samples per episode; the hand moves at most 2 cm between
samples at the fastest in-distribution speed). The sweep is run once per
preset and every geometry test reads from it. ``SCENE_VALIDITY_PHYSICS_EPISODES``
(default 10) full physics episodes are used for the guards. The geometry sweep
does not step physics: the human arm is kinematic, so its geometry depends
only on the trajectory and on the code that writes it into MuJoCo.

Run as one file per process, like the rest of this test suite.
"""
from __future__ import annotations

import functools
import os
from dataclasses import dataclass, field

try:  # same import-order guard as tests/conftest.py (OSMesa vs torch/triton LLVM)
    import torch  # noqa: F401
    import triton  # noqa: F401
except Exception:  # pragma: no cover
    pass

import mujoco
import numpy as np
import pytest

from piper_rl.config import PROJECT_DIR
from piper_rl.hazard.scene import KNIFE_HUMAN_MODEL_PATH
from piper_rl.human_aware_env import PiperHumanAwarePickPlaceEnv
from piper_rl.human_config import HUMAN_MODEL_PATH, HumanAwareEnvConfig

N_EPISODES = int(os.environ.get("SCENE_VALIDITY_EPISODES", "1000"))
N_PHYSICS_EPISODES = int(os.environ.get("SCENE_VALIDITY_PHYSICS_EPISODES", "10"))
TABLE_TOP = 0.20
LENGTH_TOL = 1e-3          # 1 mm, as specified
PENETRATION_TOL = 1e-6     # numerical zero for signed distances
MIN_HALF_LENGTH = 0.01     # the env floors the capsule half-length here
HUMAN_SEGMENT_GEOMS = ("human_upper_arm_geom", "human_forearm_geom")
HUMAN_GEOMS = HUMAN_SEGMENT_GEOMS + ("human_hand_geom",)
HUMAN_BODIES = ("human_upper_arm", "human_forearm", "human_hand")
CONTROL_HZ = 20
HORIZON_S = 10.0

# Every human preset the campaign uses. A new preset (for example the fast-hand
# test set) is added here so the sweep covers it.
PRESETS: dict[str, dict] = {
    "no_human": dict(preset="no_human"),
    "training_id": dict(preset="randomized"),
    "curriculum_d0": dict(preset="curriculum", difficulty=0.0),
    "curriculum_d05": dict(preset="curriculum", difficulty=0.5),
    "curriculum_d1": dict(preset="curriculum", difficulty=1.0),
    "fixed_height_randomized": dict(preset="fixed"),
    "fixed_exact_h036_v1": dict(preset="fixed_exact_h036_v1"),
    "fixed_exact_shifted_h036_v1": dict(preset="fixed_exact_shifted_h036_v1"),
    "evaluation_id": dict(preset="evaluation_id"),
    "evaluation_ood": dict(preset="evaluation_ood"),
}
SCENES = {
    "cup": str(HUMAN_MODEL_PATH),
    "knife": str(KNIFE_HUMAN_MODEL_PATH),
}
# Bodies that must stay rigidly attached to the gripper and be touchable by
# the person. Only the ones present in a scene are checked.
HELD_OBJECT_BODIES = ("knife", "pan")
PAN_MODEL_PATH = PROJECT_DIR / "agilex_piper" / "piper_human_pan_task.xml"
if PAN_MODEL_PATH.exists():
    SCENES["pan"] = str(PAN_MODEL_PATH)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def make_config(spec: dict, scene: str = "cup", human_state: bool = True) -> HumanAwareEnvConfig:
    cfg = HumanAwareEnvConfig.from_preset(spec["preset"], spec.get("difficulty", 1.0))
    cfg.model_path = SCENES[scene]
    cfg.curriculum = False
    cfg.domain_rand.enabled = False
    cfg.noise.enabled = False
    cfg.include_human_state = human_state
    return cfg


@dataclass
class Capsule:
    name: str
    p1: np.ndarray
    p2: np.ndarray
    radius: float
    half_length: float
    rbound: float


def drawn_capsule(model: mujoco.MjModel, data: mujoco.MjData, gid: int) -> Capsule:
    """The capsule as MuJoCo draws and collides it (world frame)."""
    pos = data.geom_xpos[gid]
    axis = data.geom_xmat[gid].reshape(3, 3)[:, 2]
    radius = float(model.geom_size[gid, 0])
    half = float(model.geom_size[gid, 1])
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
    return Capsule(name, pos - half * axis, pos + half * axis, radius, half,
                   float(model.geom_rbound[gid]))


def static_physical_geoms(model: mujoco.MjModel, exclude: set[int]) -> list[int]:
    """World-body geoms that take part in collisions: table, floor, later the
    stove, trivet and board. Visual-only pads are skipped."""
    out = []
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] != 0 or gid in exclude:
            continue
        if model.geom_contype[gid] == 0 and model.geom_conaffinity[gid] == 0:
            continue
        out.append(gid)
    return out


_FROMTO = np.zeros(6)


def signed_distance(model, data, gid_a: int, gid_b: int) -> float:
    return float(mujoco.mj_geomDistance(model, data, gid_a, gid_b, 2.0, _FROMTO))


def lowest_point_in_footprint(cap: Capsule, model: mujoco.MjModel, table_gid: int,
                              n: int = 64) -> float:
    """Analytic check from the drawn geometry only: the lowest surface point of
    the capsule whose axis point lies over the table footprint (expanded by the
    radius). Returns +inf when the capsule is wholly outside the footprint."""
    centre = model.geom_pos[table_gid]
    half = model.geom_size[table_gid]
    pts = cap.p1[None, :] + np.linspace(0.0, 1.0, n)[:, None] * (cap.p2 - cap.p1)[None, :]
    inside = (np.abs(pts[:, 0] - centre[0]) <= half[0] + cap.radius) & \
             (np.abs(pts[:, 1] - centre[1]) <= half[1] + cap.radius)
    if not inside.any():
        return np.inf
    return float(pts[inside, 2].min() - cap.radius)


@dataclass
class SweepReport:
    preset: str
    n_episodes: int
    n_samples: int = 0
    worst_length_err: float = 0.0            # |drawn length - joint distance|
    worst_endpoint_err: float = 0.0          # sum of endpoint offsets
    worst_hand_centre_err: float = 0.0
    worst_rbound_err: float = 0.0
    deepest_in_footprint: float = np.inf     # lowest z - r over the table footprint
    min_static_distance: dict = field(default_factory=dict)   # static geom name -> min signed distance
    min_base_distance_at_rest: float = np.inf
    first: dict = field(default_factory=dict)                 # check -> first failing message
    nonfinite: int = 0

    def note(self, check: str, msg: str) -> None:
        self.first.setdefault(check, msg)


@functools.lru_cache(maxsize=None)
def sweep(preset: str, n_episodes: int = N_EPISODES, scene: str = "cup") -> SweepReport:
    """Write every human state on the 20 Hz grid into MuJoCo, run kinematics,
    and record the geometry MuJoCo sees. One pass per preset serves all tests."""
    env = PiperHumanAwarePickPlaceEnv(make_config(PRESETS[preset], scene=scene))
    m, d = env.model, env.data
    rep = SweepReport(preset, n_episodes)
    gids = [m.geom(n).id for n in HUMAN_GEOMS]
    up, fore, hand = gids
    statics = static_physical_geoms(m, exclude=set(gids))
    snames = {g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in statics}
    rep.min_static_distance = {snames[g]: np.inf for g in statics}
    base = m.geom("base_col").id
    table = m.geom("table").id
    times = np.arange(0, int(HORIZON_S * CONTROL_HZ) + 1) / CONTROL_HZ
    for ep in range(n_episodes):
        env.reset(seed=100_000 + ep)
        for g in gids:   # rest state: the reset pose
            dist = signed_distance(m, d, g, base)
            rep.min_base_distance_at_rest = min(rep.min_base_distance_at_rest, dist)
            if dist < -PENETRATION_TOL:
                rep.note("base", f"ep {ep}: {HUMAN_GEOMS[gids.index(g)]} intersects the robot base at rest ({dist*1000:.1f} mm)")
        traj = env.human_trajectory
        for t in times:
            st = traj.state(float(t))
            env._set_human_state(st)
            mujoco.mj_kinematics(m, d)
            rep.n_samples += 1
            caps = [drawn_capsule(m, d, g) for g in gids]
            joints = ((st.shoulder_position, st.elbow_position),
                      (st.elbow_position, st.hand_position))
            for cap, (a, b) in zip(caps[:2], joints):
                if not (np.isfinite(cap.p1).all() and np.isfinite(cap.p2).all()):
                    rep.nonfinite += 1
                    rep.note("finite", f"ep {ep} t={t:.2f}: {cap.name} has non-finite pose")
                    continue
                joint_len = float(np.linalg.norm(b - a))
                err = abs(2 * cap.half_length - max(joint_len, 2 * MIN_HALF_LENGTH))
                if err > rep.worst_length_err:
                    rep.worst_length_err = err
                if err > LENGTH_TOL:
                    rep.note("length", f"ep {ep} t={t:.2f}s: {cap.name} drawn {2*cap.half_length:.4f} m, joint distance {joint_len:.4f} m")
                end_err = min(np.linalg.norm(cap.p1 - a) + np.linalg.norm(cap.p2 - b),
                              np.linalg.norm(cap.p1 - b) + np.linalg.norm(cap.p2 - a))
                if end_err > rep.worst_endpoint_err:
                    rep.worst_endpoint_err = end_err
                if end_err > 2 * LENGTH_TOL and joint_len >= 2 * MIN_HALF_LENGTH:
                    rep.note("endpoints", f"ep {ep} t={t:.2f}s: {cap.name} endpoints off by {end_err*1000:.1f} mm")
            hc = 0.5 * (caps[2].p1 + caps[2].p2)
            herr = float(np.linalg.norm(hc - st.hand_position))
            rep.worst_hand_centre_err = max(rep.worst_hand_centre_err, herr)
            if herr > LENGTH_TOL:
                rep.note("hand", f"ep {ep} t={t:.2f}s: hand capsule centre off by {herr*1000:.1f} mm")
            for cap in caps:
                rerr = abs(cap.rbound - (cap.radius + cap.half_length))
                rep.worst_rbound_err = max(rep.worst_rbound_err, rerr)
                if rerr > 1e-9:
                    rep.note("rbound", f"ep {ep} t={t:.2f}s: {cap.name} rbound {cap.rbound:.4f} vs {cap.radius + cap.half_length:.4f}")
                low = lowest_point_in_footprint(cap, m, table)
                if low < rep.deepest_in_footprint:
                    rep.deepest_in_footprint = low
                if low < TABLE_TOP - PENETRATION_TOL:
                    rep.note("footprint", f"ep {ep} t={t:.2f}s: {cap.name} reaches z={low:.3f} m over the table footprint (top {TABLE_TOP} m)")
            for g in gids:
                for s in statics:
                    dist = signed_distance(m, d, g, s)
                    if dist < rep.min_static_distance[snames[s]]:
                        rep.min_static_distance[snames[s]] = dist
                    if dist < -PENETRATION_TOL:
                        rep.note("penetration", f"ep {ep} t={t:.2f}s: {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)} penetrates {snames[s]} by {-dist*1000:.1f} mm")
    env.close()
    return rep


# --------------------------------------------------------------------------- #
# bug detectors
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("preset", list(PRESETS))
class TestDrawnArmGeometry:
    """Fails on the pre-fix code for every preset: half-lengths were 0.5 m."""

    def test_capsule_length_equals_joint_distance(self, preset):
        rep = sweep(preset)
        assert rep.nonfinite == 0, rep.first.get("finite")
        assert rep.worst_length_err <= LENGTH_TOL, (
            f"{preset}: worst length error {rep.worst_length_err*1000:.1f} mm over "
            f"{rep.n_samples} samples; first: {rep.first.get('length')}")

    def test_capsule_endpoints_coincide_with_joint_points(self, preset):
        """Length alone could be right with the wrong orientation."""
        rep = sweep(preset)
        assert "endpoints" not in rep.first, f"{preset}: {rep.first['endpoints']}"
        assert rep.worst_hand_centre_err <= LENGTH_TOL, f"{preset}: {rep.first.get('hand')}"

    def test_broadphase_bound_matches_drawn_size(self, preset):
        """Contact culling uses geom_rbound; it must equal radius + half-length."""
        rep = sweep(preset)
        assert rep.worst_rbound_err <= 1e-9, f"{preset}: {rep.first.get('rbound')}"

    def test_person_never_below_table_top_inside_footprint(self, preset):
        """Analytic, from the drawn geometry alone."""
        rep = sweep(preset)
        assert rep.deepest_in_footprint >= TABLE_TOP - PENETRATION_TOL, (
            f"{preset}: lowest point over the footprint z={rep.deepest_in_footprint:.3f} m; "
            f"first: {rep.first.get('footprint')}")

    def test_person_never_penetrates_static_geometry(self, preset):
        """MuJoCo's own signed distance against every static physical geom
        (table, floor; later stove, trivet, board). This is the geometry the
        contact solver uses, so agreement with the analytic test above shows
        the collision code and the drawn geometry are the same thing."""
        rep = sweep(preset)
        assert rep.min_static_distance, "no static physical geoms found"
        bad = {k: v for k, v in rep.min_static_distance.items() if v < -PENETRATION_TOL}
        assert not bad, f"{preset}: penetrations (m) {bad}; first: {rep.first.get('penetration')}"

    def test_person_clear_of_robot_base_at_rest(self, preset):
        rep = sweep(preset)
        assert rep.min_base_distance_at_rest >= -PENETRATION_TOL, f"{preset}: {rep.first.get('base')}"


PAN_PRESETS = ("training_id", "evaluation_ood", "fixed_exact_h036_v1")


@pytest.mark.skipif("pan" not in SCENES, reason="pan scene not generated")
@pytest.mark.parametrize("preset", PAN_PRESETS)
class TestPanSceneGeometry:
    """The same sweep on the hot-pan scene: the person must also stay clear of
    the stove, the hob ring and the trivet (all static collision geoms)."""

    def test_person_never_penetrates_static_geometry(self, preset):
        rep = sweep(preset, scene="pan")
        names = set(rep.min_static_distance)
        assert {"table", "floor", "stove", "hob", "trivet"} <= names, names
        bad = {k: v for k, v in rep.min_static_distance.items() if v < -PENETRATION_TOL}
        assert not bad, f"{preset}: penetrations (m) {bad}; first: {rep.first.get('penetration')}"

    def test_capsule_length_equals_joint_distance(self, preset):
        rep = sweep(preset, scene="pan")
        assert rep.worst_length_err <= LENGTH_TOL, rep.first.get("length")
        assert rep.deepest_in_footprint >= TABLE_TOP - PENETRATION_TOL, rep.first.get("footprint")


class TestHazardBridgeGeometry:
    """The hazard module reads the same capsules; it must see the joint lengths."""

    def test_bridge_capsules_match_joint_points(self):
        from piper_rl.hazard.mujoco_bridge import read_human_capsules
        env = PiperHumanAwarePickPlaceEnv(make_config(PRESETS["evaluation_id"], scene="knife"))
        times = np.arange(0, int(HORIZON_S * CONTROL_HZ) + 1) / CONTROL_HZ
        for ep in range(max(1, N_EPISODES // 20)):
            env.reset(seed=500_000 + ep)
            traj = env.human_trajectory
            for t in times:
                st = traj.state(float(t))
                env._set_human_state(st)
                mujoco.mj_kinematics(env.model, env.data)
                caps = read_human_capsules(env.model, env.data)
                assert len(caps) == 3, [c.name for c in caps]
                lengths = sorted(float(np.linalg.norm(c.a - c.b)) for c in caps)
                joint = sorted([
                    float(np.linalg.norm(st.elbow_position - st.shoulder_position)),
                    float(np.linalg.norm(st.hand_position - st.elbow_position)),
                    0.11,  # hand capsule, fixed 2 * 0.055
                ])
                for dl, j in zip(lengths, joint):
                    assert abs(dl - max(j, 2 * MIN_HALF_LENGTH)) <= LENGTH_TOL, (lengths, joint)
        env.close()


class TestHeldObject:
    """Scenes with a tool in the gripper."""

    @pytest.mark.parametrize("scene", [s for s in SCENES if s != "cup"])
    def test_held_object_touchable_by_person(self, scene):
        """Fails on the committed knife scene: contype=conaffinity=0."""
        model = mujoco.MjModel.from_xml_path(SCENES[scene])
        human = [model.geom(n).id for n in HUMAN_GEOMS]
        bodies = [b for b in HELD_OBJECT_BODIES
                  if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) >= 0]
        assert bodies, f"{scene}: no held-object body found"
        for body in bodies:
            bid = model.body(body).id
            geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == bid]
            assert geoms, f"{body} has no geoms"
            for g in geoms:
                for h in human:
                    touch = (model.geom_contype[g] & model.geom_conaffinity[h]) or \
                            (model.geom_contype[h] & model.geom_conaffinity[g])
                    gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
                    assert touch, (
                        f"{scene}: {gname} (contype={model.geom_contype[g]}, "
                        f"conaffinity={model.geom_conaffinity[g]}) cannot contact the person")

    @pytest.mark.parametrize("scene", [s for s in SCENES if s != "cup"])
    def test_held_object_contact_is_detected(self, scene):
        """Place the hand capsule on the held object and check MuJoCo reports a
        contact between them (the full pipeline, not only the bitmasks)."""
        env = PiperHumanAwarePickPlaceEnv(make_config(PRESETS["evaluation_id"], scene=scene))
        env.reset(seed=1)
        m, d = env.model, env.data
        bodies = [b for b in HELD_OBJECT_BODIES
                  if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) >= 0]
        hand = env.human_mocap_ids["human_hand"]
        for body in bodies:
            bid = m.body(body).id
            geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] == bid}
            # put the hand on the object's first geom centre
            d.mocap_pos[hand] = d.geom_xpos[next(iter(geoms))]
            mujoco.mj_forward(m, d)
            pairs = {(int(d.contact[i].geom1), int(d.contact[i].geom2)) for i in range(d.ncon)}
            hg = env.human_hand_geom
            hit = any((hg in p) and (set(p) & geoms) for p in pairs)
            assert hit, f"{scene}: no contact reported between the hand and {body}"
        env.close()

    @pytest.mark.parametrize("scene", [s for s in SCENES if s != "cup"])
    def test_held_object_rigid_in_gripper(self, scene):
        """The held object's pose relative to the tool frame is constant through
        a random-action episode (a guard for a fixed child body; a real test once
        the pan is held by friction)."""
        env = PiperHumanAwarePickPlaceEnv(make_config(PRESETS["evaluation_id"], scene=scene))
        bodies = [b for b in HELD_OBJECT_BODIES
                  if mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, b) >= 0]
        link6 = env.model.body("link6").id
        rng = np.random.default_rng(0)
        for ep in range(min(N_PHYSICS_EPISODES, 3)):
            env.reset(seed=600_000 + ep)
            ref = {}
            for step in range(200):
                _, _, te, tr, _ = env.step(rng.uniform(-1, 1, env.action_space.shape))
                R6 = env.data.xmat[link6].reshape(3, 3)
                for b in bodies:
                    bid = env.model.body(b).id
                    rel_p = R6.T @ (env.data.xpos[bid] - env.data.xpos[link6])
                    rel_R = R6.T @ env.data.xmat[bid].reshape(3, 3)
                    if b not in ref:
                        ref[b] = (rel_p.copy(), rel_R.copy())
                    assert np.allclose(rel_p, ref[b][0], atol=1e-3), (b, step, rel_p, ref[b][0])
                    assert np.allclose(rel_R, ref[b][1], atol=1e-3), (b, step)
                if te or tr:
                    break
        env.close()


# --------------------------------------------------------------------------- #
# regression guards (cannot fail on the old code; kept so they cannot regress)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("preset", ["training_id", "evaluation_ood", "fixed_exact_h036_v1", "no_human"])
class TestGuardsFiniteAndDeterministic:

    def test_guard_no_nan_or_exploding_values(self, preset):
        env = PiperHumanAwarePickPlaceEnv(make_config(PRESETS[preset]))
        rng = np.random.default_rng(1)
        for ep in range(N_PHYSICS_EPISODES):
            obs, _ = env.reset(seed=700_000 + ep)
            assert np.isfinite(obs).all()
            for _ in range(200):
                obs, r, te, tr, info = env.step(rng.uniform(-1, 1, env.action_space.shape))
                assert np.isfinite(obs).all() and np.isfinite(r)
                assert np.abs(obs).max() < 100.0, np.abs(obs).max()
                assert np.isfinite(env.data.qpos).all() and np.isfinite(env.data.qvel).all()
                assert np.abs(env.data.qvel).max() < 1e3
                assert np.isfinite(info["human_robot_distance"])
                if te or tr:
                    break
        env.close()

    def test_guard_same_seed_same_episode(self, preset):
        def rollout(seed):
            env = PiperHumanAwarePickPlaceEnv(make_config(PRESETS[preset]))
            rng = np.random.default_rng(seed)
            obs, info = env.reset(seed=seed)
            trace = [obs.copy()]
            keys = []
            for _ in range(200):
                obs, r, te, tr, info = env.step(rng.uniform(-1, 1, env.action_space.shape))
                trace.append(obs.copy())
                keys.append((r, te, tr, info["human_robot_distance"], info["termination"]))
                if te or tr:
                    break
            env.close()
            return np.array(trace), keys

        for ep in range(min(N_PHYSICS_EPISODES, 3)):
            a, ka = rollout(800_000 + ep)
            b, kb = rollout(800_000 + ep)
            assert a.shape == b.shape and np.array_equal(a, b)
            assert ka == kb


class TestGuardNoHumanInvariance:
    """In a no-human episode the policy's inputs do not depend on the arm
    geometry, so moving the parked arm far away must not change the robot's
    motion or the outcome. If this holds, the no-human success rate cannot be
    changed by an edit to the arm geometry."""

    @pytest.fixture(scope="class")
    def policy(self):
        pytest.importorskip("stable_baselines3")
        from stable_baselines3 import SAC
        path = PROJECT_DIR / "models" / "human_aware_curriculum_hold_s3.zip"
        if not path.exists():
            pytest.skip("checkpoint not in repo")
        return SAC.load(str(path), device="cpu")

    def test_guard_no_human_outcome_invariant_to_arm_geometry(self, policy):
        cfg = make_config(PRESETS["no_human"])

        def run(seed, hide_arm):
            env = PiperHumanAwarePickPlaceEnv(cfg)
            if hide_arm:
                orig = env._set_human_state

                def far(state):
                    orig(state)
                    for name in HUMAN_BODIES:
                        env.data.mocap_pos[env.human_mocap_ids[name]] = np.array([10.0, 10.0, 10.0])
                env._set_human_state = far
            obs, _ = env.reset(seed=seed)
            qs = []
            done = False
            while not done:
                act, _ = policy.predict(obs, deterministic=True)
                obs, r, te, tr, info = env.step(act)
                qs.append(env.data.qpos.copy())
                done = te or tr
            env.close()
            return np.array(qs), bool(info["is_success"]), info["termination"]

        for ep in range(N_PHYSICS_EPISODES):
            seed = 30_000 + ep   # the pre-registered no-human suite seeds
            q_a, s_a, t_a = run(seed, hide_arm=False)
            q_b, s_b, t_b = run(seed, hide_arm=True)
            assert q_a.shape == q_b.shape and np.array_equal(q_a, q_b), f"seed {seed}"
            assert (s_a, t_a) == (s_b, t_b)

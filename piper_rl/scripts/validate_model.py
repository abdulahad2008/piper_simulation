"""Model and environment health checks. Run this first, and after any XML edit.

Replaces the original ``agilex_piper/validate_piper.py`` (which used a full 6-DoF
IK target that the PIPER's joint limits make infeasible over most of the
workspace, and therefore always reported "NEEDS TUNING").

Checks
------
1.  MJCF compiles; every named handle the env depends on exists.
2.  Actuator / joint / camera inventory.
3.  Gripper geometry: opening vs. object width, palm clearance vs. object height.
4.  Straight-down IK reachability map over (radius, height)   [``--map``]
5.  Gymnasium API conformance (``stable_baselines3.common.env_checker``).
6.  Determinism: same seed -> same trajectory.
7.  Reward sanity: a do-nothing policy must score far below the scripted one,
    and hovering must not accumulate shaping reward.
8.  Throughput benchmark.

Usage
-----
    python -m piper_rl.scripts.validate_model
    python -m piper_rl.scripts.validate_model --map
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import mujoco

from piper_rl.config import EnvConfig
from piper_rl.ik import PiperIK, ARM_JOINTS, APPROACH_AXIS_LOCAL
from piper_rl.piper_env import PiperPickPlaceEnv, TABLE_TOP, GRIP_OPEN

OK, BAD = "  ok  ", " FAIL "
_fails = []


def check(name: str, cond: bool, detail: str = "") -> bool:
    print(f"[{OK if cond else BAD}] {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        _fails.append(name)
    return cond


# --------------------------------------------------------------------------- #
def structural_checks(cfg: EnvConfig):
    print("\n--- 1. model structure " + "-" * 50)
    m = mujoco.MjModel.from_xml_path(cfg.model_path)
    d = mujoco.MjData(m)
    check("MJCF compiles", True, f"nq={m.nq} nv={m.nv} nu={m.nu} ngeom={m.ngeom}")

    needed = dict(
        site=["tcp", "target", "obj_center"],
        body=["base_link", "link6", "link7", "link8", "obj"],
        geom=["table", "floor", "obj_geom", "pad_dst", "pad_src",
              "pad_left_tip", "pad_left_base", "pad_right_tip", "pad_right_base",
              "base_col"] + [f"link{i}_col" for i in range(1, 7)],
        camera=["forehead", "wrist", "front", "side", "top", "overview"],
        actuator=[f"joint{i}" for i in range(1, 7)] + ["gripper"],
        joint=list(ARM_JOINTS) + ["joint7", "joint8", "obj_free"],
        light=["key_light"],
    )
    for kind, names in needed.items():
        missing = []
        for n in names:
            try:
                getattr(m, kind)(n)
            except (KeyError, ValueError):
                missing.append(n)
        check(f"named {kind}s present", not missing,
              "missing: " + ", ".join(missing) if missing else f"{len(names)} found")

    lim = cfg.limits
    check("action scale is consistent with the TCP speed limit",
          cfg.max_step_dist * cfg.control_hz <= lim.max_tcp_speed + 1e-9,
          f"{cfg.max_step_dist} m/step * {cfg.control_hz} Hz = "
          f"{cfg.max_step_dist*cfg.control_hz:.2f} m/s vs cap "
          f"{lim.max_tcp_speed} m/s")
    check("control timestep divides cleanly",
          abs(round((1 / cfg.control_hz) / m.opt.timestep)
              - (1 / cfg.control_hz) / m.opt.timestep) < 1e-9,
          f"{m.opt.timestep*1000:.1f} ms physics, "
          f"{1000/cfg.control_hz:.0f} ms control")

    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    check("home keyframe is collision-free at reset", d.ncon <= 6,
          f"{d.ncon} contacts (object resting on the table is 4-6)")
    return m, d


# --------------------------------------------------------------------------- #
def gripper_geometry(m, d):
    print("\n--- 2. gripper geometry " + "-" * 49)
    q7 = m.jnt_qposadr[m.joint("joint7").id]
    q8 = m.jnt_qposadr[m.joint("joint8").id]
    gl = m.geom("pad_left_tip").id
    gr = m.geom("pad_right_tip").id
    tcp = m.site("tcp").id

    seps = {}
    for op in (0.0, 0.0175, 0.035):
        d.qpos[q7], d.qpos[q8] = op, -op
        mujoco.mj_forward(m, d)
        seps[op] = float(np.linalg.norm(d.geom_xpos[gl] - d.geom_xpos[gr])) - 0.005
    print(f"       finger inner opening: closed {seps[0.0]*1000:.1f} mm | "
          f"half {seps[0.0175]*1000:.1f} mm | open {seps[0.035]*1000:.1f} mm")

    obj_w = 2 * float(m.geom_size[m.geom("obj_geom").id, 0])
    obj_h = 2 * float(m.geom_size[m.geom("obj_geom").id, 1])
    check("object fits the open gripper", seps[0.035] > obj_w + 0.008,
          f"object {obj_w*1000:.0f} mm vs opening {seps[0.035]*1000:.0f} mm "
          f"({(seps[0.035]-obj_w)/2*1000:.0f} mm clearance per side)")

    # Palm clearance: how far above the TCP does link6's collision capsule reach?
    d.qpos[q7], d.qpos[q8] = 0.035, -0.035
    mujoco.mj_forward(m, d)
    tcp_p = d.site_xpos[tcp].copy()
    l6 = m.geom("link6_col").id
    l6_p = d.geom_xpos[l6].copy()
    l6_r = float(m.geom_rbound[l6])
    palm_gap = float(np.linalg.norm(l6_p - tcp_p)) - l6_r
    check("object clears the gripper palm", obj_h / 2 < 0.046,
          f"object half-height {obj_h/2*1000:.0f} mm; palm sits ~46 mm above "
          f"the TCP (crude bound {palm_gap*1000:.0f} mm)")


# --------------------------------------------------------------------------- #
def reachability_map(m, d, full: bool = False):
    print("\n--- 3. straight-down IK reachability " + "-" * 36)
    ik = PiperIK(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    seed = d.qpos[ik.qadr].copy()

    zs = [0.235, 0.26, 0.28, 0.30, 0.32, 0.34]
    rs = [0.22, 0.26, 0.30, 0.34, 0.38, 0.42, 0.46] if full else [0.28, 0.34, 0.40]
    thetas = [-45, -20, 0, 20, 45] if full else [-40, 0, 40]

    print("       worst IK position error [mm] over azimuth "
          f"{thetas[0]}..{thetas[-1]} deg  (>= 100 means the pose is infeasible)")
    print("        r\\z  " + "".join(f"{z:>7.3f}" for z in zs))
    worst_in_task_region = 0.0
    for r in rs:
        row = []
        for z in zs:
            w = 0.0
            for th in thetas:
                t = np.array([r * np.cos(np.radians(th)),
                              r * np.sin(np.radians(th)), z])
                _, pe, ae = ik.solve(t, (0, 0, -1), q_seed=seed,
                                     qpos_full=d.qpos, iters=250)
                w = max(w, pe * 1000 + (100 if ae > np.deg2rad(3) else 0))
            row.append(w)
            if 0.30 <= r <= 0.39 and 0.235 <= z <= 0.32:
                worst_in_task_region = max(worst_in_task_region, w)
        print(f"       {r:5.2f} " + "".join(f"{v:>7.1f}" for v in row))

    check("task workspace (r 0.30-0.39, z 0.235-0.32) is fully reachable",
          worst_in_task_region < 3.0,
          f"worst residual {worst_in_task_region:.1f} mm")


# --------------------------------------------------------------------------- #
def api_and_behaviour(cfg: EnvConfig):
    print("\n--- 4. gymnasium API + behaviour " + "-" * 40)
    from stable_baselines3.common.env_checker import check_env
    env = PiperPickPlaceEnv(cfg)
    try:
        check_env(env, warn=True)
        check("stable_baselines3 env_checker", True,
              f"obs {env.observation_space.shape}, act {env.action_space.shape}")
    except Exception as e:                                   # pragma: no cover
        check("stable_baselines3 env_checker", False, str(e)[:120])

    # determinism
    def rollout(seed):
        o, _ = env.reset(seed=seed)
        rng = np.random.default_rng(123)
        rs, xs = [], []
        for _ in range(40):
            a = rng.uniform(-1, 1, env.action_space.shape[0])
            o, r, te, tr, _ = env.step(a)
            rs.append(r)
            xs.append(env.tcp_pos.copy())
            if te or tr:
                break
        return np.array(rs), np.array(xs)

    r1, x1 = rollout(7)
    r2, x2 = rollout(7)
    check("deterministic given a seed",
          np.allclose(r1, r2) and np.allclose(x1, x2),
          f"max reward delta {np.abs(r1-r2).max():.2e}")

    # observation finiteness
    o, _ = env.reset(seed=0)
    finite = True
    for _ in range(200):
        o, r, te, tr, _ = env.step(env.action_space.sample())
        finite &= bool(np.all(np.isfinite(o)) and np.isfinite(r))
        if te or tr:
            o, _ = env.reset()
    check("observations and rewards stay finite", finite)

    env.close()
    return env


# --------------------------------------------------------------------------- #
def reward_sanity(cfg: EnvConfig):
    print("\n--- 5. reward sanity " + "-" * 52)
    from piper_rl.scripts.scripted_demo import run_episode
    c = EnvConfig(**{**cfg.__dict__})
    c.curriculum = False
    c.domain_rand.enabled = False
    c.noise.enabled = False
    env = PiperPickPlaceEnv(c)

    # (a) do nothing
    idle = []
    for s in range(4):
        env.reset(seed=s)
        tot = 0.0
        for _ in range(c.max_episode_steps):
            _, r, te, tr, _ = env.step(np.array([0, 0, 0, 0, 1.0]))
            tot += r
            if te or tr:
                break
        idle.append(tot)

    # (b) hover over the object without ever grasping: the classic shaping
    #     exploit. Difference-based shaping must make this worth ~nothing.
    hover = []
    for s in range(4):
        env.reset(seed=s)
        tot = 0.0
        for i in range(c.max_episode_steps):
            tgt = env.obj_pos + np.array([0, 0, 0.05 + 0.03 * np.sin(i / 6)])
            e = tgt - env.tcp_pos
            a = np.clip(e / c.max_step_dist, -1, 1)
            _, r, te, tr, _ = env.step(np.array([a[0], a[1], a[2], 0, 1.0]))
            tot += r
            if te or tr:
                break
        hover.append(tot)

    # (c) the scripted solution
    solved = [run_episode(env, s)[0] for s in range(4)]

    print(f"       do-nothing        {np.mean(idle):8.2f}")
    print(f"       hover-and-bob     {np.mean(hover):8.2f}   (shaping exploit)")
    print(f"       scripted solution {np.mean(solved):8.2f}")
    check("solving beats hovering by a wide margin",
          np.mean(solved) > np.mean(hover) + 60,
          f"margin {np.mean(solved)-np.mean(hover):.1f}")
    check("hovering is not profitable", np.mean(hover) < 25,
          f"{np.mean(hover):.1f}")
    env.close()


# --------------------------------------------------------------------------- #
def benchmark(cfg: EnvConfig):
    print("\n--- 6. throughput " + "-" * 55)
    env = PiperPickPlaceEnv(cfg)
    env.reset(seed=0)
    n = 800
    t = time.time()
    for _ in range(n):
        _, _, te, tr, _ = env.step(env.action_space.sample())
        if te or tr:
            env.reset()
    el = time.time() - t
    print(f"       {el/n*1000:.2f} ms per env step -> {n/el:.0f} steps/s "
          f"(single env, {env._n_substeps} substeps each)")
    env.close()


# --------------------------------------------------------------------------- #
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--map", action="store_true",
                   help="print the full reachability map (slow)")
    p.add_argument("--skip-reward", action="store_true")
    args = p.parse_args(argv)

    cfg = EnvConfig()
    print("=" * 72)
    print("PIPER pick-and-place -- model & environment validation")
    print(f"model: {cfg.model_path}")
    print("=" * 72)

    m, d = structural_checks(cfg)
    gripper_geometry(m, d)
    reachability_map(m, d, full=args.map)
    api_and_behaviour(cfg)
    if not args.skip_reward:
        reward_sanity(cfg)
    benchmark(cfg)

    print("\n" + "=" * 72)
    if _fails:
        print(f"RESULT: {len(_fails)} CHECK(S) FAILED -> " + "; ".join(_fails))
        print("=" * 72)
        return 1
    print("RESULT: all checks passed -- the environment is ready to train")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())

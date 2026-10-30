"""Preview images of the hot-pan scene, rendered from the simulator.

Each preview shows three moments of one episode: the pan lifted off the stove,
the person's hand inside the 25 cm intrusion zone around the pan, and the pan
on the trivet after the hand has left. Under each image the scene-validity
checks for those three frames are printed (segment-length error, deepest
penetration of any static geometry, minimum person-table distance, minimum
hand-pan distance). A preview whose checks fail is not written.

The robot is driven by a scripted carry controller through the ordinary
action interface (5 numbers, 20 Hz, safety layer, action smoothing), so the
motion is what a policy could produce. The person is the exact fixed crossing
(preset fixed_exact_h036_v1) from the +y side, delayed so the hand crosses the
carry path while the pan is in the air. The person does not react (the
"keeps going" person type).

Figure-only colouring: upper arm in a blue sleeve, forearm and hand in skin
colour. The scene files keep the single colour.

    MUJOCO_GL=osmesa python -m piper_rl.scripts.render_previews --out out/previews
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from piper_rl.config import PROJECT_DIR
from piper_rl.human_aware_env import PiperHumanAwarePickPlaceEnv
from piper_rl.human_config import HumanAwareEnvConfig
from piper_rl.piper_env import TABLE_TOP
from piper_rl.scripts.make_pan_scene import STOVE_XY, TRIVET_XY, CENTRE_X, HEIGHT as PAN_HEIGHT

SCENES = {
    "pan": PROJECT_DIR / "agilex_piper" / "piper_human_pan_task.xml",
    "pan_torso": PROJECT_DIR / "agilex_piper" / "piper_human_pan_task_torso.xml",
}
SLEEVE = (0.25, 0.40, 0.75, 1.0)
SKIN = (0.85, 0.62, 0.48, 1.0)
INTRUSION_RADIUS = 0.25
HUMAN_GEOMS = ("human_upper_arm_geom", "human_forearm_geom", "human_hand_geom")
CARRY_Z = 0.285                                 # TCP (= pan rim) height while carrying: the crossing hand's
                                                # underside passes at 0.295 m, 1 cm above the rim
PLACE_Z = TABLE_TOP + 0.008 + PAN_HEIGHT + 0.012  # pan base 1.2 cm above the trivet top
HOLD_UNTIL_S = 4.6                              # wait on the stove so the crossing meets the carry mid-air


class PanPreviewEnv(PiperHumanAwarePickPlaceEnv):
    """Human-aware env on the pan scene with the cup parked away and the
    target site on the trivet. Preview only; the trained task env is Phase 3."""

    def _index_model(self):
        super()._index_model()
        m = self.model
        self.pan_geoms = {g for g in range(m.ngeom)
                          if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("pan_")}
        self.pan_centre_site = m.site("pan_centre").id
        self.torso_mocap = None
        try:
            self.torso_mocap = int(m.body_mocapid[m.body("human_torso").id])
        except KeyError:
            pass

    def _after_reset(self, rng, options, info_dr):
        # park the cup off the table and out of sight; it plays no part here
        self.model.geom_contype[self.obj_geom] = 0
        self.model.geom_conaffinity[self.obj_geom] = 0
        self.model.geom_rgba[self.obj_geom, 3] = 0.0
        self._set_object(np.array([-0.6, 0.6, -0.65]), 0.0)
        self.model.site_pos[self.target_site] = [TRIVET_XY[0], TRIVET_XY[1], TABLE_TOP + 0.008]
        self.target_pos = self.model.site_pos[self.target_site].copy()
        return super()._after_reset(rng, options, info_dr)

    def _set_human_state(self, state):
        super()._set_human_state(state)
        if self.torso_mocap is not None:
            self.data.mocap_pos[self.torso_mocap] = state.shoulder_position
            self.data.mocap_quat[self.torso_mocap] = [1, 0, 0, 0]

    def _check_done(self, rew_info):
        if self._step_count >= self.cfg.max_episode_steps:
            return False, True, "timeout"
        return False, False, ""

    # --- measurements ---------------------------------------------------- #
    def pan_centre(self) -> np.ndarray:
        return self.data.site_xpos[self.pan_centre_site].copy()

    def hand_pan_distance(self) -> float:
        fromto = np.zeros(6)
        best = np.inf
        for g in self.pan_geoms:
            best = min(best, mujoco.mj_geomDistance(self.model, self.data, self.human_hand_geom, g, 2.0, fromto))
        return float(best)

    def person_pan_distance(self) -> float:
        fromto = np.zeros(6)
        best = np.inf
        for h in self.human_geoms:
            for g in self.pan_geoms:
                best = min(best, mujoco.mj_geomDistance(self.model, self.data, h, g, 2.0, fromto))
        return float(best)


def scripted_action(env: PanPreviewEnv, t: float) -> np.ndarray:
    """Carry the pan: hold over the stove, lift, move over the trivet, lower.
    The pan centre (not the TCP) is steered in x, y; the TCP height in z."""
    tcp = env.tcp_pos
    pan = env.pan_centre()
    if t < HOLD_UNTIL_S:
        goal_xy, goal_z = np.array(STOVE_XY), PLACE_Z
    elif t < HOLD_UNTIL_S + 0.6:
        goal_xy, goal_z = np.array(STOVE_XY), CARRY_Z
    else:
        goal_xy = np.array(TRIVET_XY)
        goal_z = CARRY_Z if np.linalg.norm(pan[:2] - goal_xy) > 0.02 else PLACE_Z
    delta = np.array([goal_xy[0] - pan[0], goal_xy[1] - pan[1], goal_z - tcp[2]])
    act = np.zeros(5)
    act[:3] = np.clip(delta / env.cfg.max_step_dist * 0.5, -1.0, 1.0)
    act[4] = -1.0   # gripper closed on the handle
    return act


def checks_for_frame(env: PanPreviewEnv) -> dict:
    m, d = env.model, env.data
    st = env.human_state
    gids = [m.geom(n).id for n in HUMAN_GEOMS]
    # drawn segment length vs joint distance
    errs = []
    for gid, (a, b) in zip(gids[:2], ((st.shoulder_position, st.elbow_position),
                                      (st.elbow_position, st.hand_position))):
        errs.append(abs(2 * m.geom_size[gid, 1] - max(np.linalg.norm(b - a), 0.02)))
    # signed distances to every static physical geom
    fromto = np.zeros(6)
    statics = [g for g in range(m.ngeom) if m.body_geomadr[0] <= g < m.body_geomadr[0] + m.body_geomnum[0]
               and (m.geom_contype[g] or m.geom_conaffinity[g])]
    min_static = np.inf
    min_table = np.inf
    table = m.geom("table").id
    for h in gids:
        for s in statics:
            dist = mujoco.mj_geomDistance(m, d, h, s, 2.0, fromto)
            min_static = min(min_static, dist)
            if s == table:
                min_table = min(min_table, dist)
    return {
        "segment_length_error_mm": float(max(errs) * 1000),
        "deepest_penetration_mm": float(max(0.0, -min_static) * 1000),
        "min_person_table_mm": float(min_table * 1000),
        "min_hand_pan_mm": float(env.hand_pan_distance() * 1000),
        "pan_centre_to_trivet_mm": float(np.linalg.norm(env.pan_centre()[:2] - np.array(TRIVET_XY)) * 1000),
        "pan_base_above_table_mm": float((env.pan_centre()[2] - TABLE_TOP) * 1000),
    }


def run_episode(scene: str, seed: int, camera: str, width: int, height: int,
                colour_arm: bool = True, hide_torso: bool = False):
    cfg = HumanAwareEnvConfig.from_preset("fixed_exact_h036_v1").eval_variant()
    cfg.model_path = str(SCENES[scene])
    cfg.curriculum = False
    cfg.domain_rand.enabled = False
    cfg.noise.enabled = False
    cfg.human_motion.appearance_time_range = (1.2, 1.2)
    cfg.limits.tcp_table_clearance = 0.065     # the pan base hangs 5.5 cm below the TCP
    # The training env ends the control step at the first robot-person contact.
    # A preview must keep simulating through it, or the person freezes.
    cfg.human_safety.terminate_on_human_collision = False
    env = PanPreviewEnv(cfg)
    m = env.model
    renderer = mujoco.Renderer(m, height, width)
    env.reset(seed=seed)
    if colour_arm:   # after reset: the domain randomiser restores model colours at every reset
        for name, rgba in (("human_upper_arm_geom", SLEEVE), ("human_forearm_geom", SKIN), ("human_hand_geom", SKIN)):
            gid = m.geom(name).id
            m.geom_matid[gid] = -1          # otherwise the material colour wins
            m.geom_rgba[gid] = rgba
    frames, log = [], []
    for k in range(env.cfg.max_episode_steps):
        t = k * env.dt
        _, _, te, tr, info = env.step(scripted_action(env, t))
        renderer.update_scene(env.data, camera=camera)
        frames.append(renderer.render().copy())
        row = checks_for_frame(env)
        row.update(t=float(env.data.time), tcp=env.tcp_pos.tolist(), pan=env.pan_centre().tolist(), hand=env.human_state.hand_position.tolist(),
                   human_present=bool(env.human_state.human_present), robot_contact=bool(info["human_collision"]),
                   person_pan_mm=env.person_pan_distance() * 1000)
        log.append(row)
        if tr:
            break
    env.close()
    return frames, log


def pick_moments(log: list[dict]) -> tuple[int, int, int]:
    """Frame indices: pan just lifted, closest hand-pan approach, pan placed."""
    t = np.array([r["t"] for r in log])
    lifted = int(np.argmax(t >= HOLD_UNTIL_S + 0.55))
    first_contact = next((i for i, r in enumerate(log) if r["person_pan_mm"] <= 0), len(log))
    inside = [i for i, r in enumerate(log)
              if r["human_present"] and r["t"] >= HOLD_UNTIL_S + 0.6 and i < first_contact
              and 0 < r["person_pan_mm"] < INTRUSION_RADIUS * 1000]
    # the closest contact-free approach inside the zone while the pan is carried
    # (a later contact, if any, is reported in the caption)
    closest = int(min(inside, key=lambda i: log[i]["person_pan_mm"])) if inside else int(np.argmin([r["person_pan_mm"] for r in log]))
    return lifted, closest, len(log) - 1


def compose(frames, log, idx, title, caption, out_path: Path, extra_draw=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.9))
    labels = ["(1) pan lifted off the stove", "(2) hand inside the 25 cm zone, pan in the air", "(3) end of the episode: pan on the trivet"]
    for ax, i, lab in zip(axes, idx, labels):
        ax.imshow(frames[i])
        if extra_draw:
            extra_draw(ax, i)
            ax.set_xlim(0, frames[i].shape[1])
            ax.set_ylim(frames[i].shape[0], 0)
        r = log[i]
        ax.set_title(f"{lab}\nt = {r['t']:.2f} s", fontsize=11)
        ax.axis("off")
        txt = (f"segment-length error {r['segment_length_error_mm']:.2f} mm\n"
               f"penetration of static geometry {r['deepest_penetration_mm']:.2f} mm\n"
               f"min person–table {r['min_person_table_mm']:.0f} mm\n"
               f"min hand–pan {r['min_hand_pan_mm']:.0f} mm")
        ax.text(0.01, -0.02, txt, transform=ax.transAxes, va="top", ha="left", fontsize=9, family="monospace")
    fig.suptitle(title, fontsize=13, y=0.985)
    fig.text(0.5, 0.005, caption, ha="center", va="bottom", fontsize=9.5, wrap=True)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.2, wspace=0.03)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def contact_note(log) -> str:
    hits = [r for r in log if r["person_pan_mm"] <= 0]
    if not hits:
        return "The person never touches the pan in this episode."
    first = hits[0]
    return (f"This keep-working robot and non-reacting person do meet: the person's arm first touches the pan at "
            f"t = {first['t']:.2f} s (contact detected, force-free, see make_pan_scene.py); the pan still reaches the trivet.")


def validate(log, idx) -> list[str]:
    bad = []
    for i in idx:
        r = log[i]
        if r["segment_length_error_mm"] > 1.0:
            bad.append(f"frame {i}: segment-length error {r['segment_length_error_mm']:.2f} mm")
        if r["deepest_penetration_mm"] > 1e-3:
            bad.append(f"frame {i}: penetration {r['deepest_penetration_mm']:.2f} mm")
        if r["min_person_table_mm"] < 0:
            bad.append(f"frame {i}: person below the table top")
    return bad


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="out/previews")
    p.add_argument("--seed", type=int, default=31000)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=720)
    args = p.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {}

    variants = {
        "a_overview": dict(scene="pan", camera="overview",
                           title="(a) Hot-pan scene, overview camera (the preset used in the existing figures)",
                           caption="Counter at 0.90 m above the floor (EN 1116). Stove plate with hob ring (left), trivet (right), "
                                   "20 cm pan (525 g) held by the handle. Person: exact fixed crossing at 0.22 m/s from the +y side, "
                                   "no reaction. Robot: scripted carry through the 5-number action interface. Blue sleeve and skin colour are render-only."),
        "b_counter": dict(scene="pan", camera="counter",
                          title="(b) The same episode from a pulled-back camera showing the full counter height",
                          caption="Same simulation as (a); only the camera differs. The floor is 0.70 m below the robot base, "
                                  "so the worktop is at 0.90 m. The person's shoulder is 1.24 m above the floor (a 5th-percentile woman)."),
        "c_torso": dict(scene="pan_torso", camera="counter",
                        title="(c) With a visual-only torso, head and legs for context",
                        caption="Torso, head and legs are VISUAL ONLY: no collision, not seen by the robot, not in any metric. "
                                "They follow the arm's shoulder point. Everything else is identical to (b)."),
    }
    cache = {}
    for name, v in variants.items():
        key = (v["scene"], v["camera"])
        if key not in cache:
            cache[key] = run_episode(v["scene"], args.seed, v["camera"], args.width, args.height)
        frames, log = cache[key]
        idx = pick_moments(log)
        bad = validate(log, idx)
        report[name] = {"frames": list(idx), "checks": [log[i] for i in idx], "failed": bad}
        if bad:
            print(f"{name}: NOT WRITTEN, checks failed: {bad}")
            continue
        compose(frames, log, idx, v["title"], v["caption"] + " " + contact_note(log), out / f"preview_{name}.png")
        print("wrote", out / f"preview_{name}.png", "frames", idx)

    # (d) top-down: simulator render from the top camera, hand paths and intrusion zone drawn on it
    frames, log = run_episode("pan", args.seed, "top", args.width, args.height)
    idx = pick_moments(log)
    bad = validate(log, idx)
    report["d_topdown"] = {"frames": list(idx), "checks": [log[i] for i in idx], "failed": bad}
    if not bad:
        cam_pos = np.array([0.42, 0.0, 1.35])    # the `top` camera, looking straight down, fovy 45
        fovy = 45.0

        def world_to_px(xy, z):
            # pinhole: top camera xyaxes = (1 0 0, 0 1 0): image x = world x, image y = -world y
            depth = cam_pos[2] - z
            f = (args.height / 2) / np.tan(np.radians(fovy / 2))
            u = args.width / 2 + f * (xy[0] - cam_pos[0]) / depth
            vpx = args.height / 2 - f * (xy[1] - cam_pos[1]) / depth
            return u, vpx

        hand = np.array([r["hand"] for r in log if r["human_present"]])
        pan_path = np.array([r["pan"] for r in log])

        def draw(ax, i):
            hp = np.array([world_to_px(h[:2], h[2]) for h in hand])
            ax.plot(hp[:, 0], hp[:, 1], "-", color="gold", lw=2, label="hand path")
            pp = np.array([world_to_px(q[:2], q[2]) for q in pan_path])
            ax.plot(pp[:, 0], pp[:, 1], "--", color="white", lw=1.5, label="pan-centre path")
            pc = log[i]["pan"]
            cx, cy = world_to_px(np.array(pc[:2]), pc[2])
            depth = cam_pos[2] - pc[2]
            f = (args.height / 2) / np.tan(np.radians(fovy / 2))
            import matplotlib.patches as mp
            ax.add_patch(mp.Circle((cx, cy), INTRUSION_RADIUS * f / depth, fill=False, ec="red", lw=2, ls=":"))
            if i == idx[0]:
                ax.legend(loc="lower left", fontsize=8)

        compose(frames, log, idx,
                "(d) Top-down view with the hand path, the pan-centre path and the 25 cm intrusion zone",
                "Rendered from the `top` camera preset; the yellow line is the hand path of this episode, the white dashed line the pan "
                "centre's path, the red dotted circle the 25 cm intrusion zone around the pan centre (projected at the pan's height). "
                "The zone only starts the reaction clock; it is not a stop distance.",
                out / "preview_d_topdown.png", extra_draw=draw)
        print("wrote", out / "preview_d_topdown.png", "frames", idx)
    else:
        print("d_topdown: NOT WRITTEN", bad)
    (out / "preview_checks.json").write_text(json.dumps(report, indent=2))
    print("checks:", out / "preview_checks.json")


if __name__ == "__main__":
    main()

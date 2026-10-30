"""Hand-written reactions on the hot-pan task: keep working, stop, retract.

Runs each reaction against each person type on a test set and writes one row
per episode (CSV) plus a summary (JSON). No training is involved; this is the
first stop-versus-retract table of the kitchen-hazard study (Phase 3, box 8).

Task (pan preview env): the pan starts in the gripper over the stove; the
robot carries it to the trivet and lowers it. Success = pan centre within
4.5 cm of the trivet centre with the TCP moving slower than 2 cm/s, held to
the end of the episode (10 s). The robot starts its carry at a random time in
[0, 3] s (seeded), so the person's arrival overlaps the carry in a fair share
of episodes; the share is reported.

Reactions (all through the 5-number action interface, 20 Hz):
  keep     the scripted carry, blind to the person
  stop     the carry, but hold the current joint position whenever the
           person-pan distance is below the ISO/TS 15066 protective distance
           S = v_h (T_r + T_s) + v_r T_r + S_s with the measured values
           T_r = 0.05 s, T_s = 0.30 s, v_r = 0.30 m/s, S_s = 6 mm and the hand
           speed v_h from the trajectory; resume when the distance exceeds
           S + 5 cm
  retract  the carry, but when the hand enters the 25 cm intrusion zone move
           the pan away from the hand at full speed (horizontal unit vector
           from hand to pan centre, plus up) until the distance exceeds 30 cm,
           then resume. The ISO distance is sized for stopping, not escaping:
           triggered at S, a retract starts about 0.45 s before contact and
           cannot clear a 0.22 m/s hand (smoke test, 30 Oct).

Person types (the human clock runs the planned path at rate +1, 0 or -1):
  keeps_going   today's person: never reacts
  halts         stops moving 0.25 s after the pan comes within 25 cm of the
                hand, resumes when it is further than 30 cm
  pulls_back    runs the path backwards 0.25 s after the same trigger until
                parked, then resumes forwards when clear

Per episode: success, completion time, intrusion time (first person-pan
distance < 25 cm), whether the intrusion came before completion, reaction
time (trigger to first reaction step), stopped time, closest person-to-hot-
surface distance (base and walls), closest person-to-handle distance,
contacts with the hot surface, the handle and the robot links (counts of
control steps), and hot-surface exposure (steps with distance < 3 cm) in the
1 s after intrusion and overall.

    MUJOCO_GL=osmesa python -m piper_rl.scripts.reaction_baselines --set exact --episodes 200 --out out/reactions
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from piper_rl.human_config import HumanAwareEnvConfig
from piper_rl.scripts.make_pan_scene import CENTRE_X, HEIGHT as PAN_HEIGHT, STOVE_XY, TRIVET_XY, TRIVET_TOP, TOP_R
from piper_rl.scripts.render_previews import PanPreviewEnv, SCENES, CARRY_Z, PLACE_Z

TEST_SETS = {
    # name: (preset, default episodes, seed)
    "exact": ("fixed_exact_h036_v1", 200, 31_000),
    "random": ("evaluation_id", 500, 40_000),
    "ood": ("evaluation_ood", 500, 50_000),
}
INTRUSION_R = 0.25
HUMAN_REACT_DELAY = 0.25
HUMAN_RELEASE_R = 0.30
T_R, T_S, V_R, S_S = 0.05, 0.30, 0.30, 0.006
RESUME_MARGIN = 0.05
STOPPED_SPEED = 0.02
SUCCESS_R = 0.045
EXPOSURE_R = 0.03


class ReactionEnv(PanPreviewEnv):
    """Pan preview env plus the human clock and cheap per-substep distances."""

    def __init__(self, cfg, person_type: str):
        self.person_type = person_type
        super().__init__(cfg)

    def _index_model(self):
        super()._index_model()
        m = self.model
        self.hot_geoms = [g for g in self.pan_geoms
                          if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) != "pan_handle"]
        self.handle_geom = m.geom("pan_handle").id

    def _reset_extension_state(self):
        super()._reset_extension_state()
        self.hold_q = None
        self.t_h = 0.0                 # the human clock
        self.rate = 1.0
        self.react_since = None        # sim time the trigger first held
        self.reacting = False
        self.min_hot = np.inf
        self.min_handle = np.inf
        self.hot_contact_steps = 0
        self.handle_contact_steps = 0
        self.robot_contact_steps = 0

    def _after_reset(self, rng, options, info_dr):
        info = super()._after_reset(rng, options, info_dr)
        self.t_h = 0.0
        self.rate = 1.0
        self.react_since = None
        self.reacting = False
        return info

    def hand_to_pan_disc(self, hand: np.ndarray) -> float:
        """Distance from the hand centre to the pan's top disc (cheap trigger)."""
        pc = self.pan_centre()
        pc = pc + np.array([0, 0, PAN_HEIGHT])      # rim plane
        d = hand - pc
        radial = max(0.0, np.hypot(d[0], d[1]) - TOP_R)
        return float(np.hypot(radial, d[2]))

    def step(self, action):
        if self.hold_q is not None:
            # the scripted stop: hold the measured joint position, bypassing
            # the low-pass filter (a zero action alone creeps, Phase 1 audit 3)
            self._smoothed_q = self.hold_q.copy()
            self._q_cmd = self.hold_q.copy()
        return super().step(action)

    def _before_physics_substep(self, substep):
        if substep == 0:
            self._human_collision_this_step = False
            self._step_tcp_start = self.tcp_pos
        sim_time = float(self.data.time - self._human_time_zero)
        dt = float(self.model.opt.timestep)
        traj = self.human_trajectory
        if self.person_type == "keeps_going" or sim_time < traj.appearance_time:
            self.t_h = sim_time
        else:
            st = self.human_state
            d = self.hand_to_pan_disc(st.hand_position) if st.human_present else np.inf
            if d < INTRUSION_R and self.react_since is None:
                self.react_since = sim_time
            if d > HUMAN_RELEASE_R:
                self.react_since = None
                self.reacting = False
            if self.react_since is not None and sim_time - self.react_since >= HUMAN_REACT_DELAY:
                self.reacting = True
            if self.reacting:
                self.rate = 0.0 if self.person_type == "halts" else -1.0
            else:
                self.rate = 1.0
            # the clock never runs back past the appearance (start pose)
            self.t_h = max(traj.appearance_time, self.t_h + self.rate * dt)
        self._set_human_state(traj.state(self.t_h))

    def _after_physics_substep(self, substep):
        self._update_human_contacts_and_distance()
        return False

    def control_step_metrics(self) -> dict:
        """Exact distances and contacts, once per control step."""
        m, d = self.model, self.data
        fromto = np.zeros(6)
        hot = np.inf
        handle = np.inf
        for h in self.human_geoms:
            for g in self.hot_geoms:
                hot = min(hot, mujoco.mj_geomDistance(m, d, h, g, 1.0, fromto))
            handle = min(handle, mujoco.mj_geomDistance(m, d, h, self.handle_geom, 1.0, fromto))
        self.min_hot = min(self.min_hot, hot)
        self.min_handle = min(self.min_handle, handle)
        hot_c = handle_c = robot_c = False
        for i in range(d.ncon):
            c = d.contact[i]
            pair = {int(c.geom1), int(c.geom2)}
            if not (pair & self.human_geoms):
                continue
            other = next(g for g in pair if g not in self.human_geoms)
            hot_c |= other in self.hot_geoms
            handle_c |= other == self.handle_geom
            robot_c |= other in self.robot_geoms
        self.hot_contact_steps += int(hot_c)
        self.handle_contact_steps += int(handle_c)
        self.robot_contact_steps += int(robot_c)
        return {"hot": hot, "handle": handle, "hot_contact": hot_c, "handle_contact": handle_c, "robot_contact": robot_c}


class Carry:
    """The scripted carry with a per-episode start delay."""

    def __init__(self, env: ReactionEnv, start_delay: float):
        self.env = env
        self.start = start_delay

    def action(self, t: float) -> tuple[np.ndarray, bool]:
        env = self.env
        tcp = env.tcp_pos
        pan = env.pan_centre()
        if t < self.start:
            goal_xy, goal_z = np.array(STOVE_XY), PLACE_Z
        elif t < self.start + 0.6:
            goal_xy, goal_z = np.array(STOVE_XY), CARRY_Z
        else:
            goal_xy = np.array(TRIVET_XY)
            goal_z = CARRY_Z if np.linalg.norm(pan[:2] - goal_xy) > 0.02 else PLACE_Z
        delta = np.array([goal_xy[0] - pan[0], goal_xy[1] - pan[1], goal_z - tcp[2]])
        act = np.zeros(5)
        act[:3] = np.clip(delta / env.cfg.max_step_dist * 0.5, -1.0, 1.0)
        act[4] = -1.0
        placed = np.linalg.norm(pan[:2] - np.array(TRIVET_XY)) < SUCCESS_R and abs(tcp[2] - PLACE_Z) < 0.01
        return act, placed


def protective_distance(hand_speed: float) -> float:
    return hand_speed * (T_R + T_S) + V_R * T_R + S_S


def run_episode(env: ReactionEnv, reaction: str, seed: int) -> dict:
    obs, info = env.reset(seed=seed)
    rng = np.random.default_rng(seed + 7)
    start_delay = float(rng.uniform(0.0, 3.0))
    carry = Carry(env, start_delay)
    dt = env.dt
    stopped_steps = 0
    reacting = False
    t_intrusion = None
    t_react = None
    t_complete = None
    placed_steps = 0
    exposure_total = 0
    exposure_1s = 0
    min_tcp_speed_when_placed = np.inf
    prev_tcp = env.tcp_pos.copy()
    for k in range(env.cfg.max_episode_steps):
        t = k * dt
        st = env.human_state
        act, placed = carry.action(t)
        hand = st.hand_position
        d_disc = env.hand_to_pan_disc(hand) if st.human_present else np.inf
        v_h = float(np.linalg.norm(st.hand_velocity)) if st.human_present else 0.0
        if st.human_present and d_disc < INTRUSION_R and t_intrusion is None:
            t_intrusion = t
        S = protective_distance(v_h)
        if reaction == "retract":
            trigger = st.human_present and d_disc < INTRUSION_R
            release = (not st.human_present) or d_disc > HUMAN_RELEASE_R
        else:
            trigger = st.human_present and d_disc < S
            release = (not st.human_present) or d_disc > S + RESUME_MARGIN
        if reaction in ("stop", "retract"):
            if trigger and not reacting:
                reacting = True
                if t_react is None and t_intrusion is not None:
                    t_react = t
            elif reacting and release:
                reacting = False
            if reacting:
                if reaction == "stop":
                    env.hold_q = env.data.qpos[env.arm_qadr].copy()
                    act = np.zeros(5); act[4] = -1.0
                    stopped_steps += 1
                else:
                    env.hold_q = None
                    away = env.pan_centre() - hand
                    away[2] = 0.0
                    n = np.linalg.norm(away)
                    away = away / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])
                    act = np.zeros(5)
                    act[:2] = away[:2]
                    act[2] = 0.6
                    act[4] = -1.0
            else:
                env.hold_q = None
        obs, r, te, tr, info = env.step(act)
        if reaction == "stop" and env.hold_q is not None:
            env.hold_q = None
        met = env.control_step_metrics()
        tcp_speed = float(np.linalg.norm(env.tcp_pos - prev_tcp) / dt)
        prev_tcp = env.tcp_pos.copy()
        if met["hot"] < EXPOSURE_R:
            exposure_total += 1
            if t_intrusion is not None and t <= t_intrusion + 1.0:
                exposure_1s += 1
        if placed and tcp_speed < STOPPED_SPEED:
            placed_steps += 1
            if t_complete is None:
                t_complete = t + dt
        else:
            placed_steps = 0
        if tr:
            break
    pan_final = np.linalg.norm(env.pan_centre()[:2] - np.array(TRIVET_XY))
    success = pan_final < SUCCESS_R and placed_steps >= 10   # held for the last 0.5 s
    return {
        "seed": seed, "reaction": reaction, "person_type": env.person_type,
        "trajectory_type": env.human_trajectory.trajectory_type,
        "start_delay_s": start_delay,
        "success": int(success),
        "completion_time_s": t_complete if t_complete is not None else "",
        "intrusion_time_s": t_intrusion if t_intrusion is not None else "",
        "intrusion_before_completion": int(t_intrusion is not None and (t_complete is None or t_intrusion < t_complete)),
        "reaction_time_s": (t_react - t_intrusion) if (t_react is not None and t_intrusion is not None) else "",
        "stopped_time_s": stopped_steps * dt,
        "min_hot_surface_distance_m": env.min_hot,
        "min_handle_distance_m": env.min_handle,
        "hot_contact_steps": env.hot_contact_steps,
        "handle_contact_steps": env.handle_contact_steps,
        "robot_contact_steps": env.robot_contact_steps,
        "exposure_1s_after_intrusion_s": exposure_1s * dt,
        "exposure_total_s": exposure_total * dt,
        "pan_final_error_m": pan_final,
    }


def make_env(preset: str, person_type: str) -> ReactionEnv:
    cfg = HumanAwareEnvConfig.from_preset(preset).eval_variant()
    cfg.model_path = str(SCENES["pan"])
    cfg.curriculum = False
    cfg.domain_rand.enabled = False
    cfg.noise.enabled = False
    cfg.limits.tcp_table_clearance = 0.065
    cfg.human_safety.terminate_on_human_collision = False
    return ReactionEnv(cfg, person_type)


def summarise(rows: list[dict]) -> dict:
    out = {}
    keys = sorted({(r["reaction"], r["person_type"]) for r in rows})
    for reaction, person in keys:
        sel = [r for r in rows if r["reaction"] == reaction and r["person_type"] == person]
        n = len(sel)
        intr = [r for r in sel if r["intrusion_before_completion"]]
        def mean(xs):
            xs = [x for x in xs if x != "" and x is not None]
            return float(np.mean(xs)) if xs else None
        out[f"{reaction}/{person}"] = {
            "episodes": n,
            "success_pct": 100 * sum(r["success"] for r in sel) / n,
            "intrusion_before_completion_pct": 100 * len(intr) / n,
            "success_pct_given_intrusion": (100 * sum(r["success"] for r in intr) / len(intr)) if intr else None,
            "hot_contact_pct": 100 * sum(r["hot_contact_steps"] > 0 for r in sel) / n,
            "handle_contact_pct": 100 * sum(r["handle_contact_steps"] > 0 for r in sel) / n,
            "robot_contact_pct": 100 * sum(r["robot_contact_steps"] > 0 for r in sel) / n,
            "mean_exposure_1s_after_intrusion_s": mean([r["exposure_1s_after_intrusion_s"] for r in intr]),
            "mean_exposure_total_s": mean([r["exposure_total_s"] for r in sel]),
            "p5_min_hot_distance_mm": 1000 * float(np.percentile([r["min_hot_surface_distance_m"] for r in sel], 5)),
            "mean_completion_time_s": mean([r["completion_time_s"] for r in sel]),
            "mean_reaction_time_s": mean([r["reaction_time_s"] for r in intr]),
            "mean_stopped_time_s": mean([r["stopped_time_s"] for r in sel]),
        }
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set", choices=list(TEST_SETS), default="exact")
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--reactions", nargs="+", default=["keep", "stop", "retract"])
    p.add_argument("--person-types", nargs="+", default=["keeps_going", "halts", "pulls_back"])
    p.add_argument("--out", default="out/reactions")
    args = p.parse_args(argv)
    preset, n_default, seed0 = TEST_SETS[args.set]
    n = args.episodes or n_default
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.time()
    for person in args.person_types:
        env = make_env(preset, person)
        for reaction in args.reactions:
            for ep in range(n):
                rows.append(run_episode(env, reaction, seed0 + ep))
            print(f"{args.set} {person:<12}{reaction:<8} {n} episodes done ({time.time()-t0:.0f} s)")
        env.close()
    csv_path = out / f"reactions_{args.set}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = summarise(rows)
    (out / f"reactions_{args.set}_summary.json").write_text(json.dumps(
        {"set": args.set, "preset": preset, "episodes_per_cell": n, "seed0": seed0,
         "parameters": {"T_r": T_R, "T_s": T_S, "v_r": V_R, "S_s": S_S, "intrusion_r": INTRUSION_R,
                        "human_delay": HUMAN_REACT_DELAY, "exposure_r": EXPOSURE_R},
         "cells": summary}, indent=1))
    print(f"\n{'cell':<24}{'succ%':>7}{'intr%':>7}{'succ|intr':>10}{'hot%':>6}{'hndl%':>7}{'robot%':>7}{'exp1s':>7}{'p5hot mm':>9}{'t_done':>7}{'t_react':>8}")
    for k, v in summary.items():
        f = lambda x, fmt: (fmt % x) if x is not None else "-"
        print(f"{k:<24}{v['success_pct']:>7.1f}{v['intrusion_before_completion_pct']:>7.1f}{f(v['success_pct_given_intrusion'],'%.1f'):>10}"
              f"{v['hot_contact_pct']:>6.1f}{v['handle_contact_pct']:>7.1f}{v['robot_contact_pct']:>7.1f}"
              f"{f(v['mean_exposure_1s_after_intrusion_s'],'%.2f'):>7}{v['p5_min_hot_distance_mm']:>9.0f}"
              f"{f(v['mean_completion_time_s'],'%.2f'):>7}{f(v['mean_reaction_time_s'],'%.2f'):>8}")
    print("written", csv_path)


if __name__ == "__main__":
    main()

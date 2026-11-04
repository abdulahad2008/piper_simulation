"""Hot-pan carry task with the kinematic person: the Phase 2/3 training task.

The robot holds a frying pan by its handle (welded in the gripper, see
``piper_rl/scripts/make_pan_scene.py``) and must carry it from the stove to the
trivet while the seeded human arm shares the counter. Compared with the cup
task it keeps the robot, the table, the control rate, the action interface and
the observation layout unchanged; only the held object, its reward and the
scene differ.

What is deliberately *not* modelled here (Phase 1 audit, §1.7): the person is
a sensor, not an obstacle. Person-pan contacts are detected (``gap="1"``) and
counted by class (hot surface / handle / robot body) but exert no force.

Observation (same 46 + n_act + 13 layout as the cup task, so the trained
networks, the float64 actor export and ``train.py`` need no changes):

    q_norm 6 | dq_norm 6 | grip 2 | tcp 3 | approach 3 | pinch 3 |
    pan centre 3 | pan up-axis 3 | pan velocity 3 | trivet 3 | pan - tcp 3 |
    trivet - pan 3 | [lifted, over_trivet] 2 | grip cmd 1 | [pan radius, pan
    height] 2 | previous action n_act | human state 13

Success: pan centre within ``reward.place_tol_xy`` (4.5 cm) of the trivet
centre, pan base between 5 mm below and 2 cm above the trivet top, pan speed
below ``reward.settle_vel``, for 3 consecutive control steps. There is no
release: the pan stays in the gripper (the study is about the carry, not the
grasp).

Reward: a single potential (so it cannot be farmed by cycling) plus one-shot
bonuses and the same per-step penalties as the cup task:

    phi(s) = -w_transport * d_xy
             + w_lift * [ min(h, H) * s(d_xy) + (H - min(h, H)) * (1 - s(d_xy)) ]

where h is the pan base height above the table, H = reward.lift_height, and
s(d) rises smoothly from 0 inside the placing radius to 1 beyond 2x that
radius: far from the trivet, height is rewarded (carry high); at the trivet,
lowering is rewarded. The shaping reward is phi(s_t) - phi(s_{t-1}).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import gymnasium as gym
import mujoco
import numpy as np


from piper_rl.human_aware_env import PiperHumanAwarePickPlaceEnv
from piper_rl.human_config import HumanAwareEnvConfig
from piper_rl.piper_env import APPROACH_AXIS_LOCAL, GRIP_OPEN, PINCH_AXIS_LOCAL, TABLE_TOP
from piper_rl.scripts.make_pan_scene import (
    HEIGHT as PAN_HEIGHT, STOVE_XY, TOP_R, TRIVET_TOP, TRIVET_XY)

PROJECT_DIR = Path(__file__).resolve().parents[1]
PAN_SCENE = PROJECT_DIR / "agilex_piper" / "piper_human_pan_task.xml"

CARRY_CLEARANCE = 0.085      # pan base above the table while carrying (rim at 0.285+)
PLACE_CLEARANCE = 0.012      # pan base above the trivet top at the start pose
EXPOSURE_R = 0.03            # person within this of a hot surface = exposed (hazard cost 1)
PAN_START_PHASES = ("stove", "stove_lifted", "mid_carry", "over_trivet")


@dataclass
class PanEnvConfig(HumanAwareEnvConfig):
    """Human-aware config with the pan scene and the pan's TCP clearance.

    The pan base hangs 5.35 cm below the TCP, so the safety layer's table
    clearance must grow from 1.5 cm to 6.5 cm or the layer lets the pan
    scrape the counter. Everything else is inherited, including every human
    preset (``PanEnvConfig.from_preset("curriculum")``).
    """

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        self.model_path = str(PAN_SCENE)
        self.limits.tcp_table_clearance = 0.065


class PiperPanPlaceEnv(PiperHumanAwarePickPlaceEnv):
    """Carry the welded hot pan from the stove to the trivet near a person."""

    def __init__(self, cfg: Optional[PanEnvConfig] = None,
                 render_mode: Optional[str] = None):
        cfg = cfg or PanEnvConfig()
        if not isinstance(cfg, PanEnvConfig):
            raise TypeError("PiperPanPlaceEnv requires PanEnvConfig")
        super().__init__(cfg, render_mode=render_mode)

    # ------------------------------------------------------------ indexing
    def _index_model(self) -> None:
        super()._index_model()
        m = self.model
        self.pan_body = m.body("pan").id
        self.pan_centre_site = m.site("pan_centre").id
        self.pan_geoms = {g for g in range(m.ngeom)
                          if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("pan_")}
        self.handle_geom = m.geom("pan_handle").id
        self.hot_geoms = sorted(self.pan_geoms - {self.handle_geom})
        fixtures = []
        for name in ("stove", "hob", "trivet"):
            try:
                fixtures.append(m.geom(name).id)
            except KeyError:
                pass
        self.fixture_geoms = set(fixtures)
        # arm links colliding with the counter fixtures is a collision too
        self.env_geoms = set(self.env_geoms) | self.fixture_geoms
        self.trivet_xyz = np.array([TRIVET_XY[0], TRIVET_XY[1], TRIVET_TOP])
        self._pan_fromto = np.zeros(6)

    # ------------------------------------------------------------- episode
    def _reset_extension_state(self) -> None:
        super()._reset_extension_state()
        self._pan_phase = "stove"
        self._prev_phi = 0.0
        self._prev_dxy = 0.0
        self._prev_pan = np.zeros(3)
        self._pan_speed = 0.0
        self._pan_vel = np.zeros(3)
        self._hot_contact_this_step = False
        self._n_place_steps = 0
        self._pan_success = False
        self._pan_bonus_paid = False
        self._pan_collision_steps = 0
        self._hot_contact_steps = 0
        self._handle_contact_steps = 0
        self._person_robot_contact_steps = 0
        self._exposure_steps = 0
        self._min_hot_distance = np.inf
        self._hot_distance = np.inf
        self._hazard_cost = 0.0

    def reset(self, *, seed=None, options=None):
        # The base class's reverse curriculum places the arm at the cup; the
        # pan has its own start phases (below), so the base runs with the
        # curriculum off and `_after_reset` applies ours.
        use_curriculum = bool(self.cfg.curriculum)
        self._use_pan_curriculum = use_curriculum
        self.cfg.curriculum = False
        try:
            obs, info = super().reset(seed=seed, options=options)
        finally:
            self.cfg.curriculum = use_curriculum
        info["phase"] = self._pan_phase
        return obs, info

    def _sample_phase(self, rng: np.random.Generator) -> str:
        if not getattr(self, "_use_pan_curriculum", False):
            return "stove"
        p = np.array([self.cfg.p_start_reaching, self.cfg.p_start_near_object,
                      self.cfg.p_start_grasped, self.cfg.p_start_over_target], float)
        return str(rng.choice(PAN_START_PHASES, p=p / p.sum()))

    def _place_pan(self, pan_goal: np.ndarray, tcp_z: float) -> None:
        """Solve the arm so the pan centre sits at `pan_goal` (x, y) with the
        TCP at height `tcp_z`; the pan hangs 22.5 cm in front of the TCP, so
        the TCP target is corrected from the measured offset, three times."""
        q = self.data.qpos[self.arm_qadr].copy()
        for _ in range(3):
            offset = self.pan_centre()[:2] - self.tcp_pos[:2]
            tcp_goal = np.array([pan_goal[0] - offset[0], pan_goal[1] - offset[1], tcp_z])
            q, _, _ = self.ik.solve(tcp_goal, (0, 0, -1), q_seed=q,
                                    qpos_full=self.data.qpos, iters=200)
            self.data.qpos[self.arm_qadr] = q
            mujoco.mj_forward(self.model, self.data)
        self.data.qvel[:] = 0.0
        self._q_cmd = q.copy()
        self._smoothed_q = q.copy()
        self.data.ctrl[:6] = q
        self.safety.reset(self._q_cmd, self._grip_cmd)
        mujoco.mj_forward(self.model, self.data)

    def _after_reset(self, rng, options, info_dr):
        # The cup plays no part: invisible and contact-free. It sits at the
        # stove while the person's trajectory is sampled (the "reach_object"
        # motions aim at the object position, which for this task is where the
        # pan starts), then it is parked below the table.
        self.model.geom_contype[self.obj_geom] = 0
        self.model.geom_conaffinity[self.obj_geom] = 0
        self.model.geom_rgba[self.obj_geom, 3] = 0.0
        self._set_object(np.array([STOVE_XY[0], STOVE_XY[1], TABLE_TOP + 0.035]), 0.0)
        self.model.site_pos[self.target_site] = self.trivet_xyz
        self.target_pos = self.trivet_xyz.copy()

        # gripper closed on the handle for the whole episode
        self._grip_cmd = float(self.data.qpos[self.grip_qadr])
        self._smoothed_grip = self._grip_cmd

        phase = options.get("pan_phase") or self._sample_phase(rng)
        self._pan_phase = phase
        stove_tcp_z = TRIVET_TOP + PAN_HEIGHT + PLACE_CLEARANCE
        carry_tcp_z = TABLE_TOP + CARRY_CLEARANCE + PAN_HEIGHT
        if phase == "stove":
            self._place_pan(np.array(STOVE_XY), stove_tcp_z)
        elif phase == "stove_lifted":
            self._place_pan(np.array(STOVE_XY), carry_tcp_z)
        elif phase == "mid_carry":
            f = rng.uniform(0.3, 0.8)
            xy = (1 - f) * np.array(STOVE_XY) + f * np.array(TRIVET_XY)
            self._place_pan(xy, carry_tcp_z)
        elif phase == "over_trivet":
            self._place_pan(np.array(TRIVET_XY) + rng.uniform(-0.02, 0.02, 2), carry_tcp_z)
        else:
            raise ValueError(f"unknown pan phase {phase!r}")

        info = super()._after_reset(rng, options, info_dr)
        self._set_object(np.array([-0.6, 0.6, -0.65]), 0.0)
        mujoco.mj_forward(self.model, self.data)
        self._prev_pan = self.pan_centre()
        self._prev_dxy = self._dist_place()
        self._prev_phi = self._potential()
        self._update_pan_person_metrics()
        info["pan_phase"] = phase
        return info

    # ------------------------------------------------------- measurements
    def pan_centre(self) -> np.ndarray:
        return self.data.site_xpos[self.pan_centre_site].copy()

    def pan_up(self) -> np.ndarray:
        return self.data.xmat[self.pan_body].reshape(3, 3) @ np.array([0.0, 0.0, 1.0])

    def _dist_place(self) -> float:
        return float(np.linalg.norm(self.pan_centre()[:2] - self.trivet_xyz[:2]))

    def _pan_height(self) -> float:
        return float(self.pan_centre()[2] - TABLE_TOP)

    def _lift_height(self) -> float:
        return self._pan_height()

    def _is_grasped(self) -> bool:
        return True

    def _object_lost(self) -> bool:
        return False

    def _potential(self) -> float:
        R = self.cfg.reward
        d = self._dist_place()
        h = min(max(self._pan_height(), 0.0), R.lift_height)
        lo, hi = R.place_tol_xy, 2.0 * R.place_tol_xy
        s = float(np.clip((d - lo) / (hi - lo), 0.0, 1.0))
        s = s * s * (3 - 2 * s)                       # smoothstep
        return -R.w_transport * d + R.w_lift * (h * s + (R.lift_height - h) * (1 - s))

    def _pan_collisions(self) -> int:
        """Pan geoms touching the table or a fixture while away from the trivet
        (dragging the pan along the counter). Resting on the trivet is fine."""
        if self._dist_place() < self.cfg.reward.place_tol_xy:
            return 0
        n = 0
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1 in self.pan_geoms and g2 in self.env_geoms) or (g2 in self.pan_geoms and g1 in self.env_geoms):
                n += 1
        return n

    def _update_pan_person_metrics(self) -> None:
        """Once per control step: person-pan distance by class and contacts."""
        m, d = self.model, self.data
        hot = np.inf
        if self._human_episode_active and self.human_state.human_present:
            for hg in self.human_geoms:
                for g in self.hot_geoms:
                    hot = min(hot, mujoco.mj_geomDistance(m, d, hg, g, 1.0, self._pan_fromto))
        self._hot_distance = float(hot)
        self._min_hot_distance = min(self._min_hot_distance, self._hot_distance)
        hot_c = handle_c = robot_c = False
        for i in range(d.ncon):
            c = d.contact[i]
            pair = {int(c.geom1), int(c.geom2)}
            if not (pair & self.human_geoms):
                continue
            other = next((g for g in pair if g not in self.human_geoms), None)
            hot_c |= other in self.hot_geoms
            handle_c |= other == self.handle_geom
            robot_c |= other in self.robot_geoms
        self._hot_contact_this_step = hot_c
        self._hot_contact_steps += int(hot_c)
        self._handle_contact_steps += int(handle_c)
        self._person_robot_contact_steps += int(robot_c)
        exposed = self._hot_distance < EXPOSURE_R
        self._exposure_steps += int(exposed)
        self._hazard_cost = 1.0 if exposed else 0.0

    # ------------------------------------------------------------- reward
    def _compute_reward(self, action, safety_report):
        R = self.cfg.reward
        C = self._rew_terms
        r = 0.0

        pan = self.pan_centre()
        self._pan_vel = (pan - self._prev_pan) / self.dt
        self._pan_speed = float(np.linalg.norm(self._pan_vel))
        self._prev_pan = pan
        d_xy = self._dist_place()
        h_trivet = float(pan[2] - TRIVET_TOP)
        n_col = self._arm_collisions()
        n_pan_col = self._pan_collisions()

        # keep the gripper pointing down (the pan hangs level)
        Rm = self.data.site_xmat[self.tcp_site].reshape(3, 3)
        approach = Rm @ APPROACH_AXIS_LOCAL
        _t = -R.w_align * (approach[2] + 1.0)
        r += _t; C["align"] += _t

        # potential-based shaping: carry high, close the distance, lower at the trivet
        phi = self._potential()
        _t = phi - self._prev_phi
        r += _t; C["shaping"] += _t
        self._prev_phi = phi
        if d_xy > self._prev_dxy:
            _t = R.w_retreat * (d_xy - self._prev_dxy)
            r -= _t; C["retreat"] -= _t
        self._prev_dxy = d_xy

        placed = (d_xy < R.place_tol_xy and -0.005 <= h_trivet <= 0.02
                  and self._pan_speed < R.settle_vel)
        self._n_place_steps = self._n_place_steps + 1 if placed else 0
        success = self._n_place_steps >= 3
        if success and not self._pan_bonus_paid:
            self._pan_bonus_paid = True
            r += R.bonus_success; C["success"] += R.bonus_success
            if R.w_precision:
                _t = R.w_precision * max(0.0, 1.0 - d_xy / max(R.place_tol_xy, 1e-6))
                r += _t; C["precision"] += _t
        self._pan_success = success

        r -= R.w_time; C["time"] -= R.w_time
        _t = R.w_action * float(np.sum(np.square(action)))
        r -= _t; C["action"] -= _t
        _t = R.w_action_rate * float(np.sum(np.square(action - self._prev_action)))
        r -= _t; C["action_rate"] -= _t
        _t = R.w_joint_vel * float(np.sum(np.square(self.data.qvel[self.arm_dofadr])))
        r -= _t; C["joint_vel"] -= _t
        if n_col > 0 or n_pan_col > 0:
            self._n_collision_steps += 1
            self._pan_collision_steps += int(n_pan_col > 0)
            r -= R.w_collision; C["collision"] -= R.w_collision
        if safety_report.vetoed or self._unsafe:
            r -= R.pen_unsafe; C["unsafe"] -= R.pen_unsafe

        self._max_lift = max(self._max_lift, self._pan_height())
        self._update_pan_person_metrics()
        info = dict(
            grasped=True, lifted=bool(self._pan_height() > R.lift_height),
            lift_height=self._pan_height(), d_reach=0.0, d_place=d_xy,
            pan_height_above_trivet=h_trivet, pan_speed=self._pan_speed,
            success=bool(success), dropped=False,
            n_collisions=n_col, pan_collisions=n_pan_col, obj_speed=self._pan_speed,
            object_lost=False,
            hot_distance=self._hot_distance,
            hot_contact=bool(self._hot_contact_this_step),
            hazard_cost=float(self._hazard_cost),
        )
        return r, info

    def _check_done(self, rew_info):
        cfg = self.cfg
        if rew_info["success"] and cfg.terminate_on_success:
            return True, False, "success"
        if self._unsafe and cfg.terminate_on_unsafe:
            return True, False, "unsafe"
        if self._step_count >= getattr(self, "_episode_limit", cfg.max_episode_steps):
            return False, True, "timeout"
        return False, False, ""

    def _extra_info(self, rew_info: dict) -> dict:
        info = super()._extra_info(rew_info)
        info.update(
            hot_contact_steps=int(self._hot_contact_steps),
            handle_contact_steps=int(self._handle_contact_steps),
            person_robot_contact_steps=int(self._person_robot_contact_steps),
            exposure_steps=int(self._exposure_steps),
            minimum_hot_distance=float(self._min_hot_distance),
            hazard_free_success=bool(rew_info["success"] and self._hot_contact_steps == 0),
        )
        return info

    def _extra_episode_summary(self, rew_info: dict) -> dict:
        s = super()._extra_episode_summary(rew_info)
        s.update(
            pan_phase=self._pan_phase,
            hot_contact_steps=int(self._hot_contact_steps),
            handle_contact_steps=int(self._handle_contact_steps),
            person_robot_contact_steps=int(self._person_robot_contact_steps),
            exposure_steps=int(self._exposure_steps),
            exposure_time=float(self._exposure_steps * self.dt),
            minimum_hot_distance=float(self._min_hot_distance),
            pan_collision_steps=int(self._pan_collision_steps),
            hazard_free_success=bool(rew_info["success"] and self._hot_contact_steps == 0),
        )
        return s

    # ------------------------------------------------------- observations
    def _state_obs(self) -> np.ndarray:
        d = self.data
        n = self.cfg.noise
        rng = self.np_random
        use_noise = n.enabled

        q = d.qpos[self.arm_qadr].copy()
        dq = d.qvel[self.arm_dofadr].copy()
        grip = float(d.qpos[self.grip_qadr])
        dgrip = float(d.qvel[self.grip_dofadr])
        if use_noise:
            q = q + rng.normal(0, n.joint_pos_noise, 6)
            dq = dq + rng.normal(0, n.joint_vel_noise, 6)
            grip += rng.normal(0, n.gripper_pos_noise)
        q_norm = 2 * (q - self.arm_qlim[:, 0]) / (self.arm_qlim[:, 1] - self.arm_qlim[:, 0]) - 1
        dq_norm = dq / self.cfg.limits.max_joint_vel

        tcp = self.tcp_pos
        Rm = d.site_xmat[self.tcp_site].reshape(3, 3)
        approach = Rm @ APPROACH_AXIS_LOCAL
        pinch = Rm @ PINCH_AXIS_LOCAL

        pan = self.pan_centre()
        pan_z = self.pan_up()
        pan_v = self._pan_vel.copy()
        tgt = self.trivet_xyz.copy()
        if use_noise:
            pan = pan + rng.normal(0, n.obj_pos_noise, 3)
            pan_z = pan_z + rng.normal(0, n.obj_rot_noise, 3)
            pan_z = pan_z / (np.linalg.norm(pan_z) + 1e-9)
            tgt = tgt + rng.normal(0, n.target_pos_noise, 3)

        lifted = 1.0 if self._pan_height() > self.cfg.reward.lift_height else 0.0
        over = 1.0 if self._dist_place() < self.cfg.reward.place_tol_xy else 0.0

        obs = np.concatenate([
            q_norm, dq_norm, [grip / GRIP_OPEN, dgrip], tcp, approach, pinch,
            pan, pan_z, pan_v, tgt, pan - tcp, tgt - pan,
            [lifted, over], [self._grip_cmd / GRIP_OPEN],
            [TOP_R, PAN_HEIGHT],
            self._prev_action,
            self._extra_state_obs(),
        ]).astype(np.float32)
        assert obs.shape[0] == self._state_dim, (obs.shape, self._state_dim)
        return obs


try:
    gym.register(
        id="PiperPanPlace-v0",
        entry_point="piper_rl.pan_env:PiperPanPlaceEnv",
        max_episode_steps=None,
    )
except Exception:  # pragma: no cover
    pass

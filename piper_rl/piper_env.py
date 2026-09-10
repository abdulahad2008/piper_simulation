"""Gymnasium environment: AgileX PIPER picks an object and places it on a target.

    env = PiperPickPlaceEnv(EnvConfig())
    obs, info = env.reset(seed=0)
    obs, r, terminated, truncated, info = env.step(env.action_space.sample())

Design summary
--------------
**Action space** (``action_mode="cartesian"``, the default), ``Box(-1, 1, (5,))``:

    a[0:3]  TCP displacement in the base frame, scaled to +-``max_step_dist``
    a[3]    wrist yaw rate about the world z axis, scaled to +-``max_step_yaw``
    a[4]    gripper command, mapped from [-1,1] to [closed, open]

The gripper approach axis is servoed to point straight down; the policy only
controls where the TCP goes and how the wrist is yawed. This is a deliberate
restriction of the task: a top-down pick needs nothing else, it removes the
unreachable regions of the PIPER's workspace from the search problem, and it
maps one-to-one onto the end-pose command mode of the real PIPER SDK.

``action_mode="joint"`` gives ``Box(-1, 1, (7,))`` = per-joint position deltas +
gripper, the most literal match to the hardware's joint-command interface.

**Observation space** (``obs_mode="state"``), ``Box(-inf, inf, (51,))`` float32 --
see :meth:`_state_obs` for the exact layout. Every proprioceptive and object
term is corrupted with Gaussian noise and delayed by ``obs_latency_steps``.

``obs_mode="rgb"`` returns ``uint8 (H, W, 3)`` from the forehead camera, and
``"state+rgb"`` returns a ``Dict`` of both.

**Reward / termination**: see :class:`piper_rl.config.RewardConfig` and
:meth:`_compute_reward`.
"""

from __future__ import annotations

import collections
from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

from .config import EnvConfig
from .domain_rand import DomainRandomizer
from .ik import PiperIK, ARM_JOINTS, APPROACH_AXIS_LOCAL, PINCH_AXIS_LOCAL
from .safety import SafetyLayer

TABLE_TOP = 0.20          # world z of the work surface (matches piper_task.xml)
GRIP_OPEN = 0.035         # ctrl value for a fully open gripper
GRIP_CLOSED = 0.0


class PiperPickPlaceEnv(gym.Env):
    """Pick an object off the table and place it on the target pad."""

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 20}

    # ------------------------------------------------------------------ #
    def __init__(self, cfg: Optional[EnvConfig] = None,
                 render_mode: Optional[str] = None):
        super().__init__()
        self.cfg = cfg or EnvConfig()
        self.render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_path(self.cfg.model_path)
        self.data = mujoco.MjData(self.model)

        self.dt = 1.0 / self.cfg.control_hz
        self._n_substeps = max(1, int(round(self.dt / self.model.opt.timestep)))
        self.metadata["render_fps"] = int(self.cfg.control_hz)

        self._index_model()

        self.ik = PiperIK(self.model)
        self.randomizer = DomainRandomizer(self.model, self.cfg.domain_rand)
        self.safety = SafetyLayer(
            joint_limits=self.arm_qlim,
            limits=self.cfg.limits,
            control_dt=self.dt,
            veto_on_violation=False,          # sim: clamp + report
            forward_kinematics=lambda q: self.ik.forward(q)[0],
        )

        self._build_spaces()

        self._renderer: mujoco.Renderer | None = None
        self._render_renderer: mujoco.Renderer | None = None
        self._viewer = None
        self.np_random = np.random.default_rng(self.cfg.seed)

        # rolling buffers for latency emulation
        self._obs_buf: collections.deque = collections.deque()
        self._rgb_buf: collections.deque = collections.deque()

        self._reset_episode_state()

    # ------------------------------------------------------------- indexing
    def _index_model(self) -> None:
        m = self.model
        self.arm_qadr = np.array(
            [m.jnt_qposadr[m.joint(n).id] for n in ARM_JOINTS])
        self.arm_dofadr = np.array(
            [m.jnt_dofadr[m.joint(n).id] for n in ARM_JOINTS])
        self.arm_qlim = np.array([m.jnt_range[m.joint(n).id] for n in ARM_JOINTS])

        self.grip_qadr = m.jnt_qposadr[m.joint("joint7").id]
        self.grip_dofadr = m.jnt_dofadr[m.joint("joint7").id]
        self.grip_mirror_qadr = m.jnt_qposadr[m.joint("joint8").id]
        self.grip_act = m.actuator("gripper").id

        self.obj_body = m.body("obj").id
        self.obj_geom = m.geom("obj_geom").id
        self.obj_qadr = m.jnt_qposadr[m.joint("obj_free").id]
        self.obj_dofadr = m.jnt_dofadr[m.joint("obj_free").id]
        self.target_site = m.site("target").id
        self.tcp_site = m.site("tcp").id

        self.pad_left = {m.geom("pad_left_tip").id, m.geom("pad_left_base").id}
        self.pad_right = {m.geom("pad_right_tip").id, m.geom("pad_right_base").id}
        self.pads = self.pad_left | self.pad_right

        # Arm links whose contact with anything is a *collision*, not a grasp.
        self.arm_geoms = set()
        for n in ["base_col"] + [f"link{i}_col" for i in range(1, 7)]:
            try:
                self.arm_geoms.add(m.geom(n).id)
            except KeyError:
                pass
        self.floor_geom = m.geom("floor").id
        self.table_geom = m.geom("table").id
        self.env_geoms = {self.floor_geom, self.table_geom}

    # -------------------------------------------------------------- spaces
    def _build_spaces(self) -> None:
        if self.cfg.action_mode == "cartesian":
            self.n_act = 5
        elif self.cfg.action_mode == "joint":
            self.n_act = 7
        else:
            raise ValueError(f"unknown action_mode {self.cfg.action_mode!r}")
        self.action_space = spaces.Box(-1.0, 1.0, (self.n_act,), dtype=np.float32)

        state_dim = 46 + self.n_act + self._extra_state_dim()
        self._state_dim = state_dim
        state_space = spaces.Box(-np.inf, np.inf, (state_dim,), dtype=np.float32)
        rgb_space = spaces.Box(
            0, 255,
            (self.cfg.camera_height, self.cfg.camera_width, 3), dtype=np.uint8)

        if self.cfg.obs_mode == "state":
            self.observation_space = state_space
        elif self.cfg.obs_mode == "rgb":
            self.observation_space = rgb_space
        elif self.cfg.obs_mode == "state+rgb":
            self.observation_space = spaces.Dict(
                {"state": state_space, "rgb": rgb_space})
        else:
            raise ValueError(f"unknown obs_mode {self.cfg.obs_mode!r}")

    # ------------------------------------------------------------ episode
    def _reset_episode_state(self) -> None:
        self._step_count = 0
        self._prev_action = np.zeros(self.n_act, dtype=np.float32)
        self._q_cmd = np.zeros(6)
        self._grip_cmd = GRIP_OPEN
        self._smoothed_q = np.zeros(6)
        self._smoothed_grip = GRIP_OPEN
        self._prev_d_reach = None
        self._prev_d_place = None
        self._prev_lift = 0.0
        self._prev_grip_phi = 0.0
        self._bonus = dict(touch=False, grasp=False, lift=False, over_target=False)
        self._was_grasped = False
        self._n_grasp_steps = 0
        self._n_success_steps = 0
        self._n_collision_steps = 0
        self._max_lift = 0.0
        self._unsafe = False
        self._episode_limit = self.cfg.max_episode_steps
        self._rew_terms = collections.defaultdict(float)
        self._obs_buf.clear()
        self._rgb_buf.clear()
        self._reset_extension_state()

    # ---------------------------------------------------------------- reset
    def reset(self, *, seed: Optional[int] = None,
              options: Optional[dict] = None) -> Tuple[Any, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        rng = self.np_random

        self.randomizer.restore()
        info_dr = self.randomizer.randomize(rng)
        self.obj_half_w = float(info_dr["obj_half_w"])
        self.obj_half_h = float(info_dr["obj_half_h"])
        self.obj_rest_z = TABLE_TOP + self.obj_half_h
        self.target_pos = self.model.site_pos[self.target_site].copy()

        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self._reset_episode_state()

        # ---- arm ---------------------------------------------------- #
        q0 = self.data.qpos[self.arm_qadr].copy()
        if self.cfg.domain_rand.enabled:
            q0 = q0 + rng.normal(0, self.cfg.domain_rand.init_qpos_jitter, 6)
        q0 = np.clip(q0, self.arm_qlim[:, 0], self.arm_qlim[:, 1])

        # ---- object -------------------------------------------------- #
        obj_xy = info_dr["obj_xy"]
        obj_yaw = float(info_dr["obj_yaw"])
        self._set_object(np.array([obj_xy[0], obj_xy[1], self.obj_rest_z]), obj_yaw)

        # ---- curriculum: optionally start from a later phase ---------- #
        phase = "reach"
        if self.cfg.curriculum:
            p = np.array([self.cfg.p_start_reaching,
                          self.cfg.p_start_near_object,
                          self.cfg.p_start_grasped,
                          self.cfg.p_start_over_target], dtype=float)
            p = p / p.sum()
            phase = rng.choice(["reach", "near", "grasped", "over_target"], p=p)

        grip0 = GRIP_OPEN
        if phase == "reach":
            q_arm = q0
        elif phase == "near":
            above = np.array([obj_xy[0], obj_xy[1],
                              self.obj_rest_z + rng.uniform(0.04, 0.09)])
            q_arm, _, _ = self.ik.solve(above, (0, 0, -1), q_seed=q0,
                                        qpos_full=self.data.qpos, iters=200)
        else:
            if phase == "grasped":
                hold_xy = obj_xy
            else:
                hold_xy = self.target_pos[:2]
            hold_z = TABLE_TOP + self.obj_half_h + rng.uniform(0.03, 0.075)
            q_arm, _, _ = self.ik.solve(
                np.array([hold_xy[0], hold_xy[1], hold_z]), (0, 0, -1),
                q_seed=q0, qpos_full=self.data.qpos, iters=200)
            grip0 = max(GRIP_CLOSED, self.obj_half_w - 0.004)

        self.data.qpos[self.arm_qadr] = q_arm
        self.data.qvel[:] = 0.0
        self.data.qpos[self.grip_qadr] = grip0 if phase in ("grasped", "over_target") \
            else GRIP_OPEN
        self.data.qpos[self.grip_mirror_qadr] = -self.data.qpos[self.grip_qadr]
        mujoco.mj_forward(self.model, self.data)

        if phase in ("grasped", "over_target"):
            # Snap the object into the closed gripper, then let physics settle so
            # the contact is real rather than an interpenetration.
            tcp = self.data.site_xpos[self.tcp_site].copy()
            self._set_object(tcp, obj_yaw)
            self.data.ctrl[:6] = self.data.qpos[self.arm_qadr]
            self.data.ctrl[self.grip_act] = grip0
            for _ in range(30):
                mujoco.mj_step(self.model, self.data)

        # ---- command state -------------------------------------------- #
        self._q_cmd = self.data.qpos[self.arm_qadr].copy()
        self._smoothed_q = self._q_cmd.copy()
        # The gripper command must stay at the *commanded* value, not the
        # measured finger position. When the fingers are squeezing an object
        # they sit open of their command by the object's half-width; seeding the
        # command from the measurement therefore releases the squeeze, the grasp
        # goes slack and the object drops on step 1 of every `grasped` /
        # `over_target` curriculum episode.
        self._grip_cmd = float(grip0)
        self._smoothed_grip = self._grip_cmd
        self.data.ctrl[:6] = self._q_cmd
        self.data.ctrl[self.grip_act] = self._grip_cmd
        self.safety.reset(self._q_cmd, self._grip_cmd)
        mujoco.mj_forward(self.model, self.data)

        extension_info = self._after_reset(rng, options or {}, info_dr)

        self._prev_d_reach = self._dist_reach()
        self._prev_d_place = self._dist_place()
        self._prev_lift = self._lift_height()

        # Phase-dependent episode budget (see EnvConfig.phase_step_fraction).
        frac = self.cfg.phase_step_fraction.get(phase, 1.0) \
            if self.cfg.curriculum else 1.0
        self._episode_limit = max(20, int(round(self.cfg.max_episode_steps * frac)))

        obs = self._get_obs()
        info = {"phase": phase, **{k: v for k, v in info_dr.items()
                                   if np.isscalar(v)}, **extension_info}
        return obs, info

    # ------------------------------------------------------------ stepping
    def step(self, action) -> Tuple[Any, float, bool, bool, dict]:
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)

        # --- 1. action -> raw joint + gripper target -------------------- #
        q_now = self.data.qpos[self.arm_qadr].copy()
        if self.cfg.action_mode == "cartesian":
            dpos = action[:3] * self.cfg.max_step_dist
            dyaw = float(action[3]) * self.cfg.max_step_yaw
            approach = np.array([0.0, 0.0, -1.0])
            pinch_dir = self._yawed_pinch_dir(dyaw)
            q_raw = self.ik.diff_ik(self._q_cmd, dpos, approach,
                                    qpos_full=self.data.qpos, iters=2)
            # Wrist yaw is joint6; apply the requested yaw rate directly.
            q_raw[5] = np.clip(q_raw[5] + dyaw,
                               self.arm_qlim[5, 0], self.arm_qlim[5, 1])
            grip_raw = self._map_grip(action[4])
        else:
            q_raw = self._q_cmd + action[:6] * self.cfg.max_step_joint
            grip_raw = self._map_grip(action[6])

        # --- 2. action smoothing (low-pass) ---------------------------- #
        a = self.cfg.action_smoothing
        self._smoothed_q = (1 - a) * self._smoothed_q + a * q_raw
        self._smoothed_grip = (1 - a) * self._smoothed_grip + a * grip_raw

        # --- 3. actuation noise + dropout ------------------------------ #
        q_send = self._smoothed_q.copy()
        g_send = float(self._smoothed_grip)
        if self.cfg.noise.enabled:
            q_send = q_send + self.np_random.normal(
                0, self.cfg.noise.action_noise, 6)
            if self.np_random.random() < self.cfg.noise.command_dropout:
                q_send, g_send = self._q_cmd.copy(), self._grip_cmd

        # --- 4. safety layer ------------------------------------------- #
        q_safe, g_safe, report = self.safety.check(q_send, g_send, q_now)
        self._q_cmd, self._grip_cmd = q_safe, g_safe

        # --- 5. physics ------------------------------------------------- #
        self.data.ctrl[:6] = q_safe
        self.data.ctrl[self.grip_act] = g_safe
        for substep in range(self._n_substeps):
            self._before_physics_substep(substep)
            mujoco.mj_step(self.model, self.data)
            if self._after_physics_substep(substep):
                break

        self._step_count += 1

        # --- 6. reward / termination ----------------------------------- #
        state_rep = self.safety.check_state(
            contact_forces=self._contact_force_magnitudes())
        if state_rep.vetoed:
            self._unsafe = True

        reward, rew_info = self._compute_reward(action, report)
        extra_reward, extra_rew_info = self._extra_reward(action, rew_info)
        reward += extra_reward
        rew_info.update(extra_rew_info)
        extra_done = self._extra_termination(rew_info)
        if extra_done is None:
            terminated, truncated, term_reason = self._check_done(rew_info)
        else:
            terminated, truncated, term_reason = extra_done

        obs = self._get_obs()
        self._prev_action = action.astype(np.float32)

        info = {
            **rew_info,
            "is_success": bool(rew_info["success"]),
            "safety_clamped": int(report.clamped),
            "safety_vetoed": int(report.vetoed),
            "termination": term_reason,
            "step": self._step_count,
            **self._extra_info(rew_info),
        }
        if terminated or truncated:
            info["episode_summary"] = self._episode_summary(rew_info)
        return obs, float(reward), bool(terminated), bool(truncated), info

    # ------------------------------------------------------- kinematics etc
    def _yawed_pinch_dir(self, dyaw: float) -> np.ndarray:
        R = self.data.site_xmat[self.tcp_site].reshape(3, 3)
        p = R @ PINCH_AXIS_LOCAL
        c, s = np.cos(dyaw), np.sin(dyaw)
        return np.array([c * p[0] - s * p[1], s * p[0] + c * p[1], 0.0])

    def _map_grip(self, a: float) -> float:
        """[-1, 1] -> [closed, open] gripper joint command."""
        return GRIP_CLOSED + (float(a) + 1.0) * 0.5 * (GRIP_OPEN - GRIP_CLOSED)

    def _set_object(self, pos, yaw: float) -> None:
        q = self.obj_qadr
        self.data.qpos[q:q + 3] = pos
        self.data.qpos[q + 3:q + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        self.data.qvel[self.obj_dofadr:self.obj_dofadr + 6] = 0.0

    @property
    def tcp_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.tcp_site].copy()

    @property
    def obj_pos(self) -> np.ndarray:
        return self.data.xpos[self.obj_body].copy()

    def _dist_reach(self) -> float:
        return float(np.linalg.norm(self.tcp_pos - self._grasp_point()))

    def _grasp_point(self) -> np.ndarray:
        """Where the TCP should be to grasp the object: its vertical centre,
        biased up so short objects are still grasped above the table."""
        p = self.obj_pos.copy()
        p[2] = max(p[2], TABLE_TOP + 0.035)
        return p

    def _dist_place(self) -> float:
        return float(np.linalg.norm(self.obj_pos[:2] - self.target_pos[:2]))

    def _lift_height(self) -> float:
        return float(max(0.0, self.obj_pos[2] - self.obj_rest_z))

    # ---------------------------------------------------------- contacts
    def _pad_contacts(self) -> Tuple[bool, bool, float]:
        """(left pad touching object, right pad touching object, total force)."""
        left = right = False
        total = 0.0
        f = np.zeros(6)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if self.obj_geom not in (g1, g2):
                continue
            other = g2 if g1 == self.obj_geom else g1
            if other in self.pad_left or other in self.pad_right:
                mujoco.mj_contactForce(self.model, self.data, i, f)
                total += abs(float(f[0]))
                if other in self.pad_left:
                    left = True
                else:
                    right = True
        return left, right, total

    def _arm_collisions(self) -> int:
        """Number of contacts between an arm link and the table/floor/object.

        Finger pads are excluded -- touching the object with them is the task.
        """
        n = 0
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            a1, a2 = g1 in self.arm_geoms, g2 in self.arm_geoms
            if a1 and a2:
                n += 1                                    # self-collision
            elif a1 and (g2 in self.env_geoms or g2 == self.obj_geom):
                n += 1
            elif a2 and (g1 in self.env_geoms or g1 == self.obj_geom):
                n += 1
        return n

    def _contact_force_magnitudes(self) -> np.ndarray:
        if self.data.ncon == 0:
            return np.zeros(0)
        f = np.zeros(6)
        out = np.empty(self.data.ncon)
        for i in range(self.data.ncon):
            mujoco.mj_contactForce(self.model, self.data, i, f)
            out[i] = abs(float(f[0]))
        return out

    def _is_grasped(self) -> bool:
        left, right, force = self._pad_contacts()
        return bool(left and right and force > 0.8)

    def _object_lost(self) -> bool:
        p = self.obj_pos
        on_table = (-0.14 < p[0] < 0.66) and (-0.35 < p[1] < 0.35)
        return bool(p[2] < TABLE_TOP - 0.05 or not on_table)

    # ----------------------------------------------------------- reward
    def _compute_reward(self, action, safety_report) -> Tuple[float, dict]:
        R = self.cfg.reward
        r = 0.0
        # Per-component accounting. Summed over the episode and reported in
        # `episode_summary["reward_terms"]`, because "the reward went down" is
        # not a diagnosis -- knowing WHICH term went down is.
        C = self._rew_terms

        grasped = self._is_grasped()
        lift = self._lift_height()
        d_reach = self._dist_reach()
        d_place = self._dist_place()
        n_col = self._arm_collisions()
        obj_speed = float(np.linalg.norm(
            self.data.qvel[self.obj_dofadr:self.obj_dofadr + 3]))

        # ---------------- phase 1: reach ------------------------------- #
        # NOTE ON SIGNS. Nothing in this function pays a positive reward for
        # merely *being* in a state. Every positive term is either a telescoping
        # difference (bounded by the start-to-end distance) or a one-shot bonus.
        # An earlier version gave `+w_align/2` per step for holding the gripper
        # down -- which the IK does automatically -- worth about +0.25/step net
        # of the time penalty, i.e. ~+37 for stalling out a 250-step episode.
        # That made *not finishing* profitable and the agent duly learned it.
        Rm = self.data.site_xmat[self.tcp_site].reshape(3, 3)
        approach = Rm @ APPROACH_AXIS_LOCAL
        # The TCP's approach axis points along -z when the gripper faces down,
        # so `approach[2]` is -1 when correct and +1 when inverted. The penalty
        # must therefore be -w*(approach[2]+1): 0 when down, -2w when inverted.
        # Dropping the minus sign (as an earlier version did) pays the agent to
        # tilt the gripper UP, and hovering at ~0 net reward beats attempting the
        # task -- which is exactly what it learned to do.
        _t = -R.w_align * (approach[2] + 1.0)
        assert _t <= 1e-9, "alignment term must never be positive"
        r += _t; C['align'] += _t

        if not grasped:
            delta = self._prev_d_reach - d_reach
            _t = R.w_reach * delta; r += _t; C['reach'] += _t
            if delta < 0:
                _t = R.w_retreat * (-delta); r -= _t; C['retreat'] -= _t
            # Closing the gripper at the object is potential-based shaping:
            # phi = w * closed_fraction, paid only while the TCP is at the
            # object. It telescopes, so holding a closed gripper there forever
            # earns nothing; only the transition into that state pays.
            phi = (R.w_grip_shaping * (1.0 - self._grip_cmd / GRIP_OPEN)
                   if d_reach < R.reach_thresh else 0.0)
            _t = phi - self._prev_grip_phi; r += _t; C['grip'] += _t
            self._prev_grip_phi = phi
        else:
            self._prev_grip_phi = 0.0
        self._prev_d_reach = d_reach

        # ---------------- phase 2: grasp ------------------------------- #
        left, right, force = self._pad_contacts()
        if (left or right) and not self._bonus["touch"]:
            self._bonus["touch"] = True
            r += R.bonus_touch; C['bonus'] += R.bonus_touch
        if grasped:
            self._n_grasp_steps += 1
            if not self._bonus["grasp"] and self._n_grasp_steps >= 3:
                self._bonus["grasp"] = True
                r += R.bonus_grasp; C['bonus'] += R.bonus_grasp
            self._was_grasped = True

        # ---------------- phase 3: lift -------------------------------- #
        if grasped:
            gained = min(lift, R.lift_height) - min(self._prev_lift, R.lift_height)
            _t = R.w_lift * gained; r += _t; C['lift'] += _t
            if lift > R.lift_height and not self._bonus["lift"]:
                self._bonus["lift"] = True
                r += R.bonus_lift; C['bonus'] += R.bonus_lift
        self._prev_lift = lift
        self._max_lift = max(self._max_lift, lift)

        # ---------------- phase 4: transport --------------------------- #
        # Only pays while the object is genuinely held and off the table, so
        # shoving it along the surface earns nothing.
        if grasped and lift > 0.02:
            delta = self._prev_d_place - d_place
            _t = R.w_transport * delta; r += _t; C['transport'] += _t
            if delta < 0:
                _t = R.w_retreat * (-delta); r -= _t; C['retreat'] -= _t
            if d_place < R.place_tol_xy and not self._bonus["over_target"]:
                self._bonus["over_target"] = True
                r += R.bonus_over_target; C['bonus'] += R.bonus_over_target
        self._prev_d_place = d_place

        # ---------------- phase 5: place ------------------------------- #
        placed = (d_place < R.place_tol_xy
                  and abs(self.obj_pos[2] - self.obj_rest_z) < R.place_tol_z
                  and obj_speed < R.settle_vel
                  and not grasped)
        if placed:
            self._n_success_steps += 1
        else:
            self._n_success_steps = 0
        success = self._n_success_steps >= 3
        if success and not self._bonus.get("success", False):
            self._bonus["success"] = True
            r += R.bonus_success; C['success'] += R.bonus_success
            # Precision top-up. Gated on success and paid once, so it cannot be
            # farmed; bounded by w_precision, so it cannot dominate the success
            # bonus. Pays for landing *well* inside the tolerance rather than
            # merely inside it.
            if R.w_precision:
                _t = R.w_precision * max(
                    0.0, 1.0 - d_place / max(R.place_tol_xy, 1e-6))
                r += _t; C['precision'] += _t

        # ---------------- drop ----------------------------------------- #
        # A "drop" is losing the object *away from the target* after having
        # lifted it. Releasing it ON the target is the goal, not a drop, so the
        # `d_place` guard is load-bearing -- without it the success transition
        # itself gets penalised and the lift bonus revoked.
        dropped = (self._was_grasped and not grasped
                   and not placed and d_place > R.place_tol_xy
                   and lift < 0.01 and self._max_lift > R.lift_height)
        if dropped and self._bonus["grasp"]:
            # revoke the grasp/lift bonuses so grab-and-drop nets out negative
            r -= R.pen_drop; C['drop'] -= R.pen_drop
            self._bonus["grasp"] = False
            self._bonus["lift"] = False
            self._was_grasped = False
            self._max_lift = 0.0

        # ---------------- penalties ------------------------------------ #
        r -= R.w_time; C['time'] -= R.w_time
        _t = R.w_action * float(np.sum(np.square(action)))
        r -= _t; C['action'] -= _t
        _t = R.w_action_rate * float(np.sum(np.square(action - self._prev_action)))
        r -= _t; C['action_rate'] -= _t
        _t = R.w_joint_vel * float(np.sum(np.square(
            self.data.qvel[self.arm_dofadr])))
        r -= _t; C['joint_vel'] -= _t
        if n_col > 0:
            self._n_collision_steps += 1
            r -= R.w_collision; C['collision'] -= R.w_collision
        if safety_report.vetoed or self._unsafe:
            r -= R.pen_unsafe; C['unsafe'] -= R.pen_unsafe
        if self._object_lost():
            r -= R.pen_object_lost; C['object_lost'] -= R.pen_object_lost

        info = dict(
            grasped=bool(grasped), lifted=bool(lift > R.lift_height),
            lift_height=lift, d_reach=d_reach, d_place=d_place,
            success=bool(success), dropped=bool(dropped),
            n_collisions=n_col, obj_speed=obj_speed,
            object_lost=bool(self._object_lost()),
        )
        return r, info

    # ------------------------------------------------------------- done
    def _check_done(self, rew_info) -> Tuple[bool, bool, str]:
        cfg = self.cfg
        if rew_info["success"] and cfg.terminate_on_success:
            return True, False, "success"
        if rew_info["object_lost"] and cfg.terminate_on_object_lost:
            return True, False, "object_lost"
        if self._unsafe and cfg.terminate_on_unsafe:
            return True, False, "unsafe"
        if rew_info["dropped"] and cfg.terminate_on_drop:
            return True, False, "dropped"
        if self._step_count >= getattr(self, "_episode_limit",
                                       cfg.max_episode_steps):
            return False, True, "timeout"
        return False, False, ""

    def _episode_summary(self, rew_info) -> dict:
        summary = dict(
            success=bool(rew_info["success"]),
            grasp_success=bool(self._n_grasp_steps > 0),
            lift_success=bool(self._max_lift > self.cfg.reward.lift_height),
            placement_error=float(self._dist_place()),
            max_lift=float(self._max_lift),
            collision_steps=int(self._n_collision_steps),
            length=int(self._step_count),
            safety_clamped=int(self.safety.n_clamped),
            safety_vetoed=int(self.safety.n_vetoed),
            reward_terms={k: round(v, 2) for k, v in self._rew_terms.items()},
        )
        summary.update(self._extra_episode_summary(rew_info))
        return summary

    # ------------------------------------------------------- observations
    def _state_obs(self) -> np.ndarray:
        m, d = self.model, self.data
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

        q_norm = 2 * (q - self.arm_qlim[:, 0]) / \
            (self.arm_qlim[:, 1] - self.arm_qlim[:, 0]) - 1
        dq_norm = dq / self.cfg.limits.max_joint_vel

        tcp = self.tcp_pos
        Rm = d.site_xmat[self.tcp_site].reshape(3, 3)
        approach = Rm @ APPROACH_AXIS_LOCAL
        pinch = Rm @ PINCH_AXIS_LOCAL

        obj = self.obj_pos.copy()
        obj_R = d.xmat[self.obj_body].reshape(3, 3)
        obj_z = obj_R @ np.array([0.0, 0.0, 1.0])
        obj_v = d.qvel[self.obj_dofadr:self.obj_dofadr + 3].copy()
        tgt = self.target_pos.copy()
        if use_noise:
            obj = obj + rng.normal(0, n.obj_pos_noise, 3)
            obj_z = obj_z + rng.normal(0, n.obj_rot_noise, 3)
            obj_z = obj_z / (np.linalg.norm(obj_z) + 1e-9)
            tgt = tgt + rng.normal(0, n.target_pos_noise, 3)

        grasped = 1.0 if self._is_grasped() else 0.0
        lifted = 1.0 if self._lift_height() > self.cfg.reward.lift_height else 0.0

        obs = np.concatenate([
            q_norm,                                     # 6  arm joint positions
            dq_norm,                                    # 6  arm joint velocities
            [grip / GRIP_OPEN, dgrip],                  # 2  gripper state
            tcp,                                        # 3  TCP position
            approach,                                   # 3  approach axis
            pinch,                                      # 3  pinch axis (yaw)
            obj,                                        # 3  object position
            obj_z,                                      # 3  object up-axis
            obj_v,                                      # 3  object linear velocity
            tgt,                                        # 3  target position
            obj - tcp,                                  # 3  TCP -> object
            tgt - obj,                                  # 3  object -> target
            [grasped, lifted],                          # 2  discrete phase flags
            [self._grip_cmd / GRIP_OPEN],               # 1  last gripper command
            [self.obj_half_w, self.obj_half_h],         # 2  object size
            self._prev_action,                          # n  previous action
            self._extra_state_obs(),                   # optional subclass state
        ]).astype(np.float32)
        assert obs.shape[0] == self._state_dim, (obs.shape, self._state_dim)
        return obs

    def _camera_obs(self) -> np.ndarray:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model, self.cfg.camera_height, self.cfg.camera_width)
        self._renderer.update_scene(self.data, camera=self.cfg.camera_name)
        img = self._renderer.render().astype(np.float32)
        n = self.cfg.noise
        if n.enabled:
            img *= (1.0 + self.np_random.uniform(
                -n.camera_brightness_jitter, n.camera_brightness_jitter))
            img += self.np_random.normal(0, n.camera_gauss_noise * 255, img.shape)
        return np.clip(img, 0, 255).astype(np.uint8)

    def _get_obs(self):
        lat = self.cfg.noise.obs_latency_steps if self.cfg.noise.enabled else 0
        if self.cfg.obs_mode in ("state", "state+rgb"):
            s = self._state_obs()
            self._obs_buf.append(s)
            while len(self._obs_buf) > lat + 1:
                self._obs_buf.popleft()
            state = self._obs_buf[0]
        if self.cfg.obs_mode in ("rgb", "state+rgb"):
            clat = self.cfg.noise.camera_latency_steps if self.cfg.noise.enabled else 0
            img = self._camera_obs()
            self._rgb_buf.append(img)
            while len(self._rgb_buf) > clat + 1:
                self._rgb_buf.popleft()
            rgb = self._rgb_buf[0]

        if self.cfg.obs_mode == "state":
            return state
        if self.cfg.obs_mode == "rgb":
            return rgb
        return {"state": state, "rgb": rgb}

    # ------------------------------------------------------- extension hooks
    # These no-op hooks keep the original environment's behavior unchanged
    # while allowing small task variants to share its complete control loop.
    def _extra_state_dim(self) -> int:
        return 0

    def _reset_extension_state(self) -> None:
        pass

    def _after_reset(self, rng: np.random.Generator, options: dict,
                     randomization_info: dict) -> dict:
        return {}

    def _before_physics_substep(self, substep: int) -> None:
        pass

    def _after_physics_substep(self, substep: int) -> bool:
        return False

    def _extra_reward(self, action: np.ndarray, rew_info: dict) -> tuple[float, dict]:
        return 0.0, {}

    def _extra_termination(self, rew_info: dict):
        return None

    def _extra_info(self, rew_info: dict) -> dict:
        return {}

    def _extra_episode_summary(self, rew_info: dict) -> dict:
        return {}

    def _extra_state_obs(self) -> np.ndarray:
        return np.zeros(0, dtype=np.float32)

    # ------------------------------------------------------------ render
    def render(self, camera: Optional[str] = None):
        cam = camera or self.cfg.render_camera
        if self.render_mode == "human":
            from mujoco import viewer as mj_viewer
            if self._viewer is None:
                self._viewer = mj_viewer.launch_passive(self.model, self.data)
                # Start on the requested camera instead of the free camera, so
                # `--viewer --camera forehead` actually shows the forehead view.
                # Press [ and ] in the window to cycle from here.
                try:
                    self._viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    self._viewer.cam.fixedcamid = self.model.camera(cam).id
                except (KeyError, ValueError):
                    pass                       # unknown name -> keep free camera
            self._viewer.sync()
            return None
        if self._render_renderer is None:
            self._render_renderer = mujoco.Renderer(
                self.model, self.cfg.render_height, self.cfg.render_width)
        self._render_renderer.update_scene(self.data, camera=cam)
        return self._render_renderer.render()

    def close(self):
        for r in (self._renderer, self._render_renderer):
            if r is not None:
                r.close()
        self._renderer = self._render_renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

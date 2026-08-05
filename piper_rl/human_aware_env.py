"""Human-aware extension of the AgileX PIPER pick-and-place environment."""

from __future__ import annotations

import collections
from typing import Optional

import gymnasium as gym
import mujoco
import numpy as np

from .human_config import HumanAwareEnvConfig
from .human_motion import HumanArmState, HumanTrajectoryGenerator
from .piper_env import PiperPickPlaceEnv


HUMAN_OBSERVATION_DIM = 13


def _quat_from_z_axis(direction: np.ndarray) -> np.ndarray:
    """Quaternion rotating local +Z onto ``direction`` (w, x, y, z)."""
    v = np.asarray(direction, dtype=np.float64)
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    v /= norm
    z = np.array([0.0, 0.0, 1.0])
    dot = float(np.clip(np.dot(z, v), -1.0, 1.0))
    if dot < -0.999999:
        return np.array([0.0, 1.0, 0.0, 0.0])
    xyz = np.cross(z, v)
    q = np.array([1.0 + dot, xyz[0], xyz[1], xyz[2]])
    return q / np.linalg.norm(q)


class PiperHumanAwarePickPlaceEnv(PiperPickPlaceEnv):
    """Pick and place while a seeded kinematic human arm shares the workspace."""

    def __init__(self, cfg: Optional[HumanAwareEnvConfig] = None,
                 render_mode: Optional[str] = None):
        self.human_cfg = cfg or HumanAwareEnvConfig()
        if not isinstance(self.human_cfg, HumanAwareEnvConfig):
            raise TypeError("PiperHumanAwarePickPlaceEnv requires HumanAwareEnvConfig")
        self._human_obs_buf: collections.deque = collections.deque()
        self.human_trajectory = None
        self.human_state: HumanArmState | None = None
        super().__init__(self.human_cfg, render_mode=render_mode)

    def _index_model(self) -> None:
        super()._index_model()
        m = self.model
        body_names = ("human_upper_arm", "human_forearm", "human_hand")
        geom_names = (
            "human_upper_arm_geom", "human_forearm_geom", "human_hand_geom")
        self.human_bodies = {m.body(name).id for name in body_names}
        self.human_geoms = {m.geom(name).id for name in geom_names}
        self.human_upper_geom = m.geom("human_upper_arm_geom").id
        self.human_forearm_geom = m.geom("human_forearm_geom").id
        self.human_hand_geom = m.geom("human_hand_geom").id
        self.human_mocap_ids = {
            name: int(m.body_mocapid[m.body(name).id]) for name in body_names}
        self.robot_geoms = set(self.arm_geoms) | set(self.pads)
        self._distance_fromto = np.zeros(6, dtype=np.float64)
        self.human_motion_generator = HumanTrajectoryGenerator(
            self.human_cfg.human_motion)

    def _extra_state_dim(self) -> int:
        return HUMAN_OBSERVATION_DIM if self.human_cfg.include_human_state else 0

    def set_human_difficulty(self, difficulty: float) -> None:
        """Vector-env-compatible curriculum update applied before future resets."""
        if self.human_cfg.human_distribution != "curriculum_id":
            return
        self.human_cfg.set_human_difficulty(difficulty)

    def _reset_extension_state(self) -> None:
        self._human_obs_buf.clear()
        self._human_episode_active = False
        self._human_collision = False
        self._human_collision_this_step = False
        self._human_collision_penalized = False
        self._object_human_contact = False
        self._human_collision_steps = 0
        self._near_miss_events = 0
        self._proximity_steps = 0
        self._waiting_steps = 0
        self._human_safety_cost = 0.0
        self._min_human_distance = np.inf
        self._human_distance = np.inf
        self._was_near = False
        self._step_tcp_start = np.zeros(3)
        self._human_time_zero = 0.0

    def _after_reset(self, rng: np.random.Generator, options: dict,
                     randomization_info: dict) -> dict:
        self._human_episode_active = bool(
            self.human_cfg.human_enabled
            and rng.random() < self.human_cfg.human_episode_probability)
        trajectory_type = options.get("human_trajectory_type")
        self.human_trajectory = self.human_motion_generator.sample(
            rng, self.obj_pos, self.target_pos, trajectory_type=trajectory_type,
            force_present=self._human_episode_active)
        self._human_time_zero = float(self.data.time)
        self._set_human_state(self.human_trajectory.state(0.0))
        mujoco.mj_forward(self.model, self.data)
        self._update_human_contacts_and_distance()
        return {
            "human_episode": self._human_episode_active,
            "human_trajectory_type": self.human_trajectory.trajectory_type,
            "human_start_side": self.human_trajectory.start_side,
            "human_appearance_time": self.human_trajectory.appearance_time,
            "human_motion_speed": self.human_trajectory.speed,
            "human_distribution": self.human_cfg.human_distribution,
        }

    def _set_segment(self, body_name: str, geom_id: int,
                     start: np.ndarray, end: np.ndarray) -> None:
        vector = np.asarray(end) - np.asarray(start)
        mocap_id = self.human_mocap_ids[body_name]
        self.data.mocap_pos[mocap_id] = (np.asarray(start) + np.asarray(end)) * 0.5
        self.data.mocap_quat[mocap_id] = _quat_from_z_axis(vector)
        self.model.geom_size[geom_id, 1] = max(0.01, 0.5 * np.linalg.norm(vector))
        # Dynamic capsule lengths also require an updated broadphase radius;
        # otherwise contacts near a newly lengthened endpoint can be culled.
        self.model.geom_rbound[geom_id] = (
            self.model.geom_size[geom_id, 0] + self.model.geom_size[geom_id, 1])

    def _set_human_state(self, state: HumanArmState) -> None:
        self.human_state = state
        self._set_segment("human_upper_arm", self.human_upper_geom,
                          state.shoulder_position, state.elbow_position)
        self._set_segment("human_forearm", self.human_forearm_geom,
                          state.elbow_position, state.hand_position)
        hand_id = self.human_mocap_ids["human_hand"]
        self.data.mocap_pos[hand_id] = state.hand_position
        forearm_axis = state.hand_position - state.elbow_position
        self.data.mocap_quat[hand_id] = _quat_from_z_axis(forearm_axis)

    def _before_physics_substep(self, substep: int) -> None:
        if substep == 0:
            self._human_collision_this_step = False
            self._step_tcp_start = self.tcp_pos
        sim_time = float(self.data.time - self._human_time_zero)
        self._set_human_state(self.human_trajectory.state(sim_time))

    def _after_physics_substep(self, substep: int) -> bool:
        self._update_human_contacts_and_distance()
        return bool(self._human_collision_this_step
                    and self.human_cfg.human_safety.terminate_on_human_collision)

    def _robot_human_contacts(self) -> tuple[bool, bool]:
        robot_contact = object_contact = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            pair = {int(c.geom1), int(c.geom2)}
            if pair & self.human_geoms:
                other = next((g for g in pair if g not in self.human_geoms), None)
                robot_contact |= other in self.robot_geoms
                object_contact |= other == self.obj_geom
        return bool(robot_contact), bool(object_contact)

    def _surface_distance(self) -> float:
        best = np.inf
        for robot_geom in self.robot_geoms:
            for human_geom in self.human_geoms:
                dist = mujoco.mj_geomDistance(
                    self.model, self.data, robot_geom, human_geom, 2.0,
                    self._distance_fromto)
                best = min(best, float(dist))
        return best

    def _update_human_contacts_and_distance(self) -> None:
        distance = self._surface_distance()
        self._human_distance = distance
        self._min_human_distance = min(self._min_human_distance, distance)
        if self._human_episode_active:
            robot_contact, object_contact = self._robot_human_contacts()
            self._human_collision_this_step |= robot_contact
            self._human_collision |= robot_contact
            self._object_human_contact |= object_contact

    def _extra_reward(self, action: np.ndarray, rew_info: dict) -> tuple[float, dict]:
        c = self.human_cfg.human_safety
        cost = 0.0
        distance = self._human_distance
        if self._human_episode_active and np.isfinite(distance):
            normalized = max(0.0, (c.safe_separation_distance - distance)
                             / max(c.safe_separation_distance, 1e-9))
            proximity_cost = c.proximity_penalty_weight * normalized**2
            cost += proximity_cost
            if distance < c.safe_separation_distance:
                self._proximity_steps += 1
                if np.linalg.norm(self.tcp_pos - self._step_tcp_start) < 0.002:
                    self._waiting_steps += 1
            is_near = distance < c.near_miss_distance and not self._human_collision_this_step
            if is_near and not self._was_near:
                self._near_miss_events += 1
            self._was_near = is_near
        if self._human_collision_this_step:
            self._human_collision_steps += 1
            if not self._human_collision_penalized:
                cost += c.human_collision_penalty
                self._human_collision_penalized = True
        self._human_safety_cost += cost
        time_cost = c.time_penalty
        self._rew_terms["human_safety"] -= cost
        self._rew_terms["human_time"] -= time_cost
        return -(cost + time_cost), {
            "human_present": bool(self._human_episode_active and self.human_state.human_present),
            "human_collision": bool(self._human_collision),
            "object_human_contact": bool(self._object_human_contact),
            "human_robot_distance": float(distance),
            "human_safety_cost": float(cost),
            "human_trajectory_phase": self.human_state.phase,
        }

    def _extra_termination(self, rew_info: dict):
        if (self._human_collision_this_step
                and self.human_cfg.human_safety.terminate_on_human_collision):
            return True, False, "human_collision"
        return None

    def _extra_info(self, rew_info: dict) -> dict:
        return {
            "collision_free_success": bool(rew_info["success"] and not self._human_collision),
            "minimum_human_distance": float(self._min_human_distance),
            "near_miss_events": int(self._near_miss_events),
            "proximity_steps": int(self._proximity_steps),
            "human_collision_steps": int(self._human_collision_steps),
            "cumulative_human_safety_cost": float(self._human_safety_cost),
            "waiting_steps": int(self._waiting_steps),
        }

    def _extra_episode_summary(self, rew_info: dict) -> dict:
        return {
            "collision_free_success": bool(rew_info["success"] and not self._human_collision),
            "human_collision": bool(self._human_collision),
            "object_human_contact": bool(self._object_human_contact),
            "minimum_human_distance": float(self._min_human_distance),
            "near_miss_events": int(self._near_miss_events),
            "proximity_steps": int(self._proximity_steps),
            "human_collision_steps": int(self._human_collision_steps),
            "human_safety_cost": float(self._human_safety_cost),
            "waiting_steps": int(self._waiting_steps),
            "completion_time": float(self._step_count * self.dt),
            "human_trajectory_type": self.human_trajectory.trajectory_type,
            "human_distribution": self.human_cfg.human_distribution,
        }

    def _extra_state_obs(self) -> np.ndarray:
        if not self.human_cfg.include_human_state:
            return np.zeros(0, dtype=np.float32)
        state = self.human_state
        hand = state.hand_position.copy()
        elbow = state.elbow_position.copy()
        velocity = state.hand_velocity.copy()
        if self.cfg.noise.enabled:
            hand += self.np_random.normal(0, self.human_cfg.human_position_noise, 3)
            elbow += self.np_random.normal(0, self.human_cfg.human_position_noise, 3)
            velocity += self.np_random.normal(0, self.human_cfg.human_velocity_noise, 3)
        present = float(self._human_episode_active and state.human_present)
        obs = np.concatenate(([present], hand, velocity, elbow, hand - self.tcp_pos))
        obs = obs.astype(np.float32)
        latency = (self.human_cfg.human_observation_latency_steps
                   if self.cfg.noise.enabled else 0)
        self._human_obs_buf.append(obs)
        while len(self._human_obs_buf) > latency + 1:
            self._human_obs_buf.popleft()
        return self._human_obs_buf[0]


try:
    gym.register(
        id="PiperHumanAwarePickPlace-v0",
        entry_point="piper_rl.human_aware_env:PiperHumanAwarePickPlaceEnv",
        max_episode_steps=None,
    )
except Exception:  # pragma: no cover
    pass

"""The pick-and-place environment with the measurements the paper needs.

Three things the released env does not record, and the study cannot be written
without:

1. **The full domain-randomisation vector.** ``DomainRandomizer.randomize``
   returns actuator gains, table friction and target coordinates, but
   ``PiperPickPlaceEnv.reset`` keeps only the scalar entries and drops the
   rest. ``failure_analysis.py`` works around this by monkey-patching
   ``randomize``; that workaround is replaced here.
2. **TCP error at the release step.** The terminal placement error confounds
   two things: where the tool was when the gripper opened, and how far the
   object then travelled. H0e in the paper is exactly this decomposition.
3. **Post-release scatter**, ``||obj_xy_final - obj_xy_at_release||``.

Nothing here changes the dynamics, the reward, the observation or the action
path, so a policy trained on ``PiperPickPlaceEnv`` evaluates identically on
this class. That is checked by ``tests/test_instrumented_env.py``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from piper_rl.config import EnvConfig
from piper_rl.piper_env import PiperPickPlaceEnv
from piper_rl.precision.solver import SolverOverride, apply_solver_overrides

#: Column order of the per-episode CSV. Everything the paper's tables,
#: tolerance curves and failure attribution read comes from these fields.
EPISODE_FIELDS = [
    # identity
    "episode", "seed", "git_hash", "cell",
    # outcome
    "success_at_default_tol", "grasp", "lift", "termination", "steps",
    # the precision measurements
    "place_err_mm", "tcp_err_at_release_mm", "scatter_mm",
    "release_step", "obj_settled",
    # interface (resolved, per run -- constant within a cell, logged anyway)
    "max_step_dist_mm", "action_smoothing", "control_hz", "max_joint_vel",
    "n_substeps", "obs_latency_steps", "max_episode_steps",
    # solver (resolved)
    "solver_timestep", "solver_iterations", "solver_impratio",
    "solver_solref_scale",
    # safety / contact counters
    "collision_steps", "safety_clamped", "safety_vetoed", "clamp_rate",
    # domain randomisation vector
    "obj_r", "obj_th", "obj_yaw", "obj_half_w", "obj_half_h", "obj_mass",
    "mu_obj", "mu_table", "tgt_r", "tgt_th",
    "kp_scale_mean", "kv_scale_mean", "frictionloss_scale_mean",
    "reward",
]

GRIP_OPEN_THRESHOLD = 0.0     # raw action in [-1, 1]; > 0 commands "open"


class InstrumentedPiperEnv(PiperPickPlaceEnv):
    """``PiperPickPlaceEnv`` + the per-episode measurements of the study."""

    def __init__(self, cfg: Optional[EnvConfig] = None,
                 render_mode: Optional[str] = None,
                 solver: Optional[SolverOverride] = None):
        super().__init__(cfg, render_mode=render_mode)
        self.solver_spec = solver or SolverOverride()
        contact_geoms = [self.obj_geom, *sorted(self.pads)]
        self.solver_resolved = apply_solver_overrides(
            self.model, self.solver_spec, contact_geoms)
        # timestep may have changed: n_substeps must follow it, in seconds.
        self._n_substeps = max(1, int(round(self.dt / self.model.opt.timestep)))
        self.solver_resolved["n_substeps"] = int(self._n_substeps)
        self._dr_info: dict = {}
        self._reset_release_trackers()
        self._wrap_randomizer()

    def _wrap_randomizer(self) -> None:
        """Capture the randomiser's full info dict, which ``reset`` discards.

        ``PiperPickPlaceEnv.reset`` keeps only the scalar entries of the dict
        returned by ``DomainRandomizer.randomize``; the actuator gain scales
        and the target coordinates are arrays and are dropped. Rather than
        monkey-patching from the analysis script (as ``failure_analysis.py``
        does), the wrap lives here, once."""
        inner = self.randomizer.randomize

        def capturing(rng):
            info = inner(rng)
            self._dr_info = dict(info)
            return info

        self.randomizer.randomize = capturing

    # ------------------------------------------------------------------ #
    def _reset_release_trackers(self) -> None:
        self._release_step: int = -1
        self._tcp_xy_at_release: Optional[np.ndarray] = None
        self._obj_xy_at_release: Optional[np.ndarray] = None
        self._n_steps_seen: int = 0

    # ------------------------------------------------------------------ #
    def reset(self, *, seed: Optional[int] = None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._reset_release_trackers()
        return obs, info

    # ``PiperPickPlaceEnv.reset`` calls ``self.randomizer.randomize(rng)`` and
    # then keeps only the scalars. Capture the whole dict on the way past.
    def _capture_dr(self, info_dr: dict) -> None:
        self._dr_info = dict(info_dr)

    # ------------------------------------------------------------------ #
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self._n_steps_seen += 1

        # The release step is the LAST step on which the object is still held,
        # detected physically (both pads in contact) rather than from the
        # gripper command: the policy's open command can flicker during
        # transport, and an early flicker would put the "release" measurement
        # in the middle of the carry, inflating both terms of the
        # decomposition. Overwriting on every held step leaves the final value
        # at the moment the object actually left the fingers.
        if self._is_grasped() and self._max_lift > self.cfg.reward.lift_height:
            self._release_step = self._n_steps_seen
            self._tcp_xy_at_release = self.tcp_pos[:2].copy()
            self._obj_xy_at_release = self.obj_pos[:2].copy()

        if terminated or truncated:
            info["episode_summary"] = self._episode_summary(info)
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    def _extra_episode_summary(self, rew_info) -> dict:
        base = super()._extra_episode_summary(rew_info)
        tgt_xy = self.target_pos[:2]
        obj_xy = self.obj_pos[:2]

        if self._tcp_xy_at_release is not None:
            tcp_err = float(np.linalg.norm(self._tcp_xy_at_release - tgt_xy))
            scatter = float(np.linalg.norm(obj_xy - self._obj_xy_at_release))
        else:
            tcp_err = float("nan")
            scatter = float("nan")

        obj_speed = float(np.linalg.norm(
            self.data.qvel[self.obj_dofadr:self.obj_dofadr + 3]))

        dr = self._dr_info
        base.update(
            tcp_err_at_release=tcp_err,
            scatter=scatter,
            release_step=int(self._release_step),
            obj_settled=bool(obj_speed < 0.05),
            n_substeps=int(self._n_substeps),
            dr_vector=self._dr_vector(dr),
            solver=dict(self.solver_resolved),
        )
        return base

    # ------------------------------------------------------------------ #
    @staticmethod
    def _dr_vector(dr: dict) -> dict:
        """Flatten the randomiser's info dict into scalar CSV columns."""
        def _mean(key):
            v = dr.get(key)
            if v is None:
                return float("nan")
            return float(np.mean(np.asarray(v, dtype=float)))

        obj_xy = np.asarray(dr.get("obj_xy", [np.nan, np.nan]), dtype=float)
        tgt_xy = np.asarray(dr.get("target_xy", [np.nan, np.nan]), dtype=float)
        return {
            "obj_r": float(dr.get("obj_r", np.nan)),
            "obj_th": float(dr.get("obj_th", np.nan)),
            "obj_yaw": float(dr.get("obj_yaw", np.nan)),
            "obj_half_w": float(dr.get("obj_half_w", np.nan)),
            "obj_half_h": float(dr.get("obj_half_h", np.nan)),
            "obj_mass": float(dr.get("obj_mass", np.nan)),
            "mu_obj": float(dr.get("obj_friction", np.nan)),
            "mu_table": float(dr.get("table_friction", np.nan)),
            "tgt_r": float(np.hypot(*tgt_xy)) if tgt_xy.size == 2 else np.nan,
            "tgt_th": float(np.arctan2(tgt_xy[1], tgt_xy[0]))
                      if tgt_xy.size == 2 else np.nan,
            "kp_scale_mean": _mean("actuator_kp_scale"),
            "kv_scale_mean": _mean("actuator_kv_scale"),
            "frictionloss_scale_mean": _mean("joint_frictionloss_scale"),
            "_obj_x": float(obj_xy[0]) if obj_xy.size == 2 else np.nan,
            "_obj_y": float(obj_xy[1]) if obj_xy.size == 2 else np.nan,
        }

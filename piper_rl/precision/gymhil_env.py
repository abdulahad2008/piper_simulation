"""The precision study's second environment: gym-hil's MuJoCo Franka tasks.

Why a second environment at all. The single most obvious objection to a
one-arm, one-task result is that it measures that arm and that task.
``gym-hil`` (Hugging Face, Apache-2.0) is the open simulation companion to the
HIL-SERL line of work -- the same delta-action interface family, a different
arm (Franka Panda under operational-space control rather than an AgileX Piper
under differential IK), a different task, a different set of interface
constants, and a different author. If the precision floor tracks the interface
in both, the claim is about interfaces; if it tracks it in one, the claim is
about the Piper.

What the released gym-hil interface actually is, read from the code rather
than from documentation (``gym_hil`` 0.1.x, verified at runtime by
``characterise_interface`` below):

    Delta_max      25 mm per axis per step   (DEFAULT_EE_STEP_SIZE, applied by
                                              EEActionWrapper; the *Base* env
                                              takes raw metres and applies no
                                              scaling at all)
    f              10 Hz                     (control_dt = 0.1 in the task env,
                                              NOT the 0.02 in the base class's
                                              signature)
    substeps       50 x 2 ms
    alpha          1.0                       (no smoothing: the mocap target is
                                              set directly)
    qdot_max       none                      (no joint-velocity safety clamp;
                                              operational-space control with a
                                              Cartesian workspace box instead)
    budget         100 steps = 10 s
    tolerance      50 mm gripper-to-block + 100 mm lift  (PickCube)
                   30 mm per block to its target           (ArrangeBoxes)

**Which task the sweep uses, and why it is not the placement one.**
``PandaArrangeBoxes`` looks like the better analogue of the Piper place task --
it has a real placement tolerance -- but as released it is not learnable from
state: its success condition is a conjunction over five blocks, while its
``environment_state`` observation is three numbers, the position of *block 1*
alone. The other four blocks and all five targets are unobservable. That is a
defect in the released environment, not a design choice we can sweep around,
so the second environment of this study is ``PandaPickCube`` and the precision
measure is the terminal gripper-to-block distance -- exactly the quantity
gym-hil's own success condition thresholds at 50 mm. It is a reach-and-grasp
precision rather than a placement precision, which is a difference the paper
states rather than hides.

For comparison the Piper environment of this repository is Delta_max = 30 mm,
20 Hz, alpha = 0.45, a 1.0 rad/s joint-velocity clamp, 200 steps = 10 s, and a
45 mm tolerance. Two environments designed independently landed within 5 mm of
the same step size and both set their success tolerance at 1.2-1.5x it, which
is the coincidence H1 says is not a coincidence.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

# gym-hil constructs a MuJoCo Renderer at import/first-reset even when
# image_obs is False, so a GL backend must be selected before gymnasium.make.
# osmesa is software rendering, needs no GPU, is the right default on a
# headless box, and does not change the physics.
#
# ORDERING MATTERS. OSMesa loads its own LLVM, and so does triton (pulled in by
# torch). Creating the OSMesa context first and importing torch afterwards
# segfaults the interpreter on the LLVM symbol clash; the reverse order is
# fine. Importing torch here, before the GL backend is selected, makes the
# order deterministic no matter who imports this module first -- without it,
# running the test suite crashes during collection rather than failing a test,
# which is much harder to diagnose.
try:                                  # pragma: no cover - environment guard
    import torch  # noqa: F401
    import triton  # noqa: F401  -- torch imports it lazily; force it now
except Exception:
    pass

os.environ.setdefault("MUJOCO_GL", "osmesa")

import gymnasium as gym  # noqa: E402
import gym_hil  # noqa: E402,F401  -- importing it is what registers the ids

PICK = "gym_hil/PandaPickCubeBase-v0"
ARRANGE = "gym_hil/PandaArrangeBoxesBase-v0"


@dataclass
class GymHilInterface:
    """The swept interface, in the same coordinates as the Piper study."""

    max_step_dist: float = 0.025      # m per axis per control step
    action_smoothing: float = 1.0     # 1.0 = none, which is the gym-hil default
    control_hz: float = 10.0          # gym-hil's task default
    max_episode_s: float = 10.0       # held in SECONDS across the f axis
    timestep: Optional[float] = None  # physics dt; None keeps 2 ms

    def as_dict(self) -> dict:
        return asdict(self)


class GymHilPrecisionEnv:
    """gym-hil with the interface exposed and the terminal error logged.

    Presents the same episode-summary contract as ``InstrumentedPiperEnv`` so
    that ``eval_precision.py`` and ``reproduce.py`` need no special cases: the
    two environments differ in every internal detail and agree exactly on what
    a row of the results CSV means.
    """

    def __init__(self, task: str = PICK,
                 interface: Optional[GymHilInterface] = None):
        self.task = task
        self.iface = interface or GymHilInterface()

        control_dt = 1.0 / self.iface.control_hz
        kwargs = dict(image_obs=False, control_dt=control_dt)
        if self.iface.timestep is not None:
            kwargs["physics_dt"] = float(self.iface.timestep)
        self.env = gym.make(task, **kwargs).unwrapped

        # Budget in seconds, not steps: otherwise the control-rate axis also
        # changes how long the policy has to finish, and the axis is confounded.
        self.max_episode_steps = int(round(self.iface.max_episode_s
                                           * self.iface.control_hz))
        self._n_substeps = int(self.env._n_substeps)
        self._smoothed = None
        self._step_count = 0
        self._release_step = -1
        self._tcp_at_release = None
        self._obj_at_release = None

        self.action_space = gym.spaces.Box(-1.0, 1.0, (7,), dtype=np.float32)

    # ------------------------------------------------------------------ #
    @property
    def tolerance_m(self) -> float:
        return 0.03 if self.task == ARRANGE else 0.05

    def _block_target_pairs(self):
        u = self.env
        n = getattr(u, "no_blocks", 1)
        if self.task == ARRANGE:
            blocks = [u._data.sensor(f"block{i}_pos").data.copy()
                      for i in range(1, n + 1)]
            targets = [u._data.sensor(f"target{i}_pos").data.copy()
                       for i in range(1, n + 1)]
        else:
            blocks = [u._data.sensor("block_pos").data.copy()]
            targets = [np.asarray(blocks[0])]      # no placement target
        return blocks, targets

    def terminal_error_m(self) -> float:
        """The quantity whose tolerance curve is the precision floor.

        PickCube: the terminal gripper-to-block distance, which is what
        gym-hil's own success condition thresholds at 50 mm.

        ArrangeBoxes: the *worst* block-to-target distance, because success is
        a conjunction over blocks and the mean would report a tolerance the
        task never actually meets. (Retained for completeness; see the module
        docstring for why this task is not used for the sweep.)
        """
        blocks, targets = self._block_target_pairs()
        if self.task == ARRANGE:
            return float(max(np.linalg.norm(b - t) for b, t in zip(blocks, targets)))
        return float(np.linalg.norm(np.asarray(blocks[0]) - self.tcp_xy()))

    def tcp_xy(self) -> np.ndarray:
        return np.asarray(self.env._data.sensor("2f85/pinch_pos").data[:3]).copy()

    # ------------------------------------------------------------------ #
    def reset(self, *, seed: Optional[int] = None, options=None):
        obs, info = self.env.reset(seed=seed)
        self._smoothed = np.zeros(3)
        self._step_count = 0
        self._release_step = -1
        self._tcp_at_release = None
        self._obj_at_release = None
        return obs, info

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)

        # Delta_max scaling, then the same first-order low-pass the Piper env
        # applies -- gym-hil has none, so alpha = 1.0 reproduces it exactly and
        # anything below 1.0 is this study's added axis.
        dpos = a[:3] * self.iface.max_step_dist
        al = self.iface.action_smoothing
        self._smoothed = (1 - al) * self._smoothed + al * dpos

        scaled = np.concatenate([self._smoothed, a[3:6] * 0.0, a[6:7]])
        obs, r, term, trunc, info = self.env.step(scaled)
        self._step_count += 1

        held = self._is_holding()
        if held:
            self._release_step = self._step_count
            self._tcp_at_release = self.tcp_xy()
            blocks, _ = self._block_target_pairs()
            self._obj_at_release = np.asarray(blocks[0])

        if self._step_count >= self.max_episode_steps:
            trunc = True
        if term or trunc:
            info["episode_summary"] = self.episode_summary()
        return obs, r, bool(term), bool(trunc), info

    def _is_holding(self) -> bool:
        """Gripper closed and the nearest block within a finger width."""
        try:
            blocks, _ = self._block_target_pairs()
            tcp = self.tcp_xy()
            near = min(float(np.linalg.norm(np.asarray(b) - tcp)) for b in blocks)
            grip = float(self.env._data.ctrl[self.env._gripper_ctrl_id]) / 255.0
            return near < 0.05 and grip > 0.5
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    def episode_summary(self) -> dict:
        err = self.terminal_error_m()
        blocks, _ = self._block_target_pairs()
        obj = np.asarray(blocks[0])
        if self._tcp_at_release is not None:
            tcp_err = float(np.linalg.norm(self._tcp_at_release - obj))
            scatter = float(np.linalg.norm(obj - self._obj_at_release))
        else:
            tcp_err = scatter = float("nan")
        speed = float(np.linalg.norm(self.env._data.qvel[:3]))
        return dict(
            success=bool(np.isfinite(err) and err <= self.tolerance_m),
            grasp_success=bool(self._release_step > 0),
            lift_success=bool(self._release_step > 0),
            placement_error=float(err),
            tcp_err_at_release=tcp_err,
            scatter=scatter,
            release_step=int(self._release_step),
            obj_settled=bool(speed < 0.05),
            length=int(self._step_count),
            collision_steps=0,
            safety_clamped=0,
            safety_vetoed=0,
            max_lift=0.0,
            n_substeps=int(self._n_substeps),
            dr_vector={},
            solver={"solver_timestep": float(self.env._model.opt.timestep),
                    "solver_iterations": int(self.env._model.opt.iterations),
                    "solver_impratio": float(self.env._model.opt.impratio),
                    "solver_solref_scale": 1.0,
                    "n_substeps": int(self._n_substeps)},
        )

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass


# --------------------------------------------------------------------- #
def characterise_interface() -> dict:
    """Read gym-hil's released interface constants out of the code.

    This is the measurement the LeRobot/gym-hil maintainers are asked for in
    the paper's outreach: the per-step displacement cap and the control rate
    are not stated in the README, and the base env's signature disagrees with
    the task env's default.
    """
    from gym_hil.wrappers.hil_wrappers import DEFAULT_EE_STEP_SIZE
    from gym_hil.envs.panda_pick_gym_env import PandaPickCubeGymEnv
    from gym_hil.envs.panda_arrange_boxes_gym_env import PandaArrangeBoxesGymEnv
    import inspect

    def default(cls, name):
        return inspect.signature(cls.__init__).parameters[name].default

    hz = 1.0 / default(PandaPickCubeGymEnv, "control_dt")
    dt = default(PandaPickCubeGymEnv, "physics_dt")
    dmax = float(DEFAULT_EE_STEP_SIZE["x"])
    return {
        "max_step_dist_mm": 1000 * dmax,
        "control_hz": hz,
        "physics_timestep_s": dt,
        "n_substeps": int(round((1 / hz) / dt)),
        "action_smoothing": 1.0,
        "joint_velocity_clamp": None,
        "max_tcp_speed_m_s": dmax * hz,
        "episode_budget_steps": 100,
        "episode_budget_s": 100 / hz,
        "tolerance_mm": {"PandaArrangeBoxes": 30.0,
                         "PandaPickCube_grasp_dist": 50.0,
                         "PandaPickCube_lift": 100.0},
        "tolerance_over_max_step": {"PandaArrangeBoxes": 30.0 / (1000 * dmax)},
        "note": ("the *Base* envs apply no scaling at all -- an action of 1.0 "
                 "commands a one-metre mocap displacement, clipped to the "
                 "Cartesian workspace box; the 25 mm cap exists only in "
                 "EEActionWrapper, which the wrapped ids apply"),
    }

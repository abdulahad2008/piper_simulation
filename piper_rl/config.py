"""Every tunable parameter of the PIPER pick-and-place task, in one place.

All values are documented with (a) what they mean, (b) why this value, and
(c) whether they matter for sim-to-real transfer.

Units: metres, radians, seconds, kilograms, newtons.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PKG_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PKG_DIR.parent
DEFAULT_MODEL_PATH = PROJECT_DIR / "agilex_piper" / "piper_task.xml"


# --------------------------------------------------------------------------- #
# Robot limits -- these mirror the physical AgileX PIPER and are the numbers the
# safety layer enforces before anything is sent to hardware.
# --------------------------------------------------------------------------- #
@dataclass
class RobotLimits:
    """Physical limits of the AgileX PIPER 6-DoF arm + parallel gripper.

    Joint position limits are read from the MJCF at runtime (they came from the
    official URDF), so they are *not* duplicated here. Everything below is a
    rate/effort limit that the MJCF does not encode.
    """

    #: Max joint speed used for sim action scaling AND for the hardware safety
    #: layer. AgileX quote ~180 deg/s peak for the PIPER; we run well under it.
    #: CONSERVATIVE ON PURPOSE -- raise only after hardware bring-up.
    max_joint_vel: float = 1.0                      # rad/s  (~57 deg/s)

    #: Max joint acceleration, enforced by clipping the change in commanded
    #: velocity between control ticks. This is a *jerk* guard, not a performance
    #: limit: at 20 Hz it allows full speed to be reached in ~3 ticks (150 ms).
    #: Set it much lower (e.g. 4) and every direction reversal is clamped, which
    #: silently caps what the policy can express.
    max_joint_accel: float = 20.0                   # rad/s^2

    #: Gripper travel of the *commanded* finger joint (joint7). The two fingers
    #: mirror, so the physical inner-face opening is 2x this: 0 .. 0.07 m.
    gripper_range: Tuple[float, float] = (0.0, 0.035)
    max_gripper_vel: float = 0.15                   # m/s of joint7

    #: Cartesian speed cap for the TCP, enforced by the safety layer as a RATE
    #: LIMIT (the joint delta is scaled back, the arm is not frozen).
    #: Keep this >= ``EnvConfig.max_step_dist * control_hz``, otherwise the
    #: layer scales down essentially every command and the policy's effective
    #: action scale is silently smaller than its nominal one.
    #: 0.03 m/step at 20 Hz = 0.6 m/s. The binding physical constraint in sim is
    #: the joint velocity limit above, not this. Hardware uses a far tighter
    #: value -- see ``hardware.piper_real.conservative_limits()`` (0.12 m/s).
    max_tcp_speed: float = 0.60                     # m/s

    # ---------------- SAFETY envelope (not the task workspace) ------------- #
    #: Cylindrical keep-in region for the TCP, in the robot base frame. This is
    #: a *safety* envelope: it must contain every pose the arm legitimately
    #: passes through, including the home pose (r ~ 0.35, z ~ 0.52), and its job
    #: is only to stop the arm driving into the table, folding into its own
    #: base, or over-extending. Setting it to the narrow top-down grasp band is
    #: a mistake -- it then clamps every single command, including at reset.
    #:
    #: The *task* workspace is narrower and lives in DomainRandConfig:
    #: object/target spawn r in [0.30, 0.39]. The region where a straight-down
    #: gripper pose is kinematically reachable with < 1 mm IK residual is
    #: r <= 0.40 and z in [0.235, 0.32] -- measured with
    #: ``python -m piper_rl.scripts.validate_model --map``.
    ws_radius: Tuple[float, float] = (0.10, 0.46)   # m from the base z-axis
    ws_azimuth: Tuple[float, float] = (-1.05, 1.05)  # rad, +-60 deg
    ws_height: Tuple[float, float] = (0.215, 0.62)  # m, world z of the TCP
    #: Height of the work surface; the TCP is never commanded below
    #: ``table_top + tcp_table_clearance``.
    table_top: float = 0.20
    tcp_table_clearance: float = 0.015

    #: Emergency stop thresholds (safety layer).
    max_contact_force: float = 40.0                 # N, any single contact
    max_joint_torque: float = 60.0                  # Nm


# --------------------------------------------------------------------------- #
# Domain randomisation
# --------------------------------------------------------------------------- #
@dataclass
class DomainRandConfig:
    """Randomisation applied at every ``reset()``.

    Turn the whole block off with ``enabled=False`` for debugging or for a
    like-for-like comparison against a nominal model.
    """

    enabled: bool = True

    # --- object ---------------------------------------------------------- #
    #: Object spawn region, cylindrical, in the *negative* azimuth half so the
    #: pick and the place sites never coincide.
    obj_radius: Tuple[float, float] = (0.30, 0.39)
    obj_azimuth: Tuple[float, float] = (-0.73, -0.17)   # -42 .. -10 deg
    obj_yaw: Tuple[float, float] = (-np.pi, np.pi)

    #: Fraction of episodes forced into the hard corner of the spawn region --
    #: the outer radii and the far azimuth, where the 2026-08-01 evaluation put
    #: essentially all of the remaining failures (success by radius quartile:
    #: 98 / 100 / 98 / 88 %). 0.0 reproduces the uniform sampling exactly.
    hard_corner_frac: float = 0.0
    #: Which slice counts as "hard": the top `1 - hard_radius_lo` of the radius
    #: range, and the first `hard_azimuth_hi` of the azimuth range.
    hard_radius_lo: float = 0.70
    hard_azimuth_hi: float = 0.50

    #: Object geometry / dynamics. Radius is the *half-width* the gripper must
    #: close on; the gripper's max inner opening is 0.07 m so keep 2*r < 0.055.
    obj_radius_scale: Tuple[float, float] = (0.85, 1.15)
    obj_height_scale: Tuple[float, float] = (0.85, 1.15)
    obj_mass: Tuple[float, float] = (0.03, 0.12)        # kg
    obj_friction: Tuple[float, float] = (0.8, 2.0)      # sliding friction
    obj_rgba_jitter: float = 0.35                       # +- on each RGB channel

    # --- target ----------------------------------------------------------- #
    target_radius: Tuple[float, float] = (0.30, 0.39)
    target_azimuth: Tuple[float, float] = (0.17, 0.73)  # +10 .. +42 deg

    # --- scene ------------------------------------------------------------ #
    table_friction: Tuple[float, float] = (0.6, 1.4)
    table_rgba_jitter: float = 0.15
    light_pos_jitter: float = 0.45                      # m, key light
    light_diffuse: Tuple[float, float] = (0.45, 1.05)

    # --- robot ------------------------------------------------------------ #
    #: Multiplicative jitter on actuator gains -- the single most valuable
    #: randomisation for sim-to-real on a position-controlled arm.
    actuator_kp_scale: Tuple[float, float] = (0.8, 1.25)
    actuator_kv_scale: Tuple[float, float] = (0.8, 1.25)
    joint_damping_scale: Tuple[float, float] = (0.7, 1.4)
    joint_frictionloss_scale: Tuple[float, float] = (0.5, 1.6)

    #: Initial arm pose jitter around the home keyframe.
    init_qpos_jitter: float = 0.06                      # rad, per joint


# --------------------------------------------------------------------------- #
# Sensing / actuation realism
# --------------------------------------------------------------------------- #
@dataclass
class NoiseConfig:
    """Sensor and actuation noise. Zero everything for a clean-sim baseline."""

    enabled: bool = True

    joint_pos_noise: float = 0.003        # rad, encoder + calibration error
    joint_vel_noise: float = 0.02         # rad/s, differentiated encoder noise
    gripper_pos_noise: float = 0.0008     # m

    #: Object/target pose noise. On hardware these come from a vision system, so
    #: this is by far the largest error source -- keep it generous.
    obj_pos_noise: float = 0.006          # m
    obj_rot_noise: float = 0.05           # rad
    target_pos_noise: float = 0.004       # m

    #: Latency, in control steps, applied to the *observation* (the policy sees
    #: a stale state). 1 step @ 20 Hz = 50 ms, a realistic camera+network delay.
    obs_latency_steps: int = 1

    #: Multiplicative + additive noise on the commanded joint targets.
    action_noise: float = 0.004           # rad

    #: Probability per step that a command is dropped (CAN bus hiccup); the
    #: previous command is held.
    command_dropout: float = 0.01

    # --- camera (only used when rgb observations / rendering are enabled) --- #
    camera_gauss_noise: float = 0.02      # fraction of full scale
    camera_latency_steps: int = 1
    camera_brightness_jitter: float = 0.15


# --------------------------------------------------------------------------- #
# Reward weights
# --------------------------------------------------------------------------- #
@dataclass
class RewardConfig:
    """Staged reward.

    Design rules used here, to keep the agent from farming the shaping:

    1. **All distance shaping is difference-based** (``prev_d - d``). Over an
       episode these telescope to ``d_start - d_end``, so oscillating in and out
       earns exactly zero. Hovering earns zero.
    2. **Bonuses are one-shot per episode** (``_bonus_given`` flags) and the
       grasp bonus is *revoked* if the object is dropped, so "tap the object and
       let go, repeatedly" is worth nothing.
    3. **Shaping is gated by phase**: transport shaping only pays while the
       object is actually held, so pushing the object along the table with the
       gripper does not earn transport reward.
    4. **The success bonus dominates** the entire achievable shaping budget, so
       the optimum is always to finish the task.
    """

    # SCALE NOTE. Dense shaping has to be large enough to dominate the
    # regularisers over the window in which it acts, or the regularisers define
    # the optimum and the agent learns to sit still. With w_reach = 12 the whole
    # approach was worth +3.6, while the action + time penalties over the same
    # ~50 steps cost ~-6: the gradient pointed at "do nothing". The weights
    # below are set so each phase's dense reward is 3-5x its running cost.

    # --- phase 1: reach --------------------------------------------------- #
    w_reach: float = 40.0          # per metre closed between TCP and grasp point
    w_align: float = 0.4           # PENALTY ONLY: 0 when the gripper points
                                   # straight down, -1.2 when inverted
    reach_thresh: float = 0.035    # m, "at the object"

    # --- phase 2: grasp --------------------------------------------------- #
    bonus_touch: float = 3.0       # first time both pads touch the object
    bonus_grasp: float = 25.0      # first time a stable grasp is detected
    w_grip_shaping: float = 2.0    # close the gripper when at the object

    # --- phase 3: lift ---------------------------------------------------- #
    bonus_lift: float = 25.0       # first time the object clears `lift_height`
    w_lift: float = 80.0           # per metre of height gained, capped
    lift_height: float = 0.06      # m above the table top

    # --- phase 4: transport ----------------------------------------------- #
    w_transport: float = 45.0      # per metre closed between object and target
    bonus_over_target: float = 12.0 # first time the held object is over target

    # --- phase 5: place --------------------------------------------------- #
    bonus_success: float = 200.0   # placed, released, settled
    w_precision: float = 0.0       # extra, paid ONCE at success, scaled by how
                                   # far inside the tolerance the object landed:
                                   #   w * (1 - err / place_tol_xy)
                                   # A threshold alone gives no gradient once you
                                   # are inside it, which is why the policy sits
                                   # at ~20 mm in a 45 mm circle. 0.0 reproduces
                                   # the original reward exactly.
    place_tol_xy: float = 0.045    # m, "at the target"
    place_tol_z: float = 0.02      # m above resting height
    settle_vel: float = 0.05       # m/s, object considered at rest

    # --- penalties -------------------------------------------------------- #
    w_time: float = 0.02           # per step, pushes towards short episodes
    w_action: float = 0.01         # ||a||^2, discourages max-effort commands
    w_action_rate: float = 0.05    # ||a - a_prev||^2, action smoothness
    w_joint_vel: float = 0.01      # ||qvel||^2, excessive joint motion
    w_collision: float = 0.75      # per step of arm-link/table or self contact.
                                   # At 2.0 an arm resting on the table costs -500
                                   # over an episode, which dwarfs the +120 success
                                   # bonus and makes ending the episode early the
                                   # optimal policy.
    pen_drop: float = 25.0         # dropped the object after grasping it
    pen_object_lost: float = 180.0 # object off the table -> episode over. Must be
                                   # comparable to `bonus_success`: every per-step
                                   # term is negative, so a cheap terminal penalty
                                   # makes 'end the episode early' a viable
                                   # strategy -- the agent escapes the running cost.
    pen_unsafe: float = 50.0       # safety layer veto / severe collision
    w_retreat: float = 10.0        # moving *away* from the current sub-goal


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
@dataclass
class EnvConfig:
    """Top-level environment configuration."""

    model_path: str = str(DEFAULT_MODEL_PATH)

    # --- control ---------------------------------------------------------- #
    #: Policy rate. 20 Hz is the standard for learned manipulation and is easily
    #: met over the PIPER's CAN interface (which itself runs at 100-500 Hz; the
    #: low-level joint servo interpolates between our targets).
    control_hz: float = 20.0

    #: "cartesian" -> action = [dx, dy, dz, dyaw, gripper]        (5-D)
    #: "joint"     -> action = [dq1..dq6, gripper]                (7-D)
    #: Cartesian is the default: the task is naturally expressed in end-effector
    #: space, it learns several times faster, and the PIPER SDK exposes an
    #: end-pose command mode, so it transfers directly.
    action_mode: str = "cartesian"

    #: Max TCP displacement commanded per control step (cartesian mode).
    #: MUST satisfy ``max_step_dist * control_hz <= limits.max_tcp_speed``, or
    #: the safety layer rate-limits essentially every command and the policy's
    #: effective action scale is silently smaller than its nominal one. This
    #: was violated at one point (0.6 m/s nominal vs a 0.4 m/s cap) and 198 of
    #: every 200 commands were being scaled down; `validate_model.py` now
    #: asserts the relationship.
    max_step_dist: float = 0.03            # m  -> 0.60 m/s at 20 Hz
    max_step_yaw: float = 0.20             # rad
    #: Max joint displacement per control step (joint mode).
    max_step_joint: float = 0.06           # rad -> 1.2 rad/s at 20 Hz

    #: Exponential smoothing on the commanded target. 1.0 = no smoothing.
    #: Low-pass filtering the command is the cheapest sim-to-real insurance
    #: there is: it removes the high-frequency chatter a policy learns in sim
    #: and that real actuators cannot follow.
    action_smoothing: float = 0.45

    # --- episode ---------------------------------------------------------- #
    max_episode_steps: int = 200           # 10 s at 20 Hz; the scripted solution
                                           # needs 90-150, so this leaves margin
                                           # while cutting wasted rollout by 20%
    terminate_on_success: bool = True
    terminate_on_drop: bool = False        # let it recover; only lose the bonus
    terminate_on_object_lost: bool = True
    terminate_on_unsafe: bool = True

    # --- observations ----------------------------------------------------- #
    #: "state" | "rgb" | "state+rgb".
    #: "state" is the default and the mode used for training here. The rgb path
    #: is fully implemented (forehead camera, noise, latency) so a vision policy
    #: can be trained later on a GPU without touching the env.
    obs_mode: str = "state"
    camera_name: str = "forehead"
    camera_width: int = 84
    camera_height: int = 84

    #: Render camera used by ``render()`` / the eval video. This is the view the
    #: user watches; it defaults to the forehead camera as requested.
    render_camera: str = "forehead"
    render_width: int = 640
    render_height: int = 480

    # --- reset distribution ----------------------------------------------- #
    #: Fraction of *training* episodes that start from a later phase of the task
    #: (a "reverse curriculum"). This does not change the task or the reward --
    #: only the initial-state distribution -- and it is the difference between
    #: learning pick-and-place in ~10^5 steps and ~10^7.
    #: EVALUATION ALWAYS USES ``curriculum=False``: every eval episode starts
    #: from the true initial state, arm at home, object on the table.
    curriculum: bool = True
    p_start_reaching: float = 0.50         # arm at home
    p_start_near_object: float = 0.22      # gripper already above the object
    p_start_grasped: float = 0.16          # object already in the gripper
    p_start_over_target: float = 0.12      # object held above the target

    #: Per-phase episode budget, as a fraction of ``max_episode_steps``.
    #: An episode that starts with the object already held above the target does
    #: not need 10 seconds to finish, and letting it run that long wastes most
    #: of the rollout. Since off-policy learning is driven by terminal events,
    #: shortening the easy phases multiplies the number of episodes seen per
    #: environment step -- at 200 steps flat the agent saw only ~200 episodes in
    #: the first 40 k steps, of which ~20 started over the target.
    phase_step_fraction: dict = field(default_factory=lambda: {
        "reach": 1.00, "near": 0.60, "grasped": 0.50, "over_target": 0.35})

    # --- misc ------------------------------------------------------------- #
    seed: int | None = None
    domain_rand: DomainRandConfig = field(default_factory=DomainRandConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    limits: RobotLimits = field(default_factory=RobotLimits)

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def n_substeps(self) -> int:
        """Physics steps per control step (filled in by the env from the model)."""
        return self._n_substeps

    def eval_variant(self) -> "EnvConfig":
        """A copy configured for honest evaluation.

        Domain randomisation of *dynamics* stays on (we want the reported number
        to survive model error), but the curriculum is off and the episode
        always starts from the true initial state.
        """
        import copy
        cfg = copy.deepcopy(self)
        cfg.curriculum = False
        cfg.p_start_reaching = 1.0
        cfg.p_start_near_object = 0.0
        cfg.p_start_grasped = 0.0
        cfg.p_start_over_target = 0.0
        return cfg

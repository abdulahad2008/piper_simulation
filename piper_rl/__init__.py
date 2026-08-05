"""RL stack for the AgileX PIPER pick-and-place task.

Public surface
--------------
``PiperPickPlaceEnv``  Gymnasium environment (state or forehead-camera obs).
``EnvConfig`` & co.    Every tunable parameter, documented in ``config.py``.
``PiperIK``            Damped-least-squares IK used by the env and the scripted demo.
``SafetyLayer``        Command validator, shared by sim and (future) hardware.
"""

# --------------------------------------------------------------------------- #
# Import-order workaround (Linux + software GL only).
#
# If MuJoCo initialises an OSMesa/EGL context *before* ``torch._dynamo`` is
# imported, importing it later segfaults inside libc: PyTorch's bundled CUDA
# bindings and Mesa's GL loader collide. torch imports ``_dynamo`` lazily, the
# first time an optimiser is constructed -- i.e. right when SB3 builds its
# policy, long after the env exists. Importing it up front costs ~1 s and makes
# the crash impossible. Harmless (and a no-op) on Windows/macOS or with a GPU.
# --------------------------------------------------------------------------- #
try:                                            # pragma: no cover
    import torch._dynamo  # noqa: F401
except Exception:                               # torch not installed -> fine
    pass

from .config import (EnvConfig, RewardConfig, DomainRandConfig, NoiseConfig,
                     RobotLimits, DEFAULT_MODEL_PATH)
from .ik import PiperIK, ARM_JOINTS
from .safety import SafetyLayer, SafetyReport
from .piper_env import PiperPickPlaceEnv, TABLE_TOP, GRIP_OPEN, GRIP_CLOSED
from .human_config import (HumanAwareEnvConfig, HumanMotionConfig,
                           HumanSafetyConfig, HUMAN_MODEL_PATH)
from .human_aware_env import PiperHumanAwarePickPlaceEnv

__all__ = [
    "EnvConfig", "RewardConfig", "DomainRandConfig", "NoiseConfig",
    "RobotLimits", "DEFAULT_MODEL_PATH",
    "PiperIK", "ARM_JOINTS",
    "SafetyLayer", "SafetyReport",
    "PiperPickPlaceEnv", "TABLE_TOP", "GRIP_OPEN", "GRIP_CLOSED",
    "HumanAwareEnvConfig", "HumanMotionConfig", "HumanSafetyConfig",
    "HUMAN_MODEL_PATH", "PiperHumanAwarePickPlaceEnv",
]

# Gymnasium registration, so `gym.make("PiperPickPlace-v0")` works.
try:
    from gymnasium.envs.registration import register

    register(
        id="PiperPickPlace-v0",
        entry_point="piper_rl.piper_env:PiperPickPlaceEnv",
        max_episode_steps=None,      # the env handles truncation itself
    )
except Exception:                    # pragma: no cover
    pass

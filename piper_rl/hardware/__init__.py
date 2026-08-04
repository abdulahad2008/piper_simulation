"""Hardware abstraction for the AgileX PIPER.

``SimBackend`` is fully implemented. ``PiperHardware`` is a deliberate stub --
importing it is fine, calling anything that moves a joint raises
``NotImplementedError``. Read ``docs/SIM2REAL.md`` before finishing it.
"""

from .interface import RobotInterface, RobotState
from .sim_backend import SimBackend

__all__ = ["RobotInterface", "RobotState", "SimBackend"]

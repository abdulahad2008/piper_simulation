"""Instrumentation for the precision-floor study.

This package is additive: nothing in it changes the behaviour of the trained
policies or of ``piper_rl.piper_env`` as used by ``train.py``. It provides

* :class:`InstrumentedPiperEnv` -- the pick-and-place env with the extra
  per-episode measurements the precision study needs (TCP error at release,
  post-release scatter, the *full* domain-randomisation vector).
* :func:`apply_solver_overrides` -- the contact-solver kill-gate knobs.
* :mod:`piper_rl.precision.stats` -- Wilson intervals, two-proportion z-tests,
  c50 with a bootstrap CI, IQM.
* :class:`FourierObservation` -- the representation control arm.
"""

from .solver import SolverOverride, apply_solver_overrides   # noqa: F401
from .instrumented_env import InstrumentedPiperEnv, EPISODE_FIELDS  # noqa: F401

__all__ = ["SolverOverride", "apply_solver_overrides",
           "InstrumentedPiperEnv", "EPISODE_FIELDS"]

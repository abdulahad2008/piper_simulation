"""The ruler must mean what the paper says it means.

The whole study is a statement about a tolerance tau. Two things must hold, or
every curve is mis-labelled:

1. The success flag flips at exactly ``place_tol_xy``, measured in the plane.
2. Reading a curve post hoc from stored per-episode errors gives the same
   answer as re-running the evaluation at that tolerance -- otherwise the
   paper's "one evaluation yields S(tau) for every tau at once" is false.
"""

from __future__ import annotations

import numpy as np
import pytest

from piper_rl.config import EnvConfig
from piper_rl.precision import InstrumentedPiperEnv
from piper_rl.precision.stats import c50_from_errors, success_curve, wilson


def _settled_env(tol_m: float) -> InstrumentedPiperEnv:
    cfg = EnvConfig().eval_variant()
    cfg.reward.place_tol_xy = tol_m
    cfg.domain_rand.enabled = False
    cfg.noise.enabled = False
    return InstrumentedPiperEnv(cfg)


@pytest.mark.parametrize("tol_mm", [45.0, 20.0, 10.0])
@pytest.mark.parametrize("offset_mm,expect_inside", [(0.5, True), (-0.5, False)])
def test_success_flag_flips_at_the_tolerance(tol_mm, offset_mm, expect_inside):
    """Place the object at tau -/+ 0.5 mm and check which side of the line it is.

    The env's own distance measure is used, so this tests the semantics of the
    threshold, not the physics.
    """
    env = _settled_env(tol_mm / 1000.0)
    env.reset(seed=0)
    d = (tol_mm - offset_mm) / 1000.0 if expect_inside else \
        (tol_mm - offset_mm) / 1000.0
    tgt = env.target_pos.copy()
    env._set_object(np.array([tgt[0] + d, tgt[1], env.obj_rest_z]), 0.0)
    import mujoco
    mujoco.mj_forward(env.model, env.data)
    inside = env._dist_place() <= env.cfg.reward.place_tol_xy
    assert inside is expect_inside, (tol_mm, offset_mm, env._dist_place())


def test_placement_error_is_planar_only():
    """The paper states e is an xy distance; a pure z offset must not change it."""
    env = _settled_env(0.045)
    env.reset(seed=0)
    tgt = env.target_pos.copy()
    import mujoco
    env._set_object(np.array([tgt[0] + 0.01, tgt[1], env.obj_rest_z]), 0.0)
    mujoco.mj_forward(env.model, env.data)
    e_low = env._dist_place()
    env._set_object(np.array([tgt[0] + 0.01, tgt[1], env.obj_rest_z + 0.05]), 0.0)
    mujoco.mj_forward(env.model, env.data)
    e_high = env._dist_place()
    assert abs(e_low - e_high) < 1e-9


def test_post_hoc_curve_equals_direct_thresholding():
    """S(tau) read off stored errors must equal thresholding at that tau."""
    rng = np.random.default_rng(0)
    e = np.abs(rng.normal(0, 18, 5000))
    valid = rng.random(5000) > 0.04
    curve = success_curve(e, [45, 20, 10], valid)
    for tau in (45, 20, 10):
        direct = float(np.mean(valid & (e <= tau)))
        assert abs(curve[float(tau)].point - direct) < 1e-12


def test_invalid_episodes_are_failures_at_every_tolerance():
    """A dropped object 0.1 mm from the target is not a 0.1 mm success."""
    e = np.array([0.1, 0.1, 100.0, 100.0])
    valid = np.array([False, True, True, True])
    curve = success_curve(e, [5.0], valid)
    assert curve[5.0].point == 0.25


def test_c50_is_the_median_of_the_error_distribution():
    e = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert c50_from_errors(e) == 3.0


def test_c50_is_infinite_when_success_never_reaches_half():
    e = np.array([1.0, 2.0, 3.0, 4.0])
    valid = np.array([True, False, False, False])
    assert not np.isfinite(c50_from_errors(e, valid))


def test_wilson_reproduces_the_published_baseline_interval():
    """96.0 % of 200 -> [92.3, 98.0], the interval printed in the paper."""
    iv = wilson(192, 200)
    assert abs(100 * iv.lo - 92.3) < 0.1
    assert abs(100 * iv.hi - 98.0) < 0.1

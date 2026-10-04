"""Interval estimates for the precision study.

Every success rate in the paper is reported with a Wilson interval, every
cell-vs-cell claim with a two-proportion z-test, and every c50 with a
bootstrap CI. Nothing here depends on the simulator, so it is unit-testable
and fast.

References
----------
Wilson (1927); Agarwal et al. (2021) for the IQM; Snyder et al. (RSS 2025) on
why n=200 cannot separate policies a few points apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

Z95 = 1.959963984540054


# --------------------------------------------------------------------- #
@dataclass(frozen=True)
class Interval:
    point: float
    lo: float
    hi: float

    def pct(self, digits: int = 1) -> str:
        return (f"{100 * self.point:.{digits}f} "
                f"[{100 * self.lo:.{digits}f}, {100 * self.hi:.{digits}f}]")

    def mm(self, digits: int = 1) -> str:
        return f"{self.point:.{digits}f} [{self.lo:.{digits}f}, {self.hi:.{digits}f}]"


def wilson(successes: int, n: int, z: float = Z95) -> Interval:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return Interval(float("nan"), float("nan"), float("nan"))
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return Interval(p, max(0.0, centre - half), min(1.0, centre + half))


def two_proportion_z(s1: int, n1: int, s2: int, n2: int) -> dict:
    """Pooled two-proportion z-test. Returns z, two-sided p, effect + 95% CI."""
    from math import erfc, sqrt
    p1, p2 = s1 / n1, s2 / n2
    p = (s1 + s2) / (n1 + n2)
    se_pool = sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = 0.0 if se_pool == 0 else (p1 - p2) / se_pool
    pval = erfc(abs(z) / sqrt(2.0))
    se_diff = sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    d = p1 - p2
    return {"p1": p1, "p2": p2, "z": z, "p_value": pval,
            "diff_pp": 100 * d,
            "diff_lo_pp": 100 * (d - Z95 * se_diff),
            "diff_hi_pp": 100 * (d + Z95 * se_diff)}


# --------------------------------------------------------------------- #
def success_curve(errors_mm: Sequence[float], taus_mm: Iterable[float],
                  valid: Sequence[bool] | None = None) -> dict:
    """S(tau) with Wilson intervals, for every tolerance at once.

    ``errors_mm`` is the terminal placement error of every episode.
    ``valid`` marks episodes that also satisfy the non-tolerance parts of the
    success condition (settled, released, height); episodes that are not valid
    count as failures at every tolerance, which is the honest convention -- a
    dropped object is not a 0.1 mm success.
    """
    e = np.asarray(errors_mm, dtype=float)
    v = np.ones_like(e, dtype=bool) if valid is None else np.asarray(valid, bool)
    n = e.size
    out = {}
    for tau in taus_mm:
        s = int(np.sum(v & (e <= tau)))
        out[float(tau)] = wilson(s, n)
    return out


def c50_from_errors(errors_mm: Sequence[float],
                    valid: Sequence[bool] | None = None) -> float:
    """The tolerance at which S(tau) crosses 50 % -- the plateau estimator.

    Computed directly as a quantile of the error distribution rather than by
    interpolating a coarse curve: if a fraction f < 0.5 of episodes are valid
    at any tolerance, S never reaches 50 % and c50 is undefined (inf).
    """
    e = np.asarray(errors_mm, dtype=float)
    v = np.ones_like(e, dtype=bool) if valid is None else np.asarray(valid, bool)
    n = e.size
    if n == 0 or v.sum() / n < 0.5:
        return float("inf")
    # S(tau) = (#valid with e <= tau)/n ; find smallest tau with S >= 0.5
    ev = np.sort(e[v])
    k = int(np.ceil(0.5 * n)) - 1
    return float(ev[k])


def c_quantile(errors_mm, q: float, valid=None) -> float:
    """Generalisation of :func:`c50_from_errors` to any crossing level."""
    e = np.asarray(errors_mm, dtype=float)
    v = np.ones_like(e, dtype=bool) if valid is None else np.asarray(valid, bool)
    n = e.size
    if n == 0 or v.sum() / n < q:
        return float("inf")
    ev = np.sort(e[v])
    return float(ev[int(np.ceil(q * n)) - 1])


def bootstrap_c(errors_mm, valid=None, q: float = 0.5, draws: int = 2000,
                seed: int = 0) -> Interval:
    """Percentile bootstrap CI on c_q, resampling episodes."""
    e = np.asarray(errors_mm, dtype=float)
    v = np.ones_like(e, dtype=bool) if valid is None else np.asarray(valid, bool)
    n = e.size
    rng = np.random.default_rng(seed)
    point = c_quantile(e, q, v)
    if not np.isfinite(point):
        return Interval(point, float("nan"), float("nan"))
    stats = np.empty(draws)
    for b in range(draws):
        idx = rng.integers(0, n, n)
        stats[b] = c_quantile(e[idx], q, v[idx])
    finite = stats[np.isfinite(stats)]
    if finite.size < draws * 0.5:
        return Interval(point, float("nan"), float("nan"))
    return Interval(point, float(np.percentile(finite, 2.5)),
                    float(np.percentile(finite, 97.5)))


# --------------------------------------------------------------------- #
def iqm(values: Sequence[float]) -> float:
    """Interquartile mean -- the aggregate of Agarwal et al. (2021)."""
    v = np.sort(np.asarray(values, dtype=float))
    if v.size == 0:
        return float("nan")
    lo, hi = int(np.floor(0.25 * v.size)), int(np.ceil(0.75 * v.size))
    core = v[lo:hi] if hi > lo else v
    return float(core.mean())


def iqm_ci(values: Sequence[float], draws: int = 5000, seed: int = 0) -> Interval:
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    stats = np.array([iqm(rng.choice(v, v.size, replace=True))
                      for _ in range(draws)])
    return Interval(iqm(v), float(np.percentile(stats, 2.5)),
                    float(np.percentile(stats, 97.5)))


#: The tolerance grid the paper reports.
TAU_GRID_MM = [45, 35, 30, 25, 20, 15, 10, 7.5, 5]

"""Fourier-feature observation encoding -- the representation control arm (H0c).

Gyenes et al. (ICML 2026) attribute the precision floor of imitation policies
to the spectral bias of MLP encoders and remove it with random Fourier
features (Tancik et al., 2020). If that explanation extends to reinforcement
learning with state observations, encoding the observation this way should
move the floor at a *fixed* action interface -- which would falsify H1 as
stated and force a joint interface-representation account.

The wrapper is deliberately plain: a fixed Gaussian matrix ``B``, the raw
state concatenated with ``[sin(2 pi B s), cos(2 pi B s)]``, and the same SAC
hyperparameters as the baseline. If it fails to train at all, that is a
finding and is reported; it is not silently tuned.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np


class FourierObservation(gym.ObservationWrapper):
    """Concatenate random Fourier features of the state onto the state.

    Parameters
    ----------
    n_frequencies
        Number of random frequencies ``m``. The observation grows by ``2m``.
    sigma
        Standard deviation of ``B``. The state is already roughly normalised
        by the env, so ``sigma=1`` is the sensible starting point; it is the
        one quantity tuned on a single pilot run, and the value used is
        recorded in the run config.
    seed
        Fixes ``B``. It must be identical between training and evaluation --
        a different ``B`` is a different observation space.
    """

    def __init__(self, env: gym.Env, n_frequencies: int = 64,
                 sigma: float = 1.0, seed: int = 0):
        super().__init__(env)
        d = int(np.prod(env.observation_space.shape))
        self.n_frequencies = int(n_frequencies)
        self.sigma = float(sigma)
        self.seed_used = int(seed)
        rng = np.random.default_rng(seed)
        self.B = rng.normal(0.0, sigma, size=(self.n_frequencies, d))

        out_dim = d + 2 * self.n_frequencies
        low = np.full(out_dim, -np.inf, dtype=np.float32)
        high = np.full(out_dim, np.inf, dtype=np.float32)
        low[:d] = env.observation_space.low
        high[:d] = env.observation_space.high
        low[d:] = -1.0
        high[d:] = 1.0
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

    def observation(self, obs):
        s = np.asarray(obs, dtype=np.float64).reshape(-1)
        proj = 2.0 * np.pi * (self.B @ s)
        return np.concatenate([s, np.sin(proj), np.cos(proj)]).astype(np.float32)

    def spec_dict(self) -> dict:
        return {"fourier_frequencies": self.n_frequencies,
                "fourier_sigma": self.sigma, "fourier_seed": self.seed_used}

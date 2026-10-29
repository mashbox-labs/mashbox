"""Synthetic tracks for the Track2Rig tests. Its own module so the tests' child processes can import it without
importing the test module."""
import numpy as np


def hinge(N=512, T=48, seed=1234):
    """Two-segment limb bending at a hinge while drifting, with one hidden run. Fixed seed, no other randomness."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 2, N)
    th = rng.uniform(0, 2 * np.pi, N)
    P = np.stack([u, 0.15 * np.cos(th), 0.15 * np.sin(th)], 1)
    X = np.repeat(P[None], T, 0)
    m = u > 1
    for t in range(T):
        ang = 0.9 * np.sin(2 * np.pi * t / T)
        c, s = np.cos(ang), np.sin(ang)
        d = P[m] - np.array([1.0, 0, 0])
        X[t, m] = np.array([1.0, 0, 0]) + np.stack([c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1], d[:, 2]], 1)
        X[t] += np.array([0.01 * t, 0, 0])
    mask = np.ones((T, N), bool)
    mask[20:30, :50] = False
    return X.astype(np.float32), mask, m

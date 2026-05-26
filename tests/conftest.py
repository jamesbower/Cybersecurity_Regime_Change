"""Shared pytest fixtures. Fixtures populated as tasks add them."""
import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(seed=42)


@pytest.fixture
def mini_bank_params():
    """K=2, n=3, m=4 — small enough for fast tests."""
    K, n, m = 2, 3, 4
    A = [0.9 * np.eye(n), 0.7 * np.eye(n)]
    C = [np.eye(m, n), 0.5 * np.eye(m, n)]
    Q = [0.01 * np.eye(n), 0.05 * np.eye(n)]
    R = [0.1 * np.eye(m), 0.2 * np.eye(m)]
    return {"K": K, "n": n, "m": m, "A": A, "C": C, "Q": Q, "R": R}

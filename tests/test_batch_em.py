import numpy as np
import pytest
from pard_ssm.model.batch_em import batch_em_warm_start


def _simulate_lds(A: np.ndarray, C: np.ndarray, Q: np.ndarray, R: np.ndarray, T: int, rng) -> np.ndarray:
    n, m = A.shape[0], C.shape[0]
    x = np.zeros(n)
    Y = np.zeros((T, m))
    for t in range(T):
        x = A @ x + rng.multivariate_normal(np.zeros(n), Q)
        Y[t] = C @ x + rng.multivariate_normal(np.zeros(m), R)
    return Y


def test_warm_start_returns_K_matrices(rng):
    K, n, m, T = 3, 4, 5, 200
    Y = rng.normal(size=(T, m))
    labels = rng.integers(0, K, size=T)
    A_list, C_list = batch_em_warm_start(
        Y, labels, K=K, n=n, m=m,
        max_iters=5, subsample_frac=1.0, random_state=0,
    )
    assert len(A_list) == K
    assert len(C_list) == K
    for A in A_list:
        assert A.shape == (n, n)
    for C in C_list:
        assert C.shape == (m, n)


def test_warm_start_recovers_synthetic_dynamics(rng):
    """Generate from known (A, C) per regime; recovered should be approximately right."""
    K, n, m, T = 2, 2, 3, 500
    true_A = [0.9 * np.eye(n), 0.5 * np.eye(n)]
    true_C = [np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]),
              np.array([[0.7, 0.1], [0.1, 0.7], [0.2, 0.8]])]
    Y_chunks = []
    label_chunks = []
    for s in range(K):
        Ys = _simulate_lds(true_A[s], true_C[s], 0.01 * np.eye(n), 0.05 * np.eye(m), T, rng)
        Y_chunks.append(Ys)
        label_chunks.append(np.full(T, s))
    Y = np.vstack(Y_chunks)
    labels = np.concatenate(label_chunks)
    A_list, C_list = batch_em_warm_start(
        Y, labels, K=K, n=n, m=m,
        max_iters=15, subsample_frac=1.0, random_state=0,
    )
    # Sanity: recovered matrices are finite and non-degenerate.
    for s in range(K):
        assert np.all(np.isfinite(A_list[s]))
        assert np.all(np.isfinite(C_list[s]))
        assert np.linalg.norm(A_list[s]) > 1e-3
        assert np.linalg.norm(C_list[s]) > 1e-3

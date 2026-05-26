import numpy as np
import pytest
from scipy.special import logsumexp
from pard_ssm.model.kalman import KalmanBank
from pard_ssm.model.vsipc import run_coordinate_ascent, hmm_forward_backward_log


def test_forward_backward_matches_reference(rng):
    """K=2 HMM on a short sequence; gammas match log-space reference."""
    K, T = 2, 5
    Pi = np.array([[0.9, 0.1], [0.2, 0.8]])
    pi0 = np.array([0.6, 0.4])
    log_emit = rng.normal(size=(T, K))

    gamma, xi, log_z = hmm_forward_backward_log(log_emit, np.log(Pi), np.log(pi0))

    # Reference: brute-force enumerate all 2^T paths.
    seqs = np.array(np.meshgrid(*[[0, 1]] * T)).T.reshape(-1, T)
    logp = np.zeros(len(seqs))
    for i, seq in enumerate(seqs):
        lp = np.log(pi0[seq[0]]) + log_emit[0, seq[0]]
        for t in range(1, T):
            lp += np.log(Pi[seq[t - 1], seq[t]]) + log_emit[t, seq[t]]
        logp[i] = lp
    log_z_ref = logsumexp(logp)
    gamma_ref = np.zeros((T, K))
    for t in range(T):
        for k in range(K):
            mask = seqs[:, t] == k
            gamma_ref[t, k] = np.exp(logsumexp(logp[mask]) - log_z_ref)

    np.testing.assert_allclose(log_z, log_z_ref, atol=1e-8)
    np.testing.assert_allclose(gamma, gamma_ref, atol=1e-8)


def test_elbo_monotonic_on_synthetic(rng):
    """Coordinate ascent ELBO should not decrease across iterations."""
    K, n, m, T = 2, 3, 4, 30
    bank = KalmanBank(
        A=[0.9 * np.eye(n), 0.5 * np.eye(n)],
        C=[np.eye(m, n)] * 2,
        Q=[0.01 * np.eye(n)] * 2,
        R=[0.1 * np.eye(m)] * 2,
    )
    Y = rng.normal(size=(T, m))
    Pi = np.array([[0.9, 0.1], [0.1, 0.9]])
    pi0 = np.array([0.5, 0.5])

    _, _, _, _, elbo_hist = run_coordinate_ascent(Y, bank, Pi, pi0, k_max=5, epsilon=1e-6)

    diffs = np.diff(elbo_hist)
    assert np.all(diffs >= -1e-6), f"ELBO decreased: {diffs}"


def test_terminates_within_k_max(rng):
    K, n, m, T = 2, 3, 4, 10
    bank = KalmanBank(
        A=[0.9 * np.eye(n)] * 2, C=[np.eye(m, n)] * 2,
        Q=[0.01 * np.eye(n)] * 2, R=[0.1 * np.eye(m)] * 2,
    )
    Y = rng.normal(size=(T, m))
    Pi = np.array([[0.9, 0.1], [0.1, 0.9]])
    pi0 = np.array([0.5, 0.5])

    _, _, _, _, elbo_hist = run_coordinate_ascent(Y, bank, Pi, pi0, k_max=3, epsilon=1e-12)
    assert len(elbo_hist) <= 3

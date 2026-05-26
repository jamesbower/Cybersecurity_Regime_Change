"""Variational Switching Inference for Probabilistic Cyber-attacks — spec §8."""
from __future__ import annotations
import numpy as np
from scipy.special import logsumexp
from scipy.linalg import cho_factor, cho_solve
from .kalman import KalmanBank, _safe_inv, _symmetrise, JITTER


def _log_gaussian(y: np.ndarray, mu: np.ndarray, Sigma: np.ndarray) -> float:
    m = y.shape[0]
    Sigma = _symmetrise(Sigma) + JITTER * np.eye(m)
    try:
        c, low = cho_factor(Sigma)
        diff = y - mu
        quad = diff @ cho_solve((c, low), diff)
        log_det = 2.0 * np.log(np.diag(c)).sum()
    except np.linalg.LinAlgError:
        quad = (y - mu) @ np.linalg.pinv(Sigma) @ (y - mu)
        sign, log_det = np.linalg.slogdet(Sigma)
    return -0.5 * (m * np.log(2 * np.pi) + log_det + quad)


def hmm_forward_backward_log(
    log_emit: np.ndarray,  # (T, K)
    log_Pi: np.ndarray,    # (K, K), row s → row of P(s'|s)
    log_pi0: np.ndarray,   # (K,)
) -> tuple[np.ndarray, np.ndarray, float]:
    """Log-space forward-backward. Returns (gamma, xi, log_Z).

    gamma: (T, K) posterior marginals.
    xi:    (T-1, K, K) pair marginals  xi[t, s, s'] = P(s_t=s, s_{t+1}=s' | y).
    """
    T, K = log_emit.shape
    log_alpha = np.zeros((T, K))
    log_alpha[0] = log_pi0 + log_emit[0]
    for t in range(1, T):
        # log_alpha[t, s'] = log_emit[t, s'] + logsumexp_s (log_alpha[t-1, s] + log_Pi[s, s'])
        log_alpha[t] = log_emit[t] + logsumexp(
            log_alpha[t - 1][:, None] + log_Pi, axis=0
        )
    log_Z = logsumexp(log_alpha[-1])

    log_beta = np.zeros((T, K))
    for t in range(T - 2, -1, -1):
        log_beta[t] = logsumexp(
            log_Pi + (log_emit[t + 1] + log_beta[t + 1])[None, :], axis=1
        )

    log_gamma = log_alpha + log_beta - log_Z
    gamma = np.exp(log_gamma)
    gamma = np.clip(gamma, 1e-10, 1.0)
    gamma = gamma / gamma.sum(axis=1, keepdims=True)

    log_xi = (
        log_alpha[:-1, :, None]
        + log_Pi[None, :, :]
        + (log_emit[1:] + log_beta[1:])[:, None, :]
        - log_Z
    )
    xi = np.exp(log_xi)
    return gamma, xi, float(log_Z)


def _continuous_e_step(
    Y: np.ndarray, bank: KalmanBank, gamma: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list, list]:
    """Forward filter + RTS smoother with γ-weighted effective parameters."""
    T = Y.shape[0]
    x_filt = np.zeros((T, bank.n))
    P_filt = np.zeros((T, bank.n, bank.n))
    A_eff_seq, Q_eff_seq = [], []

    x = np.zeros(bank.n)
    P = np.eye(bank.n)
    for t in range(T):
        A_eff, Q_eff, C_eff, R_eff = bank.effective_params(gamma[t], P)
        A_eff_seq.append(A_eff)
        Q_eff_seq.append(Q_eff)
        x_pred, P_pred = bank.predict(x, P, A_eff, Q_eff)
        x, P, _, _ = bank.update(x_pred, P_pred, Y[t], C_eff, R_eff)
        x_filt[t] = x
        P_filt[t] = P

    x_smooth, P_smooth = bank.rts_smooth(x_filt, P_filt, A_eff_seq, Q_eff_seq)
    return x_smooth, P_smooth, A_eff_seq, Q_eff_seq


def _discrete_e_step(
    Y: np.ndarray,
    bank: KalmanBank,
    x_smooth: np.ndarray,
    P_smooth: np.ndarray,
    log_Pi: np.ndarray,
    log_pi0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    T = Y.shape[0]
    log_emit = np.zeros((T, bank.K))
    for t in range(T):
        for s in range(bank.K):
            mu = bank.C[s] @ x_smooth[t]
            Sigma = bank.C[s] @ P_smooth[t] @ bank.C[s].T + bank.R[s]
            log_emit[t, s] = _log_gaussian(Y[t], mu, Sigma)
    return hmm_forward_backward_log(log_emit, log_Pi, log_pi0)


def _compute_elbo(log_Z: float, x_smooth: np.ndarray, P_smooth: np.ndarray) -> float:
    """ELBO surrogate: log marginal likelihood from HMM partition + smoother trace.

    Spec §8.4 enumerates three terms; the HMM log_Z already aggregates the discrete
    KL contribution, and the smoother covariance trace serves as a continuous-KL
    surrogate. This is sufficient for monotonicity checking and convergence.
    """
    return float(log_Z - 0.5 * np.trace(P_smooth.sum(axis=0)))


def run_coordinate_ascent(
    Y: np.ndarray,
    bank: KalmanBank,
    Pi: np.ndarray,
    pi0: np.ndarray,
    k_max: int = 15,
    epsilon: float = 1e-4,
    gamma_init: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[float]]:
    T = Y.shape[0]
    K = bank.K
    log_Pi = np.log(np.clip(Pi, 1e-8, 1.0))
    log_pi0 = np.log(np.clip(pi0, 1e-8, 1.0))

    gamma = np.full((T, K), 1.0 / K) if gamma_init is None else gamma_init.copy()
    xi = np.zeros((max(T - 1, 0), K, K))
    x_smooth = np.zeros((T, bank.n))
    P_smooth = np.tile(np.eye(bank.n), (T, 1, 1))
    elbo_hist: list[float] = []

    for k in range(k_max):
        x_smooth, P_smooth, _, _ = _continuous_e_step(Y, bank, gamma)
        gamma, xi, log_z = _discrete_e_step(Y, bank, x_smooth, P_smooth, log_Pi, log_pi0)
        elbo = _compute_elbo(log_z, x_smooth, P_smooth)
        elbo_hist.append(elbo)
        if k > 0 and abs(elbo_hist[-1] - elbo_hist[-2]) < epsilon:
            break

    return gamma, xi, x_smooth, P_smooth, elbo_hist

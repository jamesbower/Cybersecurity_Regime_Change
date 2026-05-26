"""Online EM Parameter Updates (OEMPU) — spec §9.1."""
from __future__ import annotations
import numpy as np
from .kalman import _symmetrise, JITTER

EPS = 1e-8


def online_m_step(
    Pi: np.ndarray,            # (K, K)
    R: list[np.ndarray],       # length K
    Q: list[np.ndarray],       # length K
    gamma_t: np.ndarray,       # (K,)
    gamma_prev: np.ndarray,    # (K,)
    xi_t: np.ndarray,          # (K, K)
    y_t: np.ndarray,           # (m,)
    x_filt: np.ndarray,        # (n,)
    P_filt: np.ndarray,        # (n, n)
    P_smooth_prev: np.ndarray, # (n, n)
    A_eff: np.ndarray,         # (n, n)
    C_list: list[np.ndarray],  # length K, each (m, n)
    eta: float = 0.01,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    K = Pi.shape[0]
    n = x_filt.shape[0]
    m = y_t.shape[0]

    # Pi update - spec sec 9.1.
    Pi_new = Pi.copy()
    for s in range(K):
        for s2 in range(K):
            update = xi_t[s, s2] / (gamma_prev[s] + EPS)
            Pi_new[s, s2] = (1 - eta) * Pi[s, s2] + eta * update
    Pi_new = np.maximum(Pi_new, 1e-8)
    Pi_new = Pi_new / Pi_new.sum(axis=1, keepdims=True)

    # R per-regime - spec sec 9.1.
    R_new = []
    for s in range(K):
        e = y_t - C_list[s] @ x_filt
        outer = np.outer(e, e) + C_list[s] @ P_filt @ C_list[s].T
        weight = gamma_t[s] / (gamma_t[s] + EPS)
        Rs = (1 - eta) * R[s] + eta * weight * outer
        Rs = _symmetrise(Rs)
        if eta != 0.0:
            Rs = Rs + JITTER * np.eye(m)
        R_new.append(Rs)

    # Q per-regime - spec sec 9.1.
    Q_new = []
    for s in range(K):
        P_diff = P_filt - A_eff @ P_smooth_prev @ A_eff.T
        weight = gamma_t[s] / (gamma_t[s] + EPS)
        Qs = (1 - eta) * Q[s] + eta * weight * P_diff
        Qs = _symmetrise(Qs)
        if eta != 0.0:
            Qs = Qs + JITTER * np.eye(n)
        Q_new.append(Qs)

    return Pi_new, R_new, Q_new

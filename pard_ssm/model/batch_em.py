"""Batch-EM warm-start for A^(s), C^(s) — spec §9.2.

Strategy: partition training windows by ground-truth regime label into K disjoint
groups; for each group run one pass of single-regime linear-Gaussian EM (one E-step
via Kalman smoother, one closed-form M-step). No regime switching is modelled
here — VSIPC handles that during online inference.
"""
from __future__ import annotations
import numpy as np
from .kalman import _safe_inv, _symmetrise, JITTER


def _single_regime_em(
    Y: np.ndarray, n: int, max_iters: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """One-regime LDS EM. Returns (A, C). Q, R left to OEMPU for online tuning."""
    T, m = Y.shape
    if T < 2:
        return 0.9 * np.eye(n), rng.normal(size=(m, n)) * 0.1

    # Init: A = 0.9 I (stable), C = random Gaussian.
    A = 0.9 * np.eye(n)
    C = rng.normal(size=(m, n)) * 0.1
    Q = 0.01 * np.eye(n)
    R = 0.1 * np.eye(m)

    for _ in range(max_iters):
        # E-step: forward filter + RTS smoother to get x̂_t and lag-one cross-covs.
        x_filt = np.zeros((T, n))
        P_filt = np.zeros((T, n, n))
        x = np.zeros(n); P = np.eye(n)
        for t in range(T):
            x_pred = A @ x
            P_pred = _symmetrise(A @ P @ A.T + Q) + JITTER * np.eye(n)
            S = _symmetrise(C @ P_pred @ C.T + R) + JITTER * np.eye(m)
            K_gain = P_pred @ C.T @ _safe_inv(S)
            innov = Y[t] - C @ x_pred
            x = x_pred + K_gain @ innov
            P = _symmetrise((np.eye(n) - K_gain @ C) @ P_pred) + JITTER * np.eye(n)
            x_filt[t] = x; P_filt[t] = P

        # RTS smoother.
        x_smooth = x_filt.copy()
        P_smooth = P_filt.copy()
        G_seq = []
        for t in range(T - 2, -1, -1):
            P_pred = _symmetrise(A @ P_filt[t] @ A.T + Q) + JITTER * np.eye(n)
            G = P_filt[t] @ A.T @ _safe_inv(P_pred)
            G_seq.insert(0, G)
            x_smooth[t] = x_filt[t] + G @ (x_smooth[t + 1] - A @ x_filt[t])
            P_smooth[t] = _symmetrise(P_filt[t] + G @ (P_smooth[t + 1] - P_pred) @ G.T)

        # Lag-one cross-cov P[t, t-1 | T] = G_{t-1} P[t|T] approximation.
        P_cross = np.zeros((T, n, n))
        for t in range(1, T):
            P_cross[t] = G_seq[t - 1] @ P_smooth[t]

        # M-step: closed-form linear-Gaussian.
        Sxx_curr = sum(np.outer(x_smooth[t], x_smooth[t]) + P_smooth[t] for t in range(1, T))
        Sxx_prev = sum(np.outer(x_smooth[t - 1], x_smooth[t - 1]) + P_smooth[t - 1] for t in range(1, T))
        Sx_cross = sum(np.outer(x_smooth[t], x_smooth[t - 1]) + P_cross[t] for t in range(1, T))
        A = Sx_cross @ _safe_inv(Sxx_prev)

        Syx = sum(np.outer(Y[t], x_smooth[t]) for t in range(T))
        Sxx_all = sum(np.outer(x_smooth[t], x_smooth[t]) + P_smooth[t] for t in range(T))
        C = Syx @ _safe_inv(Sxx_all)

        # Update Q, R for next iter only (not returned; OEMPU owns them online).
        resid_x = [x_smooth[t] - A @ x_smooth[t - 1] for t in range(1, T)]
        Q = _symmetrise(sum(np.outer(r, r) for r in resid_x) / (T - 1)) + JITTER * np.eye(n)
        resid_y = [Y[t] - C @ x_smooth[t] for t in range(T)]
        R = _symmetrise(sum(np.outer(r, r) for r in resid_y) / T) + JITTER * np.eye(m)

    return A, C


def batch_em_warm_start(
    Y: np.ndarray,
    labels: np.ndarray,
    K: int,
    n: int,
    m: int,
    max_iters: int = 20,
    subsample_frac: float = 0.1,
    random_state: int = 42,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Per-regime single-LDS EM. See module docstring."""
    rng = np.random.default_rng(random_state)
    A_list: list[np.ndarray] = []
    C_list: list[np.ndarray] = []

    for s in range(K):
        Y_s = Y[labels == s]
        if 0.0 < subsample_frac < 1.0 and len(Y_s) > 0:
            # Clamp to len(Y_s): for tiny inputs (e.g. smoke tests) we may have
            # fewer windows per regime than the floor of 2.
            keep = min(max(int(len(Y_s) * subsample_frac), 2), len(Y_s))
            idx = rng.choice(len(Y_s), size=keep, replace=False)
            idx.sort()
            Y_s = Y_s[idx]

        if len(Y_s) < 2:
            # Fallback: default init per spec §7.3.
            A_list.append(0.9 * np.eye(n))
            C_list.append(rng.normal(size=(m, n)) * 0.1)
            continue

        A_s, C_s = _single_regime_em(Y_s, n, max_iters, rng)
        A_list.append(A_s)
        C_list.append(C_s)

    return A_list, C_list

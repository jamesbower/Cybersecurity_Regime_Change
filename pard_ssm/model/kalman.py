"""Regime-bank Kalman filter + RTS smoother — spec §8.2."""
from __future__ import annotations
import numpy as np
from scipy.linalg import cho_factor, cho_solve

JITTER = 1e-6


def _symmetrise(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M + M.T)


def _safe_inv(M: np.ndarray) -> np.ndarray:
    M = _symmetrise(M) + JITTER * np.eye(M.shape[0])
    try:
        c, low = cho_factor(M)
        return cho_solve((c, low), np.eye(M.shape[0]))
    except np.linalg.LinAlgError:
        return np.linalg.pinv(M)


class KalmanBank:
    def __init__(
        self,
        A: list[np.ndarray],
        C: list[np.ndarray],
        Q: list[np.ndarray],
        R: list[np.ndarray],
    ) -> None:
        assert len(A) == len(C) == len(Q) == len(R)
        self.A = [np.asarray(a, dtype=np.float64) for a in A]
        self.C = [np.asarray(c, dtype=np.float64) for c in C]
        self.Q = [np.asarray(q, dtype=np.float64) for q in Q]
        self.R = [np.asarray(r, dtype=np.float64) for r in R]
        self.K = len(A)
        self.n = self.A[0].shape[0]
        self.m = self.C[0].shape[0]

    def effective_params(
        self, gamma_t: np.ndarray, P_prev: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Spec §8.2 weighted blend with variance-inflation."""
        A_eff = sum(gamma_t[s] * self.A[s] for s in range(self.K))
        C_eff = sum(gamma_t[s] * self.C[s] for s in range(self.K))
        Q_eff = sum(gamma_t[s] * self.Q[s] for s in range(self.K))
        for s in range(self.K):
            diff = self.A[s] - A_eff
            Q_eff = Q_eff + gamma_t[s] * (diff @ P_prev @ diff.T)
        R_eff = sum(gamma_t[s] * self.R[s] for s in range(self.K))
        return A_eff, _symmetrise(Q_eff) + JITTER * np.eye(self.n), C_eff, _symmetrise(R_eff) + JITTER * np.eye(self.m)

    def predict(
        self, x: np.ndarray, P: np.ndarray, A_eff: np.ndarray, Q_eff: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        x_pred = A_eff @ x
        P_pred = _symmetrise(A_eff @ P @ A_eff.T + Q_eff) + JITTER * np.eye(self.n)
        return x_pred, P_pred

    def update(
        self,
        x_pred: np.ndarray,
        P_pred: np.ndarray,
        y: np.ndarray,
        C_eff: np.ndarray,
        R_eff: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        innov = y - C_eff @ x_pred
        S = _symmetrise(C_eff @ P_pred @ C_eff.T + R_eff) + JITTER * np.eye(self.m)
        K_gain = P_pred @ C_eff.T @ _safe_inv(S)
        x_post = x_pred + K_gain @ innov
        P_post = _symmetrise((np.eye(self.n) - K_gain @ C_eff) @ P_pred) + JITTER * np.eye(self.n)
        return x_post, P_post, innov, S

    def rts_smooth(
        self,
        x_filt: np.ndarray,
        P_filt: np.ndarray,
        A_eff_seq: list[np.ndarray],
        Q_eff_seq: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        T = x_filt.shape[0]
        x_smooth = x_filt.copy()
        P_smooth = P_filt.copy()
        for t in range(T - 2, -1, -1):
            P_pred = _symmetrise(
                A_eff_seq[t + 1] @ P_filt[t] @ A_eff_seq[t + 1].T + Q_eff_seq[t + 1]
            ) + JITTER * np.eye(self.n)
            G = P_filt[t] @ A_eff_seq[t + 1].T @ _safe_inv(P_pred)
            x_pred = A_eff_seq[t + 1] @ x_filt[t]
            x_smooth[t] = x_filt[t] + G @ (x_smooth[t + 1] - x_pred)
            P_smooth[t] = _symmetrise(P_filt[t] + G @ (P_smooth[t + 1] - P_pred) @ G.T)
        return x_smooth, P_smooth

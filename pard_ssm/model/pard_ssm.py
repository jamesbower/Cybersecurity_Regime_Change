"""PARD-SSM orchestrator — the only stateful class."""
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Optional
import numpy as np
from ..config import PARDSSMConfig
from .kalman import KalmanBank
from .vsipc import run_coordinate_ascent
from .batch_em import batch_em_warm_start
from .oempu import online_m_step
from .pkar import gate, KillChainAlert


@dataclass
class InferenceResult:
    gamma: np.ndarray
    alert: Optional[KillChainAlert]
    latency_ms: float


def _init_transitions(K: int, n: int) -> list[np.ndarray]:
    A = []
    A0 = 0.9 * np.eye(n); A.append(A0)
    A1 = 0.85 * np.eye(n); A1[0, 1] = 0.1; A.append(A1)
    A2 = 0.80 * np.eye(n); A2[4 % n, 5 % n] = 0.15; A.append(A2)
    A3 = np.eye(n); A3[0, 0] = 1.02; A.append(A3)
    while len(A) < K:
        A.append(0.9 * np.eye(n))
    return A[:K]


def _init_observation(K: int, n: int, m: int, rng: np.random.Generator) -> list[np.ndarray]:
    return [rng.normal(size=(m, n)) * 0.1 for _ in range(K)]


def _init_Pi(K: int) -> np.ndarray:
    if K == 4:
        return np.array([
            [0.90, 0.09, 0.01, 0.00],
            [0.10, 0.75, 0.14, 0.01],
            [0.00, 0.05, 0.75, 0.20],
            [0.02, 0.00, 0.04, 0.94],
        ])
    # Fallback for K != 4: strong diagonal.
    Pi = 0.9 * np.eye(K) + 0.1 / K * np.ones((K, K))
    return Pi / Pi.sum(axis=1, keepdims=True)


def _init_pi0(K: int) -> np.ndarray:
    if K == 4:
        return np.array([0.97, 0.01, 0.01, 0.01])
    v = np.full(K, 0.01 / max(K - 1, 1))
    v[0] = 0.99 - 0.01 * (K - 1) + 0.01
    return v / v.sum()


class PARDSSM:
    def __init__(self, cfg: PARDSSMConfig, random_state: int = 42) -> None:
        self.cfg = cfg
        rng = np.random.default_rng(random_state)
        K, n, m = cfg.model.K, cfg.model.n, cfg.model.m
        self.K, self.n, self.m = K, n, m

        self.A = _init_transitions(K, n)
        self.C = _init_observation(K, n, m, rng)
        self.Q = [0.01 * np.eye(n), 0.10 * np.eye(n), 0.05 * np.eye(n), 0.02 * np.eye(n)][:K]
        while len(self.Q) < K:
            self.Q.append(0.05 * np.eye(n))
        self.R = [0.1 * np.eye(m) for _ in range(K)]

        self.Pi = _init_Pi(K)
        self.pi0 = _init_pi0(K)

        self.gamma_prev = self.pi0.copy()
        self.P_prev = np.eye(n)
        self.A_eff_prev = self.A[0].copy()
        self.collapse_counter = 0
        self.warm_started = False

    def warm_start(self, Y_train: np.ndarray, labels_train: np.ndarray) -> None:
        A_list, C_list = batch_em_warm_start(
            Y_train, labels_train,
            K=self.K, n=self.n, m=self.m,
            max_iters=self.cfg.batch_em.max_iters,
            subsample_frac=self.cfg.batch_em.subsample_frac,
            random_state=self.cfg.batch_em.random_state,
        )
        self.A = A_list
        self.C = C_list
        self.warm_started = True

    def _bank(self) -> KalmanBank:
        return KalmanBank(A=self.A, C=self.C, Q=self.Q, R=self.R)

    def infer_window(self, y_t: np.ndarray, t: float) -> InferenceResult:
        start = time.perf_counter()
        bank = self._bank()

        # Single-window inference: T=1; use gamma_prev as prior on gamma_init.
        Y = y_t.reshape(1, -1)
        gamma_init = self.gamma_prev[None, :].copy()
        gamma_seq, xi_seq, x_smooth, P_smooth, _ = run_coordinate_ascent(
            Y, bank, self.Pi, self.gamma_prev,
            k_max=self.cfg.model.k_max,
            epsilon=self.cfg.model.epsilon,
            gamma_init=gamma_init,
        )
        gamma_t = gamma_seq[0]

        # Build a synthetic xi from gamma_prev -> gamma_t for the M-step.
        xi_t = np.outer(self.gamma_prev, gamma_t)
        A_eff, _, _, _ = bank.effective_params(gamma_t, self.P_prev)

        self.Pi, self.R, self.Q = online_m_step(
            Pi=self.Pi, R=self.R, Q=self.Q,
            gamma_t=gamma_t, gamma_prev=self.gamma_prev, xi_t=xi_t,
            y_t=y_t, x_filt=x_smooth[0], P_filt=P_smooth[0],
            P_smooth_prev=self.P_prev, A_eff=A_eff, C_list=self.C,
            eta=self.cfg.online_em.eta,
        )

        alert = gate(gamma_t, self.gamma_prev, self.Pi, self.cfg.kl_gating.tau_kl, t)

        # Mode-collapse tracking.
        if gamma_t.max() > 0.99:
            self.collapse_counter += 1
            if self.collapse_counter == self.cfg.mode_collapse.window_threshold:
                import logging
                logging.getLogger(__name__).warning(
                    "Mode collapse: gamma peaked >0.99 for %d windows. Pi=%s",
                    self.collapse_counter, self.Pi,
                )
        else:
            self.collapse_counter = 0

        self.gamma_prev = gamma_t
        self.P_prev = P_smooth[0]
        self.A_eff_prev = A_eff

        latency_ms = (time.perf_counter() - start) * 1000.0
        return InferenceResult(gamma=gamma_t, alert=alert, latency_ms=latency_ms)

    def state_snapshot(self) -> dict:
        return {
            "Pi": self.Pi.copy(),
            "gamma_prev": self.gamma_prev.copy(),
            "Q": [q.copy() for q in self.Q],
            "R": [r.copy() for r in self.R],
            "warm_started": self.warm_started,
        }

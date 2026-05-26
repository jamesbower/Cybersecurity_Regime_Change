"""Predictive KL Alert Records — spec §10."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

REGIME_NAMES = {0: "Normal", 1: "Reconnaissance", 2: "Intrusion", 3: "Exfiltration"}


@dataclass
class KillChainAlert:
    timestamp: float
    stage_posterior: np.ndarray
    kl_score: float
    elbo_entropy: float
    current_stage: str
    predicted_stage: str
    predicted_posterior: np.ndarray


def compute_kl(gamma_t: np.ndarray, gamma_prev: np.ndarray) -> float:
    """Discrete KL(gamma_t || gamma_prev) with clipping."""
    g_t = np.clip(gamma_t, 1e-10, 1.0)
    g_prev = np.clip(gamma_prev, 1e-10, 1.0)
    return float(np.sum(g_t * np.log(g_t / g_prev)))


def predict_ahead(gamma_t: np.ndarray, Pi: np.ndarray, tau: int) -> np.ndarray:
    """gamma_hat_{t+tau}(s') = (Pi^tau)^T gamma_t  ->  marginal at t+tau."""
    if tau == 0:
        return gamma_t.copy()
    Pi_tau = np.linalg.matrix_power(Pi, tau)
    return Pi_tau.T @ gamma_t


def gate(
    gamma_t: np.ndarray,
    gamma_prev: np.ndarray,
    Pi: np.ndarray,
    tau_kl: float,
    timestamp: float,
) -> Optional[KillChainAlert]:
    kl = compute_kl(gamma_t, gamma_prev)
    if kl <= tau_kl:
        return None
    g_t_clip = np.clip(gamma_t, 1e-10, 1.0)
    elbo_entropy = float(-np.sum(g_t_clip * np.log(g_t_clip)))
    pred = predict_ahead(gamma_t, Pi, tau=1)
    return KillChainAlert(
        timestamp=timestamp,
        stage_posterior=gamma_t.copy(),
        kl_score=kl,
        elbo_entropy=elbo_entropy,
        current_stage=REGIME_NAMES[int(np.argmax(gamma_t))],
        predicted_stage=REGIME_NAMES[int(np.argmax(pred))],
        predicted_posterior=pred,
    )

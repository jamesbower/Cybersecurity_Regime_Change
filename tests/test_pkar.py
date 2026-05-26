import numpy as np
import pytest
from scipy.stats import entropy as scipy_entropy
from pard_ssm.model.pkar import (
    compute_kl, predict_ahead, gate, KillChainAlert, REGIME_NAMES,
)


def test_kl_identical_distributions_zero():
    g = np.array([0.25, 0.25, 0.25, 0.25])
    assert compute_kl(g, g) == pytest.approx(0.0, abs=1e-10)


def test_kl_matches_scipy(rng):
    g_t = rng.dirichlet(np.ones(4))
    g_prev = rng.dirichlet(np.ones(4))
    expected = scipy_entropy(g_t, g_prev)
    assert compute_kl(g_t, g_prev) == pytest.approx(float(expected), abs=1e-8)


def test_predict_ahead_tau_zero_is_identity():
    g = np.array([0.6, 0.3, 0.05, 0.05])
    Pi = np.array([[0.9, 0.05, 0.03, 0.02],
                   [0.1, 0.7, 0.15, 0.05],
                   [0.05, 0.1, 0.7, 0.15],
                   [0.02, 0.03, 0.05, 0.9]])
    out = predict_ahead(g, Pi, tau=0)
    np.testing.assert_allclose(out, g)


def test_predict_ahead_identity_pi_returns_input():
    g = np.array([0.6, 0.3, 0.05, 0.05])
    Pi = np.eye(4)
    out = predict_ahead(g, Pi, tau=10)
    np.testing.assert_allclose(out, g)


def test_gate_below_threshold_returns_none():
    g = np.array([0.6, 0.3, 0.05, 0.05])
    Pi = np.eye(4)
    alert = gate(g, g, Pi, tau_kl=0.5, timestamp=0.0)
    assert alert is None


def test_gate_above_threshold_returns_alert():
    g_t = np.array([0.01, 0.97, 0.01, 0.01])
    g_prev = np.array([0.97, 0.01, 0.01, 0.01])
    Pi = np.eye(4)
    alert = gate(g_t, g_prev, Pi, tau_kl=0.5, timestamp=123.4)
    assert isinstance(alert, KillChainAlert)
    assert alert.timestamp == 123.4
    assert alert.current_stage == REGIME_NAMES[1]
    assert alert.kl_score > 0.5

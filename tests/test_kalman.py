import numpy as np
import pytest
from pard_ssm.model.kalman import KalmanBank


def test_single_regime_matches_hand_rolled_kalman(mini_bank_params):
    p = mini_bank_params
    bank = KalmanBank(A=p["A"], C=p["C"], Q=p["Q"], R=p["R"])

    # Use γ = [1, 0] so effective params == regime-0 params exactly.
    gamma = np.array([1.0, 0.0])
    A_eff, Q_eff, C_eff, R_eff = bank.effective_params(gamma, P_prev=np.eye(p["n"]))
    np.testing.assert_allclose(A_eff, p["A"][0])
    np.testing.assert_allclose(C_eff, p["C"][0])

    # Single-step predict + update against textbook Kalman.
    x = np.zeros(p["n"])
    P = np.eye(p["n"])
    y = np.array([1.0, 0.5, -0.3, 0.2])
    x_pred, P_pred = bank.predict(x, P, A_eff, Q_eff)
    x_post, P_post, innov, S = bank.update(x_pred, P_pred, y, C_eff, R_eff)

    # Textbook one-liners:
    x_pred_ref = p["A"][0] @ x
    P_pred_ref = p["A"][0] @ P @ p["A"][0].T + p["Q"][0]
    S_ref = p["C"][0] @ P_pred_ref @ p["C"][0].T + p["R"][0]
    K_gain = P_pred_ref @ p["C"][0].T @ np.linalg.inv(S_ref)
    innov_ref = y - p["C"][0] @ x_pred_ref
    x_post_ref = x_pred_ref + K_gain @ innov_ref
    P_post_ref = (np.eye(p["n"]) - K_gain @ p["C"][0]) @ P_pred_ref

    np.testing.assert_allclose(x_pred, x_pred_ref, atol=1e-10)
    # Covariance assertions loosened to 1e-5 to accommodate JITTER=1e-6 added
    # twice (once in effective_params for Q_eff, once in predict for P_pred).
    np.testing.assert_allclose(P_pred, P_pred_ref, atol=1e-5)
    np.testing.assert_allclose(S, S_ref, atol=1e-5)
    np.testing.assert_allclose(innov, innov_ref, atol=1e-5)
    np.testing.assert_allclose(x_post, x_post_ref, atol=1e-5)
    np.testing.assert_allclose(P_post, P_post_ref, atol=1e-5)


def test_effective_params_weighted_blend(mini_bank_params):
    p = mini_bank_params
    bank = KalmanBank(A=p["A"], C=p["C"], Q=p["Q"], R=p["R"])
    gamma = np.array([0.5, 0.5])
    A_eff, Q_eff, C_eff, R_eff = bank.effective_params(gamma, P_prev=np.eye(p["n"]))
    np.testing.assert_allclose(A_eff, 0.5 * (p["A"][0] + p["A"][1]))
    np.testing.assert_allclose(C_eff, 0.5 * (p["C"][0] + p["C"][1]))


def test_safe_inv_handles_near_singular():
    from pard_ssm.model.kalman import _safe_inv
    M = np.array([[1.0, 1.0], [1.0, 1.0]])  # singular
    Minv = _safe_inv(M)
    assert np.all(np.isfinite(Minv))


def test_rts_smoother_recovers_static_signal(mini_bank_params):
    """A=I, Q small, C=I, R small → smoothed states ≈ observations."""
    n, m = 3, 3
    bank = KalmanBank(
        A=[np.eye(n)], C=[np.eye(n)],
        Q=[1e-6 * np.eye(n)], R=[1e-6 * np.eye(n)],
    )
    T = 10
    Y = np.tile(np.array([1.0, 2.0, 3.0]), (T, 1))
    gamma = np.ones((T, 1))  # always regime 0
    # Forward pass
    x_filt = np.zeros((T, n))
    P_filt = np.zeros((T, n, n))
    x = np.zeros(n); P = np.eye(n)
    A_eff_seq = []
    for t in range(T):
        A_eff, Q_eff, C_eff, R_eff = bank.effective_params(gamma[t], P)
        A_eff_seq.append(A_eff)
        x_pred, P_pred = bank.predict(x, P, A_eff, Q_eff)
        x, P, _, _ = bank.update(x_pred, P_pred, Y[t], C_eff, R_eff)
        x_filt[t] = x; P_filt[t] = P
    x_smooth, _ = bank.rts_smooth(x_filt, P_filt, A_eff_seq, Q_eff_seq=[1e-6 * np.eye(n)] * T)
    np.testing.assert_allclose(x_smooth[-1], [1.0, 2.0, 3.0], atol=1e-3)

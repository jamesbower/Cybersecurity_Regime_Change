import numpy as np
import pytest
from pard_ssm.model.oempu import online_m_step


def _toy_inputs():
    K, n, m = 2, 3, 4
    Pi = np.array([[0.9, 0.1], [0.2, 0.8]])
    R = [0.1 * np.eye(m) for _ in range(K)]
    Q = [0.01 * np.eye(n) for _ in range(K)]
    gamma_t = np.array([0.7, 0.3])
    gamma_prev = np.array([0.6, 0.4])
    xi_t = np.array([[0.5, 0.1], [0.15, 0.25]])
    y_t = np.ones(m)
    x_filt = 0.5 * np.ones(n)
    P_filt = 0.1 * np.eye(n)
    P_smooth_prev = 0.1 * np.eye(n)
    A_eff = 0.9 * np.eye(n)
    C_list = [np.eye(m, n), 0.5 * np.eye(m, n)]
    return dict(
        Pi=Pi, R=R, Q=Q, gamma_t=gamma_t, gamma_prev=gamma_prev, xi_t=xi_t,
        y_t=y_t, x_filt=x_filt, P_filt=P_filt, P_smooth_prev=P_smooth_prev,
        A_eff=A_eff, C_list=C_list,
    )


def test_eta_zero_means_no_change():
    inp = _toy_inputs()
    Pi2, R2, Q2 = online_m_step(eta=0.0, **inp)
    np.testing.assert_allclose(Pi2, inp["Pi"])
    for s in range(2):
        np.testing.assert_allclose(R2[s], inp["R"][s])
        np.testing.assert_allclose(Q2[s], inp["Q"][s])


def test_pi_rows_sum_to_one():
    inp = _toy_inputs()
    Pi2, _, _ = online_m_step(eta=0.5, **inp)
    np.testing.assert_allclose(Pi2.sum(axis=1), np.ones(2), atol=1e-12)


def test_r_remains_psd():
    inp = _toy_inputs()
    _, R2, _ = online_m_step(eta=0.5, **inp)
    for Rs in R2:
        eigvals = np.linalg.eigvalsh(Rs)
        assert np.all(eigvals >= -1e-10), f"R has negative eigenvalue: {eigvals}"


def test_q_remains_psd():
    inp = _toy_inputs()
    _, _, Q2 = online_m_step(eta=0.5, **inp)
    for Qs in Q2:
        eigvals = np.linalg.eigvalsh(Qs)
        assert np.all(eigvals >= -1e-10)

from pathlib import Path
import numpy as np
import pytest
from pard_ssm.config import load_config
from pard_ssm.model.pard_ssm import PARDSSM, InferenceResult


CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


@pytest.fixture
def default_cfg():
    return load_config(CONFIG_DIR / "pard_ssm_default.yaml")


def test_init_with_default_config_creates_K_matrices(default_cfg):
    model = PARDSSM(default_cfg)
    assert len(model.A) == default_cfg.model.K
    assert len(model.C) == default_cfg.model.K
    assert model.Pi.shape == (default_cfg.model.K, default_cfg.model.K)


def test_warm_start_freezes_A_and_C(default_cfg, rng):
    model = PARDSSM(default_cfg)
    T = 100
    Y = rng.normal(size=(T, default_cfg.model.m))
    labels = rng.integers(0, default_cfg.model.K, size=T)
    model.warm_start(Y, labels)
    A_before = [a.copy() for a in model.A]
    C_before = [c.copy() for c in model.C]
    for _ in range(3):
        model.infer_window(rng.normal(size=default_cfg.model.m), t=0.0)
    for s in range(default_cfg.model.K):
        np.testing.assert_array_equal(model.A[s], A_before[s])
        np.testing.assert_array_equal(model.C[s], C_before[s])


def test_infer_window_returns_inference_result(default_cfg, rng):
    model = PARDSSM(default_cfg)
    Y = rng.normal(size=(50, default_cfg.model.m))
    labels = rng.integers(0, default_cfg.model.K, size=50)
    model.warm_start(Y, labels)
    out = model.infer_window(rng.normal(size=default_cfg.model.m), t=1.5)
    assert isinstance(out, InferenceResult)
    assert out.gamma.shape == (default_cfg.model.K,)
    assert np.isclose(out.gamma.sum(), 1.0, atol=1e-8)
    assert out.latency_ms >= 0.0


def test_gamma_prev_persists_across_calls(default_cfg, rng):
    model = PARDSSM(default_cfg)
    Y = rng.normal(size=(50, default_cfg.model.m))
    labels = rng.integers(0, default_cfg.model.K, size=50)
    model.warm_start(Y, labels)
    out1 = model.infer_window(rng.normal(size=default_cfg.model.m), t=0.0)
    saved_prev = model.gamma_prev.copy()
    assert np.allclose(saved_prev, out1.gamma)
    out2 = model.infer_window(rng.normal(size=default_cfg.model.m), t=1.0)
    # After a second call, gamma_prev should now equal out2.gamma.
    assert np.allclose(model.gamma_prev, out2.gamma)

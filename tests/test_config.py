from pathlib import Path
import pytest
from pard_ssm.config import load_config, PARDSSMConfig


CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def test_load_default_returns_dataclass():
    cfg = load_config(CONFIG_DIR / "pard_ssm_default.yaml")
    assert isinstance(cfg, PARDSSMConfig)
    assert cfg.model.K == 4
    assert cfg.model.n == 8
    assert cfg.model.m == 17
    assert cfg.online_em.eta == pytest.approx(0.01)
    assert cfg.kl_gating.tau_kl == pytest.approx(2.0)


def test_load_with_dataset_override_merges():
    cfg = load_config(
        CONFIG_DIR / "pard_ssm_default.yaml",
        CONFIG_DIR / "cicids2017.yaml",
    )
    assert cfg.dataset.name == "cicids2017"
    assert cfg.dataset.split_ratios == [0.6, 0.2, 0.2]
    assert cfg.model.K == 4  # base config preserved


def test_validation_rejects_negative_k(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "model: {K: -1, n: 8, m: 17, k_max: 15, epsilon: 1.0e-4}\n"
        "online_em: {eta: 0.01}\n"
        "kl_gating: {tau_kl: 2.0}\n"
        "window: {W: 1.0, overlap_s: 0.5}\n"
        "batch_em: {max_iters: 20, subsample_frac: 0.1, random_state: 42}\n"
        "mode_collapse: {window_threshold: 100}\n"
        "logging: {level: INFO}\n"
    )
    with pytest.raises(ValueError, match="K"):
        load_config(bad)

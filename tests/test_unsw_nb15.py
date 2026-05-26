from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from pard_ssm.data.unsw_nb15 import (
    load_raw, one_hot_encode, map_labels, fit_pca, UNSW_REGIME_MAP,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _setup_raw(tmp_path):
    import shutil
    dest = tmp_path / "unsw_nb15"
    dest.mkdir()
    shutil.copy(FIXTURE_DIR / "unsw_tiny_train.csv", dest / "UNSW_NB15_training-set.csv")
    shutil.copy(FIXTURE_DIR / "unsw_tiny_test.csv", dest / "UNSW_NB15_testing-set.csv")
    return dest


def test_load_raw_returns_train_test_tuple(tmp_path):
    dest = _setup_raw(tmp_path)
    train, test = load_raw(dest)
    assert len(train) == 5
    assert len(test) == 3
    assert "attack_cat" in train.columns


def test_one_hot_encode_expands_categoricals(tmp_path):
    dest = _setup_raw(tmp_path)
    train, _ = load_raw(dest)
    enc = one_hot_encode(train, cols=["proto", "service", "state"])
    assert "proto" not in enc.columns
    assert any(c.startswith("proto_") for c in enc.columns)


def test_map_labels_resolves_attack_cats(tmp_path):
    dest = _setup_raw(tmp_path)
    train, _ = load_raw(dest)
    out = map_labels(train)
    assert "regime" in out.columns
    assert (out.loc[out["attack_cat"] == "Normal", "regime"] == 0).all()
    assert (out.loc[out["attack_cat"] == "Reconnaissance", "regime"] == 1).all()
    assert (out.loc[out["attack_cat"] == "Exploits", "regime"] == 2).all()


def test_map_labels_raises_on_unknown():
    df = pd.DataFrame({"attack_cat": ["Normal", "MysteryAttack"]})
    with pytest.raises(ValueError, match="MysteryAttack"):
        map_labels(df)


def test_pca_fits_and_transforms_to_n_components(rng):
    X = rng.normal(size=(100, 30))
    pca = fit_pca(X, n_components=17, random_state=0)
    Xt = pca.transform(X)
    assert Xt.shape == (100, 17)

"""UNSW-NB15 loader + PCA-to-17 projection — spec §5.2."""
from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

log = logging.getLogger(__name__)

UNSW_REGIME_MAP: dict[str, int] = {
    "Normal": 0, "normal": 0,
    "Reconnaissance": 1, "Fuzzers": 1, "Analysis": 1,
    "Backdoors": 2, "Backdoor": 2, "Exploits": 2, "Shellcode": 2, "Worms": 2, "Generic": 2,
    "DoS": 3,
}


def load_raw(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_dir = Path(raw_dir)
    train_path = raw_dir / "UNSW_NB15_training-set.csv"
    test_path = raw_dir / "UNSW_NB15_testing-set.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Expected {train_path} and {test_path}; place per README data placement section."
        )
    train = pd.read_csv(train_path, low_memory=False)
    test = pd.read_csv(test_path, low_memory=False)
    log.info("UNSW load: %d train + %d test rows", len(train), len(test))
    return train, test


def one_hot_encode(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    present = [c for c in cols if c in df.columns]
    if not present:
        return df.copy()
    return pd.get_dummies(df, columns=present, drop_first=False, dtype=np.float64)


def map_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["attack_cat"] = df["attack_cat"].astype(str).str.strip()
    unknown = set(df["attack_cat"].unique()) - set(UNSW_REGIME_MAP.keys())
    if unknown:
        raise ValueError(f"Unknown UNSW attack_cat: {sorted(unknown)}")
    df["regime"] = df["attack_cat"].map(UNSW_REGIME_MAP).astype(int)
    return df


def fit_pca(X_train: np.ndarray, n_components: int = 17, random_state: int = 42) -> PCA:
    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(X_train)
    return pca

#!/usr/bin/env python3
"""Preprocess UNSW-NB15: load → one-hot → label-map → PCA-to-N → parquet."""
from __future__ import annotations
import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pard_ssm.data.unsw_nb15 import load_raw, one_hot_encode, map_labels, fit_pca


DROP_COLS = ["id", "Label", "attack_cat", "regime"]


def _features_matrix(df: pd.DataFrame) -> np.ndarray:
    feat = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
    feat = feat.select_dtypes(include=[np.number, "bool"])
    return feat.to_numpy(dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/unsw_nb15"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--n-components", type=int, default=17)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = load_raw(args.raw_dir)
    train_df = map_labels(train_df)
    test_df = map_labels(test_df)
    train_oh = one_hot_encode(train_df, cols=["proto", "service", "state"])
    test_oh = one_hot_encode(test_df, cols=["proto", "service", "state"])
    # Align columns so PCA sees identical feature ordering.
    test_oh = test_oh.reindex(columns=train_oh.columns, fill_value=0.0)

    X_train = _features_matrix(train_oh)
    X_test = _features_matrix(test_oh)

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    pca = fit_pca(X_train_s, n_components=args.n_components, random_state=args.random_state)
    Y_train = pca.transform(X_train_s)
    Y_test = pca.transform(X_test_s)

    for name, Y, regimes in [("train", Y_train, train_df["regime"].to_numpy()),
                             ("test", Y_test, test_df["regime"].to_numpy())]:
        cols = {f"y_{i + 1}": Y[:, i] for i in range(Y.shape[1])}
        cols["label"] = regimes
        cols["t_window"] = np.arange(Y.shape[0], dtype=np.float64)
        out = pd.DataFrame(cols)
        out_path = args.out_dir / f"unsw_nb15_{name}.parquet"
        out.to_parquet(out_path, index=False)
        logging.getLogger(__name__).info("wrote %s (%d rows)", out_path, len(out))


if __name__ == "__main__":
    main()

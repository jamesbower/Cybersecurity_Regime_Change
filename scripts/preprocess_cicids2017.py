#!/usr/bin/env python3
"""Preprocess CICIDS2017: load → label-map → time-split → FEOV → parquet."""
from __future__ import annotations
import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from pard_ssm.data.cicids2017 import load_raw, map_labels, time_split
from pard_ssm.features.feov import compute_feov


def _windowed_labels(df: pd.DataFrame, n_windows: int) -> np.ndarray:
    """Majority-attack label per window: if any attack flow present → most common attack regime; else 0."""
    if "Timestamp" not in df.columns or "regime" not in df.columns:
        return np.zeros(n_windows, dtype=int)
    if n_windows == 0:
        return np.zeros(0, dtype=int)
    buckets = np.array_split(df["regime"].to_numpy(), n_windows)
    return np.array([
        int(np.bincount(b[b != 0]).argmax()) if (b != 0).any() else 0
        for b in buckets
    ], dtype=int)


def _split_to_parquet(name: str, df: pd.DataFrame, out_dir: Path, window_s: float, overlap_s: float) -> None:
    Y = compute_feov(df, window_s=window_s, overlap_s=overlap_s)
    labels = _windowed_labels(df, n_windows=Y.shape[0])
    cols = {f"y_{i + 1}": Y[:, i] for i in range(17)}
    cols["label"] = labels
    cols["t_window"] = np.arange(Y.shape[0]) * (window_s - overlap_s)
    out = pd.DataFrame(cols)
    out_path = out_dir / f"cicids2017_{name}.parquet"
    out.to_parquet(out_path, index=False)
    logging.getLogger(__name__).info("wrote %s (%d rows)", out_path, len(out))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/cicids2017"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--window-s", type=float, default=1.0)
    parser.add_argument("--overlap-s", type=float, default=0.5)
    parser.add_argument("--split-ratios", type=str, default="0.6,0.2,0.2")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ratios = tuple(float(x) for x in args.split_ratios.split(","))

    df = load_raw(args.raw_dir)
    df = map_labels(df)
    train, val, test = time_split(df, ratios=ratios)
    for name, split in [("train", train), ("val", val), ("test", test)]:
        _split_to_parquet(name, split, args.out_dir, args.window_s, args.overlap_s)


if __name__ == "__main__":
    main()

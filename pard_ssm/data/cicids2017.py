"""CICIDS2017 loader + kill-chain label mapping — spec §5.1."""
from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

CICIDS_REGIME_MAP: dict[str, int] = {
    "BENIGN": 0,
    "PortScan": 1, "FTP-Patator": 1, "SSH-Patator": 1,
    "Bot": 2, "Infiltration": 2,
    "Web Attack - Brute Force": 2, "Web Attack - XSS": 2, "Web Attack - Sql Injection": 2,
    "Heartbleed": 2,
    "DoS Hulk": 3, "DoS GoldenEye": 3, "DoS slowloris": 3, "DoS Slowhttptest": 3, "DDoS": 3,
}


def load_raw(raw_dir: Path) -> pd.DataFrame:
    """Concat all *.csv under raw_dir, strip col whitespace, drop NaN/Inf/dupes."""
    raw_dir = Path(raw_dir)
    csvs = sorted(raw_dir.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files under {raw_dir}")
    frames = []
    for p in csvs:
        df = pd.read_csv(p, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    before = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df.drop_duplicates().reset_index(drop=True)
    log.info("CICIDS load: %d rows → %d after dropna/dedupe", before, len(df))
    return df


def map_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Label"] = df["Label"].astype(str).str.strip()
    unknown = set(df["Label"].unique()) - set(CICIDS_REGIME_MAP.keys())
    if unknown:
        raise ValueError(f"Unknown CICIDS labels: {sorted(unknown)}")
    df["regime"] = df["Label"].map(CICIDS_REGIME_MAP).astype(int)
    return df


def time_split(
    df: pd.DataFrame, ratios: tuple[float, float, float] = (0.6, 0.2, 0.2)
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1.0"
    if "Timestamp" in df.columns:
        df = df.copy()
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df = df.sort_values("Timestamp").reset_index(drop=True)
    n = len(df)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train = df.iloc[:n_train].reset_index(drop=True)
    val = df.iloc[n_train:n_train + n_val].reset_index(drop=True)
    test = df.iloc[n_train + n_val:].reset_index(drop=True)
    return train, val, test

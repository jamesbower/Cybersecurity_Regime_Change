from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from pard_ssm.data.cicids2017 import load_raw, map_labels, time_split, CICIDS_REGIME_MAP


FIXTURE = Path(__file__).parent / "fixtures" / "cicids_tiny.csv"


def test_load_raw_strips_column_whitespace(tmp_path):
    import shutil
    dest_dir = tmp_path / "cicids2017"
    dest_dir.mkdir()
    shutil.copy(FIXTURE, dest_dir / "tiny.csv")
    df = load_raw(dest_dir)
    assert "Timestamp" in df.columns  # leading space stripped
    assert "Label" in df.columns


def test_map_labels_resolves_known_attacks(tmp_path):
    import shutil
    dest_dir = tmp_path / "cicids2017"
    dest_dir.mkdir()
    shutil.copy(FIXTURE, dest_dir / "tiny.csv")
    df = load_raw(dest_dir)
    df = map_labels(df)
    assert "regime" in df.columns
    assert df["regime"].isin([0, 1, 2, 3]).all()
    assert (df.loc[df["Label"] == "PortScan", "regime"] == 1).all()
    assert (df.loc[df["Label"] == "DDoS", "regime"] == 3).all()


def test_map_labels_raises_on_unknown(tmp_path):
    df = pd.DataFrame({"Label": ["BENIGN", "Mystery"]})
    with pytest.raises(ValueError, match="Mystery"):
        map_labels(df)


def test_map_labels_strips_label_whitespace(tmp_path):
    df = pd.DataFrame({"Label": [" BENIGN ", "BENIGN"]})
    out = map_labels(df)
    assert (out["regime"] == 0).all()


def test_time_split_preserves_temporal_order():
    df = pd.DataFrame({
        "Timestamp": pd.to_datetime(["2024-01-01 00:00:0%d" % i for i in range(10)]),
        "value": list(range(10)),
    })
    train, val, test = time_split(df, ratios=(0.6, 0.2, 0.2))
    assert len(train) == 6
    assert len(val) == 2
    assert len(test) == 2
    assert train["value"].tolist() == [0, 1, 2, 3, 4, 5]
    assert val["value"].tolist() == [6, 7]
    assert test["value"].tolist() == [8, 9]

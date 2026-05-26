import subprocess
import sys
import shutil
from pathlib import Path
import pandas as pd
import pytest


FIXTURE = Path(__file__).parent / "fixtures" / "cicids_tiny.csv"
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "preprocess_cicids2017.py"


def test_preprocess_writes_three_parquets(tmp_path):
    raw_dir = tmp_path / "raw" / "cicids2017"
    raw_dir.mkdir(parents=True)
    shutil.copy(FIXTURE, raw_dir / "tiny.csv")
    out_dir = tmp_path / "processed"
    out_dir.mkdir()

    result = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--raw-dir", str(raw_dir),
         "--out-dir", str(out_dir),
         "--window-s", "1.0",
         "--overlap-s", "0.0",
         "--split-ratios", "0.5,0.25,0.25"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    for split in ["train", "val", "test"]:
        p = out_dir / f"cicids2017_{split}.parquet"
        assert p.exists(), f"missing {p}"
        df = pd.read_parquet(p)
        assert "label" in df.columns
        assert all(f"y_{i}" in df.columns for i in range(1, 18))

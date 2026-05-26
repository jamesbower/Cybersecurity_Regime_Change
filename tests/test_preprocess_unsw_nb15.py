import subprocess
import sys
import shutil
from pathlib import Path
import pandas as pd
import pytest


FIXTURE_DIR = Path(__file__).parent / "fixtures"
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "preprocess_unsw_nb15.py"


def test_preprocess_writes_train_test_parquets(tmp_path):
    raw_dir = tmp_path / "raw" / "unsw_nb15"
    raw_dir.mkdir(parents=True)
    shutil.copy(FIXTURE_DIR / "unsw_tiny_train.csv", raw_dir / "UNSW_NB15_training-set.csv")
    shutil.copy(FIXTURE_DIR / "unsw_tiny_test.csv", raw_dir / "UNSW_NB15_testing-set.csv")
    out_dir = tmp_path / "processed"
    out_dir.mkdir()

    result = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--raw-dir", str(raw_dir),
         "--out-dir", str(out_dir),
         "--n-components", "3"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    for split in ["train", "test"]:
        p = out_dir / f"unsw_nb15_{split}.parquet"
        assert p.exists()
        df = pd.read_parquet(p)
        assert "label" in df.columns
        assert any(c.startswith("y_") for c in df.columns)

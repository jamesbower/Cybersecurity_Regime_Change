import subprocess
import sys
import shutil
import json
from pathlib import Path
import pytest


REPO = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO / "tests" / "fixtures"


def test_smoke_cicids_end_to_end(tmp_path):
    # Stage raw fixture into a tmp data dir.
    raw_dir = tmp_path / "raw" / "cicids2017"
    raw_dir.mkdir(parents=True)
    shutil.copy(FIXTURE_DIR / "cicids_tiny.csv", raw_dir / "tiny.csv")
    out_dir = tmp_path / "processed"
    out_dir.mkdir()
    # Preprocess.
    pre = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "preprocess_cicids2017.py"),
         "--raw-dir", str(raw_dir),
         "--out-dir", str(out_dir),
         "--window-s", "1.0", "--overlap-s", "0.0",
         "--split-ratios", "0.6,0.2,0.2"],
        capture_output=True, text=True,
    )
    assert pre.returncode == 0, pre.stderr

    # Run experiment.
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    run = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "run_experiment.py"),
         "--dataset", "cicids2017",
         "--config", str(REPO / "config" / "pard_ssm_default.yaml"),
         "--processed-dir", str(out_dir),
         "--results-dir", str(results_dir),
         "--run-id", "smoke"],
        capture_output=True, text=True,
    )
    assert run.returncode == 0, run.stderr

    metrics_path = results_dir / "cicids2017_smoke" / "metrics.json"
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text())
    for key in ["f1", "fpr", "saa", "latency"]:
        assert key in metrics
    assert "mean_ms" in metrics["latency"]

import time
import numpy as np
import pytest
from sklearn.metrics import f1_score, confusion_matrix
from pard_ssm.evaluation.metrics import (
    weighted_f1, fpr_binary, stage_attribution_accuracy, LatencyProfiler,
)


def test_perfect_predictions():
    y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    y_pred = y_true.copy()
    assert weighted_f1(y_true, y_pred) == pytest.approx(1.0)
    assert fpr_binary(y_true, y_pred) == pytest.approx(0.0)
    assert stage_attribution_accuracy(y_true, y_pred) == pytest.approx(1.0)


def test_weighted_f1_matches_sklearn(rng):
    y_true = rng.integers(0, 4, size=100)
    y_pred = rng.integers(0, 4, size=100)
    assert weighted_f1(y_true, y_pred) == pytest.approx(
        float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    )


def test_fpr_collapses_regimes():
    y_true = np.array([0, 0, 0, 1, 2, 3])  # 3 normal, 3 attack
    y_pred = np.array([1, 0, 0, 0, 0, 3])  # one FP, two FN
    # binary: true=[0,0,0,1,1,1], pred=[1,0,0,0,0,1]
    tn, fp, fn, tp = confusion_matrix([0, 0, 0, 1, 1, 1], [1, 0, 0, 0, 0, 1]).ravel()
    assert fpr_binary(y_true, y_pred) == pytest.approx(fp / (fp + tn))


def test_saa_excludes_normal_class():
    y_true = np.array([0, 0, 1, 2, 3])
    y_pred = np.array([3, 3, 1, 2, 0])  # all normal predictions are "wrong"
    # SAA looks only at attack windows: true=[1,2,3], pred=[1,2,0] → 2/3 correct
    assert stage_attribution_accuracy(y_true, y_pred) == pytest.approx(2.0 / 3.0)


def test_latency_profiler_reports_plausible_values():
    profiler = LatencyProfiler()
    for _ in range(20):
        with profiler.timed():
            time.sleep(0.001)
    summary = profiler.summary()
    assert summary["count"] == 20
    assert summary["mean_ms"] > 0.5
    assert summary["p99_ms"] >= summary["p50_ms"]

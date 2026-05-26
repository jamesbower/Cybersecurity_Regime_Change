"""Evaluation metrics — spec §12."""
from __future__ import annotations
import time
from contextlib import contextmanager
from typing import Iterator
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix


def weighted_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))


def fpr_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Collapse regimes 1..K-1 → 'attack'; compute false-positive rate."""
    yt = (np.asarray(y_true) != 0).astype(int)
    yp = (np.asarray(y_pred) != 0).astype(int)
    if (yt == 0).sum() == 0:
        return 0.0
    cm = confusion_matrix(yt, yp, labels=[0, 1])
    tn, fp = cm[0, 0], cm[0, 1]
    denom = tn + fp
    return float(fp / denom) if denom > 0 else 0.0


def stage_attribution_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Accuracy on attack-only mask: fraction of attack windows with argmax = true regime."""
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    mask = yt != 0
    if mask.sum() == 0:
        return 0.0
    return float((yp[mask] == yt[mask]).mean())


class LatencyProfiler:
    def __init__(self) -> None:
        self.records: list[float] = []

    @contextmanager
    def timed(self) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.records.append((time.perf_counter() - start) * 1000.0)

    def add(self, ms: float) -> None:
        self.records.append(ms)

    def summary(self) -> dict:
        if not self.records:
            return {"count": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p99_ms": 0.0}
        arr = np.array(self.records)
        return {
            "count": int(arr.size),
            "mean_ms": float(arr.mean()),
            "p50_ms": float(np.percentile(arr, 50)),
            "p99_ms": float(np.percentile(arr, 99)),
        }

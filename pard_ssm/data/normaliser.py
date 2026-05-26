"""Welford's online mean/variance algorithm — spec §6.3."""
from __future__ import annotations
import numpy as np


class WelfordNormaliser:
    def __init__(self, n_features: int) -> None:
        self.n_features = n_features
        self.n = 0
        self.mean = np.zeros(n_features, dtype=np.float64)
        self.M2 = np.zeros(n_features, dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.M2 += delta * (x - self.mean)

    def fit(self, X: np.ndarray) -> "WelfordNormaliser":
        for row in X:
            self.update(row)
        return self

    @property
    def variance(self) -> np.ndarray:
        return self.M2 / max(self.n - 1, 1)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / (np.sqrt(self.variance) + 1e-8)

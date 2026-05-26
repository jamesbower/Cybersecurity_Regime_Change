import numpy as np
import pandas as pd
import pytest
from pard_ssm.features.feov import compute_feov, shannon_entropy


def test_shannon_entropy_uniform_two_states_is_one_bit_in_nats():
    h = shannon_entropy([1, 1])
    assert h == pytest.approx(np.log(2), abs=1e-10)


def test_shannon_entropy_concentrated_is_zero():
    assert shannon_entropy([10]) == pytest.approx(0.0)
    assert shannon_entropy([0, 0, 5, 0]) == pytest.approx(0.0)


def test_compute_feov_returns_17_dims_per_window():
    # 5 flows within a single 1-second window.
    df = pd.DataFrame({
        "Timestamp": pd.to_datetime(["2024-01-01 00:00:00.1"] * 5),
        "Flow IAT Mean": [100.0, 200.0, 150.0, 120.0, 180.0],
        "Flow IAT Std": [10.0, 20.0, 15.0, 12.0, 18.0],
        "Total Length of Fwd Packets": [500, 600, 400, 700, 550],
        "Total Length of Bwd Packets": [300, 400, 200, 500, 350],
        "Total Fwd Packets": [10, 12, 8, 14, 11],
        "Total Backward Packets": [5, 6, 4, 7, 5],
        "Flow Duration": [1_000_000, 1_200_000, 800_000, 1_500_000, 1_100_000],
        "Destination Port": [80, 80, 443, 53, 22],
        "Protocol": [6, 6, 17, 17, 1],
        "SYN Flag Count": [1, 1, 0, 0, 1],
        "ACK Flag Count": [5, 6, 4, 3, 5],
        "Packet Length Mean": [50.0, 55.0, 45.0, 60.0, 52.0],
        "Packet Length Variance": [10.0, 12.0, 8.0, 15.0, 11.0],
        "Fwd Header Length": [40, 48, 32, 56, 44],
        "RST Flag Count": [0, 0, 1, 0, 0],
    })
    Y = compute_feov(df, window_s=1.0, overlap_s=0.5)
    assert Y.ndim == 2
    assert Y.shape[1] == 17
    assert np.all(np.isfinite(Y))


def test_compute_feov_zero_flow_window_emits_zero_vector():
    # Two flows 10 seconds apart - middle windows will be empty.
    df = pd.DataFrame({
        "Timestamp": pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 00:00:10"]),
        "Flow IAT Mean": [100.0, 100.0],
        "Flow IAT Std": [10.0, 10.0],
        "Total Length of Fwd Packets": [100, 100],
        "Total Length of Bwd Packets": [100, 100],
        "Total Fwd Packets": [1, 1],
        "Total Backward Packets": [1, 1],
        "Flow Duration": [1_000_000, 1_000_000],
        "Destination Port": [80, 80],
        "Protocol": [6, 6],
        "SYN Flag Count": [1, 1],
        "ACK Flag Count": [1, 1],
        "Packet Length Mean": [50.0, 50.0],
        "Packet Length Variance": [10.0, 10.0],
        "Fwd Header Length": [40, 40],
        "RST Flag Count": [0, 0],
    })
    Y = compute_feov(df, window_s=1.0, overlap_s=0.0)
    # Some windows should have all-zero rows (empty windows). All rows must be finite.
    assert Y.shape[1] == 17
    assert np.all(np.isfinite(Y))
    assert (Y.sum(axis=1) == 0).any(), "Expected at least one empty window -> zero vector"

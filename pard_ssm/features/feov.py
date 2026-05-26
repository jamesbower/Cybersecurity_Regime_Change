"""FEOV - 17-dim observation vector for CICIDS2017 - spec §6.

Proxy notes (spec §15):
- Dim 9 (H_payload): proxied by entropy of Packet Length Mean values per window.
- Dim 13 (F_rate): proxied by Fwd Header Length / Total Fwd Packets.
- Dim 15 (DNS_QR): proxied by fraction of flows with Destination Port == 53; DGA score = 0.
- Dim 17 (H_http): proxied by entropy of (port==80 vs port==443) flow counts.
"""
from __future__ import annotations
from collections.abc import Iterable
import logging
import numpy as np
import pandas as pd

EPS = 1e-8
log = logging.getLogger(__name__)


def shannon_entropy(counts: Iterable[float]) -> float:
    arr = np.asarray(list(counts), dtype=np.float64)
    total = arr.sum()
    if total <= 0:
        return 0.0
    p = arr / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def _window_feov(flows: pd.DataFrame) -> np.ndarray:
    if len(flows) == 0:
        return np.zeros(17, dtype=np.float64)

    fwd_bytes = flows["Total Length of Fwd Packets"].sum()
    bwd_bytes = flows["Total Length of Bwd Packets"].sum()
    fwd_pkts = flows["Total Fwd Packets"].sum()
    bwd_pkts = flows["Total Backward Packets"].sum()
    total_bytes = fwd_bytes + bwd_bytes
    total_pkts = fwd_pkts + bwd_pkts

    # Group 1: temporal / IAT
    mu_iat = flows["Flow IAT Mean"].mean()
    sigma2_iat = (flows["Flow IAT Std"] ** 2).mean()
    bpp = total_bytes / (total_pkts + EPS)
    delta_t = flows["Flow Duration"].mean() / 1e6  # us -> s

    # Group 2: protocol / port diversity
    port_counts = flows["Destination Port"].value_counts().values
    h_dport = shannon_entropy(port_counts)
    n_dip = float(len(port_counts))  # spec §6.1 uses distinct dst IPs; CICFlowMeter lacks dst IP column -> use distinct port count as a proxy
    proto_counts = flows["Protocol"].value_counts().values
    h_proto = shannon_entropy(proto_counts)
    syn = flows["SYN Flag Count"].sum()
    ack = flows["ACK Flag Count"].sum()
    r_syn = syn / (ack + EPS)

    # Group 3: payload (with documented proxies)
    h_payload = shannon_entropy(np.round(flows["Packet Length Mean"]).astype(int).value_counts().values)
    sigma2_plen = flows["Packet Length Variance"].mean()
    r_bytes = bwd_bytes / (fwd_bytes + EPS)
    r_pkts = bwd_pkts / (fwd_pkts + EPS)
    f_rate = flows["Fwd Header Length"].sum() / (fwd_pkts + EPS)

    # Group 4: stateful / behavioural
    rst = flows["RST Flag Count"].sum()
    cfr = rst / (fwd_pkts + EPS)
    dns_count = (flows["Destination Port"] == 53).sum()
    dns_qr = float(dns_count / len(flows))
    icmp_count = (flows["Protocol"] == 1).sum()
    icmp_rate = float(icmp_count / max(delta_t, EPS))
    http_mask = flows["Destination Port"].isin([80, 443])
    h_http = shannon_entropy(flows.loc[http_mask, "Destination Port"].value_counts().values)

    return np.array([
        mu_iat, sigma2_iat, bpp, delta_t,
        h_dport, n_dip, h_proto, r_syn,
        h_payload, sigma2_plen, r_bytes, r_pkts, f_rate,
        cfr, dns_qr, icmp_rate, h_http,
    ], dtype=np.float64)


def compute_feov(flow_df: pd.DataFrame, window_s: float = 1.0, overlap_s: float = 0.5) -> np.ndarray:
    """Bucket flows into sliding windows, compute 17-dim FEOV per window."""
    if flow_df.empty:
        return np.zeros((0, 17), dtype=np.float64)

    df = flow_df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)

    t0 = df["Timestamp"].iloc[0]
    t_end = df["Timestamp"].iloc[-1]
    stride = max(window_s - overlap_s, 1e-3)
    n_windows = max(int(np.ceil(((t_end - t0).total_seconds() + window_s) / stride)), 1)

    starts = [t0 + pd.Timedelta(seconds=i * stride) for i in range(n_windows)]
    Y = np.zeros((n_windows, 17), dtype=np.float64)
    for i, ws in enumerate(starts):
        we = ws + pd.Timedelta(seconds=window_s)
        mask = (df["Timestamp"] >= ws) & (df["Timestamp"] < we)
        chunk = df.loc[mask]
        if len(chunk) == 0:
            log.debug("FEOV window %d empty (t=%s)", i, ws)
        Y[i] = _window_feov(chunk)
    return Y

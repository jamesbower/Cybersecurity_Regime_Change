#!/usr/bin/env python3
"""Warm-start + online inference + metrics + result dump."""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
from dataclasses import asdict
import numpy as np
import pandas as pd
from pard_ssm.config import load_config
from pard_ssm.data.normaliser import WelfordNormaliser
from pard_ssm.model.pard_ssm import PARDSSM
from pard_ssm.evaluation.metrics import (
    weighted_f1, fpr_binary, stage_attribution_accuracy, LatencyProfiler,
)


def _load_split(processed_dir: Path, dataset: str, split: str) -> pd.DataFrame:
    p = processed_dir / f"{dataset}_{split}.parquet"
    if not p.exists():
        raise FileNotFoundError(
            f"Missing {p}. Run scripts/preprocess_{dataset}.py first."
        )
    return pd.read_parquet(p)


def _to_Y(df: pd.DataFrame) -> np.ndarray:
    y_cols = sorted(
        [c for c in df.columns if c.startswith("y_")],
        key=lambda c: int(c.split("_")[1]),
    )
    return df[y_cols].to_numpy(dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["cicids2017", "unsw_nb15"])
    parser.add_argument("--config", type=Path, default=Path("config/pard_ssm_default.yaml"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--run-id", type=str, default="run1")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("run_experiment")

    cfg = load_config(args.config)
    out_dir = args.results_dir / f"{args.dataset}_{args.run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = _load_split(args.processed_dir, args.dataset, "train")
    test_df = _load_split(
        args.processed_dir, args.dataset,
        "test" if (args.processed_dir / f"{args.dataset}_test.parquet").exists() else "val",
    )

    Y_train = _to_Y(train_df)
    Y_test = _to_Y(test_df)
    labels_train = train_df["label"].to_numpy().astype(int)
    labels_test = test_df["label"].to_numpy().astype(int)

    # CICIDS uses 17-dim FEOV. UNSW PCA may be < 17 in smoke (n_components arg);
    # PARDSSM expects cfg.model.m features → pad if needed.
    m_expected = cfg.model.m
    if Y_train.shape[1] < m_expected:
        pad_cols = m_expected - Y_train.shape[1]
        Y_train = np.hstack([Y_train, np.zeros((Y_train.shape[0], pad_cols))])
        Y_test = np.hstack([Y_test, np.zeros((Y_test.shape[0], pad_cols))])

    # Normalise.
    norm = WelfordNormaliser(n_features=m_expected).fit(Y_train)
    Y_train_n = norm.transform(Y_train)
    Y_test_n = norm.transform(Y_test)

    model = PARDSSM(cfg)
    log.info("warm-starting on %d training windows", len(Y_train_n))
    model.warm_start(Y_train_n, labels_train)

    profiler = LatencyProfiler()
    gammas = np.zeros((len(Y_test_n), cfg.model.K))
    preds = np.zeros(len(Y_test_n), dtype=int)
    alerts = []
    for t, y in enumerate(Y_test_n):
        res = model.infer_window(y, t=float(t))
        profiler.add(res.latency_ms)
        gammas[t] = res.gamma
        preds[t] = int(np.argmax(res.gamma))
        if res.alert is not None:
            a = asdict(res.alert)
            a["stage_posterior"] = a["stage_posterior"].tolist()
            a["predicted_posterior"] = a["predicted_posterior"].tolist()
            alerts.append(a)

    metrics = {
        "f1": weighted_f1(labels_test, preds),
        "fpr": fpr_binary(labels_test, preds),
        "saa": stage_attribution_accuracy(labels_test, preds),
        "latency": profiler.summary(),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    # gamma parquet
    gamma_cols = {f"gamma_{k}": gammas[:, k] for k in range(cfg.model.K)}
    gamma_cols["pred"] = preds
    gamma_cols["true"] = labels_test
    pd.DataFrame(gamma_cols).to_parquet(out_dir / "gamma.parquet", index=False)
    # alerts jsonl
    with open(out_dir / "alerts.jsonl", "w") as fh:
        for a in alerts:
            fh.write(json.dumps(a) + "\n")

    log.info("done. metrics=%s", metrics)
    if metrics["latency"]["mean_ms"] > 1.2:
        log.warning("mean latency %.3f ms exceeds spec target of 1.2 ms",
                    metrics["latency"]["mean_ms"])


if __name__ == "__main__":
    main()

# PARD-SSM (Phase 1 core)

Python 3.12 reimplementation of the core model from Hiremath et al.,
*"PARD-SSM: Probabilistic Cyber-Attack Regime Detection via Variational Switching
State-Space Models"* (arXiv:2604.02299v1, April 2026).

**Scope (this implementation):** scaffold + data loaders + FEOV + PARD-SSM core
(KalmanBank / VSIPC / batch-EM warm-start / OEMPU / PKAR) + minimal evaluation
(weighted F₁, FPR, SAA, inference latency).

**Deferred:** Snort / BiLSTM / Isolation Forest / KF-Anomaly baselines, EDM
metric, learned-Π̂ reporting, ablation runner, Phase 2 extensions.

**Deviation from paper spec:** Python 3.12 (paper specifies 3.11). All pinned
dependency versions in `requirements.txt` support 3.12.

## Install

```bash
python3.12 -m venv .venv-pard-ssm
source .venv-pard-ssm/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Run tests

```bash
pytest -v -m "not slow"
```

## Reproducing paper results

### 1. Place raw data

CICIDS2017 (8 day-specific CSVs from CICFlowMeter):
```
data/raw/cicids2017/MachineLearningCSV/Monday-WorkingHours.pcap_ISCX.csv
data/raw/cicids2017/MachineLearningCSV/Tuesday-WorkingHours.pcap_ISCX.csv
... (8 files total per spec §5.1)
```

UNSW-NB15 (pre-split CSVs):
```
data/raw/unsw_nb15/UNSW_NB15_training-set.csv
data/raw/unsw_nb15/UNSW_NB15_testing-set.csv
```

### 2. Preprocess (one-shot per dataset)

```bash
python scripts/preprocess_cicids2017.py
python scripts/preprocess_unsw_nb15.py
```

Outputs land in `data/processed/`.

### 3. Run experiments

```bash
python scripts/run_experiment.py --dataset cicids2017 --run-id paper_repro
python scripts/run_experiment.py --dataset unsw_nb15  --run-id paper_repro
```

Results land in `results/<dataset>_paper_repro/`:
- `metrics.json` — F₁, FPR, SAA, latency summary
- `gamma.parquet` — per-window regime posteriors + predictions
- `alerts.jsonl` — KL-gated KillChainAlert records

### 4. Validation targets (spec §14)

| Metric | CICIDS2017 target | UNSW-NB15 target | Tolerance |
|---|---|---|---|
| F₁ (weighted) | 0.982 | 0.971 | ± 0.010 |
| SAA | 0.861 | 0.834 | ± 0.015 |
| FPR | "Low" (< 0.05) | "Low" | qualitative |
| Latency (mean) | < 1.2 ms | < 1.2 ms | hard ceiling |

If the latency target is exceeded, the runner logs a `WARNING` but does not fail.

## Layout

See `docs/superpowers/specs/2026-05-26-pard-ssm-phase1-core-design.md` for the
architectural design and `docs/superpowers/plans/2026-05-26-pard-ssm-phase1-core.md`
for the task-by-task implementation plan (both files are gitignored).

## Known limitations

- FEOV dims 9, 13, 15, 17 use documented proxies (spec §15) since exact
  computation requires raw PCAPs not present in CICFlowMeter CSVs. Expect up
  to ± 0.010 F₁ drift from paper numbers.
- UNSW-NB15 PCA uses all numerical + one-hot-encoded categorical features
  (spec §15 doesn't specify which to exclude).
- Mode-collapse detection logs a warning but does not auto-restart in this
  implementation slice.

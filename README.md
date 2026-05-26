# PARD-SSM (Phase 1, core slice)

Faithful Python 3.12 reimplementation of the core model from Hiremath et al.,
*"PARD-SSM: Probabilistic Cyber-Attack Regime Detection via Variational Switching
State-Space Models"* (arXiv:2604.02299v1, April 2026).

**Scope (this slice):** data loaders, FEOV features, PARD-SSM model
(Kalman / VSIPC / batch-EM warm-start / OEMPU / PKAR), minimal evaluation.
Baselines, report generation, and dataset download scripts are deferred.

**Deviation from spec:** Python 3.12 instead of 3.11 (per repo owner). Pinned dep
versions in `requirements.txt` all support 3.12.

## Quickstart

```bash
python3.12 -m venv .venv-pard-ssm
source .venv-pard-ssm/bin/activate
pip install -r requirements.txt
pip install -e .
pytest -v -m "not slow"
```

## Data placement

The user places raw datasets manually under `data/raw/`:

- `data/raw/cicids2017/MachineLearningCSV/*.csv` — 8 day-specific CICFlowMeter CSVs
- `data/raw/unsw_nb15/UNSW_NB15_training-set.csv`
- `data/raw/unsw_nb15/UNSW_NB15_testing-set.csv`

Then run preprocessing once:

```bash
python scripts/preprocess_cicids2017.py
python scripts/preprocess_unsw_nb15.py
```

## Running an experiment

```bash
python scripts/run_experiment.py --dataset cicids2017
python scripts/run_experiment.py --dataset unsw_nb15
```

Results go to `results/<dataset>_<runid>/`.

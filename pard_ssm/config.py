"""OmegaConf-backed config loader with dataclass validation."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from omegaconf import OmegaConf


@dataclass
class ModelCfg:
    K: int
    n: int
    m: int
    k_max: int
    epsilon: float


@dataclass
class OnlineEMCfg:
    eta: float


@dataclass
class KLGatingCfg:
    tau_kl: float


@dataclass
class WindowCfg:
    W: float
    overlap_s: float


@dataclass
class BatchEMCfg:
    max_iters: int
    subsample_frac: float
    random_state: int


@dataclass
class ModeCollapseCfg:
    window_threshold: int


@dataclass
class LoggingCfg:
    level: str


@dataclass
class DatasetCfg:
    name: str = ""
    raw_dir: str = ""
    processed_dir: str = ""
    split_ratios: list = field(default_factory=lambda: [0.6, 0.2, 0.2])
    baseline_hours: float = 1.0
    pca_components: int = 17
    pca_random_state: int = 42


@dataclass
class PARDSSMConfig:
    model: ModelCfg
    online_em: OnlineEMCfg
    kl_gating: KLGatingCfg
    window: WindowCfg
    batch_em: BatchEMCfg
    mode_collapse: ModeCollapseCfg
    logging: LoggingCfg
    dataset: DatasetCfg = field(default_factory=DatasetCfg)


def _validate(cfg: PARDSSMConfig) -> None:
    if cfg.model.K <= 0:
        raise ValueError(f"model.K must be positive, got {cfg.model.K}")
    if cfg.model.n <= 0:
        raise ValueError(f"model.n must be positive, got {cfg.model.n}")
    if cfg.model.m <= 0:
        raise ValueError(f"model.m must be positive, got {cfg.model.m}")
    if cfg.model.k_max <= 0:
        raise ValueError(f"model.k_max must be positive, got {cfg.model.k_max}")
    if cfg.model.epsilon < 0:
        raise ValueError(f"model.epsilon must be non-negative, got {cfg.model.epsilon}")
    if cfg.online_em.eta < 0:
        raise ValueError(f"online_em.eta must be non-negative, got {cfg.online_em.eta}")
    if cfg.kl_gating.tau_kl < 0:
        raise ValueError(f"kl_gating.tau_kl must be non-negative, got {cfg.kl_gating.tau_kl}")


def load_config(default_path: Path, dataset_path: Optional[Path] = None) -> PARDSSMConfig:
    base = OmegaConf.load(default_path)
    if dataset_path is not None:
        override = OmegaConf.load(dataset_path)
        base = OmegaConf.merge(base, override)
    schema = OmegaConf.structured(PARDSSMConfig)
    merged = OmegaConf.merge(schema, base)
    cfg: PARDSSMConfig = OmegaConf.to_object(merged)  # type: ignore[assignment]
    _validate(cfg)
    return cfg

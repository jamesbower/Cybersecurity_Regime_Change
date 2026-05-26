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
    split_ratios: list[float] = field(default_factory=lambda: [0.6, 0.2, 0.2])
    baseline_hours: Optional[float] = None
    pca_components: Optional[int] = None
    pca_random_state: Optional[int] = None


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


_VALID_LOG_LEVELS = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}


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
    if cfg.window.W <= 0:
        raise ValueError(f"window.W must be positive, got {cfg.window.W}")
    if cfg.window.overlap_s < 0:
        raise ValueError(
            f"window.overlap_s must be non-negative, got {cfg.window.overlap_s}"
        )
    if cfg.window.overlap_s >= cfg.window.W:
        raise ValueError(
            f"window.overlap_s ({cfg.window.overlap_s}) must be less than "
            f"window.W ({cfg.window.W})"
        )
    if cfg.batch_em.max_iters <= 0:
        raise ValueError(
            f"batch_em.max_iters must be positive, got {cfg.batch_em.max_iters}"
        )
    if not (0.0 < cfg.batch_em.subsample_frac <= 1.0):
        raise ValueError(
            f"batch_em.subsample_frac must be in (0.0, 1.0], got "
            f"{cfg.batch_em.subsample_frac}"
        )
    if cfg.mode_collapse.window_threshold <= 0:
        raise ValueError(
            f"mode_collapse.window_threshold must be positive, got "
            f"{cfg.mode_collapse.window_threshold}"
        )
    if cfg.logging.level not in _VALID_LOG_LEVELS:
        raise ValueError(
            f"logging.level must be one of {sorted(_VALID_LOG_LEVELS)}, got "
            f"{cfg.logging.level!r}"
        )


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

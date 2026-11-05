from dataclasses import asdict, dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class DataConfig:
    start_year: int = 1992
    end_year: int = 2025
    min_brent_observations: int = 200


@dataclass(frozen=True)
class ModelConfig:
    min_observations: int = 15
    hac_lags: int = 1
    hac_small_sample_correction: bool = True
    diagnostic_lags: int = 1
    adf_maxlag: int = 2
    coint_maxlag: int = 1
    coint_sensitivity_lags: tuple[int, ...] = (0, 1, 2)
    sensitivity_start_years: tuple[int, ...] = (1993, 1995, 2000)
    sensitivity_hac_lags: tuple[int, ...] = (0, 1, 2)
    reference_model: str = "ar1_oil"


@dataclass(frozen=True)
class Config:
    data: DataConfig
    model: ModelConfig

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: Path) -> Config:
    with path.open("rb") as stream:
        raw = tomllib.load(stream)
    config = Config(DataConfig(**raw["data"]), ModelConfig(**raw["model"]))
    if config.data.start_year > config.data.end_year:
        raise ValueError("start_year must not exceed end_year")
    if config.data.min_brent_observations < 1:
        raise ValueError("min_brent_observations must be positive")
    if config.model.min_observations < 10:
        raise ValueError("At least 10 observations must be required")
    lags = [
        config.model.hac_lags,
        config.model.adf_maxlag,
        config.model.coint_maxlag,
        *config.model.coint_sensitivity_lags,
        *config.model.sensitivity_hac_lags,
    ]
    if any(lag < 0 for lag in lags) or config.model.diagnostic_lags < 1:
        raise ValueError("Invalid lag configuration")
    return config

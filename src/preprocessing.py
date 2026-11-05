from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DataConfig
from src.data_loader import filter_dates, validate_fred, validate_ppp


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Construct differences only inside a sorted, contiguous yearly sample."""
    df = frame[["year", "brent", "nominal_usd_rub", "ppp_basket_usd_rub"]].copy()
    df = df.sort_values("year").reset_index(drop=True)
    if df["year"].duplicated().any() or (df["year"].diff().dropna() != 1).any():
        raise ValueError("Years must be unique and contiguous; cannot difference across gaps")
    values = df[["brent", "nominal_usd_rub", "ppp_basket_usd_rub"]]
    if not np.isfinite(values.to_numpy()).all() or (values <= 0).any().any():
        raise ValueError("Input levels must be finite and positive")
    df["basket_real_usd_rub"] = df["nominal_usd_rub"] / df["ppp_basket_usd_rub"]
    for variable in ["brent", "basket_real_usd_rub"]:
        df[f"log_{variable}"] = np.log(df[variable])
        df[f"dlog_{variable}"] = df[f"log_{variable}"].diff()
        df[f"dlog_{variable}_lag1"] = df[f"dlog_{variable}"].shift(1)
    df["dummy_1998_crisis"] = (df["year"] == 1998).astype(int)
    df["dummy_2014_2015_shock"] = df["year"].isin([2014, 2015]).astype(int)
    df["dummy_2022_plus"] = (df["year"] >= 2022).astype(int)
    return df


def _yearly(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df = frame.assign(year=frame["date"].dt.year, month=frame["date"].dt.month)
    return df.groupby("year").agg(
        **{
            prefix: ("value", "mean"),
            f"{prefix}_observations": ("value", "size"),
            f"{prefix}_months": ("month", "nunique"),
        }
    )


def build_final_dataset(
    brent_daily: pd.DataFrame,
    usd_rub_monthly: pd.DataFrame,
    ppp_yearly: pd.DataFrame,
    processed_dir: Path,
    config: DataConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = config or DataConfig()
    start, end = f"{config.start_year}-01-01", f"{config.end_year}-12-31"
    brent = filter_dates(validate_fred(brent_daily), start, end)
    usd = filter_dates(validate_fred(usd_rub_monthly), start, end)
    if usd["date"].dt.to_period("M").duplicated().any():
        raise ValueError("USD/RUB must have at most one observation per month")
    ppp = validate_ppp(ppp_yearly).set_index("year")
    calendar = pd.DataFrame(index=pd.Index(range(config.start_year, config.end_year + 1), name="year"))
    coverage = calendar.join(_yearly(brent, "brent")).join(_yearly(usd, "nominal_usd_rub")).join(ppp)
    coverage["brent_complete"] = coverage["brent_months"].eq(12) & coverage["brent_observations"].ge(
        config.min_brent_observations
    )
    coverage["fx_complete"] = coverage["nominal_usd_rub_months"].eq(12) & coverage[
        "nominal_usd_rub_observations"
    ].eq(12)
    coverage["ppp_available"] = coverage["ppp_basket_usd_rub"].notna()
    coverage["included"] = coverage[["brent_complete", "fx_complete", "ppp_available"]].all(axis=1)
    coverage["exclusion_reason"] = coverage.apply(
        lambda row: "; ".join(
            text
            for field, text in [
                ("brent_complete", "incomplete_brent"),
                ("fx_complete", "incomplete_fx"),
                ("ppp_available", "missing_ppp"),
            ]
            if not row[field]
        ),
        axis=1,
    )
    coverage = coverage.reset_index()
    processed_dir.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(processed_dir / "data_coverage.csv", index=False)
    selected = coverage.loc[coverage["included"]]
    if selected.empty:
        raise ValueError("No complete years in requested range; see data_coverage.csv")
    df = add_features(selected)
    df.to_csv(processed_dir / "final_dataset.csv", index=False)
    return df, coverage

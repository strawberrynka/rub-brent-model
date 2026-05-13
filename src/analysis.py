from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_VARS = [
    "brent",
    "nominal_usd_rub",
    "ppp_basket_usd_rub",
    "basket_real_usd_rub",
    "log_brent",
    "log_basket_real_usd_rub",
    "dlog_brent",
    "dlog_basket_real_usd_rub",
    "dlog_brent_lag1",
]


def save_descriptive_statistics(df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    stats = df[BASE_VARS].describe().T
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(out_path)
    return stats


def save_correlation_matrix(df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    corr = df[BASE_VARS].corr()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    corr.to_csv(out_path)
    return corr

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def aggregate_to_yearly_mean(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    out = df.copy()
    out["year"] = pd.to_datetime(out["date"]).dt.year
    out = out.groupby("year", as_index=False)[value_col].mean()
    return out


def build_final_dataset(
    brent_daily: pd.DataFrame,
    usd_rub_monthly: pd.DataFrame,
    ppp_yearly: pd.DataFrame,
    processed_dir: Path,
) -> pd.DataFrame:
    brent_yearly = aggregate_to_yearly_mean(brent_daily, "value").rename(
        columns={"value": "brent"}
    )
    usd_yearly = aggregate_to_yearly_mean(usd_rub_monthly, "value").rename(
        columns={"value": "nominal_usd_rub"}
    )

    df = brent_yearly.merge(usd_yearly, on="year", how="inner").merge(
        ppp_yearly[["year", "ppp_basket_usd_rub"]], on="year", how="inner"
    )

    df = df.dropna(subset=["brent", "nominal_usd_rub", "ppp_basket_usd_rub"]).copy()
    df["basket_real_usd_rub"] = df["nominal_usd_rub"] / df["ppp_basket_usd_rub"]

    df["log_brent"] = np.log(df["brent"])
    df["log_basket_real_usd_rub"] = np.log(df["basket_real_usd_rub"])
    df["dlog_brent"] = df["log_brent"].diff()
    df["dlog_basket_real_usd_rub"] = df["log_basket_real_usd_rub"].diff()
    df["dlog_brent_lag1"] = df["dlog_brent"].shift(1)

    df = df.sort_values("year").reset_index(drop=True)

    processed_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_dir / "final_dataset.csv", index=False)
    return df

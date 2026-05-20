from __future__ import annotations

from pathlib import Path

import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint


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


def run_adf_tests(df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    variables = ["log_brent", "log_basket_real_usd_rub", "dlog_brent", "dlog_basket_real_usd_rub"]
    results = []
    
    for var in variables:
        series = df[var].dropna()
        if len(series) > 0:
            adf_stat, p_value, lags, nobs, crit_values, icbest = adfuller(series)
            results.append({
                "variable": var,
                "adf_stat": adf_stat,
                "p_value": p_value,
                "lags": lags,
                "n_obs": nobs,
                "crit_1%": crit_values["1%"],
                "crit_5%": crit_values["5%"],
                "crit_10%": crit_values["10%"]
            })
            
    res_df = pd.DataFrame(results)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(out_path, index=False)
    return res_df


def run_cointegration_test(df: pd.DataFrame, out_path: Path) -> dict:
    sub_df = df[["log_brent", "log_basket_real_usd_rub"]].dropna()
    coint_t, p_value, crit_value = coint(sub_df["log_brent"], sub_df["log_basket_real_usd_rub"])
    
    result = {
        "coint_t": coint_t,
        "p_value": p_value,
        "crit_1%": crit_value[0],
        "crit_5%": crit_value[1],
        "crit_10%": crit_value[2],
    }
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_csv(out_path, index=False)
    return result

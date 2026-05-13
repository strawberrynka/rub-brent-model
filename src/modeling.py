from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson


@dataclass
class ModelBundle:
    name: str
    equation: str
    result: sm.regression.linear_model.RegressionResultsWrapper


def _fit_ols(df: pd.DataFrame, y_col: str, x_cols: List[str]) -> sm.regression.linear_model.RegressionResultsWrapper:
    model_df = df[[y_col] + x_cols].dropna().copy()
    y = model_df[y_col]
    x = sm.add_constant(model_df[x_cols], has_constant="add")
    model = sm.OLS(y, x)
    return model.fit()


def run_models(df: pd.DataFrame) -> Dict[str, ModelBundle]:
    m1 = _fit_ols(df, "log_basket_real_usd_rub", ["log_brent"])
    m2 = _fit_ols(df, "dlog_basket_real_usd_rub", ["dlog_brent"])
    m3 = _fit_ols(df, "dlog_basket_real_usd_rub", ["dlog_brent", "dlog_brent_lag1"])

    return {
        "model_1": ModelBundle(
            name="Модель 1",
            equation="log_basket_real_usd_rub = const + beta * log_brent",
            result=m1,
        ),
        "model_2": ModelBundle(
            name="Модель 2",
            equation="dlog_basket_real_usd_rub = const + beta * dlog_brent",
            result=m2,
        ),
        "model_3": ModelBundle(
            name="Модель 3",
            equation=(
                "dlog_basket_real_usd_rub = const + beta1 * dlog_brent + beta2 * dlog_brent_lag1"
            ),
            result=m3,
        ),
    }


def save_regression_results(models: Dict[str, ModelBundle], out_path: Path) -> pd.DataFrame:
    rows: List[dict] = []
    for model_key, bundle in models.items():
        res = bundle.result
        for param in res.params.index:
            rows.append(
                {
                    "model": model_key,
                    "equation": bundle.equation,
                    "parameter": param,
                    "coef": res.params[param],
                    "std_err": res.bse[param],
                    "t_stat": res.tvalues[param],
                    "p_value": res.pvalues[param],
                    "r_squared": res.rsquared,
                    "adj_r_squared": res.rsquared_adj,
                    "aic": res.aic,
                    "bic": res.bic,
                    "n_obs": int(res.nobs),
                    "durbin_watson": durbin_watson(res.resid),
                }
            )

    out = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out


def save_model_comparison(models: Dict[str, ModelBundle], out_path: Path) -> pd.DataFrame:
    rows = []
    for model_key, bundle in models.items():
        res = bundle.result
        rows.append(
            {
                "model": model_key,
                "equation": bundle.equation,
                "r_squared": res.rsquared,
                "adj_r_squared": res.rsquared_adj,
                "aic": res.aic,
                "bic": res.bic,
                "n_obs": int(res.nobs),
                "durbin_watson": durbin_watson(res.resid),
            }
        )

    out = pd.DataFrame(rows).sort_values("aic")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out


def extract_residuals_for_plot(models: Dict[str, ModelBundle]) -> Tuple[pd.Series, str]:
    bundle = models["model_3"]
    return bundle.result.resid, bundle.name

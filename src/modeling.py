from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss


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


def _hac_stats_by_param(
    res: sm.regression.linear_model.RegressionResultsWrapper,
    maxlags: int = 1,
) -> Dict[str, Dict[str, float]]:
    robust = res.get_robustcov_results(cov_type="HAC", maxlags=maxlags)
    return {
        param: {
            "hac_std_err": float(robust.bse[idx]),
            "hac_t_stat": float(robust.tvalues[idx]),
            "hac_p_value": float(robust.pvalues[idx]),
        }
        for idx, param in enumerate(res.model.exog_names)
    }


def _ljung_box_pvalue_lag1(res: sm.regression.linear_model.RegressionResultsWrapper) -> float:
    lb = acorr_ljungbox(res.resid, lags=[1], return_df=True)
    return float(lb["lb_pvalue"].iloc[0])


def run_models(df: pd.DataFrame) -> Dict[str, ModelBundle]:
    m1 = _fit_ols(df, "log_basket_real_usd_rub", ["log_brent"])
    m2 = _fit_ols(df, "dlog_basket_real_usd_rub", ["dlog_brent"])
    m3 = _fit_ols(df, "dlog_basket_real_usd_rub", ["dlog_brent", "dlog_brent_lag1"])
    m4 = _fit_ols(
        df,
        "dlog_basket_real_usd_rub",
        ["dlog_brent", "dlog_brent_lag1", "dlog_basket_real_usd_rub_lag1"],
    )
    m5 = _fit_ols(
        df,
        "dlog_basket_real_usd_rub",
        [
            "dlog_brent",
            "dlog_brent_lag1",
            "dlog_basket_real_usd_rub_lag1",
            "dummy_1998_crisis",
            "dummy_2014_2015_shock",
            "dummy_2022_plus",
        ],
    )
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
        "model_4": ModelBundle(
            name="Модель 4",
            equation=(
                "dlog_basket_real_usd_rub = const + beta1 * dlog_brent + "
                "beta2 * dlog_brent_lag1 + phi * dlog_basket_real_usd_rub_lag1"
            ),
            result=m4,
        ),
        "model_5": ModelBundle(
            name="Модель 5",
            equation=(
                "dlog_basket_real_usd_rub = const + beta1 * dlog_brent + "
                "beta2 * dlog_brent_lag1 + phi * dlog_basket_real_usd_rub_lag1 + crisis_dummies"
            ),
            result=m5,
        ),
    }


def save_regression_results(models: Dict[str, ModelBundle], out_path: Path) -> pd.DataFrame:
    rows: List[dict] = []
    for model_key, bundle in models.items():
        res = bundle.result
        hac_stats = _hac_stats_by_param(res)
        ljung_box_pvalue_lag1 = _ljung_box_pvalue_lag1(res)
        for param in res.params.index:
            param_hac = hac_stats[param]
            rows.append(
                {
                    "model": model_key,
                    "equation": bundle.equation,
                    "parameter": param,
                    "coef": res.params[param],
                    "std_err": res.bse[param],
                    "t_stat": res.tvalues[param],
                    "p_value": res.pvalues[param],
                    "hac_std_err": param_hac["hac_std_err"],
                    "hac_t_stat": param_hac["hac_t_stat"],
                    "hac_p_value": param_hac["hac_p_value"],
                    "r_squared": res.rsquared,
                    "adj_r_squared": res.rsquared_adj,
                    "aic": res.aic,
                    "bic": res.bic,
                    "n_obs": int(res.nobs),
                    "durbin_watson": durbin_watson(res.resid),
                    "ljung_box_pvalue_lag1": ljung_box_pvalue_lag1,
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
                "ljung_box_pvalue_lag1": _ljung_box_pvalue_lag1(res),
            }
        )

    out = pd.DataFrame(rows).sort_values("aic")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out


def extract_residuals_for_plot(models: Dict[str, ModelBundle]) -> Tuple[pd.Series, str]:
    bundle = min(models.values(), key=lambda item: item.result.aic)
    return bundle.result.resid, bundle.name


def _stationarity_row(series: pd.Series, variable: str) -> dict:
    clean = series.dropna()
    row = {
        "variable": variable,
        "n_obs": int(clean.shape[0]),
        "adf_stat": float("nan"),
        "adf_p_value": float("nan"),
        "kpss_stat": float("nan"),
        "kpss_p_value": float("nan"),
    }
    if clean.shape[0] < 8:
        return row

    adf_stat, adf_p_value, *_ = adfuller(clean, autolag="AIC")
    row["adf_stat"] = float(adf_stat)
    row["adf_p_value"] = float(adf_p_value)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InterpolationWarning)
        kpss_stat, kpss_p_value, *_ = kpss(clean, regression="c", nlags="auto")
    row["kpss_stat"] = float(kpss_stat)
    row["kpss_p_value"] = float(kpss_p_value)
    return row


def save_stationarity_tests(df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    variables = [
        "log_brent",
        "log_basket_real_usd_rub",
        "dlog_brent",
        "dlog_basket_real_usd_rub",
    ]
    out = pd.DataFrame([_stationarity_row(df[var], var) for var in variables])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out

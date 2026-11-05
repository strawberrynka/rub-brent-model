"""Small, explicitly specified models; comparisons use identical observations."""

from dataclasses import dataclass, replace
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, acorr_ljungbox
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, coint, kpss

from src.config import ModelConfig
from src.preprocessing import add_features

Y = "dlog_basket_real_usd_rub"
OIL = "dlog_brent"
LAG_Y = Y + "_lag1"
LAG_OIL = OIL + "_lag1"
DUMMIES = ["dummy_1998_crisis", "dummy_2014_2015_shock", "dummy_2022_plus"]
SPECS = {
    "ar1": ("Инерция курса", [LAG_Y]),
    "ar1_oil": ("Инерция + Brent", [LAG_Y, OIL]),
    "ar1_oil_lag": ("Инерция + Brent + лаг Brent", [LAG_Y, OIL, LAG_OIL]),
    "ar1_oil_crises": ("Динамика + кризисные индикаторы", [LAG_Y, OIL, LAG_OIL, *DUMMIES]),
    "changes": ("Только изменение Brent", [OIL]),
    "changes_lag": ("Brent + лаг Brent", [OIL, LAG_OIL]),
}


@dataclass
class ModelBundle:
    name: str
    group: str
    result: object
    omitted: tuple[str, ...] = ()


def _fit(df: pd.DataFrame, y: str, predictors: list[str], config: ModelConfig):
    frame = df.set_index("year")[[y, *predictors]].dropna()
    if len(frame) < config.min_observations:
        raise ValueError(f"Only {len(frame)} observations; at least {config.min_observations} required")
    omitted = tuple(column for column in predictors if column in DUMMIES and frame[column].nunique() < 2)
    predictors = [column for column in predictors if column not in omitted]
    x = sm.add_constant(frame[predictors], has_constant="add")
    if not np.isfinite(x.to_numpy()).all() or not np.isfinite(frame[y]).all():
        raise ValueError("Non-finite regression inputs")
    if np.linalg.matrix_rank(x.to_numpy()) != x.shape[1]:
        raise ValueError("Regression design is rank deficient")
    if len(frame) <= x.shape[1] + 2:
        raise ValueError("Insufficient residual degrees of freedom")
    return sm.OLS(frame[y], x).fit(), omitted


def run_models(df: pd.DataFrame, config: ModelConfig | None = None) -> dict[str, ModelBundle]:
    config = config or ModelConfig()
    common = df.dropna(subset=[Y, OIL, LAG_Y, LAG_OIL])
    models = {}
    for key, (label, predictors) in SPECS.items():
        result, omitted = _fit(common, Y, predictors, config)
        models[key] = ModelBundle(label, "changes", result, omitted)
    result, omitted = _fit(df, "log_basket_real_usd_rub", ["log_brent"], config)
    models["levels"] = ModelBundle("Уровни (исследовательская оценка)", "levels", result, omitted)
    return models


def robust_result(result, config: ModelConfig):
    return result.get_robustcov_results(
        cov_type="HAC",
        maxlags=config.hac_lags,
        use_correction=config.hac_small_sample_correction,
        use_t=True,
    )


def regression_table(models: dict, config: ModelConfig) -> pd.DataFrame:
    rows = []
    for key, bundle in models.items():
        res = bundle.result
        hac = robust_result(res, config)
        intervals = hac.conf_int(alpha=0.05)
        for idx, name in enumerate(res.model.exog_names):
            rows.append(
                {
                    "model": key,
                    "label": bundle.name,
                    "group": bundle.group,
                    "parameter": name,
                    "coef": float(res.params[name]),
                    "std_err": float(res.bse[name]),
                    "p_value": float(res.pvalues[name]),
                    "hac_std_err": float(hac.bse[idx]),
                    "hac_p_value": float(hac.pvalues[idx]),
                    "ci_low": float(intervals[idx, 0]),
                    "ci_high": float(intervals[idx, 1]),
                    "hac_lags": config.hac_lags,
                    "hac_small_sample_correction": config.hac_small_sample_correction,
                    "use_t": True,
                    "n_obs": int(res.nobs),
                    "start_year": int(res.model.data.row_labels[0]),
                    "end_year": int(res.model.data.row_labels[-1]),
                    "omitted_constant_dummies": ";".join(bundle.omitted),
                }
            )
    return pd.DataFrame(rows)


def comparison_table(models: dict, config: ModelConfig) -> pd.DataFrame:
    comparable = {key: bundle for key, bundle in models.items() if bundle.group == "changes"}
    if not comparable:
        raise ValueError("No comparable change models")
    reference = next(iter(comparable.values())).result
    rows = []
    for key, bundle in comparable.items():
        res = bundle.result
        if list(res.model.data.row_labels) != list(
            reference.model.data.row_labels
        ) or not np.array_equal(res.model.endog, reference.model.endog):
            raise ValueError("Comparison requires identical years and response")
        bg = acorr_breusch_godfrey(res, nlags=config.diagnostic_lags, result_object=False)
        lb = acorr_ljungbox(res.resid, lags=[config.diagnostic_lags], return_df=True)
        rows.append(
            {
                "model": key,
                "label": bundle.name,
                "n_obs": int(res.nobs),
                "n_parameters": len(res.params),
                "start_year": int(res.resid.index[0]),
                "end_year": int(res.resid.index[-1]),
                "r_squared": res.rsquared,
                "adj_r_squared": res.rsquared_adj,
                "aic": res.aic,
                "bic": res.bic,
                "bg_lm_p_value": float(bg[1]),
                "bg_f_p_value": float(bg[3]),
                "ljung_box_p_value": float(lb["lb_pvalue"].iloc[0]),
                "diagnostic_lags": config.diagnostic_lags,
            }
        )
    out = pd.DataFrame(rows)
    out["delta_aic"] = out["aic"] - out["aic"].min()
    out["delta_bic"] = out["bic"] - out["bic"].min()
    return out.sort_values("aic").reset_index(drop=True)


def stationarity_table(df: pd.DataFrame, config: ModelConfig) -> pd.DataFrame:
    rows = []
    for name in ["log_brent", "log_basket_real_usd_rub", OIL, Y]:
        clean = df[name].dropna()
        adf = adfuller(
            clean, maxlag=config.adf_maxlag, autolag=None, regression="c", result_object=False
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", InterpolationWarning)
            kp = kpss(clean, regression="c", nlags="auto")
        boundary = ""
        if any(issubclass(w.category, InterpolationWarning) for w in caught):
            boundary = ">=" if kp[1] == 0.1 else "<="
        rows.append(
            {
                "variable": name,
                "n_obs": len(clean),
                "trend": "c",
                "adf_stat": adf[0],
                "adf_p_value": adf[1],
                "adf_usedlag": adf[2],
                "adf_effective_n": adf[3],
                "adf_autolag": "None",
                "kpss_stat": kp[0],
                "kpss_p_value": kp[1],
                "kpss_p_bound": boundary,
                "kpss_usedlag": kp[2],
            }
        )
    return pd.DataFrame(rows)


def cointegration_table(df: pd.DataFrame, config: ModelConfig) -> pd.DataFrame:
    rows = []
    for y0, y1 in [("log_basket_real_usd_rub", "log_brent"), ("log_brent", "log_basket_real_usd_rub")]:
        for lag in sorted(set([config.coint_maxlag, *config.coint_sensitivity_lags])):
            statistic, pvalue, critical = coint(df[y0], df[y1], trend="c", maxlag=lag, autolag=None)
            rows.append(
                {
                    "y0": y0,
                    "y1": y1,
                    "trend": "c",
                    "maxlag": lag,
                    "autolag": "None",
                    "primary": y0 == "log_basket_real_usd_rub" and lag == config.coint_maxlag,
                    "n_obs": len(df),
                    "start_year": int(df.year.min()),
                    "end_year": int(df.year.max()),
                    "coint_t": statistic,
                    "p_value": pvalue,
                    "crit_1pct": critical[0],
                    "crit_5pct": critical[1],
                    "crit_10pct": critical[2],
                }
            )
    return pd.DataFrame(rows)


def sensitivity_table(df: pd.DataFrame, config: ModelConfig) -> pd.DataFrame:
    rows = []
    starts = sorted(set([int(df.year.min()), *config.sensitivity_start_years]))
    for start in starts:
        sub = add_features(df.loc[df.year >= start])
        if len(sub) - 2 < config.min_observations:
            rows.append({"start_year_requested": start, "status": "insufficient_observations"})
            continue
        models = run_models(sub, config)
        for lag in config.sensitivity_hac_lags:
            settings = replace(config, hac_lags=lag)
            reg = regression_table(models, settings)
            selected = reg.loc[
                reg.parameter.eq(OIL) & reg.model.isin(["ar1_oil", "ar1_oil_lag", "ar1_oil_crises"])
            ].copy()
            selected["start_year_requested"] = start
            selected["status"] = "ok"
            rows.extend(selected.to_dict("records"))
    return pd.DataFrame(rows)


def influence_tables(bundle: ModelBundle) -> tuple[pd.DataFrame, pd.DataFrame]:
    res = bundle.result
    influence = res.get_influence()
    diagnostics = pd.DataFrame(
        {
            "year": res.resid.index,
            "cooks_distance": influence.cooks_distance[0],
            "leverage": influence.hat_matrix_diag,
            "studentized_residual": influence.resid_studentized_internal,
        }
    )
    rows = []
    for index, year in enumerate(res.resid.index):
        keep = np.arange(int(res.nobs)) != index
        # Fixed design sensitivity: do not bridge years by recomputing lags.
        refit = sm.OLS(res.model.endog[keep], res.model.exog[keep]).fit()
        position = res.model.exog_names.index(OIL)
        rows.append(
            {
                "omitted_year": int(year),
                "brent_coef": float(refit.params[position]),
                "coefficient_change": float(refit.params[position] - res.params[OIL]),
            }
        )
    return diagnostics, pd.DataFrame(rows)


def save_table(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from src.modeling import ModelBundle, comparison_table, cointegration_table, run_models
from src.preprocessing import add_features, build_final_dataset


def dataset(sources, config, directory):
    return build_final_dataset(
        sources["brent"], sources["usd_rub"], sources["ppp"], directory, config.data
    )[0]


def test_identical_sample_and_separate_levels(sources, config, tmp_path):
    df = dataset(sources, config, tmp_path)
    models = run_models(df, config.model)
    comparison = comparison_table(models, config.model)
    assert "levels" not in set(comparison.model)
    assert set(comparison.n_obs) == {31}
    assert set(comparison.start_year) == {1995}
    expected = list(range(1995, 2026))
    for bundle in models.values():
        if bundle.group == "changes":
            assert list(bundle.result.resid.index) == expected
    assert models["levels"].result.nobs == 33


def test_comparison_rejects_different_observations(sources, config, tmp_path):
    models = run_models(dataset(sources, config, tmp_path), config.model)
    original = models["ar1"].result
    result = sm.OLS(
        pd.Series(original.model.endog[1:], index=original.resid.index[1:]),
        pd.DataFrame(original.model.exog[1:], index=original.resid.index[1:]),
    ).fit()
    models["ar1"] = ModelBundle("short sample", "changes", result)
    with pytest.raises(ValueError, match="identical"):
        comparison_table(models, config.model)


def test_constant_crisis_dummy_is_explicitly_omitted(sources, config, tmp_path):
    df = dataset(sources, config, tmp_path)
    sub = add_features(df[df.year >= 2000])
    model = run_models(sub, config.model)["ar1_oil_crises"]
    assert "dummy_1998_crisis" in model.omitted
    assert "dummy_1998_crisis" not in model.result.params.index
    assert np.linalg.matrix_rank(model.result.model.exog) == model.result.model.exog.shape[1]


def test_cointegration_records_direction_and_exact_settings(sources, config, tmp_path):
    df = dataset(sources, config, tmp_path)
    result = cointegration_table(df, config.model)
    assert len(result) == 6
    assert result.primary.sum() == 1
    assert set(result.y0) == {"log_brent", "log_basket_real_usd_rub"}
    assert set(result.autolag) == {"None"}
    assert set(result.n_obs) == {33}
    assert result.loc[result.primary, "maxlag"].iloc[0] == config.model.coint_maxlag
    assert result.p_value.between(0, 1).all()

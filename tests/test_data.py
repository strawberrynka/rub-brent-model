import numpy as np
import pandas as pd
import pytest

from src.data_loader import filter_dates, load_sources, validate_fred, validate_ppp
from src.preprocessing import build_final_dataset


def build(sources, directory, config):
    return build_final_dataset(
        sources["brent"], sources["usd_rub"], sources["ppp"], directory, config.data
    )


def test_partial_first_year_excluded_before_differences(sources, config, tmp_path):
    df, coverage = build(sources, tmp_path, config)
    first = coverage.set_index("year").loc[1992]
    assert first.nominal_usd_rub_months == 7
    assert not first.included
    assert df.year.tolist() == list(range(1993, 2026))
    assert pd.isna(df.loc[0, "dlog_basket_real_usd_rub"])
    assert pd.isna(df.loc[1, "dlog_basket_real_usd_rub_lag1"])
    expected = np.log(df.loc[1, "basket_real_usd_rub"] / df.loc[0, "basket_real_usd_rub"])
    assert df.loc[1, "dlog_basket_real_usd_rub"] == pytest.approx(expected)


def test_missing_interior_year_cannot_be_bridged(sources, config, tmp_path):
    sources["ppp"] = sources["ppp"].loc[sources["ppp"].year.ne(2005)]
    with pytest.raises(ValueError, match="contiguous"):
        build(sources, tmp_path, config)


def test_duplicate_ppp_year_is_rejected(sources):
    duplicate = pd.concat([sources["ppp"], sources["ppp"].iloc[[0]]])
    with pytest.raises(ValueError, match="Duplicate PPP"):
        validate_ppp(duplicate)


@pytest.mark.parametrize("value", [0, -1, np.inf])
def test_invalid_levels_rejected(value):
    with pytest.raises(ValueError, match="positive"):
        validate_fred(pd.DataFrame({"date": ["2000-01-01"], "value": [value]}))


def test_duplicate_month_is_not_counted_as_coverage(sources, config, tmp_path):
    extra = sources["usd_rub"].iloc[[20]].copy()
    extra["date"] = extra["date"] + pd.Timedelta(days=1)
    sources["usd_rub"] = pd.concat([sources["usd_rub"], extra])
    with pytest.raises(ValueError, match="one observation per month"):
        build(sources, tmp_path, config)


def test_shuffled_sources_produce_same_features(sources, config, tmp_path):
    expected, _ = build(sources, tmp_path / "ordered", config)
    shuffled = {key: frame.sample(frac=1, random_state=5) for key, frame in sources.items()}
    actual, _ = build(shuffled, tmp_path / "shuffled", config)
    pd.testing.assert_frame_equal(actual, expected)


def test_offline_dates_are_respected(sources, config, tmp_path):
    from dataclasses import replace

    restricted = replace(config, data=replace(config.data, start_year=2000, end_year=2020))
    df, coverage = build(sources, tmp_path, restricted)
    assert df.year.tolist() == list(range(2000, 2021))
    assert coverage.year.min() == 2000 and coverage.year.max() == 2020
    filtered = filter_dates(sources["usd_rub"], "2000-06-01", "2001-02-28")
    assert filtered.date.min() == pd.Timestamp("2000-06-01")
    assert filtered.date.max() == pd.Timestamp("2001-02-01")


def test_offline_missing_file_fails_without_download(tmp_path):
    with pytest.raises(FileNotFoundError, match="offline source"):
        load_sources(tmp_path)


@pytest.mark.parametrize("date_header", ["DATE", "observation_date"])
def test_refresh_accepts_both_fred_date_headers(date_header, monkeypatch, tmp_path):
    import requests

    class Response:
        def __init__(self, url):
            series = "DCOILBRENTEU" if "DCOILBRENTEU" in url else "CCUSMA02RUM618N"
            self.text = f"{date_header},{series}\n2000-01-01,30\n2000-01-02,.\n"

        def raise_for_status(self):
            pass

        def json(self):
            return [{"pages": 1}, [{"date": "2000", "value": 7.5}]]

    monkeypatch.setattr(requests, "get", lambda url, **kwargs: Response(url))
    data, info = load_sources(tmp_path, refresh=True)
    assert len(data["brent"]) == 1
    assert data["ppp"].iloc[0].ppp_basket_usd_rub == 7.5
    assert info["mode"] == "refresh"
    assert (tmp_path / "source_metadata.json").exists()


def test_failed_refresh_preserves_entire_snapshot(monkeypatch, tmp_path):
    import requests
    from src.data_loader import SOURCES

    before = {}
    for source in SOURCES.values():
        path = tmp_path / source["file"]
        path.write_bytes(b"original snapshot")
        before[path] = path.read_bytes()

    class Response:
        def __init__(self, url):
            self.url = url
            series = "DCOILBRENTEU" if "DCOILBRENTEU" in url else "CCUSMA02RUM618N"
            self.text = f"observation_date,{series}\n2000-01-01,30\n"

        def raise_for_status(self):
            pass

        def json(self):
            return {"error": "unavailable"}

    monkeypatch.setattr(requests, "get", lambda url, **kwargs: Response(url))
    with pytest.raises(ValueError, match="World Bank"):
        load_sources(tmp_path, refresh=True)
    assert all(path.read_bytes() == contents for path, contents in before.items())

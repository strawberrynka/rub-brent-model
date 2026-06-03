from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from pandas_datareader import data as web


FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
WB_URL = "https://api.worldbank.org/v2/country/RUS/indicator/PA.NUS.PRVT.PP?format=json&per_page=20000"


def _save_raw(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _load_cached_fred(raw_path: Path) -> pd.DataFrame:
    series = pd.read_csv(raw_path)
    series["date"] = pd.to_datetime(series["date"])
    series["value"] = pd.to_numeric(series["value"], errors="coerce")
    return series.dropna(subset=["value"]).sort_values("date")


def _load_cached_ppp(raw_path: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_path)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["ppp_basket_usd_rub"] = pd.to_numeric(df["ppp_basket_usd_rub"], errors="coerce")
    df = df[["year", "ppp_basket_usd_rub"]].dropna().sort_values("year")
    df["year"] = df["year"].astype(int)
    return df


def _load_fred_series(
    series_id: str,
    start_date: str,
    end_date: Optional[str],
    raw_path: Path,
) -> pd.DataFrame:
    try:
        series = web.DataReader(series_id, "fred", start_date, end_date)
        series = series.reset_index().rename(columns={"DATE": "date", series_id: "value"})
    except Exception as e_datareader:
        print(f"[WARN] DataReader не сработал для {series_id}: {e_datareader}. Пробуем fallback CSV...")
        try:
            url = FRED_CSV_URL.format(series_id=series_id)
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            series = pd.read_csv(StringIO(response.text))
            series = series.rename(columns={"DATE": "date", series_id: "value"})
            series["date"] = pd.to_datetime(series["date"])
            if end_date is not None:
                mask = (series["date"] >= pd.to_datetime(start_date)) & (
                    series["date"] <= pd.to_datetime(end_date)
                )
            else:
                mask = series["date"] >= pd.to_datetime(start_date)
            series = series.loc[mask].copy()
        except Exception as e_csv:
            if raw_path.exists():
                print(f"[WARN] Используем сохраненный raw-файл для {series_id}: {raw_path}")
                return _load_cached_fred(raw_path)
            raise RuntimeError(
                f"Не удалось загрузить серию {series_id} ни через DataReader, ни через CSV fallback."
            ) from e_csv

    series["date"] = pd.to_datetime(series["date"])
    series["value"] = pd.to_numeric(series["value"], errors="coerce")
    series = series.dropna(subset=["value"]).sort_values("date")
    _save_raw(series, raw_path)
    return series


def load_brent_fred(
    start_date: str,
    end_date: Optional[str],
    raw_dir: Path,
) -> pd.DataFrame:
    return _load_fred_series(
        series_id="DCOILBRENTEU",
        start_date=start_date,
        end_date=end_date,
        raw_path=raw_dir / "brent_fred_raw.csv",
    )


def load_usd_rub_fred(
    start_date: str,
    end_date: Optional[str],
    raw_dir: Path,
) -> pd.DataFrame:
    return _load_fred_series(
        series_id="CCUSMA02RUM618N",
        start_date=start_date,
        end_date=end_date,
        raw_path=raw_dir / "usd_rub_fred_raw.csv",
    )


def load_ppp_world_bank(raw_dir: Path) -> pd.DataFrame:
    raw_path = raw_dir / "ppp_world_bank_raw.csv"
    try:
        response = requests.get(WB_URL, timeout=30)
        response.raise_for_status()
        payload = response.json()
    except Exception as e:
        if raw_path.exists():
            print(f"[WARN] Используем сохраненный raw-файл World Bank PPP: {raw_path}")
            return _load_cached_ppp(raw_path)
        raise RuntimeError(f"Ошибка при загрузке World Bank API: {e}") from e

    if not isinstance(payload, list) or len(payload) < 2 or not payload[1]:
        raise RuntimeError(
            "World Bank API вернул пустой или некорректный ответ для PA.NUS.PRVT.PP (RUS)."
        )

    rows = payload[1]
    df = pd.DataFrame(rows)
    if "date" not in df.columns or "value" not in df.columns:
        raise RuntimeError("В ответе World Bank отсутствуют поля 'date' или 'value'.")

    df = df[["date", "value"]].copy()
    df["year"] = pd.to_numeric(df["date"], errors="coerce")
    df["ppp_basket_usd_rub"] = pd.to_numeric(df["value"], errors="coerce")
    df = df[["year", "ppp_basket_usd_rub"]].dropna().sort_values("year")
    df["year"] = df["year"].astype(int)

    _save_raw(df, raw_path)
    return df

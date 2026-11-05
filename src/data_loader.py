"""Explicit offline/refresh modes; failed refreshes never silently use stale data."""

from datetime import datetime, timezone
from io import StringIO
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
WB_URL = "https://api.worldbank.org/v2/country/RUS/indicator/PA.NUS.PRVT.PP?format=json&per_page=20000"
SOURCES = {
    "brent": {
        "file": "brent_fred_raw.csv",
        "series": "DCOILBRENTEU",
        "url": FRED_URL.format(series="DCOILBRENTEU"),
    },
    "usd_rub": {
        "file": "usd_rub_fred_raw.csv",
        "series": "CCUSMA02RUM618N",
        "url": FRED_URL.format(series="CCUSMA02RUM618N"),
    },
    "ppp": {"file": "ppp_world_bank_raw.csv", "series": "PA.NUS.PRVT.PP", "url": WB_URL},
}


def validate_fred(frame: pd.DataFrame) -> pd.DataFrame:
    if not {"date", "value"} <= set(frame):
        raise ValueError("FRED data must have date and value columns")
    df = frame[["date", "value"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="raise").dt.normalize()
    df["value"] = pd.to_numeric(df["value"].replace(".", np.nan), errors="raise")
    if df["date"].isna().any() or df["date"].duplicated().any():
        raise ValueError("Missing or duplicate dates in FRED data")
    df = df.dropna(subset=["value"]).sort_values("date").reset_index(drop=True)
    if df.empty or not np.isfinite(df["value"]).all() or (df["value"] <= 0).any():
        raise ValueError("FRED observations must be finite and positive")
    return df


def validate_ppp(frame: pd.DataFrame) -> pd.DataFrame:
    if not {"year", "ppp_basket_usd_rub"} <= set(frame):
        raise ValueError("PPP data must have year and ppp_basket_usd_rub columns")
    df = frame[["year", "ppp_basket_usd_rub"]].copy()
    df["year"] = pd.to_numeric(df["year"], errors="raise")
    if df["year"].isna().any() or (df["year"] % 1 != 0).any():
        raise ValueError("PPP years must be integers")
    df["year"] = df["year"].astype(int)
    if df["year"].duplicated().any():
        raise ValueError("Duplicate PPP years")
    df["ppp_basket_usd_rub"] = pd.to_numeric(df["ppp_basket_usd_rub"], errors="raise")
    df = df.dropna(subset=["ppp_basket_usd_rub"]).sort_values("year").reset_index(drop=True)
    values = df["ppp_basket_usd_rub"]
    if df.empty or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("PPP observations must be finite and positive")
    return df


def filter_dates(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    if start > end:
        raise ValueError("Invalid date range")
    return df.loc[df["date"].between(start, end)].copy()


def load_sources(raw_dir: Path, *, refresh: bool = False) -> tuple[dict, dict]:
    """Validate all responses before replacing any cached source."""
    frames = {}
    fetched_at = None
    if refresh:
        for name, source in SOURCES.items():
            response = requests.get(source["url"], timeout=(10, 60))
            response.raise_for_status()
            if name == "ppp":
                payload = response.json()
                if not isinstance(payload, list) or len(payload) != 2 or not payload[1]:
                    raise ValueError("Invalid or empty World Bank response")
                if int(payload[0].get("pages", 1)) != 1:
                    raise ValueError("World Bank response requires pagination")
                frame = pd.DataFrame(payload[1]).rename(
                    columns={"date": "year", "value": "ppp_basket_usd_rub"}
                )
                frames[name] = validate_ppp(frame)
            else:
                frame = pd.read_csv(StringIO(response.text)).rename(
                    columns={"DATE": "date", "observation_date": "date", source["series"]: "value"}
                )
                frames[name] = validate_fred(frame)
        raw_dir.mkdir(parents=True, exist_ok=True)
        for name, frame in frames.items():
            path = raw_dir / SOURCES[name]["file"]
            temporary = path.with_suffix(".csv.tmp")
            frame.to_csv(temporary, index=False)
            temporary.replace(path)
        fetched_at = datetime.now(timezone.utc).isoformat()
        (raw_dir / "source_metadata.json").write_text(
            json.dumps({"fetched_at_utc": fetched_at, "sources": SOURCES}, indent=2), encoding="utf-8"
        )
    else:
        for name, source in SOURCES.items():
            path = raw_dir / source["file"]
            if not path.exists():
                raise FileNotFoundError(f"Missing offline source: {path}. Run --refresh explicitly.")
            frame = pd.read_csv(path)
            frames[name] = validate_ppp(frame) if name == "ppp" else validate_fred(frame)
        metadata_path = raw_dir / "source_metadata.json"
        if metadata_path.exists():
            fetched_at = json.loads(metadata_path.read_text(encoding="utf-8")).get("fetched_at_utc")
    return frames, {"mode": "refresh" if refresh else "offline", "fetched_at_utc": fetched_at}

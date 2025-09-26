#!/usr/bin/env python3
"""
India Macro Data Pipeline (without ICB)

What this does
--------------
Pulls key Indian macroeconomic time series from **IMF IFS (SDMX JSON API)** and **World Bank API**,
merges them into a clean monthly panel, and writes to `data/processed/india_macro_monthly.csv`.

Included series (all monthly unless noted):
- CPI index (IMF IFS: `PCPI_IX`)
- Industrial Production Index / IIP (IMF IFS: `AIP_IX`)
- USD/INR, domestic currency per USD, period average (IMF IFS: `ENDA_XDC_USD_RATE`)
- Total reserves excl. gold, USD (IMF IFS: `RAXG_USD`)
- Annual GDP, current US$ (World Bank: `NY.GDP.MKTP.CD`) — backfilled to monthly (constant within year)

You can easily add more IMF/World Bank indicators by extending the `IMF_SERIES` and
`WB_SERIES` dictionaries below.

Usage
-----
python india_macro_pipeline.py --start 2000-01 --end 2025-12

Requirements
------------
Python 3.9+ and the following pip packages:
    pandas requests python-dateutil

Project layout:
    data/raw/        # downloaded raw files (optional)
    data/processed/  # final merged CSV written here

"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List

import pandas as pd
import numpy as np
import requests

# -----------------------------
# Configuration
# -----------------------------
IMF_BASE = "https://dataservices.imf.org/REST/SDMX_JSON.svc/CompactData"
IMF_DATASET = "IFS"  # International Financial Statistics
IMF_AREA = "IN"      # India
IMF_FREQ = "M"       # Monthly

IMF_SERIES: Dict[str, str] = {
    "cpi_index": "PCPI_IX",
    "iip_index": "AIP_IX",
    "usdinr_avg": "ENDA_XDC_USD_RATE",
    "fx_reserves_usd": "RAXG_USD",
}

WB_BASE = "https://api.worldbank.org/v2"
WB_COUNTRY = "IND"
WB_SERIES: Dict[str, str] = {
    "gdp_current_usd": "NY.GDP.MKTP.CD",
}

@dataclass
class SeriesSpec:
    dataset: str
    freq: str
    area: str
    indicator: str


def _imf_compact_url(spec: SeriesSpec, start: Optional[str] = None, end: Optional[str] = None) -> str:
    key = f"{spec.dataset}/{spec.freq}.{spec.area}.{spec.indicator}"
    url = f"{IMF_BASE}/{key}"
    params = []
    if start:
        params.append(f"startPeriod={start}")
    if end:
        params.append(f"endPeriod={end}")
    if params:
        url += "?" + "&".join(params)
    return url


def fetch_imf_series(indicator: str, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    spec = SeriesSpec(dataset=IMF_DATASET, freq=IMF_FREQ, area=IMF_AREA, indicator=indicator)
    url = _imf_compact_url(spec, start, end)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    try:
        series = data["CompactData"]["DataSet"].get("Series")
    except Exception as exc:
        raise RuntimeError(f"IMF response structure unexpected for {indicator}: {exc}")

    if series is None:
        return pd.DataFrame(columns=["date", "value"])

    if isinstance(series, list):
        series = series[0]

    obs = series.get("Obs", [])
    rows = []
    for o in obs:
        t = o.get("@TIME_PERIOD")
        v = o.get("@OBS_VALUE")
        if t is None or v in (None, ""):
            continue
        try:
            rows.append({"date": pd.Period(t, freq="M"), "value": float(v)})
        except Exception:
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("date")
    df["date"] = df["date"].dt.to_timestamp(how="end")
    return df


def fetch_world_bank_annual(indicator: str) -> pd.DataFrame:
    url = f"{WB_BASE}/country/{WB_COUNTRY}/indicator/{indicator}"
    params = {"format": "json", "per_page": 20000}
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    payload = resp.json()
    records = payload[1] or []
    rows = []
    for r in records:
        year = r.get("date")
        val = r.get("value")
        if year and val is not None:
            dt = pd.Timestamp(int(year), 12, 1) + pd.offsets.MonthEnd(0)
            rows.append({"date": dt, "value": float(val)})
    df = pd.DataFrame(rows).sort_values("date")
    return df


def run_pipeline(start: Optional[str], end: Optional[str], out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    imf_frames = {}
    for out_name, code in IMF_SERIES.items():
        df = fetch_imf_series(code, start, end)
        df = df.rename(columns={"value": out_name})
        imf_frames[out_name] = df

    wb_frames = {}
    for out_name, code in WB_SERIES.items():
        df = fetch_world_bank_annual(code)
        df = df.rename(columns={"value": out_name})
        wb_frames[out_name] = df

    min_date, max_date = None, None
    frames = list(imf_frames.values()) + list(wb_frames.values())
    for f in frames:
        if f.empty:
            continue
        d0, d1 = f["date"].min(), f["date"].max()
        min_date = d0 if (min_date is None or d0 < min_date) else min_date
        max_date = d1 if (max_date is None or d1 > max_date) else max_date

    if start:
        syear, smonth = map(int, start.split("-"))
        min_date = pd.Timestamp(syear, smonth, 1) + pd.offsets.MonthEnd(0)
    if end:
        eyear, emonth = map(int, end.split("-"))
        max_date = pd.Timestamp(eyear, emonth, 1) + pd.offsets.MonthEnd(0)

    date_index = pd.date_range(min_date, max_date, freq="M")
    panel = pd.DataFrame({"date": date_index})

    def merge_in(df: pd.DataFrame):
        nonlocal panel
        if not df.empty:
            panel = panel.merge(df, on="date", how="left")

    for f in imf_frames.values():
        merge_in(f)
    for f in wb_frames.values():
        if not f.empty:
            f = f.set_index("date").reindex(date_index, method="ffill").reset_index().rename(columns={"index": "date"})
        merge_in(f)

    panel = panel.sort_values("date").reset_index(drop=True)
    panel.to_csv(out_csv, index=False)
    print(f"Saved macro monthly panel -> {out_csv}")
    print(f"Rows: {len(panel)} | Columns: {len(panel.columns)})")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="India Macro pipeline (IMF IFS + World Bank)")
    p.add_argument("--start", type=str, default=None, help="Start period YYYY-MM for IMF pulls")
    p.add_argument("--end", type=str, default=None, help="End period YYYY-MM for IMF pulls")
    p.add_argument("--out", type=Path, default=Path("data/processed/india_macro_monthly.csv"), help="Output CSV path")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(start=args.start, end=args.end, out_csv=args.out)

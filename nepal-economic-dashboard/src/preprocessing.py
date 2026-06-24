"""
Preprocessing & Feature Engineering Pipeline
=============================================
Builds a unified monthly time-series DataFrame from parsed NRB datasets,
ready for ML modeling.
"""

import numpy as np
import pandas as pd
import re
from pathlib import Path

from .data_loader import (
    load_cpi_yoy_series, load_wpi_yoy_series,
    load_exchange_rate, load_interest_rate_series,
    load_money_supply, load_tourist_arrivals,
    load_cpi_province, load_cpi_ecology,
)

# ─────────────────────────────────────────────────────────────────────────────
# Re-export simple series builders (thin wrappers for clean API)
# ─────────────────────────────────────────────────────────────────────────────

def build_cpi_series() -> pd.Series:
    """Overall CPI YoY % change as monthly pd.Series."""
    return load_cpi_yoy_series()


def build_wpi_series() -> pd.Series:
    """Overall WPI YoY % change as monthly pd.Series."""
    return load_wpi_yoy_series()


# ─────────────────────────────────────────────────────────────────────────────
# Build feature: Exchange rate (mid-month middle rate)
# ─────────────────────────────────────────────────────────────────────────────

_MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

def build_exchange_rate_series() -> pd.Series:
    """Monthly USD/NPR middle rate as pd.Series."""
    df = load_exchange_rate()
    records = []
    for _, row in df.iterrows():
        mo_str = str(row["month"])[:3].lower()
        mo = _MONTH_MAP.get(mo_str)
        fy = str(row["fiscal_year"])
        yr_match = re.search(r"(\d{4})", fy)
        yr = int(yr_match.group(1)) if yr_match else None
        if mo and yr:
            actual_yr = yr if mo >= 7 else yr + 1
            ts = pd.Timestamp(actual_yr, mo, 1)
            val = row.get("avg_middle") if pd.notna(row.get("avg_middle")) else row.get("middle")
            if pd.notna(val):
                records.append((ts, float(val)))

    if not records:
        return pd.Series([], name="usd_npr", dtype=float)

    series = (pd.DataFrame(records, columns=["date", "usd_npr"])
                .set_index("date")["usd_npr"]
                .sort_index()
                .drop_duplicates()
                .rename("usd_npr"))
    series.index.name = "date"
    return series


# ─────────────────────────────────────────────────────────────────────────────
# Build feature: Policy interest rate (repo rate)
# ─────────────────────────────────────────────────────────────────────────────

def build_repo_rate_series() -> pd.Series:
    """Monthly repo rate (NRB policy rate) as pd.Series."""
    return load_interest_rate_series("Fixed Repo")


# ─────────────────────────────────────────────────────────────────────────────
# Build feature: Money Supply (M2)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_period_from_monetary(text: str) -> pd.Timestamp | None:
    """Parse period strings like '2024 Jul', '2025 May (R)' → Timestamp."""
    text = re.sub(r"\s*\(.*?\)\s*", " ", str(text)).strip()
    m = re.search(r"(\d{4})\s+([A-Za-z]+)", text)
    if m:
        yr = int(m.group(1))
        mo_str = m.group(2)[:3].lower()
        mo = _MONTH_MAP.get(mo_str)
        if mo:
            return pd.Timestamp(yr, mo, 1)
    return None


def build_m2_series() -> pd.Series:
    """Monthly M2 money supply as pd.Series (Rs. million)."""
    df = load_money_supply()
    if df.empty:
        return pd.Series([], name="m2", dtype=float)

    m2_keys = ["M2", "Broad Money", "M 2"]
    row = None
    for k in m2_keys:
        matches = [idx for idx in df.index if k.lower() in str(idx).lower()]
        if matches:
            row = df.loc[matches[0]]
            break
    if row is None:
        row = df.iloc[0]

    ts_dict = {}
    for col, val in row.items():
        ts = _parse_period_from_monetary(str(col))
        if ts is not None and pd.notna(val):
            ts_dict[ts] = float(val)

    series = pd.Series(ts_dict, name="m2").sort_index()
    series.index.name = "date"
    return series


# ─────────────────────────────────────────────────────────────────────────────
# Build feature: Tourist arrivals (annual total per year)
# ─────────────────────────────────────────────────────────────────────────────

def build_tourist_series() -> pd.Series:
    """Annual total tourist arrivals as pd.Series."""
    df = load_tourist_arrivals()
    totals = {}
    for col in df.columns:
        yr_match = re.search(r"(\d{4})", str(col))
        if yr_match:
            yr = int(yr_match.group(1))
            val = pd.to_numeric(df[col], errors="coerce").sum()
            if pd.notna(val) and val > 0:
                totals[pd.Timestamp(yr, 12, 31)] = val

    series = pd.Series(totals, name="tourist_arrivals").sort_index()
    series.index.name = "date"
    return series


# ─────────────────────────────────────────────────────────────────────────────
# Assemble unified monthly feature matrix
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_matrix() -> pd.DataFrame:
    """
    Merge all monthly series into one DataFrame aligned on date.
    Missing values are forward-filled then backward-filled.
    Returns a DataFrame with 'cpi_yoy' as the target and all others as features.
    """
    cpi  = build_cpi_series()
    wpi  = build_wpi_series()
    fx   = build_exchange_rate_series()
    repo = build_repo_rate_series()
    m2   = build_m2_series()

    dfs = [s.to_frame() for s in [cpi, wpi, fx, repo, m2] if not s.empty]
    if not dfs:
        raise ValueError("No data series could be built.")

    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.join(df, how="outer")

    merged = merged.sort_index()
    merged = merged.dropna(subset=["cpi_yoy"])
    merged = merged.ffill().bfill()

    if "m2" in merged.columns:
        merged["m2_yoy"] = merged["m2"].pct_change(12) * 100

    if "usd_npr" in merged.columns:
        merged["fx_change"] = merged["usd_npr"].pct_change(1) * 100

    for lag in [1, 3, 6, 12]:
        merged[f"cpi_lag_{lag}"] = merged["cpi_yoy"].shift(lag)

    merged = merged.dropna()
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Train/test split utility
# ─────────────────────────────────────────────────────────────────────────────

def train_test_split_ts(df: pd.DataFrame,
                         target: str = "cpi_yoy",
                         test_ratio: float = 0.2):
    """Chronological train/test split (no shuffling)."""
    feature_cols = [c for c in df.columns if c != target]
    n = len(df)
    split = int(n * (1 - test_ratio))

    X = df[feature_cols]
    y = df[target]

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    df = build_feature_matrix()
    print(f"Feature matrix shape: {df.shape}")
    print(df.tail())

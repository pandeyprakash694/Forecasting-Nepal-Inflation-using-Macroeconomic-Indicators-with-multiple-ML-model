"""
NRB Data Loader
================
Parses all NRB CSV files (originally exported from Excel with multi-row headers).
Each loader returns a clean pandas DataFrame ready for analysis.
"""

import os
import re
import numpy as np
import pandas as pd
from pathlib import Path

# ── Root path: auto-resolve relative to this file ──────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT.parent / "nrb" / "data"


def _data(filename: str) -> Path:
    return DATA_DIR / filename


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_MONTH_NUM = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

def _fiscal_year_base(fy_str: str) -> int:
    """'2022/23' → 2022  (first year of fiscal year)."""
    m = re.search(r"(\d{4})", str(fy_str))
    return int(m.group(1)) if m else 0


def _month_to_calendar_year(month_name: str, fy_base: int) -> int:
    """
    Nepal fiscal year runs Aug–Jul.
    Aug–Dec → first year of FY;  Jan–Jul → second year of FY.
    """
    mo = _MONTH_NUM.get(month_name.strip()[:3].lower(), 0)
    return fy_base if mo >= 8 else fy_base + 1


def _parse_fy_month_table(filepath: str,
                           fy_row_idx: int = 4,
                           subtype_row_idx: int = 5,
                           data_start_idx: int = 6,
                           month_col: int = 0,
                           pct_change_label: str = "% Change") -> pd.DataFrame:
    """
    Parse tables that have:
      - Month names in column `month_col`
      - Fiscal year labels in row `fy_row_idx` (cols 1, 3, 5, ...)
      - Sub-type labels in row `subtype_row_idx` (e.g. Index / % Change)
      - Data from row `data_start_idx` onwards

    Returns a long-format DataFrame with columns:
        fiscal_year, month, cal_year, cal_date, index_val, pct_change
    """
    raw = pd.read_csv(filepath, header=None, dtype=str)

    fy_row = raw.iloc[fy_row_idx]
    sub_row = raw.iloc[subtype_row_idx]

    # Determine last meaningful column to avoid iterating over thousands of empties
    # A column is meaningful if fy_row or sub_row has a non-empty value
    last_meaningful = 1
    for j in range(1, raw.shape[1]):
        fy_v = str(fy_row.iloc[j]).strip()
        sub_v = str(sub_row.iloc[j]).strip()
        if fy_v not in ("nan", "NaN", "") or sub_v not in ("nan", "NaN", ""):
            last_meaningful = j

    # Collect (col_index, fiscal_year, sub_type) up to last_meaningful
    col_meta = []
    current_fy = None
    for j in range(1, last_meaningful + 1):
        fy = str(fy_row.iloc[j]).strip()
        if fy and fy not in ("nan", "NaN"):
            current_fy = fy
        sub = str(sub_row.iloc[j]).strip()
        if sub in ("nan", "NaN"):
            sub = ""
        if current_fy:
            col_meta.append((j, current_fy, sub))

    records = []
    for i in range(data_start_idx, len(raw)):
        row = raw.iloc[i]
        month_name = str(row.iloc[month_col]).strip()
        if not month_name or month_name in ("nan", "NaN", "Average"):
            continue
        mo = _MONTH_NUM.get(month_name[:3].lower(), 0)
        if mo == 0:
            continue
        for j, fy, sub in col_meta:
            val = pd.to_numeric(row.iloc[j], errors="coerce")
            yr_base = _fiscal_year_base(fy)
            cal_yr = _month_to_calendar_year(month_name, yr_base)
            if cal_yr > 0:
                records.append({
                    "fiscal_year": fy,
                    "month": month_name,
                    "cal_year": cal_yr,
                    "cal_date": pd.Timestamp(cal_yr, mo, 1),
                    "sub_type": sub,
                    "value": val,
                })

    return pd.DataFrame(records)


def _to_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if col not in ("date", "year", "month", "fiscal_year", "period"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 1. Selected Macroeconomic Indicators (Annual)
# ─────────────────────────────────────────────────────────────────────────────

def load_smi() -> pd.DataFrame:
    """Annual macroeconomic summary (GDP, CPI, M2, reserves, etc.)."""
    raw = pd.read_csv(_data("1_SMIs.csv"), header=None, dtype=str)
    # Row 3 has fiscal year headers starting from col 3
    year_row = raw.iloc[3]
    years = [str(v).strip() for v in year_row.iloc[3:]
             if pd.notna(v) and re.match(r"\d{4}", str(v).strip())]

    records = []
    for i in range(4, len(raw)):
        row = raw.iloc[i]
        indicator = str(row.iloc[2]).strip() if pd.notna(row.iloc[2]) else ""
        if not indicator or indicator in ("nan", "NaN"):
            continue
        values = list(row.iloc[3: 3 + len(years)])
        rec = {"indicator": indicator}
        for yr, val in zip(years, values):
            rec[str(yr).strip()] = pd.to_numeric(val, errors="coerce")
        records.append(rec)
    df = pd.DataFrame(records).set_index("indicator")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. CPI Year-over-Year (Monthly)
# ─────────────────────────────────────────────────────────────────────────────

def load_cpi_yoy() -> pd.DataFrame:
    """
    Monthly CPI Year-over-Year long-format DataFrame.
    Columns: fiscal_year, month, cal_year, cal_date, sub_type, value
    """
    return _parse_fy_month_table(str(_data("5_CPI_Y-O-Y.csv")))


def load_cpi_yoy_series() -> pd.Series:
    """
    Overall CPI YoY % change as a monthly pd.Series indexed by calendar date.
    """
    df = load_cpi_yoy()
    pct = df[df["sub_type"].str.contains("Change", case=False, na=False)].copy()
    pct = pct.dropna(subset=["value", "cal_date"])
    pct = pct.sort_values("cal_date")
    # Take only one value per date (deduplicate: keep first)
    series = (pct.drop_duplicates("cal_date")
                 .set_index("cal_date")["value"]
                 .rename("cpi_yoy")
                 .sort_index())
    series.index.name = "date"
    return series


# ─────────────────────────────────────────────────────────────────────────────
# 3. CPI Province-wise (cross-section snapshot)
# ─────────────────────────────────────────────────────────────────────────────

def load_cpi_province() -> pd.DataFrame:
    """Province-wise CPI snapshot table."""
    raw = pd.read_csv(_data("3_CPI_Province.csv"), header=None, dtype=str)

    # Find where actual province rows begin (rows with content in col 0)
    # Col 0 has: S.N. → province numbers like 1, 2, ...
    records = []
    # Build column headers from rows 7-10
    period_cols = []
    for j in range(3, raw.shape[1]):
        parts = []
        for r in range(7, 11):
            if r < len(raw):
                v = str(raw.iloc[r, j]).strip()
                if v and v not in ("nan", "NaN"):
                    parts.append(v)
        if parts:
            period_cols.append((j, " ".join(parts)))

    for i in range(11, len(raw)):
        row = raw.iloc[i]
        label = str(row.iloc[1]).strip()
        if not label or label in ("nan", "NaN"):
            continue
        rec = {"entity": label}
        for j, col_name in period_cols:
            rec[col_name] = pd.to_numeric(row.iloc[j], errors="coerce")
        records.append(rec)

    return pd.DataFrame(records).set_index("entity") if records else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# 4. CPI Ecology-wise (cross-section snapshot)
# ─────────────────────────────────────────────────────────────────────────────

def load_cpi_ecology() -> pd.DataFrame:
    """Ecology-wise CPI (Mountain / Hill / Terai) snapshot."""
    raw = pd.read_csv(_data("4_CPI_Ecology.csv"), header=None, dtype=str)

    period_cols = []
    for j in range(2, raw.shape[1]):
        parts = []
        for r in range(7, 11):
            if r < len(raw):
                v = str(raw.iloc[r, j]).strip()
                if v and v not in ("nan", "NaN"):
                    parts.append(v)
        if parts:
            period_cols.append((j, " ".join(parts)))

    records = []
    for i in range(11, len(raw)):
        row = raw.iloc[i]
        label = str(row.iloc[1]).strip()
        if not label or label in ("nan", "NaN"):
            continue
        rec = {"entity": label}
        for j, col_name in period_cols:
            rec[col_name] = pd.to_numeric(row.iloc[j], errors="coerce")
        records.append(rec)

    return pd.DataFrame(records).set_index("entity") if records else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# 5. WPI Year-over-Year
# ─────────────────────────────────────────────────────────────────────────────

def load_wpi_yoy() -> pd.DataFrame:
    """Monthly WPI Year-over-Year long-format DataFrame."""
    return _parse_fy_month_table(str(_data("8_WPI__Y-O-Y_.csv")))


def load_wpi_yoy_series() -> pd.Series:
    """Overall WPI YoY % change as a monthly pd.Series."""
    df = load_wpi_yoy()
    pct = df[df["sub_type"].str.contains("Change", case=False, na=False)].copy()
    pct = pct.dropna(subset=["value", "cal_date"])
    series = (pct.drop_duplicates("cal_date")
                 .set_index("cal_date")["value"]
                 .rename("wpi_yoy")
                 .sort_index())
    series.index.name = "date"
    return series


# ─────────────────────────────────────────────────────────────────────────────
# 6. Exchange Rate (USD/NPR)
# ─────────────────────────────────────────────────────────────────────────────

def load_exchange_rate() -> pd.DataFrame:
    """Monthly USD/NPR exchange rate (buying, selling, middle)."""
    raw = pd.read_csv(_data("32_33_Exchange_Rate.csv"), header=None, dtype=str)

    # Row 2 has sub-column headers; row 3+ has data
    # Structure: col1=fiscal_year, col2=month, col3=buying, col4=selling, col5=middle
    records = []
    for i in range(3, len(raw)):
        row = raw.iloc[i]
        fy = str(row.iloc[1]).strip()
        month = str(row.iloc[2]).strip()
        if not month or month in ("nan", "NaN"):
            continue
        if fy in ("nan", "NaN"):
            fy = records[-1]["fiscal_year"] if records else ""
        records.append({
            "fiscal_year": fy,
            "month": month,
            "buying": pd.to_numeric(row.iloc[3], errors="coerce"),
            "selling": pd.to_numeric(row.iloc[4], errors="coerce"),
            "middle": pd.to_numeric(row.iloc[5], errors="coerce"),
            "avg_buying": pd.to_numeric(row.iloc[6], errors="coerce"),
            "avg_selling": pd.to_numeric(row.iloc[7], errors="coerce"),
            "avg_middle": pd.to_numeric(row.iloc[8], errors="coerce"),
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Interest Rates (Policy + Market Rates, Monthly)
# ─────────────────────────────────────────────────────────────────────────────

def load_interest_rates() -> pd.DataFrame:
    """
    Monthly interest rates wide-format:
    Rows = rate types, columns = date strings (e.g. '2016 Oct').
    """
    raw = pd.read_csv(_data("55_Interest_Rate.csv"), header=None, dtype=str)

    # Row 3 has date strings (2016 Oct, 2016 Nov, ...)
    period_row = raw.iloc[3]
    periods = [str(v).strip() for v in period_row.iloc[1:]
               if str(v).strip() not in ("nan", "NaN", "", "Year")]

    records = []
    for i in range(4, len(raw)):
        row = raw.iloc[i]
        rate_name = str(row.iloc[0]).strip()
        if not rate_name or rate_name in ("nan", "NaN"):
            continue
        values = [pd.to_numeric(row.iloc[j + 1], errors="coerce")
                  for j in range(len(periods))]
        # Skip pure section-header rows
        if all(pd.isna(v) for v in values):
            continue
        rec = {"rate_type": rate_name}
        for p, val in zip(periods, values):
            rec[p] = val
        records.append(rec)

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).set_index("rate_type")


def load_interest_rate_series(rate_name_pattern: str = "Fixed Repo") -> pd.Series:
    """
    Extract a specific interest rate as a monthly pd.Series.
    Columns in interest rate table look like '2016 Oct', '2016 Nov', ...
    """
    df = load_interest_rates()
    if df.empty:
        return pd.Series([], name="interest_rate", dtype=float)

    matches = [idx for idx in df.index
               if rate_name_pattern.lower() in str(idx).lower()]
    if not matches:
        return pd.Series([], name="interest_rate", dtype=float)

    row = df.loc[matches[0]]
    ts_dict = {}
    for col, val in row.items():
        col_clean = re.sub(r"\s+", " ", str(col)).strip()
        # Remove "Mid-" prefix (e.g. "2026 Mid-Jan" → "2026 Jan")
        col_clean = re.sub(r"\bMid-?", "", col_clean, flags=re.IGNORECASE).strip()
        # Parse "YYYY Mon" or "YYYY MonAbbr"
        m = re.search(r"(\d{4})\s+([A-Za-z]{3})", col_clean)
        if m:
            yr = int(m.group(1))
            mo_str = m.group(2)[:3].lower()
            mo = _MONTH_NUM.get(mo_str)
            if mo and pd.notna(val):
                ts_dict[pd.Timestamp(yr, mo, 1)] = float(val)

    return pd.Series(ts_dict, name="repo_rate").sort_index()


# ─────────────────────────────────────────────────────────────────────────────
# 8. Forex Reserves
# ─────────────────────────────────────────────────────────────────────────────

def load_forex_reserves() -> pd.DataFrame:
    """Gross foreign assets of the banking sector."""
    raw = pd.read_csv(_data("30_ReserveRs.csv"), header=None, dtype=str)

    records = []
    # Find data rows (after the header section)
    for i in range(5, len(raw)):
        row = raw.iloc[i]
        item = str(row.iloc[1]).strip()
        if not item or item in ("nan", "NaN"):
            continue
        rec = {"item": item}
        for j, col_name in enumerate(["jul_2024", "may_2025r", "jul_2025r", "may_2026p",
                                       "pct_change_2024_25", "pct_change_2025_26"]):
            rec[col_name] = pd.to_numeric(row.iloc[3 + j], errors="coerce")
        records.append(rec)

    return pd.DataFrame(records).set_index("item")


# ─────────────────────────────────────────────────────────────────────────────
# 9. Money Supply (Monetary Survey)
# ─────────────────────────────────────────────────────────────────────────────

def load_money_supply() -> pd.DataFrame:
    """Monthly monetary aggregates (Foreign assets, domestic credit, M2, etc.)."""
    raw = pd.read_csv(_data("37_MS.csv"), header=None, dtype=str)

    # Row 4: year labels (2024, 2025, 2025, 2026)
    # Row 5: month labels (Jul, May(R), Jul(R), May(P))
    # Combine to form period columns
    year_row = raw.iloc[4]
    month_row = raw.iloc[5]

    col_names = {}
    current_yr = ""
    for j in range(1, raw.shape[1]):
        yr = str(year_row.iloc[j]).strip()
        mo = str(month_row.iloc[j]).strip()
        if yr and yr not in ("nan", "NaN"):
            current_yr = re.sub(r"\D", "", yr)
        mo_clean = re.sub(r"\s*\(.*?\)\s*", "", mo).strip()
        if mo_clean and mo_clean not in ("nan", "NaN", "") and current_yr:
            col_names[j] = f"{current_yr} {mo_clean}"

    records = []
    for i in range(7, len(raw)):
        row = raw.iloc[i]
        agg = str(row.iloc[0]).strip()
        if not agg or agg in ("nan", "NaN"):
            continue
        # Clean indicator name
        agg = re.sub(r"^\d+\.\s*", "", agg).strip()
        rec = {"aggregate": agg}
        for j, col_name in col_names.items():
            rec[col_name] = pd.to_numeric(row.iloc[j], errors="coerce")
        records.append(rec)

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).set_index("aggregate")


# ─────────────────────────────────────────────────────────────────────────────
# 10. Deposits
# ─────────────────────────────────────────────────────────────────────────────

def load_deposits() -> pd.DataFrame:
    """Deposit details of banks and financial institutions."""
    raw = pd.read_csv(_data("46_Deposits.csv"), header=None, dtype=str)

    records = []
    for i in range(5, len(raw)):
        row = raw.iloc[i]
        item = str(row.iloc[0]).strip()
        if not item or item in ("nan", "NaN"):
            continue
        rec = {"item": item}
        for j, col in enumerate(["jul_2024", "may_2025r", "jul_2025r", "may_2026p",
                                   "change_amt_2024_25", "change_pct_2024_25",
                                   "change_amt_2025_26", "change_pct_2025_26"]):
            rec[col] = pd.to_numeric(row.iloc[1 + j], errors="coerce")
        records.append(rec)

    return pd.DataFrame(records).set_index("item")


# ─────────────────────────────────────────────────────────────────────────────
# 11. Trade – Direction (Imports + Exports combined)
# ─────────────────────────────────────────────────────────────────────────────

def load_trade_direction() -> pd.DataFrame:
    """Trade by direction (country-wise total)."""
    raw = pd.read_csv(_data("13_Direction.csv"), header=None, dtype=str)

    records = []
    for i in range(4, len(raw)):
        row = raw.iloc[i]
        country = str(row.iloc[1]).strip()
        if not country or country in ("nan", "NaN"):
            continue
        rec = {"country": country}
        for j, col in enumerate(["exports_2023_24_annual", "exports_2024_25_10m",
                                   "exports_2024_25_annual", "exports_2025_26_10m",
                                   "imports_2023_24_annual", "imports_2024_25_10m",
                                   "imports_2024_25_annual", "imports_2025_26_10m"]):
            rec[col] = pd.to_numeric(row.iloc[2 + j], errors="coerce")
        records.append(rec)

    return pd.DataFrame(records).set_index("country")


# ─────────────────────────────────────────────────────────────────────────────
# 12. Top Imports
# ─────────────────────────────────────────────────────────────────────────────

def load_top_imports() -> pd.DataFrame:
    """Top imports by commodity (Rs. in million)."""
    raw = pd.read_csv(_data("18_Top_M.csv"), header=None, dtype=str)

    records = []
    for i in range(4, len(raw)):
        row = raw.iloc[i]
        commodity = str(row.iloc[2]).strip()
        if not commodity or commodity in ("nan", "NaN"):
            continue
        rec = {"commodity": commodity}
        for j, col in enumerate(["annual_2023_24", "ten_month_2024_25",
                                   "annual_2024_25", "ten_month_2025_26"]):
            rec[col] = pd.to_numeric(row.iloc[3 + j], errors="coerce")
        records.append(rec)

    return pd.DataFrame(records).set_index("commodity")


# ─────────────────────────────────────────────────────────────────────────────
# 13. Top Exports
# ─────────────────────────────────────────────────────────────────────────────

def load_top_exports() -> pd.DataFrame:
    """Top exports by commodity (Rs. in million)."""
    raw = pd.read_csv(_data("14_Top_X.csv"), header=None, dtype=str)

    records = []
    for i in range(4, len(raw)):
        row = raw.iloc[i]
        commodity = str(row.iloc[2]).strip()
        if not commodity or commodity in ("nan", "NaN"):
            continue
        rec = {"commodity": commodity}
        for j, col in enumerate(["annual_2023_24", "ten_month_2024_25",
                                   "annual_2024_25", "ten_month_2025_26"]):
            rec[col] = pd.to_numeric(row.iloc[3 + j], errors="coerce")
        records.append(rec)

    return pd.DataFrame(records).set_index("commodity")


# ─────────────────────────────────────────────────────────────────────────────
# 14. India / China / Other Trade
# ─────────────────────────────────────────────────────────────────────────────

def _load_bilateral_trade(filename: str) -> pd.DataFrame:
    raw = pd.read_csv(_data(filename), header=None, dtype=str)
    records = []
    for i in range(4, len(raw)):
        row = raw.iloc[i]
        commodity = str(row.iloc[2]).strip()
        if not commodity or commodity in ("nan", "NaN"):
            continue
        rec = {"commodity": commodity}
        for j, col in enumerate(["annual_2023_24", "ten_month_2024_25",
                                   "annual_2024_25", "ten_month_2025_26"]):
            rec[col] = pd.to_numeric(row.iloc[3 + j], errors="coerce")
        records.append(rec)
    return pd.DataFrame(records).set_index("commodity")


def load_exports_india() -> pd.DataFrame:
    return _load_bilateral_trade("15_X-India.csv")


def load_exports_china() -> pd.DataFrame:
    return _load_bilateral_trade("16_X-China.csv")


def load_imports_india() -> pd.DataFrame:
    return _load_bilateral_trade("19_M-India.csv")


def load_imports_china() -> pd.DataFrame:
    return _load_bilateral_trade("20_M-China.csv")


# ─────────────────────────────────────────────────────────────────────────────
# 15. Balance of Payments (Cumulative)
# ─────────────────────────────────────────────────────────────────────────────

def load_bop() -> pd.DataFrame:
    """Balance of Payments cumulative data (BPM6)."""
    raw = pd.read_csv(_data("28_A__BoP_Cumulative.csv"), header=None, dtype=str)

    # Find the row with column headers (has "Credit", "Debit")
    col_header_idx = 3
    records = []
    for i in range(col_header_idx + 1, len(raw)):
        row = raw.iloc[i]
        sn = str(row.iloc[2]).strip()
        item = str(row.iloc[3]).strip()
        if not item or item in ("nan", "NaN"):
            continue
        rec = {"sn": sn, "item": item}
        for j, col in enumerate(["credit_2023_24", "debit_2023_24",
                                   "credit_2024_25r", "debit_2024_25r",
                                   "credit_2025_26p", "debit_2025_26p"]):
            rec[col] = pd.to_numeric(row.iloc[4 + j], errors="coerce")
        records.append(rec)

    return pd.DataFrame(records).set_index("item")


# ─────────────────────────────────────────────────────────────────────────────
# 16. Migrant Workers (Labor Permits)
# ─────────────────────────────────────────────────────────────────────────────

def load_migrant_workers() -> pd.DataFrame:
    """Country-wise labor permits for foreign employment."""
    raw = pd.read_csv(_data("26_Migrant_Worker.csv"), header=None, dtype=str)

    records = []
    for i in range(4, len(raw)):
        row = raw.iloc[i]
        country = str(row.iloc[2]).strip()
        if not country or country in ("nan", "NaN"):
            continue
        rec = {"country": country}
        for j, col in enumerate(["fy_2023_24", "fy_2024_25r", "fy_2025_26p"]):
            rec[col] = pd.to_numeric(row.iloc[3 + j], errors="coerce")
        records.append(rec)

    return pd.DataFrame(records).set_index("country")


# ─────────────────────────────────────────────────────────────────────────────
# 17. Tourist Arrivals (Monthly by Year)
# ─────────────────────────────────────────────────────────────────────────────

def load_tourist_arrivals() -> pd.DataFrame:
    """Monthly tourist arrivals from 2005 to present (month × year pivot)."""
    raw = pd.read_csv(_data("27_Tourist_Arrival.csv"), header=None, dtype=str)

    # Row 2 has years (col 2 onwards), Row 4+ has monthly data
    year_row = raw.iloc[2]
    years = [str(v).strip() for v in year_row.iloc[2:]
             if str(v).strip() not in ("nan", "NaN", "")]

    records = []
    for i in range(4, len(raw)):
        row = raw.iloc[i]
        month = str(row.iloc[1]).strip()
        if not month or month in ("nan", "NaN", "Month"):
            continue
        rec = {"month": month}
        for yr, val in zip(years, row.iloc[2: 2 + len(years)]):
            rec[yr] = pd.to_numeric(val, errors="coerce")
        records.append(rec)

    return pd.DataFrame(records).set_index("month") if records else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# 18. Share Market Indicators
# ─────────────────────────────────────────────────────────────────────────────

def load_share_market() -> pd.DataFrame:
    """NEPSE stock market key indicators."""
    raw = pd.read_csv(_data("58_Share_Market_Indicators.csv"), header=None, dtype=str)

    records = []
    for i in range(4, len(raw)):
        row = raw.iloc[i]
        item = str(row.iloc[0]).strip()
        if not item or item in ("nan", "NaN"):
            continue
        rec = {"indicator": item}
        for j, col in enumerate(["may_2024", "may_2025", "may_2026",
                                   "pct_change_1_to_2", "pct_change_2_to_3"]):
            rec[col] = pd.to_numeric(row.iloc[1 + j], errors="coerce")
        records.append(rec)

    return pd.DataFrame(records).set_index("indicator")


# ─────────────────────────────────────────────────────────────────────────────
# 19. Listed Companies & Market Capitalisation
# ─────────────────────────────────────────────────────────────────────────────

def load_listed_companies() -> pd.DataFrame:
    """Listed companies and market capitalisation by sector."""
    raw = pd.read_csv(_data("60_Listed_co.csv"), header=None, dtype=str)

    records = []
    for i in range(4, len(raw)):
        row = raw.iloc[i]
        sector = str(row.iloc[0]).strip()
        if not sector or sector in ("nan", "NaN"):
            continue
        rec = {"sector": sector}
        for j, col in enumerate(["listed_2024", "listed_2025", "listed_2026",
                                   "mktcap_2024", "mktcap_2025", "mktcap_2026"]):
            rec[col] = pd.to_numeric(row.iloc[1 + j], errors="coerce")
        records.append(rec)

    return pd.DataFrame(records).set_index("sector")


# ─────────────────────────────────────────────────────────────────────────────
# 20. Securities Market Turnover
# ─────────────────────────────────────────────────────────────────────────────

def load_turnover() -> pd.DataFrame:
    """Securities market turnover details."""
    raw = pd.read_csv(_data("62_Turnover_Details.csv"), header=None, dtype=str)

    records = []
    for i in range(4, len(raw)):
        row = raw.iloc[i]
        item = str(row.iloc[0]).strip()
        if not item or item in ("nan", "NaN"):
            continue
        rec = {"item": item}
        for j, col in enumerate(["units_2023_24", "value_2023_24", "pct_share_2023_24",
                                   "units_2024_25", "value_2024_25", "pct_share_2024_25",
                                   "units_2025_26", "value_2025_26", "pct_share_2025_26"]):
            rec[col] = pd.to_numeric(row.iloc[1 + j], errors="coerce")
        records.append(rec)

    return pd.DataFrame(records).set_index("item")


# ─────────────────────────────────────────────────────────────────────────────
# 21. Interbank Transaction Rates
# ─────────────────────────────────────────────────────────────────────────────

def load_interbank() -> pd.DataFrame:
    """Monthly interbank transaction amounts and weighted average interest rates."""
    raw = pd.read_csv(_data("56_Inter_bank.csv"), header=None, dtype=str)

    records = []
    for i in range(4, len(raw)):
        row = raw.iloc[i]
        period = str(row.iloc[0]).strip()
        if not period or period in ("nan", "NaN"):
            continue
        rec = {"period": period}
        for j, col in enumerate(["cb_amount_2024_25", "cb_wair_2024_25",
                                   "cb_amount_2025_26", "cb_wair_2025_26",
                                   "others_amount_2024_25", "others_wair_2024_25",
                                   "others_amount_2025_26", "others_wair_2025_26"]):
            rec[col] = pd.to_numeric(row.iloc[1 + j], errors="coerce")
        records.append(rec)

    return pd.DataFrame(records).set_index("period")


# ─────────────────────────────────────────────────────────────────────────────
# 22. Annual Migrant Worker Monthly Data
# ─────────────────────────────────────────────────────────────────────────────

def load_migrant_monthly() -> pd.DataFrame:
    """Monthly migrant worker labor permit data (longer series)."""
    raw = pd.read_csv(_data("A11_Migrant_Worker_Monthly.csv"), header=None, dtype=str)

    # Find period header row
    period_row = raw.iloc[2]
    periods = [str(v).strip() for v in period_row.iloc[1:]
               if str(v).strip() not in ("nan", "NaN", "")]

    records = []
    for i in range(3, len(raw)):
        row = raw.iloc[i]
        item = str(row.iloc[0]).strip()
        if not item or item in ("nan", "NaN"):
            continue
        values = [pd.to_numeric(row.iloc[j + 1], errors="coerce")
                  for j in range(len(periods))]
        if all(pd.isna(v) for v in values):
            continue
        rec = {"item": item}
        for p, val in zip(periods, values):
            rec[p] = val
        records.append(rec)

    return pd.DataFrame(records).set_index("item") if records else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# 23. Annual CPI Summary
# ─────────────────────────────────────────────────────────────────────────────

def load_cpi_annual() -> pd.DataFrame:
    """Annual CPI data across fiscal years."""
    raw = pd.read_csv(_data("A1_CPI_Annual.csv"), header=None, dtype=str)

    header_idx = 2
    period_row = raw.iloc[header_idx]
    periods = [str(v).strip() for v in period_row.iloc[2:]
               if str(v).strip() not in ("nan", "NaN", "")]

    records = []
    for i in range(header_idx + 1, len(raw)):
        row = raw.iloc[i]
        cat = str(row.iloc[1]).strip()
        if not cat or cat in ("nan", "NaN"):
            continue
        values = [pd.to_numeric(row.iloc[j + 2], errors="coerce")
                  for j in range(len(periods))]
        if all(pd.isna(v) for v in values):
            continue
        rec = {"category": cat}
        for p, val in zip(periods, values):
            rec[p] = val
        records.append(rec)

    return pd.DataFrame(records).set_index("category")


# ─────────────────────────────────────────────────────────────────────────────
# Master loader: load all datasets at once
# ─────────────────────────────────────────────────────────────────────────────

def load_all() -> dict:
    """
    Load all NRB datasets and return a dict of {name: DataFrame}.
    Any failed load is stored as empty DataFrame with a warning printed.
    """
    loaders = {
        "smi":               load_smi,
        "cpi_yoy":           load_cpi_yoy,
        "cpi_yoy_series":    load_cpi_yoy_series,
        "cpi_province":      load_cpi_province,
        "cpi_ecology":       load_cpi_ecology,
        "wpi_yoy":           load_wpi_yoy,
        "wpi_yoy_series":    load_wpi_yoy_series,
        "exchange_rate":     load_exchange_rate,
        "interest_rates":    load_interest_rates,
        "forex_reserves":    load_forex_reserves,
        "money_supply":      load_money_supply,
        "deposits":          load_deposits,
        "trade_direction":   load_trade_direction,
        "top_imports":       load_top_imports,
        "top_exports":       load_top_exports,
        "exports_india":     load_exports_india,
        "exports_china":     load_exports_china,
        "imports_india":     load_imports_india,
        "imports_china":     load_imports_china,
        "bop":               load_bop,
        "migrant_workers":   load_migrant_workers,
        "tourist_arrivals":  load_tourist_arrivals,
        "share_market":      load_share_market,
        "listed_companies":  load_listed_companies,
        "turnover":          load_turnover,
        "interbank":         load_interbank,
    }

    datasets = {}
    for name, fn in loaders.items():
        try:
            datasets[name] = fn()
        except Exception as e:
            print(f"[WARNING] Could not load '{name}': {e}")
            datasets[name] = pd.DataFrame()

    return datasets


if __name__ == "__main__":
    all_data = load_all()
    for k, v in all_data.items():
        shape = v.shape if isinstance(v, pd.DataFrame) else "N/A"
        print(f"  {k:25s} -> shape {shape}")

"""
Nepal Economic Dashboard — Streamlit App
=========================================
Multi-tab dashboard covering:
  1. Overview     — Key macro snapshot
  2. Inflation    — CPI/WPI trends, province, ecology
  3. Trade        — Imports, exports, trade balance
  4. Forex        — Reserves, exchange rate
  5. Banking      — Deposits, credit, interest rates, M2
  6. Stock Market — NEPSE indicators, listed cos, turnover
  7. External     — Remittances, tourists, migrants
  8. ML Forecast  — Model comparison + live prediction
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.data_loader import (
    load_cpi_yoy, load_wpi_yoy, load_cpi_province, load_cpi_ecology,
    load_trade_direction, load_top_imports, load_top_exports,
    load_forex_reserves, load_exchange_rate, load_deposits,
    load_interest_rates, load_money_supply, load_share_market,
    load_listed_companies, load_tourist_arrivals, load_migrant_workers,
    load_turnover,
)
from src.preprocessing import (
    build_cpi_series, build_wpi_series, build_exchange_rate_series,
    build_repo_rate_series, build_m2_series, build_feature_matrix,
    train_test_split_ts,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Nepal Economic Dashboard",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = px.colors.qualitative.Plotly
NRB_RED = "#c0392b"

# ─────────────────────────────────────────────────────────────────────────────
# Cached loaders (st.cache_data reuses results across reruns)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _cpi_series(): return build_cpi_series()

@st.cache_data(show_spinner=False)
def _wpi_series(): return build_wpi_series()

@st.cache_data(show_spinner=False)
def _fx_series(): return build_exchange_rate_series()

@st.cache_data(show_spinner=False)
def _repo_series(): return build_repo_rate_series()

@st.cache_data(show_spinner=False)
def _m2_series(): return build_m2_series()

@st.cache_data(show_spinner=False)
def _cpi_yoy_df(): return load_cpi_yoy()

@st.cache_data(show_spinner=False)
def _wpi_yoy_df(): return load_wpi_yoy()

@st.cache_data(show_spinner=False)
def _cpi_province_df(): return load_cpi_province()

@st.cache_data(show_spinner=False)
def _cpi_ecology_df(): return load_cpi_ecology()

@st.cache_data(show_spinner=False)
def _trade_dir_df(): return load_trade_direction()

@st.cache_data(show_spinner=False)
def _top_imports_df(): return load_top_imports()

@st.cache_data(show_spinner=False)
def _top_exports_df(): return load_top_exports()

@st.cache_data(show_spinner=False)
def _forex_df(): return load_forex_reserves()

@st.cache_data(show_spinner=False)
def _deposits_df(): return load_deposits()

@st.cache_data(show_spinner=False)
def _interest_df(): return load_interest_rates()

@st.cache_data(show_spinner=False)
def _ms_df(): return load_money_supply()

@st.cache_data(show_spinner=False)
def _share_mkt_df(): return load_share_market()

@st.cache_data(show_spinner=False)
def _listed_df(): return load_listed_companies()

@st.cache_data(show_spinner=False)
def _tourists_df(): return load_tourist_arrivals()

@st.cache_data(show_spinner=False)
def _migrants_df(): return load_migrant_workers()

@st.cache_data(show_spinner=False)
def _turnover_df(): return load_turnover()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _series_to_line_fig(series: pd.Series, title: str,
                         yaxis: str = "Value", color: str = NRB_RED) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series.index, y=series.values,
        mode="lines+markers",
        line=dict(color=color, width=2),
        marker=dict(size=5),
        name=series.name or "",
    ))
    fig.update_layout(title=title, xaxis_title="Date",
                       yaxis_title=yaxis, template="plotly_white",
                       height=380)
    return fig


def _kpi_card(col, label: str, value, delta=None, unit: str = ""):
    with col:
        if delta is not None:
            st.metric(label, f"{value:.2f}{unit}", delta=f"{delta:+.2f}{unit}")
        else:
            st.metric(label, f"{value:.2f}{unit}")


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/9/9b/Flag_of_Nepal.svg",
             width=60)
    st.title("Nepal Economic\nDashboard")
    st.caption("Data Source: Nepal Rastra Bank (NRB)")
    st.divider()
    tab_choice = st.radio(
        "Navigate",
        ["📊 Overview", "📈 Inflation", "🚢 Trade",
         "💱 Forex", "🏦 Banking", "📉 Stock Market",
         "✈️ External Sector", "🤖 ML Forecast"],
    )
    st.divider()
    st.caption("Built with ❤️ using Streamlit + Plotly")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Overview
# ─────────────────────────────────────────────────────────────────────────────

if tab_choice == "📊 Overview":
    st.title("🏔️ Nepal Economic Dashboard")
    st.markdown("**Key macroeconomic indicators at a glance — Nepal Rastra Bank data**")
    st.divider()

    cpi   = _cpi_series().dropna()
    wpi   = _wpi_series().dropna()
    fx    = _fx_series().dropna()
    repo  = _repo_series().dropna()
    m2    = _m2_series().dropna()

    # KPI Row
    k1, k2, k3, k4, k5 = st.columns(5)
    if len(cpi) >= 2:
        _kpi_card(k1, "CPI YoY (%)", cpi.iloc[-1], cpi.iloc[-1] - cpi.iloc[-2], "%")
    if len(wpi) >= 2:
        _kpi_card(k2, "WPI YoY (%)", wpi.iloc[-1], wpi.iloc[-1] - wpi.iloc[-2], "%")
    if len(fx) >= 2:
        _kpi_card(k3, "USD/NPR", fx.iloc[-1], fx.iloc[-1] - fx.iloc[-2])
    if len(repo) >= 1:
        _kpi_card(k4, "Repo Rate (%)", repo.iloc[-1], unit="%")
    if len(m2) >= 2:
        m2_b = m2.iloc[-1] / 1e6
        _kpi_card(k5, "M2 (Rs. Trillion)", m2_b, unit=" T")

    st.divider()

    # Mini charts
    c1, c2 = st.columns(2)
    with c1:
        if not cpi.empty:
            st.plotly_chart(
                _series_to_line_fig(cpi.tail(36), "CPI Inflation YoY (3-Year)", "% Change"),
                use_container_width=True
            )
    with c2:
        if not fx.empty:
            st.plotly_chart(
                _series_to_line_fig(fx.tail(36), "USD/NPR Exchange Rate", "NPR", color="#2980b9"),
                use_container_width=True
            )

    c3, c4 = st.columns(2)
    with c3:
        if not repo.empty:
            st.plotly_chart(
                _series_to_line_fig(repo.tail(60), "NRB Policy (Repo) Rate", "%", color="#27ae60"),
                use_container_width=True
            )
    with c4:
        if not m2.empty:
            m2_plot = (m2 / 1e6).rename("M2 (Rs. Trillion)")
            st.plotly_chart(
                _series_to_line_fig(m2_plot.tail(36), "M2 Money Supply", "Rs. Trillion", color="#8e44ad"),
                use_container_width=True
            )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Inflation
# ─────────────────────────────────────────────────────────────────────────────

elif tab_choice == "📈 Inflation":
    st.title("📈 Inflation Analysis")
    st.caption("CPI & WPI trends — national, province-wise, and ecology-wise")
    st.divider()

    subtab = st.selectbox("View", ["CPI YoY Trend", "WPI Trend",
                                    "Province-wise CPI", "Ecology-wise CPI",
                                    "CPI vs WPI Comparison"])

    if subtab == "CPI YoY Trend":
        cpi = _cpi_series().dropna()
        st.subheader("Overall CPI Year-over-Year (%)")
        fig = _series_to_line_fig(cpi, "Nepal CPI Inflation (YoY %)", "% Change")
        # Add NRB target band
        fig.add_hrect(y0=5, y1=7, fillcolor="green", opacity=0.07,
                       annotation_text="NRB Target Band (5–7%)")
        st.plotly_chart(fig, use_container_width=True)

        df_cpi = _cpi_yoy_df()
        if not df_cpi.empty:
            st.subheader("CPI by Category (Select rows from table)")
            cat = st.multiselect("Categories", df_cpi.index.tolist(),
                                  default=df_cpi.index.tolist()[:4])
            if cat:
                subset = df_cpi.loc[cat].T
                subset.index = [str(i) for i in subset.index]
                fig2 = px.line(subset, title="CPI YoY by Category",
                                labels={"value": "% Change", "index": "Period"},
                                template="plotly_white")
                st.plotly_chart(fig2, use_container_width=True)

    elif subtab == "WPI Trend":
        wpi = _wpi_series().dropna()
        st.subheader("Overall WPI Year-over-Year (%)")
        st.plotly_chart(
            _series_to_line_fig(wpi, "Nepal WPI Inflation (YoY %)", "% Change", color="#e67e22"),
            use_container_width=True
        )
        df_wpi = _wpi_yoy_df()
        if not df_wpi.empty:
            st.dataframe(df_wpi.replace({np.nan: None}), use_container_width=True)

    elif subtab == "Province-wise CPI":
        df = _cpi_province_df()
        st.subheader("Province-wise CPI")
        if not df.empty:
            # Take last few periods
            last_cols = [c for c in df.columns[-6:]]
            subset = df[last_cols].replace({np.nan: None})
            fig = px.bar(subset.T, barmode="group",
                          title="Province-wise CPI — Recent Periods",
                          labels={"value": "CPI Index", "index": "Period"},
                          template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(subset, use_container_width=True)

    elif subtab == "Ecology-wise CPI":
        df = _cpi_ecology_df()
        st.subheader("Ecology-wise CPI (Mountain / Hill / Terai)")
        if not df.empty:
            last_cols = [c for c in df.columns[-6:]]
            subset = df[last_cols].replace({np.nan: None})
            fig = px.line(subset.T, title="Ecology-wise CPI Trend",
                           labels={"value": "CPI Index", "index": "Period"},
                           template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(subset, use_container_width=True)

    elif subtab == "CPI vs WPI Comparison":
        cpi = _cpi_series().dropna()
        wpi = _wpi_series().dropna()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cpi.index, y=cpi.values, name="CPI YoY",
                                  line=dict(color=NRB_RED, width=2)))
        fig.add_trace(go.Scatter(x=wpi.index, y=wpi.values, name="WPI YoY",
                                  line=dict(color="#e67e22", width=2, dash="dot")))
        fig.update_layout(title="CPI vs WPI Year-over-Year Comparison",
                           yaxis_title="% Change", template="plotly_white", height=400)
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Trade
# ─────────────────────────────────────────────────────────────────────────────

elif tab_choice == "🚢 Trade":
    st.title("🚢 Trade Analysis")
    st.caption("Imports, Exports, Trade Balance — Nepal Rastra Bank")
    st.divider()

    subtab = st.selectbox("View", ["Country-wise Trade",
                                    "Top Imports", "Top Exports",
                                    "Trade Balance"])

    if subtab == "Country-wise Trade":
        df = _trade_dir_df()
        if not df.empty:
            st.subheader("Trade Direction (Country-wise)")
            st.dataframe(df.replace({np.nan: None}), use_container_width=True)

            # Bar chart for top partners
            if "exports_2025_26_10m" in df.columns and "imports_2025_26_10m" in df.columns:
                top = df[["exports_2025_26_10m", "imports_2025_26_10m"]].dropna().head(15)
                top.columns = ["Exports", "Imports"]
                fig = px.bar(top, barmode="group",
                              title="Top Trading Partners (2025/26 — 10 months)",
                              labels={"value": "Rs. million", "index": "Country"},
                              template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

    elif subtab == "Top Imports":
        df = _top_imports_df()
        if not df.empty:
            st.subheader("Top Import Commodities (Rs. million)")
            fig = px.bar(df.dropna(subset=["ten_month_2025_26"]).head(15),
                          x="ten_month_2025_26", y=df.dropna(subset=["ten_month_2025_26"]).index[:15],
                          orientation="h",
                          title="Top Imports — 2025/26 (10 months)",
                          labels={"ten_month_2025_26": "Rs. million"},
                          template="plotly_white")
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df.replace({np.nan: None}), use_container_width=True)

    elif subtab == "Top Exports":
        df = _top_exports_df()
        if not df.empty:
            st.subheader("Top Export Commodities (Rs. million)")
            fig = px.bar(df.dropna(subset=["ten_month_2025_26"]).head(15),
                          x="ten_month_2025_26", y=df.dropna(subset=["ten_month_2025_26"]).index[:15],
                          orientation="h",
                          title="Top Exports — 2025/26 (10 months)",
                          labels={"ten_month_2025_26": "Rs. million"},
                          template="plotly_white")
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df.replace({np.nan: None}), use_container_width=True)

    elif subtab == "Trade Balance":
        df_imp = _top_imports_df()
        df_exp = _top_exports_df()
        periods = ["annual_2023_24", "ten_month_2024_25", "annual_2024_25", "ten_month_2025_26"]
        labels = ["FY2023/24 Annual", "FY2024/25 (10M)", "FY2024/25 Annual", "FY2025/26 (10M)"]
        totals_imp = [df_imp[p].sum() for p in periods if p in df_imp.columns]
        totals_exp = [df_exp[p].sum() for p in periods if p in df_exp.columns]
        n = min(len(totals_imp), len(totals_exp))
        balance = [e - i for e, i in zip(totals_exp[:n], totals_imp[:n])]

        fig = go.Figure()
        fig.add_bar(x=labels[:n], y=totals_imp[:n], name="Imports",
                     marker_color="#e74c3c")
        fig.add_bar(x=labels[:n], y=totals_exp[:n], name="Exports",
                     marker_color="#27ae60")
        fig.add_trace(go.Scatter(x=labels[:n], y=balance, name="Trade Balance",
                                  mode="lines+markers",
                                  line=dict(color="#2c3e50", width=3, dash="dot")))
        fig.update_layout(barmode="group", title="Trade Balance Summary",
                           yaxis_title="Rs. million", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Forex
# ─────────────────────────────────────────────────────────────────────────────

elif tab_choice == "💱 Forex":
    st.title("💱 Foreign Exchange")
    st.divider()

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("USD/NPR Exchange Rate")
        fx = _fx_series().dropna()
        if not fx.empty:
            fig = _series_to_line_fig(fx, "USD/NPR Monthly Rate", "NPR", color="#2980b9")
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Gross Forex Reserves")
        df = _forex_df()
        if not df.empty:
            st.dataframe(df.replace({np.nan: None}), use_container_width=True)

    st.subheader("Detailed Exchange Rate History")
    df_raw = load_exchange_rate()
    if not df_raw.empty:
        st.dataframe(df_raw.replace({np.nan: None}), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Banking
# ─────────────────────────────────────────────────────────────────────────────

elif tab_choice == "🏦 Banking":
    st.title("🏦 Banking Sector")
    st.divider()

    subtab = st.selectbox("View", ["Interest Rates", "Money Supply (M2)",
                                    "Deposits", "Monetary Survey"])

    if subtab == "Interest Rates":
        repo = _repo_series().dropna()
        if not repo.empty:
            st.subheader("NRB Policy (Repo) Rate — Monthly")
            st.plotly_chart(
                _series_to_line_fig(repo, "Fixed Repo Rate (%)", "% p.a.", color="#27ae60"),
                use_container_width=True
            )
        df = _interest_df()
        if not df.empty:
            st.subheader("Full Interest Rate Structure")
            # Show last 12 columns only for readability
            last_cols = df.columns[-12:]
            st.dataframe(df[last_cols].replace({np.nan: None}), use_container_width=True)

    elif subtab == "Money Supply (M2)":
        m2 = _m2_series().dropna()
        if not m2.empty:
            fig = _series_to_line_fig((m2 / 1e6).rename("M2"), "M2 Money Supply",
                                       "Rs. Trillion", color="#8e44ad")
            st.plotly_chart(fig, use_container_width=True)
        df = _ms_df()
        if not df.empty:
            st.subheader("Monetary Survey Table")
            last_cols = df.columns[-6:]
            st.dataframe(df[last_cols].replace({np.nan: None}), use_container_width=True)

    elif subtab == "Deposits":
        df = _deposits_df()
        if not df.empty:
            st.subheader("Deposit Details — Banks & Financial Institutions")
            st.dataframe(df.replace({np.nan: None}), use_container_width=True)
            # Simple bar
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            if numeric_cols:
                col = st.selectbox("Select column", numeric_cols)
                subset = df[col].dropna()
                fig = px.bar(x=subset.index, y=subset.values,
                              title=f"Deposits — {col}", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

    elif subtab == "Monetary Survey":
        df = _ms_df()
        if not df.empty:
            last_cols = df.columns[-8:]
            subset = df[last_cols].replace({np.nan: None})
            st.subheader("Monetary Survey — Recent Data")
            fig = px.bar(subset.T, barmode="group",
                          title="Monetary Aggregates Trend",
                          labels={"value": "Rs. million", "index": "Period"},
                          template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(subset, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Stock Market
# ─────────────────────────────────────────────────────────────────────────────

elif tab_choice == "📉 Stock Market":
    st.title("📉 NEPSE Stock Market")
    st.divider()

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Key Market Indicators")
        df = _share_mkt_df()
        if not df.empty:
            st.dataframe(df.replace({np.nan: None}), use_container_width=True)
            numeric = df.select_dtypes(include="number")
            if not numeric.empty:
                fig = px.bar(numeric, barmode="group",
                              title="NEPSE Indicators Comparison",
                              template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Listed Companies & Market Cap")
        df2 = _listed_df()
        if not df2.empty:
            st.dataframe(df2.replace({np.nan: None}), use_container_width=True)

    st.subheader("Securities Market Turnover")
    df3 = _turnover_df()
    if not df3.empty:
        st.dataframe(df3.replace({np.nan: None}), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# 7. External Sector
# ─────────────────────────────────────────────────────────────────────────────

elif tab_choice == "✈️ External Sector":
    st.title("✈️ External Sector")
    st.divider()

    subtab = st.selectbox("View", ["Tourist Arrivals", "Migrant Workers"])

    if subtab == "Tourist Arrivals":
        df = _tourists_df()
        if not df.empty:
            st.subheader("Monthly Tourist Arrivals by Year")
            # Melt for plotly
            df_plot = df.copy()
            df_plot.index.name = "Month"
            df_long = df_plot.reset_index().melt(id_vars="Month",
                                                   var_name="Year",
                                                   value_name="Arrivals")
            df_long["Arrivals"] = pd.to_numeric(df_long["Arrivals"], errors="coerce")
            # Recent 5 years
            recent_years = sorted(df_long["Year"].unique())[-5:]
            df_recent = df_long[df_long["Year"].isin(recent_years)]

            fig = px.line(df_recent, x="Month", y="Arrivals", color="Year",
                           title="Tourist Arrivals — Last 5 Years",
                           template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

            # Annual total trend
            annuals = df_long.groupby("Year")["Arrivals"].sum().dropna()
            annuals = annuals[pd.to_numeric(annuals.index, errors="coerce").notna()]
            fig2 = px.bar(x=annuals.index, y=annuals.values,
                           title="Annual Tourist Arrivals",
                           labels={"x": "Year", "y": "Total Arrivals"},
                           template="plotly_white")
            st.plotly_chart(fig2, use_container_width=True)

    elif subtab == "Migrant Workers":
        df = _migrants_df()
        if not df.empty:
            st.subheader("Nepalese Labour Permits — Top Destination Countries")
            latest_col = df.columns[-1]
            top15 = df[latest_col].dropna().sort_values(ascending=False).head(15)
            fig = px.bar(x=top15.values, y=top15.index,
                          orientation="h",
                          title=f"Labor Permits ({latest_col})",
                          labels={"x": "Number of Permits", "y": "Country"},
                          template="plotly_white")
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df.replace({np.nan: None}), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# 8. ML Forecast
# ─────────────────────────────────────────────────────────────────────────────

elif tab_choice == "🤖 ML Forecast":
    st.title("🤖 ML-Based CPI Inflation Forecast")
    st.caption("Trained on NRB macroeconomic time-series data")
    st.divider()

    # ── Train button ──────────────────────────────────────────────────────────
    if "model_results" not in st.session_state:
        st.session_state["model_results"] = None

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        train_btn = st.button("🚀 Train / Retrain All Models", type="primary")
    with col_info:
        st.info("Trains Ridge, Random Forest, XGBoost, LightGBM, "
                "Gradient Boosting, Prophet, and ARIMA on NRB data.")

    if train_btn:
        with st.spinner("Training models — please wait…"):
            try:
                from src.preprocessing import build_feature_matrix, train_test_split_ts, build_cpi_series
                from src.models import train_all, compare_models

                df = build_feature_matrix()
                X_tr, X_te, y_tr, y_te = train_test_split_ts(df)
                cpi_s = build_cpi_series()
                results = train_all(X_tr, X_te, y_tr, y_te, cpi_s)
                comparison = compare_models(results)

                st.session_state["model_results"] = results
                st.session_state["model_comparison"] = comparison
                st.session_state["X_test"] = X_te
                st.session_state["y_test"] = y_te
                st.success("✅ All models trained successfully!")
            except Exception as e:
                st.error(f"Training failed: {e}")

    # ── Results ───────────────────────────────────────────────────────────────
    if st.session_state.get("model_results"):
        results = st.session_state["model_results"]
        comparison = st.session_state["model_comparison"]
        X_te = st.session_state["X_test"]
        y_te = st.session_state["y_test"]

        st.subheader("📊 Model Comparison")
        st.dataframe(comparison.set_index("Model").style.highlight_min(
            subset=["MAE", "RMSE", "MAPE"], color="#d5f5e3"
        ).highlight_max(subset=["R2"], color="#d5f5e3"),
            use_container_width=True)

        # Actual vs Predicted chart
        st.subheader("📈 Actual vs. Predicted — Test Set")
        fig_avp = go.Figure()
        fig_avp.add_trace(go.Scatter(x=X_te.index, y=y_te.values,
                                      name="Actual", line=dict(color="black", width=2)))
        colors_list = COLORS
        for i, (key, res) in enumerate(results.items()):
            if "predictions" in res:
                fig_avp.add_trace(go.Scatter(
                    x=X_te.index, y=res["predictions"],
                    name=res.get("name", key),
                    line=dict(dash="dot", color=colors_list[i % len(colors_list)])
                ))
        fig_avp.update_layout(xaxis_title="Date", yaxis_title="CPI YoY (%)",
                               template="plotly_white", height=400)
        st.plotly_chart(fig_avp, use_container_width=True)

        # Feature importance
        st.subheader("🔍 Feature Importance")
        imp_models = {k: v for k, v in results.items() if "feature_importance" in v}
        if imp_models:
            sel_model = st.selectbox("Select Model", list(imp_models.keys()))
            imp_dict = imp_models[sel_model]["feature_importance"]
            items = sorted(imp_dict.items(), key=lambda x: x[1], reverse=True)[:12]
            fig_imp = px.bar(x=[v for _, v in items], y=[k for k, _ in items],
                              orientation="h", template="plotly_white",
                              title=f"Feature Importance — {results[sel_model]['name']}",
                              labels={"x": "Importance", "y": "Feature"})
            fig_imp.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_imp, use_container_width=True)

        # Prophet forecast
        if "prophet" in results and "forecast" in results["prophet"]:
            st.subheader("🔮 Prophet 6-Month Forecast")
            fc = results["prophet"]["forecast"]
            fig_fc = go.Figure()
            fig_fc.add_trace(go.Scatter(x=fc["ds"], y=fc["yhat"], name="Forecast",
                                         line=dict(color=NRB_RED, width=2)))
            if "yhat_lower" in fc.columns and "yhat_upper" in fc.columns:
                fig_fc.add_trace(go.Scatter(
                    x=pd.concat([fc["ds"], fc["ds"][::-1]]),
                    y=pd.concat([fc["yhat_upper"], fc["yhat_lower"][::-1]]),
                    fill="toself", fillcolor="rgba(192,57,43,0.15)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name="Confidence Interval"
                ))
            fig_fc.update_layout(title="Prophet — CPI Inflation Forecast",
                                   yaxis_title="CPI YoY (%)", template="plotly_white")
            st.plotly_chart(fig_fc, use_container_width=True)

    # ── Live prediction form ──────────────────────────────────────────────────
    st.divider()
    st.subheader("🎯 Live Prediction")
    st.caption("Enter current macro values to get an inflation forecast")

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            wpi     = st.number_input("WPI YoY (%)", value=5.0, step=0.1)
            usd_npr = st.number_input("USD/NPR Rate", value=134.0, step=0.5)
        with c2:
            repo    = st.number_input("Repo Rate (%)", value=5.0, step=0.25)
            m2_val  = st.number_input("M2 (Rs. million)", value=6500000.0, step=50000.0)
        with c3:
            lag1  = st.number_input("CPI lag 1m (%)", value=6.5, step=0.1)
            lag3  = st.number_input("CPI lag 3m (%)", value=6.8, step=0.1)
            lag6  = st.number_input("CPI lag 6m (%)", value=7.0, step=0.1)
            lag12 = st.number_input("CPI lag 12m (%)", value=7.5, step=0.1)

        submitted = st.form_submit_button("Predict CPI YoY", type="primary")

    if submitted and st.session_state.get("model_results"):
        import joblib
        from pathlib import Path

        MODELS_DIR = Path(__file__).parent.parent / "models"
        predictions_out = {}
        feature_data = {
            "wpi_yoy": wpi, "usd_npr": usd_npr, "repo_rate": repo,
            "m2": m2_val, "m2_yoy": 0.0, "fx_change": 0.0,
            "cpi_lag_1": lag1, "cpi_lag_3": lag3,
            "cpi_lag_6": lag6, "cpi_lag_12": lag12,
        }
        X_in = pd.DataFrame([feature_data])
        for model_name in ["ridge", "random_forest", "xgboost", "lightgbm"]:
            mp = MODELS_DIR / f"{model_name}.pkl"
            if mp.exists():
                try:
                    m = joblib.load(mp)
                    # Align columns to training columns
                    try:
                        df_feat = build_feature_matrix()
                        feat_cols = [c for c in df_feat.columns if c != "cpi_yoy"]
                        X_aligned = X_in.reindex(columns=feat_cols, fill_value=0)
                    except Exception:
                        X_aligned = X_in.fillna(0)
                    pred = float(m.predict(X_aligned)[0])
                    predictions_out[model_name] = pred
                except Exception:
                    pass

        if predictions_out:
            st.success("**CPI Inflation Predictions:**")
            out_cols = st.columns(len(predictions_out))
            for col, (name, val) in zip(out_cols, predictions_out.items()):
                col.metric(name.replace("_", " ").title(), f"{val:.2f}%")
    elif submitted:
        st.warning("Train models first by clicking '🚀 Train / Retrain All Models'")

"""
Nepal Economic Dashboard — FastAPI Backend
==========================================
Endpoints:
  GET  /                          Health check
  GET  /api/v1/inflation/cpi      CPI YoY time series
  GET  /api/v1/inflation/wpi      WPI YoY time series
  GET  /api/v1/inflation/province CPI by province
  GET  /api/v1/inflation/ecology  CPI by ecology
  GET  /api/v1/trade/summary      Trade direction summary
  GET  /api/v1/trade/imports      Top imports
  GET  /api/v1/trade/exports      Top exports
  GET  /api/v1/forex/reserves     Forex reserves
  GET  /api/v1/forex/rate         USD/NPR exchange rate series
  GET  /api/v1/banking/deposits   Deposit data
  GET  /api/v1/banking/rates      Interest rate series
  GET  /api/v1/banking/money      Money supply (M2)
  GET  /api/v1/stock/indicators   Share market indicators
  GET  /api/v1/stock/companies    Listed companies
  GET  /api/v1/external/tourists  Tourist arrivals
  GET  /api/v1/external/migrants  Migrant workers
  GET  /api/v1/models/compare     Model metrics comparison table
  POST /api/v1/predict/{model}    Predict CPI using a trained model
"""

import sys
import os
from pathlib import Path

# Allow importing from parent src/
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import joblib

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.data_loader import (
    load_cpi_yoy, load_wpi_yoy, load_cpi_province, load_cpi_ecology,
    load_trade_direction, load_top_imports, load_top_exports,
    load_forex_reserves, load_exchange_rate, load_deposits,
    load_interest_rates, load_money_supply, load_share_market,
    load_listed_companies, load_tourist_arrivals, load_migrant_workers,
)
from src.preprocessing import (
    build_feature_matrix, train_test_split_ts, build_cpi_series
)
from src.models import train_all, compare_models
from api.schemas import PredictRequest, PredictResponse, MetricsResponse

app = FastAPI(
    title="Nepal Economic Dashboard API",
    description="Macroeconomic indicators & ML forecasting from NRB data.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = Path(__file__).parent.parent / "models"


# ─────────────────────────────────────────────────────────────────────────────
# Cache for trained models (loaded lazily on first request)
# ─────────────────────────────────────────────────────────────────────────────

_model_cache: dict = {}
_metrics_cache: dict = {}


def _get_model(name: str):
    if name not in _model_cache:
        model_path = MODELS_DIR / f"{name}.pkl"
        if not model_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Model '{name}' not trained yet. "
                       "Run POST /api/v1/models/train first."
            )
        _model_cache[name] = joblib.load(model_path)
    return _model_cache[name]


def _df_to_records(df: pd.DataFrame) -> list:
    """Convert DataFrame to list of dicts, replacing NaN with None."""
    return df.reset_index().replace({np.nan: None}).to_dict(orient="records")


# ─────────────────────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "service": "Nepal Economic Dashboard API"}


# ─────────────────────────────────────────────────────────────────────────────
# Inflation
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/v1/inflation/cpi", tags=["Inflation"])
def get_cpi_yoy():
    """CPI Year-over-Year percentage change by category."""
    try:
        df = load_cpi_yoy()
        return {"data": _df_to_records(df)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/inflation/wpi", tags=["Inflation"])
def get_wpi_yoy():
    """WPI Year-over-Year percentage change."""
    try:
        df = load_wpi_yoy()
        return {"data": _df_to_records(df)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/inflation/province", tags=["Inflation"])
def get_cpi_province():
    """CPI by province."""
    try:
        df = load_cpi_province()
        return {"data": _df_to_records(df)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/inflation/ecology", tags=["Inflation"])
def get_cpi_ecology():
    """CPI by ecological zone (Mountain / Hill / Terai)."""
    try:
        df = load_cpi_ecology()
        return {"data": _df_to_records(df)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Trade
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/v1/trade/summary", tags=["Trade"])
def get_trade_direction():
    """Trade by direction (country-wise imports + exports)."""
    try:
        df = load_trade_direction()
        return {"data": _df_to_records(df)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/trade/imports", tags=["Trade"])
def get_top_imports():
    """Top imports by commodity."""
    try:
        df = load_top_imports()
        return {"data": _df_to_records(df)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/trade/exports", tags=["Trade"])
def get_top_exports():
    """Top exports by commodity."""
    try:
        df = load_top_exports()
        return {"data": _df_to_records(df)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Forex
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/v1/forex/reserves", tags=["Forex"])
def get_forex_reserves():
    """Gross foreign assets of the banking sector."""
    try:
        df = load_forex_reserves()
        return {"data": _df_to_records(df)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/forex/rate", tags=["Forex"])
def get_exchange_rate():
    """Monthly USD/NPR exchange rate."""
    try:
        df = load_exchange_rate()
        return {"data": df.replace({np.nan: None}).to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Banking
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/v1/banking/deposits", tags=["Banking"])
def get_deposits():
    """Deposit details of banks and financial institutions."""
    try:
        df = load_deposits()
        return {"data": _df_to_records(df)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/banking/rates", tags=["Banking"])
def get_interest_rates():
    """Structure of interest rates (policy rates + market rates)."""
    try:
        df = load_interest_rates()
        return {"data": _df_to_records(df)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/banking/money", tags=["Banking"])
def get_money_supply():
    """Monetary survey — M1, M2, deposits, credit."""
    try:
        df = load_money_supply()
        return {"data": _df_to_records(df)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Stock Market
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/v1/stock/indicators", tags=["Stock Market"])
def get_share_market():
    """NEPSE stock market indicators."""
    try:
        df = load_share_market()
        return {"data": _df_to_records(df)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/stock/companies", tags=["Stock Market"])
def get_listed_companies():
    """Listed companies and market capitalisation."""
    try:
        df = load_listed_companies()
        return {"data": _df_to_records(df)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# External Sector
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/v1/external/tourists", tags=["External Sector"])
def get_tourists():
    """Monthly tourist arrivals by year."""
    try:
        df = load_tourist_arrivals()
        return {"data": _df_to_records(df)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/external/migrants", tags=["External Sector"])
def get_migrants():
    """Country-wise Nepalese labour permits for foreign employment."""
    try:
        df = load_migrant_workers()
        return {"data": _df_to_records(df)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# ML Models
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/v1/models/train", tags=["ML Models"])
def train_models():
    """
    Train all ML models on the latest NRB data.
    This may take 30-60 seconds.
    """
    global _model_cache, _metrics_cache
    try:
        df = build_feature_matrix()
        X_train, X_test, y_train, y_test = train_test_split_ts(df)
        cpi_series = build_cpi_series()

        results = train_all(X_train, X_test, y_train, y_test, cpi_series)
        comparison = compare_models(results)

        _metrics_cache = results
        _model_cache = {}  # reload from disk on next predict

        return {
            "status": "trained",
            "samples": {"train": len(X_train), "test": len(X_test)},
            "comparison": comparison.replace({np.nan: None}).to_dict(orient="records"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/models/compare", tags=["ML Models"], response_model=list[MetricsResponse])
def get_model_comparison():
    """Return the latest model metrics comparison."""
    if not _metrics_cache:
        raise HTTPException(
            status_code=404,
            detail="No trained models found. Call POST /api/v1/models/train first."
        )
    rows = []
    for key, res in _metrics_cache.items():
        if "metrics" in res:
            rows.append(MetricsResponse(
                model=res.get("name", key),
                **res["metrics"]
            ))
    return rows


@app.post("/api/v1/predict/{model_name}",
          tags=["ML Models"],
          response_model=PredictResponse)
def predict(model_name: str, request: PredictRequest):
    """
    Predict CPI YoY using a trained model.
    Available models: ridge, random_forest, xgboost, lightgbm, gradient_boosting
    """
    valid_models = {"ridge", "random_forest", "xgboost", "lightgbm", "gradient_boosting"}
    if model_name not in valid_models:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model. Choose from: {valid_models}"
        )

    model = _get_model(model_name)

    feature_data = request.model_dump()
    feature_df = pd.DataFrame([feature_data])

    # Load a sample feature matrix to align column order
    try:
        df = build_feature_matrix()
        feature_cols = [c for c in df.columns if c != "cpi_yoy"]
        for col in feature_cols:
            if col not in feature_df.columns:
                feature_df[col] = df[col].median()
        feature_df = feature_df[feature_cols].fillna(0)
    except Exception:
        feature_df = feature_df.fillna(0)

    pred = float(model.predict(feature_df)[0])

    return PredictResponse(
        model=model_name,
        prediction=round(pred, 4),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Run directly: uvicorn api.main:app --reload
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)

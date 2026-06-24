"""
ML Models for Nepal CPI Inflation Forecasting
==============================================
Trains and evaluates multiple models:
  - Ridge Regression (baseline)
  - Random Forest
  - XGBoost
  - LightGBM
  - Prophet (univariate time-series)
  - ARIMA (statsmodels)

Each model returns a standard result dict: {name, predictions, metrics, model_obj}
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4),
            "R2": round(r2, 4), "MAPE": round(mape, 4)}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Ridge Regression
# ─────────────────────────────────────────────────────────────────────────────

def train_ridge(X_train, y_train, X_test, y_test, alpha: float = 1.0) -> dict:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=alpha)),
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    joblib.dump(pipe, MODELS_DIR / "ridge.pkl")
    return {
        "name": "Ridge Regression",
        "predictions": preds,
        "metrics": compute_metrics(y_test.values, preds),
        "model": pipe,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Random Forest
# ─────────────────────────────────────────────────────────────────────────────

def train_random_forest(X_train, y_train, X_test, y_test,
                         n_estimators: int = 200, max_depth: int = 6) -> dict:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(n_estimators=n_estimators,
                                         max_depth=max_depth,
                                         random_state=42,
                                         n_jobs=-1)),
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    importances = dict(zip(X_train.columns,
                           pipe.named_steps["model"].feature_importances_))
    joblib.dump(pipe, MODELS_DIR / "random_forest.pkl")
    return {
        "name": "Random Forest",
        "predictions": preds,
        "metrics": compute_metrics(y_test.values, preds),
        "model": pipe,
        "feature_importance": importances,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. XGBoost
# ─────────────────────────────────────────────────────────────────────────────

def train_xgboost(X_train, y_train, X_test, y_test,
                   n_estimators: int = 300, learning_rate: float = 0.05,
                   max_depth: int = 4) -> dict:
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)
    preds = model.predict(X_test)
    importances = dict(zip(X_train.columns, model.feature_importances_))
    joblib.dump(model, MODELS_DIR / "xgboost.pkl")
    return {
        "name": "XGBoost",
        "predictions": preds,
        "metrics": compute_metrics(y_test.values, preds),
        "model": model,
        "feature_importance": importances,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. LightGBM
# ─────────────────────────────────────────────────────────────────────────────

def train_lightgbm(X_train, y_train, X_test, y_test,
                    n_estimators: int = 300, learning_rate: float = 0.05,
                    max_depth: int = 4) -> dict:
    model = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    importances = dict(zip(X_train.columns, model.feature_importances_))
    joblib.dump(model, MODELS_DIR / "lightgbm.pkl")
    return {
        "name": "LightGBM",
        "predictions": preds,
        "metrics": compute_metrics(y_test.values, preds),
        "model": model,
        "feature_importance": importances,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Gradient Boosting (sklearn)
# ─────────────────────────────────────────────────────────────────────────────

def train_gradient_boosting(X_train, y_train, X_test, y_test) -> dict:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingRegressor(n_estimators=200,
                                              learning_rate=0.05,
                                              max_depth=4,
                                              random_state=42)),
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    importances = dict(zip(X_train.columns,
                           pipe.named_steps["model"].feature_importances_))
    joblib.dump(pipe, MODELS_DIR / "gradient_boosting.pkl")
    return {
        "name": "Gradient Boosting",
        "predictions": preds,
        "metrics": compute_metrics(y_test.values, preds),
        "model": pipe,
        "feature_importance": importances,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. Prophet (univariate)
# ─────────────────────────────────────────────────────────────────────────────

def train_prophet(cpi_series: pd.Series, periods: int = 6) -> dict:
    """
    Fits Prophet on the full CPI YoY series.
    Returns forecast for `periods` months ahead.
    """
    try:
        from prophet import Prophet
    except ImportError:
        return {"name": "Prophet", "error": "prophet not installed"}

    df_p = pd.DataFrame({"ds": cpi_series.index, "y": cpi_series.values})
    df_p = df_p.dropna()

    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="additive",
    )
    m.fit(df_p)

    future = m.make_future_dataframe(periods=periods, freq="MS")
    forecast = m.predict(future)

    in_sample = forecast[forecast["ds"].isin(df_p["ds"])][["ds", "yhat"]].set_index("ds")
    actual = df_p.set_index("ds")["y"]
    common = actual.index.intersection(in_sample.index)
    metrics = compute_metrics(actual.loc[common].values,
                               in_sample.loc[common, "yhat"].values)

    return {
        "name": "Prophet",
        "forecast": forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods + 6),
        "metrics": metrics,
        "model": m,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. ARIMA (statsmodels auto-order p=1, d=1, q=1)
# ─────────────────────────────────────────────────────────────────────────────

def train_arima(cpi_series: pd.Series, order: tuple = (2, 1, 1)) -> dict:
    """Fits ARIMA on CPI YoY series. Returns in-sample fit + forecast."""
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError:
        return {"name": "ARIMA", "error": "statsmodels not installed"}

    series = cpi_series.dropna()
    model = ARIMA(series, order=order)
    result = model.fit()

    preds_in = result.fittedvalues
    metrics = compute_metrics(series.values[order[1]:], preds_in.values[order[1]:])

    forecast = result.forecast(steps=6)

    return {
        "name": f"ARIMA{order}",
        "fitted": preds_in,
        "forecast": forecast,
        "metrics": metrics,
        "model": result,
        "summary": result.summary().as_text(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward cross-validation
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward_cv(X: pd.DataFrame, y: pd.Series,
                     model_fn, n_splits: int = 5,
                     min_train: int = 12) -> list[dict]:
    """
    Walk-forward (expanding window) cross-validation for time-series.
    Returns list of {fold, train_size, metrics}.
    """
    n = len(X)
    step = max(1, (n - min_train) // n_splits)
    results = []

    for fold in range(n_splits):
        split = min_train + fold * step
        if split >= n:
            break
        X_tr, X_te = X.iloc[:split], X.iloc[split: split + step]
        y_tr, y_te = y.iloc[:split], y.iloc[split: split + step]

        if len(X_te) == 0:
            break

        try:
            result = model_fn(X_tr, y_tr, X_te, y_te)
            results.append({
                "fold": fold + 1,
                "train_size": split,
                "test_size": len(X_te),
                **result["metrics"],
            })
        except Exception as e:
            results.append({"fold": fold + 1, "error": str(e)})

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Train all models at once
# ─────────────────────────────────────────────────────────────────────────────

def train_all(X_train, X_test, y_train, y_test,
               cpi_series: pd.Series = None) -> dict:
    """
    Train every model and return a dict of results.
    """
    results = {}

    print("Training Ridge...")
    results["ridge"] = train_ridge(X_train, y_train, X_test, y_test)

    print("Training Random Forest...")
    results["random_forest"] = train_random_forest(X_train, y_train, X_test, y_test)

    print("Training XGBoost...")
    results["xgboost"] = train_xgboost(X_train, y_train, X_test, y_test)

    print("Training LightGBM...")
    results["lightgbm"] = train_lightgbm(X_train, y_train, X_test, y_test)

    print("Training Gradient Boosting...")
    results["gradient_boosting"] = train_gradient_boosting(X_train, y_train, X_test, y_test)

    if cpi_series is not None:
        print("Training Prophet...")
        results["prophet"] = train_prophet(cpi_series)

        print("Training ARIMA...")
        results["arima"] = train_arima(cpi_series)

    return results


def compare_models(results: dict) -> pd.DataFrame:
    """Build a comparison table of all model metrics."""
    rows = []
    for key, res in results.items():
        if "metrics" in res:
            row = {"Model": res.get("name", key)}
            row.update(res["metrics"])
            rows.append(row)
    return pd.DataFrame(rows).sort_values("RMSE")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.preprocessing import build_feature_matrix, train_test_split_ts, build_cpi_series

    df = build_feature_matrix()
    X_train, X_test, y_train, y_test = train_test_split_ts(df)
    cpi_series = build_cpi_series()

    results = train_all(X_train, X_test, y_train, y_test, cpi_series)
    comparison = compare_models(results)
    print("\n=== Model Comparison ===")
    print(comparison.to_string(index=False))

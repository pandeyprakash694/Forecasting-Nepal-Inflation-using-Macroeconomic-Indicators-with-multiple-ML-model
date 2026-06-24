"""
Pydantic schemas for the Nepal Economic Dashboard API.
"""

from pydantic import BaseModel
from typing import Optional


class PredictRequest(BaseModel):
    """Input features for CPI inflation prediction."""
    wpi_yoy: Optional[float] = None
    usd_npr: Optional[float] = None
    repo_rate: Optional[float] = None
    m2: Optional[float] = None
    m2_yoy: Optional[float] = None
    fx_change: Optional[float] = None
    cpi_lag_1: float
    cpi_lag_3: float
    cpi_lag_6: float
    cpi_lag_12: float

    model_config = {
        "json_schema_extra": {
            "example": {
                "wpi_yoy": 5.2,
                "usd_npr": 133.5,
                "repo_rate": 5.0,
                "m2": 6500000,
                "m2_yoy": 12.3,
                "fx_change": 0.5,
                "cpi_lag_1": 6.8,
                "cpi_lag_3": 7.1,
                "cpi_lag_6": 7.5,
                "cpi_lag_12": 7.9,
            }
        }
    }


class PredictResponse(BaseModel):
    model: str
    prediction: float
    unit: str = "CPI YoY (%)"


class MetricsResponse(BaseModel):
    model: str
    MAE: float
    RMSE: float
    R2: float
    MAPE: float

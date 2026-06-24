"""
Evaluation utilities for model comparison and reporting.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def residual_plot(y_true: np.ndarray, y_pred: np.ndarray,
                   model_name: str = "") -> go.Figure:
    """Residual scatter plot."""
    residuals = y_true - y_pred
    fig = px.scatter(x=y_pred, y=residuals,
                     labels={"x": "Predicted CPI YoY (%)", "y": "Residual"},
                     title=f"Residuals — {model_name}",
                     template="plotly_white")
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    return fig


def actual_vs_predicted_plot(dates, y_true, results_dict: dict) -> go.Figure:
    """Overlay actual vs. predicted for all models."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=y_true, name="Actual",
                              line=dict(color="black", width=2)))
    colors = px.colors.qualitative.Plotly
    for i, (key, res) in enumerate(results_dict.items()):
        if "predictions" in res:
            fig.add_trace(go.Scatter(
                x=dates, y=res["predictions"],
                name=res.get("name", key),
                line=dict(dash="dot", color=colors[i % len(colors)])
            ))
    fig.update_layout(title="Actual vs Predicted — CPI YoY (%)",
                       xaxis_title="Date",
                       yaxis_title="CPI YoY (%)",
                       template="plotly_white",
                       legend=dict(orientation="h", y=-0.2))
    return fig


def feature_importance_plot(importance_dict: dict,
                              model_name: str = "",
                              top_n: int = 10) -> go.Figure:
    """Horizontal bar chart for feature importance."""
    items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    features, values = zip(*items)
    fig = px.bar(x=values, y=features, orientation="h",
                  title=f"Feature Importance — {model_name}",
                  labels={"x": "Importance", "y": "Feature"},
                  template="plotly_white")
    fig.update_yaxes(autorange="reversed")
    return fig


def metrics_heatmap(comparison_df: pd.DataFrame) -> go.Figure:
    """Heatmap of model metrics for visual comparison."""
    metrics_cols = ["MAE", "RMSE", "MAPE"]
    available = [c for c in metrics_cols if c in comparison_df.columns]
    z = comparison_df[available].values
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=available,
        y=comparison_df["Model"].tolist(),
        colorscale="RdYlGn_r",
        text=np.round(z, 3),
        texttemplate="%{text}",
    ))
    fig.update_layout(title="Model Metrics Comparison",
                       template="plotly_white")
    return fig

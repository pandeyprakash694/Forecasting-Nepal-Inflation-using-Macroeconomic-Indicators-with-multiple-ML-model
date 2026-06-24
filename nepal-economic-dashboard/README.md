# Nepal Economic Dashboard

A full-stack data science project for analysing and forecasting Nepal's macroeconomic
indicators using data published by **Nepal Rastra Bank (NRB)**.

## Live Demo Scope

| Layer | Technology |
|-------|------------|
| Data | NRB Monthly Economic Bulletin (CSV exports) |
| EDA & Modelling | Jupyter Notebooks + Python |
| ML Backend | FastAPI |
| Interactive Dashboard | Streamlit |

---

## Data Coverage

| Category | Datasets |
|----------|----------|
| **Inflation** | CPI YoY, WPI YoY, Province-wise CPI, Ecology-wise CPI |
| **Trade** | Top Imports, Top Exports, Country-wise Direction |
| **Forex** | USD/NPR Exchange Rate, Gross Forex Reserves |
| **Banking** | Money Supply (M2), Deposits, Interest Rates |
| **Stock Market** | NEPSE Indicators, Listed Companies, Turnover |
| **External Sector** | Tourist Arrivals, Migrant Workers |

---

## ML Models

| Model | Type |
|-------|------|
| Ridge Regression | Linear (baseline) |
| Random Forest | Ensemble (bagging) |
| Gradient Boosting | Ensemble (boosting) |
| XGBoost | Gradient boosting |
| LightGBM | Gradient boosting |
| Prophet | Time-series (Facebook) |
| ARIMA | Classical time-series |

**Target variable:** CPI Year-over-Year inflation (%)  
**Evaluation:** Walk-forward cross-validation (preserves time order)

---

## Project Structure

```
nepal-economic-dashboard/
├── src/
│   ├── data_loader.py       # Parsers for all NRB CSV files
│   ├── preprocessing.py     # Feature engineering & train/test split
│   ├── models.py            # All ML models
│   └── evaluate.py          # Metrics & visualisation helpers
├── notebooks/
│   ├── 01_EDA.ipynb         # Exploratory Data Analysis
│   └── 03_models.ipynb      # Model training & evaluation
├── api/
│   ├── main.py              # FastAPI application
│   └── schemas.py           # Pydantic request/response schemas
├── app/
│   └── streamlit_app.py     # Streamlit dashboard
├── models/                  # Saved .pkl model files (auto-created)
├── requirements.txt
└── README.md
```

---

## Getting Started

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit Dashboard
```bash
# from the nepal-economic-dashboard/ folder
streamlit run app/streamlit_app.py
```

### 3. Run the FastAPI backend
```bash
# from the nepal-economic-dashboard/ folder
uvicorn api.main:app --reload --port 8000
```
Then open: http://localhost:8000/docs

### 4. Train models via API
```bash
curl -X POST http://localhost:8000/api/v1/models/train
```

### 5. Predict via API
```bash
curl -X POST http://localhost:8000/api/v1/predict/xgboost \
  -H "Content-Type: application/json" \
  -d '{
    "cpi_lag_1": 6.8, "cpi_lag_3": 7.1,
    "cpi_lag_6": 7.5, "cpi_lag_12": 7.9,
    "wpi_yoy": 5.2, "repo_rate": 5.0
  }'
```

---

## Key Findings (EDA)

- Nepal CPI shows strong **autocorrelation** — past inflation is the best single predictor
- **WPI and CPI** are highly correlated (imported commodity prices pass through quickly)
- **Repo rate increases** (2022–2023) preceded a fall in CPI — monetary policy effectiveness visible
- **USD/NPR depreciation** correlates positively with inflation (import cost channel)
- Tourism has a **post-COVID V-shaped recovery** with 2023–2025 surpassing pre-pandemic peaks

---

## Note on Data Limitations

NRB CSV files have ~10 annual data points (SMI table) and ~36 monthly points (CPI YoY).
With small samples, walk-forward CV is essential to avoid overfitting.
Tree-based models are regularised and evaluated honestly — MAPE and RMSE are reported.

---

## Author

Prakash — Freelance Data Scientist | FinTech & Macroeconomic Analysis

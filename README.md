# 🏦 Probability of Default (PD) Model — Digital Lending

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.3-9ACD32?logo=lightgbm&logoColor=white)](https://lightgbm.readthedocs.io/)
[![MLflow](https://img.shields.io/badge/MLflow-2.11-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![dbt](https://img.shields.io/badge/dbt-1.7-FF694B?logo=dbt&logoColor=white)](https://www.getdbt.com/)
[![DVC](https://img.shields.io/badge/DVC-3.x-945DD5?logo=dvc&logoColor=white)](https://dvc.org/)
[![Optuna](https://img.shields.io/badge/Optuna-3.x-6C63FF)](https://optuna.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **A production-grade end-to-end machine learning pipeline** that predicts the
> probability of default (PD) for short-term installment loans in an emerging
> market digital lending context. Built to IFRS 9 / Basel II standards with full
> experiment tracking, SHAP explainability, and a live public scoring API.

🔗 **[Live Demo](https://your-streamlit-url.streamlit.app)** &nbsp;|&nbsp;
📡 **[API Docs](https://your-railway-url.up.railway.app/docs)** &nbsp;|&nbsp;
📊 **[Model Card](reports/model_card.md)**

---

## 📌 Project Overview

| | |
|---|---|
| **Business problem** | Estimate default risk at loan origination for risk-based pricing and portfolio loss reduction |
| **Dataset** | 619,655 loans (2023–2025) with borrower demographics, FirstCentral bureau data, and loan behaviour |
| **Target** | Binary: 1 = default (90+ DPD or written-off), 0 = paid. Default rate: **20.79%** |
| **Segments** | **C1** — first-time borrowers (DR: 42.9%) modelled separately from **C2+** — returning borrowers (DR: 5–28%) |
| **Best C1 model** | `Ensemble [(LR+RF+LGBM) + Optuna]` — AUC: **0.6774** · Gini: **0.3548** · KS: **0.2561** |
| **Best C2+ model** | `Ensemble [(LR+RF+LGBM) + Optuna]` — AUC: **0.7572** · Gini: **0.5144** · KS: **0.3816** |

---

## 🏗️ Architecture

```
Raw CSV (619k loans)
    │
    ▼
┌─────────────────────────────────┐
│  Data Engineering               │
│  PostgreSQL SQL + dbt-duckdb    │
│  Bureau JSON parsing (FC API)   │
│  DVC for data versioning        │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  Feature Engineering            │
│  Temporal grouped split         │
│  IV ranking · Correlation filter│
│  24 engineered features         │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  Model Development              │
│  LR baseline → RF → LightGBM   │
│  Optuna tuning (150 trials)     │
│  Soft-voting ensemble           │
│  MLflow experiment tracking     │
│  SHAP explainability            │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  Deployment                     │
│  FastAPI  → Railway             │
│  Streamlit → Streamlit Cloud    │
└─────────────────────────────────┘
```

---

## 📁 Project Structure

```
PD/
├── 📂 data/
│   ├── raw/                    
│   └── processed/                
├── 📂 dbt_project/             
│   └── models/
│       ├── staging/            
│       ├── intermediate/       
│       └── marts/              
├── 📂 models/                  
│   ├── best_model_c1.pkl
│   ├── best_model_c2plus.pkl
│   └── inference_artifacts.json
├── 📂 notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_feature_engr.ipynb
│   └── 03_model_dev.ipynb
├── 📂 reports/                 
│   ├── shap_c1.png
│   ├── shap_c2plus.png
│   ├── model_comparison_c1.png
│   └── model_card.md
├── 📂 src/
│   ├── api/main.py          
│   └── streamlit_app.py    
├── .dvc/                   
├── dvc.yaml                
├── requirements.txt
├── Procfile                
└── README.md
```

---

## ⚡ Quick Start

### Prerequisites
- Python 3.10+
- Git

### 1 — Clone and install

```bash
git clone https://github.com/husainridwan/pd-model.git
cd pd-model
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2 — Pull data (DVC)

```bash
dvc pull
```

> If you don't have DVC remote access, place your anonymised CSV at
> `data/raw/loan_data.csv` and run the notebooks in order.

### 3 — Run dbt transformations

```bash
cd dbt_project
dbt deps
dbt run
dbt test
cd ..
```

### 4 — Run notebooks in order

```bash
jupyter notebook
```

Open and execute in sequence:
1. `01_data_cleaning.ipynb`
2. `02_feature_engr.ipynb`
3. `03_model_dev.ipynb`

Each notebook saves its outputs to `data/processed/` and `models/`.

### 5 — View MLflow experiment dashboard

```bash
mlflow ui --backend-store-uri ../mlruns
```

Open [http://localhost:5000](http://localhost:5000) to compare all runs.

### 6 — Run the API locally

```bash
uvicorn src.api.main:app --reload
```

Open [http://localhost:8000/docs](http://localhost:8000/docs) for interactive API docs.

### 7 — Run the Streamlit dashboard locally

```bash
streamlit run src/app/streamlit_app.py
```

---

## 🔬 Methodology

### Data Pipeline

| Stage | Tool | Description |
|---|---|---|
| SQL extraction | PostgreSQL + Metabase | Loan, user, bureau data joined across 3 tables |
| Bureau parsing | Python + JSONB | FirstCentral `full_credit` array parsed at index 9 for payment history |
| Transformation | dbt-duckdb | 3-layer model: staging → intermediate → mart |
| Data versioning | DVC | Raw and processed data tracked, not stored in git |
| Quality checks | Great Expectations | Schema, null rate, range validation before any transformation |

### Feature Engineering

- **24 features** surviving IV ranking (≥ 0.02) and correlation filter (r ≤ 0.85)
- **Random user split**: Loans were split randomly into 70/15/15 (all loans per user in one split)
- Users spanning periods go entirely into train — no borrower-level leakage
- Prior behaviour features (`prior_loan_count`, `days_since_last_loan`) computed with expanding window sorted by `(user_id, approval_date)`
- State risk tier encoded from training-set default rates only and applied to all splits


## 📊 Model Performance

Both models were evaluated on a held‑out test set (randomly split by user).  
The **Soft Voting Ensemble** was the best performer for both segments.

| Segment | Model                   | Val AUC | Test AUC | Test Gini | Test KS |
|---------|-------------------------|---------|----------|-----------|---------|
| C1      | Logistic Regression     | 0.6629  | 0.6566   | 0.3131    | 0.2182  |
| C1      | Random Forest           | 0.6848  | 0.6792   | 0.3584    | 0.2536  |
| C1      | LightGBM + Optuna       | 0.6823  | 0.6759   | 0.3517    | 0.2513  |
| C1      | **Ensemble (LR+RF+LGBM)** | **0.6851** | **0.6774** | **0.3548** | **0.2561** |
| C2+     | Logistic Regression     | 0.7481  | 0.7433   | 0.4866    | 0.3628  |
| C2+     | Random Forest           | 0.7614  | 0.7572   | 0.5144    | 0.3789  |
| C2+     | LightGBM + Optuna       | 0.7592  | 0.7556   | 0.5112    | 0.3770  |
| C2+     | **Ensemble (LR+RF+LGBM)** | **0.7619** | **0.7572** | **0.5144** | **0.3816** |

Risk bands were calibrated from the validation set.  
(See `models/inference_artifacts.json` for thresholds.)

All models calibrated with isotonic regression on validation set.
Cross-validation uses `StratifiedGroupKFold(groups=user_id)` to prevent borrower-level leakage within the training fold.

### Why Two Segments?

EDA revealed a 30pp default rate gap between first-time borrowers (C1: 42.9%)
and returning borrowers (C25+: 5.4%). A single model averages across these
populations and underperforms on both. C1 relies on bureau and demographic
signals; C2+ relies on prior loan count and borrowing frequency.

---

## 📊 Key Findings

- 🔑 **Loan sequence** is the strongest predictor: `cardinal_log` IV = 0.596
- ⚠️ **Full credit utilisation** (borrowing at exactly the approved limit) increases
  default probability by ~2.5× — 74% of all loans in the dataset hit this threshold
- 📉 **Medium tenure** (31–60 days) has the highest default rate (31.2%) —
  non-monotonic signal captured by `is_medium_tenure` flag
- 🏦 **Bureau data** adds more lift for C1 borrowers (no prior loan history)
  than for C2+ borrowers where prior loan count dominates
- 🚫 **Gender excluded** on both IV grounds (0.2pp gap) and CBN fair lending
  regulatory compliance

---

## 🚀 Deployment

### API (FastAPI on Railway)

```bash
# POST /predict
curl -X POST https://your-railway-url.up.railway.app/predict \
  -H "Content-Type: application/json" \
  -d '{
    "loan_number": 1,
    "principal_amount": 50000,
    "tenure": 30,
    "monthly_income": 150000,
    "age": 32,
    "state": "Lagos",
    "employment_status": "Employed",
    "product_type": "Standard",
    "lender_count": 0
  }'
```

Response:

```json
{
  "probability_of_default": 0.187,
  "risk_band": "Medium",
  "model_used": "C1",
  "top_drivers": {
    "cardinal_log": "+5.2%",
    "is_full_utilisation": "+3.8%",
    "state_risk_tier_enc": "+1.9%"
  }
}
```

### Streamlit Dashboard

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://pdapp.streamlit.app)

Three tabs: **Credit Scorer** · **Model Performance** · **Methodology**

---

## ⚠️ Limitations

- **Bureau data is self-reported** at inference — no live bureau API call in the
  public demo. Model accuracy is higher in production where real bureau data is available.
- **Prior default rate excluded** — Lender's business rule (defaulters cannot
  re-borrow) means `prior_default_rate` has IV ≈ 0 in the training data and is not
  a usable feature despite being theoretically the strongest predictor.
- **Temporal coverage**: training data from 2023–2025. Model should be retrained
  quarterly and monitored monthly using PSI.

---

## 📋 Model Card

See [`reports/model_card.md`](reports/model_card.md) for full documentation
including intended use, training data description, evaluation results, and
fairness considerations.

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Data engineering | PostgreSQL · dbt-duckdb · DVC · Great Expectations |
| Feature engineering | Python · Pandas · NumPy |
| Modeling | Scikit-learn · LightGBM · Optuna · SHAP |
| Experiment tracking | MLflow |
| API | FastAPI · Uvicorn · Pydantic |
| Dashboard | Streamlit · Plotly |
| Deployment | Railway (API) · Streamlit Community Cloud (UI) |
| Version control | Git · GitHub |

---

## 👤 Author

**Ridwanllah Husain** — Risk & Data Analyst  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin)](https://www.linkedin.com/in/ridwanllah-husain-655458195/)
[![GitHub](https://img.shields.io/badge/GitHub-husainridwan-181717?logo=github)](https://github.com/husainridwan)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
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

> This was built while working in fintech risk analytics, this project answers a
> question I was dealing with daily: how do you decide who to lend to when
> most of your borrowers have little or no formal credit history? I took
> 619k+ real loan records, built a segmented scoring model from scratch,
> and shipped it as a live web app. Everything from data extraction to
> deployment is here.

🔗 **[Live Demo](https://defaultpredictorapp.streamlit.app)** &nbsp;|&nbsp;
📡 **[API Docs](https://defaultpredictorapp.onrender.app/docs)** &nbsp;|&nbsp;
📊 **[Model Card](https://github.com/husainridwan/ProbabilityOfDefault/blob/main/notebooks/reports/model_card.md)**

---

## 📌 Project Overview

| | |
|---|---|
| **Business problem** | Estimate default risk at loan origination for risk-based pricing and portfolio loss reduction |
| **Dataset** | 619k+ loans with borrower demographics, credit bureau data, and loan behaviour |
| **Target** | Binary: 1 = default (90+ DPD or written-off), 0 = paid. Default rate: **20.79%** |
| **Segments** | First-time borrowers (**L1**- DR: 42.9%) modelled separately from returning borrowers (**L2+**- DR: 5–28%) |
| **Best L1 model** | `Ensemble [(LR+RF+LGBM) + Optuna]` — AUC: **0.6774** · Gini: **0.3548** · KS: **0.2561** |
| **Best L2+ model** | `Ensemble [(LR+RF+LGBM) + Optuna]` — AUC: **0.7572** · Gini: **0.5144** · KS: **0.3816** |

---

## 🏗️ Architecture

```
Raw CSV (619k loans)
    │
    ▼
┌─────────────────────────────────┐
│  Data Analytics                 │
│  PostgreSQL SQL + dbt-duckdb    │
│  Bureau JSON parsing            │
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
│  LR baseline → RF → LightGBM    │
│  Optuna tuning (150 trials)     │
│  Soft-voting ensemble           │
│  MLflow experiment tracking     │
│  SHAP explainability            │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  Deployment                     │
│  FastAPI  → Render              │
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
git clone https://github.com/husainridwan/ProbabilityOfDefault.git
cd PD
python -m venv .venv
source .venv/bin/activate      
pip install -r requirements.txt
```

### 2 — Pull data (DVC)

```bash
dvc pull
```

> If you don't have DVC remote access, place your csv data at
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

---

## 🔬 Methodology

### Data Pipeline

| Stage | Tool | Description |
|---|---|---|
| Data Extraction | PostgreSQL + Metabase | Loan, user, bureau data joined across 3 tables |
| Data Transformation | dbt-duckdb | 3-layer model: staging - intermediate - mart |
| Data versioning | DVC | Raw and processed data tracked, not stored in git |
| Quality checks | Great Expectations | Schema, null rate, range validation before any transformation |

### Why two separate models?

First-time borrowers and returning borrowers are not the same credit problem.
They default at 42.9% on their first loan; by the time someone has
taken 25 loans with the same lender, their default rate is 5.4%. A single
model averaging across both groups would be too conservative with good returning
customers and not cautious enough with new ones. So I trained separate models:
the L1 model leans on bureau signals and demographics, while L2+ gets the
additional benefit of prior loan count and days between loans.

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
| L1      | Logistic Regression     | 0.6629  | 0.6566   | 0.3131    | 0.2182  |
| L1      | Random Forest           | 0.6848  | 0.6792   | 0.3584    | 0.2536  |
| L1      | LightGBM + Optuna       | 0.6823  | 0.6759   | 0.3517    | 0.2513  |
| L1      | **Ensemble (LR+RF+LGBM)** | **0.6851** | **0.6774** | **0.3548** | **0.2561** |
| L2+     | Logistic Regression     | 0.7481  | 0.7433   | 0.4866    | 0.3628  |
| L2+     | Random Forest           | 0.7614  | 0.7572   | 0.5144    | 0.3789  |
| L2+     | LightGBM + Optuna       | 0.7592  | 0.7556   | 0.5112    | 0.3770  |
| L2+     | **Ensemble (LR+RF+LGBM)** | **0.7619** | **0.7572** | **0.5144** | **0.3816** |

Risk bands were calibrated from the validation set.  
(See `models/inference_artifacts.json` for thresholds.)

All models calibrated with isotonic regression on validation set.
Cross-validation uses `StratifiedGroupKFold(groups=user_id)` to prevent borrower-level leakage within the training fold.

---

## 📊 What the Data Shows

A few things stand out from the exploratory analysis that shaped the whole
modelling approach:

- 🔑 **Loan sequence dominates everything.** `cardinal_log` had an IV of 0.596,
  the highest of any feature. Borrowing history is the clearest signal we have.
- ⚠️ **Maxing out the credit limit is a risk flag.** 74% of loans hit the approved
  limit exactly, and those borrowers default at 2.7x the rate of people who
  borrow less than their limit.
- 📉 **Medium-tenure loans are riskier than long ones.** 60-day loans default
  at 31.2%, the highest of any tenure. The pattern is non-monotonic, which
  is easy to miss if you just look at averages.
- 🏦 **Bureau data matters more for first-time borrowers.** For returning borrowers,
  prior loan count tells you more than any bureau score.
- 🚫 **Gender was excluded.** The default rate gap between groups was 0.2%, 
below the IV threshold and excluded on fair lending grounds.

---

## 📋 Model Card

See [`notebooks/reports/model_card.md`](notebooks/reports/model_card.md) for full documentation
including intended use, training data description, evaluation results, and
fairness considerations.

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Data analytics | PostgreSQL · dbt-duckdb · DVC · Great Expectations |
| Feature engineering | Python · Pandas · NumPy |
| Modelling | Scikit-learn · LightGBM · Optuna · SHAP |
| Experiment tracking | MLflow |
| API | FastAPI · Uvicorn · Pydantic |
| Dashboard | Streamlit · Plotly |
| Deployment | Railway (API) · Streamlit Community Cloud (UI) |
| Version control | GitHub |

---

## 👤 Author

**Ridwanllah Husain** — Risk & Data Analyst  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin)](https://www.linkedin.com/in/ridwanllah-husain-655458195/)
[![GitHub](https://img.shields.io/badge/GitHub-husainridwan-181717?logo=github)](https://github.com/husainridwan)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
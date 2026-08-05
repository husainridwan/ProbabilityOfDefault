# 🏦 Probability of Default — Segmented Credit Scoring for Digital Lending

**A pooled credit model told us thin-file borrowers and repeat borrowers were
the same risk. They default at 41% and 15%. This project splits them, scores
them separately, and explains every decision.**

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.3-9ACD32?logo=lightgbm&logoColor=white)](https://lightgbm.readthedocs.io/)
[![MLflow](https://img.shields.io/badge/MLflow-2.11-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![dbt](https://img.shields.io/badge/dbt-1.7-FF694B?logo=dbt&logoColor=white)](https://www.getdbt.com/)
[![DVC](https://img.shields.io/badge/DVC-3.x-945DD5?logo=dvc&logoColor=white)](https://dvc.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

🔗 **[Live demo](https://defaultpredictorapp.streamlit.app)** ·
📊 **[Model card](notebooks/reports/model_card.md)** ·
📡 **[Author](https://github.com/husainridwan)**

---

## 📌 The Business Problem

A digital lender in Nigeria writes short-tenor instalment loans to a mostly
thin-file population i.e., customers with little or no formal credit history. Two
decisions have to be made at origination: **approve or decline**, and **at what
limit**. Both need a probability of default.

The obvious approach is one model over the whole book. That approach is wrong
here, and the data says so plainly:

| Segment | Default rate |
|---|---|
| **L1** — first loan with the lender | **40.9%** |
| **L2+** — returning borrower | **14.7%** |
| Whole book (619,655 loans) | 20.8% |

A single model trained across both learns an average that fits neither. It is
too pessimistic about a customer on their twelfth loan and too optimistic about
one on their first; and those are the two decisions that matter most. The cost
of getting it wrong is asymmetric: over-declining good returning customers
kills the repeat business the unit economics depend on, while under-pricing
first-timers writes losses directly.

**So this project trains two models and routes each application to the right
one.**

---

## What was built

```
619,655 loans · 167,990 borrowers · Jan 2023 – Dec 2025
        │
        ▼
┌───────────────────────────────────────────────┐
│  Extract & model         PostgreSQL → dbt     │
│  Loan, user, and bureau tables joined;        │
│  staging → marts; DVC versions the data       │
└───────────────────┬───────────────────────────┘
                    ▼
┌───────────────────────────────────────────────┐
│  Features                24 survivors         │
│  Information Value ≥ 0.02 · correlation ≤ 0.85│
│  Grouped user-level split, no borrower leakage│
└───────────────────┬───────────────────────────┘
                    ▼
┌───────────────────────────────────────────────┐
│  Two models              L1  and  L2+         │
│  LogReg → RandomForest → LightGBM + Optuna    │
│  Soft-voting ensemble · isotonic calibration  │
│  MLflow tracking · SHAP explanations          │
└───────────────────┬───────────────────────────┘
                    ▼
┌───────────────────────────────────────────────┐
│  Serve                   FastAPI · Streamlit  │
│  Score → risk band → per-feature explanation  │
└───────────────────────────────────────────────┘
```

---

## Results

Held-out test set, split by borrower so no customer appears in both train and
test.

| Segment | Model | Test AUC | Gini | KS |
|---|---|---|---|---|
| L1 | Logistic Regression | 0.657 | 0.313 | 0.218 |
| L1 | Random Forest | 0.679 | 0.358 | 0.254 |
| L1 | LightGBM + Optuna | 0.676 | 0.352 | 0.251 |
| **L1** | **Ensemble (LR+RF+LGBM)** | **0.677** | **0.355** | **0.256** |
| L2+ | Logistic Regression | 0.743 | 0.487 | 0.363 |
| L2+ | Random Forest | 0.757 | 0.514 | 0.379 |
| L2+ | LightGBM + Optuna | 0.756 | 0.511 | 0.377 |
| **L2+** | **Ensemble (LR+RF+LGBM)** | **0.757** | **0.514** | **0.382** |

**Read these numbers honestly.** L2+ at AUC 0.757 is a usable production
model. L1 at 0.677 is weak in absolute terms and that is the expected result
when scoring applicants with no internal history. The point of the segmentation
is not that L1 becomes good; it is that L1's weakness stops contaminating the
L2+ score, where most of the lending volume and nearly all of the repeat
revenue sits.

The ensemble barely beats Random Forest alone on L2+ (0.7572 vs 0.7572 on AUC,
separating only on KS). If you are reproducing this and want a simpler
artefact to maintain, Random Forest alone is a defensible choice; the ensemble
earns its place mainly through stability across the two segments, not through a
headline metric.

### Risk bands — what the score actually authorises

Calibrated on the validation set, so each band carries an observed default rate
rather than an arbitrary cut.

| Band | L1 default rate | L1 action | L2+ default rate | L2+ action |
|---|---|---|---|---|
| Very low | 18.3% | Approve | 1.8% | Auto-approve |
| Low | 35.5% | Approve, monitor | 6.8% | Approve |
| Medium | 44.7% | Manual review | 13.5% | Approve with conditions |
| High | 54.3% | Decline or cut limit | 23.1% | Manual review |
| Very high | 66.0% | Decline | 37.9% | Decline |

Note that a *"very low"* first-time borrower (18.3%) is riskier than a *"high"*
returning one (23.1%) is close to. The bands are segment-relative on purpose —
a single global cut-off would decline nearly the entire L1 population, which is
the acquisition funnel.

---

## What the data showed

Five findings that changed the modelling approach:

**Loan sequence dominates every other signal.** `cardinal_log` - how many loans
the customer has taken carries an Information Value of 0.596, several times
any other feature. Borrowing history beats every bureau field available.

**Maxing the limit is a red flag.** 74% of loans use the approved limit
exactly, and those borrowers default at ~2.7× the rate of customers who take
less than offered. Full utilisation became an explicit feature
(`is_full_utilisation`, plus an interaction with the L1 flag).

**Risk by tenure is non-monotonic.** 60-day loans default at 31.2% - worse than
both shorter and longer tenors. A linear tenure term would have missed this
entirely; it is captured with a medium-tenure indicator.

**Bureau data earns its keep only for first-timers.** For L2+, internal prior
behaviour outperforms every purchased bureau signal. That has a direct cost
implication: bureau pulls are worth paying for on L1 applications and largely
redundant on L2+.

**Prior default rate is structurally useless here.** IV ≈ 0.000; not because
defaults do not predict defaults, but because policy blocks defaulters from
re-borrowing, so they are absent from the training data by construction. This
is the kind of feature that looks predictive in theory and is empty in
practice; `prior_loan_count` proxies borrower tenure instead.

**`gender` was excluded** on two independent grounds: an IV below the 0.02
threshold (a 0.2pp default-rate gap) and CBN fair-lending compliance. Full
exclusion rationale is in the [model card](notebooks/reports/model_card.md).

---

## Why these technical choices

**Two models rather than one interaction term.** An `is_first_loan` flag inside
a pooled model lets the tree split on segment, but it still shares
hyperparameters, calibration, and feature importances across two populations
whose *best available signals differ*; L1 leans on bureau and demographics,
L2+ on internal history. Separate models let each optimise independently and be
monitored independently.

**Grouped split, not a row-level one.** All loans belonging to a borrower go
into a single split, and cross-validation uses
`StratifiedGroupKFold(groups=user_id)`. Without this, the same customer's
loan #3 lands in train and loan #4 in test, and the model scores its own
history back — inflated AUC that vanishes in production.

**Isotonic calibration.** Ranking is not enough when the output feeds a risk
band with a stated default rate. Calibration is what makes "13.5% band" mean
13.5%.

**Soft voting over stacking.** Stacking adds a meta-learner and a second layer
to validate, monitor, and explain. On this data it was not buying enough to
justify the operational cost.

**SHAP, not just global importances.** A declined applicant is entitled to a
reason, and a reviewer overriding the model needs to see what drove the score.
Global feature importance cannot answer "why *this* application".

---

## Known limitations

Stated plainly, because a credit model whose limits are undocumented is not
deployable.

1. **Bureau features are self-reported in the public demo.** The hosted app
   asks plain-language questions ("how many lenders have you borrowed from?").
   Production would replace these with bureau API calls. Demo accuracy is
   therefore optimistic relative to what self-reported inputs would deliver in
   a real funnel, where applicants have an incentive to under-report.

2. **No temporal validation.** The split is random at borrower level, not
   forward in time, so these metrics do not evidence stability across a
   changing macro environment. A temporal split *was* attempted and produced a
   bureau coverage gap between train and test (40% vs 3%) driven by how the
   bureau data pull was constructed; an artefact of the data, not of the
   method. Until that is resolved, treat these numbers as in-sample-in-time.
   Production use needs monthly retraining on a rolling window with PSI
   monitoring.

3. **L1 AUC of 0.677 is genuinely weak.** It is deployable only alongside
   manual review on the medium band and conservative initial limits. The honest
   framing is that thin-file scoring is hard, not that this model solved it.

4. **Trained on one lender, one market, one product.** Nigerian short-tenor
   digital lending. Not transferable to mortgages, SME, or longer tenors
   without retraining — see the model card's out-of-scope section.

---

## Repository

```
├── notebooks/
│   ├── 01_data_cleaning.ipynb      raw → cleaned, bureau JSON parsed
│   ├── 02_feature_engr.ipynb       IV ranking, correlation filter, grouped split
│   ├── 03_model_dev.ipynb          LR → RF → LGBM → ensemble, Optuna, SHAP
│   └── reports/
│       ├── model_card.md           intended use, limits, fairness
│       ├── shap_c1.png             L1 explanations
│       ├── shap_c2plus.png         L2+ explanations
│       └── model_comparison_*.png  per-segment model comparison
├── dbt_project/
│   └── models/
│       ├── staging/stg_data.sql
│       └── marts/fct_model_features.sql
├── src/
│   ├── api/main.py                 FastAPI scoring service
│   └── streamlit_app.py            self-contained demo UI
├── models/                         DVC-tracked artefacts (.pkl, thresholds)
├── data/                           DVC-tracked; not in git
├── Procfile                        uvicorn entrypoint
└── requirements.txt
```

---

## Reproducing this

Requires Python 3.10+ and Git.

```bash
git clone https://github.com/husainridwan/ProbabilityOfDefault.git
cd ProbabilityOfDefault
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Data.** The loan data is proprietary and DVC-tracked, so `dvc pull` needs
remote access that this repo does not grant publicly. To run the pipeline on
your own data, place a CSV at `data/raw/data.csv` with the schema documented in
`notebooks/01_data_cleaning.ipynb` and run the notebooks in order.

```bash
cd dbt_project && dbt deps && dbt run && dbt test && cd ..   # transformations
jupyter notebook                                             # 01 → 02 → 03
mlflow ui --backend-store-uri ./mlruns                       # compare runs
uvicorn src.api.main:app --reload                            # API at :8000/docs
streamlit run src/streamlit_app.py                           # UI at :8501
```

The Streamlit app is self-contained as it carries its own scoring logic and
thresholds, so it runs without the DVC-tracked model artefacts.

---

## Stack

| Layer | Tools |
|---|---|
| Transformation | PostgreSQL · dbt (duckdb adapter) |
| Data versioning | DVC |
| Features & modelling | pandas · NumPy · scikit-learn · LightGBM · Optuna |
| Explainability | SHAP |
| Experiment tracking | MLflow |
| Serving | FastAPI · Uvicorn · Pydantic · Streamlit · Plotly |

---

## Author

**Ridwanllah Husain** — Risk & Data Analyst
[GitHub](https://github.com/husainridwan) ·
[LinkedIn](https://www.linkedin.com/in/ridwanllah-husain/) ·
h.ridwan707@gmail.com

Built while working in fintech risk analytics, on the question I dealt with
daily: how do you decide who to lend to when most of your borrowers have no
formal credit history?

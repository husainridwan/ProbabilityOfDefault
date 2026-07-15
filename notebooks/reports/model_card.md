# Model Card — Probability of Default (PD) Model

## Model Details

| | |
|---|---|
| **Model type** | Soft-voting ensemble (Logistic Regression + Random Forest + LightGBM) + Optuna |
| **Segments** | Two separate models: L1 (first-time borrowers) and L2+ (returning borrowers) |
| **Version** | 1.0 |
| **Date** | July 2026 |
| **Framework** | scikit-learn 1.4, LightGBM 4.3, Optuna 3.x |
| **Contact** | h.ridwan707@gmail.com | 

## Intended Use

**Primary use case:** Estimate the probability that a borrower will default
(miss repayment by 90+ days or be written off) on a short-term installment
loan at the point of origination.

**Intended users:** Credit risk analysts, loan officers, and automated
underwriting systems at digital lending platforms in emerging markets.

**Out-of-scope uses:** This model should not be used for mortgage lending,
SME lending, or any loan product with tenure exceeding 12 months. It was
trained on Nigerian digital lending data and may not generalise to other
markets without retraining.

## Training Data

| | |
|---|---|
| **Source** | Anonymised loan performance data from a Nigerian digital lender |
| **Size** | 619,655 loans across 167,990 unique borrowers |
| **Date range** | January 2023 – December 2025 |
| **Split strategy** | Random user-level split 70/15/15 (all loans per user in one split) |
| **Default rate** | 20.79% overall · 40.94% L1 · 14.65% L2+ |
| **Target definition** | 1 = Default | 0 = No Default |

## Features

**24 features** selected by Information Value (≥ 0.02) and correlation
filter (r ≤ 0.85).

| Category | Features |
|---|---|
| Loan characteristics | cardinal_log, credit_limit, principal_log, tenure, is_full_utilisation, is_medium_tenure, burden_score, full_util_x_c1, is_first_loan |
| Product & channel | product_type, channel_group |
| Demographics | age, salary_log, state_risk_tier_enc |
| Bureau (self-reported at inference) | distinct_lender_count, total_monthly_obligations, total_bureau_accounts, bureau_dsti, bureau_annual_dti, bureau_health_score, total_outstanding_debt, total_enquiry_count, prop_bad, arrears_ratio |
| Prior behaviour (L2+ only) | prior_loan_count, days_since_last_loan |

**Excluded features and reasons:**

- `gender` — 0.2% default rate gap (IV < 0.02) and CBN fair lending compliance
- `prior_default_rate` / `ever_defaulted` — IV ≈ 0.000; business rule prevents
  re-lending to defaulters, so prior defaults are structurally absent from training data
- `bureau_account_rating` — IV 0.003, weakest bureau signal
- `accounts_in_arrears` — IV 0.007, dominated by bureau_health_score

## Evaluation Results

### L1 Model (First-time Borrowers)

| Metric | Validation | Test |
|---|---|---|
| ROC-AUC | 0.685 | **0.677** |
| Gini | 0.370 | **0.355** |
| KS Statistic | 0.266 | **0.256** |
| PR-AUC | — | **0.590** |


### L2+ Model (Returning Borrowers)

| Metric | Validation | Test |
|---|---|---|
| ROC-AUC | 0.762 | **0.757** |
| Gini | 0.524 | **0.514** |
| KS Statistic | 0.387 | **0.382** |
| PR-AUC | — | **0.347** |


### Risk Band Calibration (from validation set)

**L1 Model:**
| Band | Default Rate | Interpretation |
|---|---|---|
| Very Low | 18.3% | Approve |
| Low | 35.5% | Approve with monitoring |
| Medium | 44.7% | Manual review |
| High | 54.3% | Decline or reduce limit |
| Very High | 66.0% | Decline |

**L2+ Model:**
| Band | Default Rate | Interpretation |
|---|---|---|
| Very Low | 1.8% | Auto-approve |
| Low | 6.8% | Approve |
| Medium | 13.5% | Approve with conditions |
| High | 23.1% | Manual review |
| Very High | 37.9% | Decline |

## Known Limitations

1. **Bureau data is self-reported at inference**: The public demo API
   collects bureau features through plain-language questions (e.g. "how many
   lenders have you borrowed from?"). In a live production use, credit bureau API
   calls would replace self-reporting and improve accuracy.

2. **Prior default rate is structurally absent**: Borrowers who defaulted
   cannot re-borrow under normal policy, so `prior_default_rate` has IV ≈ 0
   in training data. The model uses `prior_loan_count` as a proxy for
   borrower tenure instead.

3. **Temporal deployment not validated**: The model uses a random user
   split rather than a temporal split. In production, the model should be
   retrained monthly on a rolling window and monitored with PSI.
   The temporal split was tested but produced a bureau coverage gap
   (40% train vs 3% test) due to the structure of the bureau data pull.

4. **L1 model performance**: AUC 0.677 for first-time borrowers reflects
   the inherent difficulty of scoring thin-file applicants with no internal
   history. Bureau and demographic signals provide the primary lift.

## Monitoring Recommendations

- **AUC monitoring** weekly on new loan cohorts as they mature.
- **Default rate by risk band** monthly to verify bands remain calibrated.
- **Retrain trigger** if test AUC drops more than 3% below training AUC.
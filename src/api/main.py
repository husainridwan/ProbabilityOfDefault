import sys
import json
import traceback
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from sklearn.base import BaseEstimator, ClassifierMixin

class EnsembleModel(BaseEstimator, ClassifierMixin):
    _estimator_type = "classifier"
    def __init__(self, lr=None, rf=None, lgbm=None):
        self.lr = lr; self.rf = rf; self.lgbm = lgbm
        self.classes_ = np.array([0, 1])
    def predict_proba(self, X):
        p_lr   = self.lr.predict_proba(X)[:, 1]
        p_rf   = self.rf.predict_proba(X)[:, 1]
        p_lgbm = self.lgbm.predict_proba(X)[:, 1]
        avg = (p_lr + p_rf + p_lgbm) / 3
        return np.column_stack([1 - avg, avg])
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

class LGBMPipeline(BaseEstimator, ClassifierMixin):
    _estimator_type = "classifier"
    def __init__(self, preprocessor=None, model=None, feat_names=None, cat_indices=None):
        self.preprocessor = preprocessor
        self.model        = model
        self.feat_names   = feat_names
        self.cat_indices  = cat_indices
        self.classes_     = np.array([0, 1])
    def fit(self, X, y=None):
        return self
    def predict_proba(self, X):
        X_pp = self.preprocessor.transform(X)
        if self.feat_names is not None:
            X_pp = pd.DataFrame(X_pp, columns=self.feat_names)
        return self.model.predict_proba(X_pp)
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

sys.modules['__main__'].EnsembleModel = EnsembleModel
sys.modules['__main__'].LGBMPipeline = LGBMPipeline

# Load artifacts and models
with open("models/inference_artifacts.json") as f:
    ARTIFACTS = json.load(f)

FEATURES_C1 = ARTIFACTS.get("features_c1")
FEATURES_C2 = ARTIFACTS.get("features_c2plus")
if not FEATURES_C1 or not FEATURES_C2:
    raise RuntimeError("Feature lists missing – regenerate the modelling notebook.")

MODEL_C1 = joblib.load("models/best_model_c1.pkl")
MODEL_C2PLUS = joblib.load("models/best_model_c2plus.pkl")

THRESHOLDS_C1 = {k: float(v) for k, v in ARTIFACTS["thresholds_c1"].items()}
THRESHOLDS_C2 = {k: float(v) for k, v in ARTIFACTS["thresholds_c2plus"].items()}

STATE_TIER_MAP = ARTIFACTS.get("state_tier_map", {})
PRODUCT_MAP = ARTIFACTS.get("product_map", {})
CHANNEL_MAP = ARTIFACTS.get("channel_map", {})

print(f"✅ Models loaded. L1 features: {len(FEATURES_C1)}, L2+ features: {len(FEATURES_C2)}")

# FastAPI app
app = FastAPI(title="PD Model API – Digital Lending", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class LoanApplication(BaseModel):
    loan_number: int = Field(..., ge=1, example=4)
    principal_amount: float = Field(..., gt=0, example=100000)
    tenure: int = Field(..., ge=7, le=365, example=90)
    credit_limit: Optional[float] = Field(150000, example=150000)
    monthly_income: float = Field(..., ge=0, example=150000)
    age: Optional[int] = Field(28, ge=18, le=75, example=28)
    state: Optional[str] = Field("Lagos", example="Lagos")
    employment_status: Optional[str] = Field("Employed", example="Employed")
    marital_status: Optional[str] = Field("Married", example="Married")
    education_level: Optional[str] = Field("Tertiary", example="Tertiary")
    product_type: Optional[str] = Field("Standard", example="Standard")
    channel_group: Optional[str] = Field("Organic", example="Organic")
    total_bureau_accounts: Optional[int] = Field(2, example=2)
    lender_count: Optional[int] = Field(1, example=1)
    enquiry_count: Optional[int] = Field(2, example=2)
    monthly_obligations: Optional[float] = Field(30000, example=30000)
    total_debt: Optional[float] = Field(45000, example=45000)
    has_written_off: Optional[int] = Field(0, example=0)
    has_arrears: Optional[int] = Field(0, example=0)
    days_since_last_loan: Optional[int] = Field(16, example=16)

class PredictionResponse(BaseModel):
    probability_of_default: float
    risk_band: str
    decision: str
    model_used: str

def preprocess(inp: LoanApplication):
    credit_limit = inp.credit_limit or inp.principal_amount
    income = max(inp.monthly_income, 1)
    return {
        "cardinal_log": np.log1p(inp.loan_number),
        "is_first_loan": int(inp.loan_number == 1),
        "credit_limit": credit_limit,
        "principal_log": np.log1p(inp.principal_amount),
        "tenure": inp.tenure,
        "is_full_utilisation": int(inp.principal_amount >= credit_limit * 0.99),
        "is_medium_tenure": int(30 < inp.tenure <= 60),
        "burden_score": (inp.principal_amount / income) * (inp.tenure / 30),
        "full_util_x_c1": int(inp.principal_amount >= credit_limit * 0.99) * int(inp.loan_number == 1),
        "product_type": PRODUCT_MAP.get(inp.product_type, 0),
        "channel_group": CHANNEL_MAP.get(inp.channel_group, 0),
        "age": inp.age or 35,
        "salary_log": np.log1p(income),
        "state_risk_tier_enc": STATE_TIER_MAP.get(inp.state, 1),
        "distinct_lender_count": inp.lender_count or 0,
        "total_bureau_accounts": inp.total_bureau_accounts or 0,
        "total_monthly_obligations": inp.monthly_obligations or 0,
        "total_outstanding_debt": inp.total_debt or 0,
        "total_enquiry_count": inp.enquiry_count or 0,
        "distinct_enquiring_lenders": min(inp.enquiry_count or 0, inp.lender_count or 0),
        "bureau_dsti": (inp.monthly_obligations or 0) / income,
        "bureau_annual_dti": (inp.total_debt or 0) / (income * 12),
        "prop_bad": float(inp.has_written_off or 0),
        "arrears_ratio": float(inp.has_arrears or 0),
        "bureau_health_score": _bureau_health(inp),
        "prior_loan_count": inp.loan_number - 1,
        "days_since_last_loan": inp.days_since_last_loan or 0,
    }

def _bureau_health(inp: LoanApplication):
    if inp.total_bureau_accounts == 0 and inp.lender_count == 0:
        return 0
    if inp.total_bureau_accounts == 0:
        return 1
    if inp.has_written_off:
        return 1
    if inp.has_arrears:
        return 2
    return 3

def get_risk_band(prob, thresholds):
    if prob <= thresholds["very_low"]:
        return "Very Low"
    elif prob <= thresholds["low"]:       
        return "Low"
    elif prob <= thresholds["medium"]:    
        return "Medium"
    elif prob <= thresholds["high"]:      
        return "High"
    return "Very High"

def get_decision(band: str) -> str:
    mapping = {
        "Very Low":  "Approve",
        "Low":       "Approve",
        "Medium":    "Review",
        "High":      "Decline",
        "Very High": "Decline",
    }
    return mapping.get(band, "Review")

@app.get("/")
def root():
    return {"message": "PD Model API – go to /docs"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict(loan: LoanApplication):
    try:
        is_c1 = int(loan.loan_number == 1)   # 1 = first loan, 0 = returning

        # Index 0 = C2+, Index 1 = C1
        model       = (MODEL_C2PLUS, MODEL_C1)[is_c1]
        thresholds  = (THRESHOLDS_C2, THRESHOLDS_C1)[is_c1]
        feature_list = (FEATURES_C2, FEATURES_C1)[is_c1]

        feat_dict = preprocess(loan)
        X = pd.DataFrame([{k: feat_dict.get(k, 0) for k in feature_list}])

        proba_raw = model.predict_proba(X)
        if isinstance(proba_raw, list):
            proba_raw = np.array(proba_raw)

        if proba_raw.ndim == 1:
            prob = float(proba_raw[0].item())
        elif proba_raw.ndim == 2:
            prob = float(proba_raw[0, 1].item())
        else:
            raise ValueError(f"Unexpected predict_proba shape: {proba_raw.shape}")

        band = get_risk_band(prob, thresholds)
        decision = get_decision(band)

        return PredictionResponse(
            probability_of_default=round(prob, 4),
            risk_band=band,
            decision=decision,
            model_used="L1" if is_c1 else "L2+",
        )
    except Exception as e:
        print("=" * 60)
        print("ERROR in /predict:")
        traceback.print_exc()
        print("=" * 60)
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/model-info")
def model_info():
    return {
        "l1_features": len(FEATURES_C1),
        "l2_features": len(FEATURES_C2),
        "l1_test_auc": 0.677,
        "l2_test_auc": 0.757,
    }
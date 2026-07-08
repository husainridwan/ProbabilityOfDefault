import mlflow
import mlflow.sklearn
import joblib
import json
import os
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score, brier_score_loss

# EnsembleModel class 
class EnsembleModel(BaseEstimator, ClassifierMixin):
    _estimator_type = "classifier"

    def __init__(self, lr, rf, lgbm):
        self.lr = lr
        self.rf = rf
        self.lgbm = lgbm
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        p_lr   = self.lr.predict_proba(X)[:, 1]
        p_rf   = self.rf.predict_proba(X)[:, 1]
        p_lgbm = self.lgbm.predict_proba(X)[:, 1]
        avg = (p_lr + p_rf + p_lgbm) / 3
        return np.column_stack([1 - avg, avg])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

# MLflow setup
# os.environ['MLFLOW_ALLOW_FILE_STORE'] = 'true'
mlflow.set_tracking_uri('sqlite:///./mlflow.db')
mlflow.set_experiment('PD_Model_Risk')

# Load data
train = pd.read_parquet('./data/processed/train_features.parquet')
val   = pd.read_parquet('./data/processed/val_features.parquet')
test  = pd.read_parquet('./data/processed/test_features.parquet')

with open('./models/inference_artifacts.json') as f:
    art = json.load(f)

# Models to log
models = [
    ('./models/lr_c1.pkl',          'Logistic Regression', 'C1'),
    ('./models/lr_c2plus.pkl',      'Logistic Regression', 'C2+'),
    ('./models/rf_c1.pkl',          'Random Forest',       'C1'),
    ('./models/rf_c2plus.pkl',      'Random Forest',       'C2+'),
    ('./models/lgbm_c1.pkl',        'LightGBM',            'C1'),
    ('./models/lgbm_c2plus.pkl',    'LightGBM',            'C2+'),
    ('./models/ensemble_c1.pkl',    'Ensemble',            'C1'),
    ('./models/ensemble_c2plus.pkl','Ensemble',            'C2+'),
    ('./models/best_model_c1.pkl',  'Best (C1)',           'C1'),
    ('./models/best_model_c2plus.pkl','Best (C2+)',        'C2+'),
]

# Trusted types for skops (needed for calibration and LightGBM)
trusted = [
    'numpy.dtype',
    'sklearn.calibration._CalibratedClassifier',
    'collections.OrderedDict',
    'lightgbm.basic.Booster',
    'lightgbm.sklearn.LGBMClassifier',
    '__main__.EnsembleModel'
]

for path, model_name, segment in models:
    try:
        model = joblib.load(path)
    except FileNotFoundError:
        print(f'Skipping {path} (not found)')
        continue
    except Exception as e:
        print(f'Error loading {path}: {e}')
        continue

    # Get feature list for this segment
    features = art['features_c1'] if segment == 'C1' else art['features_c2plus']
    # Filter val/test for this segment
    mask_val  = (val['is_first_loan'] == 1) if segment == 'C1' else (val['is_first_loan'] == 0)
    mask_test = (test['is_first_loan'] == 1) if segment == 'C1' else (test['is_first_loan'] == 0)
    X_val_seg  = val[mask_val][features]
    y_val_seg  = val[mask_val]['target']
    X_test_seg = test[mask_test][features]
    y_test_seg = test[mask_test]['target']

    try:
        proba_val  = model.predict_proba(X_val_seg)[:, 1]
        proba_test = model.predict_proba(X_test_seg)[:, 1]
    except Exception as e:
        print(f'Error predicting with {model_name} ({segment}): {e}')
        continue

    val_auc  = roc_auc_score(y_val_seg, proba_val)
    test_auc = roc_auc_score(y_test_seg, proba_test)
    val_brier= brier_score_loss(y_val_seg, proba_val)
    test_brier=brier_score_loss(y_test_seg, proba_test)

    with mlflow.start_run(run_name=f'{model_name}_{segment}'):
        mlflow.log_param('model', model_name)
        mlflow.log_param('segment', segment)
        mlflow.log_metric('val_auc', round(val_auc, 4))
        mlflow.log_metric('test_auc', round(test_auc, 4))
        mlflow.log_metric('val_brier', round(val_brier, 4))
        mlflow.log_metric('test_brier', round(test_brier, 4))
        # Log model with trusted types
        mlflow.sklearn.log_model(
            model,
            f'{model_name.lower().replace(" ","_")}_{segment.lower()}',
            skops_trusted_types=trusted
        )
        print(f'Logged {model_name} ({segment}) – Val AUC={val_auc:.4f}, Test AUC={test_auc:.4f}')

print('Done. All models logged to MLflow.')
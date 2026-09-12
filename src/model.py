"""
src/model.py — §3.4

External deps: xgboost, sklearn, shap, numpy, pandas, pickle, os
Internal deps: feature_extractor.FEATURE_NAMES
"""

import os
import pickle

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)
from xgboost import XGBClassifier, XGBRegressor

from src.feature_extractor import FEATURE_NAMES

DEFAULT_MODEL_PATH = os.path.join('models', 'gpugate_model.pkl')


def train_classifier(X, y, **kwargs) -> XGBClassifier:
    """y is binary (speedup_log2 > log2(1.2))."""
    params = dict(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, reg_alpha=0.1, base_score=0.5,
        eval_metric='logloss', use_label_encoder=False,
    )
    params.update(kwargs)
    model = XGBClassifier(**params)
    model.fit(X, y)
    return model


def train_regressor(X, y, **kwargs) -> XGBRegressor:
    """y is continuous (speedup_log2). Strictly more informative than the
    classifier once it works — predicts magnitude, not just a binary
    verdict. TreeSHAP works identically on it. This is the upgrade path."""
    params = dict(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, reg_alpha=0.1, base_score=0.5,
        objective='reg:squarederror',
    )
    params.update(kwargs)
    model = XGBRegressor(**params)
    model.fit(X, y)
    return model


def evaluate_model(model, X_test, y_test, threshold: float = 0.263) -> dict:
    """threshold is on log2-speedup / classifier decision space, default
    matches PROFITABLE_THRESHOLD_LOG2 = log2(1.2). For a classifier, y_test
    should be the binary labels and predictions come from predict(); the
    `threshold` param is kept for the regressor upgrade path (thresholding
    predicted speedup_log2 against it) and is unused for a plain
    XGBClassifier evaluation."""
    if isinstance(model, XGBRegressor):
        y_pred_cont = model.predict(X_test)
        y_pred = (y_pred_cont > threshold).astype(int)
        y_true = (np.asarray(y_test) > threshold).astype(int)
    else:
        y_pred = model.predict(X_test)
        y_true = np.asarray(y_test)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    try:
        if isinstance(model, XGBRegressor):
            auc = roc_auc_score(y_true, model.predict(X_test))
        else:
            auc = roc_auc_score(y_true, model.predict_proba(X_test)[:, 1])
    except ValueError:
        auc = float('nan')  # only one class present in y_true

    n_test = len(y_true)
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc_roc': auc,
        'confusion_matrix': cm.tolist(),
        'false_positive_rate': fp / (fp + tn) if (fp + tn) else 0.0,
        'false_negative_rate': fn / (fn + tp) if (fn + tp) else 0.0,
        'n_test': n_test,
        'n_correct': int(tp + tn),
    }


def get_shap_values(model, features: np.ndarray) -> np.ndarray:
    """Exact (not approximate) SHAP via TreeExplainer, since it's a tree
    model. For classifier, take the class-1 SHAP values."""
    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(features)
    if isinstance(raw, list):
        # older shap API: list per class -> take class 1
        return np.asarray(raw[1])
    raw = np.asarray(raw)
    if raw.ndim == 3:
        # (n_samples, n_features, n_classes) -> class 1
        return raw[:, :, 1]
    return raw


def get_shap_for_single(model, features: dict) -> dict:
    """dict -> array in FEATURE_NAMES order -> SHAP values -> dict keyed by
    feature name."""
    arr = np.array([[features[name] for name in FEATURE_NAMES]], dtype=float)
    shap_vals = get_shap_values(model, arr)[0]
    return {name: float(val) for name, val in zip(FEATURE_NAMES, shap_vals)}


def get_feature_importance(model) -> dict:
    """model.feature_importances_, gain-based."""
    return {
        name: float(imp)
        for name, imp in zip(FEATURE_NAMES, model.feature_importances_)
    }


def save_model(model, path: str = DEFAULT_MODEL_PATH) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)


def load_model(path: str = DEFAULT_MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No trained model found at '{path}'. Run `python -m src.train` first."
        )
    with open(path, 'rb') as f:
        return pickle.load(f)

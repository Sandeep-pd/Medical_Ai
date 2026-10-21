"""Synthetic demonstration classifier and prediction helper."""

from functools import lru_cache
from typing import Sequence

import numpy as np
import pandas as pd


@lru_cache(maxsize=1)
def _train_and_evaluate_demo_model() -> tuple[object, dict]:
    try:
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
        from sklearn.model_selection import train_test_split
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError(
            "Model evaluation requires scikit-learn and xgboost."
        ) from exc

    rng = np.random.default_rng(42)
    data = pd.DataFrame(rng.normal(size=(2000, 4)), columns=[f"feature_{i}" for i in range(1, 5)])
    logits = 1.2 * data["feature_1"] - 0.8 * data["feature_2"] + 0.5 * data["feature_3"]
    probabilities = 1 / (1 + np.exp(-logits))
    target = (rng.random(len(data)) < probabilities).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=0.2, random_state=42, stratify=target
    )
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    prediction_scores = model.predict_proba(X_test)[:, 1]
    evaluation = {
        "dataset": "synthetic demonstration data",
        "training_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, prediction_scores)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
        "feature_importance": {
            feature: float(importance)
            for feature, importance in zip(data.columns, model.feature_importances_)
        },
        "random_seed": 42,
        "test_size": 0.2,
    }
    return model, evaluation


def evaluate_demo_model() -> dict:
    """Return holdout metrics for the synthetic demonstration classifier."""
    _, evaluation = _train_and_evaluate_demo_model()
    return evaluation


def predict_demo(features: Sequence[float]) -> float:
    if len(features) != 4:
        raise ValueError("Exactly four feature values are required.")
    if not all(np.isfinite(value) for value in features):
        raise ValueError("Feature values must be finite numbers.")
    model, _ = _train_and_evaluate_demo_model()
    return float(model.predict_proba(pd.DataFrame([features], columns=[f"feature_{i}" for i in range(1, 5)]))[0, 1])

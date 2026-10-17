import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score


def evaluate_classifier(model, X, y, threshold: float = 0.5) -> dict:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, proba)),
        "average_precision": float(average_precision_score(y, proba)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }


def cross_validate(model, X, y, folds: int = 5) -> dict:
    cv = StratifiedKFold(folds, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    return {"cv_roc_auc_mean": float(np.mean(scores)), "cv_roc_auc_std": float(np.std(scores))}


def evaluate_anomaly(detector, X, y) -> dict:
    scores = -detector.decision_function(X)  # higher = more anomalous
    return {"anomaly_roc_auc": float(roc_auc_score(y, scores))}

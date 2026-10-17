import json
from pathlib import Path

import joblib
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from app.config import settings
from app.ml.evaluate import cross_validate, evaluate_anomaly, evaluate_classifier
from app.ml.features import FEATURES, add_features, make_dataset


def _pipeline(estimator) -> Pipeline:
    return Pipeline([
        ("features", FunctionTransformer(add_features)),
        ("scale", StandardScaler()),
        ("model", estimator),
    ])


def train(n: int = 6000, seed: int = 42, out_dir: Path | None = None) -> dict:
    out_dir = Path(out_dir or settings.models_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = make_dataset(n, seed)
    X, y = df[FEATURES], df["is_fraud"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)

    clf = _pipeline(RandomForestClassifier(
        n_estimators=150, class_weight="balanced_subsample", n_jobs=-1, random_state=seed))
    clf.fit(X_tr, y_tr)

    # Anomaly detector learns what "normal" looks like (legitimate transactions only).
    detector = _pipeline(IsolationForest(n_estimators=150, contamination=0.04, random_state=seed))
    detector.fit(X_tr[y_tr == 0])

    metrics = {
        **evaluate_classifier(clf, X_te, y_te, settings.fraud_threshold),
        **cross_validate(_pipeline(RandomForestClassifier(
            n_estimators=60, class_weight="balanced_subsample", n_jobs=-1, random_state=seed)),
            X_tr, y_tr),
        **evaluate_anomaly(detector, X_te, y_te),
        "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
    }
    bundle = {"classifier": clf, "anomaly": detector, "metrics": metrics}
    joblib.dump(bundle, out_dir / "models.joblib")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return bundle


if __name__ == "__main__":
    print(json.dumps(train()["metrics"], indent=2))

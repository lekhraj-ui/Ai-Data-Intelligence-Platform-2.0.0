import joblib
import pandas as pd

from app.config import settings
from app.ml.features import FEATURES
from app.ml.train import train


class ModelService:
    """Loads the trained bundle (training one on first run) and serves predictions."""

    def __init__(self, models_dir=None):
        self.models_dir = models_dir or settings.models_dir
        self.bundle: dict | None = None

    def load(self) -> "ModelService":
        path = self.models_dir / "models.joblib"
        self.bundle = joblib.load(path) if path.exists() else train(out_dir=self.models_dir)
        return self

    @property
    def metrics(self) -> dict:
        return self.bundle["metrics"]

    def predict(self, record: dict) -> dict:
        df = pd.DataFrame([{k: float(record[k]) for k in FEATURES}])
        proba = float(self.bundle["classifier"].predict_proba(df)[0, 1])
        detector = self.bundle["anomaly"]
        return {
            "fraud_probability": round(proba, 4),
            "is_fraud": proba >= settings.fraud_threshold,
            "threshold": settings.fraud_threshold,
            "anomaly_score": round(float(-detector.decision_function(df)[0]), 4),
            "is_anomaly": bool(detector.predict(df)[0] == -1),
        }

import numpy as np
import pandas as pd

FEATURES = ["amount", "hour", "tx_count_24h", "merchant_risk", "distance_km"]
MODEL_FEATURES = FEATURES + ["log_amount", "is_night", "amount_per_tx"]


def make_dataset(n: int = 6000, seed: int = 42, fraud_rate: float = 0.04) -> pd.DataFrame:
    """Synthetic card transactions; fraud skews to night, high amount, far away, risky merchants."""
    rng = np.random.default_rng(seed)
    n_fraud = int(n * fraud_rate)
    n_ok = n - n_fraud
    ok = pd.DataFrame({
        "amount": rng.lognormal(3.5, 0.8, n_ok),
        "hour": rng.integers(7, 23, n_ok),
        "tx_count_24h": rng.poisson(3, n_ok),
        "merchant_risk": rng.beta(2, 8, n_ok),
        "distance_km": rng.exponential(8, n_ok),
        "is_fraud": 0,
    })
    fr = pd.DataFrame({
        "amount": rng.lognormal(5.5, 0.9, n_fraud),
        "hour": rng.choice([0, 1, 2, 3, 4, 23, 14], n_fraud, p=[.2, .2, .2, .15, .1, .1, .05]),
        "tx_count_24h": rng.poisson(9, n_fraud),
        "merchant_risk": rng.beta(6, 3, n_fraud),
        "distance_km": rng.exponential(80, n_fraud),
        "is_fraud": 1,
    })
    return pd.concat([ok, fr], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering step (module-level so the fitted pipeline can be pickled)."""
    out = df[FEATURES].astype(float).copy()
    out["log_amount"] = np.log1p(out["amount"])
    out["is_night"] = ((out["hour"] >= 23) | (out["hour"] <= 5)).astype(float)
    out["amount_per_tx"] = out["amount"] / (out["tx_count_24h"] + 1)
    return out[MODEL_FEATURES]

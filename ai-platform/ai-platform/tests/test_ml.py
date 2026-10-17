from app.ml.train import train


def test_training_quality(tmp_path):
    metrics = train(n=3000, out_dir=tmp_path)["metrics"]
    assert metrics["roc_auc"] > 0.9
    assert metrics["anomaly_roc_auc"] > 0.8
    assert (tmp_path / "models.joblib").exists()

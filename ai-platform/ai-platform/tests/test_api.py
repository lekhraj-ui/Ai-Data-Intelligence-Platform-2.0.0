FRAUD = {"amount": 2500, "hour": 3, "tx_count_24h": 12, "merchant_risk": 0.9, "distance_km": 400}
NORMAL = {"amount": 25, "hour": 14, "tx_count_24h": 2, "merchant_risk": 0.1, "distance_km": 3}


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_predict(client):
    assert client.post("/predict", json=FRAUD).json()["is_fraud"] is True
    assert client.post("/predict", json=NORMAL).json()["is_fraud"] is False


def test_predict_validation(client):
    assert client.post("/predict", json={**NORMAL, "hour": 99}).status_code == 422


def test_rag(client):
    r = client.post("/rag/query", json={"question": "Which algorithm is used for anomaly detection?"})
    body = r.json()
    assert "IsolationForest" in body["answer"]
    assert body["sources"][0]["source"] == "fraud_detection.md"


def test_agent_multi_step(client):
    q = "calculate 12*(3+4); what roles exist in the multi-agent system?"
    body = client.post("/agent/run", json={"query": q}).json()
    assert len(body["tasks"]) == 2
    assert "= 84" in body["answer"]
    assert body["review"]["approved"] is True


def test_events_and_metrics(client):
    assert len(client.get("/events").json()) > 0
    assert "http_requests_total" in client.get("/metrics").text

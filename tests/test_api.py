def test_healthz(client):
    assert client.get("/healthz").json() == {"status": "ok"}


def test_readyz(client):
    r = client.get("/readyz")
    assert r.status_code == 200
    assert r.json()["db"] is True


def test_classify_then_fetch(client):
    body = {"ad_id": "hi-01", "language": "hi", "transcript": "तेज़ इंटरनेट", "ocr_text": "VAYU"}
    r = client.post("/v1/classify", json=body)
    assert r.status_code == 200
    data = r.json()
    assert data["result"]["brand"] == "Vayu Telecom"
    assert data["prompt_version"] == "v2"
    assert "x-request-id" in r.headers

    got = client.get("/v1/classifications/hi-01")
    assert got.status_code == 200
    assert got.json()["category"] == "telecom"


def test_request_id_is_propagated(client):
    r = client.get("/healthz", headers={"x-request-id": "trace-me-123"})
    assert r.headers["x-request-id"] == "trace-me-123"


def test_unknown_ad_returns_404(client):
    assert client.get("/v1/classifications/nope").status_code == 404


def test_invalid_input_returns_422(client):
    r = client.post("/v1/classify", json={"ad_id": "", "language": "fr"})
    assert r.status_code == 422


def test_model_returning_junk_gives_502(client, fake_llm):
    fake_llm.fail_first_n = 100
    r = client.post("/v1/classify", json={"ad_id": "x", "transcript": "something"})
    assert r.status_code == 502

"""Failure-mode tests: the service must degrade, not fall over."""

from fastapi.testclient import TestClient

from app.cache import RedisCache
from app.db import make_engine
from app.main import create_app
from tests.test_cache import BrokenRedis


def test_classify_survives_redis_outage(settings, fake_llm):
    app = create_app(
        settings,
        llm=fake_llm,
        cache=RedisCache(client=BrokenRedis()),
        engine=make_engine(settings.database_url),
    )
    with TestClient(app) as c:
        r = c.post("/v1/classify", json={"ad_id": "x", "ocr_text": "PIXELON"})
        assert r.status_code == 200
        assert r.json()["result"]["brand"] == "Pixelon"
        ready = c.get("/readyz")
        assert ready.status_code == 200  # cache down != not ready
        assert ready.json() == {"db": True, "cache": False}

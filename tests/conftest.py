import pytest
from fastapi.testclient import TestClient

from app.cache import InMemoryCache
from app.config import Settings
from app.db import make_engine
from app.llm import FakeLLM
from app.main import create_app


@pytest.fixture
def settings() -> Settings:
    # _env_file=None: tests must not depend on whatever is in a developer's .env
    return Settings(
        _env_file=None, database_url="sqlite:///:memory:", llm_provider="fake", prompt_version="v2"
    )


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def client(settings, fake_llm):
    app = create_app(
        settings, llm=fake_llm, cache=InMemoryCache(), engine=make_engine(settings.database_url)
    )
    with TestClient(app) as c:  # `with` runs the lifespan (startup/shutdown)
        yield c

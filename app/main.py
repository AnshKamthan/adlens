"""HTTP service. Run locally:  uv run uvicorn app.main:app --reload"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy import Engine
from sqlalchemy.orm import sessionmaker
from starlette.concurrency import run_in_threadpool

from app.cache import Cache, InMemoryCache, RedisCache
from app.classifier import ClassificationError, Classifier
from app.config import Settings, get_settings
from app.db import db_ping, get_latest, init_db, make_engine, save_classification
from app.llm import FakeLLM, LLMClient, LLMError, OpenAICompatClient
from app.observability import request_context_middleware, setup_logging
from app.schemas import AdInput, ClassifyResponse, StoredClassification

log = logging.getLogger("adlens.api")


def build_llm(settings: Settings) -> LLMClient:
    if settings.llm_provider == "fake":
        return FakeLLM()
    return OpenAICompatClient(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key.get_secret_value(),
        model=settings.llm_model,
        timeout_s=settings.llm_timeout_s,
        max_retries=settings.llm_max_retries,
    )


def build_cache(settings: Settings) -> Cache:
    return RedisCache(settings.redis_url) if settings.redis_url else InMemoryCache()


def create_app(
    settings: Settings | None = None,
    *,
    llm: LLMClient | None = None,
    cache: Cache | None = None,
    engine: Engine | None = None,
) -> FastAPI:
    """App factory. Tests inject fakes; production builds real clients from settings."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        setup_logging(settings.log_level)
        st = app.state
        st.settings = settings
        st.engine = engine or make_engine(settings.database_url)
        init_db(st.engine)  # prod: run Alembic migrations as a separate deploy step instead
        st.sessions = sessionmaker(st.engine, expire_on_commit=False)
        st.llm = llm or build_llm(settings)
        st.cache = cache or build_cache(settings)
        st.classifier = Classifier(st.llm, st.cache, settings.prompt_version, settings.cache_ttl_s)
        log.info(
            "startup",
            extra={
                "env": settings.app_env,
                "model": st.llm.model,
                "prompt": settings.prompt_version,
            },
        )
        yield
        await st.llm.aclose()
        await st.cache.aclose()
        st.engine.dispose()

    app = FastAPI(title="adlens", version="0.1.0", lifespan=lifespan)
    app.middleware("http")(request_context_middleware)

    @app.get("/healthz")
    def healthz() -> dict:
        """Liveness: the process is up. Must NOT check dependencies (or k8s restarts
        every pod when the DB blips)."""
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz(request: Request):
        """Readiness: can this pod serve traffic right now?"""
        st = request.app.state
        db_ok = await run_in_threadpool(db_ping, st.engine)
        cache_ok = await st.cache.ping()  # reported, not required: cache is optional
        body = {"db": db_ok, "cache": cache_ok}
        return JSONResponse(body, status_code=200 if db_ok else 503)

    @app.post("/v1/classify", response_model=ClassifyResponse)
    async def classify(ad: AdInput, request: Request) -> ClassifyResponse:
        st = request.app.state
        t0 = time.perf_counter()
        try:
            result, cached = await st.classifier.classify(ad)
        except LLMError as e:
            log.error("llm_unavailable", extra={"ad_id": ad.ad_id, "error": str(e)[:300]})
            raise HTTPException(status_code=503, detail="upstream model unavailable") from e
        except ClassificationError as e:
            log.error("llm_invalid_output", extra={"ad_id": ad.ad_id, "error": str(e)[:300]})
            raise HTTPException(status_code=502, detail="model returned invalid output") from e

        # Sync DB call inside an async endpoint -> run it in a thread so it doesn't block
        # the event loop (which would stall every other in-flight request).
        await run_in_threadpool(
            save_classification,
            st.sessions,
            ad.ad_id,
            result,
            st.llm.model,
            settings.prompt_version,
        )
        latency_ms = round((time.perf_counter() - t0) * 1000)
        log.info(
            "classified",
            extra={
                "ad_id": ad.ad_id,
                "brand": result.brand,
                "category": result.category,
                "confidence": result.confidence,
                "cached": cached,
                "prompt_version": settings.prompt_version,
                "model": st.llm.model,
            },
        )
        return ClassifyResponse(
            ad_id=ad.ad_id,
            result=result,
            model=st.llm.model,
            prompt_version=settings.prompt_version,
            cached=cached,
            latency_ms=latency_ms,
        )

    @app.get("/v1/classifications/{ad_id}", response_model=StoredClassification)
    def get_classification(ad_id: str, request: Request) -> StoredClassification:
        # plain `def`: FastAPI runs it in a threadpool automatically
        rec = get_latest(request.app.state.sessions, ad_id)
        if rec is None:
            raise HTTPException(status_code=404, detail=f"no classification for {ad_id}")
        return rec

    return app


app = create_app()

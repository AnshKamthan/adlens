"""Cache layer. The cache is an optimisation, never a dependency:
if Redis is down, requests get slower, not broken."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Protocol

import redis.asyncio as redis
from redis.exceptions import RedisError

from app.schemas import AdInput

log = logging.getLogger("adlens.cache")


def make_cache_key(*, prompt_version: str, model: str, ad: AdInput) -> str:
    """Key on the *content* that determines the answer, not on ad_id.

    The same creative airs thousands of times under different airing ids; keying on
    content means we pay for the LLM once per creative. prompt_version and model are in
    the key, so shipping a new prompt or model naturally bypasses stale entries.
    """
    payload = json.dumps(
        {"pv": prompt_version, "m": model, "t": ad.transcript, "o": ad.ocr_text, "l": ad.language},
        sort_keys=True,
        ensure_ascii=False,
    )
    return "adlens:cls:" + hashlib.sha256(payload.encode()).hexdigest()


class Cache(Protocol):
    async def get(self, key: str) -> str | None: ...
    async def set(self, key: str, value: str, ttl_s: int) -> None: ...
    async def ping(self) -> bool: ...
    async def aclose(self) -> None: ...


class NullCache:
    """Always misses. Used by evals: you must measure the model, not the cache."""

    async def get(self, key: str) -> str | None:
        return None

    async def set(self, key: str, value: str, ttl_s: int) -> None:
        return None

    async def ping(self) -> bool:
        return True

    async def aclose(self) -> None:
        return None


class InMemoryCache:
    """Per-process cache for local dev/tests. Not shared across replicas."""

    def __init__(self) -> None:
        self._data: dict[str, tuple[float, str]] = {}

    async def get(self, key: str) -> str | None:
        item = self._data.get(key)
        if item is None:
            return None
        expires_at, value = item
        if time.monotonic() > expires_at:
            del self._data[key]
            return None
        return value

    async def set(self, key: str, value: str, ttl_s: int) -> None:
        self._data[key] = (time.monotonic() + ttl_s, value)

    async def ping(self) -> bool:
        return True

    async def aclose(self) -> None:
        return None


class RedisCache:
    def __init__(self, url: str | None = None, *, client: redis.Redis | None = None) -> None:
        if client is None and url is None:
            raise ValueError("RedisCache needs a url or a client")
        # short socket timeout: a slow cache must not make the API slow
        self._r = client or redis.from_url(url, decode_responses=True, socket_timeout=0.5)

    async def get(self, key: str) -> str | None:
        try:
            return await self._r.get(key)
        except RedisError as e:
            log.warning("cache_get_failed", extra={"error": repr(e)})
            return None

    async def set(self, key: str, value: str, ttl_s: int) -> None:
        try:
            await self._r.set(key, value, ex=ttl_s)  # always set a TTL
        except RedisError as e:
            log.warning("cache_set_failed", extra={"error": repr(e)})

    async def ping(self) -> bool:
        try:
            return bool(await self._r.ping())
        except RedisError:
            return False

    async def aclose(self) -> None:
        await self._r.aclose()

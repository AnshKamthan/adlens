import fakeredis
from redis.exceptions import ConnectionError as RedisConnectionError

from app.cache import RedisCache


async def test_redis_cache_roundtrip_with_ttl():
    r = fakeredis.FakeAsyncRedis(decode_responses=True)
    cache = RedisCache(client=r)
    await cache.set("k", "v", ttl_s=60)
    assert await cache.get("k") == "v"
    assert 0 < await r.ttl("k") <= 60


class BrokenRedis:
    async def get(self, *a, **k):
        raise RedisConnectionError("down")

    async def set(self, *a, **k):
        raise RedisConnectionError("down")

    async def ping(self):
        raise RedisConnectionError("down")

    async def aclose(self):
        pass


async def test_redis_outage_degrades_to_cache_miss():
    cache = RedisCache(client=BrokenRedis())
    assert await cache.get("k") is None  # no exception: slower, not broken
    await cache.set("k", "v", ttl_s=60)
    assert await cache.ping() is False

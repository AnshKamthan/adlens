"""Test the HTTP client without network: httpx.MockTransport plays the provider."""

import json

import httpx
import pytest

from app.llm import LLMError, OpenAICompatClient

OK_BODY = {
    "model": "m",
    "choices": [{"message": {"role": "assistant", "content": '{"ok": true}'}}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 3},
}


def make_client(handler, max_retries=2):
    return OpenAICompatClient(
        "http://llm.test/v1",
        "secret",
        "m",
        max_retries=max_retries,
        backoff_base_s=0,
        transport=httpx.MockTransport(handler),
    )


async def test_retries_429_then_succeeds():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if len(seen) == 1:
            return httpx.Response(429, headers={"retry-after": "0"}, json={"error": "slow down"})
        return httpx.Response(200, json=OK_BODY)

    client = make_client(handler)
    resp = await client.chat([{"role": "user", "content": "hi"}], json_mode=True)
    assert len(seen) == 2
    assert resp.input_tokens == 10
    assert str(seen[0].url) == "http://llm.test/v1/chat/completions"
    assert seen[0].headers["authorization"] == "Bearer secret"
    assert json.loads(seen[0].content)["response_format"] == {"type": "json_object"}
    await client.aclose()


async def test_does_not_retry_client_errors():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(401, json={"error": "bad key"})

    client = make_client(handler)
    with pytest.raises(LLMError, match="401"):
        await client.chat([{"role": "user", "content": "hi"}])
    assert len(calls) == 1
    await client.aclose()


async def test_gives_up_after_max_retries():
    client = make_client(lambda r: httpx.Response(503), max_retries=2)
    with pytest.raises(LLMError, match="503"):
        await client.chat([{"role": "user", "content": "hi"}])
    await client.aclose()


async def test_parses_tool_calls():
    body = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {
                                "name": "get_classification",
                                "arguments": '{"ad_id": "hi-01"}',
                            },
                        }
                    ],
                }
            }
        ]
    }
    client = make_client(lambda r: httpx.Response(200, json=body))
    resp = await client.chat([{"role": "user", "content": "?"}], tools=[{"type": "function"}])
    assert resp.tool_calls[0].name == "get_classification"
    assert resp.tool_calls[0].arguments == {"ad_id": "hi-01"}
    await client.aclose()

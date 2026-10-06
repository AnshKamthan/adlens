"""LLM client layer.

Everything provider-specific lives here, behind one small interface (`LLMClient`).
The rest of the code never imports a vendor SDK, so swapping OpenAI <-> Azure <->
a self-hosted vLLM <-> an internal gateway is a config change, not a refactor.

We call the OpenAI-compatible HTTP API directly with httpx. Ollama, vLLM, LiteLLM
proxy, OpenAI and most enterprise gateways speak this format.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Protocol

import httpx

log = logging.getLogger("adlens.llm")

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class LLMError(Exception):
    """The model call failed (network, rate limit exhausted, 4xx/5xx)."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""

    def as_assistant_message(self) -> dict:
        """Echo the model's turn back into the conversation (needed for tool loops)."""
        msg: dict = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                }
                for c in self.tool_calls
            ]
        return msg


class LLMClient(Protocol):
    model: str

    async def chat(
        self, messages: list[dict], *, tools: list[dict] | None = None, json_mode: bool = False
    ) -> LLMResponse: ...

    async def aclose(self) -> None: ...


class OpenAICompatClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        timeout_s: float = 30.0,
        max_retries: int = 2,
        backoff_base_s: float = 0.5,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.model = model
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self._http = httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(timeout_s, connect=5.0),
            transport=transport,
        )

    async def chat(
        self, messages: list[dict], *, tools: list[dict] | None = None, json_mode: bool = False
    ) -> LLMResponse:
        payload: dict = {"model": self.model, "messages": messages, "temperature": 0}
        if tools:
            payload["tools"] = tools
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        for attempt in range(self.max_retries + 1):
            t0 = time.perf_counter()
            try:
                r = await self._http.post("chat/completions", json=payload)
            except httpx.TransportError as e:  # timeouts, connection resets, DNS
                if attempt == self.max_retries:
                    raise LLMError(f"transport error after {attempt + 1} attempts: {e!r}") from e
                await self._backoff(attempt, None)
                continue

            latency_ms = round((time.perf_counter() - t0) * 1000)
            if r.status_code in RETRYABLE_STATUS and attempt < self.max_retries:
                log.warning("llm_retry", extra={"status": r.status_code, "attempt": attempt})
                await self._backoff(attempt, r.headers.get("retry-after"))
                continue
            if r.status_code >= 400:
                # 400/401/403/404 are bugs or config problems: retrying won't help.
                raise LLMError(f"LLM HTTP {r.status_code}: {r.text[:300]}")

            resp = self._parse(r.json())
            log.info(
                "llm_call",
                extra={
                    "model": resp.model,
                    "latency_ms": latency_ms,
                    "input_tokens": resp.input_tokens,
                    "output_tokens": resp.output_tokens,
                    "tool_calls": len(resp.tool_calls),
                },
            )
            return resp
        raise LLMError("unreachable")  # pragma: no cover

    async def _backoff(self, attempt: int, retry_after: str | None) -> None:
        try:
            delay = float(retry_after) if retry_after is not None else None
        except ValueError:
            delay = None
        if delay is None:
            # exponential backoff with jitter, so 50 workers don't retry in lockstep
            delay = min(8.0, self.backoff_base_s * 2**attempt) * random.uniform(0.5, 1.5)
        await asyncio.sleep(delay)

    def _parse(self, data: dict) -> LLMResponse:
        msg = data["choices"][0]["message"]
        calls = []
        for c in msg.get("tool_calls") or []:
            raw_args = c["function"].get("arguments") or "{}"
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {"__invalid_json__": raw_args}  # let the caller report it to the model
            calls.append(ToolCall(id=c["id"], name=c["function"]["name"], arguments=args))
        usage = data.get("usage") or {}
        return LLMResponse(
            content=msg.get("content"),
            tool_calls=calls,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            model=data.get("model", self.model),
        )

    async def aclose(self) -> None:
        await self._http.aclose()


# --------------------------------------------------------------------------------------
# Offline stand-in. NOT a model: a deterministic keyword matcher that only "sees" what
# the prompt shows it. That makes prompt changes (v1 -> v2) change its answers, so the
# whole pipeline, tests and eval harness run with no API key and no cost.
# --------------------------------------------------------------------------------------

FAKE_BRANDS: dict[str, tuple[str, str]] = {
    "zento": ("Zento Motors", "automotive"),
    "kiranmart": ("Kiranmart", "ecommerce"),
    "glowveda": ("GlowVeda", "fmcg_personal_care"),
    "nimbufresh": ("NimbuFresh", "fmcg_food"),
    "vayu": ("Vayu Telecom", "telecom"),
    "orbitel": ("Orbitel", "telecom"),
    "suraksha": ("Suraksha Bank", "finance"),
    "pixelon": ("Pixelon", "electronics"),
}


class FakeLLM:
    def __init__(self, model: str = "fake-keyword-v0", fail_first_n: int = 0) -> None:
        self.model = model
        self.fail_first_n = fail_first_n  # simulate a model that returns junk N times
        self.calls = 0

    async def chat(
        self, messages: list[dict], *, tools: list[dict] | None = None, json_mode: bool = False
    ) -> LLMResponse:
        self.calls += 1
        user_text = next(m["content"] for m in messages if m["role"] == "user")
        tokens_in = sum(len(str(m.get("content") or "")) for m in messages) // 4

        if self.calls <= self.fail_first_n:
            return LLMResponse(
                "Sure! The brand is probably... let me think.", [], tokens_in, 12, self.model
            )

        text = user_text.lower()
        hits = {k: text.find(k) for k in FAKE_BRANDS if k in text}
        if hits:
            brand, category = FAKE_BRANDS[min(hits, key=hits.get)]  # earliest mention wins
            out = {
                "brand": brand,
                "category": category,
                "confidence": 0.9,
                "rationale": f"'{brand}' appears in the provided text.",
            }
        else:
            out = {
                "brand": "unknown",
                "category": "other",
                "confidence": 0.2,
                "rationale": "No brand name found in the provided text.",
            }
        content = json.dumps(out)
        return LLMResponse(content, [], tokens_in, len(content) // 4, self.model)

    async def aclose(self) -> None:
        return None

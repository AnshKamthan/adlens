"""The AI component: prompt -> LLM -> validate -> (repair) -> cache.

Two different retry loops exist in this codebase, on purpose:
  * transport retries (llm.py): the request failed (429, 503, timeout) -> same request again
  * repair retries (here): the request succeeded but the output is invalid -> tell the
    model what was wrong and ask again
"""

from __future__ import annotations

import logging

from pydantic import ValidationError

from app.cache import Cache, make_cache_key
from app.llm import LLMClient
from app.prompts import build_messages
from app.schemas import AdInput, Classification

log = logging.getLogger("adlens.classifier")


class ClassificationError(Exception):
    """The model kept returning output we could not validate."""


def extract_json(text: str) -> str:
    """Models sometimes wrap JSON in ```json fences or add chatter. Take the outer {...}."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in model output")
    return text[start : end + 1]


class Classifier:
    def __init__(
        self,
        llm: LLMClient,
        cache: Cache,
        prompt_version: str,
        cache_ttl_s: int = 86_400,
        max_repair_attempts: int = 1,
    ) -> None:
        self.llm = llm
        self.cache = cache
        self.prompt_version = prompt_version
        self.cache_ttl_s = cache_ttl_s
        self.max_repair_attempts = max_repair_attempts

    async def classify(self, ad: AdInput) -> tuple[Classification, bool]:
        """Returns (result, was_cached)."""
        key = make_cache_key(prompt_version=self.prompt_version, model=self.llm.model, ad=ad)
        cached = await self.cache.get(key)
        if cached is not None:
            try:
                return Classification.model_validate_json(cached), True
            except ValidationError:
                log.warning("cache_entry_invalid", extra={"key": key})  # schema changed

        messages = build_messages(ad, self.prompt_version)
        last_error: Exception | None = None
        for attempt in range(self.max_repair_attempts + 1):
            resp = await self.llm.chat(messages, json_mode=True)
            raw = resp.content or ""
            try:
                result = Classification.model_validate_json(extract_json(raw))
            except (ValueError, ValidationError) as e:
                last_error = e
                log.warning(
                    "llm_output_invalid",
                    extra={"attempt": attempt, "ad_id": ad.ad_id, "error": str(e)[:200]},
                )
                messages = [
                    *messages,
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": f"That output was invalid: {str(e)[:300]}. "
                        "Reply with ONLY the JSON object described in the instructions.",
                    },
                ]
                continue
            await self.cache.set(key, result.model_dump_json(), self.cache_ttl_s)
            return result, False

        raise ClassificationError(f"invalid model output after retries: {last_error}")

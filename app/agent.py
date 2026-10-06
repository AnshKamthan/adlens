"""A minimal tool-calling agent loop: what LangGraph / agent SDKs do under the hood.

    model sees question + tool schemas
      -> model asks to call a tool (name + JSON args)
      -> WE execute it (the model never touches the DB) and append the result
      -> repeat until the model answers in plain text, or we hit max_steps

Run (needs a tool-calling model, e.g. Ollama llama3.1 or any OpenAI-compatible API):
    uv run python -m app.agent "What was ad hi-01 classified as?"
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass, field

from sqlalchemy.orm import sessionmaker

from app.db import get_latest, list_recent
from app.llm import LLMClient, ToolCall
from app.schemas import CATEGORIES

log = logging.getLogger("adlens.agent")

SYSTEM = (
    "You answer questions about ad classifications stored in our database. "
    "Use the tools to look things up; never guess ids or values. "
    "If a tool returns an error, say so plainly. Answer in one or two sentences."
)

TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "get_classification",
            "description": "Get the latest stored classification for one ad by its ad_id.",
            "parameters": {
                "type": "object",
                "properties": {"ad_id": {"type": "string"}},
                "required": ["ad_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_recent",
            "description": "List recent classifications, optionally filtered by category.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "enum": list(CATEGORIES)},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                },
            },
        },
    },
]


@dataclass
class AgentResult:
    answer: str
    steps: int
    trace: list[dict] = field(default_factory=list)  # every tool call + result, for debugging


class Agent:
    def __init__(self, llm: LLMClient, sessions: sessionmaker, max_steps: int = 5) -> None:
        self.llm = llm
        self.max_steps = max_steps
        # Read-only tools. Write tools (update a label, send an email) need much more care:
        # confirmation, permissions, idempotency, audit logs.
        self.tools: dict[str, Callable[..., dict]] = {
            "get_classification": lambda ad_id: self._get(sessions, ad_id),
            "list_recent": lambda category=None, limit=5: {
                "items": [r.model_dump() for r in list_recent(sessions, category, min(limit, 20))]
            },
        }

    @staticmethod
    def _get(sessions: sessionmaker, ad_id: str) -> dict:
        rec = get_latest(sessions, ad_id)
        return rec.model_dump() if rec else {"error": f"no classification found for {ad_id}"}

    def _execute(self, call: ToolCall) -> dict:
        fn = self.tools.get(call.name)
        if fn is None:
            return {"error": f"unknown tool: {call.name}"}
        if "__invalid_json__" in call.arguments:
            return {"error": "arguments were not valid JSON"}
        try:
            return fn(**call.arguments)
        except TypeError as e:  # wrong/missing arguments from the model
            return {"error": f"bad arguments: {e}"}

    async def run(self, question: str) -> AgentResult:
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question},
        ]
        trace: list[dict] = []
        for step in range(1, self.max_steps + 1):
            resp = await self.llm.chat(messages, tools=TOOL_SPECS)
            if not resp.tool_calls:
                return AgentResult(answer=resp.content or "", steps=step, trace=trace)
            messages.append(resp.as_assistant_message())
            for call in resp.tool_calls:
                result = await asyncio.to_thread(self._execute, call)  # DB is sync
                trace.append({"tool": call.name, "args": call.arguments, "result": result})
                log.info("tool_call", extra={"tool": call.name, "step": step})
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)}
                )
        return AgentResult(
            answer="Stopped: step limit reached without a final answer.",
            steps=self.max_steps,
            trace=trace,
        )


async def _main(question: str) -> None:
    from app.config import get_settings
    from app.db import make_engine
    from app.main import build_llm

    settings = get_settings()
    if settings.llm_provider == "fake":
        print("The agent needs a real tool-calling model. Set LLM_PROVIDER=openai_compat in .env")
        return
    llm = build_llm(settings)
    try:
        agent = Agent(llm, sessionmaker(make_engine(settings.database_url)))
        result = await agent.run(question)
        print(json.dumps(result.trace, indent=2, ensure_ascii=False))
        print("\nANSWER:", result.answer)
    finally:
        await llm.aclose()


if __name__ == "__main__":
    asyncio.run(_main(" ".join(sys.argv[1:]) or "List the 3 most recent telecom ads."))

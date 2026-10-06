"""Agent loop tested with a scripted model: we control exactly what the 'LLM' says."""

from sqlalchemy.orm import sessionmaker

from app.agent import Agent
from app.db import init_db, make_engine, save_classification
from app.llm import LLMResponse, ToolCall
from app.schemas import Classification


class ScriptedLLM:
    model = "scripted"

    def __init__(self, script: list[LLMResponse]):
        self.script = list(script)
        self.seen: list[list[dict]] = []

    async def chat(self, messages, *, tools=None, json_mode=False):
        self.seen.append(list(messages))
        return self.script.pop(0)

    async def aclose(self):
        pass


def sessions_with_one_row():
    engine = make_engine("sqlite:///:memory:")
    init_db(engine)
    sessions = sessionmaker(engine, expire_on_commit=False)
    save_classification(
        sessions,
        "hi-01",
        Classification(brand="Vayu Telecom", category="telecom", confidence=0.9),
        "m",
        "v2",
    )
    return sessions


async def test_agent_calls_tool_then_answers():
    llm = ScriptedLLM(
        [
            LLMResponse(None, [ToolCall("c1", "get_classification", {"ad_id": "hi-01"})]),
            LLMResponse("hi-01 is a Vayu Telecom ad (telecom)."),
        ]
    )
    result = await Agent(llm, sessions_with_one_row()).run("What is hi-01?")
    assert result.steps == 2
    assert result.trace[0]["result"]["brand"] == "Vayu Telecom"
    tool_msg = llm.seen[1][-1]  # the 2nd model call must see the tool result
    assert tool_msg["role"] == "tool" and "Vayu Telecom" in tool_msg["content"]


async def test_agent_reports_tool_errors_to_model():
    llm = ScriptedLLM(
        [
            LLMResponse(None, [ToolCall("c1", "delete_everything", {})]),
            LLMResponse("I can't do that."),
        ]
    )
    result = await Agent(llm, sessions_with_one_row()).run("drop the table")
    assert "unknown tool" in result.trace[0]["result"]["error"]


async def test_agent_stops_at_max_steps():
    loop = [LLMResponse(None, [ToolCall(f"c{i}", "list_recent", {})]) for i in range(10)]
    result = await Agent(ScriptedLLM(loop), sessions_with_one_row(), max_steps=3).run("loop")
    assert result.steps == 3
    assert result.answer.startswith("Stopped")

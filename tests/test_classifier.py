import pytest

from app.cache import InMemoryCache
from app.classifier import ClassificationError, Classifier, extract_json
from app.llm import FakeLLM
from app.prompts import build_messages
from app.schemas import AdInput

TAMIL_AD = AdInput(
    ad_id="ta-01", language="ta", transcript="வேகமான இணையம்.", ocr_text="VAYU TELECOM 5G"
)


def test_v1_prompt_ignores_ocr_and_v2_includes_it():
    v1 = build_messages(TAMIL_AD, "v1")[-1]["content"]
    v2 = build_messages(TAMIL_AD, "v2")[-1]["content"]
    assert "VAYU" not in v1
    assert "VAYU" in v2


async def test_v2_uses_ocr_to_find_brand():
    result, cached = await Classifier(FakeLLM(), InMemoryCache(), "v2").classify(TAMIL_AD)
    assert result.brand == "Vayu Telecom"
    assert result.category == "telecom"
    assert cached is False


async def test_second_call_is_served_from_cache():
    llm = FakeLLM()
    clf = Classifier(llm, InMemoryCache(), "v2")
    await clf.classify(TAMIL_AD)
    # same creative, different airing id -> same cache key
    _, cached = await clf.classify(TAMIL_AD.model_copy(update={"ad_id": "ta-01-airing-2"}))
    assert cached is True
    assert llm.calls == 1


async def test_prompt_version_change_bypasses_cache():
    llm, cache = FakeLLM(), InMemoryCache()
    await Classifier(llm, cache, "v1").classify(TAMIL_AD)
    _, cached = await Classifier(llm, cache, "v2").classify(TAMIL_AD)
    assert cached is False


async def test_invalid_output_is_repaired_once():
    llm = FakeLLM(fail_first_n=1)
    result, _ = await Classifier(llm, InMemoryCache(), "v2").classify(TAMIL_AD)
    assert result.brand == "Vayu Telecom"
    assert llm.calls == 2


async def test_gives_up_after_repair_attempts():
    llm = FakeLLM(fail_first_n=10)
    with pytest.raises(ClassificationError):
        await Classifier(llm, InMemoryCache(), "v2", max_repair_attempts=1).classify(TAMIL_AD)
    assert llm.calls == 2


@pytest.mark.parametrize(
    "raw",
    ['{"a": 1}', '```json\n{"a": 1}\n```', 'Sure! Here you go: {"a": 1} Hope that helps.'],
)
def test_extract_json_handles_common_model_wrapping(raw):
    assert extract_json(raw) == '{"a": 1}'

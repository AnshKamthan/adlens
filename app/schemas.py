"""API and LLM contracts. Pydantic models are the boundary between untrusted input
(HTTP requests, LLM output) and the rest of the code."""

from typing import Literal, get_args

from pydantic import BaseModel, Field

Language = Literal["en", "hi", "ta", "other"]
Category = Literal[
    "automotive",
    "fmcg_food",
    "fmcg_personal_care",
    "telecom",
    "finance",
    "ecommerce",
    "electronics",
    "other",
]
CATEGORIES: tuple[str, ...] = get_args(Category)


class AdInput(BaseModel):
    ad_id: str = Field(min_length=1, max_length=64)
    transcript: str = Field(
        default="", max_length=20_000, description="Speech-to-text of the audio"
    )
    ocr_text: str = Field(
        default="", max_length=5_000, description="Text read off the video frames"
    )
    language: Language = "en"


class Classification(BaseModel):
    """What we require the LLM to return. Anything else is rejected and retried."""

    brand: str = Field(min_length=1, max_length=100)
    category: Category
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(default="", max_length=500)


class ClassifyResponse(BaseModel):
    ad_id: str
    result: Classification
    model: str
    prompt_version: str
    cached: bool
    latency_ms: int


class StoredClassification(Classification):
    ad_id: str
    model: str
    prompt_version: str

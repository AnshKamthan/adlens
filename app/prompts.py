"""Prompts are code: versioned, reviewed in PRs, and evaluated before rollout.

v1: transcript only (the original behaviour).
v2: adds on-screen OCR text (ticket AIE-1423) because regional-language ads often
    show the brand name only on screen.
"""

from app.schemas import CATEGORIES, AdInput

SYSTEM_PROMPT = f"""You classify TV/digital ad creatives.
Return ONLY a JSON object with keys:
  "brand": the advertised brand name, or "unknown" if it cannot be determined,
  "category": one of {list(CATEGORIES)},
  "confidence": number between 0 and 1,
  "rationale": one short sentence.
Do not add markdown, code fences, or any text outside the JSON object."""


def build_messages(ad: AdInput, version: str) -> list[dict]:
    if version == "v1":
        user = f"Language: {ad.language}\nAudio transcript:\n{ad.transcript or '(none)'}"
    elif version == "v2":
        user = (
            f"Language: {ad.language}\n"
            f"On-screen text (OCR, may be noisy):\n{ad.ocr_text or '(none)'}\n\n"
            f"Audio transcript:\n{ad.transcript or '(none)'}"
        )
    else:
        raise ValueError(f"unknown prompt version: {version}")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]

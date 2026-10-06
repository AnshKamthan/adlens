"""Offline evaluation over a labelled golden set.

    uv run python -m evals.run_eval                              # current PROMPT_VERSION
    uv run python -m evals.run_eval --versions v1 v2             # compare prompt versions
    uv run python -m evals.run_eval --versions v2 --check evals/thresholds.json   # CI gate

Uses whatever model .env points at (FakeLLM by default -> free and deterministic).
The output is a markdown table you can paste straight into a PR description.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from app.cache import NullCache
from app.classifier import ClassificationError, Classifier
from app.config import get_settings
from app.llm import LLMClient, LLMError
from app.main import build_llm
from app.schemas import AdInput

GOLDEN = Path(__file__).parent / "golden.jsonl"


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


async def run_version(
    examples: list[dict], version: str, llm: LLMClient, concurrency: int
) -> list[dict]:
    clf = Classifier(llm, NullCache(), version)  # NullCache: measure the model, not the cache
    sem = asyncio.Semaphore(concurrency)  # respect provider rate limits

    async def one(ex: dict) -> dict:
        async with sem:
            pred, err = None, None
            try:
                res, _ = await clf.classify(AdInput(**ex["input"]))
                pred = res.model_dump()
            except (ClassificationError, LLMError) as e:
                err = str(e)[:200]
        exp = ex["expected"]
        return {
            "id": ex["id"],
            "language": ex["input"]["language"],
            "expected": exp,
            "pred": pred,
            "error": err,
            "brand_ok": bool(pred) and norm(pred["brand"]) == norm(exp["brand"]),
            "category_ok": bool(pred) and pred["category"] == exp["category"],
        }

    return await asyncio.gather(*(one(e) for e in examples))


def summarize(rows: list[dict]) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups["overall"].append(r)
        groups[r["language"]].append(r)
    return {
        g: {
            "n": len(rs),
            "brand_acc": round(sum(r["brand_ok"] for r in rs) / len(rs), 3),
            "category_acc": round(sum(r["category_ok"] for r in rs) / len(rs), 3),
            "errors": sum(r["error"] is not None for r in rs),
        }
        for g, rs in groups.items()
    }


def table(summaries: dict[str, dict[str, dict]]) -> str:
    versions = list(summaries)
    slices = list(next(iter(summaries.values())))
    head = "| slice | n | " + " | ".join(f"{v} brand | {v} category" for v in versions) + " |"
    sep = "|---|---|" + "---|---|" * len(versions)
    lines = [head, sep]
    for s in slices:
        cells = " | ".join(
            f"{summaries[v][s]['brand_acc']:.2f} | {summaries[v][s]['category_acc']:.2f}"
            for v in versions
        )
        lines.append(f"| {s} | {summaries[versions[0]][s]['n']} | {cells} |")
    return "\n".join(lines)


def diff(a: list[dict], b: list[dict], va: str, vb: str) -> str:
    by_id = {r["id"]: r for r in b}
    fixed = [r["id"] for r in a if not r["brand_ok"] and by_id[r["id"]]["brand_ok"]]
    broke = [r["id"] for r in a if r["brand_ok"] and not by_id[r["id"]]["brand_ok"]]
    out = [
        f"\nFixed by {vb} (wrong in {va}): {fixed or 'none'}",
        f"Broken by {vb} (right in {va}): {broke or 'none'}",
    ]
    for i in broke:
        p = by_id[i]["pred"] or {}
        out.append(f"  {i}: expected {by_id[i]['expected']['brand']!r}, got {p.get('brand')!r}")
    return "\n".join(out)


def check(summary: dict[str, dict], thresholds: dict) -> list[str]:
    failures = []
    for slice_name, mins in thresholds.items():
        for metric, minimum in mins.items():
            got = summary.get(slice_name, {}).get(metric)
            if got is None or got < minimum:
                failures.append(f"{slice_name}.{metric} = {got} < {minimum}")
    return failures


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--versions", nargs="+", default=None)
    ap.add_argument("--golden", type=Path, default=GOLDEN)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--check", type=Path, help="thresholds JSON; exit 1 if the LAST version fails")
    ap.add_argument("--out", type=Path, default=Path("evals/results"))
    args = ap.parse_args()

    settings = get_settings()
    versions = args.versions or [settings.prompt_version]
    examples = load(args.golden)
    llm = build_llm(settings)
    try:
        results = {v: await run_version(examples, v, llm, args.concurrency) for v in versions}
    finally:
        await llm.aclose()

    summaries = {v: summarize(rows) for v, rows in results.items()}
    print(f"model={llm.model}  examples={len(examples)}\n")
    print(table(summaries))
    if len(versions) >= 2:
        print(diff(results[versions[0]], results[versions[-1]], versions[0], versions[-1]))

    args.out.mkdir(parents=True, exist_ok=True)
    for v, rows in results.items():
        (args.out / f"{v}.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))

    if args.check:
        failures = check(summaries[versions[-1]], json.loads(args.check.read_text()))
        if failures:
            print("\nEVAL GATE FAILED:\n  " + "\n  ".join(failures))
            return 1
        print("\nEval gate passed.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

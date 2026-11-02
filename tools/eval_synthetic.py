"""Run the synthetic gold questions through ChatManager and report metrics.

Metrics per category: accuracy, median / max latency. Numbers are compared
after removing thousand separators, dates are compared as sets (recall and
precision), names by surname.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
import os
import re
import statistics
import time
from datetime import date
from typing import Any

MONTHS = {
    "январ": 1, "феврал": 2, "март": 3, "апрел": 4, "ма": 5, "июн": 6,
    "июл": 7, "август": 8, "сентябр": 9, "октябр": 10, "ноябр": 11, "декабр": 12,
}


NOT_FOUND_PHRASES: list[str] = [
    "нет информации", "не найден", "не указ", "не упомина", "отсутству", "нет данных",
    "не содерж", "нет сведений", "не приводится", "не определ",
]


def normalize_numbers(text: str) -> str:
    text = text.replace(" ", " ").replace(" ", " ")
    return re.sub(r"(?<=\d)[ ](?=\d{3}(?!\d))", "", text)


def extract_dates(text: str, default_year: int = 2025) -> set[str]:
    found: set[str] = set()
    for d, m, y in re.findall(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b", text):
        try:
            found.add(date(int(y), int(m), int(d)).isoformat())
        except ValueError:
            pass
    for d, month_word, y in re.findall(r"\b(\d{1,2})\s+([а-яё]+)(?:\s+(\d{4}))?", text.lower()):
        month = next((num for stem, num in MONTHS.items() if month_word.startswith(stem) and (stem != "ма" or month_word in {"мая", "май"})), None)
        if month is None:
            continue
        try:
            found.add(date(int(y or default_year), month, int(d)).isoformat())
        except ValueError:
            pass
    return found


def check_case(case: dict[str, Any], answer: str) -> tuple[bool, dict[str, Any]]:
    normalized = normalize_numbers(answer)
    lowered = answer.lower()
    details: dict[str, Any] = {}
    ok = True
    category = case.get("category")
    if case.get("expect_numbers"):
        missing = []
        for number in case["expect_numbers"]:
            token = str(number)
            if category == "person_days_count":
                pattern = rf"(?<![\d.]){token}(?![\d.])\s*(?:дн|дня|день|смен|раз|рабоч)|(?:дн\w*|смен\w*)\D{{0,25}}(?<![\d.]){token}(?![\d.])"
                hit = re.search(pattern, normalized.lower()) is not None
            else:
                hit = re.search(rf"(?<!\d){token}(?!\d)", normalized) is not None
            if not hit:
                missing.append(number)
        details["missing_numbers"] = missing
        ok = ok and not missing
    if case.get("forbid_numbers"):
        details["mentions_forbidden"] = [n for n in case["forbid_numbers"] if re.search(rf"(?<!\d){n}(?!\d)", normalized)]
    if case.get("expect_text"):
        missing_text = [item for item in case["expect_text"] if item.lower() not in lowered]
        details["missing_text"] = missing_text
        ok = ok and not missing_text
    if case.get("expect_any"):
        hits = [item for item in case["expect_any"] if item.lower() in lowered]
        need = int(case.get("min_hits", 1))
        details["any_hits"] = hits
        ok = ok and len(hits) >= need
    if case.get("expect_any_2"):
        # a second independent fact of the answer (e.g. place AND action)
        hits2 = [item for item in case["expect_any_2"] if item.lower() in lowered]
        details["any_hits_2"] = hits2
        ok = ok and bool(hits2)
    if case.get("expect_not_found"):
        # An honest "not in the documents" answer, not a made-up value.
        said = any(phrase in lowered for phrase in NOT_FOUND_PHRASES)
        details["said_not_found"] = said
        ok = ok and said
    if case.get("forbid_any"):
        bad = [item for item in case["forbid_any"] if item.lower() in lowered]
        details["forbidden_hits"] = bad
        ok = ok and not bad
    if "expect_dates" in case:
        expected = set(case["expect_dates"])
        got = extract_dates(answer)
        hit = expected & got
        recall = len(hit) / max(1, len(expected))
        precision = len(hit) / max(1, len(got)) if got else (1.0 if not expected else 0.0)
        details.update({"date_recall": round(recall, 2), "date_precision": round(precision, 2), "dates_missing": sorted(expected - got)[:10], "dates_extra": sorted(got - expected)[:10]})
        ok = ok and recall >= 0.9 and precision >= 0.8
    return ok, details


def build_manager(docs: Path, with_llm: bool):
    from core.chat_mode import ChatManager
    from core.document_processor import DocumentProcessor
    from core.models import AIModelLoader
    from core.search import HybridSearch

    loader = AIModelLoader()
    if not loader.load_models():
        raise RuntimeError("embedding model is not available")
    processor = DocumentProcessor(docs_folder=str(docs), recursive=True, max_file_size_mb=100)
    processor.load_documents()
    search = HybridSearch(embedding_model=loader.embedding_model, doc_processor=processor, use_document_level=True)
    llm = None
    if with_llm:
        from core.local_llm import load_llm_model

        llm = load_llm_model(lightweight=True)
    return ChatManager(llm_model=llm, hybrid_search=search), llm


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="runtime/synthetic")
    parser.add_argument("--with-llm", action="store_true")
    parser.add_argument("--only", default="", help="comma-separated categories")
    parser.add_argument("--report", default="")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    corpus = Path(args.corpus).resolve()
    cases = json.loads((corpus / "cases.json").read_text(encoding="utf-8"))["cases"]
    if args.only:
        wanted = {item.strip() for item in args.only.split(",") if item.strip()}
        cases = [case for case in cases if case["category"] in wanted]
    if args.limit:
        cases = cases[: args.limit]

    from core.chat_mode import ChatMode

    started = time.perf_counter()
    manager, llm = build_manager(corpus / "docs", args.with_llm)
    setup_seconds = time.perf_counter() - started
    manager.set_mode(ChatMode.RAG)

    rows = []
    for case in cases:
        t0 = time.perf_counter()
        try:
            answer = manager.process_message(case["question"])
        except Exception as exc:  # report, do not abort the run
            answer = f"EXCEPTION: {exc!r}"
        elapsed = time.perf_counter() - t0
        ok, details = check_case(case, answer)
        trace = manager.get_last_search_trace() if hasattr(manager, "get_last_search_trace") else {}
        rows.append({
            "category": case["category"],
            "question": case["question"],
            "ok": ok,
            "seconds": round(elapsed, 2),
            "details": details,
            "answer": answer[:4000],
            "trace": {k: trace.get(k) for k in ("question_type", "context_mode", "context_chars", "structured_answer", "direct_answer", "grounded_reasoning", "reasoned_answer", "fact_lookup", "scan_terms", "scan_rows") if k in trace},
        })
        print(f"[{'OK ' if ok else 'BAD'}] {elapsed:6.2f}s {case['category']:<18} {case['question']}", flush=True)

    summary: dict[str, Any] = {}
    for category in sorted({row["category"] for row in rows}):
        items = [row for row in rows if row["category"] == category]
        times = [row["seconds"] for row in items]
        summary[category] = {
            "accuracy": round(sum(row["ok"] for row in items) / len(items), 3),
            "passed": sum(row["ok"] for row in items),
            "total": len(items),
            "median_s": round(statistics.median(times), 2),
            "max_s": round(max(times), 2),
        }
    total_ok = sum(row["ok"] for row in rows)
    payload = {
        "llm": bool(llm),
        "setup_seconds": round(setup_seconds, 1),
        "overall_accuracy": round(total_ok / max(1, len(rows)), 3),
        "summary": summary,
        "rows": rows,
    }
    report = Path(args.report) if args.report else corpus / ("report_llm.json" if args.with_llm else "report.json")
    report.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("llm", "setup_seconds", "overall_accuracy", "summary")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("CHATKODI_DISABLE_TELEMETRY", "1")
    raise SystemExit(main())

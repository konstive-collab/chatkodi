"""CLI-бенчмарк для прогонов типовых запросов и записи результатов в логи."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from core.config import get_config
from core.logging_utils import get_logger, setup_logging
from core.telemetry import TelemetryLogger

logger = get_logger(__name__)


def _load_cases(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding='utf-8'))
    if isinstance(data, dict):
        data = data.get('cases', [])
    if not isinstance(data, list):
        raise ValueError('benchmark file must contain a list of cases or {"cases": [...]}')
    return [dict(item) for item in data]


def _load_llm_if_requested(enabled: bool):
    if not enabled:
        return None
    try:
        from core.local_llm import load_llm_model
        model = load_llm_model(lightweight=True)
        if model and model.is_available():
            return model
    except Exception as exc:
        logger.warning('LLM benchmark setup failed: %s', exc)
    return None


def _build_runtime(load_llm: bool, docs_folder: str):
    from core.chat_mode import ChatManager, ChatMode
    from core.document_processor import DocumentProcessor
    from core.models import AIModelLoader
    from core.search import HybridSearch

    loader = AIModelLoader()
    if not loader.load_models():
        raise RuntimeError('Не удалось загрузить embedding-модель для benchmark')

    processor = DocumentProcessor(docs_folder=docs_folder, recursive=True, max_file_size_mb=100)
    processor.load_documents()
    search = HybridSearch(embedding_model=loader.embedding_model, doc_processor=processor, use_document_level=True)
    llm_model = _load_llm_if_requested(load_llm)
    manager = ChatManager(llm_model=llm_model, hybrid_search=search)
    return manager, search, processor


def _check_case(case: dict[str, Any], response: str) -> tuple[bool, dict[str, Any]]:
    lowered = response.lower()
    must_contain = [str(x) for x in case.get("must_contain", []) if str(x)]
    legacy_text = str(case.get("expected_text_contains", "") or "")
    if legacy_text:
        must_contain.append(legacy_text)
    must_not_contain = [str(x) for x in case.get("must_not_contain", []) if str(x)]
    expected_sources = [str(x) for x in case.get("expected_sources", []) if str(x)]
    legacy_doc = str(case.get("expected_doc_contains", "") or "")
    if legacy_doc:
        expected_sources.append(legacy_doc)
    forbidden_sources = [str(x) for x in case.get("forbidden_sources", []) if str(x)]

    missing = [x for x in must_contain if x.lower() not in lowered]
    forbidden_text = [x for x in must_not_contain if x.lower() in lowered]
    missing_sources = [x for x in expected_sources if x.lower() not in lowered]
    present_forbidden_sources = [x for x in forbidden_sources if x.lower() in lowered]

    expected_count = case.get("expected_event_count")
    actual_count = None
    count_ok = True
    if expected_count is not None:
        match = re.search(r"\bНайдено\s+(\d+)\b", response, flags=re.IGNORECASE)
        if not match and int(expected_count) == 0 and re.search(r"\bне найдено\b", response, flags=re.IGNORECASE):
            actual_count = 0
        elif match:
            actual_count = int(match.group(1))
        count_ok = actual_count == int(expected_count)

    checks = {
        "missing_text": missing,
        "forbidden_text": forbidden_text,
        "missing_sources": missing_sources,
        "forbidden_sources_present": present_forbidden_sources,
        "expected_event_count": expected_count,
        "actual_event_count": actual_count,
        "event_count_passed": count_ok,
    }
    passed = not missing and not forbidden_text and not missing_sources and not present_forbidden_sources and count_ok
    return passed, checks


def run_benchmark(cases_path: Path, docs_folder: str = 'docs', load_llm: bool = False) -> dict[str, Any]:
    cfg = get_config()
    telemetry = TelemetryLogger.from_config_data(cfg.data)
    from core.chat_mode import ChatMode

    manager, _search, _processor = _build_runtime(load_llm=load_llm, docs_folder=docs_folder)
    cases = _load_cases(cases_path)

    results: list[dict[str, Any]] = []
    passed = 0
    for idx, case in enumerate(cases, 1):
        question = str(case.get('question', '')).strip()
        mode_name = str(case.get('mode', 'extract')).strip().lower()
        if mode_name == 'rag':
            manager.set_mode(ChatMode.RAG)
        elif mode_name == 'chat':
            manager.set_mode(ChatMode.CHAT)
        else:
            manager.set_mode(ChatMode.EXTRACT)

        started = time.perf_counter()
        response = manager.process_message(question)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        case_ok, checks = _check_case(case, response)
        if case_ok:
            passed += 1

        results.append({
            'index': idx,
            'mode': mode_name,
            'question': question,
            'elapsed_ms': round(elapsed_ms, 2),
            'checks': checks,
            'passed': case_ok,
            'response_preview': telemetry.preview(response, 700),
            'notes': case.get('notes', ''),
        })

    payload = {
        'summary': {
            'cases_total': len(results),
            'cases_passed': passed,
            'cases_failed': len(results) - passed,
            'docs_folder': docs_folder,
            'load_llm': bool(load_llm),
        },
        'cases_count': len(results),
        'cases': results,
    }
    output_path = telemetry.log_benchmark_run(payload)
    payload["output_path"] = str(output_path) if output_path else ""
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description='Run ChatKodi benchmark cases and save results to logs.')
    parser.add_argument('--cases', default='benchmark_cases.sample.json', help='Path to benchmark cases JSON file')
    parser.add_argument('--docs-folder', default='docs', help='Documents folder to index')
    parser.add_argument('--with-llm', action='store_true', help='Also load local LLM for RAG/chat cases')
    args = parser.parse_args()

    setup_logging()
    payload = run_benchmark(Path(args.cases), docs_folder=args.docs_folder, load_llm=args.with_llm)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload.get("summary", {}).get("cases_failed", 1) == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())

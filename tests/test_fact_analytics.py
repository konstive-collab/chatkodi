"""Totals, extremes, rankings, comparisons and plan/fact on the synthetic corpus.

Expected values come from the corpus generator, not from the answers.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from core.document_processor import DocumentProcessor
from core.fact_lookup import answer_question
from tools.synthetic_corpus import STAFF, build_corpus, write_corpus


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory) -> tuple[DocumentProcessor, object]:
    out = tmp_path_factory.mktemp("synthetic")
    corpus = build_corpus(seed=7, days=62)
    docs = write_corpus(corpus, Path(out), seed=7)
    processor = DocumentProcessor(docs_folder=str(docs), recursive=True, max_file_size_mb=100)
    processor.load_documents()
    return processor, corpus


def _fmt(value: int) -> str:
    return f"{value:,}".replace(",", " ")


def _month(corpus, month: int):
    return [record for record in corpus.days if record.day.month == month]


def test_month_total_excludes_plan_document(synthetic) -> None:
    processor, corpus = synthetic
    total = sum(record.revenue for record in _month(corpus, 8))
    answer = answer_question("Какая общая выручка за август 2025?", processor)
    assert answer is not None and answer.kind == "period_aggregate"
    assert answer.value == _fmt(total)


def test_average_ignores_average_check_lines(synthetic) -> None:
    processor, corpus = synthetic
    rows = _month(corpus, 7)
    average = round(sum(record.revenue for record in rows) / len(rows))
    answer = answer_question("Средняя дневная выручка в июле 2025?", processor)
    assert answer is not None and answer.value == _fmt(average)


def test_day_of_maximum_in_genitive_month(synthetic) -> None:
    processor, corpus = synthetic
    best = max(_month(corpus, 8), key=lambda record: record.revenue)
    answer = answer_question("В какой день августа 2025 была самая большая выручка?", processor)
    assert answer is not None and answer.value == _fmt(best.revenue)
    assert best.day.strftime("%d.%m.%Y") in answer.text.split("\n")[0]


def test_driver_ranking_counts_trips_not_shifts(synthetic) -> None:
    processor, corpus = synthetic
    trips = Counter(STAFF[trip["driver"]][0] for record in _month(corpus, 8) for trip in record.trips)
    top, count = trips.most_common(1)[0]
    answer = answer_question("Какой водитель чаще всех выезжал в августе 2025?", processor)
    assert answer is not None and answer.kind == "ranking"
    assert top in answer.value and answer.value.endswith(str(count))


def test_routes_of_vehicle_over_all_documents(synthetic) -> None:
    processor, corpus = synthetic
    plate = corpus.plates[0]
    routes = {trip["route"] for record in corpus.days for trip in record.trips if trip["plate"] == plate}
    answer = answer_question(f"Какие маршруты были у машины {plate}?", processor)
    assert answer is not None and answer.kind == "entity_values"
    assert answer.value == str(len(routes))


def test_compare_two_days_with_shared_year(synthetic) -> None:
    processor, corpus = synthetic
    first, second = corpus.days[0], corpus.days[-1]
    question = f"Сравни выручку {first.day.day} июля и {second.day.day} августа 2025"
    answer = answer_question(question, processor)
    assert answer is not None and answer.kind == "comparison"
    assert _fmt(first.revenue) in answer.text and _fmt(second.revenue) in answer.text
    assert _fmt(abs(second.revenue - first.revenue)) in answer.text


def test_dash_date_range_is_a_period_not_a_comparison(synthetic) -> None:
    processor, _corpus = synthetic
    answer = answer_question("выручка 01.08.2025-15.08.2025", processor)
    assert answer is None or answer.kind != "comparison"


def test_plan_versus_actual_verdict(synthetic) -> None:
    processor, corpus = synthetic
    record = next(r for r in corpus.days if r.revenue != r.plan)
    verdict = "не выполнен" if record.revenue < record.plan else "план выполнен"
    months = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа"]
    answer = answer_question(f"Выполнен ли план {record.day.day} {months[record.day.month - 1]} 2025?", processor)
    assert answer is not None
    assert verdict in answer.text.split("\n")[0]
    assert _fmt(record.plan) in answer.text and _fmt(record.revenue) in answer.text

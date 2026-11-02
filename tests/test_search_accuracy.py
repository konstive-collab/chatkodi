from datetime import datetime

from core.answer_builders.structured import build_structured_answer
from core.analytics import build_statistics
from core.date_range_parser import parse_date_constraints
from core.fact_extraction import (
    EvidenceRecord,
    build_evidence_records,
    extract_indexed_events,
    filter_records_to_intent_scope,
)
from core.query_understanding import analyze_question
from core.documents.processor import DocumentProcessor
from core.event_store import EventStore
from core.portable_settings import normalize_docs_folder_setting


def test_shared_trailing_year_is_propagated_to_range_start():
    parsed = parse_date_constraints("с 10 июня по 11 июня 2025 года")
    assert parsed["start"] == datetime(2025, 6, 10)
    assert parsed["end"] == datetime(2025, 6, 11)


def test_cross_year_range_uses_previous_year_for_start():
    parsed = parse_date_constraints("с 30 декабря по 2 января 2026 года")
    assert parsed["start"] == datetime(2025, 12, 30)
    assert parsed["end"] == datetime(2026, 1, 2)


def test_english_month_range_and_document_date_are_supported(tmp_path):
    parsed = parse_date_constraints("from 30 December to 2 January 2026")
    assert parsed["start"] == datetime(2025, 12, 30)
    assert parsed["end"] == datetime(2026, 1, 2)

    document = tmp_path / "report.txt"
    document.write_text("Operational report on 14 March 2025\nArbitrary content.", encoding="utf-8")
    processor = DocumentProcessor(str(tmp_path))
    assert processor.load_documents()
    assert processor.documents[0].metadata["document_date"] == "2025-03-14"


def test_english_binary_fact_extracts_object_action_method_and_location():
    intent = analyze_question("Was Service Alpha deployed on 14 March 2025?")
    assert intent.question_type == "binary_fact"
    assert "service alpha" in intent.object_terms
    assert intent.contract.actions == ("deployed",)

    text = (
        "Service Alpha was deployed in North Facility on 14 March 2025 "
        "using Device-Z."
    )
    records = build_evidence_records(
        intent,
        [(0.9, {"doc_id": "alpha", "source": "alpha.txt", "text": text})],
    )
    scoped = filter_records_to_intent_scope(records, intent)

    assert scoped
    assert scoped[0].action == "deployed"
    assert scoped[0].polarity == "positive"
    assert "Device-Z" in scoped[0].methods
    assert "North Facility" in scoped[0].locations


def test_query_answer_contract_keeps_metric_date_and_answer_type() -> None:
    intent = analyze_question("Какая выручка 5 августа 2025 года?")

    assert type(intent).__name__ == "QueryUnderstandingResult"
    assert intent.contract.indicators == ("revenue",)
    assert intent.contract.dates == ("2025-08-05",)
    assert intent.contract.answer_type == "number"


def test_revenue_fact_rejects_profit_and_neighboring_date() -> None:
    intent = analyze_question("Какая выручка 5 августа 2025 года?")
    text = (
        "5 августа 2025 года выручка составила 12 млн рублей. "
        "5 августа 2025 года прибыль составила 3 млн рублей. "
        "6 августа 2025 года выручка составила 99 млн рублей."
    )
    records = build_evidence_records(
        intent,
        [(0.9, {"doc_id": "finance", "source": "finance.txt", "text": text})],
    )
    scoped = filter_records_to_intent_scope(records, intent)

    assert len(scoped) == 1
    assert scoped[0].indicator == "revenue"
    assert "12 млн" in scoped[0].value
    answer = build_structured_answer(intent, scoped, {})
    assert answer is not None
    assert "12 млн" in answer.text
    assert "99 млн" not in answer.text
    assert "прибыль" not in answer.text.lower()


def test_revenue_fact_is_scoped_to_requested_organization() -> None:
    intent = analyze_question("Какая выручка ООО Альфа 5 августа 2025 года?")
    candidates = [
        (0.9, {"doc_id": "alpha", "source": "alpha.txt", "text": "5 августа 2025 года выручка ООО Альфа составила 12 млн рублей."}),
        (0.95, {"doc_id": "beta", "source": "beta.txt", "text": "5 августа 2025 года выручка ООО Бета составила 99 млн рублей."}),
    ]

    scoped = filter_records_to_intent_scope(build_evidence_records(intent, candidates), intent)
    answer = build_structured_answer(intent, scoped, {})

    assert intent.contract.objects == ("ооо альфа",)
    assert len(scoped) == 1
    assert "ООО Альфа" in scoped[0].sentence
    assert answer is not None
    assert "12 млн" in answer.text
    assert "99 млн" not in answer.text


def test_specialist_work_days_are_aggregated_across_documents_with_conflicts() -> None:
    intent = analyze_question("Сколько дней работал специалист Иванов Иван?")
    candidates = [
        (0.9, {"doc_id": "d1", "source": "one.txt", "text": "1 августа 2025 года специалист Иванов Иван работал в смене."}),
        (0.8, {"doc_id": "d2", "source": "duplicate.txt", "text": "1 августа 2025 года специалист Иванов Иван работал в смене."}),
        (0.9, {"doc_id": "d3", "source": "two.txt", "text": "2 августа 2025 года специалист Иванов Иван работал в смене."}),
        (0.9, {"doc_id": "d4", "source": "namesake.txt", "text": "4 августа 2025 года специалист Иванов Петр работал в смене."}),
        (0.9, {"doc_id": "d5", "source": "positive.txt", "text": "3 августа 2025 года специалист Иванов Иван работал в смене."}),
        (0.9, {"doc_id": "d6", "source": "negative.txt", "text": "3 августа 2025 года специалист Иванов Иван не работал."}),
    ]
    scoped = filter_records_to_intent_scope(build_evidence_records(intent, candidates), intent)
    stats = build_statistics(scoped, count_unit=intent.count_unit, intent=intent)
    answer = build_structured_answer(intent, scoped, stats)

    assert intent.count_unit == "date"
    assert stats["total_records"] == 3
    assert stats["conflicting_dates"] == ["2025-08-03"]
    assert answer is not None
    assert "3" in answer.text
    assert "противореч" in answer.text.lower()


def test_event_store_supplies_multi_document_work_facts(tmp_path) -> None:
    store = EventStore(str(tmp_path / "events.db"), str(tmp_path / "events_meta.json"))
    store.sync_chunks([
        {"doc_id": "d1", "chunk_id": 1, "source": "one.txt", "text": "1 августа 2025 года специалист Иванов Иван работал."},
        {"doc_id": "d2", "chunk_id": 2, "source": "two.txt", "text": "2 августа 2025 года специалист Иванов Иван работал."},
        {"doc_id": "d3", "chunk_id": 3, "source": "other.txt", "text": "3 августа 2025 года специалист Иванов Петр работал."},
    ])
    intent = analyze_question("Сколько дней работал специалист Иванов Иван?")
    records = filter_records_to_intent_scope(store.query_as_evidence(intent), intent)
    stats = build_statistics(records, count_unit=intent.count_unit, intent=intent)

    assert stats["total_records"] == 2
    assert stats["date_span"]["active_days"] == 2


def test_observed_section_extracts_typed_timed_targets():
    text = (
        "Силами и средствами установлено: 10.06.2025 г.:\n"
        "рБпЛА №11 (Изюмское направление):\n"
        "02.45 – две ед. АТ типа «внедорожник», ПЕТРОВСКОЕ;\n"
        "08.35 – позиция отделения, ЗЕЛЕНЫЙ ЯР;\n"
        "рБпЛА №22 (Запорожское направление):\n"
        "09.05 – блокпост, СВЕТЛОЕ."
    )
    events = extract_indexed_events(
        "doc",
        "10.docx",
        1,
        text,
        {"document_date": "2025-06-10"},
    )
    observed_11 = [
        event
        for event in events
        if event.metadata.get("section_number") == "11" and event.target
    ]
    assert [event.event_time for event in observed_11] == ["02:45", "08:35"]
    assert all(event.event_type == "any" for event in observed_11)
    assert all(event.event_dates == ["2025-06-10"] for event in observed_11)
    assert not any(
        event.metadata.get("section_number") == "11" and "СВЕТЛОЕ" in event.target
        for event in events
    )


def test_single_date_and_event_type_are_hard_scope():
    intent = analyze_question("Какие цели были обнаружены рБпЛА №11 на 10 июня 2025 года?")
    good = EvidenceRecord(
        doc_id="10",
        source="10.docx",
        chunk_id=1,
        sentence="02.45 – блокпост, ПЕТРОВСКОЕ",
        full_text="",
        score=1.0,
        event_dates=["2025-06-10"],
        target="блокпост, ПЕТРОВСКОЕ",
        event_type="any",
        event_time="02:45",
        metadata={"section_family": "рбпла", "section_number": "11"},
    )
    wrong_date = EvidenceRecord(
        **{
            **good.__dict__,
            "doc_id": "36",
            "source": "36.docx",
            "event_dates": ["2025-07-05"],
        }
    )
    scoped = filter_records_to_intent_scope([wrong_date, good], intent)
    assert scoped == [good]

    answer = build_structured_answer(intent, scoped, build_statistics(scoped, intent=intent))
    assert answer is not None
    assert "Найдено 1" in answer.text
    assert "ПЕТРОВСКОЕ" in answer.text
    assert answer.confidence >= 0.9


def test_date_range_does_not_turn_fact_question_into_summary():
    intent = analyze_question(
        "Есть ли информация о состоянии системы с 10 июня по 15 июня 2025 года?"
    )
    assert intent.question_type == "binary_fact"
    assert intent.date_mode == "range"
    assert not intent.wants_statistics
    assert not intent.wants_range_aggregation


def test_plain_fact_is_not_sent_to_statistics_summary():
    intent = analyze_question("На каких позициях находится выбранная система?")
    assert intent.question_type == "fact"
    assert build_structured_answer(intent, [], {}) is None


def test_corpus_ignore_is_data_driven_and_does_not_ban_filenames(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("User-owned knowledge base", encoding="utf-8")

    processor = DocumentProcessor(str(tmp_path))
    assert readme in processor._collect_all_files()

    (tmp_path / ".chatkodiignore").write_text("*.md\n", encoding="utf-8")
    processor = DocumentProcessor(str(tmp_path))
    assert readme not in processor._collect_all_files()


def test_switching_corpus_reloads_data_driven_ignore(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "initial.txt").write_text("Initial corpus document", encoding="utf-8")
    ignored = second / "internal.md"
    ignored.write_text("Internal corpus note", encoding="utf-8")
    visible = second / "report.txt"
    visible.write_text("Visible corpus report", encoding="utf-8")
    (second / ".chatkodiignore").write_text("*.md\n", encoding="utf-8")

    processor = DocumentProcessor(str(first))
    assert processor.update_index(new_docs_folder=str(second))
    assert visible in processor._collect_all_files()
    assert ignored not in processor._collect_all_files()


def test_statistics_query_uses_corpus_reference_year():
    intent = analyze_question(
        'Статистика применения "Система-2" с 11 по 13 июня',
        reference_date=datetime(2025, 1, 1),
    )
    assert intent.question_type == "count"
    assert intent.wants_statistics
    assert intent.start_date == datetime(2025, 6, 11)
    assert intent.end_date == datetime(2025, 6, 13)


def test_portable_settings_drop_absolute_docs_path_from_other_app_root(tmp_path, monkeypatch):
    current_root = tmp_path / "current" / "ChatKodi"
    old_root = tmp_path / "old" / "ChatKodi"
    external = tmp_path / "external_docs"
    current_root.mkdir(parents=True)
    old_docs = old_root / "docs"
    old_docs.mkdir(parents=True)
    external.mkdir()

    monkeypatch.setenv("CHATKODI_PRODUCTION", "1")
    monkeypatch.setenv("CHATKODI_PORTABLE", "1")
    monkeypatch.setenv("CHATKODI_APP_ROOT", str(current_root))
    monkeypatch.setenv("CHATKODI_RUNTIME_DIR", str(current_root / "runtime"))

    assert (
        normalize_docs_folder_setting(str(old_docs), stored_app_root=str(old_root), default="docs")
        == "docs"
    )
    assert (
        normalize_docs_folder_setting(str(external), stored_app_root=str(current_root), default="docs")
        == str(external)
    )

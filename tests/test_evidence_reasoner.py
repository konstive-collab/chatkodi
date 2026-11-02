import json

from core.evidence_reasoner import (
    GroundingEvidence,
    collect_grounding_evidence,
    format_grounded_answer,
    reason_over_evidence,
    validate_grounded_payload,
)


def _evidence():
    return [
        GroundingEvidence(
            evidence_id="E1",
            source="policy.txt",
            text="Service Alpha entered maintenance mode on 14 March 2025 at 09:30.",
            line_start=4,
            line_end=4,
        )
    ]


def test_grounded_payload_accepts_exact_quote():
    payload = {
        "status": "supported",
        "answer": "Service Alpha entered maintenance mode on 14 March 2025.",
        "claims": [{
            "text": "Service Alpha entered maintenance mode on 14 March 2025.",
            "evidence_ids": ["E1"],
            "quotes": ["Service Alpha entered maintenance mode on 14 March 2025 at 09:30."],
        }],
        "confidence": 0.96,
    }
    result = validate_grounded_payload(payload, _evidence())
    assert result.valid
    rendered = format_grounded_answer(result, _evidence())
    assert "policy.txt" in rendered
    assert "09:30" in rendered


def test_grounded_payload_rejects_invented_quote():
    payload = {
        "status": "supported",
        "answer": "The service was permanently deleted.",
        "claims": [{
            "text": "The service was permanently deleted.",
            "evidence_ids": ["E1"],
            "quotes": ["The service was permanently deleted."],
        }],
        "confidence": 0.99,
    }
    result = validate_grounded_payload(payload, _evidence())
    assert not result.valid
    assert result.validation_error == "quote_not_found"


def test_grounded_payload_rejects_unsupported_answer_with_real_quote():
    evidence = [
        GroundingEvidence(
            evidence_id="E1",
            source="policy.txt",
            text="The backup starts on Thursday at 21:15.",
        )
    ]
    result = validate_grounded_payload(
        {
            "status": "supported",
            "answer": "The backup starts on Friday at 09:00.",
            "claims": [{
                "text": "The backup starts on Thursday at 21:15.",
                "evidence_ids": ["E1"],
                "quotes": ["The backup starts on Thursday at 21:15."],
            }],
            "confidence": 0.9,
        },
        evidence,
    )
    assert not result.valid
    assert result.validation_error == "answer_number_not_in_claims"


def test_grounded_payload_rejects_evidence_for_wrong_metric_or_date():
    payload = {
        "status": "supported",
        "answer": "3 млн рублей.",
        "claims": [{
            "text": "Прибыль составила 3 млн рублей.",
            "evidence_ids": ["E1"],
            "quotes": ["5 августа 2025 года прибыль составила 3 млн рублей."],
        }],
        "confidence": 0.9,
    }
    wrong_metric = [GroundingEvidence(
        evidence_id="E1",
        source="finance.txt",
        text="5 августа 2025 года прибыль составила 3 млн рублей.",
    )]
    result = validate_grounded_payload(
        payload,
        wrong_metric,
        question="Какая выручка 5 августа 2025 года?",
    )
    assert not result.valid
    assert result.validation_error == "question_contract_mismatch"

    wrong_date = [GroundingEvidence(
        evidence_id="E1",
        source="finance.txt",
        text="6 августа 2025 года выручка составила 3 млн рублей.",
    )]
    payload["claims"][0]["text"] = "Выручка составила 3 млн рублей."
    payload["claims"][0]["quotes"] = [wrong_date[0].text]
    result = validate_grounded_payload(
        payload,
        wrong_date,
        question="Какая выручка 5 августа 2025 года?",
    )
    assert not result.valid
    assert result.validation_error == "question_contract_mismatch"


def test_grounded_payload_rejects_evidence_for_wrong_object():
    evidence = [GroundingEvidence(
        evidence_id="E1",
        source="beta.txt",
        text="5 августа 2025 года выручка ООО Бета составила 99 млн рублей.",
    )]
    result = validate_grounded_payload(
        {
            "status": "supported",
            "answer": "99 млн рублей.",
            "claims": [{
                "text": "Выручка ООО Бета составила 99 млн рублей.",
                "evidence_ids": ["E1"],
                "quotes": [evidence[0].text],
            }],
            "confidence": 0.9,
        },
        evidence,
        question="Какая выручка ООО Альфа 5 августа 2025 года?",
    )

    assert not result.valid
    assert result.validation_error == "question_contract_mismatch"

def test_reasoner_parses_json_from_fake_llm():
    class FakeLlm:
        def generate_with_callback(self, *, callback, **_kwargs):
            payload = {
                "status": "supported",
                "answer": "Maintenance mode began at 09:30.",
                "claims": [{
                    "text": "Maintenance mode began at 09:30.",
                    "evidence_ids": ["E1"],
                    "quotes": ["entered maintenance mode on 14 March 2025 at 09:30"],
                }],
                "confidence": 0.9,
            }
            callback("prefix\n" + json.dumps(payload), False)
            callback("", True)
            return ""

    result = reason_over_evidence(FakeLlm(), "When did maintenance begin?", _evidence())
    assert result.valid
    assert result.status == "supported"


def test_collect_grounding_evidence_prefers_source_text_and_deduplicates():
    result = {
        "results_all": [
            (0.9, {"source": "a.txt", "text": "Heading\nactual fact", "source_text": "actual fact", "line_start": 2}),
            (0.8, {"source": "a.txt", "text": "actual fact", "source_text": "actual fact", "line_start": 2}),
        ]
    }
    evidence = collect_grounding_evidence(result)
    assert len(evidence) == 1
    assert evidence[0].text == "actual fact"


def test_soft_answer_check_requires_numbers_and_names_from_context() -> None:
    from core.evidence_reasoner import unsupported_answer_details

    context = "Фактическая выручка за день составила 1 875 250 руб. Отчет составил: руководитель филиала Белов С.Н."
    assert unsupported_answer_details("Выручка — 1 875 250 руб., отчёт составил Белов С.Н.", context) == []
    missing = unsupported_answer_details("Выручка — 1 900 000 руб., отчёт составил Петров И.И.", context)
    assert "1 900 000" in missing and "Петров И.И." in missing


def test_not_found_with_empty_answer_is_valid() -> None:
    from core.evidence_reasoner import validate_grounded_payload

    result = validate_grounded_payload({"status": "not_found", "answer": "", "claims": []}, [])
    assert result.valid and result.answer

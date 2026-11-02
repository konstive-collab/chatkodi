from core.generation.service import GenerationService
from core.generation.types import GenerationKind


def test_training_request_asks_for_missing_fields_before_llm_or_catalog(tmp_path):
    service = GenerationService(None, {"catalog_root": str(tmp_path)})
    result = service.handle("Составь план тренировок на месяц для развития выносливости")
    assert result is not None
    assert result.generation_kind == GenerationKind.TRAINING_PLAN
    assert "возрастная группа" in result.text
    assert service.pending is not None


def test_partial_training_clarification_does_not_skip_remaining_fields(tmp_path):
    service = GenerationService(None, {"catalog_root": str(tmp_path)})
    service.handle("Составь план тренировок на месяц для развития выносливости")
    result = service.handle("Взрослый, средний уровень, 3 раза в неделю")
    assert result is not None
    assert "длительность одного занятия" in result.text
    assert "ограничения по здоровью" in result.text
    assert service.pending is not None


def test_ordinary_question_is_not_intercepted(tmp_path):
    service = GenerationService(None, {"catalog_root": str(tmp_path)})
    assert service.handle("Что сказано в документе о тренировках?") is None

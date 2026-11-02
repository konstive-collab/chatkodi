from core.generation.router import detect_generation_kind
from core.generation.types import GenerationKind


def test_detects_training_plan_without_breaking_fact_questions():
    assert detect_generation_kind("Составь план тренировок на месяц для развития выносливости") == GenerationKind.TRAINING_PLAN
    assert detect_generation_kind("Какие нормативы по бегу указаны в документе?") == GenerationKind.NONE
    assert detect_generation_kind("Где в документе находится план тренировок?") == GenerationKind.NONE


def test_detects_exercise_ideation_and_legal_creation():
    assert detect_generation_kind("Предложи упражнения с плаванием") == GenerationKind.EXERCISE_IDEATION
    assert detect_generation_kind("Составь рапорт на командировку") == GenerationKind.LEGAL_DOCUMENT
    assert detect_generation_kind("Найди рапорт на командировку") == GenerationKind.NONE

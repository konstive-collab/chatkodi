from core.conversation import normalize_query, resolve_followup, small_talk_answer


def test_greeting_and_capabilities_answer_instantly() -> None:
    answer = small_talk_answer("Привет! Что ты умеешь?", document_count=55)
    assert answer is not None
    assert "55 документов" in answer
    assert small_talk_answer("кто ты") is not None
    assert small_talk_answer("Спасибо!") is not None


def test_greeting_with_real_question_goes_to_search() -> None:
    assert small_talk_answer("Привет, какая выручка была 4 августа 2025?") is None
    assert small_talk_answer("Какая выручка 4 августа 2025?") is None


def test_capability_words_inside_real_questions_go_to_search() -> None:
    assert small_talk_answer("нужна помощь по отчёту за август") is None
    assert small_talk_answer("что ты знаешь о выручке за август 2025?") is None
    assert small_talk_answer("help me find revenue") is None
    for canned in ("помощь", "кто ты?", "что ты умеешь?", "Привет! Что ты умеешь?", "Что ты можешь делать?"):
        assert small_talk_answer(canned) is not None, canned


def test_colloquial_words_and_misspelled_months_are_fixed() -> None:
    assert normalize_query("скока выручки было 4 авгста 2025") == "сколько выручки было 4 августа 2025"
    assert normalize_query("Выручка 5 сентебря") == "Выручка 5 сентября"
    # ordinary words close to nothing stay untouched
    assert normalize_query("Какой водитель ездил на маршрут") == "Какой водитель ездил на маршрут"
    # real words one letter away from a month are not dates
    assert normalize_query("Какая марка у автомобиля А123ВС77?") == "Какая марка у автомобиля А123ВС77?"
    assert normalize_query("Марка и маяк") == "Марка и маяк"


def test_day_only_followup_keeps_month_and_year() -> None:
    result = resolve_followup("а 15-го?", "Какая выручка 4 августа 2025 года?")
    assert result.resolved
    assert result.question == "Какая выручка 15 августа 2025 года?"
    assert "Понял вопрос как" in result.note


def test_month_and_person_followups() -> None:
    month = resolve_followup("а в августе?", "Сколько дней работал Попов А.С. в июле 2025?")
    assert month.question == "Сколько дней работал Попов А.С. в августе 2025?"
    person = resolve_followup("а Лебедев П.И.?", "Сколько дней работал Попов А.С. в июле 2025?")
    assert person.question == "Сколько дней работал Лебедев П.И. в июле 2025?"


def test_full_date_followup_and_bare_date() -> None:
    full = resolve_followup("а 1 июля?", "Какая выручка 4 августа 2025 года?")
    assert full.question.startswith("Какая выручка 1 июля 2025")
    bare = resolve_followup("15.08.2025", "Какая выручка 4 августа 2025 года?")
    assert bare.resolved and "15.08.2025" in bare.question


def test_independent_questions_are_not_merged() -> None:
    assert not resolve_followup("Какая выручка 4 августа 2025?", "Сколько дней работал Попов А.С.?").resolved
    assert not resolve_followup("выручка 04.08.25", "Кто ездил на М653ОЕ77 2 августа 2025?").resolved
    assert not resolve_followup("а 15-го?", None).resolved

from core.corpus_scan import deterministic_answer, question_terms, scan

DOCS = [
    {"name": "d1.txt", "content": "ОТЧЁТ НА 1 МАРТА 2025\nПроведено две проверки склада.\nНасос Н-3 заменён.\nКлиент «Север» оплатил счёт."},
    {"name": "d2.txt", "content": "ОТЧЁТ НА 2 МАРТА 2025\nПроведена одна проверка склада.\nКлиент «Север» прислал претензию."},
    {"name": "d3.txt", "content": "ОТЧЁТ НА 3 МАРТА 2025\nПроведено три проверки склада.\nНасос Н-3 заменён повторно."},
    {"name": "d4.txt", "content": "ОТЧЁТ НА 4 АПРЕЛЯ 2025\nПроведено пять проверок склада.\nКлиент «Север» оплатил счёт."},
]


def test_terms_come_from_the_question() -> None:
    terms = question_terms("Сколько раз клиент «Север» оплатил счёт за весь период?")
    assert "север" in terms and "сколько" not in terms and "период" not in terms


def test_count_of_mentions_is_the_number_of_lines() -> None:
    result = scan(DOCS, "Сколько раз за весь период менялся насос Н-3?")
    answer = deterministic_answer("Сколько раз за весь период менялся насос Н-3?", result)
    assert answer.startswith("**2**")


def test_quantities_written_in_lines_are_summed() -> None:
    q = "Сколько всего проверок склада было с 1 по 3 марта?"
    result = scan(DOCS, q, (1, 3), (3, 3))
    assert len(result.rows) == 3 and result.documents == 3
    assert deterministic_answer(q, result).startswith("**6**")  # 2 + 1 + 3, April excluded


def test_series_and_unknown_names() -> None:
    from core.corpus_scan import is_series

    many = DOCS + [{"name": f"e{i}.txt", "content": f"ОТЧЁТ НА {i} МАЯ 2025\nстрока"} for i in range(1, 4)]
    assert is_series(many)
    assert not is_series([{"content": "Договор поставки"}, {"content": "Приказ"}, {"content": "Протокол"}] * 2)
    q = "Сколько раз клиент «Юг» оплатил счёт?"
    assert scan(DOCS, q).rows == []  # «Юг» is in no document: no answer about «Север»


def test_sum_question_is_left_to_the_llm() -> None:
    q = "Сколько всего выручки за весь период по клиенту «Север»?"
    assert deterministic_answer(q, scan(DOCS, q)) is None


def test_days_are_listed_by_the_program() -> None:
    q = "В какие дни клиент «Север» оплатил счёт?"
    answer = deterministic_answer(q, scan(DOCS, q))
    assert "1 марта" in answer and "4 апреля" in answer and "2 марта" not in answer.split("Найдено")[0]

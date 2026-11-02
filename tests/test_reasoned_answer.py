from core.reasoned_answer import safe_eval, verify

CONTRACT = (
    "Общая стоимость товара составляет 3 480 000 рублей. За просрочку поставки Поставщик уплачивает "
    "неустойку 0,1 % от стоимости непоставленного товара за каждый день просрочки, но не более 10 % от цены договора."
)


def test_safe_eval_allows_only_arithmetic() -> None:
    assert safe_eval("3480000 * 0.001 * 20") == 69600
    assert safe_eval("3 480 000 * 0,1") == 348000
    assert safe_eval("(18.2 + 21.7)") == 39.9
    assert safe_eval("__import__('os').system('x')") is None
    assert safe_eval("2 ** 10") is None
    assert safe_eval("1/0") is None


def test_calculation_is_done_by_the_program() -> None:
    payload = {
        "found": True,
        "answer": "Неустойка 0,1 % за каждый из 20 дней от 3 480 000 рублей.",
        "quotes": ["неустойку 0,1 % от стоимости непоставленного товара за каждый день просрочки"],
        "calculation": "3480000 * 0.001 * 20",
    }
    result = verify(payload, CONTRACT, "Какую неустойку заплатят за 20 дней просрочки?")
    assert result.found and result.value == 69600
    assert "69 600" in result.text and "3480000 * 0.001 * 20" in result.text


def test_decimal_in_calculation_is_not_a_date() -> None:
    payload = {
        "found": True,
        "answer": "Не более 10 % от цены договора.",
        "quotes": ["но не более 10 % от цены договора"],
        "calculation": "3480000 * 0.10",
    }
    result = verify(payload, CONTRACT, "Какой максимальный размер неустойки в рублях?")
    assert result.found and result.value == 348000


def test_dates_and_millions_in_documents_are_known_numbers() -> None:
    context = "Отпуск с 16.06.2025 по 29.06.2025. Выручка I квартал: 18,2 млн руб., II квартал: 21,7 млн руб."
    dates = {"found": True, "answer": "Отпуск с 16 по 29 июня.", "quotes": ["Отпуск с 16.06.2025 по 29.06.2025"], "calculation": ""}
    assert verify(dates, context, "Когда отпуск?").found
    total = {"found": True, "answer": "Всего за полугодие.", "quotes": ["18,2 млн руб."], "calculation": "18200000 + 21700000"}
    result = verify(total, context, "Сколько всего выручки?")
    assert result.found and result.value == 39900000


def test_wrong_number_in_model_text_is_replaced_by_the_result() -> None:
    payload = {
        "found": True,
        "answer": "Поставщик заплатит неустойку в размере 348 рублей.",
        "quotes": ["неустойку 0,1 % от стоимости непоставленного товара за каждый день просрочки"],
        "calculation": "3480000 * 0.001 * 20",
    }
    result = verify(payload, CONTRACT, "Какую неустойку заплатят за 20 дней просрочки?")
    assert result.found and "348 рублей" not in result.text
    assert "69 600" in result.text and "Основание" in result.text


def test_counting_is_done_by_the_program() -> None:
    context = (
        "рБпЛА №11:\n02.45 – две ед. АТ, ПЕТРОВСКОЕ;\n08.35 – позиция отделения, ЗЕЛЕНЫЙ ЯР;\n"
        "рБпЛА №22:\n07.40 – ПВД, РОВНОЕ;\n"
    )
    payload = {
        "found": True,
        "answer": "Обнаружено 30 целей.",
        "quotes": ["07.40 – ПВД, РОВНОЕ"],
        "counted_items": ["02.45 – две ед. АТ, ПЕТРОВСКОЕ", "08.35 – позиция отделения, ЗЕЛЕНЫЙ ЯР",
                          "07.40 – ПВД, РОВНОЕ", "07.40 – ПВД, РОВНОЕ", "11.00 – выдуманная цель"],
        "calculation": "",
    }
    result = verify(payload, context, "Сколько целей обнаружено?")
    assert result.found and result.value == 3  # duplicate and invented items are not counted
    assert "Итог: **3**" in result.text and "выдуманная" not in result.text


def test_number_words_in_documents_are_known() -> None:
    context = "уничтожено пять, ранено три чел.; уничтожено три, ранен один чел.; уничтожено четыре, ранено три чел."
    payload = {"found": True, "answer": "Уничтожено 12 человек.", "quotes": ["уничтожено пять, ранено три чел."], "calculation": "5 + 3 + 4"}
    result = verify(payload, context, "Сколько уничтожено?")
    assert result.found and result.value == 12


def test_invented_operand_is_rejected() -> None:
    payload = {
        "found": True,
        "answer": "Неустойка составит 104 400 рублей.",
        "quotes": ["неустойку 0,1 % от стоимости непоставленного товара"],
        "calculation": "3480000 * 0.001 * 30",
    }
    result = verify(payload, CONTRACT, "Какую неустойку заплатят за 20 дней просрочки?")
    assert not result.found
    assert result.rejected.startswith("calc_number_not_in_documents")


def test_quote_must_be_verbatim_and_numbers_known() -> None:
    fake_quote = {"found": True, "answer": "Срок 30 дней.", "quotes": ["срок поставки два месяца"], "calculation": ""}
    assert verify(fake_quote, CONTRACT, "Какой срок?").rejected == "no_exact_quote"
    stray = {
        "found": True,
        "answer": "Стоимость 5 200 000 рублей.",
        "quotes": ["Общая стоимость товара составляет 3 480 000 рублей"],
        "calculation": "",
    }
    assert verify(stray, CONTRACT, "Сколько стоит?").rejected.startswith("answer_number_not_in_documents")


def test_answer_for_another_date_is_rejected() -> None:
    context = "ЕЖЕДНЕВНЫЙ ОТЧЕТ за 9 августа 2025 года. Фактическая выручка за день составила 1 277 750 руб."
    payload = {
        "found": True,
        "answer": "Выручка составила 1 277 750 руб.",
        "quotes": ["Фактическая выручка за день составила 1 277 750 руб."],
        "calculation": "",
    }
    result = verify(payload, context, "Какая выручка была 9 сентября 2025 года?")
    assert not result.found and result.rejected == "date_not_in_documents"
    assert verify(payload, context, "Какая выручка была 09.08.2025?").found


def test_not_found_is_passed_through() -> None:
    result = verify({"found": False, "answer": "", "quotes": [], "calculation": ""}, CONTRACT, "Кто директор?")
    assert not result.found and "нет информации" in result.text


RULES = (
    "Итого: 34 000,00 руб.\n"
    "Счёт на сумму не более 30 000 руб. оплачивается без согласования с финансовым директором. "
    "Счета на большую сумму оплачиваются только после согласования с финансовым директором."
)


def test_limit_with_thousands_is_one_number() -> None:
    from core.requirements import requirement_holds

    assert requirement_holds("25 000,00 руб.", "на сумму не более 30 000 руб.") is True
    assert requirement_holds("34 000,00 руб.", "на сумму не более 30 000 руб.") is False
    assert requirement_holds("34 000 руб.", "свыше 30 000 руб.") is True


def test_comparison_is_done_by_the_program_and_the_model_answers_again() -> None:
    import json

    from core.reasoned_answer import compared, reasoned_answer

    first = {"found": True, "answer": "Да, можно.", "quotes": ["Счёт на сумму не более 30 000 руб. оплачивается без согласования"],
             "counted_items": [], "calculation": "",
             "comparisons": [{"value": "34 000,00 руб.", "requirement": "на сумму не более 30 000 руб."},
                             {"value": "50 000 руб.", "requirement": "не более 30 000 руб."}]}  # 50 000 is not in the documents
    assert compared(first, RULES, "Можно оплатить?") == ["34 000,00 руб. — условие «на сумму не более 30 000 руб.» не выполнено"]
    second = dict(first, answer="Нет, нужно согласование: сумма больше 30 000 руб.", comparisons=[])
    prompts = []

    class _LLM:
        supports_json_schema = True

        def generate_with_callback(self, messages, callback, **kw):
            prompts.append(messages[0]["content"])
            callback(json.dumps(first if len(prompts) == 1 else second, ensure_ascii=False), True)

    result = reasoned_answer(_LLM(), "Можно оплатить без согласования?", RULES)
    assert len(prompts) == 2 and "ПРОВЕРЕНО ПРОГРАММОЙ" in prompts[1] and "не выполнено" in prompts[1]
    assert result.text.startswith("Нет") and "Проверено программой: 34 000,00 руб." in result.text


def test_a_count_given_as_a_formula_is_asked_again_for_the_items() -> None:
    import json

    from core.reasoned_answer import reasoned_answer

    log = "рБпЛА №11:\n02.45 – АТ, ПЕТРОВСКОЕ;\n08.35 – позиция, ЗЕЛЕНЫЙ ЯР;\nрБпЛА №22:\n07.40 – ПВД, РОВНОЕ;"
    first = {"found": True, "answer": "Обнаружено 6 целей.", "quotes": ["02.45 – АТ, ПЕТРОВСКОЕ"], "counted_items": [],
             "calculation": "3 + 3"}
    second = dict(first, answer="Всего целей:", calculation="",
                  counted_items=["02.45 – АТ, ПЕТРОВСКОЕ", "08.35 – позиция, ЗЕЛЕНЫЙ ЯР", "07.40 – ПВД, РОВНОЕ"])
    prompts = []

    class _LLM:
        supports_json_schema = True

        def generate_with_callback(self, messages, callback, **kw):
            prompts.append(messages[0]["content"])
            callback(json.dumps(first if len(prompts) == 1 else second, ensure_ascii=False), True)

    result = reasoned_answer(_LLM(), "Сколько целей обнаружили все рБпЛА?", log)
    assert len(prompts) == 2 and "СКОЛЬКО было объектов" in prompts[1]
    assert result.value == 3 and "Посчитано по документам (3)" in result.text


def test_items_named_alike_count_as_often_as_the_document_has_them() -> None:
    log = "08.35 – блокпост, ЗЕЛЕНЫЙ ЯР;\n11.25 – блокпост, БЕРЕСТОВАЯ;\n12.50 – блиндаж, ДОНЦОВКА."
    payload = {"found": True, "answer": "Целей:", "quotes": ["блокпост, ЗЕЛЕНЫЙ ЯР"], "calculation": "",
               "counted_items": ["блокпост", "блокпост", "блокпост", "блиндаж"]}
    result = verify(payload, log, "Сколько целей обнаружено?")
    assert result.value == 3

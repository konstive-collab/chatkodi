"""Presentations: Markdown slides to PPTX, the look of an example deck, routing of «сделай презентацию»."""

from __future__ import annotations

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches

from core.agent.slides import markdown_to_pptx, parse_slides, pptx_outline

DECK = """# Итоги работы склада № 2 за октябрь
ООО «Вектор», 28.10.2026

## Главное
- Отгружено **1 240** заказов
- Возвраты: 1,8 %
  - из них по браку 0,4 %

## Показатели
| Показатель | План | Факт |
|---|---|---|
| Заказы | 1 200 | 1 240 |

## Предложения
""" + "\n".join(f"- пункт {i}" for i in range(1, 11))


def _example(path, colour="5B2333"):
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    fill = prs.slide_master.background.fill
    fill.solid()
    fill.fore_color.rgb = RGBColor.from_string(colour)
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    slide.shapes.title.text = "Образец"
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "Цифры месяца"
    slide.placeholders[1].text_frame.text = "Показатель 1"
    prs.save(str(path))
    return path


def test_markdown_becomes_slides_with_lists_tables_and_continuations(tmp_path):
    title, subtitle, slides = parse_slides(DECK)
    assert title.startswith("Итоги работы") and subtitle == ["ООО «Вектор», 28.10.2026"]
    assert [s.title for s in slides] == ["Главное", "Показатели", "Предложения"]
    assert slides[0].bullets[2] == (1, "из них по браку 0,4 %") and slides[1].table[1] == ["Заказы", "1 200", "1 240"]
    prs = Presentation(str(markdown_to_pptx(DECK, tmp_path / "a.pptx")))
    assert [s.shapes.title.text for s in prs.slides][-2:] == ["Предложения", "Предложения (продолжение)"]
    assert prs.slide_width == Inches(13.333)
    outline = pptx_outline(tmp_path / "a.pptx")
    assert "| Заказы | 1 200 | 1 240 |" in outline and "- Отгружено 1 240 заказов" in outline


def test_example_deck_gives_the_look_and_loses_its_slides(tmp_path):
    example = _example(tmp_path / "example.pptx")
    prs = Presentation(str(markdown_to_pptx(DECK, tmp_path / "b.pptx", base_pptx=example)))
    assert str(prs.slide_master.background.fill.fore_color.rgb) == "5B2333"
    titles = [s.shapes.title.text for s in prs.slides]
    assert "Образец" not in titles and "Цифры месяца" not in titles and titles[0].startswith("Итоги работы")


def test_making_slides_is_routed_to_a_presentation_and_a_question_is_not(tmp_path):
    from core.agent.agent import Agent, _demands
    from core.agent.attachments import AttachmentStore

    assert "pptx" in _demands("Сделай презентацию по этому отчёту")
    assert "pptx" in _demands("Оформи отчёт в презентацию как в приложенном образце")
    assert "pptx" in _demands("Сделай из этого презентацию на несколько слайдов")
    assert "pptx" not in _demands("Сколько слайдов в этой презентации и о чём они?")

    class _Manager:
        llm_model = None
        hybrid_search = None

    agent = Agent(_Manager(), store=AttachmentStore(tmp_path))
    route = {"action": "summarize", "source": "attachments", "output": "chat", "language": "", "task": "т", "search_query": ""}
    routed = agent._reroute_by_request(dict(route), "Сделай презентацию по этому отчёту")
    assert routed["action"] == "make_document" and routed["output"] == "pptx"


def test_lines_copied_from_the_example_are_dropped_unless_the_data_has_them():
    from core.agent.slides import drop_example_lines

    example = "# Отчёт подразделения\nОтдел, дата\n\n## Цифры месяца\n- Показатель 1\n- Показатель 2"
    body = "# Отчёт склада\nОтдел, дата\n\n## Цифры месяца\n- Показатель 1\n- Заказы: 1 240"
    kept = drop_example_lines(body, example, data="Отгружено 1 240 заказов")
    assert "Показатель 1" not in kept and "Отдел, дата" not in kept
    assert "## Цифры месяца" in kept and "- Заказы: 1 240" in kept


def test_the_example_is_shown_to_the_model_as_form_only(tmp_path):
    from core.agent.slides import pptx_form

    form = pptx_form(_example(tmp_path / "e.pptx"))
    assert "## Цифры месяца" in form and "(список: 1 пункт)" in form
    assert "Показатель 1" not in form and "Образец" not in form


def test_open_a_known_program_is_an_app_not_a_console_command(tmp_path):
    from core.agent.agent import Agent
    from core.agent.attachments import AttachmentStore

    class _Manager:
        llm_model = None
        hybrid_search = None

    agent = Agent(_Manager(), store=AttachmentStore(tmp_path))
    route = {"action": "command", "source": "none", "output": "chat", "language": "", "task": "т", "search_query": ""}
    assert agent._reroute_by_request(dict(route), "Открой калькулятор")["action"] == "app"
    assert agent._reroute_by_request(dict(route), "Открой порт 8080")["action"] == "command"


def test_a_question_without_norms_is_not_a_norms_check(tmp_path):
    from core.agent.agent import Agent
    from core.agent.attachments import AttachmentStore

    class _Manager:
        llm_model = None
        hybrid_search = None

    agent = Agent(_Manager(), store=AttachmentStore(tmp_path))
    agent._last_action = "compare"
    route = {"action": "check", "source": "attachments", "output": "chat", "language": "", "task": "т", "search_query": ""}
    routed = agent._reroute_by_request(dict(route), "Добавилось ли что-то новое про страхование?")
    assert routed["action"] == "answer" and routed["source"] == "previous_answer"
    assert agent._reroute_by_request(dict(route), "Соответствует ли прочность нормам?")["action"] == "check"

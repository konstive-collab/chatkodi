"""Prompt templates and answer cleanup for the local LLM (backend independent)."""

from __future__ import annotations

import re


def rag_prompt_fact_only(question: str, context: str) -> str:
    return f"""Ты анализируешь документы и отвечаешь только по предоставленному контексту.

ПРАВИЛА:
1. Это обычный фактологический вопрос по документам.
2. Не отказывай из-за приватности или конфиденциальности, если в контексте нет персональных данных людей.
3. Используй только текст документов ниже.
4. Если ответ найден, скажи его прямо и кратко.
5. Если ответа нет, напиши ровно: В документах нет информации.
6. Не выдумывай и не добавляй внешние знания.

КОНТЕКСТ:
{context}

ВОПРОС: {question}

ОТВЕТ:"""


def rag_prompt_full_documents(question: str, context: str) -> str:
    return f"""# ЗАДАЧА: Анализ документов

## РОЛЬ:
Ты - профессиональный аналитик документов. Твоя задача - проанализировать предоставленные документы и ответить на вопрос пользователя.

## ИНСТРУКЦИИ:
1. ВНИМАТЕЛЬНО прочитай ВСЕ предоставленные документы
2. Найди информацию, относящуюся к вопросу
3. Если информация есть в документах - дай ТОЧНЫЙ ответ
4. Если информации нет - скажи "В документах нет информации"
5. НИКОГДА не выдумывай информацию
6. Цитируй точные фразы из документов, когда это возможно
7. Указывай, из какого документа взята информация

## ВАЖНО:
- Документы разделены линией из знаков равенства (=======)
- Каждый документ начинается с "📄 ДОКУМЕНТ: [имя файла]"
- Используй информацию ИЗ ВСЕХ документов
- Сравнивай информацию между документами
- Выявляй противоречия если они есть

## ВОПРОС ПОЛЬЗОВАТЕЛЯ:
{question}

## ДОКУМЕНТЫ ДЛЯ АНАЛИЗА:
{context}

## ФОРМАТ ОТВЕТА:
1. КРАТКИЙ ОТВЕТ: [одно-два предложения]
2. ПОДРОБНЫЙ АНАЛИЗ: [развернутый ответ с цитатами]
3. ИСТОЧНИКИ: [из каких документов взята информация]

## НАЧНИ ОТВЕТ:
"""


def chat_prompt(question: str) -> str:
    return f"""Ты - полезный AI ассистент. Ответь на вопрос пользователя.

ВОПРОС: {question}

ОТВЕТ:"""


_REFUSAL_MARKERS = (
    "не могу предоставить информацию",
    "не могу помочь с этим",
    "не могу выполнить этот запрос",
    "конфиденциаль",
    "персональных данных",
    "законодательством о защите",
    "privacy",
    "sensitive personal data",
)


def looks_like_rag_refusal(text: str) -> bool:
    sample = (text or "").strip().lower()
    return bool(sample) and any(marker in sample for marker in _REFUSAL_MARKERS)


def trim_context(context: str, limit: int) -> str:
    """Keep whole paragraphs up to ``limit`` characters."""
    if not context or len(context) <= limit:
        return context
    blocks = [b.strip() for b in context.split("\n\n") if b.strip()]
    if not blocks:
        return context[:limit]
    kept = []
    used = 0
    for block in blocks:
        block_len = len(block) + 2
        if used + block_len > limit:
            remain = limit - used
            if remain > 300:
                kept.append(block[:remain])
            break
        kept.append(block)
        used += block_len
    return "\n\n".join(kept)


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_thinking(text: str) -> str:
    return _THINK_RE.sub("", text or "").strip()


def clean_response(response: str, question: str) -> str:
    """Remove template markers and an echoed question; collapse whitespace."""
    response = strip_thinking(response)
    if not response:
        return "Не удалось сгенерировать ответ."
    cleaned_lines = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith(("<|im_start|>", "<|im_end|>", "<|endoftext|>")):
            continue
        if not cleaned_lines and question.lower() in line.lower() and len(line.split()) < 10:
            continue
        cleaned_lines.append(line)
    cleaned = re.sub(r"\s+", " ", "\n".join(cleaned_lines)).strip()
    return cleaned if cleaned else "Ответ пуст."


def clean_rag_response(response: str, question: str, context: str) -> str:
    """Cleanup for the full-documents answer format."""
    response = strip_thinking(response)
    if not response:
        return "Не удалось проанализировать документы."
    cleaned_lines = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith(("#", "##", "###", "РОЛЬ:", "ИНСТРУКЦИИ:", "ФОРМАТ:")):
            continue
        if question.lower() in line.lower() and len(line.split()) < 15:
            continue
        cleaned_lines.append(line)
    result = re.sub(r"\s+", " ", "\n".join(cleaned_lines)).strip()
    if len(result) < 20:
        return "На основе анализа документов не удалось найти ответ на ваш вопрос."
    return result

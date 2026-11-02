from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
from typing import Dict, List, Optional

from core.document_catalog import CatalogRecord, DocumentCatalog
from core.logging_utils import get_logger
from core.paths import resolve_project_path, runtime_root
from core.structured_output import extract_json_object

from .docx_engine import create_legal_draft
from .router import detect_generation_kind
from .types import ArtifactDraft, AssistantResult, GenerationKind, PendingGeneration

logger = get_logger(__name__)


_TRAINING_LABELS = {
    "goal": "цель подготовки",
    "age": "возрастная группа",
    "level": "текущий уровень подготовки",
    "schedule": "число занятий в неделю",
    "duration": "длительность одного занятия",
    "equipment": "доступный инвентарь и место занятий",
    "limitations": "ограничения по здоровью или явное «ограничений нет»",
}


class GenerationService:
    def __init__(self, llm_model=None, config: Optional[Dict] = None):
        self.llm_model = llm_model
        self.config = dict(config or {})
        self.enabled = bool(self.config.get("enabled", True))
        catalog_root = str(self.config.get("catalog_root") or "documents")
        root_path = Path(catalog_root)
        if not root_path.is_absolute():
            root_path = resolve_project_path(root_path)
        catalog_file = runtime_root() / "document_catalog.json"
        self.catalog = DocumentCatalog(root_path, catalog_file)
        self.pending: Optional[PendingGeneration] = None
        self.last_result: Optional[AssistantResult] = None
        self.artifacts: Dict[str, ArtifactDraft] = {}

    def handle(self, question: str) -> Optional[AssistantResult]:
        question = (question or "").strip()
        if not question or not self.enabled:
            return None
        if self.pending:
            result = self._continue_pending(question)
            self.last_result = result
            return result
        kind = detect_generation_kind(question)
        if kind == GenerationKind.NONE:
            return None
        if kind == GenerationKind.TRAINING_PLAN:
            result = self._training_plan(question)
        elif kind == GenerationKind.EXERCISE_IDEATION:
            result = self._exercise_ideas(question)
        else:
            result = self._legal_document(question)
        self.last_result = result
        return result

    def _continue_pending(self, answer: str) -> AssistantResult:
        pending = self.pending
        self.pending = None
        assert pending is not None
        if pending.missing_fields == ["template_choice"]:
            paths = json.loads(pending.collected.get("candidate_paths", "[]"))
            selected = self._select_candidate(paths, answer)
            if not selected:
                self.pending = pending
                return AssistantResult(
                    "Не удалось определить выбранный образец. Укажите его номер или точное имя файла.",
                    generation_kind=GenerationKind.LEGAL_DOCUMENT,
                    clarification_fields=["template_choice"],
                )
            return self._legal_document(pending.original_request, selected_template=Path(selected))
        details = answer
        if pending.kind == GenerationKind.TRAINING_PLAN:
            return self._training_plan(pending.original_request, details=details)
        if pending.kind == GenerationKind.EXERCISE_IDEATION:
            return self._exercise_ideas(pending.original_request, details=details)
        selected = pending.collected.get("selected_template", "")
        return self._legal_document(
            pending.original_request,
            details=details,
            selected_template=Path(selected) if selected else None,
        )

    @staticmethod
    def _select_candidate(paths: List[str], answer: str) -> str:
        normalized = answer.strip().lower()
        number = re.search(r"\b([1-9])\b", normalized)
        if number:
            index = int(number.group(1)) - 1
            if 0 <= index < len(paths):
                return paths[index]
        for value in paths:
            name = Path(value).name.lower()
            if name in normalized or normalized in name:
                return value
        return ""

    def _training_missing(self, text: str) -> List[str]:
        q = text.lower().replace("ё", "е")
        checks = {
            "goal": bool(re.search(r"\b(?:цель|для\s+(?:развит|улучш|подготов|сдач|снижен|набор|вынослив|сил))", q)),
            "age": bool(re.search(r"\b(?:\d{1,2}\s*(?:лет|год)|взросл\w*|курсант\w*|подрост\w*|юнош\w*|старше|младше)\b", q)),
            "level": bool(re.search(r"\b(?:нович\w*|начальн\w*|средн\w*|продвинут\w*|подготовлен\w*|уровень)\b", q)),
            "schedule": bool(re.search(r"\b\d+\s*(?:раз|занят)\w*\s+(?:в|на)\s+недел", q)),
            "duration": bool(re.search(r"\b\d+\s*(?:минут|мин\.?|час|ч\.)\b", q)),
            "equipment": any(token in q for token in ("инвентар", "турник", "зал", "стадион", "бассейн", "без оборудования", "дома")),
            "limitations": any(token in q for token in ("ограничен", "травм", "здоров", "противопоказ", "ограничений нет")),
        }
        return [key for key, present in checks.items() if not present]

    def _training_plan(self, request: str, details: str = "") -> AssistantResult:
        combined = f"{request}\n{details}".strip()
        missing = self._training_missing(combined)
        if missing:
            self.pending = PendingGeneration(GenerationKind.TRAINING_PLAN, combined, missing)
            labels = [f"- {_TRAINING_LABELS[key]}" for key in missing]
            return AssistantResult(
                "Чтобы план был применимым и безопасным, уточните одним сообщением:\n" + "\n".join(labels),
                generation_kind=GenerationKind.TRAINING_PLAN,
                clarification_fields=missing,
            )
        if not self._ensure_llm():
            return self._llm_unavailable(GenerationKind.TRAINING_PLAN)
        sources = self.catalog.find_sources("sport", combined, limit=4)
        context = self._source_context(sources, total_chars=10000)
        creative_rule = (
            "упражнения и идеи, которых нет в контексте, включая плавание, разрешены только в разделе «Дополнение модели»;"
            if bool(self.config.get("allow_creative_exercises", True))
            else "не добавляй упражнения, которых нет в предоставленном контексте;"
        )
        prompt = f"""Ты методист по физической подготовке. Составь практический тренировочный план.

Верни ТОЛЬКО JSON:
{{"grounded": "календарный план, подтверждаемый контекстом", "creative": "дополнения вне контекста или пустая строка"}}

Требования:
- используй календарную структуру, нагрузку, восстановление, прогрессию и контрольные точки;
- не ставь медицинских диагнозов;
- в grounded помещай только сведения, поддержанные контекстом;
- в creative: {creative_rule}
- не приписывай дополнения источникам;
- заверши напоминанием о проверке плана преподавателем и учёте допуска врача при ограничениях.

Запрос и уточнения:
{combined}

Локальные источники:
{context or '[релевантные методички не найдены]'}
"""
        answer = self._generate_sport_sections(prompt, fallback=self._training_fallback(combined))
        return self._result_with_sources(answer, GenerationKind.TRAINING_PLAN, sources)

    def _exercise_ideas(self, request: str, details: str = "") -> AssistantResult:
        if not self._ensure_llm():
            return self._llm_unavailable(GenerationKind.EXERCISE_IDEATION)
        combined = f"{request}\n{details}".strip()
        sources = self.catalog.find_sources("sport", combined, limit=4)
        context = self._source_context(sources, total_chars=9000)
        creative_rule = (
            "Можно предлагать плавание и другие уместные упражнения вне документов, но нельзя выдавать их за положения методичек."
            if bool(self.config.get("allow_creative_exercises", True))
            else "Предлагай только упражнения, явно присутствующие в локальных методичках."
        )
        prompt = f"""Предложи упражнения по запросу пользователя.
Верни ТОЛЬКО JSON: {{"grounded": "упражнения из контекста", "creative": "новые упражнения вне контекста или пустая строка"}}.
{creative_rule}
Для каждого упражнения укажи цель, технику, дозировку, облегчение и условие прекращения. Не давай медицинских диагнозов.

Запрос: {combined}

Контекст локальных методичек:
{context or '[релевантные методички не найдены]'}
"""
        fallback = (
            "Безопасный структурированный ответ модели не получен. Возможные варианты для обсуждения с преподавателем: "
            "плавание в спокойном темпе, интервальная ходьба и облегчённые круговые упражнения. "
            "Для каждого варианта нагрузку и технику необходимо назначить после очной оценки подготовленности."
        )
        return self._result_with_sources(
            self._generate_sport_sections(prompt, fallback=fallback),
            GenerationKind.EXERCISE_IDEATION,
            sources,
        )

    def _legal_document(self, request: str, details: str = "", selected_template: Optional[Path] = None) -> AssistantResult:
        combined = f"{request}\n{details}".strip()
        if selected_template is None:
            candidates = self.catalog.find_legal_templates(request, limit=3)
            if not candidates:
                return AssistantResult(
                    "Подходящий безопасный DOCX-образец не найден. Добавьте образец или уточните вид документа; без подтверждения типовой файл не создаётся.",
                    generation_kind=GenerationKind.LEGAL_DOCUMENT,
                )
            if len(candidates) > 1:
                paths = [item.path for item in candidates]
                self.pending = PendingGeneration(
                    GenerationKind.LEGAL_DOCUMENT, request, ["template_choice"],
                    {"candidate_paths": json.dumps(paths, ensure_ascii=False)},
                )
                lines = [f"{index}. {item.name}" for index, item in enumerate(candidates, 1)]
                return AssistantResult(
                    "Найдено несколько подходящих образцов. Выберите номер или имя файла:\n" + "\n".join(lines),
                    generation_kind=GenerationKind.LEGAL_DOCUMENT,
                    sources=[item.name for item in candidates],
                    clarification_fields=["template_choice"],
                )
            selected_template = Path(candidates[0].path)

        missing = self._legal_missing(combined)
        if missing:
            self.pending = PendingGeneration(
                GenerationKind.LEGAL_DOCUMENT, combined, missing,
                {"selected_template": str(selected_template)},
            )
            return AssistantResult(
                "Для безопасной замены реквизитов уточните одним сообщением:\n" + "\n".join(f"- {item}" for item in missing),
                generation_kind=GenerationKind.LEGAL_DOCUMENT,
                sources=[selected_template.name],
                clarification_fields=missing,
            )
        if not self._ensure_llm():
            return self._llm_unavailable(GenerationKind.LEGAL_DOCUMENT)

        artifact = create_legal_draft(
            selected_template,
            combined,
            runtime_root() / "drafts",
            self._generate_raw,
        )
        self.artifacts[artifact.artifact_id] = artifact
        preview = artifact.preview_text[:6000].strip()
        if artifact.can_save:
            text = (
                f"Черновик подготовлен по образцу «{selected_template.name}».\n\n"
                f"{preview}\n\n"
                "Документ прошёл проверку реквизитов. Используйте кнопку «Сохранить как DOCX». "
                "Перед подписанием проверьте проект у ответственного специалиста."
            )
        else:
            errors = "\n".join(f"- {item}" for item in artifact.validation_errors)
            text = (
                f"Черновик по образцу «{selected_template.name}» создан, но сохранение заблокировано:\n{errors}\n\n"
                "Уточните адресата, автора, должность/звание, даты, место и номер документа одним сообщением."
            )
            self.pending = PendingGeneration(
                GenerationKind.LEGAL_DOCUMENT, request, ["legal_requisites"],
                {"selected_template": str(selected_template)},
            )
        return AssistantResult(
            text,
            generation_kind=GenerationKind.LEGAL_DOCUMENT,
            sources=[selected_template.name],
            artifact=artifact,
        )

    @staticmethod
    def _legal_missing(text: str) -> List[str]:
        q = text.lower().replace("ё", "е")
        if "командиров" not in q:
            return [] if len(q) >= 100 else ["адресат", "автор и его должность/звание", "содержание и основание", "дата и номер"]
        checks = [
            (bool(re.search(r"\b(?:для|направить|командировать)\s+[а-яёa-z]", q)), "ФИО и должность командируемого"),
            (bool(re.search(r"\b(?:в\s+город|в\s+г\.|место назначения|в\s+[А-ЯЁ][а-яё]+)", text)), "город или место назначения"),
            (len(re.findall(r"\b\d{1,2}[./-]\d{1,2}(?:[./-]\d{2,4})?\b", q)) >= 2 or " с " in q and " по " in q, "даты начала и окончания"),
            (any(token in q for token in ("цель", "для участия", "для проведения", "в связи", "основан")), "цель и основание командировки"),
            (any(token in q for token in ("адресат", "начальник", "командир", "руководител")), "адресат рапорта"),
            (any(token in q for token in ("от ", "автор", "подпис", "должност", "звание")), "автор, должность/звание и дата подписи"),
        ]
        return [label for present, label in checks if not present]

    def _source_context(self, sources: List[CatalogRecord], total_chars: int) -> str:
        parts: List[str] = []
        used = 0
        for item in sources:
            remaining = total_chars - used
            if remaining <= 200:
                break
            text = self.catalog.read_text(item, max_chars=min(7000, remaining))
            if not text:
                continue
            block = f"\n=== {item.name} ===\n{text}"
            parts.append(block)
            used += len(block)
        return "".join(parts)[:total_chars]

    def _result_with_sources(self, answer: str, kind: GenerationKind, sources: List[CatalogRecord]) -> AssistantResult:
        names = [item.name for item in sources]
        text = (answer or "Не удалось сформировать ответ.").strip()
        if names:
            text += "\n\n📚 Источники: " + ", ".join(names)
        return AssistantResult(text, generation_kind=kind, sources=names)

    def _ensure_llm(self) -> bool:
        model = self.llm_model
        if not model:
            return False
        try:
            if hasattr(model, "is_loaded") and not model.is_loaded() and hasattr(model, "load_now"):
                loaded = model.load_now()
                if loaded is not None:
                    model = loaded
            return bool(model and model.is_available())
        except Exception as exc:
            logger.warning("Generation LLM unavailable: %s", type(exc).__name__)
            return False

    def _generate(self, prompt: str) -> str:
        try:
            result = self.llm_model.generate_clean_answer(
                prompt, context="", rag_mode=False, streaming_callback=False, use_cache=False,
            )
            return str(result or "").strip()
        except TypeError:
            return str(self.llm_model.generate_clean_answer(prompt, rag_mode=False, use_cache=False) or "").strip()

    def _generate_raw(self, prompt: str) -> str:
        model = self.llm_model
        if hasattr(model, "generate_with_callback"):
            chunks: List[str] = []
            model.generate_with_callback(
                messages=[{"role": "user", "content": prompt}],
                callback=lambda token, is_final: chunks.append(token) if token else None,
                max_tokens=int(self.config.get("legal_patch_max_tokens", 1200)),
                temperature=0.05,
                top_p=0.8,
                context_limit=int(self.config.get("legal_context_limit", 24000)),
            )
            return "".join(chunks)
        return self._generate(prompt)

    def _generate_sport_sections(self, prompt: str, *, fallback: str) -> str:
        raw = self._generate_raw_limited(
            prompt,
            max_tokens=int(self.config.get("sport_max_tokens", 560)),
            context_limit=int(self.config.get("sport_context_limit", 14000)),
        )
        payload = extract_json_object(raw) or {}
        grounded = str(payload.get("grounded") or "").strip() if isinstance(payload, dict) else ""
        creative = str(payload.get("creative") or "").strip() if isinstance(payload, dict) else ""
        if not grounded and not creative:
            # Conservative fallback: uncertain free text is never attributed to
            # the manuals. This preserves the source/creation boundary.
            creative = fallback.strip()
            grounded = "Структурированные положения источников не удалось безопасно отделить от генерации."
        if not creative:
            creative = "Дополнительные упражнения вне найденных методичек не добавлялись."
        return (
            "Основано на документах\n"
            + grounded
            + "\n\nДополнение модели\n"
            + creative
            + "\n\nПроверьте проект у преподавателя физической подготовки и учитывайте медицинский допуск при наличии ограничений."
        )

    @staticmethod
    def _training_fallback(request: str) -> str:
        """Useful conservative draft when a local model returns malformed output."""
        q = request.lower().replace("ё", "е")
        sessions_match = re.search(r"\b([2-6])\s*(?:раз|занят)\w*\s+(?:в|на)\s+недел", q)
        sessions = int(sessions_match.group(1)) if sessions_match else 3
        duration_match = re.search(r"\b(\d{2,3})\s*(?:минут|мин\.?)\b", q)
        duration = int(duration_match.group(1)) if duration_match else 60
        pool = "бассейн" in q or "плаван" in q
        session_names = [
            "равномерная аэробная работа и техника бега",
            "силовая работа с собственным весом и турником",
            "интервальная работа умеренной интенсивности",
            "восстановительное плавание" if pool else "восстановительная ходьба и мобильность",
            "смешанная круговая тренировка",
            "лёгкая координационная работа",
        ]
        weeks = []
        loads = ["60–65%", "65–70%", "70–75%", "50–60%"]
        notes = [
            "входная неделя; зафиксировать самочувствие и исходный результат",
            "увеличить основной объём примерно на 10% при нормальном восстановлении",
            "пиковая неделя; не повышать одновременно объём и интенсивность",
            "разгрузка; повторить контрольную точку в сопоставимых условиях",
        ]
        for week in range(4):
            items = []
            for index in range(sessions):
                name = session_names[index % len(session_names)]
                items.append(f"занятие {index + 1} — {name}, до {duration} мин")
            weeks.append(f"Неделя {week + 1} ({loads[week]}): " + "; ".join(items) + f". {notes[week]}.")
        return (
            "Резервный проект календаря (создан программой, не является цитатой из методичек):\n"
            + "\n".join(weeks)
            + "\nМежду тяжёлыми занятиями оставлять не менее одного дня восстановления; "
            "уменьшить или прекратить нагрузку при боли, головокружении или необычном ухудшении самочувствия."
        )

    def _generate_raw_limited(self, prompt: str, *, max_tokens: int, context_limit: int) -> str:
        model = self.llm_model
        if hasattr(model, "generate_with_callback"):
            chunks: List[str] = []
            model.generate_with_callback(
                messages=[{"role": "user", "content": prompt}],
                callback=lambda token, is_final: chunks.append(token) if token else None,
                max_tokens=max_tokens,
                temperature=0.1,
                top_p=0.85,
                context_limit=context_limit,
            )
            return "".join(chunks).strip()
        return self._generate(prompt)

    @staticmethod
    def _llm_unavailable(kind: GenerationKind) -> AssistantResult:
        return AssistantResult(
            "Для создания нового материала нужна локальная LLM. Откройте лаунчер, выберите установленную модель и разрешите её запуск. Поиск по документам остаётся доступен без генерации.",
            generation_kind=kind,
        )

    def get_last_result(self) -> Optional[AssistantResult]:
        return self.last_result

    def set_catalog_root(self, root: Path | str) -> None:
        self.catalog = DocumentCatalog(Path(root), runtime_root() / "document_catalog.json")

    def get_result_for_text(self, text: str) -> Optional[AssistantResult]:
        result = self.last_result
        return result if result and result.text == (text or "") else None

    def save_artifact(self, artifact_id: str, destination: Path | str) -> Path:
        artifact = self.artifacts.get(artifact_id)
        if not artifact:
            raise ValueError("Черновик не найден в текущей сессии")
        if not artifact.can_save:
            raise ValueError("Сохранение заблокировано проверкой реквизитов")
        target = Path(destination)
        if target.suffix.lower() != ".docx":
            target = target.with_suffix(".docx")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(artifact.draft_path, target)
        return target

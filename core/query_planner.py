"""Optional LLM-assisted query planner for ChatKodi 1.7.

Planner never answers the user question. It only refines routing/instruction
selection on top of the deterministic parser from ``core.query_understanding``.
If anything fails, the old rule-based logic remains the source of truth.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from core.logging_utils import get_logger
from core.query_understanding import QueryUnderstandingResult, analyze_question
from core.system_profile import detect_system_profile
from core.structured_output import extract_json_object

logger = get_logger(__name__)

_ALLOWED_TYPES = {
    'binary_fact', 'location', 'method', 'count', 'frequency',
    'timeline', 'comparison', 'range_summary', 'summary', 'activity_report', 'target', 'fact'
}
_ALLOWED_FOCUS = {'general', 'direction', 'method', 'target', 'date', 'count'}


@dataclass
class PlannerDecision:
    question_type: Optional[str] = None
    wants_statistics: Optional[bool] = None
    wants_summary: Optional[bool] = None
    wants_comparison: Optional[bool] = None
    wants_timeline: Optional[bool] = None
    wants_range_aggregation: Optional[bool] = None
    statistics_focus: Optional[str] = None
    prefer_structured_answer: Optional[bool] = None
    llm_presentation_allowed: Optional[bool] = None
    confidence: float = 0.0
    source: str = 'rules'
    notes: List[str] | None = None
    raw_json: Dict[str, Any] | None = None


class QueryPlanner:
    def __init__(self, llm_model=None, planner_cfg: Optional[Dict[str, Any]] = None):
        self.llm_model = llm_model
        self.cfg = dict(planner_cfg or {})
        self.enabled = bool(self.cfg.get('enabled', False))
        self.use_llm = bool(self.cfg.get('use_llm_router', True))
        self.min_confidence = float(self.cfg.get('min_confidence', 0.72))
        self.max_tokens = int(self.cfg.get('max_tokens', 160))
        self.context_limit = int(self.cfg.get('context_limit', 1800))
        self.temperature = float(self.cfg.get('temperature', 0.05))
        self.use_on_simple = bool(self.cfg.get('use_on_simple_questions', False))
        self.timeout_seconds = float(self.cfg.get('timeout_seconds', 2.5))
        self.cache_size = int(self.cfg.get('cache_size', 128))
        self._cache: Dict[str, QueryUnderstandingResult] = {}
        self._failures = 0
        self._slow_calls = 0
        self._disabled_until = 0.0

        # На CPU-only/low-RAM машине LLM-router опасен: он может незаметно
        # загрузить локальную LLM ради одного JSON. Поэтому он автоматически
        # выключается, а rule-based планировщик остаётся рабочим.
        try:
            profile = detect_system_profile()
            if (not profile.has_cuda and profile.total_ram_gb and profile.total_ram_gb <= 16.5):
                self.use_llm = False
                logger.info("Query planner LLM-router disabled for CPU/low-RAM profile: %s", profile.summary_line)
        except Exception:
            pass

    def analyze(self, question: str, reference_date=None) -> QueryUnderstandingResult:
        reference_key = reference_date.strftime("%Y-%m-%d") if reference_date else ""
        cache_key = ((question or "").strip().lower(), reference_key)
        if cache_key in self._cache:
            return self._cache[cache_key]
        base = analyze_question(question, reference_date=reference_date)
        if not self.enabled:
            self._remember(cache_key, base)
            return base
        if time.time() < self._disabled_until:
            self._remember(cache_key, base)
            return base
        if not self._should_consult_llm(base, question):
            self._remember(cache_key, base)
            return base
        started = time.time()
        decision = self._ask_llm(question, base)
        elapsed = time.time() - started
        if elapsed > self.timeout_seconds:
            self._slow_calls += 1
            if self._slow_calls >= 2:
                # Железо стабильно не успевает: роутер — оптимизация, а не
                # обязательный шаг. Отключаем его до перезапуска, чтобы каждый
                # вопрос не ждал генерацию служебного JSON.
                self.use_llm = False
                logger.warning("Query planner LLM-router disabled for this session: %.2fs per call", elapsed)
            else:
                self._disabled_until = time.time() + 300.0
                logger.warning("Query planner LLM-router was too slow: %.2fs; paused for 5 minutes", elapsed)
        else:
            self._slow_calls = 0
        if not decision:
            self._remember(cache_key, base)
            return base
        merged = self._merge(base, decision)
        self._remember(cache_key, merged)
        return merged

    def _remember(self, key: str, intent: QueryUnderstandingResult) -> None:
        if not key:
            return
        self._cache[key] = intent
        if len(self._cache) > self.cache_size:
            try:
                first_key = next(iter(self._cache))
                self._cache.pop(first_key, None)
            except Exception:
                self._cache.clear()

    def _should_consult_llm(self, base: QueryUnderstandingResult, question: str) -> bool:
        if not self.use_llm:
            return False
        llm = self.llm_model
        if not llm or not hasattr(llm, 'generate_with_callback') or not llm.is_available():
            return False
        if self.use_on_simple:
            return True
        q = (question or '').strip()
        if base.question_type in {'frequency', 'timeline', 'comparison', 'range_summary', 'summary'}:
            return True
        if len(q) > 90:
            return True
        if q.count("?") > 1 or q.count(";") > 1:
            return True
        return False

    def _ask_llm(self, question: str, base: QueryUnderstandingResult) -> Optional[PlannerDecision]:
        prompt = self._build_prompt(question, base)
        result: List[str] = []
        try:
            self.llm_model.generate_with_callback(
                messages=[{'role': 'user', 'content': prompt}],
                callback=lambda token, is_final: result.append(token) if token else None,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=0.8,
                context_limit=self.context_limit,
                deadline_seconds=self.timeout_seconds,
            )
            raw = ''.join(result).strip()
            payload = self._extract_json(raw)
            if not payload:
                logger.info('Query planner ignored non-JSON output: %s', raw[:200])
                return None
            decision = self._parse_payload(payload)
            if not decision or decision.confidence < self.min_confidence:
                return None
            return decision
        except Exception as e:
            self._failures += 1
            if self._failures >= 2:
                self._disabled_until = time.time() + min(120.0, 15.0 * self._failures)
            logger.warning('Query planner LLM failed: %s', e)
            return None

    def _build_prompt(self, question: str, base: QueryUnderstandingResult) -> str:
        base_summary = {
            'question_type': base.question_type,
            'date_mode': base.date_mode,
            'object_terms': base.object_terms[:4],
            'identifiers': base.identifiers[:4],
            'keywords': base.keywords[:10],
            'statistics_focus': base.statistics_focus,
        }
        return f"""Ты — Query Planner. Ты НЕ отвечаешь на вопрос пользователя.
Твоя задача — выбрать режим обработки запроса для алгоритмической системы.

Верни ТОЛЬКО JSON без пояснений и markdown.

Разрешённые question_type:
- binary_fact
- location
- method
- count
- frequency
- timeline
- comparison
- range_summary
- summary
- activity_report
- target
- fact

Разрешённые statistics_focus:
- general
- direction
- method
- target
- date
- count

Если вопрос про диапазон, частоту, сравнение, по дням, где чаще, чем чаще — это НЕ single-fact.
Если вопрос вида «подробная работа/активность X за период» — используй activity_report.
Если вопрос про один конкретный факт/место/способ — используй binary_fact/location/method/fact.
Если пользователь просит перечислить объекты или результаты событий — используй target.

Базовый rule-based разбор:
{json.dumps(base_summary, ensure_ascii=False)}

Вопрос пользователя:
{question}

Формат ответа:
{{
  "question_type": "frequency",
  "wants_statistics": true,
  "wants_summary": false,
  "wants_comparison": false,
  "wants_timeline": false,
  "wants_range_aggregation": true,
  "statistics_focus": "direction",
  "prefer_structured_answer": true,
  "llm_presentation_allowed": true,
  "confidence": 0.84,
  "notes": ["range", "stats_by_direction"]
}}
"""

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        return extract_json_object(text)

    def _parse_payload(self, payload: Dict[str, Any]) -> Optional[PlannerDecision]:
        qtype = str(payload.get('question_type') or '').strip()
        if qtype and qtype not in _ALLOWED_TYPES:
            qtype = ''
        focus = str(payload.get('statistics_focus') or '').strip() or None
        if focus and focus not in _ALLOWED_FOCUS:
            focus = None
        try:
            confidence = float(payload.get('confidence', 0.0) or 0.0)
        except Exception:
            confidence = 0.0
        notes = payload.get('notes') or []
        if isinstance(notes, str):
            notes = [notes]
        notes = [str(x)[:80] for x in notes][:8]
        return PlannerDecision(
            question_type=qtype or None,
            wants_statistics=self._maybe_bool(payload.get('wants_statistics')),
            wants_summary=self._maybe_bool(payload.get('wants_summary')),
            wants_comparison=self._maybe_bool(payload.get('wants_comparison')),
            wants_timeline=self._maybe_bool(payload.get('wants_timeline')),
            wants_range_aggregation=self._maybe_bool(payload.get('wants_range_aggregation')),
            statistics_focus=focus,
            prefer_structured_answer=self._maybe_bool(payload.get('prefer_structured_answer')),
            llm_presentation_allowed=self._maybe_bool(payload.get('llm_presentation_allowed')),
            confidence=confidence,
            source='llm_router',
            notes=notes,
            raw_json=payload,
        )

    @staticmethod
    def _maybe_bool(value: Any) -> Optional[bool]:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {'true', 'yes', '1'}:
                return True
            if lowered in {'false', 'no', '0'}:
                return False
        return None

    def _merge(self, base: QueryUnderstandingResult, decision: PlannerDecision) -> QueryUnderstandingResult:
        qtype = decision.question_type or base.question_type
        # Safety: do not downgrade strong exact fact unless range/statistics intent is explicit.
        if base.question_type in {'binary_fact', 'location', 'method'} and qtype in {'summary', 'fact'}:
            qtype = base.question_type
        base.question_type = qtype
        if decision.wants_statistics is not None:
            base.wants_statistics = decision.wants_statistics
        if decision.wants_summary is not None:
            base.wants_summary = decision.wants_summary
        if decision.wants_comparison is not None:
            base.wants_comparison = decision.wants_comparison
        if decision.wants_timeline is not None:
            base.wants_timeline = decision.wants_timeline
        if decision.wants_range_aggregation is not None:
            base.wants_range_aggregation = decision.wants_range_aggregation
        if decision.statistics_focus:
            base.statistics_focus = decision.statistics_focus
        if decision.prefer_structured_answer is not None:
            base.prefer_structured_answer = decision.prefer_structured_answer
        if decision.llm_presentation_allowed is not None:
            base.llm_presentation_allowed = decision.llm_presentation_allowed
        base.routing_source = decision.source
        base.routing_confidence = decision.confidence
        base.routing_notes = list(decision.notes or [])

        # Recompute derived flags conservatively.
        if base.question_type in {'count', 'frequency', 'timeline', 'comparison', 'summary', 'range_summary', 'target'}:
            base.wants_statistics = True
        if base.question_type in {'timeline', 'range_summary'}:
            base.wants_timeline = True
        if base.question_type == 'comparison':
            base.wants_comparison = True
        if base.question_type in {'summary', 'comparison', 'timeline', 'range_summary'}:
            base.wants_summary = True
        if base.question_type in {'binary_fact', 'location', 'method', 'fact'}:
            base.wants_exact_fact = True
        if base.date_mode in {'range', 'discrete_dates'} and base.question_type in {'count', 'frequency', 'timeline', 'comparison', 'summary', 'range_summary', 'target'}:
            base.wants_range_aggregation = True
        return base

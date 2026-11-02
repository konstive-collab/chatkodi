"""Опциональные вспомогательные средства для русской лемматизации.

The search stack can work without ``pymorphy3``. When it is installed we
use it to normalize Russian words; otherwise the code falls back to
lowercased text so the application still starts cleanly.
"""

from __future__ import annotations

import re
import time
import threading
from functools import lru_cache
from typing import Any, Dict, List, Optional

from core.logging_utils import get_logger


logger = get_logger(__name__)

try:
    import pymorphy3

    PYMORPHY_AVAILABLE = True
except ImportError:
    pymorphy3 = None
    PYMORPHY_AVAILABLE = False
    logger.info("pymorphy3 is not installed; lemmatization is disabled")


class RussianLemmatizer:
    """Singleton-лемматизатор, используемый при токенизации для поиска."""

    _instance: Optional["RussianLemmatizer"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "RussianLemmatizer":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init()
            return cls._instance

    def _init(self) -> None:
        """Инициализирует опциональный анализатор pymorphy и runtime-состояние."""
        self.available = PYMORPHY_AVAILABLE
        self.word_pattern = re.compile(r"\b[а-яёА-ЯЁ]+\b")
        self.short_word_pattern = re.compile(r"\b[а-яёА-ЯЁ]{1,2}\b")
        self.start_time = time.time()
        self.morph = None

        if self.available and pymorphy3 is not None:
            try:
                self.morph = pymorphy3.MorphAnalyzer()
            except Exception as exc:
                logger.warning("Failed to initialize pymorphy3: %s", exc)
                self.available = False
                self.morph = None

    @lru_cache(maxsize=100000)
    def _lemmatize_word_cached(self, word: str) -> str:
        """Нормализует одно слово и запоминает результат для повторного использования."""
        source = (word or "").lower()
        if not self.available or not self.morph or len(source) <= 2:
            return source

        try:
            parsed = self.morph.parse(source)
            if parsed:
                lemma = parsed[0].normal_form
                if lemma:
                    return lemma
        except Exception:
            return source

        return source

    def lemmatize_word(self, word: str) -> str:
        """Возвращает лемму одного слова или исходное значение как fallback."""
        if not word:
            return word
        return self._lemmatize_word_cached(word)

    def lemmatize_text(self, text: str) -> str:
        """Извлекает русские слова из текста и объединяет их леммы."""
        if not text:
            return ""
        if not self.available:
            return text.lower()

        words = self.word_pattern.findall(text)
        if not words:
            return ""
        return " ".join(self.lemmatize_word(word) for word in words)

    def lemmatize_batch(self, texts: List[str]) -> List[str]:
        """Лемматизирует несколько текстов по порядку."""
        return [self.lemmatize_text(text) for text in texts]

    def batch_lemmatize_words(self, words: List[str]) -> List[str]:
        """Лемматизирует несколько отдельных слов по порядку."""
        return [self.lemmatize_word(word) for word in words]

    def is_short_word(self, word: str) -> bool:
        """Возвращает ``True`` для очень коротких слов, которые обычно несут мало сигнала."""
        return bool(self.short_word_pattern.fullmatch(word))

    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику кэша и производительности лемматизатора."""
        cache_info = self._lemmatize_word_cached.cache_info()
        total = cache_info.hits + cache_info.misses
        uptime = max(time.time() - self.start_time, 1e-9)
        return {
            "available": self.available,
            "total_words_processed": total,
            "cache_hits": cache_info.hits,
            "cache_misses": cache_info.misses,
            "cache_hit_rate": cache_info.hits / total if total else 0.0,
            "cache_size": cache_info.currsize,
            "uptime_seconds": uptime,
            "words_per_second": total / uptime,
        }

    def clear_cache(self) -> None:
        """Очищает кэшированные леммы, накопленные за текущий процесс."""
        self._lemmatize_word_cached.cache_clear()
        logger.info("Lemmatizer cache cleared")

    def __str__(self) -> str:
        stats = self.get_stats()
        if not self.available:
            return "Lemmatizer: unavailable"
        return (
            f"Lemmatizer: ready | words={stats['total_words_processed']} "
            f"| hit_rate={stats['cache_hit_rate']:.1%}"
        )


_lemmatizer_instance: Optional[RussianLemmatizer] = None


def get_lemmatizer() -> RussianLemmatizer:
    """Возвращает глобальный singleton-лемматизатор для текущего процесса."""
    global _lemmatizer_instance
    if _lemmatizer_instance is None:
        _lemmatizer_instance = RussianLemmatizer()
    return _lemmatizer_instance

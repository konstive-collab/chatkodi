"""Общие вспомогательные функции логирования для всего приложения.

Проект пишет в единый корневой логгер и даёт лёгкие дочерние логгеры
для отдельных модулей. Это позволяет держать консольное и файловое
логирование единообразным без дублирования настройки в каждом файле.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from core.paths import ensure_parent


def setup_logging(log_file: str = '.chatkodi/logs/chatkodi.log', level: int = logging.INFO) -> logging.Logger:
    """Один раз создаёт общий логгер приложения и затем переиспользует его."""
    logger = logging.getLogger('chatkodi')
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    try:
        file_path = ensure_parent(log_file)
        file_handler = logging.FileHandler(file_path, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception:
        # Файловое логирование опционально. Приложение может работать
        # только с консольным логом, например в среде только для чтения.
        pass

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Возвращает корневой логгер ChatKodi или дочерний логгер с пространством имён."""
    base = setup_logging()
    if not name or name == 'chatkodi':
        return base
    return logging.getLogger(f'chatkodi.{name}')

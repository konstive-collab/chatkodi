"""Совместимая точка входа для подготовки локальных моделей.

Основная реализация перенесена в ``tools.model_manager.downloader`` без изменения поведения.
"""

from tools.model_manager.downloader import *  # noqa: F401,F403

if __name__ == "__main__":
    import sys
    from tools.model_manager.downloader import main
    sys.exit(main())

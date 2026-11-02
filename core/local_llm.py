"""Совместимая точка входа для локальной LLM.

Основная реализация перенесена в ``core.llm.runtime`` без изменения поведения.
"""

from core.llm.runtime import *  # noqa: F401,F403

if __name__ == "__main__":
    import runpy
    runpy.run_module("core.llm.runtime", run_name="__main__")

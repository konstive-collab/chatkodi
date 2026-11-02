#!/usr/bin/env python3
"""Совместимая точка входа для сборки support bundle."""

from tools.support_bundle import *  # noqa: F401,F403

if __name__ == "__main__":
    from tools.support_bundle import main
    raise SystemExit(main())

#!/usr/bin/env python3
"""Совместимая точка входа для healthcheck."""

from tools.healthcheck import *  # noqa: F401,F403

if __name__ == "__main__":
    from tools.healthcheck import main
    raise SystemExit(main())

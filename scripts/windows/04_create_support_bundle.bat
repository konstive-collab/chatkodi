@echo off
cd /d "%~dp0..\.."
if exist .venv\Scripts\python.exe (
  .venv\Scripts\python.exe support_bundle.py
) else (
  python support_bundle.py
)
pause

@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo [ChatKodi] Быстрая установка лаунчера. Полная подготовка выполняется внутри лаунчера.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\windows\install_launcher.ps1"
if errorlevel 1 pause

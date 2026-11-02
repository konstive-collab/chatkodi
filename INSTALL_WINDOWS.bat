@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo [ChatKodi] Installing launcher. Complete setup from the launcher window.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\windows\install_launcher.ps1"
if errorlevel 1 pause

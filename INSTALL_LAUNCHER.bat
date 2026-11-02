@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
echo.
echo [ChatKodi] Installing and starting the launcher...
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\windows\install_launcher.ps1"
if errorlevel 1 (
  echo.
  echo [ChatKodi] Launcher installation failed.
  pause
  exit /b 1
)
endlocal

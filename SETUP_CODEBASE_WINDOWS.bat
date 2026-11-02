@echo off
setlocal
cd /d "%~dp0"
if not exist "chatkodi_config.json" copy /y "chatkodi_config.portable.json" "chatkodi_config.json" >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\windows\install_all.ps1" -SkipModelCheck
if errorlevel 1 exit /b %errorlevel%
echo.
echo Base installation completed. Models are installed separately:
echo   .venv\Scripts\python.exe download_models.py
endlocal

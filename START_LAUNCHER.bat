@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto install
".venv\Scripts\python.exe" -c "import PyQt5" >nul 2>nul
if errorlevel 1 goto install
call scripts\windows\05_launcher.bat
exit /b %errorlevel%
:install
call INSTALL_LAUNCHER.bat

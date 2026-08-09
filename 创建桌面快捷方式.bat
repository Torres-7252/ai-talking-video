@echo off
cd /d "%~dp0"
powershell -ExecutionPolicy Bypass -File "创建桌面快捷方式.ps1"
pause

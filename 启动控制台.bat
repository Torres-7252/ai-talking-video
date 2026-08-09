@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0启动控制台.ps1"
if errorlevel 1 pause

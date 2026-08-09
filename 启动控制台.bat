@echo off
chcp 65001 >nul
title AI数字人口播视频生成器

cd /d "D:\workspaces\ai-talking-video"

echo.
echo  ╔══════════════════════════════════════════════╗
echo  ║     AI数字人口播视频生成器 v0.2.0           ║
echo  ║     文案 → 声音 → 数字人 → 字幕 → 视频     ║
echo  ╚══════════════════════════════════════════════╝
echo.

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python 未安装，请先安装 Python 3.10+
    pause
    exit /b 1
)

:: Check if project exists
if not exist "scripts\web_server.py" (
    echo [ERROR] 项目目录不完整，请确认路径: D:\workspaces\ai-talking-video
    pause
    exit /b 1
)

echo [1/3] 检查环境...
python scripts\check_environment.py
if %errorlevel% neq 0 (
    echo.
    echo [WARN] 部分环境检查未通过，但仍可启动控制台
    echo.
)

echo.
echo [2/3] 启动 Web 服务器...
echo [3/3] 打开浏览器...
echo.
echo  控制台地址: http://127.0.0.1:8080
echo  按 Ctrl+C 停止服务器
echo.

start http://127.0.0.1:8080

python scripts\web_server.py

pause

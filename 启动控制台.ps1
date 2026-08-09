# AI数字人口播视频生成器 - 一键启动
Set-Location "D:\workspaces\ai-talking-video"

Write-Host ""
Write-Host "  ╔══════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "  ║     AI数字人口播视频生成器 v0.2.0           ║" -ForegroundColor Cyan
Write-Host "  ╚══════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Check Python
try {
    python --version 2>$null | Out-Null
} catch {
    Write-Host "[ERROR] Python 未安装" -ForegroundColor Red
    Read-Host "按 Enter 退出"
    exit 1
}

# Check project
if (-not (Test-Path "scripts\web_server.py")) {
    Write-Host "[ERROR] 项目目录不完整" -ForegroundColor Red
    Read-Host "按 Enter 退出"
    exit 1
}

# Check environment
Write-Host "[1/3] 检查环境...`n" -ForegroundColor Yellow
python scripts\check_environment.py
Write-Host ""

# Start
Write-Host "[2/3] 启动 Web 服务器..." -ForegroundColor Yellow
Write-Host "[3/3] 打开浏览器...`n" -ForegroundColor Yellow
Write-Host "  控制台地址: http://127.0.0.1:8080" -ForegroundColor Green
Write-Host "  按 Ctrl+C 停止服务器`n" -ForegroundColor Gray

Start-Process "http://127.0.0.1:8080"
python scripts\web_server.py

Read-Host

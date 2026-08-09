# Create desktop shortcut
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ws = New-Object -ComObject WScript.Shell
$s = $ws.CreateShortcut("$env:USERPROFILE\Desktop\AI数字人口播视频生成器.lnk")
$s.TargetPath = 'powershell.exe'
$s.Arguments = "-NoExit -ExecutionPolicy Bypass -File `"$scriptDir\启动控制台.ps1`""
$s.WorkingDirectory = $scriptDir
$s.IconLocation = 'powershell.exe,0'
$s.Description = 'AI数字人口播视频生成器'
$s.Save()

Write-Host "桌面快捷方式已创建" -ForegroundColor Green

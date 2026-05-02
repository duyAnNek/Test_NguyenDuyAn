# Chay: ros2 launch gps_visual_b system_b.launch.py
# Goi: D:/NguyenDuyAn_test/gps_system/scripts/launch_system_b.ps1

$ErrorActionPreference = "Stop"
$here = $PSScriptRoot -replace "\\", "/"
. (Join-Path $here "source_ros2_workspace.ps1")

if (-not (Get-Command ros2 -ErrorAction SilentlyContinue)) {
    Write-Host "Lenh ros2 van khong co trong PATH sau khi source." -ForegroundColor Red
    exit 1
}

ros2 launch gps_visual_b system_b.launch.py @args

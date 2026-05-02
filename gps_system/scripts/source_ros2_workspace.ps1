# Source ROS 2 Humble + workspace (dung duong dan /).
# Cach dung (tu PowerShell):
#   . D:/NguyenDuyAn_test/gps_system/scripts/source_ros2_workspace.ps1
#
# Neu ROS khong o vi tri mac dinh, dat truoc:
#   $env:ROS2_SETUP_SCRIPT = "D:/path/to/ros2_humble/local_setup.ps1"

$ErrorActionPreference = "Stop"

$ros2Setup = $env:ROS2_SETUP_SCRIPT
if (-not $ros2Setup) {
    $candidates = @(
        "C:/opt/ros/humble/local_setup.ps1",
        "C:/dev/ros2_humble/local_setup.ps1",
        "C:/pixi_ros2_ws/install/setup.ps1"
    )
    foreach ($p in $candidates) {
        if (Test-Path -LiteralPath $p) {
            $ros2Setup = $p
            break
        }
    }
}

if (-not $ros2Setup -or -not (Test-Path -LiteralPath $ros2Setup)) {
    Write-Host "Khong tim thay local_setup.ps1 cua ROS 2 Humble." -ForegroundColor Red
    Write-Host "Hay cai ROS 2 Humble (Windows) hoac dat:" -ForegroundColor Yellow
    Write-Host '  $env:ROS2_SETUP_SCRIPT = "D:/duong/dan/toi/local_setup.ps1"' -ForegroundColor Yellow
    Write-Host "Roi chay lai: . D:/NguyenDuyAn_test/gps_system/scripts/source_ros2_workspace.ps1"
    return
}

Write-Host "Dang source ROS 2: $ros2Setup" -ForegroundColor Cyan
. $ros2Setup

$scriptDir = $PSScriptRoot -replace "\\", "/"
$ros2Ws = (Split-Path -Parent $scriptDir) -replace "\\", "/"
$wsSetup = Join-Path $ros2Ws "install/local_setup.ps1"
$wsSetup = $wsSetup -replace "\\", "/"

if (-not (Test-Path -LiteralPath $wsSetup)) {
    Write-Host "Chua co $wsSetup - hay chay colcon build trong gps_system truoc." -ForegroundColor Red
    return
}

Write-Host "Dang source workspace: $wsSetup" -ForegroundColor Cyan
. $wsSetup

Write-Host 'OK. Kiem tra: ros2 pkg list (tim gps_visual)' -ForegroundColor Green

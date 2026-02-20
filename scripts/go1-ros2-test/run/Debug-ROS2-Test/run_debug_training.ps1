<#
.SYNOPSIS
    Debug Go1 ROS2 training - short cycle (20 iterations) for quick verification.

.DESCRIPTION
    1) Starts WSL ROS2 command publisher (constant vx=1.0)
    2) Runs Isaac Lab training for only 20 iterations
    3) Monitors output for ROS2 bridge debug messages
    4) Cleans up WSL process on exit
#>

param(
    [string]$Profile = "constant",
    [int]$NumEnvs = 64,
    [int]$MaxIter = 20,
    [double]$Vx = 1.0,
    [double]$Vy = 0.0,
    [double]$Wz = 0.0,
    [int]$Rate = 30,
    [string]$Task = "Isaac-Velocity-Flat-Unitree-Go1-ROS2Cmd-v0",
    [int]$Seed = 42
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Resolve-Path (Join-Path $ScriptDir "..\..\..\.." )).Path
$EffectiveRunName = "debug_ros2_test"

Write-Host "============================================"
Write-Host "  DEBUG Go1 ROS2 Training (Short Cycle)"
Write-Host "============================================"
Write-Host "  Profile   : $Profile"
Write-Host "  NumEnvs   : $NumEnvs"
Write-Host "  MaxIter   : $MaxIter (SHORT FOR DEBUG)"
Write-Host "  Vx/Vy/Wz  : $Vx / $Vy / $Wz"
Write-Host "  Rate      : ${Rate}Hz"
Write-Host "  Task      : $Task"
Write-Host "  Seed      : $Seed"
Write-Host "  ProjectRoot: $ProjectRoot"
Write-Host "============================================"

# Convert Windows path to WSL path properly
$ProjectRootForward = $ProjectRoot -replace '\\', '/'
$WslProjectRoot = (wsl -d Ubuntu-22.04 bash -c "wslpath -u '$ProjectRootForward'").Trim()
Write-Host "  WslProjectRoot: $WslProjectRoot"
$WslRos2Script = "${WslProjectRoot}/scripts/go1-ros2-test/run/Debug-ROS2-Test/run_ros2_cmd_node.sh"
$WslRos2Args = "--profile $Profile --vx $Vx --vy $Vy --wz $Wz --rate $Rate"

$WslProcess = $null

try {
    Write-Host ""
    Write-Host "[Step 1/4] Starting WSL ROS2 command node (background)..."

    # Start WSL ROS2 node in background using wsl command directly
    # The node will run independently and we'll kill it at cleanup
    $WslCmd = "cd '$WslProjectRoot' && bash scripts/go1-ros2-test/run/Debug-ROS2-Test/run_ros2_cmd_node.sh --profile $Profile --vx $Vx --vy $Vy --wz $Wz --rate $Rate"
    Write-Host "  CMD: wsl -d Ubuntu-22.04 bash -c `"$WslCmd`""
    
    # Start in a new window so it doesn't block
    $WslProcess = Start-Process -FilePath "wsl" `
        -ArgumentList "-d", "Ubuntu-22.04", "bash", "-c", $WslCmd `
        -PassThru -WindowStyle Hidden

    Write-Host "  WSL PID: $($WslProcess.Id)"

    Write-Host ""
    Write-Host "[Step 2/4] Waiting 5 seconds for ROS2 stabilization..."
    Start-Sleep -Seconds 5

    $PublisherPid = (wsl -d Ubuntu-22.04 bash -lc "pgrep -f go1_cmd_script_node.py | head -n 1" 2>$null).Trim()
    if ([string]::IsNullOrWhiteSpace($PublisherPid)) {
        Write-Host "[ERROR] WSL ROS2 node is not running (go1_cmd_script_node.py not found)."
        exit 1
    }
    Write-Host "  ROS2 publisher PID (WSL): $PublisherPid"

    Write-Host ""
    Write-Host "[Step 3/4] Starting Isaac Lab DEBUG training (20 iterations)..."

    $CondaHook = "conda shell.powershell hook | Out-String | Invoke-Expression"
    Invoke-Expression $CondaHook
    conda activate env_isaaclab

    $TrainScript = Join-Path $ProjectRoot "scripts\go1-ros2-test\train.py"
    $IsaacLabBat = Join-Path $ProjectRoot "IsaacLab\isaaclab.bat"

    $TrainArgs = @(
        "-p", $TrainScript,
        "--task", $Task,
        "--num_envs", $NumEnvs,
        "--max_iterations", $MaxIter,
        "--run_name", $EffectiveRunName,
        "--headless"
    )

    if ($Seed -ge 0) {
        $TrainArgs += @("--seed", $Seed)
    }

    Write-Host "  CMD: $IsaacLabBat $($TrainArgs -join ' ')"
    Write-Host ""
    Write-Host ">>> WATCH FOR [ROS2 Bridge] DEBUG MESSAGES <<<"
    Write-Host ""
    & $IsaacLabBat @TrainArgs

    Write-Host ""
    Write-Host "[Step 4/4] Debug training completed."
}
finally {
    Write-Host ""
    Write-Host "[Cleanup] Terminating WSL ROS2 node..."

    if ($null -ne $WslProcess -and -not $WslProcess.HasExited) {
        try {
            wsl -d Ubuntu-22.04 bash -c "pkill -f go1_cmd_script_node.py 2>/dev/null || true"
            Start-Sleep -Seconds 1

            if (-not $WslProcess.HasExited) {
                $WslProcess.Kill()
            }
        }
        catch {
            Write-Host "  Warning: Could not cleanly terminate WSL process: $_"
        }
    }

    Write-Host "[Done] Check console output for [ROS2 Bridge] debug messages."
}

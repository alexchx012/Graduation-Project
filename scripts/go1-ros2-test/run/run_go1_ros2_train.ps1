<#
.SYNOPSIS
    One-click orchestration: launch WSL ROS2 command node + Windows Isaac Lab training.

.DESCRIPTION
    1. Starts go1_cmd_script_node.py in WSL Ubuntu-22.04 (background)
    2. Waits 3 seconds for ROS2 stabilization
    3. Activates conda env_isaaclab and runs training via isaaclab.bat
    4. Cleans up WSL process on exit

.PARAMETER Profile
    Command profile: constant, sine, step (default: constant)

.PARAMETER NumEnvs
    Number of simulation environments (default: 1)

.PARAMETER MaxIter
    Maximum training iterations (default: 50)

.PARAMETER Vx
    Forward velocity in m/s (default: 0.5)

.PARAMETER Rate
    ROS2 publish rate in Hz (default: 20)

.EXAMPLE
    .\run_go1_ros2_train.ps1 -Profile sine -MaxIter 100 -Vx 0.3
#>

param(
    [string]$Profile = "constant",
    [int]$NumEnvs = 1,
    [int]$MaxIter = 50,
    [float]$Vx = 0.5,
    [float]$Vy = 0.0,
    [float]$Wz = 0.0,
    [int]$Rate = 20,
    [string]$Task = "Isaac-Velocity-Flat-Unitree-Go1-ROS2Cmd-v0",
    [int]$Seed = -1
)

$ErrorActionPreference = "Stop"

# Resolve project root (two levels up from this script)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Resolve-Path (Join-Path $ScriptDir "..\..\.." )).Path

Write-Host "============================================"
Write-Host "  Go1 ROS2 + Isaac Lab Training Orchestrator"
Write-Host "============================================"
Write-Host "  Profile   : $Profile"
Write-Host "  NumEnvs   : $NumEnvs"
Write-Host "  MaxIter   : $MaxIter"
Write-Host "  Vx/Vy/Wz  : $Vx / $Vy / $Wz"
Write-Host "  Rate      : ${Rate}Hz"
Write-Host "  Task      : $Task"
Write-Host "  Seed      : $Seed"
Write-Host "  Project   : $ProjectRoot"
Write-Host "============================================"

# -- Convert Windows project path to WSL path --
$WslProjectRoot = wsl -d Ubuntu-22.04 wslpath -u "$ProjectRoot"

$WslRos2Script = "${WslProjectRoot}/scripts/go1-ros2-test/run/run_ros2_cmd_node.sh"
$WslRos2Args = "--profile $Profile --vx $Vx --vy $Vy --wz $Wz --rate $Rate"

$WslProcess = $null

try {
    # ========== Step 1: Start WSL ROS2 node (background) ==========
    Write-Host ""
    Write-Host "[Step 1/4] Starting WSL ROS2 command node (background)..."

    $WslCommand = "bash `"$WslRos2Script`" $WslRos2Args"
    Write-Host "  CMD: wsl -d Ubuntu-22.04 bash -lc `"$WslCommand`""

    $WslProcess = Start-Process -FilePath "wsl" `
        -ArgumentList "-d", "Ubuntu-22.04", "bash", "-lc", $WslCommand `
        -PassThru -NoNewWindow

    Write-Host "  WSL PID: $($WslProcess.Id)"

    # ========== Step 2: Wait for ROS2 stabilization ==========
    Write-Host ""
    Write-Host "[Step 2/4] Waiting 3 seconds for ROS2 to stabilize..."
    Start-Sleep -Seconds 3

    if ($WslProcess.HasExited) {
        Write-Host "[ERROR] WSL ROS2 node exited prematurely (exit code: $($WslProcess.ExitCode))"
        exit 1
    }
    Write-Host "  ROS2 node is running."

    # ========== Step 3: Run training ==========
    Write-Host ""
    Write-Host "[Step 3/4] Starting Isaac Lab training..."

    # Activate conda and run training
    $CondaHook = "conda shell.powershell hook | Out-String | Invoke-Expression"
    Invoke-Expression $CondaHook
    conda activate env_isaaclab

    $TrainScript = Join-Path $ProjectRoot "scripts\go1-ros2-test\train.py"
    $IsaacLabBat = Join-Path $ProjectRoot "IsaacLab\isaaclab.bat"

    $TrainArgs = @(
        "-p", $TrainScript,
        "--task", $Task,
        "--num_envs", $NumEnvs,
        "--max_iterations", $MaxIter
    )

    if ($Seed -ge 0) {
        $TrainArgs += @("--seed", $Seed)
    }

    Write-Host "  CMD: $IsaacLabBat $($TrainArgs -join ' ')"

    & $IsaacLabBat @TrainArgs

    Write-Host ""
    Write-Host "[Step 4/4] Training completed successfully."

} finally {
    # ========== Cleanup: Kill WSL ROS2 process ==========
    Write-Host ""
    Write-Host "[Cleanup] Terminating WSL ROS2 node..."

    if ($null -ne $WslProcess -and -not $WslProcess.HasExited) {
        try {
            # Kill the go1_cmd_script_node.py process inside WSL
            wsl -d Ubuntu-22.04 bash -c "pkill -f go1_cmd_script_node.py 2>/dev/null || true"
            Start-Sleep -Seconds 1

            if (-not $WslProcess.HasExited) {
                $WslProcess.Kill()
            }
        } catch {
            Write-Host "  Warning: Could not cleanly terminate WSL process: $_"
        }
    }

    Write-Host "[Done] Check logs/ for training results."
}

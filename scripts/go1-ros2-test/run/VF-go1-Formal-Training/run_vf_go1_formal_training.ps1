<#
.SYNOPSIS
    Formal Go1 ROS2 training orchestration for stable vx=1.0 m/s tracking.

.DESCRIPTION
    1) Starts WSL ROS2 command publisher (constant vx=1.0 by default)
    2) Runs Isaac Lab training for 1500 iterations
    3) Forces W&B logger with project name go1-flat-locomotion
    4) Cleans up WSL process on exit
#>

param(
    [string]$Profile = "constant",
    [int]$NumEnvs = 4096,
    [int]$MaxIter = 1500,
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
$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$EffectiveRunName = "vf_go1_formal_${Timestamp}"

Write-Host "============================================"
Write-Host "  VF Go1 Formal Training (ROS2Cmd)"
Write-Host "============================================"
Write-Host "  Profile   : $Profile"
Write-Host "  NumEnvs   : $NumEnvs"
Write-Host "  MaxIter   : $MaxIter"
Write-Host "  Vx/Vy/Wz  : $Vx / $Vy / $Wz"
Write-Host "  Rate      : ${Rate}Hz"
Write-Host "  Task      : $Task"
Write-Host "  Seed      : $Seed"
Write-Host "  Logger    : wandb"
Write-Host "  Project   : go1-flat-locomotion"
Write-Host "  RunName   : $EffectiveRunName"
Write-Host "============================================"

$WslProjectRoot = wsl -d Ubuntu-22.04 wslpath -u "$ProjectRoot"
$WslRos2Script = "${WslProjectRoot}/scripts/go1-ros2-test/run/VF-go1-Formal-Training/run_ros2_cmd_node.sh"
$WslRos2Args = "--profile $Profile --vx $Vx --vy $Vy --wz $Wz --rate $Rate"

$WslProcess = $null

try {
    Write-Host ""
    Write-Host "[Step 1/4] Starting WSL ROS2 command node (background)..."

    $WslCommand = "bash `"$WslRos2Script`" $WslRos2Args"
    Write-Host "  CMD: wsl -d Ubuntu-22.04 bash -lc `"$WslCommand`""

    $WslProcess = Start-Process -FilePath "wsl" `
        -ArgumentList "-d", "Ubuntu-22.04", "bash", "-lc", $WslCommand `
        -PassThru -NoNewWindow

    Write-Host "  WSL PID: $($WslProcess.Id)"

    Write-Host ""
    Write-Host "[Step 2/4] Waiting 5 seconds for ROS2 stabilization..."
    Start-Sleep -Seconds 5

    if ($WslProcess.HasExited) {
        Write-Host "[ERROR] WSL ROS2 node exited prematurely (exit code: $($WslProcess.ExitCode))"
        exit 1
    }

    Write-Host ""
    Write-Host "[Step 3/4] Starting Isaac Lab formal training..."

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
        "--logger", "wandb",
        "--log_project_name", "go1-flat-locomotion",
        "--run_name", $EffectiveRunName,
        "--headless"
    )

    if ($Seed -ge 0) {
        $TrainArgs += @("--seed", $Seed)
    }

    Write-Host "  CMD: $IsaacLabBat $($TrainArgs -join ' ')"
    & $IsaacLabBat @TrainArgs

    Write-Host ""
    Write-Host "[Step 4/4] Formal training completed."
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

    Write-Host "[Done] Check logs/ and W&B for results."
}

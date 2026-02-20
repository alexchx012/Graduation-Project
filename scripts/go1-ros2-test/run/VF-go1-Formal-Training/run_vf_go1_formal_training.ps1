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
    [int]$Rate = 50,
    [string]$Task = "Isaac-Velocity-Flat-Unitree-Go1-ROS2Cmd-v0",
    [int]$Seed = 42,
    [string]$RunName = "vf_go1_formal",
    [switch]$Resume,
    [string]$LoadRun = ".*",
    [string]$Checkpoint = "model_.*.pt",
    [switch]$DisableRos2TrackingTune,
    [string]$QosReliability = "reliable",
    [string]$QosDurability = "volatile",
    [int]$QosHistoryDepth = 5
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Resolve-Path (Join-Path $ScriptDir "..\..\..\.." )).Path
$EffectiveRunName = $RunName

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
Write-Host "  Resume    : $Resume"
Write-Host "  LoadRun   : $LoadRun"
Write-Host "  Checkpoint: $Checkpoint"
Write-Host "  Tune      : $(-not $DisableRos2TrackingTune)"
Write-Host "  QoS       : $QosReliability / $QosDurability / depth=$QosHistoryDepth"
Write-Host "  Logger    : wandb"
Write-Host "  Project   : go1-flat-locomotion"
Write-Host "  RunName   : $EffectiveRunName"
Write-Host "============================================"

$ProjectRootForward = $ProjectRoot -replace '\\', '/'
$WslProjectRoot = (wsl -d Ubuntu-22.04 bash -c "wslpath -u '$ProjectRootForward'").Trim()
$WslRos2Script = "${WslProjectRoot}/scripts/go1-ros2-test/run/VF-go1-Formal-Training/run_ros2_cmd_node.sh"
$WslRos2Args = "--profile $Profile --vx $Vx --vy $Vy --wz $Wz --rate $Rate --qos_reliability $QosReliability --qos_durability $QosDurability --qos_history_depth $QosHistoryDepth"

$WslProcess = $null

try {
    Write-Host ""
    Write-Host "[Step 1/4] Starting WSL ROS2 command node (background)..."

    $WslCommand = "cd '$WslProjectRoot' && bash '$WslRos2Script' $WslRos2Args"
    Write-Host "  CMD: wsl -d Ubuntu-22.04 bash -c `"$WslCommand`""

    $WslProcess = Start-Process -FilePath "wsl" `
        -ArgumentList "-d", "Ubuntu-22.04", "bash", "-c", $WslCommand `
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

    if ($WslProcess.HasExited) {
        Write-Host "[ERROR] WSL ROS2 node exited prematurely (exit code: $($WslProcess.ExitCode))"
        exit 1
    }

    Write-Host "  ROS2 publisher PID (WSL): $PublisherPid"

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

    if ($Resume) {
        $TrainArgs += @("--resume", "--load_run", $LoadRun, "--checkpoint", $Checkpoint)
    }

    if ($DisableRos2TrackingTune) {
        $TrainArgs += "--disable_ros2_tracking_tune"
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

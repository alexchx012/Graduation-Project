<#
.SYNOPSIS
    Phase 1 — Flat ROS2Cmd baseline reproduction (300 iters).

.DESCRIPTION
    1) Starts WSL ROS2 command publisher (constant vx=1.0, 50Hz)
    2) Trains Isaac-Velocity-Flat-Unitree-Go1-ROS2Cmd-v0 for 300 iterations
    3) Logs to W&B: project=go1-flat-locomotion, run_name=baseline_flat_ros2cmd
    4) Forces --disable_ros2_tracking_tune (baseline reproduction rule)
    5) Cleans up WSL process on exit
#>

param(
    [int]$NumEnvs   = 4096,
    [int]$MaxIter   = 300,
    [int]$Seed      = 42,
    [switch]$Resume,
    [string]$LoadRun    = ".*",
    [string]$Checkpoint = "model_.*.pt"
)

$ErrorActionPreference = "Stop"

$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Resolve-Path (Join-Path $ScriptDir "..\..\..")).Path

$Task       = "Isaac-Velocity-Flat-Unitree-Go1-ROS2Cmd-v0"
$WandbProj  = "go1-flat-locomotion"
$RunName    = "baseline_flat_ros2cmd"

Write-Host "============================================"
Write-Host "  Phase 1 — Flat Baseline (ROS2Cmd)"
Write-Host "============================================"
Write-Host "  Task      : $Task"
Write-Host "  NumEnvs   : $NumEnvs"
Write-Host "  MaxIter   : $MaxIter"
Write-Host "  Seed      : $Seed"
Write-Host "  W&B proj  : $WandbProj"
Write-Host "  RunName   : $RunName"
Write-Host "  Tune      : DISABLED (baseline)"
Write-Host "============================================"

# --- WSL ROS2 publisher ---
$ProjectRootFwd = $ProjectRoot -replace '\\', '/'
$WslProjectRoot = (wsl -d Ubuntu-22.04 bash -c "wslpath -u '$ProjectRootFwd'").Trim()
$WslRos2Script  = "${WslProjectRoot}/scripts/baseline-repro/Phase1-Baseline/run_ros2_cmd.sh"

$WslProcess = $null

try {
    Write-Host ""
    Write-Host "[Step 1/4] Starting WSL ROS2 command node (background)..."

    $WslCommand = "cd '$WslProjectRoot' && bash '$WslRos2Script'"
    $WslProcess = Start-Process -FilePath "wsl" `
        -ArgumentList "-d", "Ubuntu-22.04", "bash", "-c", $WslCommand `
        -PassThru -WindowStyle Hidden

    Write-Host "  WSL PID: $($WslProcess.Id)"

    Write-Host ""
    Write-Host "[Step 2/4] Waiting 5 seconds for ROS2 stabilization..."
    Start-Sleep -Seconds 5

    $PubPid = (wsl -d Ubuntu-22.04 bash -lc "pgrep -f go1_cmd_script_node.py | head -n 1" 2>$null).Trim()
    if ([string]::IsNullOrWhiteSpace($PubPid)) {
        Write-Host "[ERROR] WSL ROS2 node not running."
        exit 1
    }
    if ($WslProcess.HasExited) {
        Write-Host "[ERROR] WSL ROS2 node exited prematurely (code: $($WslProcess.ExitCode))"
        exit 1
    }
    Write-Host "  ROS2 publisher PID (WSL): $PubPid"

    Write-Host ""
    Write-Host "[Step 3/4] Starting Isaac Lab Flat baseline training..."

    Invoke-Expression (conda shell.powershell hook | Out-String)
    conda activate env_isaaclab

    $TrainScript = Join-Path $ProjectRoot "scripts\go1-ros2-test\train.py"

    $TrainArgs = @(
        $TrainScript,
        "--task",           $Task,
        "--num_envs",       $NumEnvs,
        "--max_iterations", $MaxIter,
        "--seed",           $Seed,
        "--logger",         "wandb",
        "--log_project_name", $WandbProj,
        "--run_name",       $RunName,
        "--headless",
        "--disable_ros2_tracking_tune"
    )

    if ($Resume) {
        $TrainArgs += @("--resume", "--load_run", $LoadRun, "--checkpoint", $Checkpoint)
    }

    Write-Host "  CMD: python $($TrainArgs -join ' ')"
    python @TrainArgs

    Write-Host ""
    Write-Host "[Step 4/4] Flat baseline training completed."
}
finally {
    Write-Host ""
    Write-Host "[Cleanup] Terminating WSL ROS2 node..."

    if ($null -ne $WslProcess -and -not $WslProcess.HasExited) {
        try {
            wsl -d Ubuntu-22.04 bash -c "pkill -f go1_cmd_script_node.py 2>/dev/null || true"
            Start-Sleep -Seconds 1
            if (-not $WslProcess.HasExited) { $WslProcess.Kill() }
        } catch {
            Write-Host "  Warning: Could not cleanly terminate WSL process: $_"
        }
    }

    Write-Host "[Done] Check logs/rsl_rl/unitree_go1_flat/ and W&B ($WandbProj) for results."
}


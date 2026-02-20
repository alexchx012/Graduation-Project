<#
.SYNOPSIS
    Two-stage Go1 ROS2 formal training pipeline.

.DESCRIPTION
    Stage-1 trains with a lower forward command for stabilization.
    Stage-2 resumes from stage-1 checkpoint and shifts to vx=1.0 tracking.
#>

param(
    [int]$NumEnvs = 4096,
    [string]$Task = "Isaac-Velocity-Flat-Unitree-Go1-ROS2Cmd-v0",
    [int]$Seed = 42,
    [int]$Rate = 50,

    [double]$Stage1Vx = 0.6,
    [int]$Stage1Iter = 600,
    [string]$Stage1RunName = "vf_go1_stage1_vx06",

    [double]$Stage2Vx = 1.0,
    [int]$Stage2Iter = 900,
    [string]$Stage2RunName = "vf_go1_stage2_vx10",

    [string]$Checkpoint = "model_.*.pt",
    [string]$WandbRunId = "",
    [switch]$DisableRos2TrackingTune
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Resolve-Path (Join-Path $ScriptDir "..\..\..\.." )).Path
$FormalScript = Join-Path $ScriptDir "run_vf_go1_formal_training.ps1"
$RunRoots = @(
    (Join-Path $ProjectRoot "logs\rsl_rl\unitree_go1_flat"),
    (Join-Path $ScriptDir "logs\rsl_rl\unitree_go1_flat")
)

if ([string]::IsNullOrWhiteSpace($WandbRunId)) {
    $WandbRunId = "vfgo1twostage$([guid]::NewGuid().ToString('N').Substring(0, 12))"
}

$PrevWandbRunId = $env:WANDB_RUN_ID
$PrevWandbResume = $env:WANDB_RESUME
$PrevWandbAllowValChange = $env:WANDB_ALLOW_VAL_CHANGE

Write-Host "============================================"
Write-Host "  VF Go1 Two-Stage ROS2 Training"
Write-Host "============================================"
Write-Host "  Stage1: vx=$Stage1Vx, iter=$Stage1Iter, run=$Stage1RunName"
Write-Host "  Stage2: vx=$Stage2Vx, iter=$Stage2Iter, run=$Stage2RunName"
Write-Host "  NumEnvs: $NumEnvs, Rate: $Rate, Seed: $Seed"
Write-Host "  W&B RunID: $WandbRunId (shared across both stages)"
Write-Host "============================================"

if (-not (Test-Path $FormalScript)) {
    throw "Missing formal training script: $FormalScript"
}

$env:WANDB_RUN_ID = $WandbRunId
$env:WANDB_RESUME = "allow"
$env:WANDB_ALLOW_VAL_CHANGE = "true"

try {
    Write-Host ""
    Write-Host "[Stage 1/2] Warm-up training run..."
    & $FormalScript `
        -NumEnvs $NumEnvs `
        -Task $Task `
        -Seed $Seed `
        -Rate $Rate `
        -MaxIter $Stage1Iter `
        -Vx $Stage1Vx `
        -Vy 0.0 `
        -Wz 0.0 `
        -RunName $Stage1RunName `
        -DisableRos2TrackingTune:$DisableRos2TrackingTune

    $Stage1Candidates = @()
    foreach ($root in $RunRoots) {
        if (-not (Test-Path $root)) {
            continue
        }
        $Stage1Candidates += Get-ChildItem -Path $root -Directory |
            Where-Object { $_.Name -like "*_$Stage1RunName" }
    }
    $Stage1Candidates = $Stage1Candidates | Sort-Object LastWriteTime -Descending

    if ($Stage1Candidates.Count -eq 0) {
        throw "Cannot find stage-1 run directory with suffix _$Stage1RunName under: $($RunRoots -join ', ')"
    }

    $Stage1LoadRun = $Stage1Candidates[0].Name
    Write-Host "  Stage-1 resume source: $Stage1LoadRun"

    Write-Host ""
    Write-Host "[Stage 2/2] Resume training at vx=$Stage2Vx ..."
    & $FormalScript `
        -NumEnvs $NumEnvs `
        -Task $Task `
        -Seed $Seed `
        -Rate $Rate `
        -MaxIter $Stage2Iter `
        -Vx $Stage2Vx `
        -Vy 0.0 `
        -Wz 0.0 `
        -RunName $Stage2RunName `
        -Resume `
        -LoadRun $Stage1LoadRun `
        -Checkpoint $Checkpoint `
        -DisableRos2TrackingTune:$DisableRos2TrackingTune

    Write-Host ""
    Write-Host "[Done] Two-stage training finished."
}
finally {
    if ([string]::IsNullOrWhiteSpace($PrevWandbRunId)) {
        Remove-Item Env:WANDB_RUN_ID -ErrorAction SilentlyContinue
    }
    else {
        $env:WANDB_RUN_ID = $PrevWandbRunId
    }

    if ([string]::IsNullOrWhiteSpace($PrevWandbResume)) {
        Remove-Item Env:WANDB_RESUME -ErrorAction SilentlyContinue
    }
    else {
        $env:WANDB_RESUME = $PrevWandbResume
    }

    if ([string]::IsNullOrWhiteSpace($PrevWandbAllowValChange)) {
        Remove-Item Env:WANDB_ALLOW_VAL_CHANGE -ErrorAction SilentlyContinue
    }
    else {
        $env:WANDB_ALLOW_VAL_CHANGE = $PrevWandbAllowValChange
    }
}

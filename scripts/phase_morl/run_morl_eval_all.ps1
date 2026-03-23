<#
.SYNOPSIS
    Batch MORL physical evaluation orchestration with integrated ROS2 publisher management.

.DESCRIPTION
    1) Starts the WSL ROS2 command publisher
    2) Discovers MORL training run directories under logs/rsl_rl/unitree_go1_rough
    3) Runs scripts/phase_morl/run_morl_eval.py for each selected run
    4) Writes summary JSON files to logs/eval/phase_morl
    5) Stops the WSL ROS2 publisher on exit, even on failure

    Default discovery pattern:
    - any run directory whose name contains morl_p<id>_seed<seed>

    Example:
        .\scripts\phase_morl\run_morl_eval_all.ps1
        .\scripts\phase_morl\run_morl_eval_all.ps1 -PolicyIds P1,P2,P10 -Seeds 42,43,44
        .\scripts\phase_morl\run_morl_eval_all.ps1 -SkipExisting
#>

param(
    [string]$Task = "Isaac-Velocity-MORL-Unitree-Go1-ROS2Cmd-Play-v0",
    [int]$NumEnvs = 64,
    [int]$EvalSteps = 3000,
    [int]$WarmupSteps = 300,
    [int]$Seed = 42,
    [string]$PolicyIds = "",
    [string]$Seeds = "",
    [string]$Checkpoint = "model_1499.pt",
    [double]$RecoveryErrThresh = 0.25,
    [int]$RecoveryMinStableSteps = 1,
    [string]$SummaryDir = "logs/eval/phase_morl",
    [string]$Ros2Script = "",
    [switch]$DryRun,
    [switch]$SkipRos2,
    [switch]$SkipExisting
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Resolve-Path (Join-Path $ScriptDir "..\..")).Path
$TrainLogRoot = Join-Path $ProjectRoot "logs\rsl_rl\unitree_go1_rough"
$EvalScript = Join-Path $ProjectRoot "scripts\phase_morl\run_morl_eval.py"
$SummaryRoot = Join-Path $ProjectRoot $SummaryDir

$Ros2Process = $null
$EvaluationResults = New-Object System.Collections.Generic.List[object]

function Write-Section {
    param([string]$Message)
    Write-Host ""
    Write-Host "============================================================"
    Write-Host $Message
    Write-Host "============================================================"
}

function Convert-ToWslPath {
    param([Parameter(Mandatory=$true)][string]$WindowsPath)

    $pathForward = $WindowsPath -replace '\\', '/'
    try {
        $resolved = (wsl -d Ubuntu-22.04 bash -c "wslpath -u '$pathForward'").Trim()
        if (-not [string]::IsNullOrWhiteSpace($resolved)) {
            return $resolved
        }
    }
    catch {
    }

    if ($pathForward.Length -gt 1 -and $pathForward[1] -eq ':') {
        return "/mnt/$($pathForward[0].ToString().ToLower())$($pathForward.Substring(2))"
    }
    return $pathForward
}

function Ensure-CondaEnv {
    Write-Host "[Setup] Activating conda env: env_isaaclab"
    Invoke-Expression (conda shell.powershell hook | Out-String)
    conda activate env_isaaclab
    $condaPrefix = $env:CONDA_PREFIX
    if (-not $condaPrefix -or ($condaPrefix -notmatch 'env_isaaclab')) {
        throw "Failed to activate env_isaaclab. Current CONDA_PREFIX=$condaPrefix"
    }
}

function Get-Ros2ScriptPath {
    if (-not [string]::IsNullOrWhiteSpace($Ros2Script)) {
        return (Resolve-Path $Ros2Script).Path
    }
    return (Join-Path $ProjectRoot "scripts\baseline-repro\Phase1-Baseline\run_ros2_cmd.sh")
}

function Start-Ros2Publisher {
    if ($SkipRos2) {
        Write-Host "[ROS2] SkipRos2 enabled, not starting publisher."
        return
    }

    $ros2ScriptPath = Get-Ros2ScriptPath
    if (-not (Test-Path $ros2ScriptPath)) {
        throw "ROS2 script not found: $ros2ScriptPath"
    }

    $wslProjectRoot = Convert-ToWslPath -WindowsPath $ProjectRoot
    $wslRos2Script = Convert-ToWslPath -WindowsPath $ros2ScriptPath
    $wslCommand = "cd '$wslProjectRoot' && bash '$wslRos2Script'"

    Write-Host "[ROS2] Cleaning stale publishers..."
    wsl -d Ubuntu-22.04 bash -c "pkill -f go1_cmd_script_node.py 2>/dev/null || true" | Out-Null
    Start-Sleep -Seconds 1

    Write-Host "[ROS2] Starting WSL ROS2 publisher..."
    $script:Ros2Process = Start-Process -FilePath "wsl" `
        -ArgumentList "-d", "Ubuntu-22.04", "bash", "-c", $wslCommand `
        -PassThru -WindowStyle Hidden

    Start-Sleep -Seconds 5

    $publisherPid = (wsl -d Ubuntu-22.04 bash -lc "pgrep -f go1_cmd_script_node.py | head -n 1" 2>$null).Trim()
    if ([string]::IsNullOrWhiteSpace($publisherPid)) {
        throw "WSL ROS2 publisher is not running (go1_cmd_script_node.py not found)."
    }
    if ($script:Ros2Process.HasExited) {
        throw "WSL ROS2 publisher exited prematurely (code: $($script:Ros2Process.ExitCode))."
    }

    Write-Host "[ROS2] Publisher running, WSL PID: $publisherPid"
}

function Stop-Ros2Publisher {
    if ($SkipRos2) {
        return
    }

    Write-Host "[Cleanup] Terminating WSL ROS2 publisher..."
    try {
        wsl -d Ubuntu-22.04 bash -c "pkill -f go1_cmd_script_node.py 2>/dev/null || true" | Out-Null
        Start-Sleep -Seconds 1
    }
    catch {
        Write-Warning "Could not pkill go1_cmd_script_node.py in WSL: $_"
    }

    if ($null -ne $script:Ros2Process -and -not $script:Ros2Process.HasExited) {
        try {
            $script:Ros2Process.Kill()
        }
        catch {
            Write-Warning "Could not terminate local WSL host process: $_"
        }
    }
}

function Parse-NameSet {
    param([string]$Raw)
    if ([string]::IsNullOrWhiteSpace($Raw)) {
        return @()
    }
    return $Raw.Split(",") | ForEach-Object { $_.Trim().ToLower() } | Where-Object { $_ -ne "" }
}

function Get-MorlRunDirectories {
    if (-not (Test-Path $TrainLogRoot)) {
        throw "Training log root not found: $TrainLogRoot"
    }

    $policySet = Parse-NameSet -Raw $PolicyIds
    $seedSet = Parse-NameSet -Raw $Seeds

    $runs = Get-ChildItem $TrainLogRoot -Directory | Where-Object {
        $_.Name -match 'morl_p\d+_seed\d+$'
    } | Sort-Object Name

    if ($policySet.Count -gt 0) {
        $runs = $runs | Where-Object {
            $match = [regex]::Match($_.Name, '(morl_p\d+)_seed\d+$')
            if (-not $match.Success) { return $false }

            $fullId = $match.Groups[1].Value.ToLower()
            $shortId = $fullId -replace '^morl_', ''
            ($policySet -contains $fullId) -or ($policySet -contains $shortId)
        }
    }

    if ($seedSet.Count -gt 0) {
        $runs = $runs | Where-Object {
            $match = [regex]::Match($_.Name, 'seed(\d+)$')
            $match.Success -and ($seedSet -contains $match.Groups[1].Value.ToLower())
        }
    }

    return @($runs)
}

function Invoke-MorlEvaluation {
    param(
        [Parameter(Mandatory=$true)][System.IO.DirectoryInfo]$RunDir
    )

    $runName = $RunDir.Name
    $summaryPath = Join-Path $SummaryRoot "$runName.json"

    $modelPath = Join-Path $RunDir.FullName $Checkpoint
    if (-not (Test-Path $modelPath)) {
        Write-Host "[SKIP] Checkpoint not found: $modelPath"
        $script:EvaluationResults.Add([pscustomobject]@{
            RunName = $runName
            Passed = $false
            Skipped = $true
            SummaryJson = "No checkpoint"
        })
        return
    }

    if ($SkipExisting -and (Test-Path $summaryPath)) {
        Write-Host "[SKIP] Summary already exists: $summaryPath"
        $script:EvaluationResults.Add([pscustomobject]@{
            RunName = $runName
            Passed = $true
            Skipped = $true
            SummaryJson = $summaryPath
        })
        return
    }

    Write-Section "[Eval] $runName"
    Write-Host "  RunDir     : $($RunDir.FullName)"
    Write-Host "  SummaryJson: $summaryPath"

    $args = @(
        $EvalScript,
        "--task", $Task,
        "--load_run", $RunDir.FullName,
        "--checkpoint", $Checkpoint,
        "--num_envs", $NumEnvs,
        "--eval_steps", $EvalSteps,
        "--warmup_steps", $WarmupSteps,
        "--seed", $Seed,
        "--recovery_err_thresh", $RecoveryErrThresh,
        "--recovery_min_stable_steps", $RecoveryMinStableSteps,
        "--summary_json", $summaryPath,
        "--headless"
    )
    if ($SkipRos2) {
        $args += "--skip_ros2"
    }

    Write-Host "  CMD: python $($args -join ' ')"
    if ($DryRun) {
        Write-Host "  [DRY-RUN] Skipping execution."
        $script:EvaluationResults.Add([pscustomobject]@{
            RunName = $runName
            Passed = $true
            Skipped = $true
            SummaryJson = $summaryPath
        })
        return
    }

    python @args
    if ($LASTEXITCODE -ne 0) {
        throw "Evaluation failed for $runName (exit code $LASTEXITCODE)."
    }
    if (-not (Test-Path $summaryPath)) {
        throw "Evaluation finished for $runName but summary_json was not created: $summaryPath"
    }

    $script:EvaluationResults.Add([pscustomobject]@{
        RunName = $runName
        Passed = $true
        Skipped = $false
        SummaryJson = $summaryPath
    })
}

try {
    Write-Section "MORL Batch Evaluation"
    Write-Host "  ProjectRoot : $ProjectRoot"
    Write-Host "  TrainLogRoot: $TrainLogRoot"
    Write-Host "  EvalScript  : $EvalScript"
    Write-Host "  SummaryRoot : $SummaryRoot"
    Write-Host "  Task        : $Task"
    Write-Host "  NumEnvs     : $NumEnvs"
    Write-Host "  EvalSteps   : $EvalSteps"
    Write-Host "  WarmupSteps : $WarmupSteps"
    Write-Host "  Seed        : $Seed"
    Write-Host "  PolicyIds   : $PolicyIds"
    Write-Host "  Seeds filter: $Seeds"
    Write-Host "  DryRun      : $DryRun"
    Write-Host "  SkipRos2    : $SkipRos2"
    Write-Host "  SkipExisting: $SkipExisting"

    if (-not (Test-Path $EvalScript)) {
        throw "Evaluation script not found: $EvalScript"
    }

    Ensure-CondaEnv
    New-Item -ItemType Directory -Force -Path $SummaryRoot | Out-Null
    Start-Ros2Publisher

    $runs = Get-MorlRunDirectories
    if ($runs.Count -eq 0) {
        throw "No MORL run directories matched the requested filters."
    }

    Write-Host "[Discover] Found $($runs.Count) MORL run directories."
    foreach ($run in $runs) {
        Write-Host "  - $($run.Name)"
    }

    foreach ($run in $runs) {
        Invoke-MorlEvaluation -RunDir $run
    }

    Write-Section "Evaluation Summary"
    $passedCount = @($EvaluationResults | Where-Object { $_.Passed -and -not $_.Skipped }).Count
    $skippedCount = @($EvaluationResults | Where-Object { $_.Skipped }).Count
    Write-Host "  Passed : $passedCount"
    Write-Host "  Skipped: $skippedCount"
    Write-Host "  Total  : $($EvaluationResults.Count)"
    foreach ($item in $EvaluationResults) {
        $tag = if ($item.Skipped) { "SKIP" } else { "PASS" }
        Write-Host ("  [{0}] {1} -> {2}" -f $tag, $item.RunName, $item.SummaryJson)
    }
}
catch {
    Write-Error $_
    exit 1
}
finally {
    Stop-Ros2Publisher
}

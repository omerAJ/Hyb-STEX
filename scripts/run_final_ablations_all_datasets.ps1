<#
.SYNOPSIS
Runs the finalized Hyb-STEX ablation suite sequentially for NYCBike1,
NYCBike2, and BJTaxi.

.DESCRIPTION
Every dataset gets its own result directories and manifests.  The Python
runners write their progress after each completed seed/variant and verify the
checkpoint before reusing it.  If this script, Python, or the machine stops,
run the same command again: incomplete result rows are discarded by the
runners and the first missing seed/variant is trained again.

NYCTaxi is intentionally excluded because its finalized ablations already
exist.  Do not use -Restart unless you deliberately want to discard a whole
dataset's saved result set.

.EXAMPLE
  .\scripts\run_final_ablations_all_datasets.ps1

.EXAMPLE
  .\scripts\run_final_ablations_all_datasets.ps1 -DataRoot E:\datasets\preprocessed_data
#>
[CmdletBinding()]
param(
    # Directory containing NYCBike1, NYCBike2, and BJTaxi subdirectories.
    [string]$DataRoot,

    # A separate root prevents manifest conflicts and makes per-dataset resume safe.
    [string]$OutputRoot,

    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda",

    # Deliberately starts fresh result directories. Never needed for normal resume.
    [switch]$Restart,

    # Omit threshold sensitivity or figures only when intentionally reducing scope.
    [switch]$SkipThresholdSensitivity,
    [switch]$SkipFigures
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
if ([string]::IsNullOrWhiteSpace($DataRoot)) {
    $DataRoot = Join-Path $RepoRoot "preprocessed_data"
}
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $RepoRoot "final_all_dataset_ablations"
}

function Invoke-PythonRunner {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,
        [Parameter(Mandatory = $true)]
        [string]$Label
    )

    Write-Host "`n==== $Label ====" -ForegroundColor Cyan
    & conda run --no-capture-output -n sds-venv python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE. Re-run this script to resume."
    }
}

function Test-FigureSetComplete {
    param([Parameter(Mandatory = $true)][string]$Directory)
    $expected = @(
        "severity_metrics_per_seed.csv",
        "mae_by_test_flow_percentile.png",
        "underprediction_by_test_flow_percentile.png",
        "maximum_observed_event_windows.png"
    )
    return ($expected | ForEach-Object { Test-Path -LiteralPath (Join-Path $Directory $_) }) -notcontains $false
}

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "Conda is not available on PATH. Initialize Conda, then run this script again."
}
& conda run --no-capture-output -n sds-venv python -c "import sys; print(sys.executable)"
if ($LASTEXITCODE -ne 0) {
    throw "Conda environment 'sds-venv' is not available or Python could not start."
}
if (-not (Test-Path -LiteralPath $DataRoot -PathType Container)) {
    throw "DataRoot does not exist: $DataRoot"
}

$DataRoot = (Resolve-Path -LiteralPath $DataRoot).Path
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$OutputRoot = (Resolve-Path -LiteralPath $OutputRoot).Path

# This order is intentional. Finish all stages for one dataset before moving on.
$Datasets = @(
    @{ Name = "NYCBike1"; Config = "configs/NYCBike1.yaml" },
    @{ Name = "NYCBike2"; Config = "configs/NYCBike2.yaml" },
    @{ Name = "BJTaxi";   Config = "configs/BJTaxi.yaml" }
)
$Seeds = @("1", "2", "3")
$EventWeight = "0.25"

Push-Location $RepoRoot
try {
    foreach ($Dataset in $Datasets) {
        $Name = $Dataset.Name
        $Config = $Dataset.Config
        $DatasetData = Join-Path $DataRoot $Name
        $GraphFile = Join-Path $DatasetData "adj_mx.npz"
        if (-not (Test-Path -LiteralPath $DatasetData -PathType Container)) {
            throw "Missing dataset directory: $DatasetData"
        }
        foreach ($FileName in @("train.npz", "val.npz", "test.npz", "adj_mx.npz")) {
            if (-not (Test-Path -LiteralPath (Join-Path $DatasetData $FileName) -PathType Leaf)) {
                throw "Missing $FileName for $Name in $DatasetData"
            }
        }

        $DatasetRoot = Join-Path $OutputRoot $Name
        $PaperResults = Join-Path $DatasetRoot "paper_rescue"
        $ScheduleResults = Join-Path $DatasetRoot "residual_schedule"
        $PhaseResults = Join-Path $DatasetRoot "phase_training"
        $ThresholdResults = Join-Path $DatasetRoot "threshold_sensitivity"
        $FigureResults = Join-Path $DatasetRoot "tail_diagnostics"
        New-Item -ItemType Directory -Force -Path $DatasetRoot | Out-Null

        $Common = @(
            "--config-filenames", $Config,
            "--data-dir", $DataRoot,
            "--device", $Device,
            "--seeds"
        ) + $Seeds + @(
            "--event-loss-weight", $EventWeight
        )
        $RestartArgument = if ($Restart) { @("--restart") } else { @() }

        # Sources required by the later schedule and phase-training runners.
        # F/G are intentionally excluded: they are exploratory dual-residual
        # controls, not part of the paper's finalized argument.
        Invoke-PythonRunner -Label "${Name}: architecture and objective controls" -Arguments (
            @("scripts/run_paper_rescue_ablation.py") + $Common + @(
                "--output-dir", $PaperResults,
                "--variants",
                "A_base_mae",
                "B_base_event_weighted",
                "C_ungated_residual_mae",
                "D_ungated_residual_event_weighted",
                "E_classifier_gated_residual_mae"
            ) + $RestartArgument
        )

        Invoke-PythonRunner -Label "${Name}: frozen-versus-joint schedule controls" -Arguments (
            @("scripts/run_residual_training_schedule_ablation.py") + $Common + @(
                "--source-results", $PaperResults,
                "--output-dir", $ScheduleResults
            ) + $RestartArgument
        )

        Invoke-PythonRunner -Label "${Name}: phase-training ablation table" -Arguments (
            @("scripts/run_event_weighted_phase_training_ablation.py") + $Common + @(
                "--paper-results", $PaperResults,
                "--schedule-results", $ScheduleResults,
                "--output-dir", $PhaseResults
            ) + $RestartArgument
        )

        if (-not $SkipThresholdSensitivity) {
            Invoke-PythonRunner -Label "${Name}: 90th-versus-95th threshold sensitivity" -Arguments (
                @("scripts/run_event_threshold_ablation.py") + $Common + @(
                    "--source-results", $ScheduleResults,
                    "--output-dir", $ThresholdResults,
                    "--event-percentiles", "90", "95"
                ) + $RestartArgument
            )
        }

        if (-not $SkipFigures) {
            if (Test-FigureSetComplete -Directory $FigureResults) {
                Write-Host "`n[resume] ${Name}: tail diagnostics already complete" -ForegroundColor Green
            }
            else {
                Invoke-PythonRunner -Label "${Name}: tail diagnostics" -Arguments @(
                    "scripts/plot_extreme_underprediction_evidence.py",
                    "--config-filename", $Config,
                    "--data-dir", $DataRoot,
                    "--graph-file", $GraphFile,
                    "--paper-results", $PaperResults,
                    "--schedule-results", $ScheduleResults,
                    "--output-dir", $FigureResults
                )
            }
        }

        Write-Host "`n[complete] $Name" -ForegroundColor Green
    }
}
finally {
    Pop-Location
}

Write-Host "`nAll requested datasets completed. Results: $OutputRoot" -ForegroundColor Green

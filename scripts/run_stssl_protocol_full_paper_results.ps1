<# Run all four datasets sequentially under the ST-SSL benchmark protocol. #>
[CmdletBinding()]
param(
    [string]$CondaEnv = "sds-test",
    [string]$DataRoot,
    [string]$OutputRoot,
    [ValidateSet("cuda", "cpu")][string]$Device = "cuda",
    [switch]$Restart
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
if ([string]::IsNullOrWhiteSpace($DataRoot)) { $DataRoot = Join-Path $RepoRoot "preprocessed_data" }
if ([string]::IsNullOrWhiteSpace($OutputRoot)) { $OutputRoot = Join-Path $RepoRoot "stssl_protocol_full_paper_results" }
function Invoke-Runner {
    param([string]$Label, [string[]]$Arguments)
    Write-Host "RUN: $Label" -ForegroundColor Cyan
    & conda run --no-capture-output -n $CondaEnv python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Label failed with exit code $LASTEXITCODE. Re-run this command to resume." }
}
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) { throw "Conda is not available on PATH." }
& conda run --no-capture-output -n $CondaEnv python -c "import sys; print(sys.executable)"
if ($LASTEXITCODE -ne 0) { throw "Conda environment '$CondaEnv' could not start Python." }
if (-not (Test-Path -LiteralPath $DataRoot -PathType Container)) { throw "DataRoot does not exist: $DataRoot" }
$DataRoot = (Resolve-Path -LiteralPath $DataRoot).Path
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$OutputRoot = (Resolve-Path -LiteralPath $OutputRoot).Path
$Datasets = @(
    @{ Name = "NYCBike1"; Config = "configs/NYCBike1.yaml" },
    @{ Name = "NYCBike2"; Config = "configs/NYCBike2.yaml" },
    @{ Name = "NYCTaxi"; Config = "configs/NYCTaxi.yaml" },
    @{ Name = "BJTaxi"; Config = "configs/BJTaxi.yaml" }
)
$Seeds = @("1", "2", "3")
$RestartArgument = if ($Restart) { @("--restart") } else { @() }
Push-Location $RepoRoot
try {
    foreach ($Dataset in $Datasets) {
        $Name = $Dataset.Name; $DatasetDir = Join-Path $DataRoot $Name; $GraphFile = Join-Path $DatasetDir "adj_mx.npz"
        foreach ($FileName in @("train.npz", "val.npz", "test.npz", "adj_mx.npz")) {
            if (-not (Test-Path -LiteralPath (Join-Path $DatasetDir $FileName) -PathType Leaf)) { throw "Missing $FileName for $Name at $DatasetDir" }
        }
        $DatasetRoot = Join-Path $OutputRoot $Name; $Paper = Join-Path $DatasetRoot "paper_rescue"; $Schedule = Join-Path $DatasetRoot "residual_schedule"; $Phase = Join-Path $DatasetRoot "phase_training"
        $Common = @("--config-filenames", $Dataset.Config, "--data-dir", $DataRoot, "--graph-file", $GraphFile, "--device", $Device, "--seeds") + $Seeds + @("--scaler-fit", "train_val", "--bias-param-scope", "head_only", "--event-mask-protocol", "train_all_node_flow_p90_valid_v2", "--event-label-source", "file_verified", "--event-loss-weight", "0.25")
        Invoke-Runner "${Name}: architecture/objective ablations" (@("scripts/run_paper_rescue_ablation.py") + $Common + @("--output-dir", $Paper, "--variants", "A_base_mae", "B_base_event_weighted", "C_ungated_residual_mae", "D_ungated_residual_event_weighted", "E_classifier_gated_residual_mae") + $RestartArgument)
        Invoke-Runner "${Name}: frozen/joint residual ablations" (@("scripts/run_residual_training_schedule_ablation.py") + $Common + @("--source-results", $Paper, "--output-dir", $Schedule) + $RestartArgument)
        Invoke-Runner "${Name}: final phase-training ablation table" (@("scripts/run_event_weighted_phase_training_ablation.py") + $Common + @("--paper-results", $Paper, "--schedule-results", $Schedule, "--output-dir", $Phase) + $RestartArgument)
    }
} finally { Pop-Location }
Write-Host "Complete. Per-dataset main and full ablation summaries: $OutputRoot" -ForegroundColor Green

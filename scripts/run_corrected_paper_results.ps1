<#
.SYNOPSIS
Launches the corrected five-model paper evaluation across all four datasets.

.DESCRIPTION
This is a thin, resumable wrapper around run_corrected_paper_results.py. It
selects either an explicit Python executable or a Conda environment, verifies
that NumPy, PyTorch, and PyYAML import, and forwards all runner arguments as an
array. The import preflight intentionally does not probe CUDA availability.

Running the same full command again resumes from the correction runner's
manifest and per-unit records. Use -Restart only when intentionally discarding
compatible artifacts inside the selected correction output root.

.EXAMPLE
  .\scripts\run_corrected_paper_results.ps1 `
    -PythonExe C:\path\to\python.exe `
    -DataRoot .\preprocessed_data `
    -Device cuda `
    -DryRun

.EXAMPLE
  .\scripts\run_corrected_paper_results.ps1 `
    -CondaEnv sds-test `
    -DataRoot .\preprocessed_data `
    -OutputRoot .\corrected_valid_target_p90_results `
    -Device cuda
#>
[CmdletBinding()]
param(
    # Directory containing NYCTaxi, NYCBike1, NYCBike2, and BJTaxi.
    [string]$DataRoot,

    # Fresh protocol-v2 output root. Relative paths are resolved from the repository.
    [string]$OutputRoot,

    [ValidatePattern("^(cpu|cuda(?::[0-9]+)?)$")]
    [string]$Device = "cuda",

    # Use an explicit interpreter. Mutually exclusive with -CondaEnv.
    [string]$PythonExe,

    # Or run Python through `conda run -n <environment>`.
    [string]$CondaEnv,

    # Validate the complete plan without creating the output root.
    [switch]$DryRun,

    # Use seed 1 and one train/evaluation batch in a smoke-specific output.
    [switch]$SmokeTest,

    # Deliberately restart only the selected correction output root.
    [switch]$Restart,

    [ValidateNotNullOrEmpty()]
    [int[]]$Seeds = @(1, 2, 3)
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$SeedsWereExplicit = $PSBoundParameters.ContainsKey("Seeds")
$RepositoryRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$RunnerPath = Join-Path $PSScriptRoot "run_corrected_paper_results.py"

if (-not (Test-Path -LiteralPath $RunnerPath -PathType Leaf)) {
    throw "Corrected-results Python runner is missing: $RunnerPath"
}

function Resolve-RepositoryPath {
    param(
        [AllowEmptyString()]
        [string]$Value,
        [Parameter(Mandatory = $true)]
        [string]$DefaultRelativePath,
        [switch]$RequireDirectory
    )

    $Candidate = $Value
    if ([string]::IsNullOrWhiteSpace($Candidate)) {
        $Candidate = Join-Path $RepositoryRoot $DefaultRelativePath
    }
    elseif (-not [System.IO.Path]::IsPathRooted($Candidate)) {
        $Candidate = Join-Path $RepositoryRoot $Candidate
    }

    $FullPath = [System.IO.Path]::GetFullPath($Candidate)
    if ($RequireDirectory) {
        if (-not (Test-Path -LiteralPath $FullPath -PathType Container)) {
            throw "Required directory does not exist: $FullPath"
        }
        return (Resolve-Path -LiteralPath $FullPath).Path
    }
    if ((Test-Path -LiteralPath $FullPath) -and
        -not (Test-Path -LiteralPath $FullPath -PathType Container)) {
        throw "OutputRoot exists but is not a directory: $FullPath"
    }
    return $FullPath
}

$ResolvedDataRoot = Resolve-RepositoryPath `
    -Value $DataRoot `
    -DefaultRelativePath "preprocessed_data" `
    -RequireDirectory
$ResolvedOutputRoot = Resolve-RepositoryPath `
    -Value $OutputRoot `
    -DefaultRelativePath "corrected_valid_target_p90_results"

if ($DryRun -and $Restart) {
    throw "-Restart cannot be combined with -DryRun; dry-run is strictly non-mutating."
}
if ($null -eq $Seeds -or $Seeds.Count -eq 0) {
    throw "-Seeds must contain at least one positive integer."
}
if (@($Seeds | Where-Object { $_ -lt 1 }).Count -gt 0) {
    throw "-Seeds must contain only positive integers."
}
if (@($Seeds | Sort-Object -Unique).Count -ne $Seeds.Count) {
    throw "-Seeds must not contain duplicates."
}

$EffectiveSeeds = @($Seeds)
if ($SmokeTest) {
    if ($SeedsWereExplicit -and ($Seeds.Count -ne 1 -or $Seeds[0] -ne 1)) {
        throw "-SmokeTest uses exactly seed 1; omit -Seeds or pass -Seeds 1."
    }
    $EffectiveSeeds = @(1)
}

if (-not [string]::IsNullOrWhiteSpace($PythonExe) -and
    -not [string]::IsNullOrWhiteSpace($CondaEnv)) {
    throw "Choose either -PythonExe or -CondaEnv, not both."
}

$PythonCommand = $null
$PythonPrefixArguments = @()
$PythonLauncherDescription = $null

if (-not [string]::IsNullOrWhiteSpace($CondaEnv)) {
    if ($CondaEnv.Trim().Length -eq 0) {
        throw "-CondaEnv cannot be empty."
    }
    if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
        throw "Conda is not available on PATH; use -PythonExe or initialize Conda."
    }
    $PythonCommand = "conda"
    $PythonPrefixArguments = @(
        "run", "--no-capture-output", "-n", $CondaEnv, "python"
    )
    $PythonLauncherDescription = "conda:$CondaEnv"
}
else {
    if ([string]::IsNullOrWhiteSpace($PythonExe)) {
        $PythonExe = "python"
    }
    if (-not (Get-Command $PythonExe -ErrorAction SilentlyContinue)) {
        throw "Python executable is unavailable: $PythonExe"
    }
    $PythonCommand = $PythonExe
    $PythonLauncherDescription = "python:$PythonExe"
}

function Invoke-CheckedPython {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$PythonArguments,
        [Parameter(Mandatory = $true)]
        [string]$Label
    )

    $InvocationArguments = @($PythonPrefixArguments) + @($PythonArguments)
    Write-Host "`n[$Label]" -ForegroundColor Cyan
    & $PythonCommand @InvocationArguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE."
    }
}

# Import-only by design. CUDA availability/initialization belongs to the actual
# training runtime and must not be added to this preflight.
# Conda 23.x rejects embedded newlines in arguments passed to `conda run`.
# Keep this import-only probe on one physical line so it is safe for both
# `conda run ... python -c` and direct Python invocation.
$PreflightCode = "import json, platform, sys, numpy, torch, yaml; print(json.dumps({'executable': sys.executable, 'python': platform.python_version(), 'numpy': numpy.__version__, 'torch': torch.__version__, 'pyyaml': yaml.__version__}, sort_keys=True, separators=(',', ':')))"
$PreflightInvocationArguments = @($PythonPrefixArguments) + @("-c", $PreflightCode)
Write-Host "`n[environment preflight (imports only; CUDA not probed)]" -ForegroundColor Cyan
$PreflightOutput = @(& $PythonCommand @PreflightInvocationArguments)
$PreflightExitCode = $LASTEXITCODE
$PreflightOutput | ForEach-Object { Write-Host $_ }
if ($PreflightExitCode -ne 0) {
    throw "Environment preflight failed with exit code $PreflightExitCode."
}
$PreflightJson = @($PreflightOutput | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })[-1]
try {
    $PreflightMetadata = $PreflightJson | ConvertFrom-Json
}
catch {
    throw "Environment preflight did not return valid JSON metadata: $PreflightJson"
}
foreach ($RequiredField in @("executable", "python", "numpy", "torch", "pyyaml")) {
    if ([string]::IsNullOrWhiteSpace([string]$PreflightMetadata.$RequiredField)) {
        throw "Environment preflight metadata is missing '$RequiredField'."
    }
}
$PythonEnvironmentMetadata = [ordered]@{
    launcher = $PythonLauncherDescription
    executable = [string]$PreflightMetadata.executable
    python = [string]$PreflightMetadata.python
    numpy = [string]$PreflightMetadata.numpy
    torch = [string]$PreflightMetadata.torch
    pyyaml = [string]$PreflightMetadata.pyyaml
} | ConvertTo-Json -Compress

$RunnerArguments = @(
    "-m", "scripts.run_corrected_paper_results",
    "--data-dir", $ResolvedDataRoot,
    "--output-dir", $ResolvedOutputRoot,
    "--device", $Device,
    "--seeds"
) + @($EffectiveSeeds | ForEach-Object { [string]$_ }) + @(
    "--event-loss-weight", "0.25",
    "--python-environment", $PythonEnvironmentMetadata
)

if ($DryRun) {
    $RunnerArguments += "--dry-run"
}
if ($SmokeTest) {
    $RunnerArguments += @(
        "--smoke-test",
        "--max-train-batches", "1",
        "--max-eval-batches", "1"
    )
}
if ($Restart) {
    $RunnerArguments += "--restart"
}

$OutputExistedBefore = Test-Path -LiteralPath $ResolvedOutputRoot
Write-Host "`nRepository: $RepositoryRoot"
Write-Host "Data root:  $ResolvedDataRoot"
Write-Host "Output root: $ResolvedOutputRoot"
Write-Host "Device:     $Device"
Write-Host "Seeds:      $($EffectiveSeeds -join ', ')"
Push-Location $RepositoryRoot
try {
    Invoke-CheckedPython -Label "corrected paper-results runner" -PythonArguments $RunnerArguments
}
finally {
    Pop-Location
}

if ($DryRun -and -not $OutputExistedBefore -and
    (Test-Path -LiteralPath $ResolvedOutputRoot)) {
    throw "Dry-run unexpectedly created the output root: $ResolvedOutputRoot"
}

if ($DryRun) {
    Write-Host "`nDry-run complete; no correction output was created." -ForegroundColor Green
}
elseif ($SmokeTest) {
    Write-Host "`nSmoke run complete. It is not eligible for full-run resume." -ForegroundColor Green
}
else {
    Write-Host "`nRequested corrected paper run complete: $ResolvedOutputRoot" -ForegroundColor Green
}

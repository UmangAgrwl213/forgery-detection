param([int]$WaitForPid = 0)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if ($WaitForPid -gt 0) {
    $previous = Get-Process -Id $WaitForPid -ErrorAction SilentlyContinue
    if ($previous) { $previous.WaitForExit() }
}

$env:NO_ALBUMENTATIONS_UPDATE = '1'
$env:MPLBACKEND = 'Agg'
$python = '.\.venv\Scripts\python.exe'
$manifest = 'outputs/benchmarks/casia-source-v2.json'
$run = 'outputs/casia-source-v2-' + (Get-Date -Format yyyyMMdd-HHmmss)
Write-Output "Benchmark artifacts: $run"
& $python -u main.py --epochs 30 --seed 42 --split-manifest $manifest --output-dir $run
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python -u evaluate_casia.py --split-manifest $manifest --run-dir $run
exit $LASTEXITCODE

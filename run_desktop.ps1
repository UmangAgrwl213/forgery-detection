param(
    [string]$Weights = '',
    [string]$AiWeights = '',
    [ValidateSet('cpu', 'cuda', 'auto')][string]$Device = 'cpu'
)

$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw 'Create the project .venv and install requirements before launching the app.'
}
if (-not $Weights) {
    $calibrated = Join-Path $PSScriptRoot 'outputs\casia-source-v2-20261127-183558\best_model.pth'
    $calibration = Join-Path $PSScriptRoot 'outputs\casia-source-v2-20261127-183558\heldout_eval.json'
    $development = Join-Path $PSScriptRoot 'outputs\v2-30ep-20261127-181840\best_model.pth'
    if ((Test-Path -LiteralPath $calibrated -PathType Leaf) -and (Test-Path -LiteralPath $calibration -PathType Leaf)) {
        $Weights = $calibrated
    }
    elseif (Test-Path -LiteralPath $development -PathType Leaf) { $Weights = $development }
    else { $Weights = Join-Path $PSScriptRoot 'outputs\best_model.pth' }
}
$Weights = (Resolve-Path -LiteralPath $Weights).Path
$arguments = @('desktop_app.py', '--weights', $Weights, '--device', $Device)
if ($AiWeights) { $arguments += @('--ai-weights', (Resolve-Path -LiteralPath $AiWeights).Path) }
Push-Location -LiteralPath $PSScriptRoot
try { & $python @arguments; exit $LASTEXITCODE }
finally { Pop-Location }

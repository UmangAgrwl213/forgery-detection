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
    $development = Join-Path $PSScriptRoot 'outputs\v2-30ep-20261127-181840\best_model.pth'
    if (Test-Path -LiteralPath $development -PathType Leaf) { $Weights = $development }
    else { $Weights = Join-Path $PSScriptRoot 'outputs\best_model.pth' }
}
$Weights = (Resolve-Path -LiteralPath $Weights).Path
$arguments = @('desktop_app.py', '--weights', $Weights, '--device', $Device)
if ($AiWeights) { $arguments += @('--ai-weights', (Resolve-Path -LiteralPath $AiWeights).Path) }
Push-Location -LiteralPath $PSScriptRoot
try { & $python @arguments; exit $LASTEXITCODE }
finally { Pop-Location }

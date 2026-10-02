$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

git add -A
if ($LASTEXITCODE -ne 0) {
    throw 'git add failed.'
}

git diff --cached --quiet
$diffExitCode = $LASTEXITCODE
if ($diffExitCode -eq 0) {
    Write-Output 'No changes to commit.'
    exit 0
}
if ($diffExitCode -ne 1) {
    throw "git diff failed with exit code $diffExitCode."
}

$commitDate = Get-Date -Format 'yyyy-MM-dd'
git commit -m "Daily CUDA update $commitDate"
if ($LASTEXITCODE -ne 0) {
    throw 'git commit failed.'
}

git push origin main
if ($LASTEXITCODE -ne 0) {
    throw 'git push failed.'
}
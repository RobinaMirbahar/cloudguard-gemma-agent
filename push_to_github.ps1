# ============================================================================
# One-Click GitHub Repository Creator & Pusher for CloudGuard Gemma Agent
# ============================================================================
$env:PATH = "$HOME\.local\mingit\cmd;C:\Program Files\GitHub CLI;" + $env:PATH

Write-Host "1. Checking GitHub CLI Authentication..." -ForegroundColor Cyan
gh auth status
if ($LASTEXITCODE -ne 0) {
    Write-Host "Launching GitHub browser login..." -ForegroundColor Yellow
    gh auth login --hostname github.com --git-protocol https --web
}

Write-Host "`n2. Creating public GitHub repository 'cloudguard-gemma-agent' and pushing main branch..." -ForegroundColor Cyan
gh repo create cloudguard-gemma-agent --public --source=. --remote=origin --push

Write-Host "`nDone! Your repository is live on GitHub." -ForegroundColor Green

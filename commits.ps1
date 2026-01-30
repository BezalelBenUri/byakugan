# Split-commit script (Backend only)
# Commits each changed file under this Backend folder as a separate commit
# using the same commit message, then pushes once at the end.
#
# Usage examples (run from repo root or any folder):
#   powershell -ExecutionPolicy Bypass -File Backend\split-commit.ps1 -Message "Backend: API fixes"
#   ./Backend/split-commit.ps1 -Message "Backend: listings"
#   ./Backend/split-commit.ps1 -Message "WIP" -NoPush

param(
  [string]$Message = "",
  [switch]$NoPush
)

function Ensure-GitRepo {
  if (-not (git rev-parse --git-dir 2>$null)) { throw "Not a git repository. Run from repo root or make sure git is available." }
}

function Collect-ScopedFiles([string]$ScopePath) {
  $files = @()
  # Staged (cached) changes
  $staged = git diff --name-only --cached -- "$ScopePath" | Out-String
  if ($staged) { $files += ($staged -split "`r?`n" | Where-Object { $_ -ne '' }) }
  # Modified (unstaged) tracked
  $mods = git ls-files -m -- "$ScopePath" | Out-String
  if ($mods) { $files += ($mods -split "`r?`n" | Where-Object { $_ -ne '' }) }
  # Untracked
  $untracked = git ls-files -o --exclude-standard -- "$ScopePath" | Out-String
  if ($untracked) { $files += ($untracked -split "`r?`n" | Where-Object { $_ -ne '' }) }
  # Deleted (unstaged)
  $deleted = git ls-files -d -- "$ScopePath" | Out-String
  if ($deleted) { $files += ($deleted -split "`r?`n" | Where-Object { $_ -ne '' }) }
  # De-dup preserve order
  $seen = @{}
  $unique = @()
  foreach ($f in $files) { if (-not $seen.ContainsKey($f)) { $seen[$f] = $true; $unique += $f } }
  return $unique
}

try { Ensure-GitRepo } catch { Write-Error $_; exit 1 }

if ([string]::IsNullOrWhiteSpace($Message)) {
  $Message = Read-Host -Prompt "Enter commit message (Backend)"
  if ([string]::IsNullOrWhiteSpace($Message)) { Write-Error "Commit message is required."; exit 1 }
}

# Determine repo root and scope path (supports running inside Backend repo)
$repoRoot = git rev-parse --show-toplevel
if (-not $repoRoot) { Write-Error "Could not determine repo root"; exit 1 }

Set-Location $repoRoot
$candidate = Join-Path $repoRoot "Backend"
if (Test-Path -LiteralPath $candidate) {
  $scopePath = (Resolve-Path -LiteralPath $candidate)
} else {
  # If repo root is already the Backend dir (e.g., separate repo), use root
  $scopePath = $repoRoot
}

$branch = git rev-parse --abbrev-ref HEAD
Write-Host "[split-commit] Branch:  $branch"
Write-Host "[split-commit] Scope:   $scopePath"
Write-Host "[split-commit] Message: $Message"

$files = Collect-ScopedFiles -ScopePath $scopePath
if ($files.Count -eq 0) { Write-Host "[split-commit] No Backend changes to commit."; exit 0 }

Write-Host "[split-commit] Files: $($files.Count)"; foreach ($f in $files) { Write-Host " - $f" }

foreach ($f in $files) {
  Write-Host "[split-commit] Committing: $f"
  if (Test-Path -- $f) { git add -- "$f" | Out-Null } else { git rm --cached -f -- "$f" 2>$null | Out-Null; git rm -f -- "$f" 2>$null | Out-Null }
  $hasChanges = $(git diff --cached --quiet -- "$f"; if ($LASTEXITCODE -eq 0) { $false } else { $true })
  if (-not $hasChanges) { Write-Host "  - No staged changes, skipping"; continue }
  git commit -m "$Message" -- "$f" | Out-Null
}

if (-not $NoPush) { Write-Host "[split-commit] Pushing to origin/$branch"; git push -u origin "$branch" } else { Write-Host "[split-commit] Skipping push (-NoPush)" }

Write-Host "[split-commit] Done."
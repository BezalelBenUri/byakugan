# Split-commit script for the Byakugan repo.
# Commits each changed file as a separate commit using the same commit message,
# then pushes first to your personal repo (source of truth) and next to the
# company repo.
#
# Default remote policy:
#   - `origin`  -> your GitHub repo (source of truth)
#   - `company` -> Observatory Analytics repo
#
# Usage examples:
#   powershell -ExecutionPolicy Bypass -File .\commits.ps1 -Message "Fix viewer orientation"
#   .\commits.ps1 -Message "WIP" -NoPush
#   .\commits.ps1 -Message "Measurement fixes" -PrimaryRemote origin -SecondaryRemote company

param(
  [string]$Message = "",
  [switch]$NoPush,
  [string]$PrimaryRemote = "origin",
  [string]$SecondaryRemote = "company",
  [string]$SecondaryRemoteUrl = "https://github.com/ObservatoryAnalytics/byakugan.git"
)

function Ensure-GitRepo {
  if (-not (git rev-parse --git-dir 2>$null)) {
    throw "Not a git repository. Run from the repo root or make sure git is available."
  }
}

function Collect-RepoFiles([string]$ScopePath) {
  $files = @()

  $staged = git diff --name-only --cached -- "$ScopePath" | Out-String
  if ($staged) { $files += ($staged -split "`r?`n" | Where-Object { $_ -ne '' }) }

  $mods = git ls-files -m -- "$ScopePath" | Out-String
  if ($mods) { $files += ($mods -split "`r?`n" | Where-Object { $_ -ne '' }) }

  $untracked = git ls-files -o --exclude-standard -- "$ScopePath" | Out-String
  if ($untracked) { $files += ($untracked -split "`r?`n" | Where-Object { $_ -ne '' }) }

  $deleted = git ls-files -d -- "$ScopePath" | Out-String
  if ($deleted) { $files += ($deleted -split "`r?`n" | Where-Object { $_ -ne '' }) }

  $seen = @{}
  $unique = @()
  foreach ($f in $files) {
    if (-not $seen.ContainsKey($f)) {
      $seen[$f] = $true
      $unique += $f
    }
  }
  return $unique
}

function Ensure-Remote([string]$Name, [string]$Url) {
  $existingUrl = git remote get-url $Name 2>$null
  if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace($existingUrl)) {
    return
  }
  if ([string]::IsNullOrWhiteSpace($Url)) {
    throw "Remote '$Name' does not exist and no URL was provided."
  }
  Write-Host "[split-commit] Adding remote '$Name' -> $Url"
  git remote add $Name $Url | Out-Null
}

function Push-Branch([string]$Remote, [string]$Branch) {
  Write-Host "[split-commit] Pushing to $Remote/$Branch"
  git push -u $Remote "$Branch"
  if ($LASTEXITCODE -ne 0) {
    throw "Push to $Remote/$Branch failed."
  }
}

try { Ensure-GitRepo } catch { Write-Error $_; exit 1 }

if ([string]::IsNullOrWhiteSpace($Message)) {
  $Message = Read-Host -Prompt "Enter commit message"
  if ([string]::IsNullOrWhiteSpace($Message)) {
    Write-Error "Commit message is required."
    exit 1
  }
}

$repoRoot = git rev-parse --show-toplevel
if (-not $repoRoot) {
  Write-Error "Could not determine repo root."
  exit 1
}

Set-Location $repoRoot
$scopePath = "."
$branch = git rev-parse --abbrev-ref HEAD

Write-Host "[split-commit] Branch:           $branch"
Write-Host "[split-commit] Scope:            $repoRoot"
Write-Host "[split-commit] Message:          $Message"
Write-Host "[split-commit] Primary remote:   $PrimaryRemote"
Write-Host "[split-commit] Secondary remote: $SecondaryRemote"

try {
  Ensure-Remote -Name $PrimaryRemote -Url ""
  Ensure-Remote -Name $SecondaryRemote -Url $SecondaryRemoteUrl
} catch {
  Write-Error $_
  exit 1
}

$files = Collect-RepoFiles -ScopePath $scopePath
if ($files.Count -eq 0) {
  Write-Host "[split-commit] No repo changes to commit."
  if (-not $NoPush) {
    Write-Host "[split-commit] No new file commits; pushing any existing local commits."
    try {
      Push-Branch -Remote $PrimaryRemote -Branch $branch
      Push-Branch -Remote $SecondaryRemote -Branch $branch
    } catch {
      Write-Error $_
      exit 1
    }
  } else {
    Write-Host "[split-commit] Skipping push (-NoPush)"
  }
  exit 0
}

Write-Host "[split-commit] Files: $($files.Count)"
foreach ($f in $files) {
  Write-Host " - $f"
}

foreach ($f in $files) {
  Write-Host "[split-commit] Committing: $f"
  if (Test-Path -LiteralPath $f) {
    git add -- "$f" | Out-Null
  } else {
    git rm --cached -f -- "$f" 2>$null | Out-Null
    git rm -f -- "$f" 2>$null | Out-Null
  }

  $hasChanges = $(git diff --cached --quiet -- "$f"; if ($LASTEXITCODE -eq 0) { $false } else { $true })
  if (-not $hasChanges) {
    Write-Host "  - No staged changes, skipping"
    continue
  }

  git commit -m "$Message" -- "$f"
  if ($LASTEXITCODE -ne 0) {
    Write-Error "Commit failed for '$f'."
    exit 1
  }
}

if (-not $NoPush) {
  try {
    Push-Branch -Remote $PrimaryRemote -Branch $branch
    Push-Branch -Remote $SecondaryRemote -Branch $branch
  } catch {
    Write-Error $_
    exit 1
  }
} else {
  Write-Host "[split-commit] Skipping push (-NoPush)"
}

Write-Host "[split-commit] Done."

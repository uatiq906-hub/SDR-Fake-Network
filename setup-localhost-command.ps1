<#
  Run this ONCE in PowerShell:

      .\setup-localhost-command.ps1

  It adds a "localhost" function to your PowerShell profile, pointed at
  this exact folder. After that, opening a new PowerShell window anywhere
  and typing:

      localhost

  starts server.py, no matter what directory you're in. You don't need to
  cd into the project folder first.

  If PowerShell refuses to run this because of its execution policy, run
  this once first (in an admin PowerShell), then try again:

      Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
#>

$projectDir = $PSScriptRoot
$profilePath = $PROFILE

if (-not (Test-Path $profilePath)) {
    New-Item -ItemType File -Path $profilePath -Force | Out-Null
}

$marker = "# --- net control: localhost command ---"
$existing = Get-Content $profilePath -Raw -ErrorAction SilentlyContinue

if ($existing -and $existing.Contains($marker)) {
    Write-Host "The 'localhost' command is already set up in your profile."
    Write-Host "Edit $profilePath by hand if you need to point it at a different folder."
} else {
    $block = @"

$marker
function localhost {
    Push-Location "$projectDir"
    try { python server.py }
    finally { Pop-Location }
}
# --- end net control ---
"@
    Add-Content -Path $profilePath -Value $block
    Write-Host "Done. Close and reopen PowerShell, then type 'localhost' from anywhere."
}

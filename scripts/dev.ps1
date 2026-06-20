<#
.SYNOPSIS
  Serve the whole FrugalRoute stack on Windows PowerShell (the `make dev` equivalent).

  Starts uvicorn (api) in the background and the app static server in the foreground
  with a same-origin /api reverse-proxy, so the browser never hits CORS. Ctrl-C stops
  both.

.EXAMPLE
  pwsh scripts/dev.ps1
  pwsh scripts/dev.ps1 -ApiPort 8000 -AppPort 5500
#>
param(
  [int]$ApiPort = 8000,
  [int]$AppPort = 5500
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot

Write-Host "FrugalRoute dev stack:  app http://localhost:$AppPort/   (api on :$ApiPort)"

$api = Start-Process -PassThru -NoNewWindow -FilePath "python" `
  -ArgumentList @("-m", "uvicorn", "frugalroute_api.app:app", "--port", "$ApiPort") `
  -WorkingDirectory $repo

try {
  $env:FRUGALROUTE_API_PROXY = "http://localhost:$ApiPort"
  & node (Join-Path $repo "app/tests/e2e/static-server.mjs") $AppPort
}
finally {
  if ($api -and -not $api.HasExited) { Stop-Process -Id $api.Id -Force }
}

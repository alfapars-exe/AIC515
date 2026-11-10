# env.ps1 — Faz 3 ortak ortam (dot-source edin: . .\env.ps1)
$env:USERNAME = if ($env:USERNAME) { $env:USERNAME } else { "user" }
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"   # Windows konsolu cp1252: Türkçe/oklu print'ler çökmesin
$env:PYTHONUTF8 = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:VIZE_ROOT = (Resolve-Path "$PSScriptRoot\..").Path
$global:PY = if ($env:P3_PYTHON) { $env:P3_PYTHON } else { "python" }   # docforensics ortamının python.exe yolu için P3_PYTHON ayarlayın
$global:P3 = "$env:VIZE_ROOT\phase3"
$global:P3Y = "$env:VIZE_ROOT\phase3_yolo"
$global:P2E = "$env:VIZE_ROOT\phase2\experiments"
New-Item -ItemType Directory -Force -Path "$P3\logs" | Out-Null
function global:Step([string]$name, [scriptblock]$block) {
  $log = "$P3\logs\$name.log"
  if (Test-Path "$P3\logs\$name.done") { Write-Host "skip $name (done)"; return }
  Write-Host "=== $name $(Get-Date -Format HH:mm:ss) ===" -ForegroundColor Cyan
  "$(Get-Date -Format s) START $name" | Out-File -Append "$P3\logs\queue.log"
  $t = Get-Date
  $global:LASTEXITCODE = 0
  & $block 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log
  $sec = [int]((Get-Date) - $t).TotalSeconds
  if ($LASTEXITCODE -eq 0) { "$sec" | Out-File "$P3\logs\$name.done"; "$(Get-Date -Format s) DONE $name ${sec}s" | Out-File -Append "$P3\logs\queue.log" }
  else { "$(Get-Date -Format s) FAIL $name ${sec}s exit=$LASTEXITCODE" | Out-File -Append "$P3\logs\queue.log" }
}

# env.ps1 — Faz 3 ortak ortam (dot-source edin: . .\env.ps1)
$env:USERNAME = if ($env:USERNAME) { $env:USERNAME } else { "user" }
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"   # Windows konsolu cp1252: Türkçe/oklu print'ler çökmesin
$env:PYTHONUTF8 = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$global:KOD = (Resolve-Path "$PSScriptRoot\..").Path                                                    # kod çatısı
$env:VIZE_ROOT = if ((Split-Path $KOD -Leaf) -eq "kod") { Split-Path $KOD -Parent } else { $KOD }       # proje kökü (veri + çıktılar)
$global:PY = if ($env:P3_PYTHON) { $env:P3_PYTHON } else { "python" }   # docforensics ortamının python.exe yolu için P3_PYTHON ayarlayın
$global:P3 = "$KOD\phase3"                        # kod klasörleri
$global:P3Y = "$KOD\phase3_yolo"
$global:P2E = "$KOD\phase2\experiments"
$global:P3LOG = "$env:VIZE_ROOT\phase3\logs"   # kuyruk günlükleri ve .done işaretleri (çıktı tarafı)
$global:P3YW = "$env:VIZE_ROOT\phase3_yolo"     # YOLO iş klasörü: yolo_data\, runs\, yolo11*.pt
New-Item -ItemType Directory -Force -Path "$P3LOG" | Out-Null
function global:Step([string]$name, [scriptblock]$block) {
  $log = "$P3LOG\$name.log"
  if (Test-Path "$P3LOG\$name.done") { Write-Host "skip $name (done)"; return }
  Write-Host "=== $name $(Get-Date -Format HH:mm:ss) ===" -ForegroundColor Cyan
  "$(Get-Date -Format s) START $name" | Out-File -Append "$P3LOG\queue.log"
  $t = Get-Date
  $global:LASTEXITCODE = 0
  & $block 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log
  $sec = [int]((Get-Date) - $t).TotalSeconds
  if ($LASTEXITCODE -eq 0) { "$sec" | Out-File "$P3LOG\$name.done"; "$(Get-Date -Format s) DONE $name ${sec}s" | Out-File -Append "$P3LOG\queue.log" }
  else { "$(Get-Date -Format s) FAIL $name ${sec}s exit=$LASTEXITCODE" | Out-File -Append "$P3LOG\queue.log" }
}

# run_everything.ps1 — Tüm hattı sırayla çalıştırır (TR). Proje kökünde çalıştırın:
#   powershell -ExecutionPolicy Bypass -File run_everything.ps1 -Python "<docforensics>\python.exe" [-SkipDownload] [-SkipTrain]
# Ön koşullar: baselines\ENV_SETUP.md (conda env 'docforensics', torch cu128), HF_TOKEN (veri seti gated).
# Süreler (RTX 5080): audit ~3 dk · pilot baseline ~25 dk · GT+kontrol ~15 dk · zero-shot ~40 dk · Faz 2 eğitim+değerlendirme ~4 s · işaretçi probu ~60 dk · istatistik ~10 dk.
param([string]$Python = "python", [switch]$SkipDownload, [switch]$SkipTrain)
$ErrorActionPreference = "Stop"
$ROOT = $PSScriptRoot; Set-Location $ROOT
$env:VIZE_ROOT = $ROOT                      # tüm betikler kökü bu değişkenden okur
$env:USERNAME = if ($env:USERNAME) { $env:USERNAME } else { "user" }
$env:PYTHONUNBUFFERED = "1"
New-Item -ItemType Directory -Force -Path logs_master | Out-Null
function Step($name, $block) { Write-Host "=== $name ===" -ForegroundColor Cyan; $t=Get-Date; & $block 2>&1 | Tee-Object -FilePath "logs_master\$name.log"; Write-Host ("    done in {0:n0}s" -f ((Get-Date)-$t).TotalSeconds) }

if (-not $SkipDownload) { Step "00_download" { & $Python tools\download_dataset.py --out data\AIForge-Doc-v1 } }
Step "01_audit"     { & $Python dataset_audit\audit_dataset.py --data data\AIForge-Doc-v1 --out dataset_audit --workers 24 }
if (-not $SkipTrain) { Step "02_pilot_baselines" { Push-Location baselines; & powershell -ExecutionPolicy Bypass -File run_all.ps1 -Python $Python; Pop-Location } }
Step "03_gt"        { & $Python phase2\gt\make_gt.py --root $ROOT --splits training testing --workers 12 }
Step "03b_gt_grid"  { & $Python phase2\gt\make_verification_grid.py --root $ROOT --n 100 --seed 42 }
Step "04_controls"  { & $Python phase2\controls\make_controls.py --root $ROOT --workers 12 }
Step "05_zero_shot" { Push-Location phase2\literature_baselines; & $Python run_trufor.py; & $Python run_adcd.py; & $Python evaluate_zero_shot.py; Pop-Location }   # ağırlıklar: README.md / LICENSES.md
if (-not $SkipTrain) { Step "06_phase2_experiments" { Push-Location phase2\experiments; & powershell -ExecutionPolicy Bypass -File run_phase2.ps1 -Python $Python -Epochs 10; Pop-Location } }
Step "07_green_marker_probe" { Push-Location phase2\experiments; & $Python probe_green_marker.py; & $Python analyze_green_marker.py; Pop-Location }
Step "08_stats"     { Push-Location phase2\stats; & $Python stats_analysis.py; Pop-Location }
Step "09_vlm_verification (opsiyonel, platform LLM gerektirir)" { Write-Host "phase2\gt\vlm_verify.md içindeki yönergeye bakın (host.llm ile çalıştırıldı)" }
Step "10_build_reports" { & $Python tools\build_manuscript.py --root $ROOT }
Write-Host "Tamamlandı. Raporlar: reports\ ; paket: TESLIM_PAKETI\" -ForegroundColor Green

# run_phase4.ps1 - Faz 4 kuyrugu (ResNet-18 tam veri + isaretci onarimi + goruntu duzeyi kalibrasyon).
# Yeniden baslatilabilir: phase4_resnet\logs\<adim>.done varsa adim atlanir.
# Kullanim: powershell -ExecutionPolicy Bypass -File kod\phase4_resnet\run_phase4.ps1 -Stage All   (Smoke | Data | Train | Infer | Calib | All)
param([string]$Stage = "All", [int[]]$Seeds = @(42, 43, 44))
$ErrorActionPreference = "Continue"
$PY = if ($env:P3_PYTHON) { $env:P3_PYTHON } else { "C:\Users\PARS\.claude-science\conda\envs\docforensics\python.exe" }
$HERE = $PSScriptRoot
$KOD = Split-Path $HERE -Parent
$env:VIZE_ROOT = Split-Path $KOD -Parent
$env:PYTHONUNBUFFERED = "1"; $env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUTF8 = "1"; $env:PYTHONDONTWRITEBYTECODE = "1"
$env:USERNAME = if ($env:USERNAME) { $env:USERNAME } else { "user" }
$LOG = "$env:VIZE_ROOT\phase4_resnet\logs"
New-Item -ItemType Directory -Force -Path $LOG | Out-Null
Set-Location $HERE
function Step([string]$name, [string[]]$a) {
  if (Test-Path "$LOG\$name.done") { Write-Host "skip $name (done)"; return }
  "$(Get-Date -Format s) START $name" | Out-File -Append "$LOG\queue.log" -Encoding utf8
  $t = Get-Date; $global:LASTEXITCODE = 0
  & $PY -B @a 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath "$LOG\$name.log"
  $sec = [int]((Get-Date) - $t).TotalSeconds
  if ($LASTEXITCODE -eq 0) { "$sec" | Out-File "$LOG\$name.done"; "$(Get-Date -Format s) DONE $name ${sec}s" | Out-File -Append "$LOG\queue.log" -Encoding utf8 }
  else { "$(Get-Date -Format s) FAIL $name ${sec}s exit=$LASTEXITCODE" | Out-File -Append "$LOG\queue.log" -Encoding utf8; throw "adim basarisiz: $name" }
}
if ($Stage -eq "Smoke") {
  Step "smoke_data"  @("build_patches_p4.py", "--limit", "150", "--tag", "dryrun")
  Step "smoke_train_raw" @("train_resnet_p4.py", "--variant", "raw", "--seed", "42", "--epochs", "2", "--tag", "dryrun")
  Step "smoke_train_mr"  @("train_resnet_p4.py", "--variant", "mr", "--seed", "42", "--epochs", "2", "--tag", "dryrun")
  Step "smoke_infer" @("infer_images_p4.py", "--models", "resnet18_ft_full_s42_dryrun,resnet18_ft_full_mr_s42_dryrun,resnet18_ft_best", "--limit", "24", "--tag", "dryrun")
  Step "smoke_patch_eval" @("eval_patches_p4.py", "--models", "resnet18_ft_best,resnet18_ft_full_s42_dryrun", "--tag", "dryrun")
  Step "smoke_calib" @("calibrate_p4.py", "--tag", "dryrun")
  return
}
if ($Stage -in "Data", "All")  { Step "data_patches" @("build_patches_p4.py") }
if ($Stage -in "Train", "All") {
  foreach ($s in $Seeds) { Step "train_raw_s$s" @("train_resnet_p4.py", "--variant", "raw", "--seed", "$s") }
  foreach ($s in $Seeds) { Step "train_mr_s$s"  @("train_resnet_p4.py", "--variant", "mr", "--seed", "$s") }
}
if ($Stage -in "Infer", "All") {
  $m = @(); foreach ($s in $Seeds) { $m += "resnet18_ft_full_s$s"; $m += "resnet18_ft_full_mr_s$s" }; $m += "resnet18_ft_best"
  Step "patch_eval" @("eval_patches_p4.py", "--models", ($m -join ","))
  Step "infer_images" @("infer_images_p4.py", "--models", ($m -join ","))
}
if ($Stage -in "Calib", "All") { Step "calibrate" @("calibrate_p4.py") }
"$(Get-Date -Format s) $Stage finished" | Out-File -Append "$LOG\queue.log" -Encoding utf8

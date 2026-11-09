# run_all.ps1 — AIForge-Doc v1 smoke baseline'larını sırayla çalıştırır (TR).
# Kullanım (PowerShell, docforensics ortamı aktif ya da $Python ile yol verin):
#   .\run_all.ps1                       # varsayılan epoch'lar (ResNet 12, U-Net 15)
#   .\run_all.ps1 -ResnetEpochs 6 -UnetEpochs 8
# Not: Veri kökü common/data.py içinde PROJECT_ROOT (VIZE_ROOT ortam değişkeni ile değiştirilebilir).
param(
    [string]$Python = "python",
    [int]$ResnetEpochs = 12,
    [int]$UnetEpochs = 15,
    [int]$Seed = 42,
    [switch]$SkipRobustness
)
$ErrorActionPreference = "Continue"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here
New-Item -ItemType Directory -Force -Path (Join-Path $here "logs") | Out-Null
$env:USERNAME = if ($env:USERNAME) { $env:USERNAME } else { "user" }

function Run($name, $args) {
    $log = Join-Path $here "logs\$name.log"
    Write-Host ">>> $name  ($args)"
    $t = Get-Date
    & $Python @args 2>&1 | Tee-Object -FilePath $log
    Write-Host "<<< $name bitti: $([int]((Get-Date) - $t).TotalSeconds) s"
}

# B) klasik öznitelikler + SVM / RF  (patch önbelleğini de üretir)
Run "classical" @("train_classical.py", "--seed", $Seed)
# C) ResNet18: probe / ft / scratch
foreach ($s in @("probe", "ft", "scratch")) {
    Run "resnet_$s" @("train_resnet.py", "--setting", $s, "--epochs", $ResnetEpochs, "--seed", $Seed)
}
# D) U-Net: rgb / rgbfreq
foreach ($v in @("rgb", "rgbfreq")) {
    Run "unet_$v" @("train_unet.py", "--variant", $v, "--epochs", $UnetEpochs, "--seed", $Seed)
}
# E/F) dayanıklılık probu, kaynak tablosu, figürler
if ($SkipRobustness) { Run "evaluate" @("evaluate.py", "--skip-robustness") } else { Run "evaluate" @("evaluate.py") }
Write-Host "Tamamlandı. Sonuçlar: results\, figürler: figures\, loglar: logs\, modeller: ..\models\"

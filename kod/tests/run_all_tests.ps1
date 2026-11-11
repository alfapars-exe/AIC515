# Tüm uygulama testleri (docforensics ortamı). Çıktı: <proje>\tests\run_all_<ad>.log, özet <proje>\tests\run_all_summary.txt
# Kullanım: powershell -ExecutionPolicy Bypass -File kod\tests\run_all_tests.ps1 [-N 35]
param([int]$N = 35)
$py = if ($env:P3_PYTHON) { $env:P3_PYTHON } else { "python" }
$KOD = $PSScriptRoot; while ($KOD -and -not (Test-Path (Join-Path $KOD "_paths.py"))) { $KOD = Split-Path $KOD -Parent }   # kod çatısı
$ROOT = if ((Split-Path $KOD -Leaf) -eq "kod") { Split-Path $KOD -Parent } else { $KOD }                                       # proje kökü (veri + çıktılar)
$T = "$KOD\tests"; $O = "$ROOT\tests"; New-Item -ItemType Directory -Force -Path $O | Out-Null
Set-Location $ROOT
$env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUTF8 = "1"; $env:PYTHONDONTWRITEBYTECODE = "1"; $env:AI515_TEST_N = "$N"
$env:USERNAME = if ($env:USERNAME) { $env:USERNAME } else { "user" }
$res = @()
foreach ($t in @(@("app", @("$T\test_app.py")), @("app_p3", @("$T\test_app_p3.py")), @("xai_filters", @("$T\test_xai_filters.py")), @("xai", @("-m", "pytest", "$T\test_xai.py", "-q", "-p", "no:cacheprovider")))) {
  $name = $t[0]; $a = $t[1]
  & $py -B @a *> "$O\run_all_$name.log"
  $res += "$name rc=$LASTEXITCODE"
}
$res | Set-Content "$O\run_all_summary.txt" -Encoding UTF8
$res

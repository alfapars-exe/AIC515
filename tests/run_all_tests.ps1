# Tüm uygulama testleri (proje kökü, docforensics). Çıktı: tests\run_all_<ad>.log, özet tests\run_all_summary.txt
$py = "C:\Users\PARS\.claude-science\conda\envs\docforensics\python.exe"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONIOENCODING = "utf-8"
$res = @()
foreach ($t in @(@("app", @("tests\test_app.py")), @("app_p3", @("tests\test_app_p3.py")), @("xai_filters", @("tests\test_xai_filters.py")), @("xai", @("-m","pytest","tests\test_xai.py","-q")))) {
  $name = $t[0]; $a = $t[1]
  & $py @a *> "tests\run_all_$name.log"
  $res += "$name rc=$LASTEXITCODE"
}
$res | Set-Content tests\run_all_summary.txt

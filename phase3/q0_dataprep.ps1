# q0_dataprep.ps1 — CPU veri hazırlığı: işaretçi onarımı + YOLO dönüşümleri (yeniden başlatılabilir: .done dosyaları)
. "$PSScriptRoot\env.ps1"
Step "p0_marker_repair" { Set-Location $P3; & $PY make_marker_repaired.py }
Step "y0_conv_tiles"     { Set-Location $P3Y; & $PY convert_to_yolo.py --mode tiles --tile 1024 --stride 768 --neg-per-img 1 --auth-neg 0 }
Step "y0_conv_authneg"   { Set-Location $P3Y; & $PY convert_to_yolo.py --mode tiles --tile 1024 --stride 768 --neg-per-img 1 --auth-neg 1 --suffix _authneg }
Step "y0_conv_gemini"    { Set-Location $P3Y; & $PY convert_to_yolo.py --mode tiles --train-filter gemini_only --suffix _geminionly }
Step "y0_conv_full"      { Set-Location $P3Y; & $PY convert_to_yolo.py --mode full }
"q0 finished" | Out-File -Append "$P3\logs\queue.log"

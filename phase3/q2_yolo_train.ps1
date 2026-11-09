# q2_yolo_train.ps1 — Faz 3 YOLO eğitim kuyruğu (GPU; q1 ile paralel). Plan: phase3_yolo/README.md §2.
param([int]$Epochs = 40, [int]$Workers = 6, [switch]$SmokeOnly)
. "$PSScriptRoot\env.ps1"
Set-Location $P3Y
while (-not (Test-Path "$P3\logs\y0_conv_tiles.done")) { Start-Sleep 20 }
$YDATA = "yolo_data\tiles\data.yaml"
Step "y_A_smoke" { & $PY train_yolo.py --data $YDATA --model yolo11s.pt --imgsz 1024 --epochs 4 --fraction 0.3 --seed 42 --workers $Workers --dry-run }
Step "y_A_smoke_infer" { & $PY infer_to_maps.py --run aiforge-tiles-yolo11s-img1024-conserv-s42-dryrun --split val --limit 20 }
if ($SmokeOnly) { "q2 smoke finished" | Out-File -Append "$P3\logs\queue.log"; return }
# B — ana: 3 seed
foreach ($s in 42, 43, 44) { Step "y_B_s$s" { & $PY train_yolo.py --data $YDATA --model yolo11s.pt --imgsz 1024 --epochs $Epochs --seed $s --workers $Workers } }
# C — tek faktörlü ablasyonlar (s42)
while (-not (Test-Path "$P3\logs\y0_conv_full.done")) { Start-Sleep 20 }
Step "y_C1_full1280" { & $PY train_yolo.py --data yolo_data\full\data.yaml --model yolo11s.pt --imgsz 1280 --epochs $Epochs --seed 42 --workers $Workers }
Step "y_C2_defaultaug" { & $PY train_yolo.py --data $YDATA --model yolo11s.pt --imgsz 1024 --epochs $Epochs --seed 42 --aug default --workers $Workers }
Step "y_C3_authneg" { & $PY train_yolo.py --data yolo_data\tiles_authneg\data.yaml --model yolo11s.pt --imgsz 1024 --epochs $Epochs --seed 42 --workers $Workers }
# D — model boyutu
Step "y_D1_11n" { & $PY train_yolo.py --data $YDATA --model yolo11n.pt --imgsz 1024 --epochs $Epochs --seed 42 --workers $Workers }
Step "y_D2_11m" { & $PY train_yolo.py --data $YDATA --model yolo11m.pt --imgsz 1024 --epochs $Epochs --seed 42 --batch 4 --workers $Workers }
# E — held-out generator
Step "y_E1_geminionly" { & $PY train_yolo.py --data yolo_data\tiles_geminionly\data.yaml --model yolo11s.pt --imgsz 1024 --epochs $Epochs --seed 42 --workers $Workers }
"q2 finished" | Out-File -Append "$P3\logs\queue.log"

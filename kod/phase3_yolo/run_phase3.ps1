# run_phase3.ps1 — Faz 3: YOLO tespit formülasyonu vs U-Net bbox-modeli (AIForge-Doc v1)
# Çalıştırma: conda activate docforensics ; & "...\Vize Proje\kod\phase3_yolo\run_phase3.ps1" -Stage A   (güncel kuyruklar: kod\phase3\q*.ps1)
# Aşamalar birbirine bağlı: A (duman) geçmeden B başlatma. Her aşama tek başına yeniden koşulabilir.
param([ValidateSet("Setup","Data","A","B","C","D","E","Eval","Probes","Metrics","All")][string]$Stage = "A")
$ErrorActionPreference = "Stop"
$env:USERNAME = if ($env:USERNAME) { $env:USERNAME } else { "user" }
$KOD = $PSScriptRoot; while ($KOD -and -not (Test-Path (Join-Path $KOD "_paths.py"))) { $KOD = Split-Path $KOD -Parent }   # kod çatısı
$ROOT = if ((Split-Path $KOD -Leaf) -eq "kod") { Split-Path $KOD -Parent } else { $KOD }                                       # proje kökü (veri + çıktılar)
$Y = $PSScriptRoot; $P2 = "$KOD\phase2\experiments"
Set-Location "$ROOT\phase3_yolo"   # iş klasörü = çıktı kökü (yolo_data\, runs\, yolo11*.pt)

function Run($cmd) { Write-Host ">> $cmd" -ForegroundColor Cyan; Invoke-Expression $cmd; if ($LASTEXITCODE -ne 0) { throw "HATA: $cmd" } }

if ($Stage -in "Setup","All") {
  # ultralytics'in PyPI CPU-torch'u ÜZERİNE yazmasını engelle: --no-deps + eksikleri elle
  Run "pip install ultralytics --no-deps"
  Run "pip install py-cpuinfo ultralytics-thop polars"
  Run "python -c `"import torch,ultralytics;print(torch.__version__, torch.cuda.is_available(), ultralytics.__version__)`""
}
if ($Stage -in "Data","All") {
  Run "python `"$Y\convert_to_yolo.py`" --mode tiles --tile 1024 --stride 768 --neg-per-img 1 --auth-neg 0"
  Run "python `"$Y\convert_to_yolo.py`" --mode tiles --tile 1024 --stride 768 --neg-per-img 1 --auth-neg 1 --suffix _authneg"
  Run "python `"$Y\convert_to_yolo.py`" --mode tiles --train-filter gemini_only --suffix _geminionly"
  Run "python `"$Y\convert_to_yolo.py`" --mode full"
}
if ($Stage -in "A","All") {   # Duman testi: 2 epoch, %15 veri, ~5 dk. Epoch süresi → B/C/D ETA'sı buradan.
  Run "python `"$Y\train_yolo.py`" --data yolo_data\tiles\data.yaml --model yolo11s.pt --imgsz 1024 --epochs 2 --fraction 0.15 --seed 42 --dry-run"
  Run "python `"$Y\infer_to_maps.py`" --run aiforge-tiles-yolo11s-img1024-conserv-s42-dryrun --split val --limit 20"
  Run "python `"$P2\evaluate_p2.py`" --models aiforge-tiles-yolo11s-img1024-conserv-s42-dryrun --split val --gts bbox"
}
if ($Stage -in "B","All") {   # Ana: yolo11s, tile1024, conserv × 3 seed  (U-Net bbox s42/43/44 ile eşleşir)
  foreach ($s in 42,43,44) { Run "python `"$Y\train_yolo.py`" --data yolo_data\tiles\data.yaml --model yolo11s.pt --imgsz 1024 --epochs 40 --seed $s" }
}
if ($Stage -in "C","All") {   # Tek-değişkenli ablasyonlar (seed 42): giriş çözünürlüğü / aug / authentic negatif
  Run "python `"$Y\train_yolo.py`" --data yolo_data\full\data.yaml --model yolo11s.pt --imgsz 1280 --epochs 40 --seed 42"
  Run "python `"$Y\train_yolo.py`" --data yolo_data\tiles\data.yaml --model yolo11s.pt --imgsz 1024 --epochs 40 --seed 42 --aug default"
  Run "python `"$Y\train_yolo.py`" --data yolo_data\tiles_authneg\data.yaml --model yolo11s.pt --imgsz 1024 --epochs 40 --seed 42"
}
if ($Stage -in "D","All") {   # Model boyutu (seed 42)
  Run "python `"$Y\train_yolo.py`" --data yolo_data\tiles\data.yaml --model yolo11n.pt --imgsz 1024 --epochs 40 --seed 42"
  Run "python `"$Y\train_yolo.py`" --data yolo_data\tiles\data.yaml --model yolo11m.pt --imgsz 1024 --epochs 40 --seed 42 --batch 4"
}
if ($Stage -in "E","All") {   # Held-out generator (Faz 2 C karşılığı)
  Run "python `"$Y\train_yolo.py`" --data yolo_data\tiles_geminionly\data.yaml --model yolo11s.pt --imgsz 1024 --epochs 40 --seed 42"
}
$MAIN = "aiforge-tiles-yolo11s-img1024-conserv-s42,aiforge-tiles-yolo11s-img1024-conserv-s43,aiforge-tiles-yolo11s-img1024-conserv-s44"
if ($Stage -in "Eval","All") {  # Tam-görüntü çıkarım + 3 GT metrikleri (τ* önce val'de)
  foreach ($r in $MAIN.Split(",")) { Run "python `"$Y\infer_to_maps.py`" --run $r --split val"; Run "python `"$Y\infer_to_maps.py`" --run $r --split test" }
  Run "python `"$P2\evaluate_p2.py`" --models $MAIN --split val"
  Run "python `"$P2\evaluate_p2.py`" --models $MAIN --split test"
}
if ($Stage -in "Probes","All") { # Dayanıklılık + benign kontroller (Faz 2 D4/E3 ile aynı koşullar)
  foreach ($r in $MAIN.Split(",")) {
    foreach ($c in "jpeg75","rescale075","blur1") { Run "python `"$Y\infer_to_maps.py`" --run $r --split val --condition $c" }
    Run "python `"$Y\infer_to_maps.py`" --run $r --split val --source authentic"
  }
  foreach ($c in "jpeg75","rescale075","blur1") { Run "python `"$P2\evaluate_p2.py`" --models $MAIN --split val --condition $c" }
}
if ($Stage -in "Metrics","All") { # Tespit metriği: iki aile aynı tanımla
  Run "python `"$Y\detection_metrics.py`" --yolo $MAIN --unet unet_bbox_s42,unet_bbox_s43,unet_bbox_s44,unet_pixel_eps8_morph_s42,unet_pixel_eps8_morph_s43,unet_pixel_eps8_morph_s44 --split val"
  Run "python `"$Y\detection_metrics.py`" --yolo $MAIN --unet unet_bbox_s42,unet_bbox_s43,unet_bbox_s44 --split test"
}
Write-Host "Aşama $Stage tamam." -ForegroundColor Green

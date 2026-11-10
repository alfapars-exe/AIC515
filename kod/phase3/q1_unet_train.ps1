# q1_unet_train.ps1 — Faz 3 U-Net/DeepLabV3+ eğitim kuyruğu (GPU). Her run tek faktör değiştirir; sabitler Faz 2 ile aynı.
# Yeniden başlatılabilir (logs\<step>.done). p0_marker_repair bitmeden işaretçi-onarım adımları beklenir.
. "$PSScriptRoot\env.ps1"
Set-Location $P3
$GTS = "bbox", "pixel_eps8_morph", "glyph_v2"
# W2 — ikinci mimari (DeepLabV3+, aynı ResNet-18 kodlayıcı), 3 GT × s42
foreach ($g in $GTS) { Step "t_dlv3p_${g}_s42" { & $PY train_unet_p3.py --arch dlv3p --gt $g --seed 42 } }
# W4/W5 — işaretçi-onarılmış eğitim; piksel/glif GT'den işaretçi pikselleri çıkarılmış (_nm)
while (-not (Test-Path "$P3LOG\p0_marker_repair.done")) { Start-Sleep 20 }
$env:P3_MARKER_REPAIR = "1"
Step "t_mr_bbox_s42"            { & $PY train_unet_p3.py --gt bbox --marker-repair --seed 42 }
Step "t_mr_pixel_nm_s42"        { & $PY train_unet_p3.py --gt pixel_eps8_morph_nm --marker-repair --seed 42 }
Step "t_mr_glyph_nm_s42"        { & $PY train_unet_p3.py --gt glyph_v2_nm --marker-repair --seed 42 }
$env:P3_MARKER_REPAIR = "0"
# PHASE3-a — birleşik kayıp (birincil piksel/glif maske + 0.5 × bbox yardımcı kayıp)
Step "t_joint_pixel_s42" { & $PY train_unet_p3.py --gt pixel_eps8_morph --aux-bbox 0.5 --seed 42 }
Step "t_joint_glyph_s42" { & $PY train_unet_p3.py --gt glyph_v2 --aux-bbox 0.5 --seed 42 }
# W3 — held-out generator: gemini-only × 3 GT × ek 2 seed (Faz 2'de yalnız s42)
foreach ($s in 43, 44) { foreach ($g in $GTS) { Step "t_heldout_${g}_s$s" { Set-Location $P2E; & $PY train_unet_p2.py --gt $g --seed $s --train-filter gemini_only; Set-Location $P3 } } }
# PHASE3-b — 20 epoch (Faz 2: 10 epoch'ta piksel/glif eğrileri hâlâ yükseliyordu)
foreach ($g in $GTS) { Step "t_e20_${g}_s42" { Set-Location $P2E; & $PY train_unet_p2.py --gt $g --seed 42 --epochs 20 --patience 6 --tag e20; Set-Location $P3 } }
"q1 finished" | Out-File -Append "$P3LOG\queue.log"

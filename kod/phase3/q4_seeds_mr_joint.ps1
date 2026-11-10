# q4_seeds_mr_joint.ps1 — ek tohumlar (43, 44): işaretçi-onarımlı kutu U-Net ve birleşik kayıplı piksel U-Net (GPU).
# Tek değişen etken tohumdur; komutlar q1/q3a/run_v2 içindeki s42 koşularıyla aynıdır. Plan ve öngörüler (koşmadan önce yazıldı):
#   reports/inceleme/deney_plani_ek_tohum_2026-11-10.md
# Yeniden başlatılabilir (logs\<adım>.done). Python yolu P3_PYTHON ortam değişkeninden okunur (env.ps1).
# Öneri: saatler süren bu kuyruğu etkileşimli bir uygulamanın süreç ağacının dışında başlatın
#   (bkz. reports/inceleme/debug_kuyruk_kesintisi_2026-11-09.md).
. "$PSScriptRoot\env.ps1"
Set-Location $P3
$me = Get-CimInstance Win32_Process -Filter "ProcessId=$PID"
$par = Get-CimInstance Win32_Process -Filter "ProcessId=$($me.ParentProcessId)"
"$(Get-Date -Format s) q4 start pid=$PID parent=$($me.ParentProcessId) ($($par.Name))" | Out-File -Append "$P3LOG\queue.log"
$SEEDS = 43, 44
$P3V2 = "$KOD\phase3_v2"
# 1) eğitim — q1 ile aynı komutlar, yalnız --seed farklı
$env:P3_MARKER_REPAIR = "1"
foreach ($s in $SEEDS) { Step "t_mr_bbox_s$s" { & $PY train_unet_p3.py --gt bbox --marker-repair --seed $s } }
$env:P3_MARKER_REPAIR = "0"
foreach ($s in $SEEDS) { Step "t_joint_pixel_s$s" { & $PY train_unet_p3.py --gt pixel_eps8_morph --aux-bbox 0.5 --seed $s } }
# 2) değerlendirme — q3a ile aynı aşamalar
$MR = "unet_bbox_mr_s43,unet_bbox_mr_s44"
$JT = "unetjoint_pixel_eps8_morph_s43,unetjoint_pixel_eps8_morph_s44"
Step "e_mr_s4344"    { & $PY run_eval_p3.py --models $MR --stages "val,test,robust,controls" }
Step "e_joint_s4344" { & $PY run_eval_p3.py --models $JT --stages "val,test,robust,controls,ablation,rectfit" }
# 3) sondalar — yeni etiketle ayrı dosyaya (eski sonda dosyaları ezilmez); synthesize_p3.py bu etiketi de okur
Step "p_probes_unet_mrj_s4344" { & $PY probes_p3.py --unet "$MR,$JT" --tag unet_mrj_s4344 }
# 4) AIForge-Doc v2 — yeni işaretçisiz kutu modelleri için çıkarım; değerlendirme tüm model listesiyle yeniden yazılır
#    (v2_eval.py eval sonuç dosyalarını baştan yazar; eski modellerin tahminleri diskte olduğundan sayıları değişmez)
$SEG = "unet_bbox_s42,unet_pixel_eps8_morph_s42,unet_glyph_v2_s42,unet_bbox_s43,unet_pixel_eps8_morph_s43,unet_glyph_v2_s43,unet_bbox_s44,unet_pixel_eps8_morph_s44,unet_glyph_v2_s44,dlv3p_bbox_s42,dlv3p_pixel_eps8_morph_s42,dlv3p_glyph_v2_s42,unet_bbox_mr_s42,unet_pixel_eps8_morph_nm_mr_s42,unet_glyph_v2_nm_mr_s42,unet_bbox_geminionly_s42,unet_pixel_eps8_morph_geminionly_s42,unet_glyph_v2_geminionly_s42"
$YOLO = "aiforge-tiles-yolo11s-img1024-conserv-s42,aiforge-tiles-yolo11s-img1024-conserv-s43,aiforge-tiles-yolo11s-img1024-conserv-s44,aiforge-full-yolo11s-img1280-conserv-s42,aiforge-tiles_geminionly-yolo11s-img1024-conserv-s42,aiforge-tiles_authneg-yolo11s-img1024-conserv-s42"
Step "v2_infer_mr_s4344" { Set-Location $P3V2; & $PY v2_eval.py infer --models $MR --subset primary; Set-Location $P3 }
Step "v2_eval_primary_s4344" { Set-Location $P3V2; & $PY v2_eval.py eval --models "$SEG,$MR" --yolo $YOLO --subset primary; Set-Location $P3 }
"$(Get-Date -Format s) q4 finished" | Out-File -Append "$P3LOG\queue.log"

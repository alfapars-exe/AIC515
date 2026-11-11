# q5_kalan_tohumlar.ps1 — kalan tek tohumlu deneylerin 43 ve 44 tohumları ve yeni deney: yalnız Gemini + işaretçisiz eğitim (42–44) (GPU).
# Tek değişen etken tohumdur (yeni deneyde: eğitim süzgeci + işaretçi onarımı); komutlar q1/q2/q3a/q3b/run_v2 içindeki s42 koşularıyla aynıdır.
# Plan ve öngörüler (koşmadan önce yazıldı ve depoya gönderildi): reports/inceleme/deney_plani_kalan_tohumlar_2026-11-11.md
# Yeniden başlatılabilir (logs\<adım>.done). Blok sırası: A1, B, A2, A3 (U-Net ailesi), D (YOLO ana model, kısa), C (YOLO ablasyonları, en uzun). Kuyruk istenen anda durdurulabilir, biten adımlar korunur.
# Python yolu P3_PYTHON ortam değişkeninden okunur (env.ps1). Kuyruğu etkileşimli bir uygulamanın süreç ağacının dışında başlatın
#   (bkz. reports/inceleme/debug_kuyruk_kesintisi_2026-11-09.md).
param([switch]$SkipYolo)
. "$PSScriptRoot\env.ps1"
Set-Location $P3
$me = Get-CimInstance Win32_Process -Filter "ProcessId=$PID"
$par = Get-CimInstance Win32_Process -Filter "ProcessId=$($me.ParentProcessId)"
"$(Get-Date -Format s) q5 start pid=$PID parent=$($me.ParentProcessId) ($($par.Name))" | Out-File -Append "$P3LOG\queue.log"
$S2 = 43, 44; $S3 = 42, 43, 44
$GTS = "bbox", "pixel_eps8_morph", "glyph_v2"
$P3V2 = "$KOD\phase3_v2"
$ALL = "val,test,robust,controls,ablation,rectfit"
$MRB = "unet_bbox_mr_s43,unet_bbox_mr_s44"                                   # q4 ile eklenen modeller (listelerde korunur)
$JP = "unetjoint_pixel_eps8_morph_s43,unetjoint_pixel_eps8_morph_s44"

# ================= A1) işaretçisiz piksel/glif (nm) ve birleşik kayıplı glif: 43, 44
$env:P3_MARKER_REPAIR = "1"
foreach ($s in $S2) { Step "t_mr_pixel_nm_s$s" { & $PY train_unet_p3.py --gt pixel_eps8_morph_nm --marker-repair --seed $s } }
foreach ($s in $S2) { Step "t_mr_glyph_nm_s$s" { & $PY train_unet_p3.py --gt glyph_v2_nm --marker-repair --seed $s } }
$env:P3_MARKER_REPAIR = "0"
foreach ($s in $S2) { Step "t_joint_glyph_s$s" { & $PY train_unet_p3.py --gt glyph_v2 --aux-bbox 0.5 --seed $s } }
$MRNM = "unet_pixel_eps8_morph_nm_mr_s43,unet_glyph_v2_nm_mr_s43,unet_pixel_eps8_morph_nm_mr_s44,unet_glyph_v2_nm_mr_s44"
$JG = "unetjoint_glyph_v2_s43,unetjoint_glyph_v2_s44"
Step "e_mrnm_s4344"       { & $PY run_eval_p3.py --models $MRNM --stages "val,test,robust,controls" }
Step "e_jointglyph_s4344" { & $PY run_eval_p3.py --models $JG --stages $ALL }
Step "p_probes_unet_q5a"  { & $PY probes_p3.py --unet $MRNM --tag unet_q5a }
Step "v2_infer_q5a" { Set-Location $P3V2; & $PY v2_eval.py infer --models $MRNM --subset primary; Set-Location $P3 }

# ================= B) yeni deney: yalnız Gemini + işaretçisiz eğitim (3 referans × 3 tohum); ayrı tutulan üretici = Ideogram
$env:P3_MARKER_REPAIR = "1"
foreach ($s in $S3) {
  Step "t_gomr_bbox_s$s"  { & $PY train_unet_p3.py --gt bbox --marker-repair --train-filter gemini_only --seed $s }
  Step "t_gomr_pixel_s$s" { & $PY train_unet_p3.py --gt pixel_eps8_morph_nm --marker-repair --train-filter gemini_only --seed $s }
  Step "t_gomr_glyph_s$s" { & $PY train_unet_p3.py --gt glyph_v2_nm --marker-repair --train-filter gemini_only --seed $s }
}
$env:P3_MARKER_REPAIR = "0"
$GOMR = ($S3 | ForEach-Object { "unet_bbox_geminionly_mr_s$_", "unet_pixel_eps8_morph_nm_geminionly_mr_s$_", "unet_glyph_v2_nm_geminionly_mr_s$_" }) -join ","
Step "e_gomr" { & $PY run_eval_p3.py --models $GOMR --stages "val,test" }

# ================= A2) DeepLabV3+: 43, 44
foreach ($s in $S2) { foreach ($g in $GTS) { Step "t_dlv3p_${g}_s$s" { & $PY train_unet_p3.py --arch dlv3p --gt $g --seed $s } } }
$DL = ($S2 | ForEach-Object { "dlv3p_bbox_s$_", "dlv3p_pixel_eps8_morph_s$_", "dlv3p_glyph_v2_s$_" }) -join ","
Step "e_dlv3p_s4344"     { & $PY run_eval_p3.py --models $DL --stages $ALL }
Step "p_probes_unet_q5b" { & $PY probes_p3.py --unet "dlv3p_bbox_s43,dlv3p_bbox_s44" --tag unet_q5b }
Step "v2_infer_q5b" { Set-Location $P3V2; & $PY v2_eval.py infer --models $DL --subset primary; Set-Location $P3 }
# v2 değerlendirmesi tüm model listesiyle yeniden yazılır (eski modellerin tahminleri diskte olduğundan sayıları değişmez)
$SEG = "unet_bbox_s42,unet_pixel_eps8_morph_s42,unet_glyph_v2_s42,unet_bbox_s43,unet_pixel_eps8_morph_s43,unet_glyph_v2_s43,unet_bbox_s44,unet_pixel_eps8_morph_s44,unet_glyph_v2_s44,dlv3p_bbox_s42,dlv3p_pixel_eps8_morph_s42,dlv3p_glyph_v2_s42,unet_bbox_mr_s42,unet_pixel_eps8_morph_nm_mr_s42,unet_glyph_v2_nm_mr_s42,unet_bbox_geminionly_s42,unet_pixel_eps8_morph_geminionly_s42,unet_glyph_v2_geminionly_s42"
$YOLO = "aiforge-tiles-yolo11s-img1024-conserv-s42,aiforge-tiles-yolo11s-img1024-conserv-s43,aiforge-tiles-yolo11s-img1024-conserv-s44,aiforge-full-yolo11s-img1280-conserv-s42,aiforge-tiles_geminionly-yolo11s-img1024-conserv-s42,aiforge-tiles_authneg-yolo11s-img1024-conserv-s42"
Step "v2_eval_primary_q5" { Set-Location $P3V2; & $PY v2_eval.py eval --models "$SEG,$MRB,$MRNM,$DL" --yolo $YOLO --subset primary; Set-Location $P3 }
# kutu düzeyi tespit ölçütü (AP50, AP50-95): önceki liste + DeepLabV3+ kutu modelinin yeni tohumları
$DETY = "aiforge-tiles-yolo11s-img1024-conserv-s42,aiforge-tiles-yolo11s-img1024-conserv-s43,aiforge-tiles-yolo11s-img1024-conserv-s44,aiforge-tiles-yolo11s-img1024-default-s42,aiforge-tiles_authneg-yolo11s-img1024-conserv-s42,aiforge-tiles-yolo11n-img1024-conserv-s42,aiforge-tiles-yolo11m-img1024-conserv-s42,aiforge-tiles_geminionly-yolo11s-img1024-conserv-s42,aiforge-full-yolo11s-img1280-conserv-s42"
$DETU = "unet_bbox_s42,unet_bbox_s43,unet_bbox_s44,unet_pixel_eps8_morph_s42,unet_pixel_eps8_morph_s43,unet_pixel_eps8_morph_s44,unet_glyph_v2_s42,dlv3p_bbox_s42,unet_bbox_mr_s42,unetjoint_pixel_eps8_morph_s42,$MRB,$JP,dlv3p_bbox_s43,dlv3p_bbox_s44"
Step "y_detmetrics_val_q5"  { Set-Location $P3Y; & $PY detection_metrics.py --yolo $DETY --unet $DETU --split val; Set-Location $P3 }
Step "y_detmetrics_test_q5" { Set-Location $P3Y; & $PY detection_metrics.py --yolo $DETY --unet $DETU --split test; Set-Location $P3 }

# ================= A3) 20 epok: 43, 44 (train_unet_p2.py; q1 ile aynı komut)
foreach ($s in $S2) { foreach ($g in $GTS) { Step "t_e20_${g}_s$s" { Set-Location $P2E; & $PY train_unet_p2.py --gt $g --seed $s --epochs 20 --patience 6 --tag e20; Set-Location $P3 } } }
$E20 = ($S2 | ForEach-Object { "unet_bbox_s${_}_e20", "unet_pixel_eps8_morph_s${_}_e20", "unet_glyph_v2_s${_}_e20" }) -join ","
Step "e_e20_s4344" { & $PY run_eval_p3.py --models $E20 --stages "val,test" }
# kalan sondalar (sentez tablosundaki DeepLabV3+ piksel/glif ve birleşik glif satırlarının da üç tohumlu olması için)
Step "p_probes_unet_q5c" { & $PY probes_p3.py --unet "dlv3p_pixel_eps8_morph_s43,dlv3p_glyph_v2_s43,dlv3p_pixel_eps8_morph_s44,dlv3p_glyph_v2_s44,$JG" --tag unet_q5c }
# yeni deneyin denetimi: yalnız Gemini ile eğitilen kutu modellerinde işaretçi bağımlılığı (standart ve işaretçisiz eğitim, 3 tohum)
$GOB = ($S3 | ForEach-Object { "unet_bbox_geminionly_s$_" }) -join ","
$GOMRB = ($S3 | ForEach-Object { "unet_bbox_geminionly_mr_s$_" }) -join ","
Step "p_probes_unet_q5d" { & $PY probes_p3.py --unet "$GOB,$GOMRB" --tag unet_q5d }
"$(Get-Date -Format s) q5 U-Net bloklari bitti" | Out-File -Append "$P3LOG\queue.log"
if ($SkipYolo) { "$(Get-Date -Format s) q5 finished (YOLO atlandi)" | Out-File -Append "$P3LOG\queue.log"; return }

# ================= D) YOLO11s ana modelinin 43 ve 44 tohumları: dayanıklılık koşulları ve sondalar (q3b bunları yalnız s42 için koşmuştu)
$B2 = "aiforge-tiles-yolo11s-img1024-conserv-s43", "aiforge-tiles-yolo11s-img1024-conserv-s44"
$RC = "jpeg75", "rescale075", "blur1", "clahe", "gray"
foreach ($r in $B2) { foreach ($c in $RC) { Step "yi_${r}_$c" { Set-Location $P3; & $PY -c "import p3ext,sys,runpy; sys.argv=['infer_to_maps.py','--run','$r','--split','val','--condition','$c']; sys.path.insert(0,r'$P3Y'); runpy.run_path(r'$P3Y\infer_to_maps.py', run_name='__main__')" } } }
$MB2 = ($B2 -join ",")
foreach ($c in $RC) { Step "ye_rob_${c}_s4344" { Set-Location $P3; & $PY -c "import p3ext,sys,runpy; sys.argv=['evaluate_p2.py','--models','$MB2','--split','val','--condition','$c']; runpy.run_path(r'$P2E\evaluate_p2.py', run_name='__main__')" } }
Set-Location $P3
Step "p_probes_yolo_s4344" { & $PY probes_p3.py --yolo $MB2 --tag yolo_s4344 }
"$(Get-Date -Format s) q5 D blogu bitti" | Out-File -Append "$P3LOG\queue.log"

# ================= C) YOLO11 ablasyonlarının 43 ve 44 tohumları (en uzun blok; q2 ile aynı komutlar, 30 epok)
$Epochs = 30; $Workers = 6
Set-Location $P3YW
$YDATA = "yolo_data\tiles\data.yaml"
foreach ($s in $S2) {
  Step "y_E1_geminionly_s$s" { & $PY "$P3Y\train_yolo.py" --data yolo_data\tiles_geminionly\data.yaml --model yolo11s.pt --imgsz 1024 --epochs $Epochs --seed $s --workers $Workers }
  Step "y_C1_full1280_s$s"   { & $PY "$P3Y\train_yolo.py" --data yolo_data\full\data.yaml --model yolo11s.pt --imgsz 1280 --epochs $Epochs --seed $s --workers $Workers }
  Step "y_C2_defaultaug_s$s" { & $PY "$P3Y\train_yolo.py" --data $YDATA --model yolo11s.pt --imgsz 1024 --epochs $Epochs --seed $s --aug default --workers $Workers }
  Step "y_C3_authneg_s$s"    { & $PY "$P3Y\train_yolo.py" --data yolo_data\tiles_authneg\data.yaml --model yolo11s.pt --imgsz 1024 --epochs $Epochs --seed $s --workers $Workers }
  Step "y_D1_11n_s$s"        { & $PY "$P3Y\train_yolo.py" --data $YDATA --model yolo11n.pt --imgsz 1024 --epochs $Epochs --seed $s --workers $Workers }
  Step "y_D2_11m_s$s"        { & $PY "$P3Y\train_yolo.py" --data $YDATA --model yolo11m.pt --imgsz 1024 --epochs $Epochs --seed $s --batch 4 --workers $Workers }
}
$NEWY = @(); foreach ($s in $S2) { $NEWY += "aiforge-tiles_geminionly-yolo11s-img1024-conserv-s$s", "aiforge-full-yolo11s-img1280-conserv-s$s", "aiforge-tiles-yolo11s-img1024-default-s$s", "aiforge-tiles_authneg-yolo11s-img1024-conserv-s$s", "aiforge-tiles-yolo11n-img1024-conserv-s$s", "aiforge-tiles-yolo11m-img1024-conserv-s$s" }
Set-Location $P3Y
foreach ($r in $NEWY) {
  $fx = if ($r -like "aiforge-full-*") { @("--full", "1280") } else { @() }
  Step "yi_${r}_val"  { & $PY infer_to_maps.py --run $r --split val @fx }
  Step "yi_${r}_test" { & $PY infer_to_maps.py --run $r --split test @fx }
}
$MN = ($NEWY -join ",")
Step "ye_val_q5"  { & $PY "$P2E\evaluate_p2.py" --models $MN --split val }
Step "ye_test_q5" { & $PY "$P2E\evaluate_p2.py" --models $MN --split test }
Step "y_detmetrics_val_q5y"  { & $PY detection_metrics.py --yolo "$DETY,$MN" --unet $DETU --split val }
Step "y_detmetrics_test_q5y" { & $PY detection_metrics.py --yolo "$DETY,$MN" --unet $DETU --split test }
# v2: tablodaki YOLO varyantlarının (tam 1280, yalnız Gemini, özgün negatif karo) yeni tohumları
$V2Y = ($NEWY | Where-Object { $_ -match "geminionly|aiforge-full|authneg" }) -join ","
Step "v2_infer_q5y" { Set-Location $P3V2; & $PY v2_eval.py infer --yolo $V2Y --subset primary; Set-Location $P3Y }
Step "v2_eval_primary_q5y" { Set-Location $P3V2; & $PY v2_eval.py eval --models "$SEG,$MRB,$MRNM,$DL" --yolo "$YOLO,$V2Y" --subset primary; Set-Location $P3Y }
Set-Location $P3
"$(Get-Date -Format s) q5 finished" | Out-File -Append "$P3LOG\queue.log"

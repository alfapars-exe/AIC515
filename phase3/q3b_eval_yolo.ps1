# q3b_eval_yolo.ps1 — Faz 3 YOLO değerlendirme + sondalar + tespit metriği + hız (q2 ve q3a bittikten sonra).
. "$PSScriptRoot\env.ps1"
while (-not ((Select-String -Path "$P3\logs\queue.log" -Pattern "^q2 finished" -Quiet) -and (Select-String -Path "$P3\logs\queue.log" -Pattern "^q3a finished" -Quiet))) { Start-Sleep 60 }
$B = "aiforge-tiles-yolo11s-img1024-conserv-s42", "aiforge-tiles-yolo11s-img1024-conserv-s43", "aiforge-tiles-yolo11s-img1024-conserv-s44"
$ABL = "aiforge-tiles-yolo11s-img1024-default-s42", "aiforge-tiles_authneg-yolo11s-img1024-conserv-s42", "aiforge-tiles-yolo11n-img1024-conserv-s42", "aiforge-tiles-yolo11m-img1024-conserv-s42", "aiforge-tiles_geminionly-yolo11s-img1024-conserv-s42"
$FULL = "aiforge-full-yolo11s-img1280-conserv-s42"
$ALLY = $B + $ABL + $FULL
Set-Location $P3Y
foreach ($r in $ALLY) {
  $fx = if ($r -eq $FULL) { @("--full", "1280") } else { @() }
  Step "yi_${r}_val"  { & $PY infer_to_maps.py --run $r --split val @fx }
  Step "yi_${r}_test" { & $PY infer_to_maps.py --run $r --split test @fx }
}
$M = ($ALLY -join ",")
Step "ye_val"  { & $PY "$P2E\evaluate_p2.py" --models $M --split val }
Step "ye_test" { & $PY "$P2E\evaluate_p2.py" --models $M --split test }
# Dayanıklılık + kontrol kaynakları (yalnız B, 3 seed)
foreach ($r in @($B[0])) {   # dayanıklılık/kontrol kaynakları yalnız s42 (süre bütçesi; ana metrikler 3 seed)
  foreach ($c in "jpeg75", "rescale075", "blur1", "clahe", "gray") { Step "yi_${r}_$c" { Set-Location $P3; & $PY -c "import p3ext,sys,runpy; sys.argv=['infer_to_maps.py','--run','$r','--split','val','--condition','$c']; sys.path.insert(0,r'$P3Y'); runpy.run_path(r'$P3Y\infer_to_maps.py', run_name='__main__')" } }
  foreach ($s in "authentic", "benign_pasteback", "fullimage_resample") { Step "yi_${r}_$s" { Set-Location $P3Y; & $PY infer_to_maps.py --run $r --split val --source $s } }
}
$MB = $B[0]   # rob + probes yalnız s42 (süre bütçesi)
foreach ($c in "jpeg75", "rescale075", "blur1", "clahe", "gray") { Step "ye_rob_$c" { Set-Location $P3; & $PY -c "import p3ext,sys,runpy; sys.argv=['evaluate_p2.py','--models','$MB','--split','val','--condition','$c']; runpy.run_path(r'$P2E\evaluate_p2.py', run_name='__main__')" } }
Set-Location $P3
Step "p_probes_yolo" { & $PY probes_p3.py --yolo $MB --tag yolo }
Set-Location $P3Y
Step "y_detmetrics_val"  { & $PY detection_metrics.py --yolo $M --unet "unet_bbox_s42,unet_bbox_s43,unet_bbox_s44,unet_pixel_eps8_morph_s42,unet_pixel_eps8_morph_s43,unet_pixel_eps8_morph_s44,unet_glyph_v2_s42,dlv3p_bbox_s42,unet_bbox_mr_s42,unetjoint_pixel_eps8_morph_s42" --split val }
Step "y_detmetrics_test" { & $PY detection_metrics.py --yolo $M --unet "unet_bbox_s42,unet_bbox_s43,unet_bbox_s44,unet_pixel_eps8_morph_s42,unet_pixel_eps8_morph_s43,unet_pixel_eps8_morph_s44,unet_glyph_v2_s42,dlv3p_bbox_s42,unet_bbox_mr_s42,unetjoint_pixel_eps8_morph_s42" --split test }
Set-Location $P3
Step "bench_speed" { & $PY bench_speed.py "unet_bbox_s42,dlv3p_bbox_s42,unetjoint_pixel_eps8_morph_s42" ($B[0] + ",aiforge-tiles-yolo11n-img1024-conserv-s42,aiforge-tiles-yolo11m-img1024-conserv-s42," + $FULL) }
"q3b finished" | Out-File -Append "$P3\logs\queue.log"

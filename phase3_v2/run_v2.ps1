# AIForge-Doc v2 harici test kuyruğu (TR). Her adım tamamlanınca logs\<ad>.done yazılır; yeniden çalıştırmada atlanır.
$ErrorActionPreference = "Continue"
$py = "C:\Users\PARS\.claude-science\conda\envs\docforensics\python.exe"
Set-Location $PSScriptRoot
$SEG = "unet_bbox_s42,unet_pixel_eps8_morph_s42,unet_glyph_v2_s42,unet_bbox_s43,unet_pixel_eps8_morph_s43,unet_glyph_v2_s43,unet_bbox_s44,unet_pixel_eps8_morph_s44,unet_glyph_v2_s44,dlv3p_bbox_s42,dlv3p_pixel_eps8_morph_s42,dlv3p_glyph_v2_s42,unet_bbox_mr_s42,unet_pixel_eps8_morph_nm_mr_s42,unet_glyph_v2_nm_mr_s42,unet_bbox_geminionly_s42,unet_pixel_eps8_morph_geminionly_s42,unet_glyph_v2_geminionly_s42"
$YOLO = "aiforge-tiles-yolo11s-img1024-conserv-s42,aiforge-tiles-yolo11s-img1024-conserv-s43,aiforge-tiles-yolo11s-img1024-conserv-s44,aiforge-full-yolo11s-img1280-conserv-s42,aiforge-tiles_geminionly-yolo11s-img1024-conserv-s42,aiforge-tiles_authneg-yolo11s-img1024-conserv-s42"
$MAINSEG = "unet_bbox_s42,unet_pixel_eps8_morph_s42,unet_glyph_v2_s42"
$MAINYOLO = "aiforge-tiles-yolo11s-img1024-conserv-s42"
function Step($name, [string[]]$a) {
  if (Test-Path "logs\$name.done") { Write-Host "atla $name"; return }
  Write-Host "$(Get-Date -f HH:mm:ss) basla $name"
  & $py v2_eval.py @a *>> "logs\$name.log"
  if ($LASTEXITCODE -eq 0) { New-Item "logs\$name.done" -ItemType File -Force | Out-Null; Write-Host "$(Get-Date -f HH:mm:ss) bitti $name" }
  else { Write-Host "$(Get-Date -f HH:mm:ss) FAIL $name ($LASTEXITCODE)"; New-Item "logs\$name.FAIL" -ItemType File -Force | Out-Null }
}
Step "a_seg_primary"  @("infer","--models",$SEG,"--subset","primary")
Step "a_yolo_primary" @("infer","--yolo",$YOLO,"--subset","primary")
Step "a_eval_primary" @("eval","--models",$SEG,"--yolo",$YOLO,"--subset","primary")
Step "b_seg_all"      @("infer","--models",$MAINSEG,"--subset","all")
Step "b_yolo_all"     @("infer","--yolo",$MAINYOLO,"--subset","all")
Step "b_eval_all"     @("eval","--models",$MAINSEG,"--yolo",$MAINYOLO,"--subset","all")
Write-Host "$(Get-Date -f HH:mm:ss) KUYRUK BITTI"

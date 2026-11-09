# q3a_eval_unet.ps1 — Faz 3 U-Net/DeepLabV3+ değerlendirme kuyruğu (q1 bittikten sonra; sonuç CSV'lerine sıralı yazar).
. "$PSScriptRoot\env.ps1"
Set-Location $P3
while (-not (Select-String -Path "$P3\logs\queue.log" -Pattern "^q1 finished" -Quiet)) { Start-Sleep 60 }
$ALL = "val,test,robust,controls,ablation,rectfit"
Step "e_dlv3p"   { & $PY run_eval_p3.py --models dlv3p_bbox_s42,dlv3p_pixel_eps8_morph_s42,dlv3p_glyph_v2_s42 --stages $ALL }
Step "e_mr"      { & $PY run_eval_p3.py --models unet_bbox_mr_s42,unet_pixel_eps8_morph_nm_mr_s42,unet_glyph_v2_nm_mr_s42 --stages "val,test,robust,controls" }
Step "e_joint"   { & $PY run_eval_p3.py --models unetjoint_pixel_eps8_morph_s42,unetjoint_glyph_v2_s42 --stages $ALL }
Step "e_heldout" { & $PY run_eval_p3.py --models "unet_bbox_geminionly_s43,unet_pixel_eps8_morph_geminionly_s43,unet_glyph_v2_geminionly_s43,unet_bbox_geminionly_s44,unet_pixel_eps8_morph_geminionly_s44,unet_glyph_v2_geminionly_s44" --stages "val,test" }
Step "e_e20"     { & $PY run_eval_p3.py --models unet_bbox_s42_e20,unet_pixel_eps8_morph_s42_e20,unet_glyph_v2_s42_e20 --stages "val,test" }
# Faz 2 B modelleri için yalnız YENİ test-zamanı koşulları (clahe, gray): robustness.csv'deki eski satırları ezmemek için ayrı sonda
Step "p_probes_unet" { & $PY probes_p3.py --unet "unet_bbox_s42,unet_pixel_eps8_morph_s42,unet_glyph_v2_s42,dlv3p_bbox_s42,dlv3p_pixel_eps8_morph_s42,dlv3p_glyph_v2_s42,unet_bbox_mr_s42,unet_pixel_eps8_morph_nm_mr_s42,unet_glyph_v2_nm_mr_s42,unetjoint_pixel_eps8_morph_s42,unetjoint_glyph_v2_s42" --tag unet }
Step "p_probes_unet_s4344" { & $PY probes_p3.py --unet "unet_bbox_s43,unet_pixel_eps8_morph_s43,unet_glyph_v2_s43,unet_bbox_s44,unet_pixel_eps8_morph_s44,unet_glyph_v2_s44" --tag unet_s4344 }
"q3a finished" | Out-File -Append "$P3\logs\queue.log"

# run_phase2.ps1 — Faz 2 tam kuyruk (TR). RTX 5080 (16 GB) üzerinde sıralı çalıştırma.
# Kullanım:  powershell -ExecutionPolicy Bypass -File run_phase2.ps1 [-Python <python.exe>] [-Epochs 10]
# Ön koşul: conda/venv 'docforensics' (torch cu128, segmentation-models-pytorch, opencv-python-headless, scikit-learn, pandas, matplotlib).
# Windows sandbox notu: torch import için USERNAME env; DataLoader kullanılmaz (thread prefetch, multiprocessing yok).
param(
  [string]$Python = "python",
  [int]$Epochs = 10,
  [int]$Patience = 4
)
$ErrorActionPreference = "Stop"
$env:USERNAME = if ($env:USERNAME) { $env:USERNAME } else { "user" }
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:PYTHONUNBUFFERED = "1"
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force -Path logs, results, figures, preds | Out-Null

# A) Dry-run (1 epoch, bbox GT) — epoch süresi ölçümü (README'ye yazılır)
& $Python -u train_unet_p2.py --gt bbox --seed 42 --epochs 1 --tag dryrun 2>&1 | Tee-Object -FilePath logs\dryrun_bbox.log

# B) Ana ızgara: 3 GT × 3 seed (10 epoch, patience 4) ve C) held-out generator (gemini_only × 3 GT, seed 42)
#    run_queue.py tamamlanmış run'ları (results/train_<run>.json) atlar → yeniden başlatılabilir.
& $Python -u run_queue.py --stage BC --epochs $Epochs --patience $Patience 2>&1 | Tee-Object -FilePath logs\queue_stdout.log

# D/E) Değerlendirme: val+test tam-görüntü sliding-window (preds/), 3 GT metrikleri, dayanıklılık, kontroller, sınır ablasyonu, rect-fit
foreach ($seed in 42, 43, 44) {
  $models = "unet_bbox_s$seed,unet_pixel_eps8_morph_s$seed,unet_glyph_v2_s$seed"
  & $Python -u run_eval.py --models $models --stages val,test,robust,controls,ablation,rectfit 2>&1 | Tee-Object -FilePath "logs\eval_B_s$seed.log"
}
& $Python -u run_eval.py --models unet_bbox_geminionly_s42,unet_pixel_eps8_morph_geminionly_s42,unet_glyph_v2_geminionly_s42 --stages val,test 2>&1 | Tee-Object -FilePath logs\eval_C_geminionly.log

# F) H4 global-iz probu (image-level ResNet-18)
& $Python -u h4_global_trace.py --epochs 10 2>&1 | Tee-Object -FilePath logs\h4.log

# G) Sentez: özet tablolar (seed ort ± std), held-out generator tablosu, figürler, README
& $Python -u synthesize_p2.py 2>&1 | Tee-Object -FilePath logs\synthesize.log

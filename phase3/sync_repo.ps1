# sync_repo.ps1 — Vize Proje → AIC515_repo (GitHub alfapars-exe/AIC515) senkronu.
# Yalnız kod, sonuç tabloları, şekiller, raporlar, literatür (PDF hariç) ve bildiri kopyalanır.
# HARİÇ: AIForge-Doc görüntüleri ve türevleri (CC BY-NC-SA; yeniden dağıtılmaz), model ağırlıkları, tahmin haritaları,
# üçüncü taraf kod/ağırlıklar (TruFor, ADCD-Net), önbellekler, teslim paketinin kopyası.
param([string]$Message = "Update", [switch]$NoPush)
$ErrorActionPreference = "Stop"
$SRC = "C:\Users\PARS\Desktop\PROJELER\AI 515\Vize Proje"
$DST = "C:\Users\PARS\Desktop\PROJELER\AI 515\AIC515_repo"
$XD = @("data", "models", "TESLIM_PAKETI", "_to_delete", ".archify", "__pycache__", ".git",
        "preds", "preds_boxes", "hists", "cache", "marker_repaired", "yolo_data", "weights",
        "TruFor", "ADCD-Net", "adcd", "seed_sources", "TEKMER", "vlm_verification", "xfer",
        "benign_pasteback", "fullimage_resample", "reencode_png",
        "pixel_eps4", "pixel_eps8", "pixel_eps16", "pixel_eps8_morph", "pixel_eps8_morph_nm", "glyph", "glyph_v2", "glyph_v2_nm", "bbox_and_eps8")
$XF = @("*.pt", "*.pth", "*.ckpt", "*.onnx", "*.npz", "*.npy", "*.zip", "*.tar", "*.gz", "*.parquet", "*.log", "*.done",
        "*.pdf.part", "token*", "*.env", ".env", "train_batch*.jpg", "val_batch*.jpg", "labels.jpg", "verification_grid.pdf")
robocopy $SRC $DST /E /XD $XD /XF $XF /NFL /NDL /NJH /NP /R:1 /W:1 | Out-Null
# literatür PDF'leri (telif) ve büyük dosya kontrolü
Get-ChildItem "$DST\literature" -Recurse -Filter *.pdf -ErrorAction SilentlyContinue | Remove-Item -Force
$big = Get-ChildItem $DST -Recurse -File | Where-Object { $_.FullName -notmatch '\\.git\\' -and $_.Length -gt 50MB }
if ($big) { $big | ForEach-Object { "BÜYÜK DOSYA (>50MB) çıkarıldı: $($_.FullName)" ; Remove-Item $_.FullName -Force } }
# gizli anahtar taraması (HF/GitHub/OpenAI/Google)
$hits = Get-ChildItem $DST -Recurse -File -Include *.py, *.ps1, *.md, *.json, *.yaml, *.yml, *.txt, *.ipynb |
  Where-Object { $_.FullName -notmatch '\\.git\\' } |
  Select-String -Pattern 'hf_[A-Za-z0-9]{30,}', 'ghp_[A-Za-z0-9]{30,}', 'sk-[A-Za-z0-9]{30,}', 'AIza[0-9A-Za-z_\-]{30,}' -ErrorAction SilentlyContinue
if ($hits) { $hits | ForEach-Object { "GİZLİ ANAHTAR ŞÜPHESİ: $($_.Path):$($_.LineNumber)" }; throw "Gizli anahtar şüphesi — push durduruldu" }
Set-Location $DST
git config user.name "HARUN BENLİ"
git config user.email "80422104+alfapars-exe@users.noreply.github.com"
git add -A
$st = git status --porcelain
if (-not $st) { "Değişiklik yok"; return }
git commit -q -m $Message
"commit: $(git rev-parse --short HEAD)  dosya: $((git ls-files).Count)  boyut: {0:N1} MB" -f ((Get-ChildItem $DST -Recurse -File | Where-Object { $_.FullName -notmatch '\\.git\\' } | Measure-Object Length -Sum).Sum / 1MB)
if (-not $NoPush) { git push origin main 2>&1 | ForEach-Object { "$_" } }

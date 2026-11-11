# sync_repo.ps1 — Vize Proje → AIC515_repo (GitHub alfapars-exe/AIC515) senkronu.
# Yalnız kod, sonuç tabloları, şekiller, raporlar, literatür (PDF hariç) ve bildiri kopyalanır.
# HARİÇ: AIForge-Doc görüntüleri ve türevleri (CC BY-NC-SA; yeniden dağıtılmaz), model ağırlıkları, tahmin haritaları,
# üçüncü taraf kod/ağırlıklar (TruFor, ADCD-Net), önbellekler, teslim paketinin kopyası.
param([string]$Message = "Update", [switch]$NoPush)
$ErrorActionPreference = "Stop"
$KOD = (Resolve-Path "$PSScriptRoot\..").Path
$SRC = if ((Split-Path $KOD -Leaf) -eq "kod") { Split-Path $KOD -Parent } else { $KOD }   # proje kökü
$DST = Join-Path (Split-Path $SRC -Parent) "AIC515_repo"
$XD = @("data", "models", "TESLIM_PAKETI", "_to_delete", ".archify", "__pycache__", ".git", "_arsiv", "_yedek", "_backup", "third_party", "logs_master", ".pytest_cache",
        "preds", "preds_boxes", "hists", "cache", "marker_repaired", "yolo_data", "weights",
        "TruFor", "ADCD-Net", "adcd", "seed_sources", "TEKMER", "vlm_verification", "xfer",
        "benign_pasteback", "fullimage_resample", "reencode_png", "drafts", "_v2_zip",
        "pixel_eps4", "pixel_eps8", "pixel_eps16", "pixel_eps8_morph", "pixel_eps8_morph_nm", "glyph", "glyph_v2", "glyph_v2_nm", "bbox_and_eps8")
$XF = @("*.pt", "*.pth", "*.ckpt", "*.onnx", "*.npz", "*.npy", "*.zip", "*.tar", "*.gz", "*.parquet", "*.log", "*.done",
        "*.pdf.part", "token*", "*.env", ".env", "train_batch*.jpg", "val_batch*.jpg", "labels.jpg", "verification_grid.pdf",
        "verification_grid_page*.png", "*_LOCAL_ONLY*")   # veri setinden kırpılmış görüntü + türetilmiş maske içeren, yeniden dağıtılmayan şekiller
robocopy $SRC $DST /E /XD $XD /XF $XF /NFL /NDL /NJH /NP /R:1 /W:1 | Out-Null
# kod\ çatısına taşınan betiklerin depodaki eski konumlu kopyalarını kaldır (robocopy /PURGE kullanılmıyor)
if (Test-Path "$SRC\kod") { Get-ChildItem "$SRC\kod" -Recurse -File -Include *.py, *.ps1 | ForEach-Object {
  $old = Join-Path $DST $_.FullName.Substring("$SRC\kod\".Length); if (Test-Path $old) { Remove-Item $old -Force } } }
# yeniden dağıtılmayan şekillerin depoda kalmış eski kopyalarını kaldır (robocopy /PURGE kullanılmıyor; silme git add -A ile kayda girer)
Get-ChildItem $DST -Recurse -File -Include "verification_grid_page*.png", "*_LOCAL_ONLY*" -ErrorAction SilentlyContinue |
  Where-Object { $_.FullName -notmatch '\\.git\\' } | Remove-Item -Force
# veri tabanının metin alanları (düzenlenen alanın eski ve yeni değeri) depoya gönderilmez: yayımlanan kopyalardan bu sütunlar çıkarılır.
# Çalışma ağacındaki dosyalar değişmez; değerler kapılı veri tabanının üstverisinden yeniden elde edilir.
foreach ($rel in "dataset_audit\samples_forged.csv", "phase3_v2\v2_table.csv") {
  $f = Join-Path $DST $rel
  if (Test-Path $f) {
    $rows = Import-Csv -LiteralPath $f -Encoding UTF8
    if ($rows.Count -and ($rows[0].PSObject.Properties.Name -contains "original_value")) {
      $rows | Select-Object -Property * -ExcludeProperty original_value, forged_value | Export-Csv -LiteralPath $f -NoTypeInformation -Encoding UTF8
    }
  }
}
# çalışma ağacında artık bulunmayan, depoda kalmış eski dosyaları kaldır (robocopy /PURGE kullanılmıyor; silme git add -A ile kayda girer).
# Güvenlik: yalnız ASCII adlı ve izlenen dosyalar; .gitignore ve .gitattributes korunur.
Push-Location $DST
$keepOnlyInRepo = @(".gitignore", ".gitattributes")
git ls-files | ForEach-Object {
  if (($keepOnlyInRepo -notcontains $_) -and ($_ -cmatch '^[\x20-\x7E]+$') -and ($_ -notmatch '"')) {   # -cmatch: tr-TR kültüründe büyük I harfi büyük-küçük duyarsız eşleşmede ASCII dışına düşer
    $rel = $_ -replace '/', '\'
    if (-not (Test-Path -LiteralPath (Join-Path $SRC $rel))) {
      Remove-Item -LiteralPath (Join-Path $DST $rel) -Force -ErrorAction SilentlyContinue; "eski dosya kaldırıldı: $_"
    }
  }
}
Pop-Location
# kişisel mutlak yollar: yayımlanan kopyalarda proje kökü göreli yola, kullanıcı klasörü "~" işaretine çevrilir.
# Çalışma ağacındaki dosyalar değişmez. Üç yazım da ele alınır: tek ters eğik çizgi, JSON kaçışlı çift ters eğik çizgi, düz eğik çizgi.
$pats = New-Object System.Collections.ArrayList
foreach ($pair in @(@($SRC, ""), @($env:USERPROFILE, "~"))) {
  $p = $pair[0]; $to = $pair[1]
  if ($to -eq "") {   # proje kökü: ayırıcıyla birlikte silinir (göreli yol kalır)
    [void]$pats.Add(@([regex]::Escape($p.Replace('\', '\\') + '\\'), ""))
    [void]$pats.Add(@([regex]::Escape($p + '\'), ""))
    [void]$pats.Add(@([regex]::Escape($p.Replace('\', '/') + '/'), ""))
  }
  [void]$pats.Add(@([regex]::Escape($p.Replace('\', '\\')), $(if ($to -eq "") { "." } else { $to })))
  [void]$pats.Add(@([regex]::Escape($p), $(if ($to -eq "") { "." } else { $to })))
  [void]$pats.Add(@([regex]::Escape($p.Replace('\', '/')), $(if ($to -eq "") { "." } else { $to })))
}
$userLeaf = Split-Path $env:USERPROFILE -Leaf
$textExt = @(".csv", ".json", ".yaml", ".yml", ".txt", ".err", ".md", ".ps1", ".py", ".tex", ".cfg", ".ini", ".toml")
$nSan = 0; $sanList = New-Object System.Collections.ArrayList
$ro = [System.Text.RegularExpressions.RegexOptions]::IgnoreCase -bor [System.Text.RegularExpressions.RegexOptions]::CultureInvariant   # tr-TR: I/i eşleşmesi kültürden bağımsız olsun
Get-ChildItem $DST -Recurse -File | Where-Object { $_.FullName -notmatch '\\.git\\' -and $textExt -contains $_.Extension.ToLower() } | ForEach-Object {
  $path = $_.FullName
  if (Select-String -LiteralPath $path -Pattern "Users[\\/]{1,2}$userLeaf" -Quiet) {
    $b = [System.IO.File]::ReadAllBytes($path)
    if ($b.Length -ge 2 -and $b[0] -eq 0xFF -and $b[1] -eq 0xFE) { $enc = New-Object System.Text.UnicodeEncoding($false, $true); $skip = 2 }
    elseif ($b.Length -ge 3 -and $b[0] -eq 0xEF -and $b[1] -eq 0xBB -and $b[2] -eq 0xBF) { $enc = New-Object System.Text.UTF8Encoding($true); $skip = 3 }
    else { $enc = New-Object System.Text.UTF8Encoding($false, $true); $skip = 0 }
    try { $t = $enc.GetString($b, $skip, $b.Length - $skip) }
    catch { $enc = [System.Text.Encoding]::GetEncoding(28591); $skip = 0; $t = $enc.GetString($b) }   # UTF-8 değilse baytları koruyan Latin-1
    $n = $t
    foreach ($pt in $pats) { $n = [regex]::Replace($n, $pt[0], $pt[1], $ro) }
    if ($n -ne $t) { [System.IO.File]::WriteAllBytes($path, [byte[]]($enc.GetPreamble() + $enc.GetBytes($n))); $nSan++; [void]$sanList.Add($path.Substring($DST.Length + 1)) }
  }
}
"kişisel yol temizlenen dosya: $nSan"
$sanLog = Join-Path $SRC "_arsiv\_commit\sync_last_sanitised.txt"   # depoya gitmez (_arsiv hariç tutulur)
if (Test-Path (Split-Path $sanLog -Parent)) { $sanList | Sort-Object | Set-Content -LiteralPath $sanLog -Encoding UTF8 }
$sanCode = @($sanList | Where-Object { $_ -match '\.(py|ps1)$' })
if ($sanCode.Count) { "UYARI: kişisel yol içeren kod dosyası (çalışma ağacında düzeltin): $($sanCode -join ', ')" }
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

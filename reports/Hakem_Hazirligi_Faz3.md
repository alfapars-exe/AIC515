# Hakem Hazırlığı (Lab Rat `rebuttal`) — IEEE bildirisi, Faz 3 sonrası

Hedef mekân: IEEE konferans (SIU 2027 / WIFS / ICIP tipi, 6+2 sayfa). Bildiri: `paper/main.pdf` (8 sayfa).

## Zayıflıklar

| # | Zayıflık | Tür | Önem | Durum / öneri |
|---|---|---|---|---|
| 1 | Piksel-değişim GT'si insan tarafından doğrulanmadı (yalnız VLM denetimi) | deneysel | yüksek | Sınırlama olarak yazıldı. **En ucuz güçlendirme:** `verification_sheet.csv` ile 100 çift insan denetimi (~45 dk) |
| 2 | Faz 3 tek-faktör deneyleri tek tohum (DeepLabV3+, işaretçi-onarımı, birleşik kayıp, 20 ep, YOLO ablasyonları) | istatistik | orta | Metinde her yerde etiketlendi. Gönderimden önce en kritik ikisi (işaretçi-onarımlı box, birleşik kayıp) için +2 tohum (~40 dk GPU) |
| 3 | Tek veri seti | genelleme | orta | v2 aynı kutu anotasyonunu kullanıyor; erişim açılırsa 613 görüntülük harici test bir günde eklenir |
| 4 | TruFor ince ayarı yok | taban çizgisi | orta | Eğitim kodu yayımlanmamış; ikinci mimari + kutu dedektörü ile mimari bağımsızlık gösterildi |
| 5 | Kutu AP'de kareli YOLO tam-görüntü YOLO'dan düşük (kenarda bölünen kutular) | yöntem | düşük | Metinde açıklandı; düzeltme: kareler arası kutu birleştirme (GREEDYNMM) — gelecek iş |
| 6 | Hız iddiası beklentiyi karşılamadı | sunum | düşük | Dürüstçe raporlandı (98 vs 81 ms) |
| 7 | Görüntü düzeyi tespit testte ölçülemiyor (test authentic'leri eşli değil) | deneysel | düşük | Açıklandı; val eşlerinde raporlandı |

## Ablasyon seçimi (Lab Rat `compare` mantığı: aynı protokol, tek faktör)

| Run | Değişen faktör | Ana metrik farkı | Anlatı gücü | Yer |
|---|---|---|---|---|
| Eğitim GT (box/pix/glyph) × 3 tohum | eğitim maskesi | box-GT 0.923 vs pix-GT 0.374 (aynı tahminler) | Ana iddia (H1) | Ana metin, Tablo I |
| DeepLabV3+ | dekoder | 0.939 → 0.382 | H1 mimariden bağımsız | Ana metin |
| İşaretçi-onarımlı eğitim | eğitim girdisi | onarım Δ −0.222 (3 tohum) → +0.019 (1 tohum); test AUPRC 0.950 → 0.939 | İşaretçi = kısayol, şart değil | Ana metin, Tablo II |
| YOLO11s (3 tohum) | formülasyon | test box AUPRC 0.969; pix 0.368 | Box metriği = alan tespiti | Ana metin, Tablo V |
| Render/içerik sondası | girdi | AUROC render 0.637 / içerik 0.733 (U-Net box, 3 tohum) | Mekanizma | Ana metin, Tablo III |
| Gemini-only × 3 tohum | eğitim üreticisi | Ideogram 0.412 vs 0.909 | H3 reddi | Ana metin, Tablo IV |
| Birleşik kayıp | kayıp | pix 0.610→0.622, blur −59→−43 % | Pratik öneri | Ana metin (kısa) |
| 20 epoch, YOLO n/m, varsayılan aug, authneg | eğitim ayrıntısı | ≈0 | "Yetersiz eğitim" itirazına yanıt | Kısa cümle / ek |

## En olası 10 hakem sorusu

1. **Q:** Your pixel-change GT may encode re-encoding noise rather than real edits. How do you know it is the "true" ground truth?
   *Yanıtlanabilir:* Evet. *Taslak:* We do not claim it is the true mask; the claim is that the choice of ground truth reverses the ranking. ε-sensitivity (global trace vanishes at ε ≥ 4), morphological cleaning and a glyph-restricted variant give the same ordering, and a VLM check rated 95/96 pairs OK or partial. A human audit is listed as a limitation.
2. **Q:** Is the box-vs-pixel effect specific to the U-Net? — *Evet, Tablo I:* DeepLabV3+ 0.939 → 0.382 and a YOLO11 detector 0.965 → 0.368.
3. **Q:** If the marker is removed, does the benchmark become harder or is your shortcut claim overstated? — *Evet, Tablo II:* retraining on repaired images keeps 0.939 test AUPRC but removes the marker dependence; we therefore call it a shortcut, not a prerequisite.
4. **Q:** Why should a reader care, given that a detector already solves the box task? — *Evet:* that is the point: box metrics measure field detection; pixel-change metrics are needed to measure localization of changed pixels.
5. **Q:** Many results are single-seed. — *Kısmen:* the main U-Net grid, held-out U-Nets and the main YOLO use three seeds; others are marked exploratory. Two extra seeds for the repaired and joint models can be added in the rebuttal period.
6. **Q:** Why not fine-tune TruFor or another IML model? — *Kısmen:* training code is not released; two architectures and a detector show architecture independence.
7. **Q:** Is the held-out-generator test powered? — *Kısmen:* 126 Ideogram images with group bootstrap CIs; the YOLO and U-Net gaps (0.19 vs 0.50) do not overlap.
8. **Q:** The tiled detector has lower box AP than the full-image one; is tiling a good choice? — *Evet:* tiling raises pixel coverage and residue-free AP (0.878 vs 0.756) but splits boxes at tile borders; a merging step is future work.
9. **Q:** Did you check for train/test leakage? — *Evet:* source-document group splits, a 740-image leakage-free test subset with identical results (0.949 vs 0.950).
10. **Q:** Does preprocessing (e.g., grayscale to remove the green marker) solve the problem? — *Evet:* grayscale costs 47–67 % AUPRC for every model, including marker-repaired ones, so colour carries evidence beyond the marker.

## Önerilen son düzenlemeler
- Özet ve girişte tek-tohum uyarısı bulunuyor; gönderimden önce 2. maddedeki +2 tohum eklenirse "suggestive" ifadeleri güçlendirilebilir.
- Hedef mekân 6 sayfa ise: Tablo III (render/içerik) ve §V-F'nin dayanıklılık paragrafı eke taşınabilir.

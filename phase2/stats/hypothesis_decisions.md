# Faz 3 — Hipotez karar tablosu (H1–H4), kaynak-belge-grubu bootstrap CI + Holm

Üretim: `stats_analysis.py` (B = 1000, seed = 0); tüm sayılar `stats_summary.csv` satırlarından alınmıştır.
Veri: Faz 2 görüntü-düzeyi CSV'leri (val 487 tam görüntü, test 812, 3 GT × 9 B-modeli + 3 gemini-only C-modeli). Seed'ler sabit faktör:
önce görüntü başına 3 seed ortalaması, sonra kaynak-belge-grubu (cluster) bootstrap; farklar EŞLEŞTİRİLMİŞ bootstrap (aynı grup örneklemi iki koşula).
Grup yapısı: val 487 görüntü / 469 grup (451 tekil, 18 ikili), test 812 / 795 grup. CI = %95 yüzdelik. p_boot = 2·min(P(Δ≤0), P(Δ≥0)), alt sınır 1/B;
Holm düzeltmesi her hipotez ailesinin önceden belirlenmiş karar kontrastları üzerinde (`holm_family` sütunu; H1: 12, H2: 21, H3: 14, H4: 3, EXTRA: 8 test).

**Dışlamalar (raporlanan):** val: n_pos=0 (boş GT) → F1/AP ortalamalarından dışlandı: GT ['glyph', 'pixel'], new_id [2946, 3018] (2 görüntü; metrikler zaten NaN). test: n_pos=0 (boş GT) → F1/AP ortalamalarından dışlandı: GT ['glyph', 'pixel'], new_id [723] (1 görüntü; metrikler zaten NaN). robustness: n_pos=0 (boş GT) → F1/AP ortalamalarından dışlandı: GT ['glyph', 'pixel'], new_id [2946, 3018] (2 görüntü; metrikler zaten NaN). boundary: n_pos=0 (boş GT) → F1/AP ortalamalarından dışlandı: GT ['glyph', 'pixel'], new_id [2946, 3018] (2 görüntü; metrikler zaten NaN). test: pixel-GT yalnız eşleşmiş authentic'i olan 89 çiftte mevcut; n_pos>0 olan 88 görüntü kullanıldı, 724 görüntü (GT yok/boş) dışlandı. test: glyph-GT yalnız eşleşmiş authentic'i olan 89 çiftte mevcut; n_pos>0 olan 88 görüntü kullanıldı, 724 görüntü (GT yok/boş) dışlandı.

| Hipotez | Önceden sabitlenmiş yanlışlama ölçütü | Gözlenen (CI) | Karar |
|---|---|---|---|
| H1 metrik/sıralama | ΔF1 < 0.05 VE Kendall τ ≥ 0.9 | ΔF1@τ* (bbox-GT − piksel-GT, bbox-modeli) = 0.398 [0.377, 0.418]; τ(bbox vs piksel, 9 model) = -0.22 [-0.28, 0.06] | **DOĞRULANDI** |
| H2 sınır shortcut'ı | İki model benzer düşer (fark CI 0'ı kapsar) VE benign FP ≈ authentic | boundary relΔAP bbox-modeli -1.5 % [-2.0, -1.1] vs piksel-modeli -3.0 % [-3.8, -2.0]; fark (bbox − piksel, kendi GT) = +1.4 % [+0.5, +2.4] (yön TERS); benign paste-back FP − authentic (bbox-modeli) = 0.50 [0.36, 0.66] ‰ | **KISMEN** |
| H3 transfer/dayanıklılık | Fark yok / ters (CI örtüşür) | Ideogram ΔAP piksel-modeli − bbox-modeli (piksel-GT, gemini-only) = 0.025 [-0.039, 0.088]; dayanıklılık: yön koşula göre değişiyor (JPEG +, blur −) | **YANLIŞLANDI** |
| H4 global iz | AUROC ≈ 0.5 | AUROC forged-vs-authentic (bbox-dışı crop) = 0.516 [0.513, 0.519]; simetrik resample = 0.498 [0.492, 0.503]; pozitif kontrol = 0.980 [0.972, 0.988] | **YANLIŞLANDI** (pratik olarak; bkz. not) |

---

## H1 — GT türü metrikleri ve model sıralamasını değiştirir → DOĞRULANDI

**Ölçüt.** Yanlışlama: aynı tahminler için ΔF1 (bbox-GT − piksel/glif-GT) < 0.05 VE Kendall τ ≥ 0.9.

**Gözlenen (val 487; piksel/glif-GT'de 485, 2 boş-GT görüntü dışlandı):**

| Model | Metrik | bbox-GT | piksel-GT | glif-GT | Δ bbox−piksel | Δ bbox−glif |
|---|---|---|---|---|---|---|
| bbox-GT-trained | AP | 0.896 [0.876, 0.916] | 0.377 [0.358, 0.397] | 0.282 [0.271, 0.294] | 0.522 [0.500, 0.544] (p_boot=0.001, p_Holm=0.012) | 0.617 [0.599, 0.635] (p_boot=0.001, p_Holm=0.012) |
| bbox-GT-trained | F1@τ* | 0.842 [0.820, 0.862] | 0.447 [0.427, 0.470] | 0.387 [0.374, 0.401] | 0.398 [0.377, 0.418] (p_boot=0.001, p_Holm=0.012) | 0.458 [0.443, 0.474] (p_boot=0.001, p_Holm=0.012) |
| pixel-GT-trained | AP | 0.588 [0.567, 0.608] | 0.579 [0.556, 0.604] | 0.536 [0.518, 0.557] | 0.012 [-0.006, 0.028] (p_boot=0.202, p_Holm=0.202) | 0.054 [0.035, 0.074] (p_boot=0.001, p_Holm=0.012) |
| pixel-GT-trained | F1@τ* | 0.600 [0.580, 0.619] | 0.577 [0.556, 0.600] | 0.532 [0.514, 0.550] | 0.026 [0.010, 0.041] (p_boot=0.001, p_Holm=0.012) | 0.071 [0.055, 0.086] (p_boot=0.001, p_Holm=0.012) |
| glyph-GT-trained | AP | 0.373 [0.358, 0.388] | 0.397 [0.378, 0.416] | 0.695 [0.674, 0.715] | -0.023 [-0.037, -0.009] (p_boot=0.001, p_Holm=0.012) | -0.320 [-0.337, -0.305] (p_boot=0.001, p_Holm=0.012) |
| glyph-GT-trained | F1@τ* | 0.401 [0.388, 0.414] | 0.457 [0.439, 0.475] | 0.674 [0.655, 0.692] | -0.054 [-0.068, -0.041] (p_boot=0.001, p_Holm=0.012) | -0.271 [-0.285, -0.257] (p_boot=0.001, p_Holm=0.012) |

**Sıralama kararlılığı (Kendall τ, per-image AP ortalamasına göre; `rank_kendall.csv`):**
* 9 model (3 eğitim GT × 3 seed): bbox-GT vs piksel-GT τ = -0.22 [-0.28, 0.06], P(τ ≥ 0.9) = 0.001, sırası değişen model: 9; bbox-GT vs glif-GT τ = -0.72 [-0.83, -0.61], P(τ ≥ 0.9) = 0.001, sırası değişen model: 8; piksel-GT vs glif-GT τ = 0.50 [0.22, 0.50], P(τ ≥ 0.9) = 0.001, sırası değişen model: 6.
* 3 model tipi (seed ort.): bbox vs piksel τ = -0.33, bbox vs glif τ = -1.00 (sıralama tamamen tersine döner: bbox-GT altında bbox > piksel > glif; glif-GT altında glif > piksel > bbox).
* İkincil metrikler (F1@0.5, IoU@τ*) aynı yönde (stats_summary.csv, aile H1, holm_family boş).

**Karar: DOĞRULANDI.** bbox-modeli için ΔF1@τ* = 0.398 [0.377, 0.418] ≫ 0.05 (CI alt sınırı 0.05'in çok üzerinde) ve τ < 0.9'un çok altında (CI üst sınırı 0.06); 9 modelin 9'unun sırası değişir. Yanlışlama ölçütünün iki koşulu da reddedilir. Not: piksel-modelinin AP'si bbox-GT ve piksel-GT altında neredeyse aynıdır (0.012 [-0.006, 0.028], p_boot=0.202, p_Holm=0.202): metrik şişmesi bbox-modeline özgüdür — bbox-GT, dikdörtgenin değişmeyen pikselini "doğru" sayar.

## H2 — bbox-modeli sınır/dikdörtgen kısayolu kullanır → KISMEN

**Ölçüt.** Yanlışlama: iki model de benzer düşer (modeller arası fark CI 0'ı kapsar) VE benign kontrollerde FP ≈ authentic.

**Sınır ablasyonu, relΔAP = (ablasyon − temiz)/temiz (val 487; eşleştirilmiş bootstrap):**

| Model (değerlendirme GT) | boundary_blur | placebo_ring | inner_blur |
|---|---|---|---|
| bbox-GT-trained (bbox GT, kendi GT) | -1.5 % [-2.0, -1.1] (p_boot=0.001, p_Holm=0.021) | +0.0 % [-0.1, +0.1] (p_boot=0.802, p_Holm=0.802) | +0.1 % [-0.9, +1.0] (p_boot=0.906) |
| pixel-GT-trained (pixel-change GT, kendi GT) | -3.0 % [-3.8, -2.0] (p_boot=0.001, p_Holm=0.021) | +0.0 % [-0.0, +0.1] (p_boot=0.314, p_Holm=0.678) | -7.2 % [-9.4, -4.9] (p_boot=0.001) |
| pixel-GT-trained (bbox GT) | -6.1 % [-7.1, -5.2] (p_boot=0.001) | +0.0 % [-0.0, +0.1] (p_boot=0.214) | +4.4 % [+1.7, +7.0] (p_boot=0.001) |
| glyph-GT-trained (glyph GT, kendi GT) | -3.4 % [-4.0, -2.8] (p_boot=0.001, p_Holm=0.021) | -0.0 % [-0.1, +0.0] (p_boot=0.226, p_Holm=0.678) | -10.5 % [-12.3, -9.0] (p_boot=0.001) |
| glyph-GT-trained (bbox GT) | -5.4 % [-6.3, -4.6] (p_boot=0.001) | -0.0 % [-0.1, +0.0] (p_boot=0.440) | -3.9 % [-5.9, -2.1] (p_boot=0.001) |

**Modeller arası fark (bbox-modeli − diğer; hipotez: < 0, yani bbox-modeli daha çok düşer):**
* boundary_blur, kendi GT'leri: bbox − piksel = +1.4 % [+0.5, +2.4] (p_boot=0.006, p_Holm=0.024); bbox − glif = +1.9 % [+1.4, +2.4] (p_boot=0.001, p_Holm=0.021).
* boundary_blur, ortak bbox-GT: bbox − piksel = +4.6 % [+3.6, +5.5] (p_boot=0.001, p_Holm=0.021); bbox − glif = +3.9 % [+3.0, +4.8] (p_boot=0.001, p_Holm=0.021).
* boundary − placebo (bbox-modeli, kendi GT): -1.5 % [-2.0, -1.1] (p_boot=0.001).

**Rect-fit son-işleme ΔF1@τ* (rectfit − raw):** bbox-modeli bbox-GT -0.019 [-0.026, -0.012] (p_boot=0.001, p_Holm=0.021); bbox-modeli piksel-GT -0.036 [-0.042, -0.030] (p_boot=0.001, p_Holm=0.021); piksel-modeli piksel-GT -0.183 [-0.199, -0.170]; glif-modeli glif-GT -0.482 [-0.501, -0.462].
**Tahmin dikdörtgenselliği @τ* (alan / kendi bbox alanı; seed-ortalaması, ortalama [CI]):** bbox-modeli 0.620 [0.590, 0.652], piksel-modeli 0.259 [0.234, 0.285], glif-modeli 0.106 [0.092, 0.120]; 3-seed havuzlanmış medyan: bbox 0.865, piksel 0.035, glif 0.014 (README T6: 0.877 / 0.038 / 0.013 ile yaklaşık uyumlu).

**Benign kontroller (val 487 kaynaklı; FP = pozitif piksel oranı @τ*(bbox-GT); 3 seed ort.):**

| Model | authentic FP ‰ | benign paste-back FP ‰ | Δ paste-back − authentic ‰ | full-image resample FP ‰ | Δ resample − authentic ‰ | görüntü payı >%0.1 poz. (authentic → paste-back) |
|---|---|---|---|---|---|---|
| bbox-GT-trained | 1.65 [1.21, 2.16] | 2.15 [1.67, 2.70] | 0.50 [0.36, 0.66] (p_boot=0.001, p_Holm=0.021) | 2.17 [1.69, 2.75] | 0.52 [0.41, 0.65] (p_boot=0.001, p_Holm=0.021) | 0.244 [0.205, 0.284] → 0.296 [0.255, 0.338] (Δ 0.051 [0.029, 0.073]) |
| pixel-GT-trained | 1.99 [1.39, 2.70] | 2.22 [1.61, 2.93] | 0.22 [0.16, 0.29] (p_boot=0.001, p_Holm=0.021) | 2.47 [1.84, 3.20] | 0.48 [0.40, 0.57] (p_boot=0.001, p_Holm=0.021) | 0.281 [0.239, 0.323] → 0.333 [0.291, 0.378] (Δ 0.051 [0.032, 0.073]) |
| glyph-GT-trained | 3.80 [2.87, 4.88] | 3.98 [3.06, 5.07] | 0.18 [0.14, 0.23] (p_boot=0.001, p_Holm=0.021) | 4.17 [3.21, 5.26] | 0.36 [0.31, 0.42] (p_boot=0.001, p_Holm=0.021) | 0.341 [0.300, 0.385] → 0.409 [0.363, 0.453] (Δ 0.068 [0.047, 0.090]) |

**Görüntü-düzeyi AUROC (top-1 % skor; forged val vs kontrol; seed başına AUROC → 3 seed ort.):**

| Model | vs authentic | vs benign paste-back | vs full-image resample | Δ(authentic − paste-back) | Δ(authentic − resample) | vs authentic, Gemini / Ideogram |
|---|---|---|---|---|---|---|
| bbox-GT-trained | 0.892 [0.876, 0.907] (p_boot=0.001, p_Holm=0.021) | 0.872 [0.855, 0.888] | 0.866 [0.848, 0.885] | 0.020 [0.015, 0.025] | 0.025 [0.021, 0.030] | 0.906 [0.890, 0.921] / 0.701 [0.651, 0.762] |
| pixel-GT-trained | 0.892 [0.874, 0.909] (p_boot=0.001, p_Holm=0.021) | 0.876 [0.857, 0.894] | 0.863 [0.844, 0.883] | 0.016 [0.012, 0.020] | 0.028 [0.023, 0.034] | 0.903 [0.885, 0.920] / 0.748 [0.679, 0.824] |
| glyph-GT-trained | 0.799 [0.778, 0.823] (p_boot=0.001, p_Holm=0.021) | 0.779 [0.758, 0.802] | 0.769 [0.747, 0.792] | 0.020 [0.016, 0.026] | 0.030 [0.025, 0.036] | 0.808 [0.787, 0.833] / 0.695 [0.627, 0.775] |

Not: README T7'deki AUROC_top1 değerleri (bbox 0.838 / piksel 0.896 / glif 0.814) yalnız s44 ile uyuşur; burada 3 seed ortalaması raporlanır (s42/s43/s44 bbox-modeli: 0.921 / 0.916 / 0.838).

**Karar: KISMEN.**
* Önceden sabitlenmiş *karşılaştırmalı* ölçüt başarısız: bbox-modelinin boundary_blur düşüşü piksel-/glif-modelininkinden **küçüktür** ve fark CI'ı 0'ı kapsamaz (yön ters, Holm sonrası anlamlı). "Benzer düşüş" yanlışlama koşulu da sağlanmaz (fark ≠ 0) — yani hipotezin yönsel iddiası yanlışlanmış, ama yanlışlama ölçütünün kendisi (benzer düşüş VE FP ≈ authentic) literal olarak tetiklenmemiştir.
* Shortcut lehine kanıt: bbox-modeli yalnız sınır halkasına duyarlıdır (boundary -1.5 % [-2.0, -1.1]; plasebo ≈ 0; bbox İÇİ blur +0.1 % [-0.9, +1.0] — içerik silinince skoru değişmez), tahminleri dikdörtgenseldir (medyan 0.87 vs 0.04 / 0.01); piksel-/glif-modelleri ise içeriğe duyarlıdır (inner_blur −7 % / −11 %). Duyarlılık *profili* (kenar-evet / içerik-hayır) shape-prior'u destekler; mutlak sınır düşüşünün küçük kalması 5 px'lik halkanın tavana yakın (AP 0.90) bir modelde sınırlı etkisiyle açıklanabilir, bu Faz 2 deneyiyle ayrıştırılamaz.
* Benign FP: paste-back ve resample her modelde FP'yi anlamlı artırır (Holm sonrası p = 0.021), ancak artış bbox-modeline özgü değildir (piksel/glif-modellerinde de +0.2–0.5 ‰); görüntü-düzeyi AUROC 2–3 puan düşer. Yani "benign FP ≈ authentic" koşulu sağlanmaz, fakat bu bulgu H2'nin bbox-özgü iddiasını ayırt etmez.
* Metrik artefaktı (H1 ile ilişkili): piksel-modelinde bbox içini bulanıklaştırmak bbox-GT AP'sini **yükseltir** (+4.4 % [+1.7, +7.0]) ve piksel-GT AP'sini düşürür (-7.2 % [-9.4, -4.9]): bbox-GT, içeriği silmeyi ödüllendirir.

## H3 — piksel/glif-GT daha iyi transfer ve dayanıklılık sağlar → YANLIŞLANDI

**Ölçüt.** Yanlışlama: fark yok / ters (CI örtüşür).

**Dayanıklılık (val 487; ΔAP = bozulmuş − temiz; her model kendi GT'sinde ve ortak piksel-GT'de):**

| Model (GT) | JPEG Q75 | 0.75× rescale | blur σ=1 |
|---|---|---|---|
| bbox-GT-trained (bbox GT, kendi GT) | -0.126 [-0.145, -0.106] | -0.063 [-0.077, -0.050] | -0.095 [-0.114, -0.078] |
| bbox-GT-trained (pixel-change GT) | -0.042 [-0.053, -0.030] | -0.027 [-0.035, -0.020] | -0.040 [-0.051, -0.031] |
| pixel-GT-trained (pixel-change GT, kendi GT) | -0.105 [-0.120, -0.090] | -0.071 [-0.083, -0.060] | -0.111 [-0.126, -0.097] |
| glyph-GT-trained (glyph GT, kendi GT) | -0.103 [-0.119, -0.086] | -0.105 [-0.120, -0.091] | -0.157 [-0.174, -0.142] |
| glyph-GT-trained (pixel-change GT) | -0.061 [-0.073, -0.049] | -0.050 [-0.061, -0.041] | -0.069 [-0.082, -0.057] |

**Düşüş farkı (diğer model − bbox-modeli; hipotez > 0 = piksel/glif-modeli daha az düşer; Holm ailesi H3):**

| Kontrast | JPEG Q75 | 0.75× rescale | blur σ=1 |
|---|---|---|---|
| pixel-model (kendi GT) − bbox-model (kendi GT) | 0.021 [0.007, 0.034] (p_boot=0.004, p_Holm=0.020) | -0.007 [-0.018, 0.003] (p_boot=0.144, p_Holm=0.432) | -0.016 [-0.029, -0.002] (p_boot=0.016, p_Holm=0.064) |
| pixel-model − bbox-model (ikisi de piksel-GT) | -0.063 [-0.074, -0.053] (p_boot=0.001, p_Holm=0.014) | -0.044 [-0.055, -0.034] (p_boot=0.001, p_Holm=0.014) | -0.071 [-0.084, -0.058] (p_boot=0.001, p_Holm=0.014) |
| glyph-model (kendi GT) − bbox-model (kendi GT) | 0.023 [0.013, 0.034] (p_boot=0.001, p_Holm=0.014) | -0.042 [-0.054, -0.031] (p_boot=0.001, p_Holm=0.014) | -0.062 [-0.075, -0.048] (p_boot=0.001, p_Holm=0.014) |
| glyph-model − bbox-model (ikisi de piksel-GT) | -0.019 [-0.028, -0.010] (p_boot=0.001, p_Holm=0.014) | -0.023 [-0.033, -0.013] (p_boot=0.001, p_Holm=0.014) | -0.029 [-0.040, -0.018] (p_boot=0.001, p_Holm=0.014) |

**Held-out generator (gemini-only C modelleri, 1 seed; Ideogram: bbox-GT için val 41 + test 85 = 126, piksel/glif-GT için val 41; kaynak-eşlenmiş Gemini = cord/sroie/xfund, val+test):**

| Model (GT) | Ideogram AP | kaynak-eşlenmiş Gemini AP | açık (Gemini − Ideogram) |
|---|---|---|---|
| bbox-GT-trained (bbox GT, kendi GT) | 0.395 [0.325, 0.468] (n=126) | 0.908 [0.889, 0.925] (n=636) | 0.513 [0.439, 0.590] (p_boot=0.001) |
| pixel-GT-trained (pixel-change GT, kendi GT) | 0.279 [0.175, 0.373] (n=41) | 0.582 [0.548, 0.612] (n=249) | 0.303 [0.200, 0.407] (p_boot=0.001) |
| pixel-GT-trained (bbox GT) | 0.255 [0.205, 0.305] (n=126) | 0.559 [0.541, 0.575] (n=636) | 0.304 [0.253, 0.356] (p_boot=0.001) |
| glyph-GT-trained (glyph GT, kendi GT) | 0.295 [0.206, 0.384] (n=41) | 0.717 [0.682, 0.748] (n=249) | 0.422 [0.319, 0.521] (p_boot=0.001) |
| glyph-GT-trained (bbox GT) | 0.199 [0.157, 0.239] (n=126) | 0.394 [0.379, 0.407] (n=636) | 0.195 [0.150, 0.237] (p_boot=0.001) |

* Ideogram'da modeller arası ΔAP (aynı görüntüler, eşleştirilmiş): piksel − bbox (piksel-GT, n=41) = 0.025 [-0.039, 0.088] (p_boot=0.432, p_Holm=0.864); glif − bbox (piksel-GT) = -0.021 [-0.091, 0.047] (p_boot=0.558, p_Holm=0.864); piksel − bbox (bbox-GT, n=126) = -0.140 [-0.193, -0.090]; glif − bbox (glif-GT, n=41) = 0.125 [0.049, 0.206].
* Açık farkı (bbox-modeli açığı − piksel-modeli açığı, kendi GT'leri) = 0.210 [0.103, 0.320] (p_boot=0.001): bbox-modelinin göreli düşüşü daha büyüktür, ama bu kendi GT'lerinde ölçülen farklı ölçeklerdeki değerlerin farkıdır; ortak GT'de Ideogram AP'si piksel-modeli lehine değildir.
* B modelleri (tüm araçlar, 3 seed) için aynı tablo stats_summary.csv'de (aile H3, 'B: all tools').

**Karar: YANLIŞLANDI.** Held-out: ortak piksel-GT altında piksel-/glif-modeli Ideogram'da bbox-modelinden yüksek AP vermez (CI'lar 0'ı kapsar, Holm p ≈ 0.86). Dayanıklılık: kendi GT'lerinde piksel-modeli JPEG'de daha az (0.021 [0.007, 0.034]), yeniden-ölçeklemede farksız, blur'da daha çok düşer; glif-modeli JPEG hariç daha çok düşer; ortak piksel-GT'de her iki model de her koşulda bbox-modelinden daha çok düşer (bbox-modelinin piksel-GT AP'si zaten düşük → taban etkisi). Tutarlı bir dayanıklılık avantajı yok → yanlışlama ölçütü ("fark yok/ters") sağlanır. Sınır: C modelleri tek seed; piksel/glif-GT Ideogram alt kümesi n = 41 (geniş CI; ~±0.07 AP'lik farklar tespit edilemez).

## H4 — bbox dışında öğrenilebilir global iz → YANLIŞLANDI (pratik olarak)

**Ölçüt.** Yanlışlama: AUROC ≈ 0.5. Hipotez: bbox-dışı crop forged-vs-authentic AUROC > 0.5 ve simetrik (resample) koşulda düşüş.

| Prob | Koşul | n çift | AUROC [CI] | p (null 0.5) |
|---|---|---|---|---|
| outside-bbox crops (H4) | forged vs authentic | all, n=457 | 0.516 [0.513, 0.519] | p_boot=0.001, p_Holm=0.003 |
| outside-bbox crops (H4) | forged vs fullimage_resample | all, n=457 | 0.498 [0.492, 0.503] | p_boot=0.336, p_Holm=0.336 |
| outside-bbox crops (H4) | forged vs authentic | tool=gemini, n=416 | 0.516 [0.514, 0.520] | p_boot=0.001 |
| outside-bbox crops (H4) | forged vs authentic | tool=ideogram, n=41 | 0.512 [0.500, 0.532] | p_boot=0.028 |
| outside-bbox crops (H4) | forged vs authentic | source=cord, n=121 | 0.500 [0.500, 0.500] | p_boot=1.000 |
| outside-bbox crops (H4) | forged vs authentic | source=sroie, n=119 | 0.521 [0.514, 0.533] | p_boot=0.001 |
| outside-bbox crops (H4) | forged vs authentic | source=wildreceipt, n=168 | 0.529 [0.525, 0.538] | p_boot=0.001 |
| outside-bbox crops (H4) | forged vs authentic | source=xfund, n=49 | 0.500 [0.500, 0.500] | p_boot=1.000 |
| positive control (bbox-centred crops) | forged vs authentic | all, n=476 | 0.980 [0.972, 0.988] | p_boot=0.001 |
| positive control (bbox-centred crops) | forged vs fullimage_resample | all, n=476 | 0.975 [0.966, 0.984] | p_boot=0.001 |
| positive control (bbox-centred crops) | forged vs authentic | tool=gemini, n=435 | 0.984 [0.977, 0.990] | p_boot=0.001 |
| positive control (bbox-centred crops) | forged vs authentic | tool=ideogram, n=41 | 0.939 [0.892, 0.984] | p_boot=0.001 |
| positive control (bbox-centred crops) | forged vs authentic | source=cord, n=121 | 0.949 [0.920, 0.971] | p_boot=0.001 |
| positive control (bbox-centred crops) | forged vs authentic | source=sroie, n=119 | 1.000 [0.999, 1.000] | p_boot=0.001 |
| positive control (bbox-centred crops) | forged vs authentic | source=wildreceipt, n=187 | 0.989 [0.981, 0.995] | p_boot=0.001 |
| positive control (bbox-centred crops) | forged vs authentic | source=xfund, n=49 | 0.971 [0.935, 0.996] | p_boot=0.001 |

* ΔAUROC(authentic − resample), bbox-dışı crop = 0.018 [0.014, 0.023] (p_boot=0.001, p_Holm=0.003).
* Çift-içi doğruluk p(forged) > p(negatif), eşitlik = başarısız: vs authentic 0.396 [0.349, 0.441]; 457 çiftin 260'inde skorlar özdeş (crop'lar piksel-özdeş: cord/xfund'da bbox dışı değişmiyor).
* **Post-hoc (keşifsel, önceden kayıtlı değil):** skorları farklı 197 çiftte (sroie/wildreceipt) forged, authentic'ten **0.92 [0.88, 0.96]** oranında yüksek skor alır; simetrik yeniden-örneklenmiş negatife karşı ise 0.48 [0.43, 0.53] (n=426). Yani bbox dışında değişen piksellerde çift-içi ayırt edilebilir bir **yeniden-kodlama/yeniden-örnekleme izi** vardır, ancak bu iz (i) sahtecilik-özgü değildir (simetrik kontrolde kaybolur) ve (ii) görüntüler arası varyansın içinde kaybolur (görüntü-düzeyi AUROC 0.516). CI'ların darlığı (±0.003) çift-düzeyi yeniden örneklemeden kaynaklanır: 260 özdeş çift her örneklemde tam 0.5 katkı verir.

**Karar: YANLIŞLANDI (pratik olarak).** Görüntü-düzeyi AUROC 0.516 [0.513, 0.519] istatistiksel olarak 0.5'ten ayrışır (Holm p = 0.003) ve simetrik koşulda 0.5'e iner — hipotezin iki yönsel öngörüsü de işaret olarak tutar — fakat etki büyüklüğü (+0.016) "AUROC ≈ 0.5" yanlışlama ölçütünün içindedir ve kullanılabilir bir global iz yoktur; pozitif kontrol (0.980 [0.972, 0.988]) pipeline'ın öğrenebildiğini gösterir. Post-hoc çift-içi bulgu, veri kümesi inşasına ilişkin bir uyarıdır (eşleştirilmiş authentic negatiflerle eğitilen bir model yeniden-kodlama asimetrisini öğrenebilir); H4'ün "sahtecilik izi" iddiasını desteklemez.

## Ek kontrastlar

**Test sızıntısı (bbox-modeli, bbox-GT, test):** AP tam 812 = 0.908 [0.894, 0.922], temiz 740 = 0.907 [0.891, 0.922], temiz-olmayan 72 = 0.920 [0.879, 0.959]; Δ(812 − 740) = 0.0012 [-0.0029, 0.0048] (p_boot=0.524, p_Holm=1.000); Δ(72 − 740) = 0.014 [-0.032, 0.055] (p_boot=0.524, p_Holm=1.000). F1@τ*: Δ(812 − 740) = 0.0008 [-0.0045, 0.0055] (p_boot=0.692, p_Holm=1.000). → Sızıntı şüpheli 72 görüntünün etkisi ölçülemeyecek kadar küçük; "tam 812" ve "temiz 740" sonuçları istatistiksel olarak ayırt edilemez.

**Zero-shot literatür baseline'ları (test 812, bbox-GT; eşit koşul DEĞİL — U-Net AIForge-Doc train'de eğitildi, TruFor/ADCD-Net eğitilmedi, çözünürlük protokolleri farklı):**

| Baseline | per-image AP | F1@0.5 | Δ bbox-modeli − baseline (AP) | Δ (F1@0.5) |
|---|---|---|---|---|
| TruFor (yerel çözünürlük) | 0.443 [0.416, 0.474] | 0.403 [0.378, 0.430] | 0.465 [0.436, 0.492] (p_boot=0.001, p_Holm=0.008) | 0.435 [0.409, 0.459] |
| ADCD-Net (Q100, OCR-proxy) | 0.283 [0.260, 0.308] | 0.182 [0.159, 0.203] | 0.625 [0.598, 0.652] (p_boot=0.001, p_Holm=0.008) | 0.656 [0.630, 0.682] |
| TruFor (1024 pad, makale protokolü) | 0.193 [0.172, 0.216] | 0.134 [0.118, 0.152] | 0.715 [0.691, 0.738] (p_boot=0.001) | 0.703 [0.684, 0.724] |
| ADCD-Net (Q100, boş OCR) | 0.283 [0.260, 0.308] | 0.179 [0.157, 0.200] | 0.625 [0.598, 0.652] (p_boot=0.001) | 0.658 [0.633, 0.684] |
| ADCD-Net (Q95) | 0.212 [0.192, 0.232] | 0.114 [0.098, 0.132] | 0.696 [0.671, 0.721] (p_boot=0.001) | 0.723 [0.699, 0.747] |
| ADCD-Net (Q100, hizalama kapalı) | 0.194 [0.175, 0.213] | 0.100 [0.084, 0.117] | 0.714 [0.690, 0.739] (p_boot=0.001) | 0.737 [0.714, 0.760] |

bbox-modeli test AP = 0.908 [0.894, 0.922]. Farklar bbox-GT ile ölçülmüştür; H1'e göre bu metrik bbox-modelini kayırır — piksel-GT ile adil karşılaştırma yalnız 88 eşleşmiş test çiftinde mümkündür (stats_summary'de yok; Faz 2 README T2'de özet).

## Hesaplanamayan / sınırlı kontrastlar
* Held-out Ideogram'da piksel/glif-GT metrikleri yalnız val'deki 41 görüntüde (test'in 89 çiftinde Ideogram yok); C modelleri tek seed → seed belirsizliği CI'lara dahil değil.
* reencode_png kontrolü authentic ile piksel-özdeş olduğundan ayrıca hesaplanmadı (dejenere).
* Test'te piksel/glif-GT yalnız 88 görüntüde; test düzeyinde H1 kontrastı tekrarlanmadı (val'deki 485 görüntü birincil).
* Bootstrap, seed'ler arası varyansı içermez (seed sabit faktör olarak ortalandı; 3 seed'lik std README tablolarında).

## Figürler
* figures/fig_H1_dualgt_ci.png — 3 model × 3 GT, per-image AP ve F1@τ* (grup-bootstrap %95 CI); Kendall τ.
* figures/fig_H2_ablation_ci.png — boundary / placebo / inner relΔAP, model başına CI (kendi GT ve ortak bbox-GT).
* figures/fig_H2_controls_ci.png — benign kontrollerde FP piksel oranı ve görüntü-düzeyi AUROC, CI.
* figures/fig_H3_transfer_ci.png — degradasyon ΔAP (kendi GT / piksel-GT) ve held-out Ideogram–Gemini açığı (gemini-only modeller).
* figures/fig_H4_auroc.png — H4 AUROC'ları (CI) ve post-hoc çift-içi doğruluk.

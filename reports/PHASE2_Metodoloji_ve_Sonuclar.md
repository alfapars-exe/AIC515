# PHASE 2 — Metodoloji, Deneysel Protokol, Sonuçlar ve Yayın Planı

> **ERRATA (9 Kasım 2026, Faz 3 incelemesi — bu belge tarihsel kayıttır; güncel ve bağlayıcı metin `paper/` altındaki IEEE bildirisi ve `phase3/results/tables_p3.md`'dir):**
> 1. **Faz 1 Pixel-AUPRC şişkin:** crop başına ≤20k negatif alt-örneklemeyle hesaplanmıştı (0.947 / 0.939). Tüm piksellerle kesin değer **0.805 / 0.792** (`phase3/results/phase1_exact_auprc.csv`). Faz 2 sayıları zaten kesindir.
> 2. **Tek tohum etiketleri:** "box modeli authentic val görüntülerinin %44'ünü işaretler" değeri yalnız s44'e aittir; 3 tohum ortalaması **%26** (>0.1 % piksel @0.5). Görüntü-düzeyi AUROC'lar (0.892 / 0.892 / 0.799) 3 tohum ortalamasıdır ve doğrudur. Tablo 9d'deki değerler 3 tohum ortalamasıyla değiştirilmelidir.
> 3. **GenD (Yermakov ve ark., WACV 2026):** iddianın yönü ters aktarılmıştı; 56.8 → 90.0 sayıları kaynağa yer gösterilerek doğrulanamadığı için kullanılmamalıdır.
> 4. **"İnsan doğrulaması":** İlk GT denetimi VLM destekli tutarlılık denetimidir. 10 Kasım 2026'da aynı 100 çift yazar tarafından elle denetlenmiştir (tek değerlendirici): piksel-değişim GT 36 uygun / 64 kısmen uygun / 0 hatalı, resmî kutu 99/100 uygun (`phase2/gt/human_verification_summary.md`). Bu belgedeki "insan doğrulaması bekliyor / yapılmadı" ifadeleri bu tarihten öncesine aittir.
> 5. **Kaynakça:** doğrulanmış güncel liste `literature/references_v3.bib` (77 kayıt; Helali 2026, DocDet, Shang ve ark. 2026 ve klasik kaynaklar eklendi; 5 kayıt düzeltildi).
> 6. **İşaretçi atfı:** AIForge-Doc v1 makalesi §3.4 (Gemini: 3 px parlak yeşil dikdörtgen) — doğrulandı.


**Proje:** AIC515 Görüntü İşleme — AI-Generated Financial Document Forgery Localization (AIForge-Doc v1)
**Araştırma yönü (PHASE 1 §H):** *What Do Document Forgery Localizers Learn from Box Masks? Pixel-Change Ground Truth and Shortcut Audits for AI-Inpainted Financial Documents*
**Tarih:** 20 Ekim 2026 · **Donanım:** RTX 5080 16 GB, yerel · **Toplam GPU eğitim süresi:** 124 dk (12 U-Net) + ~2 dk (H4 ResNet-18)

Bu rapor PHASE 1 raporunun devamıdır; literatür, gap analizi ve hipotezler için `reports/PHASE1_Arastirma_Raporu.md`'ye bakınız. Tüm sayılar `phase2/experiments/results/*.csv`, `phase2/gt/gt_stats.json`, `phase2/controls/controls_summary.json`, `phase2/literature_baselines/zero_shot_results.csv` ve `phase2/stats/stats_summary.csv` dosyalarından alınmıştır.

## 1. CONDITIONAL GO koşullarının durumu
| Koşul (PHASE 1 §H) | Durum |
|---|---|
| 1. Küçük-veri baseline çalışıyor, U-Net şans üstü | ✓ Sağlandı (PHASE 1) |
| 2. Piksel-değişim GT türetimi + ε-duyarlılık + 100 çift insan doğrulaması | ✓ Türetim ve ε-duyarlılık tamamlandı (7 varyant, 3.336 çift); **insan doğrulaması bekliyor** (`phase2/gt/verification_grid.pdf`, `verification_sheet.csv`) |
| 3. Benign paste-back kontrol seti + FP ölçümü | ✓ Sağlandı (487 × 3 kontrol tipi; FP ve image-level AUROC ölçüldü) |
| 4. Temiz test alt kümesi (740) + group-aware negatifler | ✓ Sağlandı (PHASE 1 `splits.json`) |

Karar: **GO** (insan doğrulaması sonuçları raporun bir sonraki sürümüne eklenecek; otomatik tutarlılık bayrakları §2.2'de).

## 2. Yeni metodoloji tasarımı

### 2.1 Piksel-değişim ve glif GT türetimi (`phase2/gt/make_gt.py`)
D = max_kanal |forged − authentic|. Varyantlar ve tam-set istatistikleri (n = 3.336 çift; medyan bbox-kapsama / IoU(bbox)): pixel_eps4 0.666/0.620 · **pixel_eps8** 0.492/0.481 · pixel_eps16 0.354/0.351 · **pixel_eps8_morph** (3×3 açma, <8 px bileşen filtresi, 3×3 kapama) 0.351/0.350 · glyph 0.293/0.276 · **glyph_v2** (polarite: bbox içi Sauvola-koyu oranı > 0.5 ise ters; spec'teki medyan-parlaklık kuralı 199 loş fotoğrafta metnin tümleyenini seçiyordu) 0.286/0.269 · bbox_and_eps8 0.492/0.492.
**ε-duyarlılık:** bbox-dışı piksellerin > %1'i değişen çift oranı ε=0'da %47.8 → ε=2 %17.2 → ε=4 %4.7 → ε=8 %1.3 → ε=16 %0.2; yani ε=0'daki global ±1–3 yeniden-kodlama izi ε ≥ 4'te kaybolur, ancak ≥1 bbox-dışı pikselli çift oranı ε=8'de hâlâ %33.5 (yerel, gerçek değişim). Bbox-dışı değişim Gemini %36.2 vs Ideogram %11.9; kaynak: WildReceipt %60.4, SROIE %32.2, CORD %0.4, XFUND %0.0.
**Veri kusurları (eğitimden dışlandı, tüm koşullarda aynı):** 10 dejenere çift (D_max ≤ 4) + 5 "tüm değişim bbox dışında" çift + 2 pair_accepted=False → train 2.748.
**Eğitimde kullanılan GT'ler:** bbox (resmî), pixel_eps8_morph ("piksel"), glyph_v2 ("glif").

![ε duyarlılık](phase2/gt/fig_eps_sensitivity_pub.png)

### 2.2 İnsan doğrulama örneklemi (100 çift, 7 tabaka)
Otomatik bayraklar: 28 çiftte ≥1 bbox-dışı D>8 px; 15'inde eps8 pozitiflerinin > %1'i bbox dışında; 3'ünde morfolojik filtre sonrası bbox-dışı bileşen; 7'sinde eps8 bbox'ın < %20'sini kaplıyor; 0 boş glif; 1 polarity_inverted_v2. `human_verdict` sütunu doldurulduğunda uyum oranı buraya eklenecek.

### 2.3 Benign kontrol setleri (`phase2/controls/make_controls.py`, 487 val authentic)
| Kontrol | D_max medyan [IQR] | D>8 piksel medyan | bbox-dışı D>8 | Yorum |
|---|---|---|---|---|
| Gerçek sahte (referans) | 180 [145–232] | 3.020 | %30.8 görüntüde | — |
| benign_pasteback (crop 2× LANCZOS up/down, yalnız bbox geri yapıştırma) | 17 [10–28] | 72 [3–348] | 0 | Gerçek sahtenin ~1/42'si; yalnız bbox içi |
| fullimage_resample (tüm görüntü 2× up/down) | 27 [17–42] | 2.627 [331–12.166] | medyan 2.466 | Piksel SAYISI gerçek sahteye yakın, genlik düşük; tüm metin kenarlarına dağılmış |
| reencode_png | 0 | 0 | 0 | Authentic ile piksel-özdeş (null kontrol) |

### 2.4 Shortcut probeları (`phase2/experiments/probes.py`)
Sınır ablasyonu (bbox kenarı 5 px halka, σ=1.5) + plasebo halka + iç-bölge blur; rectangle-fit son-işleme; tahmin dikdörtgenselliği; benign kontrollerde FP ve image-level skor; H4 global-iz probu (ResNet-18, maskeye değmeyen 224 crop forged vs aynı-koordinat authentic; pozitif kontrol: bbox-merkezli crop).

## 3. Deneysel protokol
- **Model:** smp.Unet(resnet18, ImageNet) RGB, 14.3 M parametre; 512 px natif crop (%70 GT-pozitif içeren, %30 rastgele), BCE+Dice, AdamW 1e-4, batch 16, bf16, 2 crop/görüntü/epoch, **10 epoch** (ön kayıtlı; hiçbir run erken durmadı), seed 42/43/44. Epoch ≈ 37 s eğitim + 22 s val; run ≈ 10–11 dk; peak GPU 4.66 GiB.
- **Koşullar:** 3 eğitim GT × 3 seed (B) + 3 GT × Gemini-only (C, 1 seed) = 12 run; checkpoint'ler `models/phase2/`.
- **Çıkarım:** tam görüntü sliding-window 512 / stride 448 / örtüşme ortalaması (uzun kenar > 4096 olan 15 val + 29 test görüntüsü 4096'ya küçültülüp harita geri büyütüldü).
- **Metrikler:** 256-bin histogramla **tüm pikseller** üzerinde kesin Pixel-AUPRC/AUROC (alt-örnekleme yok; sklearn ile doğrulandı), per-image AP, F1/IoU@0.5, F1/IoU@τ* (τ* val'de 0.05–0.95 ızgarasından), Oracle-F1; üç GT ile ayrı ayrı; tool/kaynak/alan-dilimi kırılımı.
- **Test:** resmî 812 (bbox-GT) + temiz 740 + 89 çiftte piksel/glif GT. **Zero-shot referanslar:** TruFor (resmî ckpt, ≤2 MP), ADCD-Net (MIT, JPEG Q100 yeniden kodlama, 512 uzun kenar) — aynı değerlendirme koduyla, ayrı tabloda.
- **İstatistik:** kaynak-belge grubu bootstrap (B=1000) %95 CI, seed'ler sabit faktör, Holm düzeltmesi (§5).

## 4. Sonuçlar (özet tablolar; tam tablolar `phase2/experiments/README.md`)

### 4.1 H1 — Aynı tahminler, üç GT (val 487, seed ort ± std)
| Eğitim GT \ Değerlendirme GT | bbox AUPRC / F1@τ* | piksel AUPRC / F1@τ* | glif AUPRC / F1@τ* |
|---|---|---|---|
| bbox | **0.923 ± 0.015** / **0.842 ± 0.023** | 0.374 ± 0.008 / 0.447 ± 0.011 | 0.257 ± 0.007 / 0.387 ± 0.009 |
| piksel (eps8_morph) | 0.579 ± 0.014 / 0.600 ± 0.014 | **0.610 ± 0.012** / **0.577 ± 0.007** | 0.511 ± 0.010 / 0.532 ± 0.008 |
| glif (glyph_v2) | 0.351 ± 0.015 / 0.401 ± 0.007 | 0.379 ± 0.012 / 0.457 ± 0.006 | **0.676 ± 0.019** / **0.674 ± 0.011** |

Her GT kendi modelini birinci sıraya koyar; bbox modelinin ΔF1 (bbox − piksel) = 0.40 ≫ 0.05 yanlışlama sınırı. Pixel-AUROC çoğu hücrede > 0.9 (bbox modeli piksel-GT altında 0.988): AUROC 1:100 dengesizliğe duyarsızdır, tek başına raporlanmamalıdır. **H1 DOĞRULANDI.**

![Çift GT](phase2/experiments/figures/fig_dual_gt_metrics.png)

### 4.2 Resmî test ve zero-shot referanslar (bbox-GT)
| Model | Alt küme | AUPRC | F1@τ* | IoU@τ* | Oracle-F1 |
|---|---|---|---|---|---|
| U-Net bbox-GT (3 seed) | tam 812 | 0.950 ± 0.011 | 0.846 ± 0.015 | 0.789 ± 0.018 | 0.879 ± 0.011 |
| U-Net bbox-GT | temiz 740 | 0.949 ± 0.012 | 0.845 ± 0.015 | 0.788 ± 0.018 | 0.878 ± 0.012 |
| U-Net piksel-GT | tam 812 | 0.638 ± 0.015 | 0.616 ± 0.013 | 0.480 ± 0.015 | 0.624 ± 0.015 |
| U-Net glif-GT | tam 812 | 0.403 ± 0.011 | 0.405 ± 0.007 | 0.266 ± 0.006 | 0.410 ± 0.008 |
| TruFor zero-shot (≤2 MP) | tam 812 | havuz 0.080 | F1@0.5 0.403 | 0.328 | 0.456 |
| ADCD-Net zero-shot (Q100) | tam 812 | havuz 0.130 | F1@0.5 0.182 (τ* 0.244) | 0.149 | 0.328 |

Sızıntı etkisi yok (tam vs temiz fark ≤ 0.001). 89 çiftte piksel-GT: bbox modeli 0.427, piksel modeli 0.654 (val örüntüsü tekrarlanır). TruFor değerleri makaledeki 0.434 / 0.358'e yakın (−0.03).

### 4.3 H2 — Bbox modeli ne öğreniyor?
- **Dikdörtgensellik** (tahmin alanı / tahmin-bbox alanı, medyan): bbox 0.87 (1 bileşen) · piksel 0.32 (7) · glif 0.05 (27).
- **Ablasyon (kendi GT'si):** bbox modeli — sınır blur relΔAUPRC −2.0 ± 0.4 % (per-image AP −1.5 % [−2.0, −1.1]), plasebo 0.0 %, iç blur +1.2 % AUPRC / +0.1 % AP (içerik dokusuna duyarsız). Piksel modeli — sınır −6.4 % AUPRC / −3.0 % AP, iç −8.3 % (piksel-GT) ama **+10.9 ± 5.6 % AUPRC / +4.4 % [1.7, 7.0] AP (bbox-GT)**: bbox metriği içeriğin silinmesini ödüllendirir. Glif modeli — iç blur −10.5 % AP. **Sınır duyarlılığı bbox modelinde piksel/glif modellerinden KÜÇÜK** (fark +1.4 % [0.5, 2.4]; H2'nin öngördüğünün tersi).
- **Rectangle-fit:** bbox/bbox ΔF1 −0.019 ± 0.039 (nötr); piksel/piksel −0.183; glif/glif −0.482.
- **Benign kontroller:** sabit 0.5 eşiğinde >%0.1 pozitif piksel içeren görüntü payı bbox modeli authentic %44 / paste-back %50 / resample %53; piksel %10 / %13 / %14; glif %20 / %23 / %22. Ancak val-kalibre τ*'da FP piksel oranları benzer (authentic: bbox 1.65‰, piksel 1.99‰) → fark büyük ölçüde kalibrasyon farkıdır. Benign paste-back her modelde FP'yi az ama anlamlı artırır (bbox +0.50‰ [0.36, 0.66]). Image-level AUROC (top-1 %, 3 seed): forged vs authentic bbox 0.892 [0.876, 0.907], piksel 0.892 [0.874, 0.909], glif 0.799; paste-back/resample negatifle 1.6–3.0 puan düşüş. (README T7'deki 0.838/0.896 değerleri tek seed'e aittir.)
**H2 KISMEN:** bbox eğitimi dikdörtgen, içerik-duyarsız ve 0.5 eşiğinde kötü kalibre tahmin üretir; bbox metriği içerik silmeyi ödüllendirir. Ancak ön kayıtlı **sınır-shortcut mekanizması yanlışlandı** (yön ters) ve paste-back yeniden-örnekleme izi davranışın küçük bir kısmını açıklar — baskın ipucu API ile yeniden işlenen alanın **bölge-düzeyi görünümü**dür (HYPOTHESIS; doğrudan test edilmedi).

![Ablasyon](phase2/experiments/figures/fig_ablation_delta.png)

### 4.4 H3 — Transfer ve dayanıklılık
| Koşul (val, ΔAUPRC kendi GT'sinde) | bbox modeli | piksel modeli | glif modeli |
|---|---|---|---|
| JPEG Q75 | −0.154 ± 0.015 | −0.137 ± 0.012 | −0.083 ± 0.009 |
| 0.75× rescale | −0.112 ± 0.037 | −0.245 ± 0.011 | −0.162 ± 0.024 |
| Gaussian blur σ=1 | −0.190 ± 0.058 | −0.361 ± 0.032 | −0.220 ± 0.030 |

Per-image AP farkları (piksel − bbox, kendi GT'sinde) küçük ve karışık yönlü: JPEG +0.021 [0.007, 0.034] (piksel lehine), rescale −0.007 [−0.018, 0.003] (fark yok), blur −0.016 [−0.029, −0.002] (bbox lehine; Holm p 0.06). Held-out generator (Gemini-only eğitim, 1 seed; kaynak-eşlenmiş): bbox modeli Ideogram AP 0.395 [0.325, 0.468] vs Gemini 0.908 (açık 0.513); piksel modeli 0.255 vs 0.559 (bbox-GT), 0.279 [0.175, 0.373] vs 0.582 (piksel-GT; açık 0.303); ortak piksel-GT altında Ideogram ΔAP piksel − bbox +0.025 [−0.039, 0.088] (Holm p 0.86). Ideogram'ı görmüş B modelleri: bbox 0.629, piksel 0.454. **H3 YANLIŞLANDI:** piksel/glif denetimi tutarlı bir dayanıklılık veya transfer avantajı sağlamaz; havuz AUPRC'de blur/rescale'de daha çok kaybeder (yüksek-frekans glif kenarlarına bağımlılık).

![Dayanıklılık](phase2/experiments/figures/fig_robustness.png)

### 4.5 H4 — Bbox dışında global iz
Maskeye değmeyen 224 crop'larda forged vs authentic AUROC **0.516** (457 çift; Gemini 0.516, Ideogram 0.512); negatif = fullimage_resample 0.498; pozitif kontrol (bbox-merkezli crop) 0.980. AUROC istatistiksel olarak 0.5'in üstünde (CI [0.513, 0.519]) ama pratikte sıfır. Post-hoc: 260/457 çiftte bbox-dışı crop'lar piksel-özdeş; kalan 197 çiftte sınıflandırıcı forged'ı authentic'in üstüne %92 oranında koyuyor, resample'a karşı yalnız %48 → bbox dışına dokunulan çiftlerde (çoğu WildReceipt/Gemini) zayıf bir **yeniden-örnekleme imzası** var, sahtecilik izi değil; simetrik ön-işleme bunu siler. **H4 YANLIŞLANDI (pratik olarak)** — GT audit'teki "ε ≥ 4'te global iz kaybolur" bulgusuyla tutarlı; image-level tespit alanın kendisinden gelmek zorundadır.

### 4.6 Örnek tahminler
![Örnekler](phase2/experiments/figures/fig_examples_predictions.png)

## 5. İstatistiksel analiz ve hipotez kararları
| Hipotez | Önceden sabitlenmiş yanlışlama ölçütü | Gözlenen (CI) | Karar |
|---|---|---|---|
| H1 metrik/sıralama | ΔF1 < 0.05 VE Kendall τ ≥ 0.9 | ΔF1@τ* (bbox-GT − piksel-GT, bbox-modeli) = 0.398 [0.377, 0.418]; τ(bbox vs piksel, 9 model) = -0.22 [-0.28, 0.06] | **DOĞRULANDI** |
| H2 sınır shortcut'ı | İki model benzer düşer (fark CI 0'ı kapsar) VE benign FP ≈ authentic | boundary relΔAP bbox-modeli -1.5 % [-2.0, -1.1] vs piksel-modeli -3.0 % [-3.8, -2.0]; fark (bbox − piksel, kendi GT) = +1.4 % [+0.5, +2.4] (yön TERS); benign paste-back FP − authentic (bbox-modeli) = 0.50 [0.36, 0.66] ‰ | **KISMEN** |
| H3 transfer/dayanıklılık | Fark yok / ters (CI örtüşür) | Ideogram ΔAP piksel-modeli − bbox-modeli (piksel-GT, gemini-only) = 0.025 [-0.039, 0.088]; dayanıklılık: yön koşula göre değişiyor (JPEG +, blur −) | **YANLIŞLANDI** |
| H4 global iz | AUROC ≈ 0.5 | AUROC forged-vs-authentic (bbox-dışı crop) = 0.516 [0.513, 0.519]; simetrik resample = 0.498 [0.492, 0.503]; pozitif kontrol = 0.980 [0.972, 0.988] | **YANLIŞLANDI** (pratik olarak; bkz. not) |

Ek kontrastlar: test tam-812 vs temiz-740 ΔAP 0.0012 [−0.003, 0.005] (sızıntı etkisi yok); bbox modeli − TruFor ΔAP 0.465 [0.436, 0.492], − ADCD-Net 0.625 [0.598, 0.652] (eşit koşul değil). Yöntem: seed'ler sabit faktör (görüntü başına 3-seed ortalaması), kaynak-belge grubu bootstrap B=1000 (gruplar çoğunlukla tekil: val 451/469 → ≈ görüntü bootstrap), eşleştirilmiş farklar, Holm düzeltmesi 58 ön-kayıtlı karar testinde; 463 kontrastın tamamı `phase2/stats/stats_summary.csv`, ayrıntılı kararlar `phase2/stats/hypothesis_decisions.md`, figürler `phase2/stats/figures/`.

![H1 CI](phase2/stats/figures/fig_H1_dualgt_ci.png)

![H2 CI](phase2/stats/figures/fig_H2_ablation_ci.png)

![H3 CI](phase2/stats/figures/fig_H3_transfer_ci.png)

![H4](phase2/stats/figures/fig_H4_auroc.png)


## 6. Adversarial review (PHASE 2 sonrası)
- **İddia "bbox metriği şişirir" GT gürültüsünden kaynaklanabilir mi?** ε-duyarlılık (ε ≥ 4'te global iz yok), morfolojik temizleme ve glif-kısıtlı GT'nin üçü de aynı örüntüyü verir; bbox modelinin iç-blur'a duyarsızlığı GT'den bağımsız bir davranış kanıtıdır. Kalan risk: insan doğrulaması (bekliyor).
- **Piksel modelinin düşük bbox-GT skoru "kötü model" mi?** Hayır: image-level AUROC'u bbox modeliyle eşit (0.892), 0.5 eşiğinde çok daha az görüntüyü işaretler (kalibrasyon) ve maskeleri daha kesin; düşük bbox-F1, değişmeyen bbox piksellerini işaretlememesinden gelir (tasarım gereği).
- **10 epoch yeterli mi?** Piksel/glif eğrileri 10. epoch'ta hafif yükseliyordu; mutlak farklar değişebilir, 3 seed'de sıralama tutarlı. PHASE 3'te 20 epoch + cosine LR önerilir.
- **Held-out generator güçsüz:** Ideogram n=41 (val) / 85 (test), tek seed; H3'ün generator kısmı "yetersiz kanıt" sınıfına yakındır, degradasyon kısmı ise 3 seed ile nettir.
- **Negatif sonuçların değeri:** H3/H4'ün yanlışlanması, veri setinin "generation-asymmetry" uyarısının pratikte sömürülebilir olmadığını ve piksel-GT'nin tek başına çözüm olmadığını gösterir — bildiri için savunulabilir, dürüst katkılardır.

## 7. Yayın odaklı araştırma planı
**Hedef mekânlar (ders şartı: kabul mektubu gerekli; Türkçe/İngilizce):** (1) IEEE SIU 2027 (Sinyal İşleme ve İletişim Uygulamaları Kurultayı; Türkçe/İngilizce, IEEE Xplore) — en uygun zamanlama/kapsam; (2) ICDAR 2027 Workshops veya IEEE WIFS 2027 (Information Forensics and Security) — uluslararası; (3) IEEE ICIP 2027 / EUSIPCO 2027. Son başvuru tarihlerini CFP'lerden doğrulayın (bu raporda tarih verilmemiştir).
**Bildiri taslağı:** `reports/Bildiri_Taslagi_IEEE.{md,docx}` (İngilizce, IEEE yapısı, yalnız ölçülen değerler).
**Turnitin ≤ %15 için kurallar:** audit ve yöntem bölümleri kendi ölçümlerimizi anlatır (özgün); literatür bölümünde makale cümleleri alıntılanmaz, yalnız sayı + atıf; tablo başlıkları ve metrik tanımları kendi ifadelerimizle; şablon metinleri (IEEE) hariç tutulur.
**Zaman çizelgesi (öneri):** Hafta 1: insan doğrulaması (100 çift) + PHASE 3 ek deneyler (20 epoch; bbox+glif birleşik kayıp; 3 seed held-out); Hafta 2: istatistik güncellemesi, figürlerin son hali; Hafta 3: bildiri yazımı + iç hakemlik; Hafta 4: Turnitin + gönderim. Vize sunumu için `reports/Vize_Raporu_Taslak.docx` güncellendi.
**PHASE 3 ek deneyler (öncelik sırası):** (a) birleşik kayıp (piksel maske + bbox yardımcı kayıp) → hem düşük FP hem dayanıklılık?; (b) 20 epoch tekrar; (c) held-out generator 3 seed + AIForge-Doc v2 (GPT-Image-2) erişimi; (d) bölge-görünümü hipotezinin testi: alan içini authentic glifleriyle değiştirip yalnız "render tarzı" bırakan sentetik kontrol; (e) YOLO-seg kutu formülasyonu (bbox-GT'yi kayırma testi).

## 8. Dosya haritası
`phase2/gt/` (GT varyantları, gt_report.md, ε figürleri, verification_grid.pdf, verification_sheet.csv, make_gt.py) · `phase2/controls/` (3 kontrol seti, controls_report.md, make_controls.py) · `phase2/literature_baselines/` (TruFor, ADCD-Net kod/ckpt kurulumu, zero_shot_results.csv, LICENSES.md, preds/) · `phase2/experiments/` (train_unet_p2.py, infer_fullimage.py, evaluate_p2.py, probes.py, h4_global_trace.py, run_phase2.ps1, results/, figures/, logs/, preds/, README.md) · `phase2/stats/` (stats_analysis.py, stats_summary.csv, hypothesis_decisions.md, figures/) · `models/phase2/` (12 checkpoint) · `reports/` (bu rapor, bildiri taslağı, güncellenmiş Vize raporu).


---

# EK — Doğrulama, Checkpoint Değerlendirmesi ve İşaretçi Artefaktı (güncelleme)


## 9. Ek (20 Ekim 2026, akşam güncellemesi)

### 9.1 İnsan doğrulamasının anlamı ve VLM ile ikamesi
PHASE 1'de CONDITIONAL GO'nun 2. koşulu, türetilmiş piksel-değişim GT'sinin "yeniden-kodlama gürültüsünden ayrıştığının" bağımsız bir gözle teyidiydi: 100 çiftlik tabakalı örneklemde bir insanın, forged/authentic çiftine bakarak (a) yeşil maskenin gerçekten değişen glifleri kapsayıp kapsamadığını (OK/PARTIAL/WRONG), (b) bbox dışındaki işaretli bölgelerin gerçek değişim olup olmadığını, (c) bbox içinde değişmeyen alan bulunup bulunmadığını işaretlemesi bekleniyordu. Annotator zamanı olmadığından bu denetim görüntü-destekli bir dil modeliyle (platform API, `claude-sonnet-5`) yapıldı; altı panelli görsel (forged / authentic / |fark|×4 / bbox / pixel_eps8_morph / glyph_v2) ve metadata değerleri verilerek yapılandırılmış JSON yanıt alındı. **Bu bir insan doğrulaması değildir**; GT'nin iç tutarlılığının bağımsız bir kontrolüdür ve `verification_sheet_vlm.csv` içinde insan sütunları boş bırakılmıştır.

Sonuç (96/100 çözümlenen yanıt): pixel_eps8_morph OK 69 / PARTIAL 26 / WRONG 1; glyph_v2 OK 55 / PARTIAL 41; "bbox içinde değişmeyen alan var" 96/96; bbox dışı işaretli bölge 21 çiftte, bunların yalnızca 4'ünde görünür değişim teyit edildi (15 belirsiz). Tekrar geçişinde (n=18) uyum: piksel kararı %83, glif %67, bbox-dışı %83. Modelin serbest metin açıklaması 78 kontrol edilebilir çiftin %68'inde metadata'daki sahte değeri içeriyordu. Yorum: türetilmiş GT'ler çoğunlukla değişen glifleri yakalıyor; PARTIAL kararlarının kaynağı tipik olarak düşük kontrastlı fotoğraflarda eksik kalan vuruşlar (glif maskesi) ve bbox dışındaki küçük, görsel olarak belirsiz bileşenlerdir.

### 9.2 Yeni bulgu: yeşil işaretçi kalıntısı (dataset artefaktı)
VLM denetimi sırasında sahte görüntülerin bbox kenarında ince yeşil bir çizgi fark edildi. Nesnel ölçüm (G − max(R,B) > 50 ve G > 110 olan, forged'da bulunup authentic'te bulunmayan pikseller; bbox kenarı ±6 px halkası): **3.336 çiftin %75.4'ünde** ≥ 20 px kalıntı var ve %75.2'sinde bu kalıntı ≥ 15 px uzunluğunda çizgi biçiminde. Kalıntı neredeyse tamamen **bbox'ın içinde** (medyan 176 px içeride, 0 dışarıda); generator'a özgü: **Gemini %83.6 vs Ideogram %1.5**; kaynak: WildReceipt %89.7, SROIE %85.6, CORD %50.3, XFUND %50.7; test çiftlerinin %93'ünde mevcut. Açıklama: makaledeki pipeline (crop üzerine 3 px yeşil dikdörtgen çizilir, yalnız bbox geri yapıştırılır) → dikdörtgenin iç kenarı yapıştırılan bölgeye düşer. VLM aynı soruya 40 evet / 24 hayır / 32 kararsız dedi; nesnel dedektörle karşılaştırıldığında evet/hayır dediği 64 çiftin 45'inde uyumlu (16 kaçırma, 3 yanlış alarm) → VLM artefaktı eksik raporluyor. Bu artefakt, bbox modelinin dikdörtgen tahminini, içerik-duyarsızlığını ve Ideogram'daki çöküşünü açıklayabilecek güçlü bir shortcut adayıdır; §9.4'teki probla doğrudan test edilmiştir.

### 9.3 Checkpoint ve eğitim dinamiği değerlendirmesi
Tüm 12 run planlanan 10 epoch'u tamamladı; en iyi checkpoint epoch 7–10'da; son 3 epoch'ta val-crop AUPRC eğimi −0.0014 … +0.0050/epoch (plato); eğitim kaybı son epoch'ta bbox 0.24, piksel 0.51–0.53, glif 0.43–0.45 (bbox hedefi belirgin biçimde daha "kolay"). Val-crop AUPRC (pozitif içeren crop'larda, tam-görüntü değerleriyle karşılaştırılamaz) bbox modellerinde 0.986–0.988, piksel/glif modellerinde 0.915–0.928'de doyuyor; seed'ler arası fark küçük. Kayıtlı checkpoint'ler: PHASE 1 5 adet (ResNet-18 ×3, U-Net ×2; 238 MB), PHASE 2 12 U-Net + 2 H4 ResNet (743 MB), üçüncü taraf TruFor/ADCD-Net ağırlıkları (803 MB; lisanslar `phase2/literature_baselines/LICENSES.md`). Yalnızca en iyi checkpoint saklandığından "son epoch" modelleri yeniden eğitilmeden değerlendirilemez; eğrilerden fark ihmal edilebilir görünmektedir.

| Run | n train | Epochs | Best epoch | Best val-crop AUPRC | Last-epoch AUPRC | Slope last 3 ep. | Train loss (last) | Time (min) | Peak GPU (GiB) |
|---|---|---|---|---|---|---|---|---|---|
| unet_bbox_geminionly_s42 | 2454 | 10 | 9 | 0.988 | 0.988 | +0.0000 | 0.231 | 9.900 | 4.660 |
| unet_bbox_s42 | 2748 | 10 | 9 | 0.986 | 0.986 | +0.0005 | 0.244 | 10.700 | 4.660 |
| unet_bbox_s43 | 2748 | 10 | 7 | 0.986 | 0.986 | -0.0001 | 0.237 | 11.000 | 4.660 |
| unet_bbox_s44 | 2748 | 10 | 7 | 0.987 | 0.985 | +0.0007 | 0.244 | 9.500 | 4.660 |
| unet_glyph_v2_geminionly_s42 | 2454 | 10 | 8 | 0.920 | 0.918 | -0.0014 | 0.446 | 10.100 | 4.660 |
| unet_glyph_v2_s42 | 2748 | 10 | 10 | 0.923 | 0.923 | +0.0018 | 0.437 | 10.100 | 4.660 |
| unet_glyph_v2_s43 | 2748 | 10 | 8 | 0.917 | 0.916 | -0.0007 | 0.429 | 10.500 | 4.660 |
| unet_glyph_v2_s44 | 2748 | 10 | 10 | 0.915 | 0.915 | +0.0036 | 0.435 | 11.100 | 4.660 |
| unet_pixel_eps8_morph_geminionly_s42 | 2454 | 10 | 8 | 0.928 | 0.927 | -0.0006 | 0.522 | 10.100 | 4.660 |
| unet_pixel_eps8_morph_s42 | 2748 | 10 | 8 | 0.925 | 0.924 | -0.0003 | 0.515 | 10.300 | 4.660 |
| unet_pixel_eps8_morph_s43 | 2748 | 10 | 10 | 0.924 | 0.924 | +0.0017 | 0.508 | 10.500 | 4.660 |
| unet_pixel_eps8_morph_s44 | 2748 | 10 | 10 | 0.924 | 0.924 | +0.0050 | 0.526 | 10.600 | 4.660 |

### 9.4 Yeşil işaretçi probu (H2'ye ek)
### Ek-C. Yeşil işaretçi kalıntısı: tüm modellerin kullandığı bir veri seti artefaktı (keşifsel sonda)

**Nasıl fark edildi.** GT doğrulama panelleri incelenirken, sahte görüntülerde kutunun iç kenarı boyunca uzanan 1–3 px kalınlığında doygun yeşil bir çizgi görüldü. Bu çizgi, veri seti üretim hattında inpainting'den önce alana çizilen yeşil dikdörtgen işaretçinin (veri seti makalesinin §3.3'ünde anlatılan adım: Gemini ikili maske kabul etmediği için bağlam kırpıntısına 3 px parlak yeşil dikdörtgen çizilir ve istemde buna atıf yapılır; Wu ve ark., 2026) tamamen silinmemiş kalıntısıdır. Bir dedektör yazıldı (HSV yeşil eşiği + kutu iç kenar halkası + çizgi biçimi); eşleşen 3.336 çiftin **2.514'ünde (%75,4)** kalıntı var: Gemini %83,6, Ideogram %1,5; kaynak bazında CORD %50, XFUND %51, SROIE %86, WildReceipt %90. Kalıntı kutunun içinde (medyan 176 px) ve dışında hemen hiç yok (%0,3). Authentic görüntü gerektirmeyen bir sürümü resmi **test bölümünde 812 sahtenin 604'ünde (%74,4)** kalıntı buldu; 600 authentic görüntüde **0 yanlış alarm** verdi.

**Sonda tasarımı.** Doğrulama bölümündeki 487 çiftin 368'inde kalıntı var. Her görüntü için dört koşul oluşturuldu: (1) sahte (olduğu gibi), (2) *işaretçi onarılmış* sahte — yalnızca işaretçi pikselleri authentic kaynaktan kopyalandı, içerik değişmedi, (3) authentic, (4) *authentic + işaretçi* — yalnızca işaretçi pikselleri authentic görüntüye aktarıldı, hiçbir karakter değişmedi. Eğitilmiş 9 model (3 GT × 3 tohum) bu dört girdide, kutu GT'ye göre görüntü-başına AP, kalibre eşikte kutu kapsama oranı ve görüntü-düzeyi skor (üst %1 ortalaması) ile ölçüldü; grup-bootstrap %95 GA.

**Bulgular (Tablo Ek-C1, Şekil Ek-C1).**
- *İşaretçi tek başına kutuyu tetikliyor.* Authentic + işaretçi girdisinde bbox modelinin AP'si **0,906 [0,887, 0,924]**, kutu kapsaması 0,80, görüntü skoru 0,10 → 0,53. Aynı authentic görüntülerde işaretçi yokken AP 0,178, kapsama 0,04.
- *İşaretçi kaldırılınca performansın dörtte biri gidiyor.* Gerçek sahtelerde yalnızca işaretçi onarılınca bbox modeli AP 0,988 → 0,763 (**Δ −0,225 [−0,262, −0,189]**), kapsama 0,94 → 0,59. Kalıntısız sahtelerde (n=119) bbox modeli zaten 0,747 — onarılmış seviye. Dolayısıyla §6'daki "Ideogram üretici farkı" büyük ölçüde bir **işaretçi farkı**dır.
- *Piksel ve glif modelleri de ipucunu miras alıyor.* İşaretçi çizgisi authentic görüntüden farklı olduğu için türetilmiş piksel-değişim GT'sinin bir parçasıdır (etkilenen çiftlerde `pixel_8_morph` pozitif piksellerinin medyan %10'u, p75 %19). Onarımda piksel modelinin kendi GT'sine göre AP'si 0,266 düşüyor; authentic + işaretçi girdisinde kutu-AP'si 0,14 → 0,64, glif modelininki 0,13 → 0,47. Etki bbox modelinde en güçlü: bbox modeli çizgiyi görmekle kalmıyor, dikdörtgeni **tamamlıyor** (tahmin–kutu IoU 0,74; piksel 0,39; glif 0,24).
- *Öğrenmesiz kural yetmiyor.* Görüntünün tamamında yeşil çizgi parçalarını bulup sınırlayıcı kutularını tahmin eden basit kural testte %53 ateşliyor, ortalama F1 **0,054**, IoU ≥ 0,5 yalnızca %3. Kalıntı genellikle alanın bir-iki kenarını işaretliyor; kutunun tamamı işaretçiden "okunamıyor". Ağlar çizgiyi komşu metin alanının görünümüyle birleştirerek öğreniyor — yani bu sığ bir piksel kuralı değil, öğrenilmiş bir kısayol.
- *Ön-kayıtlı sınır testi neden kaçırdı?* σ = 1,5 Gauss bulanıklığı 1–3 px'lik doygun yeşil çizgiyi yayar ama kromasını silmez; plasebo ve iç-bulanıklık koşulları ise çizgiye hiç dokunmaz. Ön-kayıtlı mekanizma (yeniden örnekleme dikişi) **yanlış mekanizmaydı**; gerçek ipucu bir renk çizgisidir.

**H2 kararına etkisi.** Ön-kayıtlı ölçütle karar değişmedi (sınır-halka bulanıklığı mekanizması yanlışlandı; şekil ve içerik-duyarsızlığı doğrulandı → *Kısmen*). Keşifsel sonda ise sınırda yer alan **farklı bir ipucunun** üç model ailesi tarafından da kullanıldığını büyük etki ve dar aralıklarla gösteriyor. İkisi yan yana raporlandı; keşifsel sonuç ön-kayıtlı sonucun yerine geçirilmedi.

**Çıkarımlar.** (i) Bu veri setinde raporlanan lokalizasyon doğruluğunun önemli bir kısmı adli kanıtın değil, bir üretim-hattı artefaktının ölçümüdür; test bölümünün dörtte üçü ipucunu taşır. (ii) Piksel-değişim GT'si işaretçiyi içerdiği için "işaretçisiz GT" ve "işaretçisiz eğitim görüntüsü" bir sonraki zorunlu deneydir (her iki girdiyi üreten kod pakette). (iii) Gemini–Ideogram karşılaştırmaları kalıntısız alt kümelerde yapılmalıdır. (iv) AIForge-Doc v2 (GPT-Image-2) hattında da yeşil kontur işaretçisi bildirilmiştir; aynı artefaktın orada da var olup olmadığı açık bir sorudur.

**Tablo Ek-C1.** Kalıntı taşıyan 368 doğrulama çiftinde dört koşul (3 tohum ortalaması; %95 GA).

| Model | Girdi | AP (kutu GT) | AP (piksel GT) | Kutu kapsama @τ* | IoU(tahmin, kutu) | Görüntü skoru |
|---|---|---|---|---|---|---|
| bbox | sahte | 0,988 | 0,622 | 0,940 | 0,878 | 0,634 |
| bbox | sahte, işaretçi onarılmış | 0,763 | 0,512 | 0,592 | 0,534 | 0,450 |
| bbox | authentic | 0,178 | 0,089 | 0,042 | 0,033 | 0,098 |
| bbox | authentic + işaretçi | 0,906 | 0,562 | 0,799 | 0,737 | 0,532 |
| piksel | sahte | 0,749 | 0,761 | 0,539 | 0,499 | 0,195 |
| piksel | sahte, işaretçi onarılmış | 0,499 | 0,495 | 0,280 | 0,253 | 0,085 |
| piksel | authentic | 0,138 | 0,085 | 0,032 | 0,027 | 0,035 |
| piksel | authentic + işaretçi | 0,641 | 0,634 | 0,426 | 0,391 | 0,126 |
| glif | sahte | 0,524 | 0,586 | 0,325 | 0,284 | 0,089 |
| glif | sahte, işaretçi onarılmış | 0,400 | 0,446 | 0,193 | 0,172 | 0,064 |
| glif | authentic | 0,127 | 0,080 | 0,023 | 0,016 | 0,068 |
| glif | authentic + işaretçi | 0,465 | 0,496 | 0,284 | 0,241 | 0,102 |

![Şekil Ek-C1. Yeşil işaretçi sondası.](phase2/experiments/figures/fig_green_marker.png)

Dosyalar: `phase2/gt/green_residue_pairs.csv`, `phase2/gt/green_marker_all_forged.csv`, `phase2/experiments/results/green_marker_summary.csv`, `…/green_marker_contrasts.csv`, `…/per_image_green_marker.csv`, `…/heuristic_green_test.csv`, `…/heuristic_green_summary.json`, `…/green_marker_findings.json`; kod: `phase2/experiments/probe_green_marker.py`, `analyze_green_marker.py`.


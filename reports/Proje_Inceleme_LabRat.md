# Uçtan Uca Proje İncelemesi (Lab Rat protokolü) — AIForge-Doc v1 Sahtecilik Lokalizasyonu

**Tarih:** 9 Kasım 2026 · **Kapsam:** `Vize Proje/` tamamı (kod, veri audit'i, PHASE 1–2 sonuçları, raporlar, bildiri taslağı, literatür)
**Yöntem:** `labrat:paper-check` (sayı ↔ sonuç dosyası, atıf ↔ DBLP/Crossref/arXiv, değerlendirme bütünlüğü), `labrat:literature-research` (gap ve benchmark doğrulaması, web/Consensus/HF/GitHub), `labrat:rebuttal` (hakem gözüyle zayıflıklar ve sorular). Hiçbir proje dosyası değiştirilmedi; ayrıntılı ajan raporları `reports/inceleme/paper_check.md` ve `reports/inceleme/literature_check.md` içindedir.

## 0. Karar özeti

| Boyut | Durum | Not |
|---|---|---|
| Sayıların kaynak dosyalarla uyumu | **✓ İyi** | ~130 sayı kontrol edildi; doğrudan uyuşmazlık yok, 8 küçük sunum sorunu (§2) |
| Atıfların gerçekliği | **✓ İyi** | 39/39 BibTeX kaydı gerçek (17 arXiv, 22 DOI Crossref'te birebir); 2 iddia yönü/kaynağı düzeltilmeli; klasik kaynaklar bib'de eksik |
| Değerlendirme bütünlüğü | **✓ / ⚠** | Test seti eğitim ve model seçiminde kullanılmamış; τ* val'de seçilmiş. **Ancak** PHASE 1 smoke Pixel-AUPRC değerleri negatif alt-örnekleme nedeniyle şişkin (§3) |
| Literatür boşluğu iddiası | **⚠ Daraltılmalı** | v1'de kontrollü GT deneyi yok (doğru); fakat v2'de eğitimli dedektör + "process-twin" kontrolü içeren eş zamanlı hakemsiz bir çalışma var (Helali, 3 Kas 2026) |
| Ders şartnamesine uyum | **✓** | Dönem içi rapor başlıkları (Proje Adı, Literatür, Materyal-Metot, Sonuçlar) mevcut; Turnitin kuralı için metin kendi ölçümlerine dayanıyor |
| Genel | **GO (düzeltmelerle)** | Bildiri gönderimi öncesi §6'daki 1–6 numaralı düzeltmeler zorunlu, 7–12 önerilir |

## 1. Projenin bir cümlelik özeti (incelemeci okuması)
AIForge-Doc v1 (4.061 AI-inpainting ile sahteleştirilmiş fiş/form, bbox maskeli) üzerinde klasik öznitelikler, ResNet-18 ve U-Net ile sınıflandırma/lokalizasyon baseline'ları kurulmuş; asıl katkı, **eğitim maskesinin geometrisinin (bbox / piksel-değişim / glif) model davranışını ve raporlanan skoru nasıl değiştirdiğini** 3 tohumlu kontrollü deneyle ölçmek ve veri setinde **yeşil işaretçi kalıntısı** adlı bir üretim-hattı artefaktının tüm modellerce kısayol olarak kullanıldığını göstermektir. Dört ön-kayıtlı hipotezden H1 doğrulanmış, H2 kısmen, H3 ve H4 yanlışlanmıştır — negatif sonuçlar dürüstçe raporlanmıştır.

## 2. Sayı doğrulaması (paper-check)
Uyumlu bulunanlar: Vize §3.1 ↔ `dataset_audit/audit_report.md`; Tablo B/C/D/E/F ↔ `baselines/results/*.json`; §4.b ↔ `summary_test_agg.csv`, `summary_val_agg.csv`, `stats_summary.csv`, `hypothesis_decisions.md`, `zero_shot_results.csv`; §4.c ↔ `checkpoint_summary.csv`, `green_marker_findings.json`; veri-seti makalesi sayıları (TruFor AUC 0.751 / IoU 0.358 / F1 0.434; DTD 0.563 / 0.020; GPT-4o 0.509) ↔ arXiv 2602.20569 Tablo 2.

| # | Konum | Sorun | Önem |
|---|---|---|---|
| 1 | Vize Tablo D vs E | U-Net RGB F1 0.707 / 0.708 (iki koşunun yuvarlaması) | düşük |
| 2 | Vize §4.c | "tohumlar arası std ≤ 0,01" yalnız val-crop AUPRC için doğru; tam-görüntü AUPRC std 0.015, F1 0.029 | orta |
| 3 | Vize §4.c, Bildiri Tablo 14 | İşaretçi sondası AP 0.906 / 0.988→0.763 değerleri birincil kaynak (`green_marker_summary.csv`) teslim paketinde yok | orta |
| 4 | Vize §4.b | "dikdörtgensellik medyan 0.87": `summary_val_agg.csv` ortalaması 0.59–0.62; medyan/ortalama ayrımı yazılmalı | düşük |
| 5 | Vize §2.3 | ADCD-Net "TruFor'u %20.8 aşar" → makale "SOTA'yı ortalama %20.79"; "DTD %68–73 düşer" kaynağı gösterilmeli | orta |
| 6 | Vize §6 | "38 kayıt" → 39 | düşük |
| 7 | Vize §2.5, Bildiri | Yeşil işaretçi v1 makalesinde §3.3 değil **§3.4** (Gemini maddesi) | düşük |
| 8 | PHASE2 §2.1 | "10+5+2 dışlama → 2.748": 2.762−2.748=14; 10 dejenere çiftin 3'ü val/test dışında → 7+5+2 | düşük |

## 3. Değerlendirme bütünlüğü
1. **Test seti dokunulmamış** — `baselines/common/data.py` TestingSet'e erişmez; `splits.json`'da train∩val=∅; PHASE 2'de seçim yalnız val, test yalnız `evaluate_p2.py --split test`. ✓
2. **τ\* sızıntısı yok** — eşik val'de seçilip `tau.json`'a yazılır, test/bozulmada okunur. Not: `unet_bbox_s44` için bbox τ*=0.95 iken diğer tohumlarda 0.45 → τ* kararsızlığı raporlanmalı. ✓/⚠
3. **PHASE 1 smoke Pixel-AUPRC şişkin (en önemli bulgu)** — `baselines/common/metrics.py` tüm pozitifleri alıp crop başına ≤20k negatif örnekler; 512² crop'ta pozitif önseli ≈%4.7'den ≈%38'e çıkar. AUPRC önsele bağımlı olduğundan 0.947/0.939 (ve val-crop 0.986) **gerçek havuzlanmış AUPRC değildir**; AUROC ve F1 etkilenmez. PHASE 2 `p2metrics.py` 256-bin histogramla kesin hesaplar (0.923). Vize raporu bu şişmenin **yönünü** belirtmiyor → düzeltilmeli. ⚠
4. **Grup-farkında bölme** kodda var ama teslim paketinde `samples_forged.csv`/grup kimliği olmadığından dışarıdan doğrulanamaz. ⚠
5. **"Natif çözünürlük"** iddiası: uzun kenar > 4096 px olan ~%5 görüntü 4096'ya küçültülüyor (`infer_fullimage.py`); metinde istisna olarak yazılmalı. düşük
6. Smoke sonuçlarında seçim ve raporlama aynı smoke_val'de (rapor "provizyonel" diyor ✓).

## 4. Literatür ve benchmark doğrulaması (literature-research)
**Gap (a) — "v1 üzerinde eğitilmiş model yok":** kısmen çürütüldü.
- **Helali (2026)**, *Beyond Zero-Shot on AIForge-Doc v2: A Trained Detector and a Process-Twin Control* (GitHub + Zenodo 10.5281/zenodo.23917285, 3 Kas 2026, hakemsiz): **v2** üzerinde 371k parametreli FCN eğitmiş; "process-twin" kontrolü (içerik değişmeden yeniden-örnekle/yapıştır hattından geçen orijinaller) %37.7 oranında sahte işaretleniyor → **üretim-hattı artefaktı kısayolu bağımsız olarak gösterilmiş**. Piksel metrik, v1 ve işaretçi analizi yok.
- **Joya15** (GitHub, hakemsiz): v1+v2+GPT4o-Receipt birleşik; ana model **Tiny U-Net** (piksel F1 0.322 / IoU 0.192), SegFormer-lite yalnız çapraz-veri deneyi (v2→v1 F1 0.044). Raporda "SegFormer-Lite eğitip" ifadesi düzeltilmeli.
- **leongyiquan/DocDet** (GitHub): karo sınıflayıcı ResNet-18, sayfa AUC 0.80, lokalizasyon yok.

**Gap (b) — bbox-vs-piksel GT ve yeşil işaretçi:** doğrulandı. v1 makalesi §3.4: *"we draw a 3-pixel bright-green rectangle on the context crop to mark the edit region"* (Gemini maddesi; Ideogram ikili maske aldığı için kalıntının Gemini %83.6 / Ideogram %1.5 olmasını açıklar); §3.3 adım 5 / §4.2 GT = "exact field bounding box". Hiçbir kaynak kalıntıyı ölçmüyor veya GT geometrisiyle oynayarak eğitmiyor. Semantic Scholar: v1'e 9, v2'ye 5 atıf, tümü zero-shot/anma.

**Savunulabilir iddia (önerilen cümle):** "AIForge-Doc v1 üzerinde eğitim maskesinin geometrisini kontrollü biçimde değiştiren, piksel-değişim/glif GT'si türeten ve veri seti makalesinin §3.4'ünde tarif edilen yeşil işaretçinin kalıntısını ölçen bir çalışmaya rastlanmamıştır; v2'de görüntü düzeyinde hat-artefaktı kontrolü yapan eş zamanlı, hakemsiz bir çalışma (Helali 2026) vardır."

**Benchmark tablosu (birincil kaynaktan teyitli, piksel F1):**

| Yöntem | DocTamper-Test | FCD | SCD | AIForge-Doc v1 (zero-shot) | Kaynak |
|---|---|---|---|---|---|
| DTD (CVPR 2023) | 0.792 (IoU 0.828) | 0.816 | 0.754 | F1 0.030 / IoU 0.020 / img-AUC 0.563 | Qu+23; Wu+26 |
| FFDN (ECCV 2024) | 0.86 (IoU 0.90); JPEG Q75 0.75 | 0.92 | 0.81 | — | Chen+24 |
| ADCD-Net (ICCV 2025) | 0.806 | 0.815 | 0.740 (ort. 0.787) | bu proje: 0.182 / IoU 0.149 | Wong+25; `zero_shot_results.csv` |
| TruFor (CVPR 2023) | 0.168 (resmî ckpt) | 0.109 | 0.281 | **F1 0.434 / IoU 0.358 / img-AUC 0.751** (makale); bu proje 0.403 / 0.328 | Guillaro+23; Wu+26 |
| CAT-Net / MVSS-Net | 0.700 / 0.431 | 0.553 / 0.424 | 0.631 / 0.414 | — | DTD/ADCD makaleleri |
| DOCFORGE-BENCH (zero-shot, 8 set ort. F1) | TruFor 0.34 · CAT-Net 0.33 · ADCD 0.23 · FFDN 0.17 · DTD 0.16 · IML-ViT 0.08 | | | | Zhao+26 Tablo 3–4 |
| **Bu proje: U-Net (ResNet-18), bbox-GT, 3 tohum** | — | — | — | **AUPRC 0.950±0.011 · F1@τ* 0.846±0.015 · IoU 0.789**; aynı tahminler piksel-GT ile F1 0.45 | `summary_test_agg.csv` |
| Tiny U-Net (Joya15, hakemsiz) | — | — | — | F1 0.322 / IoU 0.192 (v1+v2 birleşik) | GitHub |
| AIForge-Doc v2 (Wu+26b) | — | — | — | insan 2AFC 0.501; TruFor AUC 0.599; DocTamper 0.585; Helali FCN img-AUC 0.989 (hakemsiz) | arXiv 2604.25213; Zenodo |

**Literatür matrisinin (kesim 17 Ekim 2026) kaçırdığı ilgili çalışmalar:** TGIF2 (arXiv:2603.28613; rastgele maskeler, maske-nesne yanlılığı), INP-X (arXiv:2602.00192; inpaint dışı pikseller geri konunca dedektör doğruluğu %91→%55 — H4/işaretçi bulgusuyla paralel), DiffusionPrint (arXiv:2604.12443), Pandolfini+ (arXiv:2512.16688; küçük alanlı düzenlemelerde çöküş), TextShield-R1 (AAAI 2026), DocShield (arXiv:2604.02694), GenText-Forensics Challenge / RealText-V2 (ACM MM 2026), SC-RF (arXiv:2608.02258), Venus-DeFakerOne (arXiv:2605.14091), Helali (2026).

## 5. Hakem gözüyle zayıflıklar ve beklenen sorular (rebuttal)

| # | Zayıflık | Tür | Önem | Yanıtlanabilir mi? |
|---|---|---|---|---|
| W1 | GT doğrulaması insan yerine VLM ile yapıldı; PHASE2 durum tablosu hâlâ "insan doğrulaması bekliyor" | deneysel / tutarlılık | yüksek | Kısmen: 96 çiftte VLM OK/PARTIAL oranları var; **100 çiftlik insan denetimi hâlâ en ucuz ve en güçlü ek** |
| W2 | Tek mimari (smp U-Net ResNet-18), 10 epoch; "GT etkisi mimariden bağımsız mı?" | deneysel | yüksek | Hayır → en az bir ek mimari (SegFormer-B0 veya U-Net++/DeepLabV3+) ile 1 tohum |
| W3 | Held-out generator (Ideogram) n=41 val / 85 test, tek tohum → H3 generator kısmı "yetersiz kanıt" | istatistik | orta | Kısmen; 3 tohum + v2 (GPT-Image-2) ile güçlendirilebilir |
| W4 | Yeşil işaretçi bulgusu keşifsel; işaretçisiz yeniden eğitim yapılmadı | deneysel | yüksek | Hayır → **işaretçi onarılmış eğitim seti ile 3 GT × 1 tohum yeniden eğitim** (kod pakette) — bildirinin en güçlü ek deneyi |
| W5 | Pixel-GT'nin kendisi işaretçiyi içeriyor (pozitiflerin medyan %10'u) | metodolojik | orta | Evet: işaretçi pikselleri GT'den çıkarılmış varyant türetilmeli |
| W6 | PHASE 1 AUPRC alt-örneklemeyle şişkin; PHASE 2 ile karşılaştırılamaz | metrik | orta | Evet: `unet_*_val_probs.npz`'den kesin AUPRC yeniden hesaplanabilir |
| W7 | Zero-shot TruFor/ADCD-Net karşılaştırması eşit koşul değil (in-domain eğitim vs zero-shot) | sunum | orta | Evet: ayrı tabloda ve "referans" olarak etiketli ✓; TruFor'u v1'de fine-tune eden bir satır beklenir |
| W8 | Image-level tespit için `TestingSet/authentic` kullanılamıyor; group-aware negatifler yalnız val'de | deneysel | orta | Evet: açıklanmış; test image-level AUROC verilmediği açıkça yazılmalı |
| W9 | Bildiri Tablo 9d tek tohum (s44) değerleri "3-seed mean" etiketli; GenD iddiasının yönü ters | tutarlılık | yüksek | Evet: düzelt |
| W10 | Klasik kaynaklar (He, Ronneberger, Ojala, Haralick…) metinde atıflı, bib'de yok | sunum | düşük | Evet |

**En olası 8 hakem sorusu ve taslak yanıtlar**
1. *Q: Piksel-değişim GT'niz yeniden-kodlama gürültüsü mü yoksa gerçek değişim mi?* — ε-duyarlılık (ε≥4'te global iz kaybolur, ε=8'de bbox-dışı değişim %33.5 lokaldir), morfolojik temizleme ve glif-kısıtlı varyantın aynı örüntüyü vermesi; VLM denetiminde pixel_eps8_morph OK 69 / PARTIAL 26 / WRONG 1. **Eksik:** insan denetimi.
2. *Q: Bbox modelinin düşük piksel-GT skoru "yanlış GT ile ölçüm" değil mi — asıl hangi GT doğru?* — İddia "hangisi doğru" değil, "GT seçimi sıralamayı ve mutlak skoru değiştirir" (Kendall τ −0.22; ΔF1 0.398 [0.377, 0.418]); iç-blur duyarsızlığı GT'den bağımsız davranış kanıtı.
3. *Q: Sonuçlar tek mimariye özgü olabilir.* — Kabul; PHASE 3'te ikinci mimari (W2).
4. *Q: Yeşil işaretçi kaldırılınca modeller ne kadar kalıyor?* — Onarılmış girdide AP 0.988→0.763; kalıntısız alt kümede zaten 0.747. **Yeniden eğitim yapılmadı** (W4) — en kritik ek deney.
5. *Q: TruFor'u fine-tune etseydiniz?* — Yapılmadı; U-Net seçimi GT etkisini izole etmek için. Bir TruFor fine-tune satırı eklenebilir (ağırlıklar/kod pakette).
6. *Q: Image-level tespit performansı test'te nedir?* — Verilemiyor (test authentic'leri ayna değil); val'de top-1% skoruyla AUROC 0.892 [0.876, 0.907] (3 tohum).
7. *Q: Helali (2026) process-twin kontrolü sizin benign kontrollerinizle nasıl ilişkili?* — Benign paste-back/resample kontrolleri aynı fikrin v1'deki piksel-düzeyi karşılığı; bbox modelinde paste-back FP +0.50‰ [0.36, 0.66]. Eş zamanlı çalışma olarak anılmalı.
8. *Q: 10 epoch yeterli mi?* — Val-crop AUPRC son 3 epoch eğimi −0.0014…+0.0050 (plato); sıralama 3 tohumda tutarlı; 20 epoch PHASE 3'te.

## 6. Öncelikli düzeltme listesi
**Bildiri/rapor gönderimi öncesi zorunlu**
1. Bildiri Tablo 9d: 3-tohum değerleri (0.892 / 0.892 / 0.799; paste-back 0.872 / 0.876 / 0.779; resample 0.866 / 0.863 / 0.769) ya da "seed 44" etiketi.
2. GenD (Yermakov+ WACV 2026) iddiasının yönü: eşleşmiş eğitim ↑, eşleşmemiş ↓ AUROC; 56.8/90.0 yer gösterilemezse kaldır; yazar listesine Fritz ekle.
3. PHASE 1 Pixel-AUPRC'ye "negatif alt-örnekleme (≤20k/crop) → şişkin; PHASE 2 ile karşılaştırılamaz" dipnotu; mümkünse kesin yeniden hesap.
4. Literatür: Helali (2026) ve DocDet eklenmeli; Joya15 açıklaması düzeltilmeli; yenilik iddiası §4'teki dar cümleyle sınırlanmalı; işaretçi atfı §3.4.
5. PHASE2 durum tablosu ve Bildiri §4.3: "insan doğrulaması" → "VLM destekli tutarlılık denetimi (insan doğrulaması bekliyor)".
6. `references.bib`: klasik kaynaklar + Helali (Zenodo DOI) + 2025–26 eksikler; "39 kayıt".

**Önerilen (PHASE 3)**
7. İşaretçi onarılmış eğitim seti ile yeniden eğitim (3 GT × 1 tohum) — bildirinin ana ek deneyi.
8. 100 çiftlik insan GT denetimi (`verification_sheet.csv` hazır).
9. İkinci mimari (SegFormer-B0 / DeepLabV3+) ile H1 tekrarı.
10. Held-out generator 3 tohum; AIForge-Doc v2 erişimi.
11. Teslim paketine `samples_forged.csv`, `green_marker_summary.csv`, `excluded_ids.json`, veri kartı snapshot'ı.
12. Tohumlar arası std ifadesi ve "natif çözünürlük" istisnasının (>4096 px) metne işlenmesi.

## 7. Ders şartnamesi açısından notlar
- Dönem içi rapor başlıkları (Proje Adı / Literatür / Materyal ve Metot / Sonuçlar) mevcut; "sonuçlar çok kısıtlı veriyle" şartı smoke alt kümesiyle karşılanıyor, tam-veri sonuçları ek bölümde.
- **Veri seti hoca listesinde değil** (`VeriTabanlariGoruntuIsleme.xlsx`: Oxford Pets, ISIC, Kvasir-SEG, MVTec…). Raporda bir cümleyle gerekçelendirilmeli: HF kartı görevleri `image-classification` + `image-segmentation`; 4.061 görüntü 0/255 ikili maske; Kvasir/ISIC ile aynı "küçük ön plan, ikili maske" yapısı (medyan kurcalanmış alan %0.98); yayımlanmış zero-shot referanslar karşılaştırma tabanı sağlıyor; CC BY-NC-SA 4.0 akademik kullanım serbest. Hocanın onayı alınmışsa belirtilmeli.
- Turnitin ≤ %15: metinler kendi ölçümlerine dayanıyor; literatür bölümünde makale cümleleri alıntılanmıyor ✓. Yeni eklenen §3.4 alıntısı kısa tutulmalı.
- Sunumda hazır cevap gerektiren noktalar: W1 (VLM vs insan), W4 (işaretçi), AUPRC şişmesi.

# Paper-Check Raporu — AIC515 Vize Projesi (AIForge-Doc v1)

Kapsam: `reports/Vize_Raporu_Taslak.md` (birincil), `PHASE2_Metodoloji_ve_Sonuclar.md`, `Bildiri_Tam_Metin.md`, `Bildiri_Taslagi_IEEE.md`; kaynak dosyalar `baselines/results/*`, `phase2/experiments/results/*`, `phase2/stats/*`, `phase2/controls/*`, `phase2/gt/*`, `dataset_audit/audit_report.md`; atıflar `literature/references.bib` (39 kayıt); kod `baselines/`, `phase2/experiments/`. Tüm yollar `/mnt/user-data/uploads/AI 515/Vize Proje/` altındadır. Hiçbir dosya değiştirilmedi.

## Özet

- **Sayılar:** Vize §3.1, §4 (B–F), §4.b, §4.c ve Bildiri Tablo 1/4/5/6/7/8/14'teki ~130 sayı kaynak dosyalarla karşılaştırıldı. **Doğrudan uyuşmazlık bulunmadı**; 4 küçük sunum/yorum sorunu var (aşağıda). `data/AIForge-Doc-v1/README.md` ve `dataset_info.json` yüklemede **yok** (yalnız 12 örnek PNG var); veri-seti makalesi sayıları arXiv HTML'inden doğrulandı.
- **Atıflar:** 39 BibTeX kaydının **tamamı gerçek** (17 arXiv kimliği arXiv'de, 22 DOI Crossref'te başlık/yazar/mekân/yıl ile birebir). Uydurma DOI yok. Sorun: iki belge arasında GenD iddiasının yönü ters; 2609.13617 için "0.921 vs 0.918" iddiası özette yok, tam metin doğrulanamadı; ADCD-Net "%68–73" tam metinde bulunamadı (muhtemelen şekilden).
- **Değerlendirme bütünlüğü:** Test seti eğitim/model seçiminde kullanılmamış; τ* val'de seçilip test'e taşınıyor; Faz 2 metrikleri histogramdan kesin. **Ana bulgu:** Vize §4'teki smoke "Pixel-AUPRC 0.947/0.939" negatif piksel alt-örneklemesiyle (crop başına ≤20k negatif, tüm pozitifler) hesaplanmış → sınıf önseli ~%4.7'den ~%38'e çekilmiş, **AUPRC sistematik olarak şişkin**; Faz 2'nin kesin 0.923'üyle karşılaştırılamaz.
- **İç tutarsızlık:** Bildiri Tablo 9d'deki görüntü-düzeyi AUROC'lar (0.838/0.896/0.814) tek-seed (s44) iken "three-seed mean" etiketli; metin 0.892/0.892/0.799 diyor. PHASE2 durum tablosu "insan doğrulaması bekliyor" derken §9.1 VLM ile ikame edildiğini söylüyor.

## Sayı uyuşmazlıkları (tablo)

Doğrulanan (uyumlu) başlıca kalemler: §3.1 tüm veri istatistikleri ↔ `audit_report.md` (3.249/812, 3.247 çift, kaynak/dil/generator dağılımları, 150×168–6016×7016, medyan 750×1296, %0.98, 33 px, %49, %33.5, 86/812, %8.9, 2.762/487, 400/100); Tablo B/C ↔ `classical_results.json`/`resnet_results.json` (tüm hücreler, TN/FP/FN/TP dâhil); Tablo D/E/F ↔ `unet_results.json`/`robustness_results.json` (tüm hücreler; 23.5 dk = 64+76+65+603+601 s); §4.b ↔ `summary_test_agg.csv`, `summary_val_agg.csv`, `stats_summary.csv`, `hypothesis_decisions.md`, `h4_results.csv`, `zero_shot_results.csv`; §3.2.1 ΔAUPRC ↔ `robustness_agg.csv`; §4.c ↔ `checkpoint_summary.csv`, `green_marker_findings.json`; TruFor 0.751/0.358/0.434, DTD 0.563/0.020, GPT-4o 0.509 ↔ arXiv 2602.20569 Tablo 2.

| # | Konum | Raporlanan | Kaynak değer | Not / Önem |
|---|---|---|---|---|
| 1 | Vize §4 Tablo D s.184 vs Tablo E s.195 | RGB Pixel-F1 **0.707** (D) / **0.708** (E) | `unet_results.json` 0.70746; `robustness_results.json` 0.70750 | Aynı model, iki farklı koşudan yuvarlama farkı; tek değer kullanılmalı. DÜŞÜK |
| 2 | Vize §4.c s.257 | "Tohumlar arası std ≤ 0,01" | `summary_val_agg.csv` bbox/bbox: AUPRC std **0.0148**, F1@0.5 std **0.029**, τ std 0.29 | Yalnız val-crop AUPRC (0.986–0.987) için doğru; tam-görüntü metriklerinde değil. ORTA |
| 3 | Vize §4.c s.259; Bildiri Tablo 14 | AP 0.906 [0.887,0.924], 0.988→0.763, kapsama 0.80 | `green_marker_findings.json`'da **yok**; `green_marker_summary.csv` yüklemede yok | Yalnız PHASE2 Ek-C/Bildiri Tablo 14 ile tutarlı; birincil kaynak doğrulanamadı. ORTA (doğrulanamadı) |
| 4 | Vize §4.b s.243; Bildiri özet | "rectangularity medyan 0.87" | `summary_val_agg.csv` rect_0p5_mean 0.59, rect_tau_mean 0.62; `hypothesis_decisions.md` 3-seed havuz medyanı **0.865** | Medyan/ortalama ayrımı belirtilmeli; Bildiri Tablo 9a ile metin farklı istatistik veriyor. DÜŞÜK |
| 5 | Vize §2.3 s.25 | ADCD-Net "TruFor'u ortalama %20.8 aşar"; DTD F1 "%68–73 düşer" | arXiv 2507.16397 özet: "SOTA yöntemleri **ortalama %20.79** aşar" (TruFor dâhil ama tek başına değil); %68–73 metinde bulunamadı (Fig. 5?) | İfade düzeltilmeli; %68–73 kaynağı gösterilmeli. ORTA |
| 6 | Vize §6 s.267 | "references.bib (38 kayıt)" | bib'de **39** `@` kaydı | DÜŞÜK |
| 7 | Vize §2.5 s.55, Bildiri s.175 | yeşil işaretçi "v1 makalesi §3.3" | arXiv HTML'de işaretçi **§3.4**'te; §3.3 maske tanımı | Bölüm numarası kontrol edilmeli. DÜŞÜK |
| 8 | PHASE2 s.24 | "10 + 5 + 2 dışlama → train 2.748" | 2.762 − 2.748 = **14**; 10 dejenere çiftin 1'i test, 2'si val'de (`gt_stats.json`, `hypothesis_decisions.md`) → 7+5+2 = 14 | Sayı doğru, cümle yanıltıcı; "train'de 14" yazılmalı. DÜŞÜK |

"2.754" değeri hiçbir belgede geçmiyor (grep: yok); 2.762 = split, 2.748 = dışlama sonrası eğitim kümesi — iki sayı tutarlı ve her yerde doğru kullanılmış.

## Atıf doğrulama (tablo: anahtar, durum, not)

Yöntem: arXiv `abs` sayfaları (WebFetch) ve Crossref `works/{DOI}` kayıtları; başlık, ilk yazarlar, mekân, yıl karşılaştırıldı.

| Anahtar | Durum | Not |
|---|---|---|
| wu2026aiforgedoc (2602.20569) | ✓ | 24 Şub 2026, Wu, Zhou, Xu, Liang…; Tablo 2: TruFor AUC 0.751 / IoU 0.358 / F1 0.434, DTD 0.563/0.020, GPT-4o 0.509 — Vize s.20 ve 241 ile **birebir** |
| wu2026aiforgedocv2 (2604.25213) | ✓ | 28 Nis 2026 (v2 25 Eki 2026); 3.066 sahte, 2AFC 0.501, TruFor 0.599, DocTamper 0.585, 0.962/0.852 — özetle birebir |
| zhao2026docforgebench (2603.01433) | ✓ | 14 yöntem × 8 set, %0.27–4.17, %39–55 ✓; "Pixel-F1 ≥ 0.3 / ≥6 setin" ifadesi özette yok (tam metin) |
| raj2026images25 (2609.13617) | ✓ kayıt / **? iddia** | Başlık/yazarlar/tarih (12 Eyl 2026) doğru. "v3 alfa maskesi, TruFor piksel AUC 0.921 vs 0.918" **özette yok**; HTML tam metin oran-limiti nedeniyle okunamadı → doğrulanamadı |
| wong2025adcdnet (ICCV 2025, 10.1109/ICCV51701.2025.01792) | ✓ | Crossref: ICCV 2025, s.19280–19289, Wong, Zhou, Wu, Si, Zhou; AVG F1 .787 ✓; %20.79 "SOTA'ya karşı" (bkz. uyuşmazlık 5); "MIT lisans" makalede yok (depo lisansı olmalı) |
| chen2024ffdn (10.1007/978-3-031-73414-4_12) | ✓ | ECCV 2024 LNCS, s.200–217 |
| qu2023doctamper (10.1109/CVPR52729.2023.00575) | ✓ | CVPR 2023 |
| guillaro2023trufor (10.1109/CVPR52729.2023.01974) | ✓ | CVPR 2023 |
| mareen2024tgif (10.1109/WIFS61860.2024.10810690) | ✓ | WIFS 2024, Mareen, Karageorgiou, Van Wallendael…; "F1 ≤ 0.1" iddiası kontrol edilmedi |
| yermakov2026gend (10.1109/WACV61042.2026.00082) | ✓ kayıt / **✗ iddia yönü** | WACV 2026, Yermakov, Cech, Matas (+Fritz; Vize s.276'da 4. yazar eksik). Makale: **eşleşmiş** eğitim ↑ AUROC, eşleşmemiş ↓. Vize s.28 ("eşleşmemişte 90.0→56.8") yön doğru; **Bildiri s.36** ("unpaired … from 56.8 to 90.0") **yön ters**. 56.8/90.0 sayıları ana metinde bulunamadı (Fig. 3 / ek) |
| docqt2026 (2605.19688) | ✓ | 19 May 2026, Ronfleux-Corail…; "14.5 F1 puanı" özette yok |
| dhouib2026diffnet (2606.22285) | ✓ | DiffNet model adı; başlık doğru |
| giftbench2026 (2610.01778), textfake2026 (2606.01050), zhang2026gpt4oreceipt (2603.11442), mllmdoc2025 (2508.11021), fsts2025 (2511.12658), freqbias2025 (2511.19886), cocoinpaint2025 (2504.18361), qu2024textsleuth (2412.14816), ctpnet2023 (2308.02158), ma2023imlvit (2307.14863), qu2024omniiml (2411.14823) | ✓ | Başlık/ilk yazarlar/tarih arXiv ile uyumlu |
| wang2022tamperedic13, luo2023icdar, qu2025ostf, luo2025rtm, dong2024tce, li2025ditl2, martineztornes2023receipt, kwon2022catnet, dong2023mvssnet, liu2022psccnet, zhu2025mesorch, kwon2025safire, smeu2025shortcuts, aoudi2026beyond, cai2026zooming, geirhos2020shortcut | ✓ | Tüm DOI'ler Crossref'te; mekân/yıl bib ile birebir (luo2025rtm: PR Ocak 2025 ✓; li2025ditl2: ACM MM 2025 ✓) |
| Vize §2.4/§6 klasik kaynaklar (He 2016, Ronneberger 2015, Ojala 2002, Haralick 1973, Ahmed 1974, Cortes 1995, Breiman 2001, Milletari 2016, Deng 2009, Yosinski 2014, Gonzalez&Woods, OECD 2015) ve Bildiri [40]–[51] (Sauvola, bootstrap, Kendall) | ✗ bib'de yok | Metinde atıf var, `references.bib`'de kayıt **yok**; Vize "Tam BibTeX: references.bib" diyor. Eklenmeli |
| Joya15 GitHub (Vize s.285) | hakemsiz | Bildiri/IEEE taslağında hiç anılmıyor (aşağıda) |

## Değerlendirme bütünlüğü

1. **Test seti dokunulmamış** — `baselines/common/data.py` (s.13: TestingSet'e erişim yok; smoke_train ⊂ train, smoke_val ⊂ val, kesişim 0), `train_unet_p2.py` yalnız `split_table("val")` ile seçim yapıyor; test yalnız `evaluate_p2.py --split test`te okunuyor. ✓
2. **τ\*** — `evaluate_p2.py` s.89–111: yalnız `split=="val" & condition=="clean"` iken hesaplanır, `tau.json`'a yazılır, test/bozulma koşullarında okunur. ✓ Not: val'deki F1@τ\* örnek-içi seçimdir (hafif iyimser, tüm GT'ler için eşit). `unet_bbox_s44` bbox τ\*=0.95 (diğer seed'ler 0.45) → eğri düz; τ\* kararsızlığı raporlanmalı.
3. **AUPRC — smoke (Vize §4) ŞİŞKİN:** `baselines/common/metrics.py` s.93–99 ve `train_unet_p2.py` s.52–53: tüm pozitifler + crop başına ≤20k negatif. 512² crop'ta ~12k pozitif / ~250k negatif → önsel %4.7 yerine ~%38; temiz crop'lar da havuz dışı. AUPRC önsel-bağımlı olduğundan 0.947/0.939 ve val-crop 0.986 **gerçek havuzlanmış AUPRC değil**; AUROC etkilenmez. Faz 2 (`p2metrics.py`) 256-bin histogramla **kesin** ve tüm pikseller üzerinde → 0.923. Vize s.224 "alt-örneklenerek hesaplandı" diyor ama şişme yönünü belirtmiyor; PHASE2 s.155 "karşılaştırılamaz" uyarısı doğru.
4. **Metrik doğruluğu (Faz 2):** AP = Σ(R_n−R_{n−1})P_n (sklearn tanımı) ✓, AUROC trapez ✓, F1/IoU sayımları ✓, n_pos=0 görüntüler F1/AP ortalamasından dışlanmış ve raporlanmış ✓, havuzlanmış ve görüntü-başına ayrımı doğru etiketlenmiş ✓.
5. **Grup-farkında bölme:** `splits.json`'da grup kimliği yok; `samples_forged.csv` yüklemede yok → "val–train grup kesişimi yok" **doğrulanamadı** (yalnız audit raporu beyanı). Test'e dokunan 581 grup train'de bırakılmış (bilinçli; %8.9 sızıntı raporlanmış ✓). TrainingSet/TestingSet 9-haneli id'leri çakışıyor (train∩test 690 id) — `p2data.py` split-önekli yollarla ayırıyor ✓, ama dikkat gerektirir.
6. **Gemini-only modeller:** val seçimi yalnız Gemini görüntüleriyle (`train_unet_p2.py` s.87–88) → Ideogram gerçekten held-out ✓.
7. **"Natif çözünürlük" iddiası:** `infer_fullimage.py` s.26: uzun kenar > 4096 ise 4096'ya küçültülüyor (H p95 = 4096 → ~%5 görüntü, SROIE 4961×7016). Vize §3.2.1 "yeniden boyutlandırma olmadan" ile çelişir. DÜŞÜK.
8. Smoke: model seçimi ve raporlama aynı smoke_val üzerinde (iyimser); rapor bunu "provizyonel" diye belirtiyor ✓.

## İç tutarsızlıklar

| # | Belge(ler) | Tutarsızlık | Önem |
|---|---|---|---|
| 1 | Bildiri_Tam_Metin Tablo 9d (s.331–342) vs §6.6 metin (s.290) ve özet | Tablo: image AUROC bbox **0.838** / piksel 0.896 / glif 0.814, "three-seed mean" etiketli; metin ve özet **0.892/0.892/0.799**. `hypothesis_decisions.md`: 0.838/0.896/0.814 **yalnız s44**. | YÜKSEK |
| 2 | Bildiri s.36 vs Vize s.28 (GenD) | Eşleşmiş/eşleşmemiş eğitimin AUROC etkisi ters yönde yazılmış (bkz. atıf tablosu). | YÜKSEK |
| 3 | PHASE2 s.13/17 vs §9.1 (s.146–147) ve Bildiri §4.3 | Durum tablosu "insan doğrulaması bekliyor / sonraki sürüme eklenecek"; §9.1 ve Bildiri "VLM ile ikame edildi (claude-sonnet-5)". Vize s.157 "VLM yalnız GT denetim aracı" ile uyumlu; PHASE2 tablosu güncellenmeli. VLM model adı doğrulanmalı. | ORTA |
| 4 | Vize s.20 vs Bildiri s.13 | Vize: Joya15 deposu v1/v2'de SegFormer eğittiği için "ilk denetimli eğitim" iddiası **kullanılmamalı**; Bildiri: "No study has examined the effect of the annotation geometry…" (daha dar, savunulabilir) ama Joya15'i hiç anmıyor; Bildiri 20 Ekim tarihli, 27 Ekim bulguları (§6.6-bis) eklenmiş ama ilgili çalışma güncellenmemiş. | ORTA |
| 5 | Vize §3.5 s.134 vs §4.c | "15 epoch, batch 8" (smoke) — tam veri 10 epoch, 16 crop/batch (8 görüntü×2). §4.b "aynı kod hattı" diyor; hiperparametre farkı belirtilmeli. | DÜŞÜK |
| 6 | PHASE2 s.152 vs s.175 | İşaretçi testte "%93" (89 eşleşmiş çift) vs "%74,4" (812 sahte) — iki payda; Bildiri açık yazmış, PHASE2'de karışabilir. | DÜŞÜK |
| 7 | Bildiri Tablo 9a (s.295–297) | piksel/glif satırları "—"; `summary_val_agg.csv`'de rect_0p5_mean piksel 0.314, glif 0.136 mevcut; metin 0.26–0.32 / 0.10–0.14 veriyor. | DÜŞÜK |
| 8 | Vize s.44, 55, 264 | `literature_matrix.xlsx`, `Proje_Konumlandirma_Yenilik_OnIsleme.md`, `Bildiri_Taslagi_IEEE.docx` yüklemede yok (md'leri var). | DÜŞÜK |

## Önerilen düzeltmeler

1. **Bildiri Tablo 9d:** AUROC sütunlarını 3-seed ortalamalarıyla (0.892/0.892/0.799; paste-back 0.872/0.876/0.779; resample 0.866/0.863/0.769) değiştirin veya başlığı "seed 44" yapın; Gemini/Ideogram sütunlarını `hypothesis_decisions.md`'deki 3-seed değerlerle (0.906/0.701 vb.) hizalayın.
2. **GenD cümlesi (Bildiri s.36):** "GenD [34] shows that training on *paired* real/fake data from the same source is essential; unpaired training lowers cross-dataset AUROC" biçimine çevirin; 56.8/90.0 sayılarını yalnız makalede yerini (şekil/ek) gösterebiliyorsanız tutun.
3. **Smoke AUPRC (Vize §4 D/E, s.224):** "Pixel-AUPRC (alt-örneklenmiş; negatifler ≤20k/crop → şişkin, Faz 2 kesin değerleriyle karşılaştırılamaz)" notu ekleyin; mümkünse `unet_*_val_probs.npz`'den kesin havuzlanmış AUPRC'yi yeniden hesaplayıp yazın.
4. **Vize s.257:** "Tohumlar arası std ≤ 0,01" → "val-crop AUPRC'de ≤ 0,002; tam-görüntü AUPRC'de 0,015, F1'de 0,03".
5. **Vize s.25:** "TruFor'u ortalama %20.8 aşar" → "en iyi karşılaştırma yöntemlerini beş bozulmada ortalama %20.79 aşar"; %68–73 için şekil/tablo numarası verin veya kaldırın.
6. **Vize s.20 (2609.13617):** "piksel AUC 0.921 vs 0.918" iddiasını makaledeki tablo/bölümle gerekçelendirin; doğrulanamazsa niteliksel ifadeye dönün.
7. **references.bib:** Klasik kaynakları (He, Ronneberger, Ojala, Haralick, Ahmed, Cortes&Vapnik, Breiman, Milletari, Deng, Yosinski, Gonzalez&Woods, Sauvola, Efron, Kendall, OECD) ekleyin; Vize s.267'yi "39 kayıt" yapın; s.276'ya "Fritz M." ekleyin.
8. **PHASE2 s.13/17:** durum tablosunu VLM ikamesiyle güncelleyin; **Bildiri §2**'ye Joya15 deposunu (hakemsiz, doğrulanmamış) ekleyip yenilik iddiasını Vize s.20'deki dar biçimde tutun.
9. **Vize Tablo D/E:** RGB clean satırında tek değer (0.707 veya her ikisinde 0.708) kullanın; Tablo E'nin ayrı koşudan geldiğini dipnotlayın.
10. **Vize §3.2.1 / Bildiri §5:** ">4096 px uzun kenar 4096'ya küçültülür (çıkarım)" istisnasını yazın.
11. **Teslim paketi:** `data/AIForge-Doc-v1/README.md`, `dataset_info.json`, `samples_forged.csv`, `excluded_ids.json`, `green_marker_summary.csv` yüklemeye eklenmeli; aksi hâlde §3.1 kart iddiaları, grup-farkında bölme, 2.748 ve Tablo 14 birincil kaynaktan doğrulanamaz.

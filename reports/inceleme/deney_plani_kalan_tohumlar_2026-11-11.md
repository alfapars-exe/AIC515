# Deney planı — kalan tek tohumlu koşuların 43 ve 44 tohumları ve yeni deney: yalnız Gemini üzerinde işaretçisiz eğitim

**Yazıldığı an:** 11 Kasım 2026, koşudan önce · **Yöntem:** Lab Rat `experiment-design` ve `launch`
**Kuyruk:** `kod/phase3/q5_kalan_tohumlar.ps1` · **Donanım:** RTX 5080 16 GB

## 1. Soru

1. **Tekrarlanabilirlik.** Bildiri ve raporda tek tohuma (42) dayanan satırlar 43 ve 44 tohumlarında da aynı sonucu veriyor mu? Kapsam: işaretçisiz eğitilen piksel-değişim ve glif U-Net'leri, birleşik kayıplı glif U-Net, DeepLabV3+ (üç referans), 20 epokluk eğitim (üç referans), YOLO11 ablasyonları (altı varyant) ve YOLO11s ana modelinin dayanıklılık koşulları ile sondaları.
2. **Yeni deney.** Yalnız Gemini görüntüleriyle eğitilen modeller Ideogram görüntülerinde çöküyor (kutu U-Net: görüntü başı AP 0.909 → 0.412). İşaretçi kalıntısı Gemini görüntülerinin %83,5'inde, Ideogram görüntülerinin %1,4'ünde var. Çöküşün ne kadarı bu kalıntıya dayanıyor? Yalnız Gemini modelleri işaretçisi onarılmış görüntülerle yeniden eğitilirse Ideogram'a aktarım düzeliyor mu?

## 2. Tasarım

Tekrar koşularında tek değişen etken tohumdur; komutlar, hiperparametreler, veri bölmesi ve değerlendirme kodu s42 koşularıyla aynıdır (`q1_unet_train.ps1`, `q2_yolo_train.ps1` 30 epokla, `q3a_eval_unet.ps1`, `q3b_eval_yolo.ps1`, `run_v2.ps1`). Yeni deneyde mevcut yalnız-Gemini modellerine göre tek değişen etken işaretçi onarımıdır (piksel-değişim ve glif referanslarında, tüm-üretici işaretçisiz modellerde olduğu gibi, işaretçi pikselleri referanstan da çıkarılır: `_nm`).

| Blok | Koşular | Komut | Sayı |
|---|---|---|---|
| A1 | `unet_pixel_eps8_morph_nm_mr_s{43,44}`, `unet_glyph_v2_nm_mr_s{43,44}` | `train_unet_p3.py --gt <ref>_nm --marker-repair --seed S` (`P3_MARKER_REPAIR=1`) | 4 |
| A1 | `unetjoint_glyph_v2_s{43,44}` | `train_unet_p3.py --gt glyph_v2 --aux-bbox 0.5 --seed S` | 2 |
| B (yeni) | `unet_bbox_geminionly_mr_s{42,43,44}`, `unet_pixel_eps8_morph_nm_geminionly_mr_s{42,43,44}`, `unet_glyph_v2_nm_geminionly_mr_s{42,43,44}` | `train_unet_p3.py --gt <ref> --marker-repair --train-filter gemini_only --seed S` | 9 |
| A2 | `dlv3p_{bbox,pixel_eps8_morph,glyph_v2}_s{43,44}` | `train_unet_p3.py --arch dlv3p --gt <ref> --seed S` | 6 |
| A3 | `unet_{bbox,pixel_eps8_morph,glyph_v2}_s{43,44}_e20` | `train_unet_p2.py --gt <ref> --seed S --epochs 20 --patience 6 --tag e20` | 6 |
| D | YOLO11s ana modeli s43, s44 (eğitim yok): beş test-zamanı koşulu ve sondalar | `infer_to_maps.py --condition C`, `evaluate_p2.py --condition C`, `probes_p3.py --yolo` | 0 |
| C | YOLO11 ablasyonları s43, s44: yalnız Gemini, tam görüntü 1280, varsayılan çoğaltma, özgün negatif karolar, 11n, 11m | `train_yolo.py … --epochs 30 --seed S --workers 6` (11m: `--batch 4`) | 12 |

Toplam 39 eğitim koşusu. Sabitler: U-Net ve DeepLabV3+ için ResNet-18 kodlayıcı, 512 piksel kırpıntı, görüntü başına 2 kırpıntı, parti 16, AdamW 1e-4, kosinüs, 10 epok, sabır 4, bf16, seçim doğrulama kırpıntılarında. YOLO için 1024 piksel karo (tam görüntü varyantında 1280), parti 8, AdamW 1e-3, sabır 10, ihtiyatlı çoğaltma.

**Yeni deneyde ayrı tutma.** `--train-filter gemini_only` eğitimden Ideogram görüntülerini çıkarır (2.454 eğitim görüntüsü) ve model seçiminde kullanılan doğrulama kırpıntılarını da Gemini ile sınırlar (446 görüntü, 892 kırpıntı); Ideogram görüntüleri eğitimde ve seçimde hiç görülmez. Süzgeç `train_unet_p2.py` içindekiyle aynıdır. Bu seçenek `train_unet_p3.py` betiğine bu tur eklendi; ilk sürümde seçim kırpıntıları süzülmüyordu, duman testinden önce düzeltildi (1 epok, 5 parti: `train 2454 | val 446`, `n_val_crops 892`).

**Blok sırası.** A1, B, A2, A3 (U-Net ailesi), D (kısa), C (en uzun). Kuyruk istenen anda durdurulabilir; biten adımlar `.done` işaretiyle korunur ve kuyruk kaldığı yerden sürer.

## 3. Değerlendirme (s42 ile aynı aşamalar)

| Adım | Komut | Çıktı |
|---|---|---|
| `e_mrnm_s4344` | `run_eval_p3.py --stages val,test,robust,controls` | `phase2/experiments/results/` (özet ve görüntü başı dosyalar, `tau.json`) |
| `e_jointglyph_s4344`, `e_dlv3p_s4344` | `run_eval_p3.py --stages val,test,robust,controls,ablation,rectfit` | aynı |
| `e_gomr`, `e_e20_s4344` | `run_eval_p3.py --stages val,test` | aynı |
| `p_probes_unet_q5a` … `q5d`, `p_probes_yolo_s4344` | `probes_p3.py --tag …` | `phase3/results/per_image_probes_p3_<etiket>.csv` |
| `v2_infer_q5a`, `v2_infer_q5b`, `v2_infer_q5y`, `v2_eval_primary_q5`, `v2_eval_primary_q5y` | `v2_eval.py infer` ve `eval` (birincil 567 çift) | `phase3_v2/results/*_primary.csv` |
| `yi_*`, `ye_val_q5`, `ye_test_q5`, `ye_rob_*_s4344` | `infer_to_maps.py`, `evaluate_p2.py` | YOLO olasılık haritaları, kutu listeleri ve özetler |
| `y_detmetrics_{val,test}_q5`, `_q5y` | `detection_metrics.py` (önceki model listesi + yeni modeller) | `phase3_yolo/results/detection_{val,test}.csv` |
| sentez (elle, koşudan sonra) | `synthesize_p3.py`, `seed_extra_stats.py`, `v2_report.py`, `v2_marker_split.py`, `v2_space_meta.py`, `make_space_meta.py` | tablolar |

`p_probes_unet_q5d` yeni deneyin denetimidir: yalnız-Gemini kutu modellerinin standart ve işaretçisiz sürümlerinde (üçer tohum) kalıntı onarımının etkisi ölçülür. Üzerine yazan iki betik (`v2_eval.py eval`, `detection_metrics.py`) önceki modellerin tamamıyla birlikte çağrılır; diğerleri model bazında birleştirir. `rectfit` aşaması bu turdan itibaren eski satırları koruyarak birleştirir (`kod/phase2/experiments/probes.py`).

## 4. Öngörüler (koşmadan önce)

Referans değerler koşudan önce diskte bulunan dosyalardan okunmuştur: `phase3/results/T1_dual_gt_val.csv`, `T2_test.csv`, `T3_probes.csv`, `T4_heldout.csv`, `T5_yolo_ablation.csv`, `T6_robustness.csv`, `phase3_v2/results/T_v2.csv`. `T4_heldout.csv` dosyasına tüm üreticilerle eğitilen işaretçisiz modellerin satırları 11 Kasım'da, koşudan önce, mevcut tahminlerden eklendi (kutu U-Net: Ideogram 0.633 [0.573, 0.695], Gemini 0.898; standart model 0.629 ve 0.912). Bu satır yeni deneyin öngörüsünü belirledi: tüm üreticilerle eğitimde işaretçinin onarılması Ideogram puanını değiştirmiyor.

AUPRC havuzlanmış piksel AUPRC'sidir; AP görüntü başı ortalama AP'dir; "ortalama" üç tohumun (42–44) ortalamasıdır.

### 4.1 Tekrar koşuları

| # | Öngörü | Yanlışlama ölçütü |
|---|---|---|
| K1 | İşaretçisiz piksel-değişim U-Net işaretçiden bağımsız kalır: kalıntı onarımında ΔAP ortalaması ≥ −0.03 (s42: +0.038 [+0.021, +0.057]; standart piksel modeli −0.247) | Ortalama ΔAP < −0.08 |
| K2 | Aynı modelin doğrulama piksel-değişim AUPRC ortalaması 0.51–0.58 (s42: 0.544; standart 0.610 ± 0.012): işaretçisiz eğitimin bedeli sürer | Ortalama < 0.49 ya da ≥ 0.60 |
| K3 | Aynı model v2 birincil kümede piksel-değişim ölçütünde tablonun en iyisi olarak kalır: ortalama 0.40–0.46 (s42: 0.430; işaretçisiz kutu U-Net 0.391 ± 0.006; standart piksel U-Net 0.246 ± 0.019) | Ortalama < 0.391 ("en iyi" ifadesi düşer); ortalama ≤ 0.30 (standart modelden iyi olduğu ifadesi de düşer) |
| K4 | İşaretçisiz glif U-Net: ΔAP ortalaması ≥ −0.03 (s42: +0.001); doğrulama glif AUPRC ortalaması 0.63–0.69 (s42: 0.661; standart 0.676 ± 0.019); v2 glif AUPRC ortalaması 0.40–0.48 (s42: 0.446; standart 0.304 ± 0.010; işaretçisiz piksel modeli glif ölçütünde 0.416) | ΔAP < −0.08; doğrulama ortalaması < 0.61; v2 ortalaması ≤ 0.34; v2 ortalaması işaretçisiz piksel modelinin glif ortalamasının altına inerse "en iyi glif yerelleştirici" ifadesi düşer |
| K5 | Birleşik kayıp glif modeline yaramaz: glif ölçütünde ortalama 0.60–0.67 (s42: 0.636; standart 0.676 ± 0.019), kutu ölçütünde 0.34–0.40 (s42: 0.370; standart 0.351 ± 0.015) | Glif ortalaması > 0.686 ya da kutu ortalaması > 0.45 |
| K10 | DeepLabV3+ kutu modeli: doğrulama kutu AUPRC'si iki yeni tohumda 0.92–0.95 (s42: 0.939), piksel-değişim ölçütünde ortalama 0.36–0.40 (s42: 0.382) | Bir tohumda kutu AUPRC < 0.90 |
| K11 | DeepLabV3+'ta her referans kendi modelini birinci sıraya koyar (üç ölçüt, her tohum) | Herhangi bir tohumda sıranın bozulması |
| K12 | DeepLabV3+ kutu modeli işaretçiye bağımlıdır: ΔAP ortalaması ≤ −0.12 (s42: −0.174 [−0.206, −0.145]) | Ortalama > −0.08 |
| K13 | DeepLabV3+ kutu modeli v2'de U-Net gibi düşer: kutu AUPRC ortalaması 0.48–0.60 (s42: 0.539; U-Net 0.520 ± 0.018) | Ortalama ≥ 0.65 |
| K14 | 20 epok eğitim kendi ölçütünde 10 epoktan iyi değildir: kutuda fark ±0.02 içinde, piksel-değişim ve glifte en çok +0.015 (s42, 20 epok: 0.926, 0.594, 0.645; 10 epok: 0.923 ± 0.015, 0.610 ± 0.012, 0.676 ± 0.019) | Herhangi bir referansta 20 epok ortalaması 10 epok ortalamasını 0.03'ten fazla geçer |
| K15 | YOLO11s ana modeli işaretçiye U-Net'ten daha az bağımlıdır: ΔAP ortalaması −0.03 ile −0.10 arasında (s42: −0.062 [−0.075, −0.051]; kutu U-Net −0.222) | Ortalama < −0.12 ya da > −0.02 |
| K16 | YOLO11s ana modelinin dayanıklılık örüntüsü sürer: gri seviye ve JPEG 75 kaybı en az %35 (s42: −%58,6 ve −%48,5); bulanıklık ve yeniden ölçekleme kaybı %5'in altında (s42: −%1,4 ve −%0,7) | Ortalamada gri ya da JPEG kaybı < %25; bulanıklık kaybı > %10 |
| K17 | YOLO model boyutu: test kutu AUPRC ortalaması 11n ve 11m için 0.955–0.975 (s42: 0.963 ve 0.969; 11s 0.969 ± 0.006); 11m, 11s'yi 0.01'den fazla geçmez | 11n ortalaması < 0.95 ya da 11m − 11s > 0.01 |
| K18 | Tam görüntü 1280: test kutu AUPRC ortalaması karolu modelin en az 0.02 altında (s42: 0.932 ve 0.969), kutu AP@0.5 en az 0.05 üstünde (s42: 0.960 ve 0.880) | AUPRC farkı < 0.01 ya da AP@0.5 kazancı < 0.03 |
| K19 | Varsayılan çoğaltma test kutu AUPRC'sini değiştirmez: fark ±0.01 içinde (s42: 0.971 ve 0.969) | Mutlak fark > 0.015 |
| K20 | Özgün negatif karolar test kutu AUPRC'sini değiştirmez: fark ±0.01 içinde (s42: 0.971 ve 0.969) | Mutlak fark > 0.015 |
| K21 | Yalnız Gemini ile eğitilen YOLO11s Ideogram'da U-Net kadar çökmez: Ideogram AP ortalaması 0.70–0.80 (s42: 0.752 [0.703, 0.806]), kaynak-eşlenmiş Gemini ile fark ≤ 0.25 (s42: 0.186) | Ortalama < 0.65 ya da > 0.85 |

### 4.2 Yeni deney (yalnız Gemini üzerinde işaretçisiz eğitim)

Referans (`T4_heldout.csv`, doğrulama + test, üç tohum): standart yalnız-Gemini kutu U-Net Ideogram 0.412 [0.341, 0.481], kaynak-eşlenmiş Gemini 0.909 (fark 0.497); piksel-değişim modeli kutu ölçütüyle 0.264 ve 0.567; glif modeli 0.166 ve 0.371. Tüm üreticilerle eğitilen kutu U-Net 0.629 ve 0.912; işaretçisiz sürümü 0.633 ve 0.898.

İki açıklama yarışıyor. (a) Çöküş işaretçiye dayanır: yalnız-Gemini modeli alanı kalıntıdan bulur, Ideogram'da kalıntı olmadığı için düşer; onarımla eğitim Ideogram puanını yükseltir. (b) Çöküş üreticiye özgüdür: model Gemini'nin yeniden çizim izini öğrenir; onarım bunu değiştirmez. Tüm üreticilerle eğitimde onarımın Ideogram puanını değiştirmemesi (0.629 ve 0.633) ve onarımın kalıntısız Gemini görüntülerinde puanı yükseltmesi (test, 0.722 → 0.762) (b)'yi destekliyor. Öngörü (b) yönündedir.

| # | Öngörü | Yanlışlama ölçütü |
|---|---|---|
| K6 | İşaretçisiz eğitim yalnız-Gemini kutu U-Net'i kurtarmaz: Ideogram AP ortalaması 0.36–0.50 (standart 0.412), kaynak-eşlenmiş Gemini ile fark ≥ 0.35; aynı görüntülerde eşli fark (işaretçisiz − standart) −0.05 ile +0.09 arasında | Ideogram AP ≥ 0.55 (çöküşün yarıdan fazlası işaretçiyle açıklanır) ya da ≤ 0.33 |
| K7 | Piksel-değişim modeli (kutu ölçütüyle): Ideogram AP 0.22–0.33 (standart 0.264); Gemini AP 0.46–0.53'e iner (standart 0.567), fark Gemini puanı düştüğü için daralır | Ideogram AP ≥ 0.36 |
| K8 | Glif modeli (kutu ölçütüyle): Ideogram AP 0.13–0.22 (standart 0.166) | Ideogram AP ≥ 0.26 |
| K9 | Denetim: yalnız-Gemini standart kutu modelinde kalıntı onarımı ΔAP ≤ −0.20 (tüm-üretici standart model −0.222); işaretçisiz sürümünde ΔAP ≥ −0.03 | İşaretçisiz sürümde ΔAP < −0.08 (onarım bağımlılığı gidermemiştir; K6–K8 yorumlanamaz); standart sürümde ΔAP > −0.15 |

Sonuç hangi yönde çıkarsa çıksın bildiri ve rapordaki ilgili cümleler üç tohumlu değerlerle yeniden yazılır; öngörü tutmazsa iddia daraltılır ya da değiştirilir. K6 yanlışlanırsa (Ideogram AP ≥ 0.55) "her model Ideogram'da çöker" bulgusu "çöküşün büyük kısmı işaretçi kalıntısına dayanır" biçiminde yeniden yazılır.

## 5. Kaynak tahmini

s42 koşularının `.done` sürelerinden (`phase3/logs/`): U-Net eğitimi 502–693 s, DeepLabV3+ 521–574 s, 20 epok 846–1.004 s; değerlendirme altı aşamada model başına yaklaşık 570 s, dört aşamada 450 s, iki aşamada 120 s; sondalar model başına yaklaşık 700 s (YOLO 800 s); YOLO eğitimi varyanta göre 32–87 dk.

| Blok | Tahmin |
|---|---|
| A1 | eğitim 54 dk, değerlendirme 52 dk, sondalar 46 dk, v2 çıkarım 6 dk: yaklaşık 2,6 saat |
| B | eğitim 72 dk, değerlendirme 18 dk: yaklaşık 1,5 saat |
| A2 | eğitim 56 dk, değerlendirme 57 dk, sondalar 23 dk, v2 çıkarım ve değerlendirme 27 dk, tespit ölçütü 21 dk: yaklaşık 3,1 saat |
| A3 | eğitim 91 dk, değerlendirme 13 dk, sondalar 69 dk + yeni deneyin denetim sondaları 69 dk: yaklaşık 4 saat |
| D | çıkarım 13 dk, değerlendirme 3 dk, sondalar 27 dk: yaklaşık 0,7 saat |
| C | eğitim 2 × 308 dk, harita çıkarımı 46 dk, değerlendirme 8 dk, tespit ölçütü 31 dk, v2 33 dk: yaklaşık 12,2 saat |
| Toplam | yaklaşık 24 saat; U-Net blokları yaklaşık 11 saat. Disk: 27 U-Net ailesi ağırlığı (yaklaşık 1,5 GB), 12 YOLO koşusu, tahmin haritaları (model ve bölme başına yaklaşık 70 MB); boş alan 399 GB |

## 6. Ön-uçuş kontrolü

| Kontrol | Durum |
|---|---|
| Yapılandırma farkı | Tekrar koşuları s42 komutlarıyla aynı, yalnız `--seed` değişir (YOLO s42 koşularının `args.yaml` dosyaları: 30 epok, sabır 10, parti 8, 11m için 4, 6 işçi). Yeni deney: mevcut yalnız-Gemini komutuna `--marker-repair` ve `_nm` referansı eklenir |
| Koşu adları | Projenin adlandırma kuralı: `<mimari>_<referans>[_geminionly][_mr]_s<tohum>[_e20]`; YOLO: `aiforge-<veri>-<model>-img<boyut>-<çoğaltma>-s<tohum>` |
| Yollar | `data/AIForge-Doc-v1`, `data/AIForge-Doc-v2`, `phase3/marker_repaired`, `phase2/gt/pixel_eps8_morph_nm`, `phase2/gt/glyph_v2_nm`, `phase3_yolo/yolo_data/{tiles,full,tiles_authneg,tiles_geminionly}/data.yaml`, `phase3_yolo/yolo11{n,s,m}.pt` mevcut |
| Betik denetimi | `q5_kalan_tohumlar.ps1` PowerShell ayrıştırıcısından hatasız geçti; yeni `--train-filter` yolu duman testinden geçti (artıkları silindi) |
| İzleme | `phase3/logs/queue.log`, adım günlükleri ve `.done` işaretleri (deney izleyici yok) |
| Bitiş tahmini | Başlangıçtan yaklaşık 24 saat sonra (yerel saat, UTC+3); U-Net blokları yaklaşık 11 saat sonra |
| Kod = gönderilmiş commit | Koşudan önce çalışma ağacı GitHub'a eşitlenir; commit kimliği `phase3/logs/queue_notes.txt` dosyasına yazılır. Bu plan ve kuyruk betiği o commit'in içindedir |
| Süreç ağacı | Kuyruk, masaüstü uygulamasının süreç ağacının dışında başlatılır (9 Kasım kesintisinin dersi); uyku engelleyici ayrı süreçtir |
| GPU | Boşta; diğer uygulamalar 4,3 GB bellek tutuyor (s42 koşularındaki ortamla aynı düzen) |
| Sonuçların yedeği | `_arsiv/_backup/2026-11-11_q5_oncesi/` (mevcut sonuç dosyaları; eski satırların değişmediği koşudan sonra karşılaştırılır) |

## 7. Analiz planı (koşudan önce sabitlendi)

1. **Tablolar.** `synthesize_p3.py`: T1–T7 üç tohum ortalaması ± standart sapma; sondalarda görüntü başı değerler önce tohumlar üzerinden ortalanır, güven aralığı kaynak belge grupları üzerinden bootstrap ile hesaplanır. YOLO ablasyonları için varyant bazında üç tohumlu özet tablo eklenir.
2. **Eşli karşılaştırmalar** (`seed_extra_stats.py` genişletilir; görüntü başı AP, tohum ortalaması, grup bootstrap, B = 1000): yalnız-Gemini işaretçisiz − standart (Ideogram ve kaynak-eşlenmiş Gemini görüntülerinde, üç referans); 20 epok − 10 epok (doğrulama ve test, kendi referansı); birleşik glif − standart glif; YOLO varyantı − ana model (test, kutu referansı).
3. **v2.** `v2_report.py`, `v2_marker_split.py` (tohum ortalamalı, grup bootstrap), `v2_space_meta.py`.
4. **Eski satırların denetimi.** Yedekle karşılaştırma: `summary_val.csv`, `summary_test.csv`, `robustness.csv`, `probes_boundary.csv`, `probes_rectfit.csv`, `summary_v2_primary.csv`, `image_level_v2_primary.csv`, `detection_{val,test}.csv`, `summary_val_<koşul>.csv`.
5. **Başarısız adımlar.** `FAIL` alan adımın günlüğü okunur, neden bulunur (Lab Rat `debug`), adım yeniden koşulur. Bir blok tamamlanmadan o bloğun satırları tablolara tohum sayısı belirtilmeden girmez; tamamlanamayan koşu olursa satır kaç tohuma dayandığıyla birlikte raporlanır.
6. **Raporlama.** Her öngörü için sonuç ve karar (tuttu, tutmadı) bu dosyanın sonuna eklenir; öngörülerin kapsamadığı bulgular ayrıca yazılır. Yukarıdaki bölümlere koşudan sonra dokunulmaz.

Koşu sonrası tek tohumlu kalanlar: YOLO kontrol kaynakları (özgün, yapıştır-geri, yeniden örnekleme) ve hız ölçümü (tohuma bağlı değil).

## 8. Sonuç (ara, U-Net blokları ve D bloğu; 11 Kasım 2026, 14:45'te yazıldı; yukarıdaki bölümlere dokunulmadı)

**Koşu.** A1, B, A2, A3 ve D blokları hatasız bitti (`phase3/logs/queue.log`: `FAIL` yok). Eski modellerin satırları yedekle karşılaştırıldı (`summary_val.csv`, `summary_test.csv`, `robustness.csv`, `probes_boundary.csv`, `probes_rectfit.csv`, `per_image_rectfit.csv`, `tau.json`); değişen yok, yalnız yeni satırlar eklendi. C bloğu (YOLO ablasyonları) sürüyor; K17–K21 ve YOLO tarafı kuyruk bitince değerlendirilecek. Bu ara sentez depoya gönderilmedi.

Kaynaklar: `phase3/results/T1_dual_gt_val.csv`, `T2_test.csv`, `T3_probes.csv`, `T4_heldout.csv`, `T6_robustness.csv`, `phase3_v2/results/T_v2.csv` (üç tohum ortalaması ± standart sapma; sondalarda tohum ortalaması ve grup bootstrap).

| # | Sonuç | Karar |
|---|---|---|
| K1 | İşaretçisiz piksel U-Net onarım ΔAP +0.054 [+0.037, +0.072] | Tuttu |
| K2 | Doğrulama piksel AUPRC 0.507 ± 0.045 (0.544, 0.458, 0.521) | Öngörülen aralığın (0.51–0.58) az altında; yanlışlama eşiği (< 0.49) aşılmadı. Bedel tek tohumda görülenden büyük |
| K3 | v2 piksel AUPRC 0.422 ± 0.015; tablonun en iyisi (işaretçisiz kutu U-Net 0.391) | Tuttu |
| K4 | İşaretçisiz glif U-Net: ΔAP +0.009 [+0.002, +0.016]; doğrulama glif 0.644 ± 0.014; v2 glif 0.432 ± 0.015 (işaretçisiz piksel modeli 0.397) | Tuttu |
| K5 | Birleşik glif U-Net: glif 0.641 ± 0.063, kutu 0.395 ± 0.073 | Tuttu; tohumlar arası fark büyük (glif 0.636, 0.581, 0.707) |
| K10 | DeepLabV3+ kutu: 0.939, 0.925, 0.923; piksel ölçütünde 0.382 ± 0.002 | Tuttu |
| K11 | Üç tohumda da her referans kendi modelini birinci sıraya koyuyor | Tuttu |
| K12 | DeepLabV3+ kutu onarım ΔAP −0.173 [−0.203, −0.146] | Tuttu |
| K13 | DeepLabV3+ kutu v2 AUPRC 0.532 ± 0.009 | Tuttu |
| K14 | 20 epok − 10 epok: kutu 0.912 − 0.923 = −0.011; piksel 0.591 − 0.610 = −0.019; glif 0.598 − 0.676 = −0.078 | Tuttu; 20 epok hiçbir referansta yararlı değil, glifte belirgin biçimde kötü ve kararsız (± 0.082) |
| K15 | YOLO11s onarım ΔAP −0.058 [−0.068, −0.049] (üç tohum) | Tuttu |
| K16 | YOLO11s göreli kayıp: gri −%57,5, JPEG 75 −%45,2, bulanıklık −%1,3, yeniden ölçekleme −%0,6 | Tuttu |
| **K6** | Yalnız Gemini + işaretçisiz kutu U-Net: Ideogram 0.463 [0.400, 0.528], kaynak-eşlenmiş Gemini 0.874; fark 0.412 (standart yalnız-Gemini: 0.412 ve 0.909, fark 0.497) | Tuttu |
| **K7** | Piksel modeli (kutu ölçütü): Ideogram 0.317, Gemini 0.472 (standart: 0.264 ve 0.567) | Tuttu |
| **K8** | Glif modeli (kutu ölçütü): Ideogram 0.203 (standart 0.166) | Tuttu |
| **K9** | Onarım ΔAP: yalnız-Gemini standart kutu −0.275 [−0.313, −0.238]; işaretçisiz sürüm +0.009 [−0.004, +0.023] | Tuttu (denetim geçerli: onarım bağımlılığı kaldırıyor) |

**Yeni deneyin yorumu.** İşaretçi kalıntısını eğitimden kaldırmak yalnız-Gemini modellerinin Ideogram'daki çöküşünü yalnızca biraz azaltıyor: kutu modelinde Ideogram AP 0.412'den 0.463'e çıkıyor, Gemini ile fark 0.497'den 0.412'ye iniyor. Farkın yaklaşık altıda biri işaretçiye, kalanı üreticiye özgü izlere bağlanabilir. Tüm üreticilerle eğitilen modelde de onarım Ideogram puanını değiştirmiyordu (0.629 ve 0.633). Bildirideki "ayrı tutulan üreticide çöküş" bulgusu güçleniyor: çöküş işaretçi kısayolundan bağımsız.

**Not (öngörülerin kapsamadığı).** İşaretçisiz piksel U-Net'in tohum 43'ü belirgin biçimde zayıf (doğrulama piksel AUPRC 0.458); 20 epoklu glif ve birleşik glif modelleri de tohuma duyarlı. Bildiride bu satırlar ortalama ± standart sapma ile verilecek.

## 9. Sonuç (kuyruk sonu; 12 Kasım 2026, 01:30'da yazıldı; yukarıdaki bölümlere dokunulmadı)

**Koşu.** Kuyruk 11 Kasım 03:37:33'te başladı ve 23:30:32'de bitti (yaklaşık 19 saat 53 dakika; tahmin 24 saat). 99 adımın tamamı `DONE`, `FAIL` yok. YOLO eğitimleri tahminden kısa sürdü (s43 ve s44 için 25–72 dk; s42'de 32–87 dk).

**Ek hesap (K6'nın eşli ölçütü).** Yalnız-Gemini kutu U-Net, işaretçisiz − standart, aynı görüntülerde görüntü başı AP (üç tohum ortalaması, grup bootstrap, B = 1000): Ideogram +0.051 [+0.004, +0.096], kaynak-eşlenmiş Gemini −0.035 [−0.047, −0.023]. Piksel modeli (kutu ölçütü): +0.053 [+0.029, +0.079] ve −0.095 [−0.110, −0.081]. Glif modeli: +0.037 [+0.017, +0.057] ve −0.055 [−0.063, −0.047]. K6'nın eşli fark aralığı (−0.05 ile +0.09) tuttu.

**YOLO öngörüleri** (kaynak: `phase3/results/T1_dual_gt_val.csv`, `T2_test.csv`, `T4_heldout.csv`; üç tohum):

| # | Sonuç | Karar |
|---|---|---|
| K17 | Test kutu AUPRC: 11n 0.966 ± 0.003, 11m 0.966 ± 0.008, 11s 0.969 ± 0.006; 11m − 11s = −0.003 | Tuttu; model boyutunun etkisi yok |
| K18 | Tam görüntü 1280: test AUPRC 0.936 ± 0.005 (karolu 0.969; fark −0.033), AP@0.5 0.961 ± 0.003 (karolu 0.880; fark +0.081) | Tuttu |
| K19 | Varsayılan çoğaltma: 0.973 ± 0.002 (fark +0.004) | Tuttu |
| K20 | Özgün negatif karolar: 0.969 ± 0.002 (fark 0.000) | Tuttu |
| K21 | Yalnız-Gemini YOLO11s: Ideogram AP 0.725 [0.675, 0.778], eşlenmiş Gemini 0.938, fark 0.214 | Tuttu |

**Toplam.** 21 öngörünün 21'i yanlışlama ölçütünü geçti. Biri (K2) öngörülen aralığın biraz altında kaldı: işaretçisiz piksel U-Net doğrulama piksel AUPRC 0.507 ± 0.045 (aralık 0.51–0.58, yanlışlama < 0.49).

**İddialara etkisi.**
1. Ayrı tutulan üretici: işaretçi kalıntısı eğitimden kaldırıldığında yalnız-Gemini modellerinin Ideogram'daki çöküşü yalnızca biraz azalıyor (kutu U-Net 0.412 → 0.463, eşli +0.051); Gemini puanı düşüyor (0.909 → 0.874). Çöküşün küçük bir kısmı işaretçiye, büyük kısmı üreticiye özgü izlere bağlı. YOLO11s'de de üç tohumda çöküş sürüyor (0.725; fark 0.214).
2. Önceki tek tohumlu bulguların hepsi üç tohumda tekrarlandı; sayılar ortalama ± standart sapma ile verilecek. 20 epoklu eğitim hiçbir referansta yararlı değil, glifte belirgin biçimde kötü ve kararsız (0.598 ± 0.082).
3. YOLO ablasyonları üç tohumda değişmedi: model boyutu, çoğaltma ve özgün negatif karolar test AUPRC'sini ±0.004 içinde bırakıyor; tam görüntü girdisi AUPRC'yi düşürüp AP@0.5'i yükseltiyor.

**Tohum etiketi.** Faz 3'te tek tohumlu satır kalmadı. Tek tohumlu kalanlar: YOLO kontrol kaynakları (özgün, yapıştır-geri, yeniden örnekleme) ve hız ölçümü.

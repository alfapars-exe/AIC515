# Deney planı — ek tohumlar (43, 44): işaretçi-onarımlı kutu U-Net ve birleşik kayıplı piksel U-Net

**Yazıldığı an:** 10 Kasım 2026, koşudan önce · **Yöntem:** Lab Rat `experiment-design` ve `launch`
**Kuyruk:** `kod/phase3/q4_seeds_mr_joint.ps1` · **Donanım:** RTX 5080 16 GB

## 1. Soru

Bildirideki iki sonuç tek tohuma (42) dayanıyor:
1. İşaretçisi onarılmış görüntülerle eğitilen kutu U-Net işaretçi bağımlılığını kaybediyor, kutu puanının çoğunu koruyor ve işaretçisiz AIForge-Doc v2'ye daha iyi aktarılıyor.
2. Kutu yardımcı kayıplı (birleşik) piksel U-Net, piksel modelini hem piksel-değişim hem kutu ölçütünde iyileştiriyor.

Bu iki sonuç 43 ve 44 tohumlarında da görülüyor mu?

## 2. Tasarım

Tek değişen etken tohumdur. Betikler, hiperparametreler, veri bölmesi ve değerlendirme kodu s42 koşularıyla aynıdır (`q1_unet_train.ps1`, `q3a_eval_unet.ps1`, `run_v2.ps1`).

| Koşu | Komut (yalnız `--seed` değişir) | Eğitim verisi |
|---|---|---|
| `unet_bbox_mr_s43`, `unet_bbox_mr_s44` | `train_unet_p3.py --gt bbox --marker-repair --seed S` (`P3_MARKER_REPAIR=1`) | 2.762 eğitim görüntüsü, işaretçisi onarılmış |
| `unetjoint_pixel_eps8_morph_s43`, `_s44` | `train_unet_p3.py --gt pixel_eps8_morph --aux-bbox 0.5 --seed S` | 2.762 eğitim görüntüsü |

Sabitler: U-Net (ResNet-18 kodlayıcı), 512 piksel kırpıntı, görüntü başına 2 kırpıntı, parti 16, AdamW 1e-4, kosinüs, 10 epok, sabır 4, bf16; model seçimi doğrulama kırpıntılarında.

Kapsam dışı (bu turda koşulmuyor): işaretçisiz piksel ve glif modelleri, birleşik kayıplı glif modeli, DeepLabV3+ ve 20 epok koşuları. Bunlar tek tohumlu kalır.

## 3. Değerlendirme (s42 ile aynı aşamalar)

| Adım | Komut | Çıktı |
|---|---|---|
| `e_mr_s4344` | `run_eval_p3.py --stages val,test,robust,controls` | `phase2/experiments/results/` altındaki özet ve görüntü başı dosyalar, `tau.json` |
| `e_joint_s4344` | `run_eval_p3.py --stages val,test,robust,controls,ablation,rectfit` | aynı |
| `p_probes_unet_mrj_s4344` | `probes_p3.py --tag unet_mrj_s4344` | `phase3/results/per_image_probes_p3_unet_mrj_s4344.csv` |
| `v2_infer_mr_s4344`, `v2_eval_primary_s4344` | `v2_eval.py infer` ve `eval` (birincil 567 çift) | `phase3_v2/results/*_primary.csv` |
| sentez (elle, koşudan sonra) | `synthesize_p3.py`, `v2_report.py`, `v2_marker_split.py`, `v2_space_meta.py` | tablolar |

Karşılaştırma tohum ortalamalarıyla yapılır; görüntü başı ölçütler önce tohumlar üzerinden ortalanır, güven aralığı kaynak belge grupları üzerinden bootstrap ile hesaplanır (mevcut sentez kodu).

## 4. Öngörüler (koşmadan önce)

Referans değerler (`phase3/results/T1_dual_gt_val.csv`, `T3_probes.csv`, `phase3_v2/results/T_v2.csv`): standart kutu U-Net doğrulama kutu AUPRC 0.923 ± 0.015 (3 tohum); işaretçisiz kutu U-Net 0.916 (s42); standart piksel U-Net piksel AUPRC 0.610 ± 0.012, kutu AUPRC 0.579 ± 0.014 (3 tohum); birleşik piksel U-Net 0.622 ve 0.647 (s42).

| # | Öngörü | Yanlışlama ölçütü |
|---|---|---|
| P1 | İşaretçisiz kutu U-Net'in doğrulama kutu AUPRC'si iki yeni tohumda 0.89–0.94 aralığında kalır | Bir tohumda 0.87'nin altı |
| P2 | Kalıntının onarılması işaretçisiz kutu modeline AP kaybettirmez: üç tohum ortalamasında ΔAP ≥ −0.05 (s42: +0.019; standart model: −0.222) | Ortalama ΔAP < −0.10 |
| P3 | v2 birincil kümede işaretçisiz kutu U-Net'in kutu AUPRC'si üç tohum ortalamasında 0.62–0.78 (s42: 0.712) ve standart kutu U-Net'in 0.520'sinin üstünde kalır | Ortalama ≤ 0.57 (standart modelin bir standart sapma üstü) |
| P4 | v1'de kalıntı taşımayan tariflerde v1 → v2 AP farkı üç tohum ortalamasında sıfıra yakın kalır (s42: +0.003 [−0.06, 0.06]) | Ortalama fark < −0.08 |
| P5 | Birleşik piksel U-Net kutu ölçütünde standart piksel modelini geçer: üç tohum ortalaması ≥ 0.61 (standart 0.579) | Ortalama < 0.595 |
| P6 | Birleşik piksel U-Net piksel-değişim ölçütünde standart piksel modelinden kötü değildir: ortalama ≥ 0.605. s42'deki +0.012 fark bir standart sapma kadardır; iyileşme iddiası ancak üç tohum ortalaması 0.622'nin (standart ortalama + 1 ss) üstündeyse sürdürülür | Ortalama < 0.598 ise "kötü değil" de düşer |

Sonuç hangi yönde çıkarsa çıksın bildiri ve rapordaki ilgili cümleler üç tohumlu değerlerle yeniden yazılır; öngörü tutmazsa iddia daraltılır.

## 5. Kaynak tahmini

| Kalem | Tahmin |
|---|---|
| Eğitim | 4 koşu × yaklaşık 10,5 dk (s42 koşuları: 552–693 s) |
| Değerlendirme | işaretçisiz 2 model yaklaşık 15 dk; birleşik 2 model yaklaşık 24 dk (s42: 1363 s / 3 model, 1424 s / 2 model) |
| Sondalar | 4 model yaklaşık 50 dk (s42: 11 model 7799 s) |
| v2 | çıkarım yaklaşık 10 dk, değerlendirme (24 model, CPU) yaklaşık 15 dk |
| Toplam | yaklaşık 2,5–3 saat; disk: 4 ağırlık × 55 MB ve tahmin haritaları |

## 6. Ön-uçuş kontrolü

| Kontrol | Durum |
|---|---|
| Yapılandırma farkı | s42 komutlarıyla aynı; yalnız `--seed` |
| Koşu adları | Projenin adlandırma kuralı: `<mimari>_<referans>[_mr]_s<tohum>` |
| Yollar | `data/AIForge-Doc-v1`, `phase3/marker_repaired`, `phase2/gt/*`, `models/phase2`, `data/AIForge-Doc-v2` mevcut |
| İzleme | `phase3/logs/queue.log`, adım günlükleri ve `.done` işaretleri (deney izleyici yok) |
| Bitiş tahmini | Başlangıçtan yaklaşık 3 saat sonra (yerel saat, UTC+3) |
| Kod = gönderilmiş commit | Koşudan önce çalışma ağacı GitHub'a eşitlenir; commit kimliği `queue_notes.txt` dosyasına yazılır |
| Süreç ağacı | Kuyruk, masaüstü uygulamasının süreç ağacının dışında başlatılır (9 Kasım kesintisinin dersi) |
| Sonuçların yedeği | `_arsiv/_backup/2026-11-10_ek_tohum_oncesi/` (mevcut sonuç dosyaları; eski satırların değişmediği koşudan sonra karşılaştırılır) |

## 7. Sonuç (11 Kasım 2026, koşudan sonra yazıldı; yukarıdaki bölümlere dokunulmadı)

**Koşu.** Kuyruk 10 Kasım 21:57:21'de başladı (commit 4668b08, 21:55:51'de gönderilmişti; plan ve kuyruk betiği bu commit'in içindedir) ve 11 Kasım 00:07:00'de bitti; dokuz adımın tamamı `DONE`, `FAIL` yok (`phase3/logs/queue.log`). Süreler: eğitim 520, 502, 505, 507 s; değerlendirme 891 ve 1181 s; sondalar 2738 s; v2 çıkarım 175 s, v2 değerlendirme 760 s. Toplam 2 saat 10 dakika (tahmin 2,5–3 saat).

**Eski satırların denetimi.** Koşudan önce alınan yedekle (`_arsiv/_backup/2026-11-10_ek_tohum_oncesi/`) karşılaştırma: `summary_val.csv`, `summary_test.csv`, `robustness.csv`, `probes_boundary.csv`, `summary_v2_primary.csv`, `image_level_v2_primary.csv`, `detection_val.csv` ve `detection_test.csv` dosyalarında eski modellerin satırları değişmedi; `tau.json` dosyasına yalnız dört anahtar eklendi.

**Öngörülerin değerlendirmesi.** Kaynaklar: `phase3/results/tables_p3.md` (T1, T2, T3, T3b), `phase3/results/extra_seed_stats.md`, `phase3_v2/results/v2_tables.md`, `phase3_v2/results/T_v2_marker_split_agg.csv`.

| # | Öngörü | Sonuç | Karar |
|---|---|---|---|
| P1 | İşaretçisiz kutu U-Net doğrulama kutu AUPRC 0.89–0.94 | 0.916 (s42), 0.916 (s43), 0.905 (s44); ortalama 0.912 ± 0.006 | Tuttu |
| P2 | Onarım ΔAP ≥ −0.05 (üç tohum ortalaması) | +0.017 [+0.001, +0.033] (s42: +0.019) | Tuttu |
| P3 | v2 kutu AUPRC 0.62–0.78 ve standart modelin 0.520'sinin üstünde | 0.712, 0.718, 0.703; ortalama 0.711 ± 0.007 (standart 0.520 ± 0.018) | Tuttu |
| P4 | v1 kalıntısız tariflerde v1 → v2 AP farkı sıfıra yakın | Üç tohum ortalaması −0.008 [−0.069, +0.053]; tohum bazında +0.003, −0.026, −0.002 | Tuttu |
| P5 | Birleşik piksel U-Net kutu ölçütünde ≥ 0.61 | 0.647, 0.675, 0.682; ortalama 0.668 ± 0.019 (standart 0.579 ± 0.014); eşli görüntü başı AP +0.051 [+0.043, +0.058] | Tuttu |
| P6 | Birleşik piksel U-Net piksel-değişim ölçütünde ≥ 0.605; iyileşme iddiası için > 0.622 | 0.622, 0.648, 0.630; ortalama 0.633 ± 0.013 (standart 0.610 ± 0.012); eşli görüntü başı AP +0.011 [+0.005, +0.017] | Tuttu |

Altı öngörünün altısı da tuttu; hiçbir yanlışlama ölçütü aşılmadı.

**Öngörülerin kapsamadığı, iddiayı daraltan bulgular.**
1. Birleşik kaybın piksel-değişim ölçütündeki kazancı küçüktür ve doğrulamayla sınırlıdır: eşi bulunan 89 test görüntüsünde havuzlanmış AUPRC 0.652 ± 0.003 ve 0.654 ± 0.016, eşli görüntü başı AP farkı +0.004 [−0.008, +0.015]. Bildiri ve rapordaki "hemen her eksende iyileştirir" ifadesi "asıl olarak kutu ölçütünde iyileştirir" olarak daraltıldı.
2. Yeniden eğitimin bir bedeli vardır: işaretçisiz kutu U-Net standart modele göre görüntü başı AP'de yaklaşık 0.04 kaybeder (doğrulama −0.038 [−0.054, −0.021], test −0.040 [−0.052, −0.029]); eşi bulunan 89 test görüntüsünde piksel-değişim ölçütünde de geridedir (0.395 ± 0.027 ve 0.427 ± 0.002; eşli AP −0.053 [−0.085, −0.029]). Tek tohumda görülen 0.389 (standart 0.374) doğrulama değeri üç tohumda 0.373 ± 0.013'e indi; "işaretçisiz kutu modeli piksel-değişim ölçütünde daha iyi" izlenimi desteklenmedi (bildiride böyle bir iddia yoktu).
3. Birleşik model işaretçiye bağımlı kalır: onarım farkı −0.194 [−0.226, −0.164] (s42: −0.174).

**Plandan sapmalar.**
1. `rectfit` aşaması `probes_rectfit.csv` ve `per_image_rectfit.csv` dosyalarını yalnız yeni modellerle yeniden yazdı; eski satırlar yedekten geri birleştirildi (12 ve 5.828 satır). `probes_rectfit_agg.csv` değişmedi.
2. Kutu düzeyi tespit ölçütü (AP50, AP50-95) planda yoktu; `tables_p3.md` içindeki T2 satırları üç tohum gösterirken AP50 sütunu tek tohumda kaldığı için 11 Kasım'da kuyruğa beşinci adım olarak eklendi ve koşuldu (377 ve 707 s). Test AP50: işaretçisiz kutu U-Net 0.804 ± 0.005, birleşik piksel U-Net 0.125 ± 0.020.
3. `kod/phase3_v2/v2_marker_split.py` içindeki bootstrap üreteci döngü boyunca paylaşılıyordu; dosyaya model eklenince sonraki satırların güven aralıklarının üçüncü ondalığı oynuyordu (nokta tahminleri değişmedi). Üreteç her çağrıda yeniden tohumlanacak biçimde düzeltildi ve tohum ortalamalı, kaynak belge grupları üzerinden bootstrap'li satırlar ayrı bir dosyaya yazıldı (`T_v2_marker_split_agg.csv`).
4. Tamamlayıcı istatistikler için yeni betik: `kod/phase3/seed_extra_stats.py` → `phase3/results/extra_seed_stats.{json,md}` (tohum bazında AUPRC ve eşli görüntü başı AP farkları).

**Güncellenen belgeler.** Bildiri (özet, giriş, yöntem, Tablo I–III ve VI, §V-B, §V-C, §V-D, §V-F, §V-G, §V-H, tartışma), dönem içi raporu (Tablo 16, 17, 27–31, 34, 35, Bölüm 4.3–4.6, sınırlamalar, Ek A), README, `reports/Hakem_Hazirligi_Faz3.md`, `reports/PHASE3_Sonuclar_ve_Kapanis.md` (§7), sunum (slayt 9, 11 ve 14 ile notları), `kod/app/meta/phase3_tables.md`.

**Tek tohumlu kalanlar.** İşaretçisiz piksel-değişim ve glif modelleri, birleşik kayıplı glif modeli, DeepLabV3+, 20 epok koşuları ve YOLO ablasyonları.

**Düzeltme notu (11 Kasım 2026).** Bölüm 2'deki "2.762 eğitim görüntüsü" eğitim bölmesinin boyutudur; dışlanan 14 çiftten sonra eğitimde 2.748 görüntü kullanılır (`phase2/experiments/results/train_*.json`, `n_train`). Koşular ve sonuçlar bundan etkilenmez.

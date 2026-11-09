# Faz 3 — YOLO tespit formülasyonu vs U-Net segmentasyon (AIForge-Doc v1)

Tarih: 9 Kasım 2026 · Donanım: RTX 5080 (16 GB), Windows (multiprocessing yok → `workers=0`) · torch 2.11+cu128 · Ultralytics YOLO11.
Lab Rat `experiment-design` + `launch` metodolojisiyle hazırlandı. Kod: `phase3_yolo/` (bu klasör). Faz 2 altyapısı (`p2data`, `evaluate_p2.py`, GT varyantları, τ*, bozulmalar, kontroller) **değiştirilmeden** yeniden kullanılır.

## 0. Neden YOLO anlamlı, neden riskli

**Lehine.** Veri setindeki resmî GT bir **dikdörtgen bbox** (`bbox_xyxy`); her sahte görüntüde tam 1 bölge. Bu, piksel segmentasyonundan çok **tek-sınıf nesne tespiti** problemidir. Faz 2 H2 bulgusu da bunu söylüyor: bbox-GT ile eğitilen U-Net'in tahminleri zaten dikdörtgenimsi (rectangularity medyan 0.87) — yani U-Net, kutu öğrenmeye zorlanıyor. Kutuyu doğrudan öğrenen bir model aynı GT altında eşit/daha iyi olup **çok daha ucuz** çıkarım verebilir (kayan pencere + piksel decoder yerine tek geçiş). Raporda "formülasyon ablasyonu" olarak güçlü bir hikâye.

**Aleyhine / riskler.**
1. **Küçük hedef rejimi.** bbox yüksekliği medyan **32 px** (Q1 21, min 6), görüntüler medyan 750×1296 (max 6016×7016). `imgsz=640`'ta medyan hedef ~16 px → YOLO'nun stride-8 başlığının altında. Bu yüzden **natif çözünürlükte 1024 kare (tile) ile eğitim ve dilimli çıkarım** varsayılan yol; "tam görüntü 1280'e küçült" ise kontrol kolu.
2. **Piksel-GT altında yapısal dezavantaj.** Kutu çıktısı piksel-değişim GT'sinde (`pixel_eps8_morph`) H1 nedeniyle düşük AUPRC verecek — bu bir hata değil, **öngörülen ve raporlanacak** sonuç (H5b).
3. **Yeşil işaretçi kısayolu.** Gemini örneklerinin %83.6'sında 3 px yeşil çerçeve kalıntısı var; kutu dedektörü bu çerçeveyi öğrenmeye U-Net'ten daha yatkın olabilir. Mevcut `probe_green_marker.py` / işaretçi-onarımlı kontrol YOLO'ya da uygulanacak (H5d).
4. **Augmentasyon.** Mosaic/scale yeniden-örnekleme ve dikiş artefaktı üretir; adli sinyali bozar. Varsayılan profil `conserv` (mosaic=0, scale=0); Ultralytics varsayılanı ablasyon kolu.
5. **Lisans.** Ultralytics AGPL-3.0 — akademik kullanım/yayın serbest; kodun açık paylaşımı zaten planda. Raporda belirtilmeli.
6. **Kurulum.** `pip install ultralytics` PyPI'dan **CPU-only torch** çekip cu128 kurulumunu ezebilir → `--no-deps` ile kurulur (`run_phase3.ps1 -Stage Setup`).

## 1. Araştırma sorusu ve hipotezler

Bağımlı değişkenler: bbox-GT AUPRC / F1@τ* / IoU (Faz 2 ile aynı), **tespit AP@0.5 ve AP@0.5:0.95** (iki aile için aynı tanım, `detection_metrics.py`), çıkarım ms/görüntü, eğitim dk, peak GiB.
Bağımsız değişkenler: formülasyon (YOLO vs U-Net), giriş rejimi (tile1024 vs full1280), aug profili, authentic negatif, model boyutu (n/s/m), seed.
Baseline: **`unet_bbox_s42/43/44`** (Faz 2 B; bbox-GT test AUPRC 0.950 ± 0.011, F1@τ* 0.846, IoU 0.789; eğitim ~10.7 dk; çıkarım 512/448 kayan pencere). Yeniden koşulmaz — aynı val/test id'leri, aynı GT, aynı τ* prosedürü ile doğrudan karşılaştırılır.

| # | Hipotez | Beklenti (koşmadan önce) | Karar ölçütü |
|---|---|---|---|
| H5a | Kutu dedektörü bbox-GT altında U-Net bbox-modeline eşit veya üstündür | AP@0.5 ≥ 0.90; bbox-GT F1@τ* U-Net'in ±0.03 bandında; **çıkarım ≥3× hızlı** | 3 seed ort.; eşleştirilmiş bootstrap (görüntü) %95 GA |
| H5b | Piksel-GT altında kutu dedektörü U-Net piksel-modelinden belirgin düşüktür (H1'in teyidi) | pixel AUPRC ≈ 0.35–0.45 (U-Net bbox-modeli 0.427 gibi) vs piksel-modeli 0.654 | aynı |
| H5c | Küçük hedeflerde (<1 %) tile1024 ≫ full1280 | <1 % diliminde AP@0.5 farkı ≥ 0.10 | seed 42, alan dilimi kırılımı |
| H5d | Held-out generator (Gemini-only → Ideogram) açığı YOLO'da U-Net'ten **büyük** değildir | Ideogram AP düşüşü ≤ U-Net bbox-modelinin 0.51'lik açığı | tek seed, Faz 2 C ile aynı protokol |
| H5e | `conserv` aug `default`'tan daha iyi/eşit (yeniden-örnekleme izi bozar) | conserv ≥ default | seed 42 |

## 2. Deney matrisi (tek değişken kuralı)

Sabitler (tüm run'lar): split'ler ve dışlananlar Faz 2 ile aynı (train 2748 / val 487 / test 812), tek sınıf, AdamW lr0 0.01 cosine, AMP, 40 epoch patience 10, batch 8 (m: 4), `workers=0`, `deterministic=True`, val seçimi YOLO'nun val-tile mAP50'si ile (U-Net'te val-crop AUPRC'nin karşılığı), nihai metrikler **tam görüntüde** Faz 2 değerlendiricisiyle.

| Run ID | Faktör | Değer | Diğerleri | Beklenen |
|---|---|---|---|---|
| A · `…-s42-dryrun` | duman | 2 epoch, %15 veri | tiles/11s/1024/conserv | pipeline uçtan uca çalışır; epoch süresi ölçülür |
| B1–B3 · `aiforge-tiles-yolo11s-img1024-conserv-s{42,43,44}` | seed | 42/43/44 | tiles/11s/1024/conserv | **ana sonuç**, U-Net bbox s42–44 ile eşleşir |
| C1 · `aiforge-full-yolo11s-img1280-conserv-s42` | giriş rejimi | full-resize 1280 | 11s/conserv/s42 | <1 % diliminde belirgin düşüş (H5c) |
| C2 · `aiforge-tiles-yolo11s-img1024-default-s42` | aug | Ultralytics varsayılanı | tiles/11s/1024/s42 | ≤ conserv (H5e) |
| C3 · `aiforge-tiles_authneg-yolo11s-img1024-conserv-s42` | authentic negatif | +1 boş-etiketli authentic kare/görüntü | tiles/11s/1024/conserv/s42 | authentic FP ↓, recall ≈ |
| D1 · `aiforge-tiles-yolo11n-…-s42` / D2 · `…yolo11m…-s42` | model boyutu | n / m | tiles/1024/conserv/s42 | n: AP −0.02…−0.05, 2× hızlı; m: +0.01…+0.02, 2× yavaş |
| E1 · `aiforge-tiles_geminionly-yolo11s-img1024-conserv-s42` | eğitim üreticisi | yalnız Gemini (2454) | tiles/11s/1024/conserv/s42 | Ideogram AP düşüşü (H5d) |

Toplam eğitim run'ı: **9** (1 duman + 8). Full-factorial değil; **ardışık eleme**: B ana sonucu verir, C/D/E tek faktör sapmalarıdır. Bir ablasyon kazanırsa (ör. C3), 2 seed daha eklenir (bütçe izin veriyor).

## 3. Kaynak tahmini (duman testinden sonra güncellenecek)

- Veri: tiles ≈ 2748 × (1–3 pozitif + 1 negatif) ≈ **6–9 bin PNG kare** ≈ 8–12 GB; `_authneg` + `_geminionly` kopyaları ile ≈ **25–30 GB disk**. Alternatif: `_geminionly` yerine ana tiles'tan id filtresiyle liste dosyası (isteğe bağlı iyileştirme).
- Eğitim: YOLO11s @1024, batch 8, ~7 bin kare/epoch → tahmin **1.5–2.5 dk/epoch** → 40 epoch ≈ **1–1.7 saat/run**. 8 run ≈ **9–13 GPU-saat** (11m için ×2, 11n için ×0.5). Faz 2'nin tamamı 124 dk idi; bu faz onun ~5 katı → gecelik koşu planı gerekir.
- Çıkarım: tile1024/stride768, val 487 + test 812 görüntü × ortalama ~3 kare → dakikalar mertebesi; probelar (3 bozulma + authentic) × 3 seed ≈ 30–45 dk.
- Depolama: `best.pt` 11s ≈ 19 MB, 11m ≈ 40 MB; olasılık haritaları Faz 2 ile aynı boyutta (preds/<run>/…).
- Bütçe aşımı tetiği: duman testinde epoch > 4 dk ise → epochs 25, ya da `batch 16` + `imgsz 896` dene (kare 1024 kalır, model girişi küçülür — bunu da logla).

## 4. Yürütme sırası (`run_phase3.ps1 -Stage …`)

`Setup → Data → A (duman, zorunlu kapı) → B → Eval (B için τ* + val/test) → C → D → E → Probes → Metrics`.
A geçmeden B başlamaz. B'nin 3 seed'i ardışık (tek GPU). C/D/E birbirinden bağımsızdır; gecelik tek script. Her run `results/checkpoint_summary_yolo.csv`'ye satır ekler (Faz 2 `checkpoint_summary.csv` ile birleştirilebilir sütunlar).

## 5. Analiz planı (koşmadan önce sabitlendi)

1. **Ana tablo (T-Y1):** val ve test, `train_gt=bbox` satırlarına YOLO B ort±std eklenir: AUPRC, AUROC, F1@τ*, IoU@τ*, OracleF1 × 3 eval GT. Kaynak: `evaluate_p2.py` çıktıları (aynı dosya formatı).
2. **Tespit tablosu (T-Y2):** AP@0.5, AP@0.5:0.95, Recall@0.5 — YOLO B vs U-Net bbox (τ*+CC) vs U-Net piksel; kırılım: alan dilimi (<1 %, 1–5 %, 5–20 %), üretici (gemini/ideogram), kaynak veri seti.
3. **Maliyet tablosu (T-Y3):** eğitim dk, peak GiB, parametre, ckpt MB, **ms/görüntü** (tam görüntü, aynı makine) — `resource_usage.csv` ile aynı sütunlar.
4. **Dayanıklılık (T-Y4):** JPEG Q75 / 0.75× / blur σ1 relΔAUPRC, Faz 2 D4 ile yan yana.
5. **Benign kontrol:** authentic görüntülerde kutu sayısı ve top-conf dağılımı → image-level AUROC (top-1 conf), Faz 2 E3 ile karşılaştırma.
6. **İstatistik:** görüntü-düzeyi eşleştirilmiş bootstrap (1000 örneklem, `group` sütunuyla küme bootstrap) %95 GA; seed std ayrıca. Çöken run → aynı seed ile 1 kez yeniden; ikinci çöküş → tablodan düşülür ve not edilir.
7. **Görseller:** (i) AP@0.5 vs ms/görüntü Pareto (tüm run'lar), (ii) alan dilimi başına AP çubuk (tile vs full), (iii) örnek galeri: YOLO kutusu + U-Net haritası + 3 GT aynı görüntüde, (iv) eğitim eğrileri (val mAP50/epoch).
8. **Rapora cümle taslağı (koşmadan):** "bbox-GT altında kutu dedektörü U-Net ile eşdeğer lokalizasyon sağlarken çıkarım maliyetini ×k düşürmüştür; piksel-GT altında ise beklendiği gibi geride kalmıştır — bu, H1'in model-bağımsız bir teyididir."

## 6. Launch ön-uçuş kontrolü (Lab Rat `launch`)

| Kontrol | Durum | Not |
|---|---|---|
| 1 Config diff | ⚠ referans Faz 2 `train_unet.py` sabitleri; YOLO'da **lr0 0.01** (U-Net 1e-4 — framework farkı, bilinçli), batch 8 (U-Net 16; 1024 kare bellek), epoch 40 (U-Net 10; U-Net eğrileri 10'da doymamıştı), seed sabit, `deterministic=True`, `workers=0` | Her sapma yukarıda gerekçeli |
| 2 Run adı | ✓ `aiforge-<data>-<model>-img<imgsz>-<aug>-s<seed>` — koddan otomatik | İç kod yok, tek cümlede okunur |
| 3 Yollar | ✓ `data/AIForge-Doc-v1/{TrainingSet,TestingSet}/{images,masks,authentic}`, `dataset_audit/{splits.json,samples_forged.csv}`, `phase2/experiments/excluded_ids.json`, `models/phase2/unet_bbox_s4{2,3,4}.pt` bu oturumda doğrulandı | `yolo_data/` Data aşamasında oluşur; disk ≥ 30 GB boş olmalı |
| 4 İzleme | ⚠ wandb/neptune/mlflow **yok**; Ultralytics `results.csv` + `runs/<run>/` grafikler + `checkpoint_summary_yolo.csv` | Faz 2 ile tutarlı (CSV log). İsteğe bağlı: `yolo settings tensorboard=True` |
| 5 ETA | ⏳ duman testinden sonra: `time_min` × 20 → B ≈ 3–5 saat, tüm faz ≈ 10–14 saat (gece) | Europe/Istanbul; gece başlat, sabah `Eval` |
| 6 Kod = push edilmiş commit | ✗ **`Vize Proje` bir git deposu değil** | Öneri: `git init` + ilk commit (veri/`models/`/`yolo_data/` .gitignore'da), run'lara commit hash yaz. Waive edilebilir ama rapor tekrar-üretilebilirliği için önerilmez |

Temizlik: yeniden başlatmada `runs/<run>/` silinmez (`exist_ok=True` üzerine yazar) → eski `results.csv` karışmasın diye **önce klasörü sil**, sonra `preds/<run>/` ve `preds_boxes/<run>/`.

Temiz lansman komutu (A geçtikten sonra):
```powershell
conda activate docforensics
cd "C:\Users\PARS\Desktop\PROJELER\AI 515\Vize Proje\phase3_yolo"
.\run_phase3.ps1 -Stage B
```

## 7. Uygulama günlüğü — plandan sapmalar (9 Kasım 2026)

| Konu | Plan | Uygulanan | Gerekçe / kanıt |
|---|---|---|---|
| Öğrenme oranı | lr0 0.01 | **lr0 0.001** (AdamW) | İlk duman testinde (2 epoch, %15 veri) val kaybı `nan`, mAP50 ≈ 0. 0.01 SGD varsayılanıdır; AdamW için 10× fazla. 0.001 ile ikinci duman (4 epoch, %30 veri): val mAP50 0.43 → 0.49 → 0.55 → 0.65 (monoton). |
| Epoch | 40, patience 10 | **30**, patience 10 | Ölçülen epoch ≈ 100 s (U-Net kuyruğuyla paralel) → 8 run × 40 epoch ≈ 9+ saat; 30 epoch ile ≈ 7 saat. Ana run'da val-kare mAP50 7. epoch'ta 0.895. |
| DataLoader | workers 0 | **workers 6** | Gerçek Windows makinede çok-süreçli yükleme çalışıyor (Faz 2 kısıtı sandbox'a özeldi). |
| Ortam | — | torch 2.11.0+cu128, smp 0.6.0, ultralytics 8.4.208 | `docforensics` ortamı yeniden kuruldu; Faz 2 checkpoint'leri val'de birebir aynı haritayı üretiyor (`phase3/results/repro_check_env.json`, |Δ| = 0). |
| Gemini-only val | tüm val | **yalnız Gemini val (446)** | Faz 2 C ile aynı: Ideogram val'de de held-out. |
| YOLO sondaları/dayanıklılık | 3 seed | **s42** | Süre bütçesi (kareli çıkarım ≈ 1 s/görüntü); ana val/test metrikleri 3 seed. |

Kuyruklar: `phase3/q0_dataprep.ps1` → `q2_yolo_train.ps1 -Epochs 30` → `q3b_eval_yolo.ps1` (q3a'dan sonra). Günlük: `phase3/logs/queue.log`.

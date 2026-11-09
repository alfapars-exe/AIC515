# XAI (SHAP) test raporu

Üretildi: 2026-11-09 16:08 · `scripts/xai_benchmark.py` · cihaz **cpu** (ZeroGPU=False) · toplam 43.3 dk
Ayarlar: Partition, referans = authentic eş (varsa) yoksa blur(32,32), max_evals = 200 (durum taraması 100), 8 girdi, grup modeli `PHASE2/unet_bbox_s42` (eşik 0.167).

> **Not:** Bu çalıştırma CPU'da ve düşük bütçeyle yapıldı; süreler GPU'yu temsil etmez ve Partition haritaları daha kaba bölümlemelidir. GPU'da `python scripts/xai_benchmark.py --n 24` ile yeniden üretin.

## A) Model durumu

| Model | Ad | Aile | Kanal | Durum | Toplamsallık hatası | Değerlendirme | Süre s | Not |
|---|---|---|---|---|---|---|---|---|
| ResNet-18 · tam ince ayar | `PHASE1/resnet18_ft_best` | resnet | 3 | **OK** | 0.000000 | 100 | 2.5 |  |
| ResNet-18 · linear probe | `PHASE1/resnet18_probe_best` | resnet | 3 | **OK** | 0.000000 | 100 | 2.5 |  |
| ResNet-18 · sıfırdan | `PHASE1/resnet18_scratch_best` | resnet | 3 | **OK** | 0.000000 | 100 | 2.6 |  |
| ResNet-18 · H4 global-iz sondası | `PHASE2/h4_resnet18` | resnet | 3 | **OK** | 0.000000 | 100 | 2.4 |  |
| ResNet-18 · H4 global-iz sondası (poz. kontrol) | `PHASE2/h4_resnet18_poscontrol` | resnet | 3 | **OK** | 0.000000 | 100 | 2.6 |  |
| U-Net · bbox-GT · yalnız Gemini (s42) | `PHASE2/unet_bbox_geminionly_s42` | unet | 3 | **OK** | 0.000000 | 100 | 29.8 |  |
| U-Net · bbox-GT (s42) | `PHASE2/unet_bbox_s42` | unet | 3 | **OK** | 0.000000 | 100 | 28.1 |  |
| U-Net · bbox-GT (s43) | `PHASE2/unet_bbox_s43` | unet | 3 | **OK** | 0.000000 | 100 | 28.8 |  |
| U-Net · bbox-GT (s44) | `PHASE2/unet_bbox_s44` | unet | 3 | **OK** | 0.000000 | 100 | 28.9 |  |
| U-Net · glif-GT · yalnız Gemini (s42) | `PHASE2/unet_glyph_v2_geminionly_s42` | unet | 3 | **OK** | 0.000000 | 100 | 28.5 |  |
| U-Net · glif-GT (s42) | `PHASE2/unet_glyph_v2_s42` | unet | 3 | **OK** | 0.000000 | 100 | 27.9 |  |
| U-Net · glif-GT (s43) | `PHASE2/unet_glyph_v2_s43` | unet | 3 | **OK** | 0.000000 | 100 | 28.3 |  |
| U-Net · glif-GT (s44) | `PHASE2/unet_glyph_v2_s44` | unet | 3 | **OK** | 0.000000 | 100 | 27.8 |  |
| U-Net · piksel-GT · yalnız Gemini (s42) | `PHASE2/unet_pixel_eps8_morph_geminionly_s42` | unet | 3 | **OK** | 0.000000 | 100 | 28.1 |  |
| U-Net · piksel-GT (s42) | `PHASE2/unet_pixel_eps8_morph_s42` | unet | 3 | **OK** | 0.000000 | 100 | 27.7 |  |
| U-Net · RGB (PHASE1, 400 örnek) | `PHASE1/unet_rgb_best` | unet | 3 | **OK** | 0.000000 | 100 | 27.6 |  |
| U-Net · RGB+FFT+Laplacian (PHASE1) | `PHASE1/unet_rgbfreq_best` | unet | 5 | **OK** | 0.000000 | 100 | 46.4 |  |
| Klasik · SVM-RBF (56 öznitelik) | `CLASSICAL/svm_rbf` | classical | — | **UNSUPPORTED** | — | — | — | Model dosyası yok (yalnız results_tables.md'de sonuçları var). Dosya eklenirse KernelExplainer (56 öznitelik). |
| Klasik · RandomForest (56 öznitelik) | `CLASSICAL/random_forest` | classical | — | **UNSUPPORTED** | — | — | — | Model dosyası yok (bucket'ta .pkl/.joblib yok). Dosya eklenirse TreeExplainer (56 öznitelik). |
| YOLO (nesne tespiti) | `N/A/yolo` | none | — | **N/A** | — | — | — | Bu uygulamada YOLO modeli yok. Tespit gereksinimi U-Net bağlı bileşenleri ve ResNet top-k patch'leriyle karşılanır. |
| ViT / EfficientNet | `N/A/vit` | none | — | **N/A** | — | — | — | Projede bu mimariler yok. |

Özet: OK 17 · UNSUPPORTED 2 · N/A 2

## B) Çekirdek modeller — performans ve doğruluk

Girdiler: `testing:000000030` (TP, Gemini, alan <1%), `testing:000000038` (FN, Gemini, alan <1%), `testing:000000088:auth` (TN, Gemini, alan —), `testing:000000260` (TP, Gemini, alan 1-5%), `testing:000000086` (FN, Gemini, alan <1%), `testing:000000411:auth` (TN, Gemini, alan —), `testing:000000363` (TP, Ideogram, alan 1-5%), `testing:000000088` (FN, Gemini, alan <1%)

| Model | n | Süre s (ort / medyan) | Değerl. | Tepe MB | Toplamsallık hatası (maks) | AOPC SHAP | AOPC rastgele | SHAP > rastgele | Pointing game | İşaretçi kütlesi (lift) | Kenar halkası kütlesi (lift) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| U-Net · bbox-GT (s42) | 8 | 52.1 / 52.0 | 200 | — | 0.000000 | 0.445 | 0.191 | %100 | 0.50 (n=6) | 0.012 (4.3×) | 0.132 (5.6×) |
| U-Net · piksel-GT (s42) | 8 | 52.8 / 52.7 | 200 | — | 0.000000 | 0.321 | 0.101 | %88 | 0.50 (n=6) | 0.025 (8.6×) | 0.164 (6.2×) |
| U-Net · glif-GT (s42) | 8 | 52.4 / 51.9 | 200 | — | 0.000000 | 0.349 | 0.167 | %100 | 0.50 (n=6) | 0.022 (7.5×) | 0.131 (5.4×) |
| ResNet-18 · tam ince ayar | 8 | 4.1 / 4.1 | 200 | — | 0.000000 | 0.623 | 0.225 | %100 | 0.50 (n=4) | 0.056 (15.1×) | 0.339 (4.1×) |

### Gruplara göre (TP / FN / TN / FP)

| Model | Grup | n | f(x) ort. | AOPC SHAP | AOPC rastgele | Pointing |
|---|---|---|---|---|---|---|
| U-Net · bbox-GT (s42) | TP | 3 | 1.000 | 0.894 | 0.264 | 1.00 |
| U-Net · piksel-GT (s42) | TP | 3 | 0.868 | 0.759 | 0.236 | 0.67 |
| U-Net · glif-GT (s42) | TP | 3 | 0.861 | 0.825 | 0.377 | 0.67 |
| ResNet-18 · tam ince ayar | TP | 3 | 0.922 | 0.817 | 0.185 | 0.33 |
| U-Net · bbox-GT (s42) | FN | 3 | 0.306 | 0.280 | 0.243 | 0.00 |
| U-Net · piksel-GT (s42) | FN | 3 | 0.096 | 0.094 | 0.051 | 0.33 |
| U-Net · glif-GT (s42) | FN | 3 | 0.099 | 0.078 | 0.062 | 0.33 |
| ResNet-18 · tam ince ayar | FN | 3 | 0.682 | 0.592 | 0.421 | 1.00 |
| U-Net · bbox-GT (s42) | TN | 2 | 0.021 | 0.018 | 0.005 | — |
| U-Net · piksel-GT (s42) | TN | 2 | 0.007 | 0.005 | -0.025 | — |
| U-Net · glif-GT (s42) | TN | 2 | 0.057 | 0.044 | 0.012 | — |
| ResNet-18 · tam ince ayar | TN | 2 | 0.513 | 0.379 | -0.007 | — |

## C) Tohum kararlılığı (aynı pencere, top-%5 pozitif bölge)

| Aile | n görüntü | IoU ort. | Spearman ort. |
|---|---|---|---|
| U-Net bbox s42/43/44 | 2 | 0.610 | 0.344 |
| U-Net glif s42/43/44 | 2 | 0.867 | 0.300 |

## Tanımlar

- **Toplamsallık hatası** = |E[f] + ΣSHAP − f(x)| (Partition'da kesin olmalı; ≈ 0).
- **AOPC** (silme): pencere 16×16 bloğa bölünür; bloklar SHAP toplamına göre azalan sırada %10'luk adımlarla referansla değiştirilir; f(x) − f(adım) ortalaması. Rastgele sıra (2 permütasyon) ile karşılaştırılır; SHAP > rastgele = sadakat.
- **Pointing game**: en yüksek SHAP bloğu GT kutusuna değiyorsa isabet (yalnız GT'li sahte girdiler).
- **İşaretçi kütlesi**: pozitif SHAP kütlesinin yeşil işaretçi piksellerindeki (±1 px) payı; *lift* = bu pay / alan payı. **Kenar halkası**: GT kutusu kenarı ±6 px. Lift ≫ 1 → modelin kararı işaretçi kısayoluna dayanıyor (PHASE 2 Ek-C).
- Gruplar `unet_bbox_s42` görüntü-düzeyi kararına göre: TP/FN sahte girdiler, TN/FP authentic eşler.
- Ham sonuçlar: `docs/xai_benchmark.csv`, `docs/xai_status.csv`, `docs/xai_seed_stability.csv`.

# AIForge-Doc v1 — Küçük Veriyle Baseline Deneyleri (smoke_train → smoke_val)

**Amaç.** Vize raporu için (a) kodların uçtan uca çalıştığını göstermek, (b) Faz 1 hipotezleri için ilk ölçümleri almak.
Tüm eğitimler `splits.json` içindeki **smoke_train (400 forged görüntü)** üzerinde; tüm değerlendirmeler **smoke_val (100 görüntü)** üzerinde.
**TestingSet/ hiçbir adımda okunmadı.** Donanım: RTX 5080 (16 GB), Windows, `docforensics` ortamı (torch 2.11.0+cu128, torchvision 0.26, smp 0.5, scikit-learn 1.9). Seed = 42. Toplam GPU eğitim süresi: **23.5 dk** (bütçe 60–90 dk'nın altında; epoch sayıları planlandığı gibi).

## 1. Ne çalıştırıldı

| Adım | Betik | Özet |
|---|---|---|
| A | `common/data.py` | Patch/crop veri katmanı. Maske GT = `masks/<id>.png` (metadata bbox değil). Görüntüler RAM'de önbelleklenir (num_workers=0; sandbox'ta multiprocessing engelli). |
| B | `features_classical.py`, `train_classical.py` | 56 el-yapımı öznitelik (8×8 blok DCT, LBP, GLCM, Laplacian artığı, FFT radyal) → SVM-RBF (3-kat CV ızgara) ve RandomForest(500). |
| C | `train_resnet.py` | ResNet18 ×3: probe (dondurulmuş backbone + yeni FC), full fine-tune, scratch. 12 epoch, batch 32, AdamW + cosine, bf16 AMP, early-stop (val AUROC, patience 5). |
| D | `train_unet.py` | smp.Unet(resnet18, ImageNet) D1 = RGB, D2 = RGB + FFT Gaussian HP (D0=20) + 3×3 Laplacian (5 kanal). 512×512 natif crop, BCE+Dice, AdamW 1e-4, 15 epoch, batch 8, bf16. |
| E | `evaluate.py` | D1/D2'yi val crop'larının JPEG Q75 ve 0.75× küçült-büyüt sürümlerinde yeniden değerlendirir (eğitim yok). |
| F | `evaluate.py` | Kaynak kullanımı tablosu + figürler + `resnet_results.*`, `unet_results.*` birleştirme. |

Komutlar (sırayla; `run_all.ps1` aynısını yapar):
```powershell
python train_classical.py --seed 42
python train_resnet.py --setting probe   --epochs 12
python train_resnet.py --setting ft      --epochs 12
python train_resnet.py --setting scratch --epochs 12
python train_unet.py   --variant rgb     --epochs 15
python train_unet.py   --variant rgbfreq --epochs 15
python evaluate.py
```

### Patch veri kümesi (A)
* **tampered**: maske bbox merkezli 256×256 (±32 px jitter; bbox 256'dan genişse bbox içinde rastgele pencere). Eğitimde görüntü başına 2, val'de 1.
* **clean_same**: aynı forged görüntüde bbox ile kesişmeyen rastgele 256×256 (küçük görüntülerde bulunamazsa atlanır).
* **clean_paired**: eşleşen authentic görüntüde ilk tampered patch ile **aynı koordinat** (pair_accepted & aynı boyut) → içerik-kontrollü negatif.
* Eğitim: 1587 patch (tampered 800 / clean_paired 398 / clean_same 389); val: 296 patch (100 / 100 / 96). Index: `results/patch_index_train.csv`, `results/patch_index_val.csv`.
* Val tampered patch'lerinde maske piksel oranı medyan ≈ %11 (min %0.4) — patch'in büyük kısmı temiz içerik.
* ResNet girişi: 256 patch → 224 random-crop (eğitim) / center-crop (val); **yeniden örnekleme yok**. Augment: yalnızca ±%8 parlaklık/kontrast (flip yok — metin).

## 2. Sonuçlar (yalnızca ölçülen değerler; smoke_val)

### B) Klasik öznitelikler + ML (patch sınıflandırma, n_val = 296)
| Model | Acc | Bal.Acc | AUROC | AUPRC | F1@0.5 | AUROC tamp-vs-paired | AUROC tamp-vs-same | AUROC gemini | AUROC ideogram | AUROC <1% | AUROC 1-5% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SVM-RBF (C=0.5, γ=0.01) | 0.534 | 0.606 | 0.646 | 0.443 | 0.546 | 0.552 | 0.744 | 0.648 | 0.641 | 0.643 | 0.648 |
| RandomForest (500 ağaç) | 0.622 | 0.653 | 0.706 | 0.536 | 0.573 | 0.610 | 0.807 | 0.696 | 0.738 | 0.732 | 0.685 |

"tamp-vs-paired": yalnızca tampered ve clean_paired patch'ler (içerik aynı → yalnızca manipülasyon izi ayırt edici). "tamp-vs-same": tampered vs clean_same. Tool/alan AUROC'ları: ilgili alt kümenin tampered patch'leri vs tüm clean patch'ler.

RF öznitelik önemleri (ilk 10; tam liste `results/rf_feature_importance.csv`, figür `figures/rf_feature_importance.png`):
| Sıra | Öznitelik | RF Gini önemi |
|---|---|---|
| 1 | lbp_u4 | 0.0320 |
| 2 | lap_mean_abs | 0.0295 |
| 3 | lap_kurtosis | 0.0266 |
| 4 | glcm_contrast | 0.0259 |
| 5 | lbp_u0 | 0.0247 |
| 6 | lbp_u5 | 0.0237 |
| 7 | lbp_u6 | 0.0235 |
| 8 | lap_std | 0.0232 |
| 9 | fft_slope | 0.0218 |
| 10 | dct_E_low | 0.0216 |

Önemler düz dağılmış (0.020–0.032): tek bir baskın el-yapımı iz yok; LBP uniform-4 (kenar deseni), Laplacian artığı istatistikleri ve GLCM kontrastı önde.

### C) ResNet18 patch sınıflandırma (n_val = 296; eşik 0.5)
| Ayar | best ep / çalışan | Acc | Bal.Acc | AUROC | AUPRC | F1@0.5 | AUROC tamp-vs-paired | AUROC tamp-vs-same | AUROC gemini | AUROC ideogram | AUROC <1% | AUROC 1-5% | TN/FP/FN/TP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ResNet18 probe (pretrained, dondurulmuş + yeni FC) | 10/12 | 0.639 | 0.693 | 0.726 | 0.554 | 0.616 | 0.635 | 0.820 | 0.735 | 0.698 | 0.680 | 0.767 | 103/93/14/86 |
| ResNet18 full fine-tune (pretrained) | 6/11 | 0.885 | 0.845 | 0.899 | 0.884 | 0.809 | 0.878 | 0.922 | 0.910 | 0.867 | 0.883 | 0.913 | 190/6/28/72 |
| ResNet18 scratch (rastgele init) | 4/9 | 0.503 | 0.610 | 0.679 | 0.497 | 0.561 | 0.573 | 0.790 | 0.670 | 0.708 | 0.656 | 0.707 | 55/141/6/94 |

Eğitim eğrileri: `figures/training_curves_resnet.png`; karışıklık matrisleri: `figures/confusion_resnet.png`; epoch log'ları: `logs/resnet_*.csv`.

### D) U-Net piksel lokalizasyonu (512×512 natif crop; val = 100 bbox-içeren + 100 rastgele crop)
GT içeren crop sayısı 141 (100 bbox + 41 rastgele crop tesadüfen maske içeriyor), tamamen temiz crop sayısı 59. Crop-başına metrikler GT içeren crop'lar üzerinde ortalanır; AUPRC/AUROC bu crop'ların piksellerinin birleştirilmiş alt örneği (tüm pozitif pikseller + crop başına ≤20k negatif) üzerinde. GT içeren crop'larda pozitif piksel oranı ortalama %4.7 (şans AUPRC ≈ 0.05).
| Varyant | best ep / çalışan | Pixel-F1@0.5 | IoU@0.5 | Oracle-F1 | Global-eşik F1 (eşik) | Pixel-AUPRC | Pixel-AUROC | F1@0.5 (yalnız bbox crop) | IoU (bbox crop) | Temiz crop FP piksel oranı | Temiz crop'ların >%1 FP'li payı |
|---|---|---|---|---|---|---|---|---|---|---|---|
| U-Net RGB (D1, 3 kanal) | 11/15 | 0.707 | 0.637 | 0.740 | 0.708 (0.45) | 0.947 | 0.948 | 0.724 | 0.653 | 0.26% | 8.5% |
| U-Net RGB+FFT-HP+Laplacian (D2, 5 kanal) | 12/15 | 0.674 | 0.613 | 0.711 | 0.674 (0.45) | 0.939 | 0.940 | 0.696 | 0.635 | 0.14% | 3.4% |

Kırılımlar (yalnızca bbox-içeren 100 crop; F1@0.5):
| Varyant | F1 gemini | F1 ideogram | F1 <1% | F1 1-5% | F1 5-20% | F1 cord | F1 sroie | F1 wildreceipt | F1 xfund |
|---|---|---|---|---|---|---|---|---|---|
| U-Net RGB (D1, 3 kanal) | 0.800 (n=75) | 0.497 (n=25) | 0.606 (n=47) | 0.830 (n=52) | 0.756 (n=1) | 0.478 | 0.869 | 0.765 | 0.601 |
| U-Net RGB+FFT-HP+Laplacian (D2, 5 kanal) | 0.779 (n=75) | 0.449 (n=25) | 0.589 (n=47) | 0.792 (n=52) | 0.788 (n=1) | 0.474 | 0.839 | 0.772 | 0.475 |

| Varyant | AUPRC gemini | AUPRC ideogram | AUPRC <1% | AUPRC 1-5% | Oracle-F1 gemini | Oracle-F1 ideogram |
|---|---|---|---|---|---|---|
| U-Net RGB (D1, 3 kanal) | 0.963 | 0.912 | 0.942 | 0.960 | 0.830 | 0.542 |
| U-Net RGB+FFT-HP+Laplacian (D2, 5 kanal) | 0.955 | 0.903 | 0.938 | 0.949 | 0.804 | 0.513 |

Örnek tahminler: `figures/unet_predictions.png` (görüntü / GT / D1 / D2, her tool'dan 3). Eğitim eğrileri: `figures/training_curves_unet.png`.

### E) Dayanıklılık probu (eğitim yok; aynı val crop'ları bozularak)
| Varyant | Koşul | Pixel-F1@0.5 | IoU@0.5 | Oracle-F1 | Pixel-AUPRC | Pixel-AUROC | Temiz crop FP | F1 gemini | F1 ideogram |
|---|---|---|---|---|---|---|---|---|---|
| U-Net RGB (D1, 3 kanal) | clean (PNG) | 0.708 | 0.637 | 0.740 | 0.947 | 0.948 | 0.26% | 0.800 | 0.497 |
| U-Net RGB (D1, 3 kanal) | JPEG Q75 | 0.636 | 0.565 | 0.680 | 0.929 | 0.932 | 0.18% | 0.728 | 0.444 |
| U-Net RGB (D1, 3 kanal) | rescale 0.75x→1x | 0.638 | 0.560 | 0.676 | 0.939 | 0.943 | 1.24% | 0.734 | 0.421 |
| U-Net RGB+FFT-HP+Laplacian (D2, 5 kanal) | clean (PNG) | 0.674 | 0.613 | 0.711 | 0.939 | 0.940 | 0.14% | 0.779 | 0.449 |
| U-Net RGB+FFT-HP+Laplacian (D2, 5 kanal) | JPEG Q75 | 0.604 | 0.540 | 0.640 | 0.912 | 0.917 | 0.13% | 0.715 | 0.328 |
| U-Net RGB+FFT-HP+Laplacian (D2, 5 kanal) | rescale 0.75x→1x | 0.633 | 0.566 | 0.674 | 0.924 | 0.929 | 0.42% | 0.755 | 0.353 |

### F) Kaynak kullanımı
| Model | Eğitim süresi (s) | Peak GPU bellek (GiB) | Parametre | Çıkarım (ms/örnek) | Checkpoint (MB) | Not |
|---|---|---|---|---|---|---|
| svm_rbf | 7 | 0.00 | — | 27.9 | — | CPU; çıkarım süresi = öznitelik çıkarımı/patch |
| random_forest | 6 | 0.00 | — | 27.9 | — | CPU; çıkarım süresi = öznitelik çıkarımı/patch |
| resnet18_probe | 64 | 0.33 | 11.18 M | 8.2 | 42.7 | 12/12 epoch, 224 patch, batch 1 bf16 |
| resnet18_ft | 76 | 0.59 | 11.18 M | 7.3 | 42.7 | 11/12 epoch, 224 patch, batch 1 bf16 |
| resnet18_scratch | 65 | 0.59 | 11.18 M | 7.5 | 42.7 | 9/12 epoch, 224 patch, batch 1 bf16 |
| unet_rgb | 603 | 1.70 | 14.33 M | 15.8 | 54.8 | 15/15 epoch, 512 crop, batch 1 bf16 |
| unet_rgbfreq | 601 | 1.75 | 14.33 M | 18.8 | 54.8 | 15/15 epoch, 512 crop, batch 1 bf16 |

Çıkarım süreleri batch=1, bf16, RTX 5080 (Python döngü ek yükü dahil). Checkpoint'ler `..\models\` altında (state_dict).

## 3. Kısa yorum
1. **Pretrained + fine-tune ≫ probe ≫ scratch.** Tam fine-tune AUROC 0.899, lineer probe 0.726, sıfırdan 0.679 (sıfırdan eğitim 1.6k patch ile kararsız; 4. epoch'ta tepe, sonra early-stop). ImageNet özellikleri tek başına yeterli değil (probe) ama iyi bir başlangıç noktası: manipülasyon izi için backbone'un uyarlanması gerekiyor.
2. **İçerik-kontrollü negatif (clean_paired) ayrımı daha zor.** Her modelde tamp-vs-paired AUROC, tamp-vs-same'den düşük (ör. ft 0.878 vs 0.922; RF 0.610 vs 0.807). Olası neden (ölçülmedi): clean_same patch'leri rastgele konumdan geldiği için metin yoğunluğu/düzeni tampered patch'lerden farklı olabilir ve model kısmen bu içerik ipucunu kullanıyor olabilir. Klasik özniteliklerde bu fark en belirgin (SVM paired AUROC 0.552 ≈ şans).
3. **Klasik öznitelikler zayıf ama şans üstü** (RF AUROC 0.706). PNG kaynaklı veri olduğundan JPEG/DCT izleri yok; DCT enerji oranları orta sırada. Bu, el-yapımı tek-iz yaklaşımının (DocTamper'daki JPEG-odaklı FPH gibi) bu veri için yeterli olmayacağı hipotezini destekler (kesin değil, smoke alt küme).
4. **U-Net lokalizasyonu beklenenden güçlü:** D1 Pixel-F1@0.5 0.707, AUPRC 0.947, temiz crop'larda FP piksel oranı %0.26. Tahmin görselleri modelin bbox **dikdörtgen bandını** öğrendiğini gösteriyor; veri auditinde (smoke çiftleri, n=498) bbox içindeki piksellerin medyan %94.7'sinin (eps=0) / %47.8'inin (eps=8) değiştiği ölçülmüştü; yani araçlar bölgeyi bütünüyle yeniden sentezliyor ve bu dikdörtgen iz kolay öğreniliyor. Bu bir **GT/veri-seti etkisi** olarak yorumlanmalı; gerçek karakter-düzeyi maske ile sonuç daha düşük olur (ileride ölçülecek).
5. **Tool etkisi büyük:** D1 F1 gemini 0.800 vs ideogram 0.497 (n=75 / 25); kaynak etkisi de büyük (sroie 0.869 vs cord 0.478). Küçük alan (<1%) daha zor: 0.606 vs 1–5% 0.830. Oracle-F1 − F1@0.5 farkı ≈ 0.032 (DocForge-Bench'teki 2–10× farka göre küçük; bbox GT'nin etkisi olabilir).
6. **RGB vs RGB+frekans:** D2 bu ayarda D1'den **daha iyi değil** (F1 0.674 vs 0.707; AUPRC 0.939 vs 0.947); yalnızca temiz crop FP oranı daha düşük (%0.14 vs %0.26). Tek seed, 400 görüntü ve 15 epoch ile fark istatistiksel olarak ayırt edilemez; ekstra kanalların ortalama ağırlıkla init edilmesi de kısa eğitimde dezavantaj olabilir. Hipotez (frekans kanalı yardımcı olur) bu smoke deneyinde desteklenmedi.
7. **Dayanıklılık (Gap C):** JPEG Q75 F1'i D1'de 0.708 → 0.636 (−7.1 puan), 0.75× yeniden ölçekleme 0.638 ve temiz crop FP'yi ~5× artırıyor (%0.26 → %1.24). D2 de benzer ölçüde düşüyor (frekans kanalı dayanıklılık kazandırmadı). Düşüş var ama çöküş yok: AUPRC ≥ 0.91 kalıyor — bu, modelin öğrendiği izin yalnızca yüksek-frekans gürültü deseni olmadığını (bbox bandının doku/parlaklık farkı gibi daha kaba izler de içerdiğini) düşündürür.

## 4. Sınırlamalar
* **Smoke alt küme:** 400 eğitim / 100 val görüntü, tek seed; tüm sayılar **ön/provizyonel**. Tool (ideogram n=25) ve alan (5–20% n=1) kırılımları küçük örneklem.
* **bbox-maske GT:** maskeler karakter değil alan dikdörtgeni; piksel metrikleri gerçek karakter-düzeyi lokalizasyonu ölçmüyor. Tam-set audit'inde (3.336 çift, eps=8) çiftlerin %33.5'inde bbox dışı değişim bulunmuştu (gemini-nano %35.9 / qwen-inpaint %11.9); smoke alt kümesindeki çiftlerde aynı ölçüm %29.7'dir (gemini-nano %34.7 / qwen-inpaint %20.1 — alt küme kaynak×tool dengeli örneklendiği için tam setten farklıdır). Not: bu README'de "ideogram" olarak anılan tool, metadata'daki `qwen-inpaint` etiketidir (dataset kartına göre Ideogram v2 Edit; eşleme yalnızca kart beyanına dayanır); bu pikseller GT'de "temiz" sayılır (FP gibi görünebilir — bkz. unet_predictions satır 5'teki ikinci bölge).
* **Crop/patch düzeyi değerlendirme:** tam-görüntü (sliding-window) çıkarımı yapılmadı; görüntü-düzeyi tespit metrikleri yok.
* Küçük görüntülerde crop boyutu için kenar-kopyalama (replicate) ile doldurma yapıldı (smoke_val'de W<256: 2, H<256: 1, W<512: 20, H<512: 12 görüntü).
* SVM/RF `n_jobs=1` (sandbox'ta multiprocessing engelli); zaman ölçümleri buna göre.
* Karşılaştırma tek model boyutu (ResNet18) ve kısa eğitim bütçesiyle yapıldı; D1-vs-D2 farkı için çoklu seed gerekir.

## 5. Dosyalar
* Kod: `common/data.py`, `common/metrics.py`, `features_classical.py`, `train_classical.py`, `train_resnet.py`, `train_unet.py`, `evaluate.py`, `run_all.ps1`
* results/: `classical_results.json/csv`, `resnet_results.json/csv` (+ `resnet_<ayar>_results.json`), `unet_results.json/csv` (+ `unet_<varyant>_results.json`), `robustness_results.csv/json`, `resource_usage.csv`, `patch_index_train.csv`, `patch_index_val.csv`, `seg_val_crop_index.csv`, `rf_feature_importance.csv`, `*_val_predictions.csv`
* figures/: `training_curves_resnet.png`, `training_curves_unet.png`, `unet_predictions.png`, `rf_feature_importance.png`, `confusion_resnet.png`
* logs/: `classical.log`, `resnet_{probe,ft,scratch}.log/.csv`, `unet_{rgb,rgbfreq}.log/.csv`, `evaluate.log`
* ../models/: `resnet18_{probe,ft,scratch}_best.pt` (42.7 MB), `unet_{rgb,rgbfreq}_best.pt` (54.8 MB)
* cache/: `patches_*.npz` (patch dizileri), `classical_features.npz`, `unet_*_val_probs.npz` (val olasılık haritaları)

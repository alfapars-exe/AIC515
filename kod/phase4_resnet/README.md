# Faz 4 — ResNet-18 parça sınıflandırıcısı: tam veri, işaretçi onarımı, görüntü düzeyi kalibrasyon

**Tarih:** 10 Kasım 2026 · **Donanım:** RTX 5080 16 GB · **Ortam:** docforensics (torch 2.11.0+cu128)
Lab Rat `experiment-design` + `launch` metodolojisiyle hazırlandı. **Bu bölüm (plan) koşulardan ÖNCE yazıldı**; sonuçlar `phase4_resnet/results/tables_p4.md` ve `reports/PHASE4_ResNet_Sonuclar.md` içindedir.

## 0. Neden
Uygulamadaki ResNet-18 (`models/resnet18_ft_best.pt`) Faz 1 pilotundan kalmadır: 400 görüntü / 1587 parça, tek tohum; val parça AUROC 0.899, 0.5 eşiğinde sahte parçaların %28'i kaçıyor. Görüntü kararı "en yüksek patch >= 0.5" kuralıyla verilir ve kalibre edilmemiştir (toplu test, n = 24 + 24: AUROC 0.990 ama FPR 0.375). Tek örnekte işaretçi kısayolu gözlenmiştir. U-Net'ler Faz 2'de tam veriyle yeniden eğitildi; ResNet eğitilmedi.

## 1. Araştırma sorusu ve hipotezler
Bağımlı değişkenler: val parça AUROC; **görüntü düzeyi** AUROC / TPR / FPR (val 487 sahte + 487 eşleşen authentic; test 812 sahte + 89 eşli authentic). Baseline: Faz 1 `resnet18_ft_best` (yeniden eğitilmez; aynı kümelerde yeniden değerlendirilir).

| # | Hipotez | Beklenti (koşmadan önce) | Karar ölçütü |
|---|---|---|---|
| H6a | Veri miktarı: tam eğitim bölümü parça ayrımını belirgin iyileştirir | val parça AUROC >= 0.95 (Faz 1 modeli aynı val'de ~0.85-0.90) | 3 tohum ortalaması, aynı val parçaları |
| H6b | Karar kuralı: val'de kalibre edilmiş agregasyon + eşik, sabit "max >= 0.5" kuralının yanlış alarmını düşürür | eski kural val FPR >= 0.20; kalibre kural dengeli doğruluğu artırır; görüntü AUROC >= 0.92 | val ve test; Wilson %95 GA |
| H6c | İşaretçi onarımlı eğitim işaretçi kısayolunu azaltır | onarılmış sahtede TPR düşüşü ve "authentic + yalnız işaretçi" yanlış alarmı ham modele göre belirgin küçük; ham val AUROC'ta küçük bedel | val, kalıntılı kimlikler; 3 tohum |

## 2. Deney matrisi (tek değişken kuralı)
Sabitler (Faz 1 `ft` tarifi, `kod/baselines/train_resnet.py`): ImageNet ResNet-18, fc -> 1, AdamW lr 1e-4, wd 1e-4, cosine, batch 32, 12 epoch, sabır 5 (val parça AUROC), bf16, 256 -> 224 rastgele kesit, +-%8 parlaklık/kontrast, flip yok. Parça tanımı Faz 1 ile aynı (görüntü başına 2 tampered + 1 clean_same + 1 clean_paired; val'de 1 tampered). Parça indeksi tüm koşularda sabit (train tohumu 42, val 43); tohum yalnız başlangıç FC'sini, sırayı ve augmentasyonu değiştirir.

| Koşu | Faktör | Değer | Diğerleri | Beklenen |
|---|---|---|---|---|
| R0 `resnet18_ft_best` | (referans) | 400 görüntü, s42 | Faz 1 | yeniden eğitilmez |
| A `...-dryrun` | duman | 150 görüntü, 2 epoch | ham + onarımlı | hat uçtan uca çalışır, epoch süresi ölçülür |
| R1 `resnet18_ft_full_s{42,43,44}` | veri miktarı | 2748 görüntü (dışlananlar hariç) | Faz 1 tarifi | H6a |
| R2 `resnet18_ft_full_mr_s{42,43,44}` | işaretçi onarımı | sahte görüntüler `phase3/marker_repaired` | R1 ile aynı koordinatlar | H6c |

Karar kuralı (H6b) eğitim gerektirmez: aynı patch skorları üzerinde hesaplanır. Toplam eğitim: 1 duman (2 kısa) + 6 koşu.

## 3. Kaynak tahmini
Parça çıkarımı ~5 dk (2748 + 487 görüntü, sahte + authentic + onarılmış decode). Eğitim: ~11 bin parça/epoch, tahmin 20-40 s/epoch -> koşu başına 4-8 dk, 6 koşu ~25-50 dk. Tam görüntü çıkarım: ~2850 görüntü x 7 model, tek decode ile ~10-20 dk. **Toplam ~1-1.5 GPU-saat.** Disk: parça önbelleği ~5 GB, ağırlıklar 6 x 43 MB.

## 4. Yürütme
```powershell
powershell -ExecutionPolicy Bypass -File kod\phase4_resnet\run_phase4.ps1 -Stage Smoke   # zorunlu kapı
powershell -ExecutionPolicy Bypass -File kod\phase4_resnet\run_phase4.ps1 -Stage All     # Data -> Train -> Infer -> Calib
```
Adımlar `phase4_resnet\logs\<adım>.done` ile yeniden başlatılabilir.

## 5. Analiz planı (koşmadan önce sabitlendi)
1. **Parça düzeyi (H6a):** tüm modeller aynı val parçalarında (ham ve onarımlı): AUROC, tamp-vs-paired, tamp-vs-same, 0.5'te TPR/FPR (`eval_patches_p4.py`).
2. **Görüntü düzeyi (H6b):** uygulama protokolüyle (224 pencere, adım 112) tüm patch skorları kaydedilir. Aday agregasyonlar max / top2 / top3 / top5. **Agregasyon seçimi:** R1'in 3 tohumunun val AUROC ortalaması en yüksek olan; 0.002 içinde küçük k. **Eşik:** model başına val'de Youden-J (U-Net/YOLO `image_tau.json` ile aynı kural). Test yalnız raporlanır.
3. **Eski kural** (max >= 0.5) aynı kümelerde yan yana.
4. **İşaretçi sondaları (H6c):** kalıntılı val kimliklerinde (a) onarılmış sahte, (b) authentic + yalnız işaretçi; kalibre eşikte TPR/FPR.
5. **Kırılımlar:** test TPR'si üretici, işaretçi alt kümesi (`ring_green_px > 0`), alan dilimi, temiz test.
6. **İstatistik:** tohum ortalaması +- std; test AUROC için kaynak-belge grubu bootstrap %95 GA (1000); TPR/FPR için Wilson. Test authentic n = 89 olduğundan test FPR geniş aralıklıdır. Çöken koşu aynı tohumla bir kez yeniden koşulur.
7. **Karşılaştırma:** aynı val protokolündeki U-Net / YOLO görüntü AUROC'ları (`kod/app/meta/image_tau.json`).

## 6. Dosyalar
`p4common.py` (indeks, parça çıkarımı, model, kayan pencere) · `build_patches_p4.py` · `train_resnet_p4.py` · `eval_patches_p4.py` · `infer_images_p4.py` · `calibrate_p4.py` · `run_phase4.ps1`

# PHASE 4 — ResNet-18 parça sınıflandırıcısı: tam veri, işaretçi onarımı, görüntü düzeyi kalibrasyon (AIC515, AIForge-Doc v1)

**Tarih:** 10 Kasım 2026 · **Yazar:** Harun Benli · **Donanım:** RTX 5080 16 GB · torch 2.11.0+cu128
**Plan (koşmadan önce yazıldı):** `kod/phase4_resnet/README.md` · **Otomatik tablolar:** `phase4_resnet/results/tables_p4.md`
**Tohum etiketi:** Faz 4 modelleri **3 tohumludur (42–44)**; Faz 1 referansı (`resnet18_ft_best`) tek tohumdur. Tüm sayılar `phase4_resnet/results/*.csv` dosyalarından gelir; başlık sayıları kayıtlı patch skorlarından bağımsız bir hesapla yeniden doğrulandı.

## 1. Soru
"ResNet sahte/gerçek ayrımında neden zayıf?" Kod incelemesi üç neden gösterdi: (i) uygulamadaki model 400 görüntülük Faz 1 pilotundan kalma, (ii) görüntü kararı kalibre edilmemiş sabit bir kuralla veriliyor (en yüksek patch ≥ 0.5), (iii) model yeşil işaretçi kısayoluna yaslanıyor. Faz 4 bu üçünü ayrı ayrı ölçtü ve giderdi.

## 2. Ne yapıldı
| Koşu | Değişen tek faktör | Eğitim verisi | Tohum |
|---|---|---|---|
| R0 `resnet18_ft_best` (referans) | — | 400 görüntü / 1587 parça | 42 |
| R1 `resnet18_ft_full_s{42,43,44}` | veri miktarı | 2748 görüntü / 10 795 parça | 42–44 |
| R2 `resnet18_ft_full_mr_s{42,43,44}` | işaretçi onarımı | aynı koordinatlar; 2126 görüntüde sahte görüntü onarılmış | 42–44 |

Tarif Faz 1 `ft` ayarıyla aynıdır (ImageNet ResNet-18, AdamW 1e-4, cosine, batch 32, 12 epoch, sabır 5, bf16, 256→224 kesit). Görüntü düzeyi skorlar uygulamanın protokolüyle üretildi (224 pencere, adım 112; görüntü başına medyan 75, ortalama 229 patch). Agregasyon ve eşik **yalnız val'de** seçildi; test yalnız raporlandı. Koşu başına eğitim 65–72 s; tüm hat ≈ 20 dk.

## 3. Sonuçlar

### 3.1 Parça düzeyi (aynı val parçaları: 487 görüntü, 1431 parça)
| Model | AUROC (ham val) | TPR@0.5 | FPR@0.5 | AUROC Ideogram | AUROC (işaretçisi onarılmış val) | TPR@0.5 (onarılmış) |
|---|---|---|---|---|---|---|
| Faz 1 (400 görüntü) | 0.926 | 0.784 | 0.051 | 0.837 | 0.735 | 0.306 |
| Faz 4 tam veri | **0.983 ± 0.003** | 0.912 ± 0.022 | 0.014 ± 0.006 | 0.949 ± 0.016 | 0.900 ± 0.009 | 0.582 ± 0.046 |
| Faz 4 tam veri + işaretçi onarımı | 0.982 ± 0.001 | 0.906 ± 0.007 | 0.030 ± 0.005 | 0.946 ± 0.009 | **0.980 ± 0.001** | **0.910 ± 0.008** |

### 3.2 Görüntü düzeyi (val: 487 sahte + 487 eşleşen authentic · test: 812 sahte + 89 eşli authentic)
Seçilen agregasyon **en yüksek patch (max)**: val AUROC ortalaması max 0.9655 > top2 0.962 > top3 0.958 > top5 0.951. Eşik: model başına val'de Youden-J.

| Model | Kural | Val AUROC | Val TPR | Val FPR | Test AUROC | Test TPR | Test FPR |
|---|---|---|---|---|---|---|---|
| Faz 1 | eski (max ≥ 0.5) | 0.893 | 0.916 | 0.456 | 0.919 | 0.930 | 0.404 |
| Faz 1 | kalibre (eşik 0.938) | 0.893 | 0.791 | 0.101 | 0.919 | 0.805 | 0.079 |
| Faz 4 tam veri | eski (max ≥ 0.5) | 0.965 ± 0.002 | 0.951 | 0.242 ± 0.048 | 0.978 ± 0.003 | 0.963 | 0.109 ± 0.034 |
| Faz 4 tam veri | **kalibre** | **0.965 ± 0.002** | 0.886 ± 0.003 | **0.021 ± 0.011** | **0.978 ± 0.003** | 0.901 ± 0.014 | **0.022 ± 0.011** |
| Faz 4 + işaretçi onarımı | kalibre | 0.948 ± 0.002 | 0.840 ± 0.018 | 0.061 ± 0.017 | 0.958 ± 0.004 | 0.834 ± 0.011 | 0.064 ± 0.017 |

Test AUROC grup-bootstrap %95 GA (tam veri, tohum başına): [0.960, 0.989], [0.963, 0.986], [0.971, 0.988]. Test authentic yalnız 89 görüntü olduğundan test FPR aralığı geniştir (Wilson, tam veri: ≈ 0.2–9 %).
Aynı val protokolünde diğer modeller (`kod/app/meta/image_tau.json`): U-Net bbox-GT AUROC 0.921 / 0.916 / 0.838, YOLO11s 0.961.

### 3.3 İşaretçi sondaları (val, işaretçi kalıntılı 368 kimlik; kalibre eşik)
| Model | Sahte TPR | Sahte, işaretçi onarılmış TPR | Authentic FPR | Authentic + yalnız işaretçi FPR |
|---|---|---|---|---|
| Faz 1 | 0.910 | 0.207 | 0.073 | 0.848 |
| Faz 4 tam veri | 0.995 ± 0.005 | 0.471 ± 0.028 | 0.016 ± 0.017 | 0.955 ± 0.019 |
| Faz 4 + işaretçi onarımı | 0.880 ± 0.007 | **0.899 ± 0.005** | 0.049 ± 0.010 | **0.458 ± 0.004** |

### 3.4 Test kırılımı (kalibre kural, TPR)
| Alt küme (n) | Faz 1 | Faz 4 tam veri | Faz 4 + işaretçi onarımı |
|---|---|---|---|
| İşaretçili (627) | 0.906 | 0.985 ± 0.004 | 0.863 ± 0.008 |
| İşaretçisiz (185) | 0.465 | 0.618 ± 0.047 | **0.735 ± 0.029** |
| Gemini (727) | 0.839 | 0.934 ± 0.012 | 0.845 ± 0.011 |
| Ideogram (85) | 0.518 | 0.624 ± 0.042 | **0.737 ± 0.024** |
| Alan < %1 (429) | 0.772 | 0.849 ± 0.021 | 0.822 ± 0.009 |
| Temiz test (740) | 0.800 | 0.898 ± 0.013 | 0.827 ± 0.013 |

## 4. Hipotez kararları
| Hipotez | Beklenti (koşmadan önce) | Sonuç | Karar |
|---|---|---|---|
| H6a veri miktarı | val parça AUROC ≥ 0.95 | 0.983 ± 0.003 (Faz 1 modeli aynı val'de 0.926) | **Desteklendi** |
| H6b karar kuralı | eski kural val FPR ≥ 0.20; kalibrasyon dengeli doğruluğu artırır; görüntü AUROC ≥ 0.92 | eski FPR 0.242 (Faz 1: 0.456); dengeli doğruluk 0.855 → 0.933; AUROC 0.965. top-k ortalaması max'tan **iyi değil** | **Desteklendi** (top-k önerisi desteklenmedi) |
| H6c işaretçi onarımı | onarılmış sahtede TPR düşüşü ve yalnız-işaretçi yanlış alarmı belirgin küçülür; ham val'de küçük bedel | onarılmış sahte TPR 0.471 → 0.899; yalnız-işaretçi FPR 0.955 → 0.458 (hâlâ yüksek); bedel: görüntü AUROC 0.965 → 0.948, FPR 0.021 → 0.061 | **Kısmen desteklendi** |

## 5. Yorum
1. **Zayıflığın ana nedeni veri miktarı ve karar kuralıydı.** Aynı tarif tam veriyle parça AUROC'u 0.926'dan 0.983'e, kalibre eşik val yanlış alarmını 0.242'den 0.021'e taşıdı. Eşikler 0.94–0.995 aralığında: yüzlerce patch'in en büyüğü alındığı için authentic görüntülerde de skor yükselir; 0.5 bu yüzden yanlış eşikti.
2. **Tam veri modeli büyük ölçüde bir işaretçi dedektörüdür.** İşaretçi onarılınca sahtelerin yarısını kaçırır (0.995 → 0.471) ve yalnız işaretçi eklenmiş authentic görüntülerin %96'sına sahte der. Başlıktaki 0.978 test AUROC bu kısayolla şişmiştir; işaretçisiz test sahtelerinde TPR 0.618'dir. Bu, Faz 2–3'te U-Net ve YOLO için bulunan örüntünün üçüncü model ailesinde tekrarıdır.
3. **İşaretçisiz eğitim kısayolu kaldırır ama bedeli vardır.** İşaretçisiz ve Ideogram sahtelerinde en iyi model budur (0.735 / 0.737); genel AUROC 0.958'e iner. Yalnız-işaretçi yanlış alarmı 0.458'de kalır: model eğitimde hiç işaretçi görmediği için yapıştırılmış yeşil çerçeveyi "işaretçi zararsızdır" diye öğrenmemiştir (bu sonda görüntüsü ayrıca gerçekten yapıştırma içerir).
4. **Görüntü düzeyinde ResNet rekabetçidir.** Aynı val protokolünde tam veri ResNet (0.965) U-Net bbox-GT'nin (0.838–0.921) üstünde, YOLO11s (0.961) düzeyindedir; ancak ResNet yalnız kaba bir ısı haritası verir, lokalizasyon ölçütü yoktur.

## 6. Uygulama (yalnız yerel; HF Space ve GitHub'a gönderilmedi)
- Yeni modeller `PHASE4/resnet18_ft_full_s42` ve `PHASE4/resnet18_ft_full_mr_s42` listede; varsayılan ResNet tam veri modeli.
- ResNet görüntü kararı artık `kod/app/meta/resnet_image_tau.json` içindeki kalibre eşikle verilir (Faz 1 modeli için de: 0.938). Simülasyon, toplu test, ResNet sekmesi, XAI tahmin tablosu ve Sonuçlar sekmesi güncellendi.
- Testler: `kod/tests/run_all_tests.ps1` → app 289 kontrol 0 hata, app_p3 0 hata, xai_filters 87 kontrol 0 hata, xai rc = 0. Uygulama skoru 3 model × 40 görüntüde boru hattıyla aynı kararı verdi (en büyük |Δskor| 0.0014).
- Değiştirilen dosyaların önceki hâlleri: `phase4_resnet/app_yedek_oncesi/`.

## 7. Sınırlamalar
- Test authentic kümesi 89 görüntü; test FPR'si geniş aralıklıdır. Val negatifleri sahtelerin eşleşen authentic'leridir (eşli tasarım), bağımsız belgelerde FPR farklı olabilir.
- Eşikler doygunluğa yakındır (0.94–0.995) ve tohuma duyarlıdır (0.940 ↔ 0.995); bf16 sayısal farkı (≤ 0.0014) sınırdaki görüntülerde kararı değiştirebilir. CPU'da (fp32) eşik yeniden doğrulanmadı.
- Parça indeksi tüm tohumlarda sabittir; tohum std'si başlangıç, sıra ve augmentasyon değişkenliğini ölçer, parça örneklemesini değil.
- İşaretçisiz eğitimde model seçimi onarılmış val parçalarıyla yapıldı (Faz 3 kuralı).
- Kod henüz bir commit'e bağlı değil; koşu JSON'larında betik SHA-256 özeti kayıtlı (`script_sha`).
- Dayanıklılık (JPEG, ölçekleme) ve held-out üretici deneyleri ResNet için yapılmadı.

## 8. Yeniden üretim
```powershell
powershell -ExecutionPolicy Bypass -File kod\phase4_resnet\run_phase4.ps1 -Stage Smoke
powershell -ExecutionPolicy Bypass -File kod\phase4_resnet\run_phase4.ps1 -Stage All
```
Çıktılar: `models/phase4/*.pt` (6 × 43 MB), `phase4_resnet/results/` (tablolar, `resnet_image_tau.json`), `phase4_resnet/preds/` (patch skorları), `phase4_resnet/cache/` (parça önbelleği, 5 GB — silinebilir, yeniden üretilir).

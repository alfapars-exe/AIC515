---
title: AIC515 · AIForge-Doc Sahtecilik Lokalizasyonu
emoji: 🧾
colorFrom: indigo
colorTo: red
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12"
app_file: app.py
pinned: false
license: cc-by-nc-sa-4.0
short_description: AI belge sahteciligi lokalizasyonu (U-Net, ResNet-18)
---

# AIC515 Görüntü İşleme · AIForge-Doc — Belge Sahteciliği Tespiti ve Lokalizasyonu

Topkapı Üniversitesi AIC515 Görüntü İşleme dönem projesi. Üretken yapay zekâ ile tek bir alanı değiştirilmiş fiş/form
görüntülerinde sahteciliği **sınıflandırır** (ResNet-18) ve değiştirilen pikselleri **segmente eder** (U-Net).

**Sekmeler:** `0 · Simülasyon` (veri setinden örnek seç → GT analizi → yeşil işaretçi taraması → girdi senaryosu ve bozulma →
ResNet-18 + U-Net'ler → bbox ve piksel-değişim GT'siyle metrikler, görüntü-düzeyi karar, yakın plan, indirilebilir rapor) ·
`1 · Toplu test` (N örnekte aynı hat; TPR/FPR/AUROC, boru hattının kayıtlı sonuçlarıyla yeniden-üretim farkı) ·
`2–7` girdi, 61 ön işleme işlemi (aşamalı pipeline), U-Net, ResNet-18, yeşil işaretçi, sonuçlar · `8 · XAI Analysis` › SHAP Analysis (tahmin → SHAP Partition/Gradient, authentic/blur/inpaint referansı → tespit bölgesi bazında açıklama, bölge kütlesi, modeller arası IoU/Spearman karşılaştırması, PNG/JSON/NPZ/MD dışa aktarım; ayrıntı `docs/XAI.md`) ve › Filter Visualization / Katman Filtreleri (SHAP'tan bağımsız: ilk conv katmanlarının checkpoint'teki gerçek kernel'leri, görüntüye uygulanmış aktivasyon haritaları conv/BN/aktivasyon, filtre↔harita çiftleri, orijinalle karşılaştırma; U-Net, DeepLabV3+, ResNet-18, YOLO11; ayrıntı `docs/XAI_FILTERS.md`).

Donanım: **ZeroGPU** (NVIDIA RTX Pro 6000 Blackwell, istek başına; `@spaces.GPU`).

## Veri ve modeller
Veri seti **AIForge-Doc v1** (Scam.AI; CC BY-NC-SA 4.0, kapılı) ve eğitilmiş ağırlıklar bu Space'e **özel bir
Storage Bucket** olarak bağlanır (`AI515_DATA_ROOT`, varsayılan `/data`); bucket proje dizin yapısını izler:

```
/data/data/AIForge-Doc-v1/{TrainingSet,TestingSet}/{images,masks,authentic}
/data/models/*.pt            # PHASE 1: unet_rgb, unet_rgbfreq, resnet18_{ft,probe,scratch}
/data/models/phase2/*.pt     # PHASE 2: unet_{bbox,pixel_eps8_morph,glyph_v2}_s{42,43,44} (+ geminionly)
/data/dataset_audit/samples_forged.csv, splits.json
```
Bucket bağlı değilse arayüz yine çalışır: kendi görüntünüzü yükleyebilirsiniz; model sekmeleri checkpoint bulunamadığında uyarı verir.
Ham veri seti lisans gereği yeniden dağıtılmaz; bucket özel tutulmalıdır.

## Ana bulgular (bkz. proje raporu)
- Resmî test (812), 3 tohum, bbox-GT ile eğitilen U-Net: AUPRC 0.950 ± 0.011, F1@τ* 0.846 ± 0.015 (zero-shot TruFor F1 0.403).
- Aynı tahminler eşleşen orijinalden türetilen **piksel-değişim GT'si** ile puanlandığında F1 0.45 / AUPRC 0.37 → kutu maskesi skorları şişiriyor (H1).
- Sahtelerin %75.4'ünde üretim hattının bıraktığı **yeşil işaretçi kalıntısı** var; yalnız işaretçi authentic görüntüye aktarılınca model kutuyu çiziyor (AP 0.906).

Kod: `app.py` (Gradio), `xai/` (SHAP açıklanabilirlik paketi), `baselines/features_classical.py` (56 klasik öznitelik). Dağıtım: proje deposundaki `hf_space/deploy_hf.py`.

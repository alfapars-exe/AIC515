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
short_description: AI belge sahteciligi lokalizasyonu (U-Net, DLV3+, YOLO11)
---

# AIC515 Görüntü İşleme · AIForge-Doc — Belge Sahteciliği Tespiti ve Lokalizasyonu

Topkapı Üniversitesi AIC515 Görüntü İşleme dönem projesi. Üretken yapay zekâ ile tek bir alanı değiştirilmiş fiş/form
görüntülerinde sahteciliği **sınıflandırır** (ResNet-18) ve değiştirilen pikselleri **segmente eder** (U-Net).

**Sekmeler:** `0 · Simülasyon` (veri setinden örnek seç → GT analizi → yeşil işaretçi taraması → girdi senaryosu ve bozulma →
ResNet-18 + U-Net'ler → bbox ve piksel-değişim GT'siyle metrikler, görüntü-düzeyi karar, yakın plan, indirilebilir rapor) ·
`1 · Toplu test` (N örnekte aynı hat; TPR/FPR/AUROC, boru hattının kayıtlı sonuçlarıyla yeniden-üretim farkı) ·
`2–7` girdi, 61 ön işleme işlemi (aşamalı pipeline), U-Net, ResNet-18, yeşil işaretçi, sonuçlar · `8 · Açıklanabilirlik (XAI)` › SHAP Açıklamaları (tahmin → **tam görüntü SHAP** — orijinal görüntünün tamamı, tek hedef ve tek referans, E[f]+ΣSHAP=f(x) doğrulamalı; döşemeli yerel mozaik ve pencere hedefleri ayrı; büyük yakınlaştırılabilir paneller, orijinal↔bindirme kaydırıcısı, katkı dağılımı, 300 dpi PNG/PDF — Partition/Gradient, authentic/blur/inpaint referansı → tespit bölgesi bazında açıklama, bölge kütlesi, modeller arası IoU/Spearman karşılaştırması, PNG/JSON/NPZ/MD dışa aktarım; ayrıntı `docs/XAI.md`) ve › Katman Filtreleri (SHAP'tan bağımsız: ilk conv katmanlarının checkpoint'teki gerçek kernel'leri, görüntüye uygulanmış tam görüntü natif çözünürlükte döşemeli aktivasyon haritaları conv/BN/aktivasyon, filtre↔harita çiftleri, orijinalle karşılaştırma; U-Net, DeepLabV3+, ResNet-18, YOLO11; ayrıntı `docs/XAI_FILTERS.md`).

Donanım: **ZeroGPU** (NVIDIA RTX Pro 6000 Blackwell, istek başına; `@spaces.GPU`).

## Veri ve modeller
Veri seti **AIForge-Doc v1** (Scam.AI; CC BY-NC-SA 4.0, kapılı) ve eğitilmiş ağırlıklar bu Space'e **özel bir
Storage Bucket** olarak bağlanır (`AI515_DATA_ROOT`, varsayılan `/data`); bucket proje dizin yapısını izler:

```
/data/data/AIForge-Doc-v1/{TrainingSet,TestingSet}/{images,masks,authentic}
/data/models/*.pt            # PHASE 1: unet_rgb, unet_rgbfreq, resnet18_{ft,probe,scratch}
/data/models/phase2/*.pt     # PHASE 2: unet_{bbox,pixel_eps8_morph,glyph_v2}_s{42,43,44} (+ geminionly)
/data/models/phase2/*.pt     # PHASE 3: dlv3p_*_s42 (DeepLabV3+), unetjoint_*_s42 (kutu yardımcı kaybı), *_mr_s42 (işaretçisiz eğitim)
/data/models/phase2/*.pt     # ek tohumlar: unet_bbox_mr_s{43,44}, unetjoint_pixel_eps8_morph_s{43,44} (yalnız Katman Filtreleri sekmesinde listelenir)
/data/models/phase4/*.pt     # PHASE 4: resnet18_ft_full_s{42,43,44}, resnet18_ft_full_mr_s{42,43,44} (tam veri; işaretçi onarımlı eğitim)
/data/models/phase3_yolo/*.pt # PHASE 3: YOLO11 n/s/m kutu dedektörleri (kare 1024, tam görüntü 1280, yalnız Gemini)
/data/dataset_audit/samples_forged.csv, splits.json
```
Bucket bağlı değilse arayüz yine çalışır: kendi görüntünüzü yükleyebilirsiniz; model sekmeleri checkpoint bulunamadığında uyarı verir.
Ham veri seti lisans gereği yeniden dağıtılmaz; bucket özel tutulmalıdır.

## Faz 3 (YOLO11 ve sondalar)
Lokalizasyon listelerinde U-Net'lerin yanında **DeepLabV3+**, **kutu yardımcı kayıplı U-Net**, **işaretçisiz eğitilmiş U-Net** ve **YOLO11 kutu dedektörleri** yer alır. YOLO çıktısı (1024 kare / 768 adım, kareler arası NMS) her kutunun güveniyle bir olasılık haritasına boyanır; böylece U-Net ile **aynı** metrikler, τ* eşikleri ve görüntü-düzeyi karar kullanılır (bildiri §IV-C). Simülasyona iki bölge-görünümü sondası (*yalnız yeniden-render*, *yalnız içerik*) ve iki bozulma (CLAHE, gri seviye) eklendi; `6 · Sonuçlar` sekmesi Faz 3 tablolarını gösterir. XAI SHAP analizinde YOLO11 kutu dedektörleri **tam görüntü** hedefleriyle (kutu güveni) açıklanır; pencere hedefleri piksel/patch modelleriyle sınırlıdır.

Lisans notu: YOLO11 çıkarımı **Ultralytics** (AGPL-3.0) kütüphanesini kullanır.

## Ana bulgular (bkz. proje raporu)
- Resmî test (812), 3 tohum, bbox-GT ile eğitilen U-Net: AUPRC 0.950 ± 0.011, F1@τ* 0.846 ± 0.015 (zero-shot TruFor F1 0.403).
- Aynı tahminler eşleşen orijinalden türetilen **piksel-değişim GT'si** ile puanlandığında F1 0.45 / AUPRC 0.37 → kutu maskesi skorları şişiriyor (H1).
- Sahtelerin %75.4'ünde üretim hattının bıraktığı **yeşil işaretçi kalıntısı** var; yalnız işaretçi authentic görüntüye aktarılınca model kutuyu çiziyor (AP 0.906).
- Faz 4: tam veriyle eğitilen ResNet-18 yama sınıflandırıcısı (3 tohum, kalibre eşik) test AUROC 0.978; kalıntı onarıldığında doğru tespit oranı 0.995'ten 0.471'e düşüyor.
- İşaretçisi onarılmış görüntülerle yeniden eğitilen kutu U-Net (3 tohum) bağımlılığı kaybediyor (onarım ΔAP +0.017) ve işaretçisiz AIForge-Doc v2'de 0.711 kutu AUPRC veriyor (standart 0.520).

Kod: `app.py` (Gradio), `xai/` (SHAP açıklanabilirlik paketi), `baselines/features_classical.py` (56 klasik öznitelik). Dağıtım: proje deposundaki `hf_space/deploy_hf.py`.

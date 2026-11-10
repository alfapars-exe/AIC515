# `kod/app/` — Simülasyon + Aşamalı Görüntü İşleme + Derin Öğrenme Arayüzü (Gradio)

AIC515 Görüntü İşleme dönem projesi (AIForge-Doc v1 sahtecilik lokalizasyonu) için etkileşimli ön yüz.
Veri setinden bir örnek seçip **simülasyon** olarak adım adım test etmeye, N örnekte **toplu test** koşturmaya,
dersin 1–14. hafta konularını aşamalı uygulamaya, projenin ResNet-18 / U-Net / DeepLabV3+ / YOLO11 modellerini
karşılaştırmaya ve kararlarını **açıklamaya (XAI)** yarar.

- **Canlı:** https://huggingface.co/spaces/infinayazilim/ai515 (ZeroGPU; veri seti ve modeller özel Storage Bucket'tan)
- **Kod:** `kod/app/` — tek doğruluk kaynağı. Space paketi (`hf_space/`) elle düzenlenmez; `kod/tools/build_space.py` ile buradan türetilir.

## Kod mimarisi

`app.py` ince bir giriş noktasıdır: `appcore/` modüllerini sırayla yükler, adlarını `app.<ad>` olarak yeniden dışa aktarır ve
`demo.launch()` çağırır. Bağımlılık yönü tek yönlüdür (her modül yalnız kendinden öncekileri içe aktarır):

| Katman | Modül | Görev |
|---|---|---|
| Ortak | `appcore/_common.py` | numpy, OpenCV, pandas, matplotlib, gradio, torch içe aktarmaları; `TORCH_OK` |
| Yapılandırma | `appcore/config.py` | yollar (`ROOT`, `DATA`, `MODELS`, `META`), ZeroGPU algılama, `gpu()` dekoratörü |
| Alan çekirdeği | `appcore/imaging.py` | görüntü/frekans yardımcıları, yeşil işaretçi dedektörü, piksel-değişim GT türetimi |
| | `appcore/ops.py` | ön işleme işlem kaydı (`OPS`, 13 kategori), hazır hatlar (`PRESETS`), `run_op`, `replay` |
| Veri | `appcore/dataset.py` | veri seti dizini, filtreleme, örnek yükleme |
| Çıkarım | `appcore/models.py` | model kaydı ve önbellek; U-Net / DeepLabV3+ kayan pencere, YOLO11 kareli çıkarım, ResNet-18 patch; `gpu_infer` |
| Ölçüt | `appcore/metrics.py` | F1, IoU, AP, görüntü skoru, eşikler (`tau.json`, `image_tau.json`), son işleme, bindirme |
| Senaryo | `appcore/scenarios.py` | girdi senaryoları, test-zamanı bozulmaları, kırpma ve montaj |
| Servisler | `appcore/simulation.py`, `batch.py` | tek örnek simülasyonu ve rapor dosyaları; toplu test, özet tablo ve grafik |
| | `appcore/handlers.py` | diğer sekmelerin olay işleyicileri (girdi, taslaklar, ön işleme, U-Net, ResNet, işaretçi, sonuçlar) |
| | `appcore/xai_bridge.py` | SHAP köprüsü: kurulum, tahmin, açıklama, model karşılaştırma, dışa aktarım |
| Sunum | `appcore/theme.py` | CSS, sol panel açıklamaları, ortak görsel boyları (`H_SM/MD/LG/XL`) ve renk açıklaması |
| | `appcore/ui.py` | açılış (model ön yükleme, XAI kurulumu), sekmeler, olay bağlantıları; `demo` |
| XAI | `xai/`, `xai_filters.py` | SHAP paketi (registry → targets → adapters → manager → viz/compare → service; `fullimage`) ve katman filtreleri |
| Veri dosyaları | `meta/` | eşikler, referans sonuçlar, işaretçi listesi, Faz 3 ve v2 tabloları |

## Kurulum / yerel çalıştırma (Windows, docforensics ortamı)

```powershell
conda activate docforensics
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128   # ilk kurulum
cd "<proje kökü>"            # ör. ...\AI 515\Vize Proje
python kod\app\app.py          # tarayıcıda http://127.0.0.1:7860 açılır
```

Ortam değişkenleri (isteğe bağlı): `VIZE_ROOT` proje kökü (varsayılan: `kod/` klasörünün üstü — otomatik bulunur), `AI515_DATA_ROOT` veri/model kökü (varsayılan: proje kökü),
`APP_PORT` (7860), `APP_HOST` (127.0.0.1), `APP_NO_BROWSER=1`. GPU varsa otomatik kullanılır; yoksa CPU'da çalışır.
`spaces` paketi yalnız Space'te vardır; yerelde `@spaces.GPU` dekoratörü etkisiz kalır.
Başlığın altındaki **Durum** satırı yüklü model sayısını, cihazı, örnek sayısını ve veri kökünü gösterir; uygulamanın
gerçekten çalışır durumda olduğu buradan tek bakışta doğrulanır.

## Sekmeler

| Sekme | İçerik |
|---|---|
| **0 · Simülasyon** | (1) **Örnek seçimi:** bölüm, kaynak veri seti, üretici (Gemini/Ideogram), alan oranı, yeşil işaretçi, en fazla MP filtreleri; *Filtrele*, *Rastgele seç* ya da kimlikle seçim (`testing:000000700`). (2) **Senaryo:** sahte belge · authentic eş (negatif kontrol) · sahte, işaretçi onarılmış · authentic + yalnız işaretçi (kısayol testi); test-zamanı bozulması (JPEG Q75/Q50, 0.75×/0.5× ölçek, blur σ=1, gürültü σ=5); isteğe bağlı ön işleme pipeline'ı. (3) **Modeller:** U-Net (bbox/piksel/glif GT, PHASE 1 RGB/RGB+freq) ve ResNet-18 (ft/probe/scratch). Çıktı: 7 adımlı canlı günlük, adım adım galeri, yakın plan paneli, model başına tablo (görüntü skoru/eşiği, karar, doğru/yanlış, F1/IoU/AP/Oracle-F1 bbox-GT ve piksel-GT, kutu kapsama), otomatik yorum ve MD/JSON/CSV/PNG raporu. *Bu örneği ön işleme / model sekmelerine aktar* düğmesi. |
| **1 · Toplu test** | Aynı filtrelerle N rastgele örnek (tohumlu); mod: yalnız sahte (lokalizasyon) ya da sahte + authentic eşi (dengeli tespit). Model bazında ortalama F1/IoU/AP (iki GT), TPR/FPR/doğruluk/görüntü AUROC ve boru hattının kayıtlı sonuçlarıyla **yeniden üretim farkı**; grafik + CSV/JSON. |
| **2 · Girdi** | Sahte görüntü + isteğe bağlı GT maskesi + authentic eş yükleme **veya** veri setinden örnek seçme (metadata ve authentic eş otomatik). |
| **3 · Ön işleme (aşamalı pipeline)** | 13 kategori / 61 işlem: renk, yoğunluk dönüşümleri, histogram (eşitleme, CLAHE), eşikleme (global, Otsu, adaptif, Sauvola), gürültü ekleme/giderme, keskinleştirme, kenar (Sobel, Prewitt, Scharr, Laplacian, LoG, Canny), morfoloji, geometrik, normalizasyon, frekans domeni (FFT, ideal/Gauss/Butterworth, D2 kanalları, 8×8 DCT), klasik segmentasyon (K-Means, watershed, bağlı bileşen), dayanıklılık bozulmaları ve yeşil işaretçi onarımı. **Önizle → Pipeline'a ekle → Geri al / Sıfırla**; presetler; önce/sonra + histogram; 56 klasik öznitelik. |
| **4 · U-Net lokalizasyonu** | Tüm checkpoint'ler; `infer_fullimage.py` protokolü (512 pencere, 448 adım); eşik `tau.json`'dan; Otsu/morfolojik son-işleme; GT olarak bbox ya da piksel-değişim GT'si (H1); **Karşılaştır**: orijinal vs pipeline çıktısı. |
| **5 · ResNet-18 patch sınıflandırma** | 224 px kayan pencere → patch ısı haritası, görüntü skoru, GT ile kesişen/temiz patch skorları. |
| **6 · Yeşil işaretçi sondası** | G − max(R,B) > 50 ve G > 110 dedektörü (PHASE 2 Ek-C). |
| **7 · Sonuçlar ve model kartı** | Özet tablolar, görüntü-düzeyi eşikler, ölçüt tanımları. |
| **8 · Açıklanabilirlik (XAI) › SHAP Açıklamaları** | Tahmin → SHAP (Partition / Gradient; authentic / blur / inpaint referansı) → model karşılaştırması; `xai/` paketi (`docs/XAI.md`). **Varsayılan: tam görüntü SHAP** (`xai/fullimage.py`) — modelin uygulamadaki tam görüntü skoru / tespit / GT hedefi orijinal görüntünün tamamında, tek referansla açıklanır (U-Net, DeepLabV3+, ResNet-18, YOLO11 kutu güveni); E[f]+ΣSHAP=f(x) doğrulanınca "tam görüntü SHAP (doğrulandı)" etiketi. Döşemeli yerel mozaik ayrı ve "tam görüntü SHAP değildir" etiketli; pencere hedefleri korunur. Büyük, yakınlaştırılabilir paneller, orijinal↔bindirme kaydırıcısı, katkı dağılımı, natif PNG/NPY + 300 dpi PNG/PDF dışa aktarım. |
| **8 · Açıklanabilirlik (XAI) › Katman Filtreleri** | SHAP Açıklamaları alt sekmesinden bağımsız XAI (`xai_filters.py`, `docs/XAI_FILTERS.md`). Seçilen modelin (U-Net, DeepLabV3+, ResNet-18, YOLO11) mimarisi checkpoint'ten tespit edilir; ilk conv katmanlarının (çıkış ≥ girdinin 1/4'ü) **gerçek kernel ağırlıkları** (B), görüntüye uygulanmış **aktivasyon haritaları** — conv / BN / ReLU-SiLU aşamaları, filtre↔harita çiftleri (C) — ve orijinalle karşılaştırma, kanal katkıları, en güçlü filtreler (D). **Varsayılan analiz bölgesi: tam görüntü, natif çözünürlük, döşemeli** (1024 px çekirdek + 64 px pay; tek geçişle birebir aynı); D bölümünde orijinal koordinatlarda yakınlaştırılabilir aktivasyon görüntüleyici, kaydırıcı ve yüksek çözünürlüklü dışa aktarım. Her katmanda bağımsız `F.conv2d` = modelin conv çıktısı doğrulaması; ham değerler NPZ/CSV/JSON. Test: `python kod\tests\test_xai_filters.py` (87 kontrol); rapor: `reports/xai_filters/RAPOR.md`, ölçüm: `reports/xai_fullimage/`. |

## Ölçütler (boru hattıyla birebir)
- Olasılıklar 8-bit'e nicemlenir; AP 256 kutulu histogramla tüm pikseller üzerinden kesin hesaplanır; F1@τ: `u8 >= ceil(255·τ)`.
- Görüntü-düzeyi skor = olasılık haritasının en yüksek %1 piksel ortalaması; eşik val'de 487 sahte + 487 authentic üzerinde Youden-J (`meta/image_tau.json`). ResNet-18: en yüksek patch skoru, eşik 0.5.
- Piksel-değişim GT'si: `D = max_c |sahte − authentic|`, (D > 8) → 3×3 açma → <8 px bileşen filtresi → 3×3 kapama.

## Testler
```powershell
powershell -ExecutionPolicy Bypass -File kod\tests\run_all_tests.ps1 -N 35   # dört paketin tümü → tests\run_all_summary.txt
$env:AI515_TEST_N=35; python kod\tests\test_app.py     # işlemler, senaryolar, bozulmalar, hatalı girdiler, yeniden üretim
python kod\tests\test_app_p3.py                        # Faz 3 modelleri (DeepLabV3+, YOLO11, birleşik kayıp): kayıtlı tahminlerle karşılaştırma
python -m pytest kod\tests\test_xai.py -q              # SHAP paketi
python kod\tests\test_xai_filters.py                   # katman filtreleri
python kod\tests\ui_test.py http://127.0.0.1:7860/     # Playwright tarayıcı testi (uygulama açıkken)
python kod\tools\capture_screens.py http://127.0.0.1:7860/   # rapor için güncel ekran görüntüleri → app\ekran_*.png
python kod\hf_space\live_test.py --n 24                # canlı Space: 4 senaryo + 24 çift toplu test
```
Son sonuçlar (10 Kasım 2026, kod `kod/` altına taşındıktan ve `app.py` modüllere bölündükten sonra, RTX 5080):
`test_app` 289 kontrol / 0 hata (36 model; Faz 4 modelleri eklendikten sonra) · `test_app_p3` 0 hata · `test_xai_filters` 87 kontrol / 0 hata · `test_xai` 15/15.
Not: GPU'da çıkarım bf16 yarı duyarlıkla yapıldığından (`appcore/models.py: _autocast`) SHAP adaptör testlerinin
eşleşme toleransı CUDA'da 5e-3, CPU'da 1e-5'tir (`kod/tests/test_xai.py: _tol`).

## Hugging Face Space'e dağıtım
Space paketi `hf_space/` **türetilmiş** bir klasördür: `kod/tools/build_space.py` onu `kod/app/`, `kod/baselines/features_classical.py`,
`kod/hf_space/space_src/` (Space README + requirements) ve sonuç dosyalarından üretir; `deploy_hf.py` her çalıştığında bunu kendisi çağırır.
Veri seti ve modeller Space reposuna değil, özel **Storage Bucket**'a (`infinayazilim/ai515-data`) yüklenir ve Space'e salt-okunur
`/data` olarak bağlanır (`AI515_DATA_ROOT=/data`). Donanım: ZeroGPU (`zero-a10g` bayrağı).

```powershell
conda activate docforensics
pip install -U "huggingface_hub>=2.4" gradio_client
hf auth login                       # WRITE yetkili token (org: infinayazilim)
cd "<proje kökü>"            # ör. ...\AI 515\Vize Proje
python kod\tools\build_space.py --check                                   # hf_space/ kaynaklarla aynı mı? (yazmaz)
python kod\hf_space\deploy_hf.py --dry-run                                # plan
python kod\hf_space\deploy_hf.py --hardware zero-a10g                     # veri + modeller + Space
python kod\hf_space\deploy_hf.py --skip-data --skip-models --hardware zero-a10g   # yalnız kodu güncelle
```
Lisans: AIForge-Doc v1 CC BY-NC-SA 4.0 + kapılı → bucket **özel** kalmalıdır; ham veri yeniden dağıtılmaz.

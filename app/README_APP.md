# `app/` — Simülasyon + Aşamalı Görüntü İşleme + Derin Öğrenme Arayüzü (Gradio)

AIC515 Görüntü İşleme dönem projesi (AIForge-Doc v1 sahtecilik lokalizasyonu) için etkileşimli ön yüz.
Veri setinden bir örnek seçip **simülasyon** olarak adım adım test etmeye, N örnekte **toplu test** koşturmaya,
dersin 1–14. hafta konularını aşamalı uygulamaya ve projenin ResNet-18 / U-Net modellerini karşılaştırmaya yarar.

- **Canlı:** https://huggingface.co/spaces/infinayazilim/ai515 (ZeroGPU; veri seti ve modeller özel Storage Bucket'tan)
- **Kod:** `app/app.py` (= `hf_space/app.py`, iki dosya aynı tutulur)

## Kurulum / yerel çalıştırma (Windows, docforensics ortamı)

```powershell
conda activate docforensics
pip install "gradio>=6"        # torch, smp, opencv, skimage zaten kurulu
cd "C:\Users\PARS\Desktop\PROJELER\AI 515\Vize Proje"
python app\app.py              # tarayıcıda http://127.0.0.1:7860 açılır
```

Ortam değişkenleri (isteğe bağlı): `VIZE_ROOT` proje kökü (varsayılan: `app/` klasörünün üstü), `AI515_DATA_ROOT` veri/model kökü (varsayılan: proje kökü),
`APP_PORT` (7860), `APP_HOST` (127.0.0.1), `APP_NO_BROWSER=1`. GPU varsa otomatik kullanılır; yoksa CPU'da çalışır.
`spaces` paketi yalnız Space'te vardır; yerelde `@spaces.GPU` dekoratörü etkisiz kalır.

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

## Ölçütler (boru hattıyla birebir)
- Olasılıklar 8-bit'e nicemlenir; AP 256 kutulu histogramla tüm pikseller üzerinden kesin hesaplanır; F1@τ: `u8 >= ceil(255·τ)`.
- Görüntü-düzeyi skor = olasılık haritasının en yüksek %1 piksel ortalaması; eşik val'de 487 sahte + 487 authentic üzerinde Youden-J (`meta/image_tau.json`). ResNet-18: en yüksek patch skoru, eşik 0.5.
- Piksel-değişim GT'si: `D = max_c |sahte − authentic|`, (D > 8) → 3×3 açma → <8 px bileşen filtresi → 3×3 kapama.

## Testler
```powershell
python tests\test_app.py        # 281 kontrol (işlemler, senaryolar, bozulmalar, hatalı girdiler, yeniden üretim)
python tests\ui_test.py http://127.0.0.1:7860/ tests\ui   # Playwright tarayıcı testi + ekran görüntüleri
python hf_space\live_test.py --n 24                        # canlı Space: 4 senaryo + 24 çift toplu test → reports\canli_test.json
```
Son sonuçlar (9 Kasım 2026): yerel 281 kontrol / 0 hata; UI testi hatasız; canlı ZeroGPU'da 24 eşli test örneğinde
bbox-GT U-Net F1 0.925, AP 0.964, |ΔF1| = 0.000 (kayıtlı sonuçlarla birebir).

## Hugging Face Space'e dağıtım (`hf_space/`)
Veri seti ve modeller Space reposuna değil, özel **Storage Bucket**'a (`infinayazilim/ai515-data`, 12.208 dosya, 10.26 GiB)
yüklenir ve Space'e salt-okunur `/data` olarak bağlanır (`AI515_DATA_ROOT=/data`). Donanım: ZeroGPU (`zero-a10g` bayrağı).

```powershell
conda activate docforensics
pip install -U "huggingface_hub>=2.4" gradio_client
hf auth login                       # WRITE yetkili token (org: infinayazilim)
cd "C:\Users\PARS\Desktop\PROJELER\AI 515\Vize Proje"
python hf_space\deploy_hf.py --dry-run                               # plan
python hf_space\deploy_hf.py --hardware zero-a10g                    # veri + modeller + Space
python hf_space\deploy_hf.py --skip-data --skip-models --hardware zero-a10g   # yalnız kodu güncelle
```
Lisans: AIForge-Doc v1 CC BY-NC-SA 4.0 + kapılı → bucket **özel** kalmalıdır; ham veri yeniden dağıtılmaz.

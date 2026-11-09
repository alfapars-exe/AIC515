# XAI · Katman Filtreleri (Filter Visualization)

Uygulamanın **9 · Katman filtreleri** sekmesi. SHAP sekmesinden (8, `app/xai/`) **bağımsız** ikinci açıklanabilirlik
özelliği: eğitilmiş modellerin **ilk konvolüsyon katmanlarındaki gerçek filtre ağırlıklarını** (kernel) ve bu filtrelerin
görüntüye uygulanmasıyla oluşan **aktivasyon (öznitelik) haritalarını** gösterir.

| | SHAP Analysis (sekme 8) | Filter Visualization (sekme 9) |
|---|---|---|
| Soru | Hangi görüntü bölgeleri modelin **kararını** ne yöne itti? | Model ilk katmanlarda görüntüden **hangi düşük seviyeli özellikleri** (kenar, çizgi, doku, renk) çıkarıyor? |
| Girdi | tahmin + referans (authentic / blur / inpaint) | yalnız görüntü + kayıtlı ağırlıklar |
| Kod | `app/xai/` | `app/xai_filters.py` |

## İlkeler

- **Filtre üretilmez, eğitilmez.** Kernel = checkpoint dosyasındaki tensör (`state_dict`) ya da Ultralytics checkpoint'indeki
  model nesnesinin parametresi. Conv katmanı olmayan modelde `NoConvError` döner (gerçekte olmayan filtre gösterilmez).
- **Ağırlık ve mimari değiştirilmez.** `eval()` + `no_grad`; analiz öncesi/sonrası ağırlık SHA-256 karşılaştırılır, testte .pt
  dosyasının özeti de kontrol edilir. YOLO'da Conv+BN birleştirme (fuse) **yapılmaz**; fp16 → fp32 yalnız tür dönüşümüdür.
- **Ara çıktılar modelin kendi forward'ından** ileri kancalarla (forward hook) alınır. Her conv katmanında, katmanın gerçek
  girdisine checkpoint kernel'i `F.conv2d` ile (aynı stride / padding / dilation / groups / bias) **bağımsız olarak** da
  uygulanır ve modelin çıktısıyla karşılaştırılır → arayüzde ve raporda `max|Δ|` / göreli fark.
- **İlk katmanlar sabit sayı değildir:** forward yürütme sırasındaki `Conv2d` katmanları, çıkış çözünürlüğü girdinin en az
  1/4'ü (birikimli adım ≤ 4) olduğu sürece alınır (ilk conv her zaman; en fazla 6–8). Sınır aşılınca forward **durdurulur**:
  derin katmanlar hesaplanmaz (bellek/süre kontrolü).
- **Aşamalar:** her conv için `conv` (ham, BN öncesi) → `BatchNorm` → `aktivasyon (ReLU/SiLU)` yalnızca gerçekten
  zincirleniyorsa (girdi = önceki aşamanın çıktısı, birebir) kaydedilir. Artık (residual) toplama araya girerse zincir kapanır.
- **Kanal birleştirme (RGB):** ilk katmanda seçili filtrenin her giriş kanalına ayrı uygulanması `W[f,c] ⋆ x_c` gösterilir;
  bunların toplamı + bias = conv çıktısı (D bölümünde fark yazılır).

## Modeller (mimari checkpoint'ten otomatik tespit edilir)

| Aile | Tespit | İlk katmanlar (bu projede) | Ön işleme |
|---|---|---|---|
| `smp.Unet` (bbox/piksel/glif, Gemini-only, MR, e20), PHASE1 U-Net (`net.` öneki) | `encoder.*` + `decoder.blocks` | `encoder.conv1` 64×3×7×7 s2 (1/2) → `layer1.{0,1}.conv{1,2}` 64×64×3×3 (1/4) | ImageNet ort./std |
| PHASE1 U-Net RGB+freq | `encoder.conv1` giriş = 5 | `encoder.conv1` 64×**5**×7×7 … | + FFT-HP (D0=20) ve Laplacian kanalları, pencere başına standartlaştırma |
| `smp.DeepLabV3Plus` | `decoder.aspp` | U-Net ile aynı ResNet-18 gövdesi | ImageNet |
| Birleşik U-Net (`unetjoint_*`, 2 sınıf) | `segmentation_head` çıkış = 2 | aynı gövde | ImageNet |
| torchvision ResNet-18 (PHASE1 ft/probe/scratch, H4) | `conv1` + `fc` | `conv1` 64×3×7×7 → `layer1` | ImageNet |
| Ultralytics YOLO11s (Faz 3) | checkpoint'te `DetectionModel` | `model.0.conv` 32×3×3×3 s2 → `model.1.conv` 64×32×3×3 s2 → C3k2 (`model.2.*`) 1×1 / 3×3 | RGB/255, 114 gri dolgu |

YOLO tam checkpoint'i ultralytics paketi gerektirir. Space'te bu paket yoktur; bunun yerine
`python tools\xai_filter_report.py --export-yolo-stems` ile `models/phase3_yolo/<run>_stem.pt` dosyaları üretilir
(ilk iki düz Conv bloğunun **gerçek** ağırlıkları + BN istatistikleri; tam checkpoint ile birebir aynı çıktı — test T4).

## Arayüz

- **A · Model ve görüntü seçimi:** model, görüntü kaynağı (veri seti / Girdi sekmesi / yükleme), analiz bölgesi
  (natif çözünürlükte GT kutusu merkezli kırpma ya da küçültülmüş tam görüntü), bölge boyutu N (64–1536 px; bellek kontrolü),
  cihaz (otomatik / GPU / CPU), en fazla ilk-katman sayısı. Katman açılır listesi analizden sonra tespit edilen katmanlarla dolar.
- **B · Kernel / filtre ağırlıkları:** katman adı, filtre sayısı, kernel boyutu, giriş/çıkış kanalı, stride/padding; tüm
  filtrelerin grid'i (RGB bileşik, 0 = gri; filtre başına ya da katman geneli ölçek); seçili filtrenin kanal kanal ağırlık
  matrisi (hücrelerde değerler) ve değer tablosu; BN parametreleri.
- **C · Filtre aktivasyon haritaları:** "filtre #i ↔ öznitelik haritası #i" çiftleri (16'lık sayfalar), tüm haritaların
  grid'i, seçili haritanın **ham değer ölçeğinde** renk çubuğu + histogramı, filtre bazında ham istatistik tablosu
  (min/maks/ort/std/%>0, GT içi/dışı yanıt oranı, ilk katmanda sezgisel tür).
- **D · Orijinal görüntü ile karşılaştırma:** orijinal bölge, kanal katkıları, Σ+bias, conv/BN/aktivasyon aşamaları,
  |aktivasyon| bindirmesi; görüntüde en güçlü yanıt veren 8 filtre (GT varsa GT içi/dışı oranına göre).
- **Katman özeti ve doğrulama:** tespit edilen tüm ilk katmanlar, doğrulama farkları, SHA-256, süre, tepe GPU belleği.
- **Normalizasyon** yalnız görüntüleme içindir: harita başına min-max, %1–%99 kırpma, katman geneli simetrik (0 = gri) ya
  da katman geneli min-max. Ham değerler tabloda ve **Ham değerleri indir** (NPZ: ağırlık, bias, tüm aşamalar; CSV; JSON) ile korunur.

## GPU ve bellek

Cihaz `otomatik` iken boş GPU belleği ≥ 1.5 GiB ise cuda, değilse CPU kullanılır. Model analiz süresince cihaza taşınır,
sonra CPU'ya döner ve önbellek boşaltılır. Haritalar CPU'da saklanır (bölge > 768² ya da ZeroGPU'da float16). ZeroGPU'da
hesap `@spaces.GPU` içinde yapılır. GPU'da cuDNN TF32 nedeniyle bağımsız F.conv2d ile fark ~1e-3 göreli düzeydedir
(CPU'da ~1e-6).

## Test ve rapor

```powershell
python tests\test_xai_filters.py                       # sentetik + tüm checkpoint'ler + YOLO stem + GPU + görseller
python tools\xai_filter_report.py --export-yolo-stems  # reports/xai_filters/RAPOR.md + figs/
```

# XAI · SHAP açıklamaları (sekme 8 · XAI Analysis › SHAP Analysis ve `xai/` paketi)

> Sekme 8 iki bağımsız alt sekmeden oluşur: **SHAP Analysis** (bu belge) ve **Filter Visualization / Katman Filtreleri** (`xai_filters.py`, [XAI_FILTERS.md](XAI_FILTERS.md)).

Bu belge, AIC515 uygulamasına eklenen SHAP tabanlı açıklanabilirlik modülünü anlatır: hangi model için hangi
yöntemin neden seçildiği, mimari, kullanım ve sınırlamalar. Test sonuçları: [`xai_test_report.md`](xai_test_report.md).

## 1. Model envanteri

| Aile | Checkpoint'ler | Mimari | Görev | Girdi / ön işleme | Çıktı ve karar | SHAP |
|---|---|---|---|---|---|---|
| ResNet-18 patch | `resnet18_{ft,probe,scratch}_best`, `phase2/h4_resnet18`, `phase2/h4_resnet18_poscontrol` | torchvision resnet18, `fc = Linear(512, 1)` | ikili sınıflandırma (224 px patch) | 224 natif patch, ImageNet normalizasyonu, kayan pencere adım 112 | sigmoid patch skoru; görüntü kararı = maks patch ≥ 0.5 | ✅ Partition (+ Gradient) |
| U-Net, 3 kanal | `unet_rgb_best`, PHASE 2 `unet_{bbox,pixel_eps8_morph,glyph_v2}_s{42,43,44}`, 3× `*_geminionly_s42` | smp.Unet, resnet18 kodlayıcı, 1 sınıf | ikili segmentasyon | 512 px pencere / 448 adım, örtüşme ortalaması, uzun kenar > 4096 ise küçültme | piksel olasılığı; görüntü skoru = en yüksek %1 piksel ortalaması ≥ `IMG_TAU` (Youden-J); bölge = olasılık ≥ τ* | ✅ Partition (+ Gradient) |
| U-Net, 5 kanal | `unet_rgbfreq_best` | aynı, `in_ch = 5` | aynı | RGB + FFT yüksek-geçiren + Laplacian; frekans kanalları pencere başına standartlaştırılır | aynı | ✅ Partition (frekans kanalları maskelenmiş RGB'den yeniden hesaplanır) + Gradient (kanal grubu katkısı) |
| Klasik ML | SVM-RBF, RandomForest | 56 öznitelik (`features_classical.py`) | — | — | **model dosyası yok** | ⛔ UNSUPPORTED (sahte çıktı üretilmez) |
| YOLO11s kutu dedektörü (Faz 3) | `models/phase3_yolo/aiforge-*-yolo11s-*.pt` | Ultralytics DetectionModel | kutu tespiti (tek sınıf) | 1024 kare / 768 adım (ya da `-full-`: tam görüntü 1280), RGB/255, conf 0.001, NMS iou 0.6 | kutu + güven | ✅ yalnız **tam görüntü** hedefleri (kutu güveni) |
| ViT / EfficientNet | — | — | — | — | projede yok | N/A |

Tüm modeller tek sigmoid çıkışlıdır, yani hedefler olasılık ölçeğindedir [0, 1]. Modeller arasındaki fark ölçekte
değil, **hedef tanımında** ve **referans değerdedir**. Kayıt (registry) dinamiktir: uygulamanın listelediği
checkpoint'lerin tamamı otomatik olarak açıklanabilir hale gelir.

## 2. Ne açıklanıyor? Hedefler

SHAP her zaman modelin **gerçek** çıktısından türetilen **skaler** bir hedefi açıklar. Adaptör, uygulamanın kendi ön
işleme ve ileri geçiş fonksiyonlarını çağırır (`_unet_prepare`, `_unet_window_forward`, `norm_chw`,
`project_freq_channels`, `image_score`); kopya kod yoktur.

| Arayüz seçeneği | ResNet-18 | U-Net |
|---|---|---|
| **Karar hedefi** | kararı belirleyen **maks patch**'in p(sahte) değeri (224×224) | en yüksek %1 pikselin ağırlık merkezine ortalanmış 512 px **penceredeki görüntü skoru** |
| **Tespitler (ilk k)** | en yüksek k patch (örtüşme < 0.25) | τ* üstü **bağlı bileşenler**; güven = bölgenin ortalama olasılığı; her bölge ayrı açıklanır |
| **GT kutusu** | GT merkezli penceredeki maks patch | GT kutusu içindeki ortalama olasılık |
| **Ortak pencere** | penceredeki uygulama ızgarası patch'lerinin maksimumu | pencere görüntü skoru (tüm modeller aynı fiziksel pencerede) |

"Nesne tespiti" gereksinimi YOLO olmadığı için U-Net bağlı bileşenleri ve ResNet top-k patch'leriyle karşılanır.
Her tespit kutusu, güven değeri ve ayrı bir haritayla gösterilir.

**Önemli fark: pencere skoru ve uygulama skoru.** Uygulamanın tahmin tablosundaki U-Net görüntü skoru, 448 px
adımlı pencerelerin **örtüşme ortalamasından** hesaplanır. SHAP ise tek bir 512 px pencerenin çıktısını açıklar,
çünkü modelin eğitim ve çıkarım birimi budur ve tam görüntü üzerinde maskeleme çok pahalıdır. Bu iki sayı
farklı olabilir. Örneğin `testing:000000038` örneğinde ızgaradaki bir pencere 0.94 verirken komşu pencere ~0 verdiği
için ortalama 0.12'ye düşüyor ve görüntü FN oluyor. Arayüz her açıklamanın yanında iki değeri de gösterir; fark
0.25'ten büyükse bunu ayrıca belirtir. ResNet maks patch hedefi ise uygulama skoruyla birebir aynıdır (testte doğrulanır).

## 2a. Tam görüntü SHAP (varsayılan hedefler)

Önceki hedefler tek bir 512 px pencereyi ya da 224 px patch'i açıklıyordu: harita orijinal belgenin yalnız bir bölümünü
kapsıyordu ve U-Net'te açıklanan pencere skoru uygulamanın tam görüntü skorundan farklıydı (§2). Yeni varsayılan hedefler
(`xai/fullimage.py`) modelin **uygulamadaki tam görüntü çıktısını** açıklar:

| Arayüz seçeneği | U-Net / DeepLabV3+ (3/5 kanal) | ResNet-18 | YOLO11s |
|---|---|---|---|
| **Tam görüntü · karar** | tam olasılık haritasının top-%1 ortalaması (= uygulamanın görüntü skoru; 512/448 pencere protokolü, örtüşme ortalaması) | tüm ızgaranın maks patch skoru (= uygulama skoru) | en güvenli kutunun güveni (IoU ≥ 0.5 eşleşen tahminlerin maks güveni) |
| **Tam görüntü · tespitler** | τ* üstü bağlı bileşenin ortalama olasılığı | tek tespit patch'inin skoru | k kutu, her biri ayrı |
| **Tam görüntü · GT** | GT kutusu içi ortalama olasılık | GT ile kesişen patch'lerin maks skoru | GT kutusuyla IoU ≥ 0.5 kutuların maks güveni |
| **Döşemeli yerel mozaik** | 512 px / 128 px örtüşmeli pencereler, her biri kendi pencere skorunu açıklar | pencere içi maks patch | — |

**Mekanizma.** shap'ın piksel düzeyi `maskers.Image`'ı tam çözünürlükte pratik değildir (ölçüm: 1296×750'de modelsiz 27 s,
3 MP'de 99 s yalnız maske/ağaç maliyeti). Bu yüzden maske bir **ızgara** üzerinde tanımlanır: `PartitionExplainer`'a
(gh, gw, 1) boyutlu bir gösterge görüntüsü verilir (uzun kenar 192 hücre; 864 px'lik görüntüde 5 px hücre); her
değerlendirmede gösterge en yakın komşu ile tam çözünürlüğe taşınır ve görüntü = gösterge ? x : referans olarak kurulur.
Böylece açıklama **tek hedef ve tek referansla tüm görüntüyü** kapsar; toplamsallık kesindir: E[f] + ΣSHAP = f(x),
E[f] = f(referans görüntüsü). Hücre değeri hücre piksellerine eşit dağıtılır (toplam korunur); U-Net > 4096 px küçültmesinde
harita orijinal çözünürlüğe alan korunarak taşınır. Değerlendirmeler toplu (batch) yapılır; pencere/patch/kare çıktıları içerik
özetiyle önbelleğe alınır (maskelenmemiş pencereler yeniden hesaplanmaz).

**Döşemeli mozaik neden "tam görüntü SHAP" değildir?** Her pencere kendi skorunu kendi tabanıyla (E[f]) açıklar; farklı
tabanlı SHAP değerlerini toplamak tek bir hedefin açıklaması olmaz. Mozaik kosinüs ağırlıklarıyla orijinal koordinatlarda
birleştirilir (dikiş azaltma), kapsama ve pencere tabanlarının farkı raporlanır ve arayüzde/dışa aktarımda **"tam görüntü SHAP
DEĞİLDİR"** etiketiyle sunulur. Bir sonuç yalnız şu koşullar doğrulandığında **"TAM GÖRÜNTÜ SHAP (doğrulandı)"** olarak etiketlenir:
harita boyutu = orijinal görüntü, hedef tam görüntü hedefi, |E[f] + ΣSHAP − f(x)| < 1e-3. Aksi halde uyarı gösterilir.

**Görselleştirme.** Orijinal tam görüntü · SHAP ısı haritası · bindirme · pozitif / negatif katkı · SAHTE | GERÇEK sınıf
panelleri · **orijinal ↔ bindirme kaydırıcısı** · **katkı dağılımı** (piksel başına SHAP histogramı, birikimli kütle eğrisi —
katkının görüntünün ne kadarında yoğunlaştığı —, Σ pozitif / Σ negatif / net ve GT içi/dışı toplamları). Hepsi aynı koordinat
sisteminde, en-boy korunarak (ekranda ≤ 2400 px; hesap ve dışa aktarım natif çözünürlükte) gösterilir; görüntüler fare
tekerleğiyle yakınlaştırılır, sürüklenir, tam ekran açılır. Dışa aktarımda her tam görüntü sonucu için natif çözünürlükte
ısı haritası ve bindirme PNG, `shap_float32.npy` ve 300 dpi PNG + PDF yayın şekli (renk çubuğu, piksel eksenleri, GT ve hedef
konturu) vardır.

**ZeroGPU.** Tam görüntü işinin süresi pencere/patch/kare sayısı × bütçe ile tahmin edilir; çağrı 120 s sınırını aşacaksa
`max_evals` o iş için düşürülür ve sonuç notunda yazılır.

## 3. Yöntem seçimi ve gerekçeler

| | Partition (birincil) | Gradient (ikincil) |
|---|---|---|
| Ne yapar | Hiyerarşik Owen değerleri; görüntü bölgeleri referansla maskelenir | Beklenen gradyanlar (expected gradients), fp32, model-girdi uzayında |
| Neden | Modelden bağımsızdır; uygulamanın tam ön işlemesini (8-bit nicemleme, top-%1, frekans kanalları) olduğu gibi kullanır; **toplamsallık kesindir** (ΣSHAP = f(x) − f(referans)) | Piksel düzeyinde ince doku izi verir; 5 kanallı U-Net'te RGB / FFT-HP / Laplacian grup katkısını ayrı raporlar |
| Bütçe | `max_evals` 300–3000 (arayüz) | 64 örnek |
| Toplamsallık | kesin (test: < 1e-3, ölçülen ~1e-8) | yaklaşık |

Reddedilen yöntemler:
- **DeepExplainer:** torchvision `BasicBlock` aynı `ReLU(inplace)` modülünü iki kez çağırır ve artık toplama içerir. shap'ın PyTorch DeepLIFT kancaları bunu doğru işlemez; düzeltmek mimariyi değiştirmeyi gerektirir.
- **KernelExplainer:** görüntüde çok yavaştır; Partition onun hiyerarşik eşdeğeridir.

**Referans (arka plan)** (`shap.maskers.Image(mask_value=…)`; shap 0.54.1 numpy dizi referansını destekler):
- `authentic eş` (varsa varsayılan): gizlenen pikseller orijinal belgeyle doldurulur. ΣSHAP = f(sahte) − f(orijinal) olur ve **yalnız değiştirilmiş pikseller** katkı alabilir. En anlamlı karşı-olgusal referans budur.
- `blur(32,32)` (eş yoksa varsayılan) ve `inpaint_telea` (yalnız Partition). Kullanılan referans sonuçta ve raporda her zaman yazılır.

**Sınıflara göre katkı:** Modeller ikili ve tek çıkışlıdır: p(gerçek) = 1 − p(sahte). GERÇEK sınıfının SHAP
haritası SAHTE'nin tam negatifidir; ayrı hesaplanmaz, iki panel olarak gösterilir. **Bölge kütlesi** grafiği,
pozitif SHAP kütlesinin GT kutusu içi / kutu kenar halkası (±6 px) / yeşil işaretçi pikselleri / geri kalan
arasındaki dağılımını verir. Bu grafik projenin işaretçi kısayolu bulgusunu (PHASE 2 Ek-C) doğrudan ölçer.

## 4. Mimari — `xai/` paketi (app.py'nin yanında)

```
xai/
  registry.py   Deps (app fonksiyonları enjekte edilir; döngüsel import yok) · ModelSpec · build_registry
                · UNSUPPORTED_MODELS (SVM/RF UNSUPPORTED, YOLO/ViT N/A + gerekçe)
  targets.py    TargetSpec (pencere, bölge, geçerli pikseller, patch'ler) · decision / detection / gt / common_window
                hedef üreticileri · unet_geometry (= app._unet_prepare) · window_mask
  fullimage.py  tam görüntü hedefleri (U-Net/ResNet/YOLO) · ızgara göstergeli Partition (grid_explain) · toplu, önbellekli
                değerlendiriciler (UNetFull, ResNetFull, YoloFull) · döşemeli yerel mozaik · run_full_jobs
  viewer.py     ekran ölçekleme (en-boy korunur) · katkı dağılımı şekli · 300 dpi PNG/PDF yayın şekli · natif dışa aktarım ·
                yakınlaştırma/kaydırma JS + CSS (SHAP ve Kernel sekmeleri ortak)
  adapters.py   BaseShapAdapter → predict_batch(NHWC 0–255) → (N,1) · ResNetPatchAdapter · UNetWindowAdapter (3/5 kanal)
                · UnsupportedAdapter (XAIUnsupported) · run_jobs (GPU fonksiyonunun içinde) · model_fingerprint
  manager.py    XAIManager: LRU önbellek (görüntü sha1 + model + hedef + yöntem + referans + bütçe); yalnız eksikler hesaplanır
  viz.py        RdBu (simetrik ölçek) ısı haritası, bindirme, pozitif/negatif, SAHTE|GERÇEK, bölge kütlesi, konum, özet şekil
  compare.py    ölçek/hedef uyumu, ortak kesişime kırpma, normalizasyon, top-%5 IoU/Dice, 16×16 blok Spearman, ortak/özgü harita
  service.py    ExplanationResult · JSON / PNG / NPZ (float16) / MD dışa aktarım
```

`app.py`'deki değişiklikler (sınırlı):
1. `_unet_prepare` ve `_unet_window_forward` yardımcıları: `_predict_unet` içindeki hazırlık ve pencere ileri geçişi bu iki fonksiyona taşındı. Davranış değişmedi; 3 ve 5 kanallı ve küçültme yolunda **bit düzeyinde aynılık** test edilir.
2. `xai_setup()`: modeller yüklendikten sonra bağımlılıkların enjeksiyonu, kayıt ve yönetici.
3. `_gpu_explain` (`@gpu(duration=_dur_explain)`, `_gpu_infer` ile aynı desen). SHAP nesneleri GPU fonksiyonunun içinde kurulur; dönüş yalnız numpy (float16) ve meta veridir. `gpu_explain`, işleri ZeroGPU'nun ≤ 120 s sınırına sığacak parçalara böler.
4. Sekme 8 ve olayları. 0–7 sekmeleri, `gpu_infer`, `_predict_*`, metrikler ve ağırlıklar değişmedi.

Bütünlük garantileri (testli): SHAP öncesi ve sonrası `state_dict` sha256 aynı, tüm `param.grad is None`
(gradyan açıklamasında parametreler geçici olarak dondurulur), modeller `eval()` modunda ve `gpu_infer` çıktıları
bit düzeyinde aynı.

## 5. Kullanım

**Arayüz (sekme 8):**
1. **Görüntü kaynağı:** Girdi sekmesi / pipeline çıktısı / son simülasyon girdisi / yeni yükleme (+ isteğe bağlı authentic eş ve GT). Simülasyondaki "Bu örneği … aktar" düğmesi Girdi sekmesini doldurur.
2. **① Tahmin yap:** uygulamanın kendi çıkarımı; skor, eşik, karar ve tespit sayısı.
3. **② SHAP analizini başlat:** hedef, yöntem, referans ve `max_evals` seçilir. Önbellekte olanlar yeniden hesaplanmaz.
4. **③ Modelleri karşılaştır:** tüm modeller aynı pencerede açıklanır; IoU/Dice, Spearman ve ortak/özgü bölge çıkar. Hedef türleri farklıysa (ResNet ↔ U-Net) yalnız normalize/sıralama ölçütleri kullanılır ve uyarı gösterilir.
5. **Dışa aktarım:** PNG paneller, `shap_values.npz` (float16), `shap_report.json`, `shap_report.md`, `karsilastirma.png`.

**Program içinden:**
```python
import app as A
X = A.XAI
res = A.gpu_infer(["PHASE2/unet_bbox_s42"], ["PHASE1/resnet18_ft_best"], [rgb], 112)[0]
jobs = X.decision_targets(res, A.XAI_SPECS, rgb.shape, A.XAI_DEPS)
out = A.XAI_MGR.explain(rgb, auth, jobs, X.ExplainConfig(method="partition", reference="auto", max_evals=1000))
cmp = X.compare.compare_results(out, rgb)
```

**Test ve benchmark:**
```
pytest tests/test_xai.py -q                       # 15 test, CPU'da ~1.5 dk
python scripts/xai_benchmark.py --n 24            # GPU; → docs/xai_test_report.md + CSV
```

## 6. Sınırlamalar

- **Pencere ve tam görüntü:** "Pencere" hedefleri 512 px penceredir (bkz. §2); tam görüntü hedefleri için bkz. §2a. Tam görüntü haritasının uzamsal çözünürlüğü ızgara hücresiyle (uzun kenar / 192 px) ve `max_evals` bütçesiyle sınırlıdır; Partition hiyerarşisi bütçe azaldıkça daha kaba bloklar bırakır.
- **Tam görüntü + Gradient:** desteklenmez; ızgara maskeli Partition kullanılır ve sonuç notunda yazılır. Tam görüntüde `inpaint` referansı maskeye bağlı olduğundan `blur(32,32)` kullanılır (not edilir).
- **Partition çözünürlüğü:** harita, `max_evals` bütçesiyle belirlenen hiyerarşik bölümleme kadar ince olur. 1000 değerlendirmede 512 px pencerede bloklar onlarca piksel büyüklüğündedir. İnce doku için Gradient kullanılmalıdır. Düşük bütçede (< 200) tohumlar arası IoU yapay olarak yüksek çıkabilir, çünkü bloklar çok kabadır.
- **5 kanallı U-Net:** frekans kanalları her değerlendirmede yeniden hesaplanır. 2.5 MP'ye kadar tam görüntü kullanılır, daha büyük görüntülerde pencere ± 256 px bağlam kullanılır (yaklaşık; sonuçta not edilir). Bu model Partition'da en yavaş olanıdır.
- **> 4096 px görüntüler:** U-Net küçültülmüş görüntüyü görür, ResNet ise natif çözünürlüğü. Bu durumda ResNet ile U-Net piksel düzeyinde karşılaştırılmaz (ölçek uyarısı).
- **Pipeline çıktısı + authentic referans:** referans işlenmemiş authentic eştir; fark kısmen ön işlemeden gelir (arayüzde not edilir).
- **ZeroGPU:** çağrı başına ≤ 120 s. Çok model × yüksek `max_evals` otomatik olarak birden çok GPU çağrısına bölünür. İlk çağrıda shap'ın numba derlemesi birkaç saniye ekler.
- **Gradient toplamsallığı** yaklaşıktır. Tabloda hata ayrıca raporlanır.

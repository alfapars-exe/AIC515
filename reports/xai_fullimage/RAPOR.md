# XAI · SHAP ve Kernel analizlerinin TAM GÖRÜNTÜ üzerinde görselleştirilmesi — rapor

Tarih: 10 Kasım 2026 · Donanım (ölçüm): NVIDIA RTX 5080 16 GB, Windows · Kod: `app/xai/fullimage.py`, `app/xai/viewer.py`,
`app/xai_filters.py`, `app/app.py` (SHAP sekmesi) · Ölçümler: [`RAPOR_OLCUM.md`](RAPOR_OLCUM.md), `bench.json`, `checks.json`

## 1. Teknik analiz (değişiklikten önce)

| | SHAP Analysis (önceki) | Filter Visualization (önceki) |
|---|---|---|
| Açıklanan birim | U-Net/DeepLabV3+: top-%1 merkezli **tek 512 px pencere**; ResNet-18: **tek 224 px patch** (pencere içi maks patch) | ilk conv katmanlarının aktivasyonları **GT/merkez çevresinden 512×512 natif kırpma** ya da küçültülmüş görüntü |
| Neden kırpma/pencere | modelin eğitim/çıkarım birimi pencere/patch; shap `maskers.Image` tam çözünürlükte çok yavaş (1296×750'de modelsiz 27 s, 3 MP'de 99 s yalnız maske/ağaç maliyeti) | bellek sınırı (64 kanal × H/2 × W/2 × 3 aşama float32) |
| Sonuç | harita belgenin küçük bir bölümünü kapsıyordu; U-Net'te açıklanan pencere skoru uygulamanın tam görüntü skorundan farklı olabiliyordu (ör. 0.77 ↔ 0.44) | aktivasyon haritası belgenin yalnız merkez/GT çevresini gösteriyordu |

Modeller ve ön işleme (değişmedi): U-Net / DeepLabV3+ / birleşik U-Net (smp, ResNet-18 kodlayıcı; 3 kanal ImageNet ya da 5 kanal
RGB + FFT-HP + Laplacian; 512 pencere / 448 adım, örtüşme ortalaması, uzun kenar > 4096 ise küçültme), ResNet-18 patch
sınıflandırıcı (224 / 112 adım), YOLO11 n/s/m kutu dedektörleri (1024 kare / 768 adım ya da `-full-` modelde tam görüntü 1280;
conf 0.001, NMS 0.6). Kernel çıkarımı: modelin kendi forward'ı + ileri kancalar; her conv bağımsız `F.conv2d` ile doğrulanır.
SHAP: `PartitionExplainer` (+ Gradient), referans authentic eş / blur(32,32) / inpaint.

## 2. Yapılan değişiklikler

**SHAP — tam görüntü (varsayılan hedefler).** Hedef, modelin uygulamadaki **tam görüntü çıktısıdır** (U-Net: tam olasılık
haritasının top-%1 ortalaması = karar skoru; ResNet: tüm ızgaranın maks patch'i; YOLO: en güvenli kutunun güveni; ayrıca
tespit bölgeleri / patch'ler / kutular ve GT kutusu). Maske bir ızgarada tanımlanır (uzun kenar 192 hücre), her değerlendirmede
tam çözünürlüğe taşınır: görüntü = gösterge ? x : referans. Böylece **tek hedef + tek referans** ile tüm görüntü açıklanır ve
toplamsallık kesindir (E[f] + ΣSHAP = f(x), E[f] = f(referans)). Değerlendirmeler toplu yapılır, pencere/patch/kare çıktıları
içerik özetiyle önbelleğe alınır. Değerler orijinal koordinatlarda (H×W) saklanır; U-Net küçültmesinde alan korunarak taşınır.

**Farklı referanslı parçaların birleştirilmesi.** "Döşemeli yerel mozaik" ayrı bir seçenektir: 512 px / 128 px örtüşmeli
pencerelerin her biri kendi skorunu kendi tabanıyla açıklar, kosinüs ağırlıklarıyla orijinal koordinatlarda birleştirilir.
Tabanlar farklı olduğu için bu harita **tam görüntü SHAP olarak sunulmaz** (arayüz, rapor ve dışa aktarım etiketi; pencere
tabanlarının farkı ve kapsama raporlanır). Bir sonuç yalnız şu üç koşul doğrulanınca **"TAM GÖRÜNTÜ SHAP (doğrulandı)"**
etiketini alır: harita boyutu = orijinal görüntü, hedef tam görüntü hedefi, |E[f] + ΣSHAP − f(x)| < 1e-3.

**Kernel — tam görüntü, natif çözünürlük, döşemeli (varsayılan).** Görüntünün tamamı modelin ön işlemesinden geçer;
1024 px çekirdek + 64 px kenar paylı döşemelerle modelin kendi forward'ından geçirilir, yalnız çekirdek çıktıları orijinal
koordinatlardaki `⌈H/s⌉×⌈W/s⌉` haritaya yazılır. Kenar payı ilk katmanların alıcı alanından büyük olduğundan sonuç tek
geçişle aynıdır. Bellek bütçesi aşılırsa haritalar f×f alan ortalamasıyla saklanır (yalnız geçerli piksellerle). Kernel
ağırlıkları (B) ve aktivasyonlar (C/D) ayrı panellerde kalır.

**Arayüz (iki sekmede de).** Orijinal tam görüntü, sonuç, bindirme, **orijinal ↔ bindirme kaydırıcısı** (ImageSlider), fare
tekerleğiyle 32×'e kadar yakınlaştırma / sürükleyerek kaydırma / çift tıkla sıfırlama (tarayıcıda doğrulandı), tam ekran ve
indirme düğmeleri, en-boy korumalı büyük paneller (480–680 px; önceki 260 px), SHAP'ta **katkı dağılımı** (histogram,
birikimli kütle, Σ+/Σ−/net, GT içi/dışı). **Yüksek çözünürlüklü dışa aktarım:** natif çözünürlükte ısı haritası/bindirme PNG,
float32 NPY, 300 dpi PNG + PDF yayın şekli (renk çubuğu, piksel eksenleri, GT ve hedef konturu).

**Korunanlar.** Ağırlık/mimari değişmedi, eğitim yok, `gpu_infer` / `_predict_*` / metrikler değişmedi; pencere hedefleri ve
"Ortak pencere" karşılaştırması aynen duruyor; kernel'de kırpma ve küçültme modları seçilebilir. Desteklenmeyen birleşim için
sahte sonuç yok: YOLO pencere hedeflerinde açık hata mesajı; ZeroGPU'da 120 s'ye sığmayan tam görüntü işi (ör. çok büyük
görüntüde 5 kanallı U-Net) uyarıyla atlanır, bütçe düşürülürse sonuç notuna yazılır.

## 3. Doğrulama

| Test | Sonuç |
|---|---|
| `tests/test_xai_filters.py` (T1–T7) | **87 kontrol / 0 hata** (11 Kasım 2026 koşusunda ek tohum ağırlıklarıyla 91 kontrol / 0 hata). T7: 576×864 (1 döşeme), 1500×1100 (4), 2300×700 (3) görüntülerde 5 mimari ailesinde döşemeli ↔ tek geçiş (CPU) göreli max\|Δ\| = 0 (YOLO ≤ 6e-8), çıktı boyutu = ⌈orijinal/adım⌉, f>1 iken = tek geçişin f×f ortalaması (Δ = 0), GPU 2300×1700: 6 döşeme, 0.6–0.8 s, tepe 280–390 MB |
| SHAP uçtan uca (`xai_predict` → `xai_run`, 8 hedef türü, U-Net + ResNet + YOLO) | tüm tam görüntü sonuçları "doğrulandı" (harita = 576×864, toplamsallık hatası ≤ 1.4e-15); mozaik "tam görüntü SHAP DEĞİLDİR" etiketli, kapsama tam; pencere hedefleri ve ③ karşılaştırma değişmeden çalışıyor; PDF/300 dpi/NPY dosyaları üretiliyor |
| Tarayıcı (yerel Gradio) | SHAP: tam görüntü panelleri, kaydırıcı, katkı dağılımı görüntülendi; tekerlek yakınlaştırma 1.56× → çift tık sıfırlama doğrulandı. Kernel: varsayılan "tam görüntü · natif, döşemeli", D bölümünde tam görüntü bindirmesi |
| Ölçüm (`tools/xai_fullimage_bench.py`, 0.28–4.48 MP, max_evals 500) | ayrıntı aşağıda ve `RAPOR_OLCUM.md` |

**Ölçüm özeti** (RTX 5080, max_evals 500, 4 görüntü: 0.28 / 1.12 / 2.15 MP gerçek + 4.48 MP (1.12 MP'nin 2× büyütülmüşü); tam tablo `RAPOR_OLCUM.md`):

- **Kapsama ve koordinatlar:** 26 SHAP sonucunun tamamında harita boyutu = orijinal görüntü (H×W), sonlu; tam görüntü
  sonuçlarında toplamsallık hatası ≤ 1.1e-4 (çoğu ~1e-16). Izgara hücresi 4–14 px (uzun kenar / 192).
- **Model çıktısıyla tutarlılık:** açıklanan f(x) ile uygulamanın tahmin tablosundaki değer arasındaki fark ≤ 0.0012
  (U-Net/DeepLabV3+ top-%1 skoru, ResNet maks patch, YOLO en güvenli kutu) — yalnız toplu / karışık duyarlıklı çıkarım
  gürültüsü. Önceki pencere hedefinde bu fark 0.33'e kadar çıkabiliyordu.
- **Süre (tam görüntü, 500 değerlendirme):** U-Net 3 kanal 18–53 s, DeepLabV3+ 10–26 s, ResNet-18 4–51 s, YOLO11 11–30 s
  (0.3 → 4.5 MP); U-Net 5 kanal (RGB+frekans) 29 s (0.28 MP), 106 s (1.12 MP), 264 s (2.15 MP) — frekans kanalları her
  değerlendirmede tüm görüntüde CPU'da yeniden hesaplanır. ZeroGPU'da süre tahmini 120 s'yi aşarsa bütçe düşürülür (nota
  yazılır) ya da iş uyarıyla atlanır.
- **GPU belleği:** tepe 2.3–3.5 GB — bunun ~1.9 GB'ı uygulamanın GPU'ya önceden yüklediği 34 modeldir; tam görüntü
  açıklamanın ek payı < 1.6 GB ve çözünürlükle büyümez (değerlendirmeler pencere/kare gruplarıyla yapılır).
- **Döşemeli mozaik (U-Net, 2–35 pencere):** kapsama her durumda tam; pencere tabanları arasındaki fark 0.0008–0.0017.
  Birleştirme ağırlıkları süreklidir (kosinüs rampası; bir pencerenin kendi kenarındaki ağırlığı 1.5e-4) → birleştirmeden
  doğabilecek sıçrama ≤ 1.5e-4 · 2·max|SHAP| (testing:000000038'de ≤ 3.5e-8). Haritada kalan sıçramalar (ör. y = 512'de
  ~1e-6) Partition'ın hiyerarşik blok sınırlarıdır ve tek bir pencerenin kendi haritasında da vardır (sınırın iki yanında
  6+ satır boyunca sabit değerler). Mozaiğin ΣSHAP'ı pencere açıklamalarının ağırlıklı birleşimidir, tek bir f(x)'e eşit değildir
  — bu yüzden "tam görüntü SHAP" olarak sunulmaz.
- **Kernel (tam görüntü, GPU):** 1–6 döşeme, 0.07–1.0 s, tepe 1.9–2.2 GB (önceden yüklü modeller dahil), saklanan 86–752 MB,
  kapsama her durumda tam. Döşemeli ↔ tek geçiş: CPU'da ve GPU'da TF32 kapalıyken **0** (928×2317 ve 1392×3476 görüntülerde,
  3 ve 8 döşeme; `checks.json`); TF32 açıkken (PyTorch varsayılanı) cuDNN döşeme boyutuna göre farklı algoritma seçtiğinde
  ≤ 2.7e-2 göreli fark — GPU ↔ CPU farkıyla aynı büyüklükte, döşemeden değil TF32 sayısal duyarlılığından.


## 4. Sınırlamalar

- Tam görüntü SHAP haritasının uzamsal çözünürlüğü ızgara hücresi (uzun kenar / 192 px) ve `max_evals` ile sınırlıdır;
  Partition hiyerarşisi düşük bütçede kaba bloklar bırakır. İnce doku için pencere hedefi + Gradient kullanılabilir.
- 5 kanallı U-Net'te frekans kanalları her değerlendirmede tüm görüntüde CPU'da yeniden hesaplanır → en yavaş model.
- Tam görüntüde Gradient desteklenmez (ızgara maskeli Partition); `inpaint` referansı yerine `blur(32,32)` kullanılır (not edilir).
- Kernel'de GPU'da cuDNN TF32 açıkken döşeme boyutuna göre farklı conv algoritmaları seçilebilir; bu, tek geçişe göre ~1e-2
  göreli farka yol açabilir (TF32 kapalıyken fark 0 — bkz. `checks.json`). CPU'da döşemeli = tek geçiş birebir.
- İstenen ekran görüntüleri göreve eklenmemişti; tasarım mevcut arayüz çıktıları üzerinden yapıldı.

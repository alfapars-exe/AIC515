# Dönem İçi Proje Raporu

**Ders:** AIC515 Görüntü İşleme — Prof. Dr. Onur Osman, Topkapı Üniversitesi
**Öğrenci / Grup:** ………………………… (en fazla 3 kişi)
**Tarih:** Kasım 2026 · **Sürüm:** v3 (9 Kasım 2026: simülasyon/toplu test arayüzü, Hugging Face ZeroGPU dağıtımı ve uygulama doğrulaması eklendi; önceki taslak: `Vize_Raporu_Taslak.md`)
**Canlı uygulama:** https://huggingface.co/spaces/infinayazilim/ai515 (ZeroGPU; veri seti özel Storage Bucket'ta)

## Özet — projeye beş soruda bakış

| Soru | Yanıt |
|---|---|
| **Hangi veri tabanı?** | **AIForge-Doc v1** (Scam.AI, Hugging Face `Scam-AI/AIForge-Doc-v1`, CC BY-NC-SA 4.0, e-posta kapılı): CORD v2, WildReceipt, SROIE ve XFUND kaynaklı **4.061 fiş/form görüntüsü**nde tek bir alan (fiyat, tarih, telefon, adres) **üretken yapay zekâ inpainting**'i (Gemini 2.5 Flash Image %89.6, Ideogram v2 Edit %10.4) ile değiştirilmiş; her sahte için 0/255 ikili maske ve eşleşen orijinal (authentic) görüntü var. 3.249 eğitim + 812 test. Yerel kopya 12.187 dosya, 9.98 GB. |
| **Amaç nedir?** | (i) Bir belge görüntüsünün sahte olup olmadığını **sınıflandırmak** (ResNet-18, klasik öznitelik + SVM/RF) ve (ii) değiştirilen pikselleri **segmente etmek** (U-Net); ders konularını (yoğunluk/histogram/eşikleme/frekans/morfoloji → öznitelik → CNN → U-Net) tek bir uygulamalı hat üzerinde kullanmak. |
| **Neyi bulmaya çalışıyoruz?** | Araştırma sorusu: *Kutu (bbox) maskesiyle eğitilen lokalizasyon modelleri gerçekte neyi öğreniyor?* Eşleşen orijinalden türetilen **piksel-değişim GT'si** ile bbox GT'si aynı tahminleri çok farklı puanlıyor mu (H1); bbox modeli dikdörtgen/sınır kısayolu mu öğreniyor (H2); piksel GT'si dayanıklılık/transfer avantajı sağlıyor mu (H3); alan dışında küresel bir iz var mı (H4)? Ek bulgu: veri setinde **yeşil işaretçi kalıntısı** adlı bir üretim artefaktı. |
| **Literatürde neler yapılmış?** | Veri seti makalesi yalnız **zero-shot** sonuç veriyor (TruFor img-AUC 0.751, IoU 0.358, F1 0.434; DocTamper 0.563/0.020; GPT-4o 0.509). Belge sahteciliği lokalizasyonunda DTD (CVPR'23), FFDN (ECCV'24), ADCD-Net (ICCV'25) JPEG/DCT ipuçlarına dayanır; DOCFORGE-BENCH (2026) 14 yöntemin 8 sette zero-shot çöktüğünü gösterir. v1'de **eğitilmiş** model yalnız hakemsiz GitHub projelerinde (Joya15 Tiny U-Net F1 0.322); v2'de eş zamanlı hakemsiz bir eğitimli dedektör + hat-artefaktı kontrolü (Helali, Kas 2026) var. |
| **Sonuçlar ve benchmark'lar?** | Küçük veri (400/100): ResNet-18 tam fine-tune AUROC **0.899** (probe 0.726, sıfırdan 0.679; RF 0.706, SVM 0.646); U-Net RGB Pixel-F1 **0.707**, IoU 0.637. Tam veri, resmî test (812), 3 tohum: U-Net bbox-GT **AUPRC 0.950 ± 0.011, F1@τ\* 0.846 ± 0.015, IoU 0.789** — zero-shot TruFor F1 0.403 / ADCD-Net 0.182'nin çok üstünde (eşit koşul değil). Ancak aynı tahminler piksel-değişim GT'siyle **F1 0.45, AUPRC 0.37**'ye düşer (H1 doğrulandı); yeşil işaretçi tek başına kutuyu tetikler (authentic + işaretçi AP 0.906). |
| **Uygulama ve doğrulama?** | Gradio arayüzü (Hugging Face Space, ZeroGPU): veri setinden bir örnek seçilip **simülasyon** olarak adım adım test edilir (GT analizi → işaretçi taraması → senaryo/bozulma → ResNet-18 + U-Net → iki GT ile metrik → görüntü-düzeyi karar → indirilebilir rapor); **toplu test** N örnekte aynı hattı koşturur. Canlı Space'te 24 sahte + 24 authentic test çiftinde bbox-GT U-Net **F1 0.925, AP 0.964**, görüntü AUROC 0.988; boru hattının kayıtlı sonuçlarıyla fark **|ΔF1| = 0.000** (birebir yeniden üretim). Yerel otomatik test: **281 kontrol, 0 hata**; tarayıcı (UI) testi hatasız. |

## 1. Proje Adı

**Yapay Zekâ ile Üretilmiş Finansal Belge Sahteciliğinin Tespiti ve Manipüle Edilmiş Bölgelerin Segmentasyonu: AIForge-Doc Veri Setinde Klasik Görüntü İşleme, ResNet ve U-Net Karşılaştırması**

İngilizce çalışma başlığı (bildiri): *What Do Document Forgery Localizers Learn from Box Masks? Pixel-Change Ground Truth and Shortcut Audits for AI-Inpainted Financial Documents.*

## 2. Literatür

### 2.1 Problem
Üretken görüntü servisleri (Gemini 2.5 Flash Image, Ideogram v2 Edit, GPT-Image-2) bir fiş ya da formdaki tek bir alanı inpainting ile, gözle ayırt edilemeyecek biçimde değiştirebilmektedir. Problem iki düzeydedir: **görüntü-düzeyi sahtecilik tespiti** (sınıflandırma) ve **kurcalama lokalizasyonu** (ikili semantik segmentasyon; dersin 13. hafta U-Net konusu). Kurcalanan alan görüntünün medyan **%0.98**'idir; bu, küçük-ön-plan segmentasyonu problemidir ve sabit eşikli F1 ile AUPRC'nin birlikte raporlanmasını gerektirir.

### 2.2 Bu veri setiyle yapılmış çalışmalar
- **Wu ve ark. (2026a, arXiv:2602.20569) — veri seti makalesi.** 4.061 sahte; mevcut dedektörleri zero-shot değerlendirir: TruFor img-AUC 0.751 [0.726, 0.776], pixel-IoU 0.358, F1 0.434; DocTamper/DTD AUC 0.563, IoU 0.020; GPT-4o AUC 0.509 (Tablo 2). Makale §3.4, Gemini için bağlam kırpıntısına **3 px parlak yeşil dikdörtgen** çizildiğini; §3.3/§4.2, GT maskesinin **alanın tam bbox'ı** olduğunu belirtir. Bu iki tasarım kararı projemizin iki ana bulgusunun kaynağıdır.
- **Wu ve ark. (2026b, arXiv:2604.25213) — AIForge-Doc v2.** GPT-Image-2 ile 3.066 sahte; insan 2AFC 0.501, TruFor AUC 0.599, DocTamper 0.585; aynı dedektörler geleneksel splicing setlerinde 0.962 / 0.852 → AI-inpainting'e özgü 0.27–0.36 AUC açığı.
- **Raj ve ark. (2026, arXiv:2609.13617) — v3.** GPT-Image-2.5; lokalizasyonun zayıf kaldığını bildirir (yapıştırma-dikişi kısayolu ayrıntıları bu raporda doğrulanamadığından nitel olarak anılmaktadır).
- **Eğitilmiş modeller (hakemsiz):** Joya15 (GitHub) v1+v2+GPT4o-Receipt birleşiminde **Tiny U-Net** ile piksel F1 0.322 / IoU 0.192 ve SegFormer-lite ile çapraz-veri F1 0.044 (v2→v1); leongyiquan/DocDet karo-sınıflandırıcı ResNet-18 (sayfa AUC 0.80, lokalizasyon yok); **Helali (2026, Zenodo 10.5281/zenodo.23917285, 3 Kasım)** v2'de 371k parametreli FCN eğitmiş (img-AUC 0.989) ve "process-twin" kontrolüyle (içerik değişmeden hattan geçen orijinaller) twin'lerin %37.7'sinin sahte işaretlendiğini göstermiştir — üretim-hattı artefaktı kısayolunun bağımsız kanıtı.
- **Boşluk (savunulabilir biçimiyle):** v1 üzerinde eğitim maskesinin geometrisini kontrollü değiştiren, piksel-değişim/glif GT'si türeten ve §3.4'teki işaretçinin kalıntısını ölçen bir çalışma yoktur; v2'de görüntü-düzeyi hat-artefaktı kontrolü yapan eş zamanlı hakemsiz bir çalışma vardır. "AIForge-Doc'ta ilk denetimli eğitim" iddiası **kullanılmamaktadır**.

### 2.3 Belge sahteciliği lokalizasyonunda yöntemler ve benchmark'lar
| Yöntem (mekân) | Ana fikir | DocTamper-Test | FCD | SCD | AIForge-Doc v1 (zero-shot) |
|---|---|---|---|---|---|
| DTD / DocTamper (CVPR 2023) | RGB + 8×8 blok DCT + JPEG kuantizasyon tablosu, Swin | F1 0.792 / IoU 0.828 | 0.816 | 0.754 | F1 0.030 / IoU 0.020 / AUC 0.563 |
| FFDN (ECCV 2024) | Sıfır-başlatılmış frekans dalı + Haar dalgacık | 0.86 / 0.90 (JPEG Q75: 0.75) | 0.92 | 0.81 | — |
| ADCD-Net (ICCV 2025) | DCT hizalama skoruyla adaptif ağırlık; 1 px kaydırmada DTD'nin F1'i −%73 | 0.806 | 0.815 | 0.740 (ort. 0.787; SOTA'yı ort. %20.8 aşar) | bu proje: F1 0.182 / IoU 0.149 |
| TruFor (CVPR 2023) | Noiseprint++ gürültü artığı + RGB | 0.168 (resmî ckpt) | 0.109 | 0.281 | **F1 0.434 / IoU 0.358 / AUC 0.751** (makale); bu proje 0.403 / 0.328 |
| CAT-Net / MVSS-Net | DCT-JPEG / gürültü + Sobel kenar | 0.700 / 0.431 | 0.553 / 0.424 | 0.631 / 0.414 | — |
| DOCFORGE-BENCH (2026) | 14 yöntem × 8 set zero-shot | ort. F1: TruFor 0.34, CAT-Net 0.33, ADCD 0.23, FFDN 0.17, DTD 0.16, IML-ViT 0.08 | | | — |

Diğer ilgili çalışmalar: TGIF (WIFS 2024) ve TGIF2 (2026) — difüzyon inpainting'de görüntü tamamen yeniden üretildiğinde F1 ≤ 0.1; INP-X (2026) — inpaint dışı pikseller geri konunca ticari dedektör doğruluğu %91 → %55 (küresel hat artefaktı kısayolu); DiffusionPrint (2026); Pandolfini ve ark. (2025) — <%20 alanlı düzenlemelerde tam-görüntü dedektörlerinin çöküşü; TextShield-R1 (AAAI 2026), DocShield (2026), GenText-Forensics Challenge (ACM MM 2026). **Kısayol öğrenme** çerçevesi: Geirhos ve ark. (2020); GenD (WACV 2026) eşleşmiş gerçek–sahte eğitimin çapraz-veri AUROC'u için gerekli olduğunu; DocQT (2026) yalnız JPEG kuantizasyon tablosu farkının FFDN'yi etkilediğini raporlar. Tam matris: `literature/literature_matrix.xlsx` (38 çalışma + 2026-11 eklemeleri).

### 2.4 Projede kullanılan modellerin kaynakları
| Model / teknik | Sunulduğu çalışma |
|---|---|
| ResNet-18 | He, Zhang, Ren, Sun, CVPR 2016 |
| U-Net | Ronneberger, Fischer, Brox, MICCAI 2015 |
| ImageNet ön-eğitim / transfer | Deng ve ark., CVPR 2009; Yosinski ve ark., NeurIPS 2014 |
| LBP / GLCM / 2-B DCT | Ojala ve ark., TPAMI 2002 / Haralick ve ark., TSMC 1973 / Ahmed ve ark., IEEE TC 1974 |
| SVM / Random Forest | Cortes & Vapnik 1995 / Breiman 2001 |
| Dice kaybı | Milletari ve ark., 3DV 2016 |
| Frekans-domeni filtreleme, morfoloji, Otsu, Sauvola | Gonzalez & Woods, *Digital Image Processing* 3. baskı (Böl. 3–4, 9–10); Sauvola & Pietikäinen 2000 |

### 2.5 Veri seti seçiminin gerekçesi
AIForge-Doc v1 dersin örnek listesinde (Oxford Pets, ISIC 2018, Kvasir-SEG, MVTec AD…) yer almamakla birlikte aynı şablona oturur: HF kartı görevleri `image-classification` + `image-segmentation` olarak etiketler; 4.061 görüntü 0/255 ikili maskelidir; Kvasir-SEG/ISIC gibi "küçük ön plan, ikili maske" yapısına ve MVTec benzeri anomali-lokalizasyonu karakterine sahiptir; yayımlanmış zero-shot referanslar karşılaştırma tabanı sağlar; lisans akademik kullanıma açıktır (ham veri yeniden dağıtılmaz, yalnız türetilmiş maske/kod paylaşılır). Finansal belge sahteciliği, uygulama alanı olarak da güncel ve ekonomik değeri yüksek bir problemdir.

## 3. Materyal ve Metot

### 3.1 Veri tabanı: AIForge-Doc v1 (yerel audit ile doğrulandı)
- **İçerik:** 3.249 eğitim + 812 test sahte (RGB PNG) ve ikili maske; eğitim tarafında 3.247/3.249 sahtenin eşleşen authentic'i doğrulandı. Kaynak: WildReceipt 1.696, CORD 1.000, SROIE 946, XFUND 419; fiş %89.7, form %10.3; 9 dil (EN 2.642, ID 1.000); Gemini 3.639 (%89.6), Ideogram 422 (%10.4).
- **Manipülasyon:** tek alanın değeri değiştirilir; bbox %50 genişletilmiş bağlam penceresi inpaint edilir, yalnız alan bbox'ı geri yapıştırılır; GT maskesi = alan bbox'ı (tüm maskeler tek dikdörtgen).
- **İstatistikler:** çözünürlük 150×168 – 6016×7016 px (medyan 750×1296); kurcalanmış alan / görüntü alanı medyan **%0.98** (%50.7'si < %1); maske yüksekliği medyan 33 px → görüntü küçültülmeden **natif çözünürlükte crop** zorunlu (uzun kenar > 4096 px olan ~%5 görüntü çıkarımda 4096'ya küçültülür); piksel dengesizliği ≈ 1:100.
- **Audit bulguları:** (i) maskeler bbox düzeyinde; eşleşen authentic ile fark alındığında bbox piksellerinin medyan %49'u gerçekten değişir (ε=8); (ii) çiftlerin %33.5'inde bbox dışında da değişim vardır; (iii) `TestingSet/authentic` test sahtelerinin aynası değildir → image-level test metriği verilemez; (iv) test sahtelerinin %8.9'u eğitimdeki bir kaynak belgeyi paylaşır → "temiz test" alt kümesi (740) ayrıca raporlanır; (v) **sahtelerin %75.4'ünde bbox iç kenarında 1–3 px yeşil işaretçi kalıntısı** (Gemini %83.6, Ideogram %1.5; 600 authentic'te 0).
- **Bölme (test setine dokunulmadı):** kaynak-belge gruplarına göre train 2.762 / val 487 (15 kusurlu çift dışlanınca 2.748); dönem içi deneyler için smoke_train 400 / smoke_val 100 (alan dilimine tabakalı, kaynak×generator dengeli) — `dataset_audit/splits.json`.

![Örnek görüntü–maske çiftleri](dataset_audit/fig_examples_overlay.png)

![Kurcalanmış alan oranı dağılımı](dataset_audit/fig_tamper_ratio.png)

### 3.2 Genel işlem hattı
1. **Natif çözünürlükte crop:** sahte görüntüden maske-bbox merkezli 256×256 (sınıflandırma) / 512×512 (segmentasyon) crop; aynı görüntüden maskeyle kesişmeyen temiz crop; eşleşen authentic'ten aynı koordinatlı temiz crop (içerik-kontrollü negatif).
2. **Klasik öznitelikler (56) + SVM-RBF / Random Forest.**
3. **ResNet-18:** (a) dondurulmuş gövde + yeni FC (linear probe), (b) tam fine-tune, (c) sıfırdan.
4. **U-Net (ResNet-18 kodlayıcı):** D1 RGB; D2 RGB + FFT yüksek-geçiren + Laplacian artığı (5 kanal). PHASE 2'de aynı mimari üç farklı GT (bbox / piksel-değişim / glif) ile 3 tohum.
5. **Değerlendirme:** sınıflandırmada Acc, Bal.Acc, AUROC, F1; segmentasyonda Pixel-F1@τ, IoU, Dice, AUPRC, AUROC, Oracle-F1; generator / kaynak / alan-dilimi kırılımı; JPEG Q75, 0.75× yeniden ölçekleme, Gauss blur dayanıklılık probları; benign kontroller; kaynak-grubu bootstrap %95 GA ve Holm düzeltmesi.

### 3.3 Ön işleme ve gerekçesi
Adli sinyal düşük seviyeli istatistiklerdedir (gürültü artığı, yeniden-örnekleme/sıkıştırma izi, glif kenarı). Bu yüzden görüntü *iyileştirme* amaçlı ön işleme yapılmamış; yalnız izi koruyan, sahte/authentic için simetrik adımlar uygulanmıştır: kayıpsız PNG okuma (BGR→RGB), yeniden boyutlandırma olmadan natif crop (küçük görüntülerde kenar kopyası dolgu), ImageNet ortalama/std normalizasyonu, eğitimde ±%8 parlaklık/kontrast jitter'ı, **geometrik artırım yok** (metin yönü), çıkarımda 512 px pencere / 448 px adım / örtüşme ortalaması. Klasik hatta gri dönüşüm + LBP/GLCM/DCT/Laplacian/FFT; D2'de frekans kanalları. Morfoloji ve Sauvola yalnız GT türetiminde kullanıldı. Yapılmayanların (küçültme, gürültü giderme, JPEG) etkisi test-zamanı bozulma probuyla ölçüldü (Tablo E ve §4.3).

Ders konularının tamamı (yoğunluk dönüşümü, histogram, eşikleme, gürültü giderme, keskinleştirme, kenar, morfoloji, geometrik dönüşüm, normalizasyon, frekans domeni, K-Means/watershed) bu raporla birlikte teslim edilen **aşamalı Gradio arayüzünde** (§3.8) uygulanabilir ve her adımın model çıktısına etkisi ölçülebilir.

### 3.4 Klasik öznitelikler (Hafta 3–4, 9–11)
**8×8 blok DCT:** $F(u,v)=\tfrac14 C(u)C(v)\sum_{x=0}^{7}\sum_{y=0}^{7} f(x,y)\cos\!\big[\tfrac{(2x+1)u\pi}{16}\big]\cos\!\big[\tfrac{(2y+1)v\pi}{16}\big]$, $C(0)=1/\sqrt2$, $C(k>0)=1$; ilk 15 zigzag AC katsayısının blok-ortalama/std'si (30), DC std (1), düşük/orta/yüksek bant enerji oranları (3).
**LBP** (P=8, R=1, uniform): $LBP=\sum_{p} s(g_p-g_c)2^p$; 10 kutulu histogram.
**GLCM** (32 seviye, d=1, 4 yön ort.): kontrast, homojenlik, enerji, korelasyon.
**Laplacian artığı** $r=f*\nabla^2$ (3×3): mean|r|, std, basıklık, çarpıklık.
**FFT radyal spektrum:** düşük/orta/yüksek bant enerji oranları + log-spektrum eğimi.
**SVM-RBF:** $\min \tfrac12\|w\|^2 + C\sum\xi_i$, $K(x,x')=\exp(-\gamma\|x-x'\|^2)$. **RF:** 500 ağaç, Gini önemi.

```
for each patch P (gri):
    f_dct <- block_dct_stats(P, 8, n_ac=15);  f_lbp <- hist(LBP(P,8,1,'uniform'),10)
    f_glcm <- mean_angles([contrast, homogeneity, energy, correlation])
    f_hp <- stats(P * Laplacian3x3);  f_fft <- radial_band_ratios(|FFT2(P)|)
    x <- concat(...)                                   # 56 öznitelik
X <- StandardScaler.fit_transform(X_train); clf <- SVM_RBF | RandomForest(500)
```

### 3.5 ResNet-18 ile patch sınıflandırma (Hafta 12)
Residual blok $y=\mathcal F(x,\{W_i\})+x$; 4 aşama × 2 temel blok, GAP, FC(512→1). Kayıp: $\mathcal L_{BCE}=-[y\log\hat p+(1-y)\log(1-\hat p)]$; AdamW, cosine LR, 12 epoch, batch 32, bf16; giriş 256 natif patch'in 224 kesiti (yeniden örnekleme yok); yatay çevirme yok.
```
model <- resnet18(IMAGENET1K_V1); model.fc <- Linear(512,1)
if probe: freeze all but fc
for epoch: loss <- BCEWithLogits(model(x), y); step;  val_auroc -> best checkpoint
```

### 3.6 U-Net ile piksel segmentasyonu (Hafta 13)
Kodlayıcı ResNet-18 (ImageNet), 4 kademe kod çözücü, atlama bağlantıları $d_l=\mathrm{Up}(d_{l+1})\oplus e_l$; çıkış piksel logit'i. Kayıp $\mathcal L=\mathcal L_{BCE}+\mathcal L_{Dice}$, $\mathcal L_{Dice}=1-\frac{2\sum\hat p_i y_i+\epsilon}{\sum\hat p_i+\sum y_i+\epsilon}$. D2'de giriş $[R,G,B,g^{FFT}_{HP},r_{Lap}]$; $H_{HP}(u,v)=1-e^{-D^2/(2D_0^2)}$, $D_0=20$. Eğitim: 512 natif crop (%70 bbox içeren), AdamW 1e-4, PHASE 1: 15 epoch/batch 8; PHASE 2: 10 epoch/batch 16, 3 tohum.
```
for epoch: z <- UNet(x); loss <- BCE(z,m) + Dice(σ(z),m); step
val: F1@τ, IoU, AUPRC, Oracle-F1;  τ* <- val ızgarası (0.05..0.95) → test'e taşınır
son-işleme (Hafta 6): eşik / Otsu + morfolojik açma-kapama + en büyük bileşen bbox'ı
```
**GT türetimi (PHASE 2):** $D=\max_c|f-a|$; piksel GT = (D>8) → 3×3 açma → <8 px bileşen filtresi → 3×3 kapama; glif GT = [Sauvola(f) ∪ Sauvola(a)] ∩ dilate(D>8, 3 px), polarite azınlık sınıfı. ε-duyarlılık: bbox-dışı pikselleri > %1 değişen çift oranı ε=0'da %47.8 → ε=8'de %1.3.

### 3.7 Değerlendirme ölçütleri
$P=\frac{TP}{TP+FP}$, $R=\frac{TP}{TP+FN}$, $F_1=\frac{2PR}{P+R}$, $IoU=\frac{TP}{TP+FP+FN}$; AUROC/AUPRC eşikten bağımsız; Oracle-F1 görüntü başına en iyi eşik. PHASE 1 AUPRC'si crop başına ≤20k negatif alt-örneklemesiyle hesaplandığından **şişkindir** (pozitif önseli ≈%4.7 → ≈%38) ve PHASE 2'nin 256-bin histogramla kesin hesaplanan AUPRC'siyle karşılaştırılamaz; AUROC ve F1 bu etkiden bağımsızdır. İstatistik: kaynak-grubu bootstrap (B=1000) %95 GA, tohumlar sabit faktör, Holm düzeltmesi (58 ön-kayıtlı test).

### 3.8 Uygulama: simülasyon + aşamalı Gradio arayüzü (app/app.py)
Ders konularını ve projenin modellerini tek arayüzde, **adım adım** çalıştırmak için geliştirildi. Aynı dosya yerelde (`python app/app.py`) ve Hugging Face Space'te (https://huggingface.co/spaces/infinayazilim/ai515) çalışır. Sekmeler: **0 · Simülasyon**, **1 · Toplu test**, 2 · Girdi, 3 · Ön işleme, 4 · U-Net, 5 · ResNet-18, 6 · Yeşil işaretçi, 7 · Sonuçlar ve model kartı.

**3.8.1 Simülasyon (tek örnek).** Kullanıcı veri setinden bir örnek seçer: bölüm (train/test), kaynak veri seti (CORD, WildReceipt, SROIE, XFUND), inpainting üreticisi (Gemini/Ideogram), değiştirilen alan oranı (<%1, %1–5, >%5), yeşil işaretçi kalıntısı (var/yok) ve en fazla megapiksel filtreleriyle listeyi daraltır; **Rastgele seç** ya da kimlikle (`testing:000000700` gibi) seçim yapılır. Ardından *girdi senaryosu* ve *test-zamanı bozulması* seçilir:

| Senaryo | Modelin gördüğü girdi | Beklenen doğru karar | Amaç |
|---|---|---|---|
| Sahte belge (forged) | veri setindeki sahte görüntü | SAHTE | standart lokalizasyon |
| Authentic eş (negatif kontrol) | eşleşen orijinal görüntü | GERÇEK | yanlış alarm ölçümü |
| Sahte, yeşil işaretçi onarılmış | sahte; işaretçi pikselleri authentic'ten kopyalanmış | SAHTE | modelin işaretçiye bağımlılığı |
| Authentic + yalnız yeşil işaretçi | authentic; yalnız işaretçi pikselleri sahteden aktarılmış | GERÇEK | kısayol testi (PHASE 2 Ek-C) |

Bozulmalar: JPEG Q75/Q50, 0.75×/0.5× yeniden ölçekleme, Gauss bulanıklık (σ=1), Gauss gürültüsü (σ=5); istenirse Ön işleme sekmesindeki pipeline da uygulanır. Simülasyon 7 adımı canlı günlükle gösterir: (1) veri yükleme, (2) GT analizi (bbox piksellerinin ne kadarı gerçekten değişmiş, ε=8), (3) yeşil işaretçi taraması, (4) model girdisinin hazırlanması, (5) çıkarım, (6) değerlendirme, (7) rapor. Çıktılar: adım adım görüntü galerisi (sahte, bbox GT, authentic, |sahte−authentic| farkı, piksel-değişim GT, model olasılık haritaları ve ResNet patch ısı haritası), alan kutusu çevresinde **yakın plan paneli**, model başına sonuç tablosu, görüntü-düzeyi **karar ve doğru/yanlış işareti**, otomatik yorum (dikdörtgensellik, H1 farkı, işaretçi uyarısı, boru hattı kayıtlı sonucuyla fark) ve indirilebilir MD/JSON/CSV/PNG raporu. **Bu örneği ön işleme / model sekmelerine aktar** düğmesi aynı örneği ders sekmelerine taşır.

**3.8.2 Toplu test.** Aynı filtrelerle N rastgele örnek (tohumlu) seçilip iki modda koşturulur: *yalnız sahte* (lokalizasyon) ya da *sahte + authentic eşi* (dengeli tespit). Model başına ortalama F1/IoU/AP (bbox-GT ve piksel-GT), tespit TPR, yanlış alarm FPR, doğruluk, görüntü AUROC ve **boru hattının kayıtlı sonuçlarıyla yeniden üretim farkı** (|ΔF1@0.5|, |ΔAP|) raporlanır; özet grafik ve CSV/JSON indirilebilir.

**3.8.3 Ölçüt tanımları (boru hattıyla birebir).** Olasılık haritası 8-bit'e nicemlenir (kayıtlı PNG tahminleriyle aynı); AP 256 kutulu histogramla **tüm pikseller** üzerinden kesin hesaplanır (alt-örnekleme yok); F1@τ ikili maske u8 ≥ ⌈255·τ⌉; τ değerleri `tau.json`'dan (val'de seçilmiş τ\*), Oracle-F1 görüntü başına en iyi eşik. Görüntü-düzeyi skor = olasılık haritasının en yüksek %1 piksel ortalaması; karar eşiği val bölümündeki 487 sahte + 487 eşleşen authentic üzerinde Youden-J ile seçildi (`meta/image_tau.json`): bbox-GT s42 τ=0.167 (val TPR 0.879, FPR 0.150, AUROC 0.921), piksel-GT s42 τ=0.084 (0.862 / 0.209 / 0.885), glif-GT s42 τ=0.067 (0.823 / 0.314 / 0.796); ResNet-18 için görüntü skoru = en yüksek patch skoru, eşik 0.5. Çıkarım `infer_fullimage.py` protokolüyle aynıdır: 512 px pencere, 448 adım, örtüşme ortalaması; >4096 px kenarlar önce küçültülür.

**3.8.4 Dağıtım mimarisi.** Space, Gradio 6 ve **ZeroGPU** (dinamik GPU; NVIDIA RTX Pro 6000 Blackwell, ücretsiz kota) üzerinde çalışır: `spaces` paketi torch'tan önce yüklenir, modeller açılışta GPU'ya taşınır, çıkarım `@spaces.GPU` ile işaretli fonksiyonda (süre örnek boyutuna göre dinamik) yapılır; yerelde dekoratör etkisizdir. Veri seti lisansı (CC BY-NC-SA 4.0, kapılı) nedeniyle ham veri Space reposuna konmadı: 12.208 dosya (10.26 GiB; görüntüler, maskeler, modeller, audit tabloları) **özel** bir Hugging Face Storage Bucket'a (`infinayazilim/ai515-data`) yüklendi ve Space'e salt-okunur `/data` olarak bağlandı (`AI515_DATA_ROOT`). Space reposunda yalnız kod, sonuç tabloları ve küçük meta dosyaları vardır. Dağıtım betiği: `hf_space/deploy_hf.py`; canlı test: `hf_space/live_test.py`.

![Simülasyon sekmesi — testing:000000700 (WildReceipt, Gemini, alan <%1): 7 adım, karar, metrikler, yakın plan ve rapor dosyaları](app/screenshot_simulasyon.png)

![Simülasyon — "Authentic + yalnız yeşil işaretçi" kısayol testi: değiştirilmemiş belgede modeller kutuyu çizip SAHTE der](app/screenshot_kisayol.png)

![Toplu test sekmesi — dengeli tespit modu, model bazında özet, grafik ve örnek bazında sonuçlar](app/screenshot_toplu.png)

**3.8.5 Ders sekmeleri (2–7).**
- **Girdi:** sahte görüntü + isteğe bağlı GT maskesi + authentic eş yükleme ya da veri setinden örnek seçme (metadata ve authentic eş otomatik).
- **Ön işleme (13 kategori / 61 işlem):** renk işleme; yoğunluk dönüşümleri (negatif, log, gamma, kontrast germe, dilimleme, bit düzlemi); histogram (eşitleme, CLAHE); eşikleme (global, Otsu, adaptif, Sauvola); gürültü ekleme/giderme (box, medyan, Gauss, bilateral, NLM); keskinleştirme (Laplacian, unsharp/high-boost); kenar (Sobel, Prewitt, Scharr, Laplacian, LoG, Canny); morfoloji (erozyon, genişleme, açma, kapama, gradyan, top/black-hat, bileşen filtresi); geometrik (döndürme, ölçekleme, öteleme, aynalama, shear, kırpma); normalizasyon (min-max, z-score, ImageNet, boyut); frekans domeni (FFT spektrumu, ideal/Gauss/Butterworth LP-HP, projenin D2 kanalları, blok DCT); klasik segmentasyon (K-Means, watershed, bağlı bileşen); dayanıklılık bozulmaları (JPEG, yeniden ölçekleme, blur, işaretçi onarımı). Her adım **Önizle → Pipeline'a ekle → Geri al** ile yönetilir, önce/sonra görüntüsü ve histogramı gösterilir; hazır presetler ve 56 klasik öznitelik tablosu vardır.
- **Derin öğrenme:** U-Net (PHASE 1 ve PHASE 2 tüm checkpoint'leri; `infer_fullimage.py` ile aynı kayan-pencere protokolü; eşik `tau.json`'dan) ve ResNet-18 (224 natif kayan pencere); model **orijinal** ya da **pipeline çıktısı** üzerinde çalıştırılıp ölçütler yan yana karşılaştırılır; GT olarak bbox maskesi ya da authentic'ten türetilen piksel-değişim maskesi seçilebilir (H1 canlı gösterim); yeşil işaretçi sondası.

![Arayüz — aşamalı ön işleme (belge ikilileştirme preseti)](app/screenshot_pipeline.png)

![Arayüz — U-Net lokalizasyonu (örnek 000000329, bbox modeli; F1 0.984 bbox-GT)](app/screenshot_unet.png)

### 3.9 Donanım ve yazılım
RTX 5080 (16 GB, CUDA 12.8), Python 3.11, PyTorch 2.11 (cu128), torchvision 0.26, segmentation-models-pytorch 0.5, scikit-learn 1.9, scikit-image, OpenCV, Gradio. Kurulum: `baselines/ENV_SETUP.md`; uçtan uca yeniden üretim: `run_everything.ps1` (~6 saat); arayüz: `app/README_APP.md`. Canlı uygulama: Hugging Face Space (Python 3.12, PyTorch 2.11, Gradio 6, ZeroGPU), veri ve modeller özel Storage Bucket'tan.

## 4. Sonuçlar

### 4.1 Çok kısıtlı veri (smoke_train 400 / smoke_val 100, tek tohum) — sistemin çalıştığının gösterimi
Patch kümesi 1.587 eğitim / 296 val (tampered / clean-same / clean-paired); toplam GPU eğitim süresi 23.5 dk; test setine dokunulmadı. Değerler provizyoneldir.

**B) Klasik öznitelikler + ML**
| Model | Acc | Bal.Acc | AUROC | AUPRC | F1@0.5 | AUROC tamp-vs-paired | AUROC tamp-vs-same |
|---|---|---|---|---|---|---|---|
| SVM-RBF (C=0.5, γ=0.01) | 0.534 | 0.606 | 0.646 | 0.443 | 0.546 | 0.552 | 0.744 |
| Random Forest (500 ağaç) | 0.622 | 0.653 | 0.706 | 0.536 | 0.573 | 0.610 | 0.807 |

**C) ResNet-18**
| Ayar | best ep | Acc | Bal.Acc | AUROC | AUPRC | F1@0.5 | AUROC tamp-vs-paired | TN/FP/FN/TP |
|---|---|---|---|---|---|---|---|---|
| Linear probe (dondurulmuş gövde + yeni FC) | 10/12 | 0.639 | 0.693 | 0.726 | 0.554 | 0.616 | 0.635 | 103/93/14/86 |
| **Tam fine-tune (pretrained)** | 6/11 | **0.885** | **0.845** | **0.899** | **0.884** | **0.809** | **0.878** | 190/6/28/72 |
| Sıfırdan | 4/9 | 0.503 | 0.610 | 0.679 | 0.497 | 0.561 | 0.573 | 55/141/6/94 |

**D) U-Net (512 natif crop; val: görüntü başına 1 bbox + 1 rastgele crop)**
| Varyant | best ep | Pixel-F1@0.5 | IoU@0.5 | Oracle-F1 | Pixel-AUPRC\* | Pixel-AUROC | Temiz crop FP | F1 gemini / ideogram | F1 <1% / 1–5% |
|---|---|---|---|---|---|---|---|---|---|
| **U-Net RGB (D1)** | 11/15 | **0.707** | **0.637** | 0.740 | 0.947 | 0.948 | 0.26 % | 0.800 / 0.497 | 0.606 / 0.830 |
| U-Net RGB+FFT-HP+Laplacian (D2) | 12/15 | 0.674 | 0.613 | 0.711 | 0.939 | 0.940 | 0.14 % | 0.779 / 0.449 | 0.589 / 0.792 |

\* alt-örneklenmiş AUPRC (bkz. §3.7); PHASE 2 ile karşılaştırılamaz.

**E) Dayanıklılık (eğitim yapılmadan, aynı val crop'ları)**
| Varyant | Koşul | Pixel-F1@0.5 | IoU | Oracle-F1 | Temiz crop FP |
|---|---|---|---|---|---|
| U-Net RGB | temiz PNG | 0.707 | 0.637 | 0.740 | 0.26 % |
| U-Net RGB | JPEG Q75 | 0.636 | 0.565 | 0.680 | 0.18 % |
| U-Net RGB | 0.75× → 1× | 0.638 | 0.560 | 0.676 | 1.24 % |
| U-Net D2 | JPEG Q75 / 0.75× | 0.604 / 0.633 | 0.540 / 0.566 | 0.640 / 0.674 | 0.13 % / 0.42 % |

**F) Kaynak:** ResNet-18 eğitimi 64–76 s (peak 0.33–0.59 GiB, 11.18 M parametre, çıkarım 7–8 ms/patch); U-Net ~10 dk (peak 1.7 GiB, 14.33 M, 16–19 ms/512-crop); SVM/RF 6–7 s CPU.

![U-Net tahminleri: forged crop / GT (bbox) / RGB / RGB+frekans](baselines/figures/unet_predictions.png)

**Okuma.** (1) Ön-eğitilmiş + yeni katman vs sıfırdan: tam fine-tune (0.899) ≫ probe (0.726) ≫ sıfırdan (0.679) — pretrained model yalnız gövde de güncellendiğinde kazandırıyor. (2) İçerik-kontrollü negatif (aynı koordinatlı authentic) her modelde daha zor → modeller kısmen içerik/konum ipucu kullanıyor. (3) Klasik öznitelikler şans üstü ama zayıf; veri PNG olduğundan DCT öznitelikleri ayırt edici değil (beklenen negatif bulgu). (4) U-Net RGB küçük veriyle güçlü; tahminler **bbox dikdörtgenini bütünüyle** işaretliyor → PHASE 2'nin motivasyonu. (5) Frekans kanalı eklemek RGB'yi geçmedi. (6) JPEG/yeniden ölçekleme F1'i ~7 puan düşürüyor, çöküş yok.

### 4.2 Tam veri (train 2.748 / val 487 / resmî test 812; 3 tohum)
| Model | Alt küme | AUPRC | F1@τ\* | IoU@τ\* | Oracle-F1 |
|---|---|---|---|---|---|
| **U-Net bbox-GT (3 tohum)** | tam 812 | **0.950 ± 0.011** | **0.846 ± 0.015** | **0.789 ± 0.018** | 0.879 ± 0.011 |
| U-Net bbox-GT | temiz 740 | 0.949 ± 0.012 | 0.845 ± 0.015 | 0.788 ± 0.018 | 0.878 ± 0.012 |
| U-Net piksel-GT | tam 812 | 0.638 ± 0.015 | 0.616 ± 0.013 | 0.480 ± 0.015 | 0.624 ± 0.015 |
| U-Net glif-GT | tam 812 | 0.403 ± 0.011 | 0.405 ± 0.007 | 0.266 ± 0.006 | 0.410 ± 0.008 |
| TruFor zero-shot (≤2 MP) | tam 812 | havuz 0.080 | F1@0.5 0.403 | 0.328 | 0.456 |
| ADCD-Net zero-shot (Q100) | tam 812 | havuz 0.130 | F1@0.5 0.182 | 0.149 | 0.328 |

Sızıntı etkisi yok (tam vs temiz ΔAP 0.001). TruFor değerleri makaledekine (0.434 / 0.358) yakın (−0.03). Zero-shot satırları eşit koşul değildir; referans olarak verilmiştir.

**H1 — aynı tahminler, üç GT (val 487, AUPRC, tohum ort.):**
| Eğitim GT \ Değerlendirme GT | bbox | piksel | glif |
|---|---|---|---|
| bbox | **0.923 ± 0.015** | 0.374 ± 0.008 | 0.257 ± 0.007 |
| piksel (eps8_morph) | 0.579 ± 0.014 | **0.610 ± 0.012** | 0.511 ± 0.010 |
| glif (glyph_v2) | 0.351 ± 0.015 | 0.379 ± 0.012 | **0.676 ± 0.019** |

Her GT kendi modelini birinci sıraya koyar; bbox modelinin ΔF1 (bbox − piksel GT) = **0.398 [0.377, 0.418]** ≫ 0.05 yanlışlama sınırı; Kendall τ (bbox vs piksel sıralaması) −0.22. **H1 doğrulandı.** Tohumlar arası std: val-crop AUPRC ≤ 0.002, tam-görüntü AUPRC 0.015, F1 0.03.

![Aynı tahminler, üç GT](phase2/experiments/figures/fig_dual_gt_metrics.png)

**H2 — bbox modeli ne öğreniyor? (kısmen).** Tahmin dikdörtgenselliği (3-tohum havuz medyanı): bbox 0.87 · piksel 0.32 · glif 0.05; bbox modeli alan içindeki gliflerin bulanıklaştırılmasına duyarsız (iç-blur +1.2 % AUPRC), sınır-halka blur'u yalnız −1.5 % (plasebo 0) → ön-kayıtlı **sınır-dikişi kısayolu yanlışlandı**; 0.5 eşiğinde authentic görüntülerin %44'ünde > %0.1 pozitif piksel (piksel modeli %10) ama kalibre τ*'da FP oranları benzer (kalibrasyon farkı). Image-level AUROC (top-1 %, val, 3 tohum): bbox 0.892 [0.876, 0.907], piksel 0.892, glif 0.799.

**H3 — dayanıklılık / transfer (yanlışlandı).** ΔAUPRC (kendi GT'si): JPEG Q75 bbox −0.154 / piksel −0.137; 0.75× −0.112 / −0.245; blur σ=1 −0.190 / −0.361. Held-out generator (Gemini-only eğitim → Ideogram): piksel − bbox ΔAP +0.025 [−0.039, 0.088] (Holm p 0.86; n=41, tek tohum → zayıf kanıt).

**H4 — bbox dışında küresel iz (yanlışlandı).** Maskeye değmeyen 224 crop'larda forged-vs-authentic AUROC **0.516** [0.513, 0.519] (pozitif kontrol 0.980); 260/457 çiftte bbox-dışı crop'lar piksel-özdeş.

### 4.3 Yeşil işaretçi kalıntısı (keşifsel sonda, 368 val çifti, 9 model)
| Model | Girdi | AP (bbox GT) | Kutu kapsama @τ\* | Görüntü skoru |
|---|---|---|---|---|
| bbox | sahte | 0.988 | 0.940 | 0.634 |
| bbox | sahte, **işaretçi onarılmış** | 0.763 (Δ −0.225 [−0.262, −0.189]) | 0.592 | 0.450 |
| bbox | authentic | 0.178 | 0.042 | 0.098 |
| bbox | **authentic + işaretçi** | **0.906 [0.887, 0.924]** | 0.799 | 0.532 |
| piksel | sahte / onarılmış / authentic + işaretçi | 0.749 / 0.499 / 0.641 | 0.539 / 0.280 / 0.426 | 0.195 / 0.085 / 0.126 |
| glif | sahte / onarılmış / authentic + işaretçi | 0.524 / 0.400 / 0.465 | 0.325 / 0.193 / 0.284 | 0.089 / 0.064 / 0.102 |

Yalnız işaretçi pikselleri authentic'e aktarılınca bbox modeli kutuyu çizer; gerçek sahtede yalnız işaretçi onarılınca performansın dörtte biri gider; öğrenmesiz "yeşil çizgi → kutu" kuralı F1 0.054 verir (ipucu öğrenilmiş bir kısayoldur). Gemini–Ideogram "üretici farkı" büyük ölçüde işaretçi farkıdır. Ara yüzdeki örnek 000000329'da 74 px kalıntı tespit edilmiş, onarım adımı 240 px'i authentic'ten kopyalamıştır.

![Yeşil işaretçi sondası](phase2/experiments/figures/fig_green_marker.png)

### 4.4 Arayüzle doğrulanan örnek (training:000000329, CORD, Gemini, alan < %1)
| Model | GT | Eşik | F1 | IoU | AP | Dikdörtgensellik |
|---|---|---|---|---|---|---|
| U-Net bbox-GT (s42) | bbox | 0.45 | **0.984** | 0.969 | 1.000 | 0.96 |
| U-Net bbox-GT (s42) | piksel-değişim | 0.95 | **0.635** | 0.466 | 0.799 | 0.97 |
| U-Net piksel-GT (s42) | bbox | 0.05 | 0.808 | 0.679 | 0.764 | 0.11 |
| U-Net glif-GT (s42) | bbox | 0.05 | 0.518 | 0.349 | 0.485 | 0.03 |
| U-Net RGB (PHASE 1, smoke) | bbox | 0.50 | 0.375 | 0.231 | 0.979 | 0.38 |
| U-Net bbox-GT (s42), **JPEG Q75 girdi** | bbox | 0.45 | 0.953 | 0.909 | 0.996 | 0.90 |
| ResNet-18 ft | — | — | GT'yle kesişen patch 0.65 / 0.98 (ort/maks); temiz patch 0.11 / 0.50 | | | |

Tek görüntüde bile H1 örüntüsü (aynı tahmin: bbox-GT F1 0.98 → piksel-GT 0.64) ve JPEG'in etkisi (0.98 → 0.95) yeniden üretilmektedir.

### 4.5 Uygulamanın doğrulanması (yerel otomatik test + canlı ZeroGPU)
Arayüzün raporlanan sonuçları değiştirmeden yeniden ürettiğini göstermek için üç düzeyde test yapıldı.

**(a) Yerel otomatik test — 281 kontrol, 0 hata** (`tests/test_app.py`, CPU). Kapsam: 61 ön işleme işleminin her biri (çıktı tipi, boyut, sonlu değer), presetler ve geri al/sıfırla; örnek filtreleri ve rastgele seçim; dört senaryo ve altı bozulma; simülasyon çıktılarının tamamı (galeri, tablo, karar, yakın plan, 4 rapor dosyası); hatalı girdiler (geçersiz kimlik, boş seçim, geometrik pipeline sonrası boyut uyuşmazlığı); toplu testin iki modu. **Yeniden üretim:** kayıtlı sonucu bulunan 33 test görüntüsünde uygulamanın hesapladığı F1@0.5 ile boru hattının kayıtlı `per_image` sonuçları arasındaki ortalama mutlak fark bbox-GT 0.006, piksel-GT 0.007, glif-GT 0.001 (CPU'da bf16/float yuvarlama farkı; en büyük tekil fark 0.145, küçük alanlı bir görüntüde).

**(b) Tarayıcı (UI) testi** (`tests/ui_test.py`, Playwright): örnek seçimi → simülasyon → kısayol senaryosu → ön işleme preseti → toplu test akışı gerçek bir tarayıcıda tıklanarak çalıştırıldı; hata yok (simülasyon 30 s, 6+6 örnek toplu test 51 s, CPU). Bu bölümdeki ekran görüntüleri bu testten alınmıştır.

**(c) Canlı Space testi** (`hf_space/live_test.py`, ZeroGPU, 9 Kasım 2026): `gradio_client` ile Space'in API'si üzerinden.

*Simülasyon — rastgele seçilen `testing:000000452` (işaretçili, eşli), dört senaryo:*

| Senaryo | ResNet-18 ft | U-Net bbox-GT | U-Net piksel-GT | U-Net glif-GT |
|---|---|---|---|---|
| Sahte belge | SAHTE ✓ (0.997) | SAHTE ✓ (1.000) · F1 bbox 0.971 · F1 piksel 0.483 | SAHTE ✓ (0.980) · 0.768 / 0.604 | SAHTE ✓ (0.987) · 0.600 / 0.468 |
| Authentic eş | GERÇEK ✓ (0.263) | GERÇEK ✓ (0.001) | GERÇEK ✓ (0.002) | GERÇEK ✓ (0.001) |
| Sahte, işaretçi onarılmış | GERÇEK ✗ (0.248) | GERÇEK ✗ (0.001) · F1 0 | GERÇEK ✗ (0.002) · F1 0 | GERÇEK ✗ (0.001) · F1 0 |
| Authentic + yalnız işaretçi | SAHTE ✗ (0.997) | SAHTE ✗ (1.000) · kutu kapsama 0.907 | SAHTE ✗ (0.980) · 0.610 | SAHTE ✗ (0.985) · 0.442 |

Parantez içi: görüntü-düzeyi skor. Bu örnek, PHASE 2 Ek-C'deki kısayolun **tek görüntüde, dört modelin hepsinde** görülebildiği en net durumdur: sahte metin yerinde kalıp yalnız işaretçi silindiğinde modeller sahteciliği kaçırır; hiçbir karakter değişmeden yalnız işaretçi eklendiğinde ise kutuyu çizip SAHTE der. Modellerin bu örnekte öğrendiği şey inpainting izi değil işaretçidir. (Her senaryo çağrısı ~36 s; GPU tahsisi ve 4 modelin tam çözünürlükte çalışması dâhil.)

*Toplu test — test bölümü, 24 rastgele eşli örnek (tohum 42), dengeli tespit (24 sahte + 24 authentic), 49 s:*

| Model | F1 bbox-GT | IoU bbox-GT | AP bbox-GT | F1 piksel-GT | AP piksel-GT | TPR | FPR | Doğruluk | Görüntü AUROC | Yeniden üretim \|ΔF1@0.5\| |
|---|---|---|---|---|---|---|---|---|---|---|
| ResNet-18 tam ince ayar | — | — | — | — | — | 1.000 | 0.375 | 0.812 | 0.990 | — |
| U-Net bbox-GT (s42) | **0.925** | 0.871 | **0.964** | 0.528 | 0.410 | 1.000 | 0.042 | 0.979 | 0.988 | **0.000** (n=24) |
| U-Net piksel-GT (s42) | 0.739 | 0.602 | 0.730 | **0.635** | **0.609** | 1.000 | 0.083 | 0.958 | 1.000 | **0.000** (n=24) |

GPU'da uygulamanın hesapladığı F1 ve AP, aynı görüntülerde boru hattının kayıtlı sonuçlarıyla **birebir aynıdır** (|ΔF1| = |ΔAP| = 0.000). Örneklem küçük olduğu için mutlak değerler resmî test sonuçlarından (§4.2: F1 0.846, AUPRC 0.950) sapabilir; H1 örüntüsü (bbox-GT modeli bbox GT'siyle 0.925, aynı tahminler piksel GT'siyle 0.528) burada da korunur. ResNet-18'in yanlış alarm oranı (0.375) sabit 0.5 eşiğinden kaynaklanır; AUROC 0.990 olduğundan sıralama iyidir, eşik kalibrasyonu sonraki adımdır.

### 4.6 Sınırlamalar ve sonraki adımlar
- Uygulamadaki görüntü-düzeyi eşikler val'de tek tohumla (s42) seçildi; test bölümünde yalnız 89 eşli authentic olduğu için test FPR'si geniş güven aralıklıdır. ResNet-18 için eşik kalibre edilmedi (0.5).
- Smoke sonuçları tek tohum ve alt-örneklenmiş AUPRC ile provizyoneldir; tam-veri sonuçları 3 tohumludur ama tek mimariye (U-Net/ResNet-18) dayanır.
- GT doğrulaması insan yerine görsel dil modeli ile yapılmıştır (96 çift: piksel GT OK 69 / PARTIAL 26 / WRONG 1); insan denetimi beklemektedir.
- Held-out generator deneyi zayıf (Ideogram n=41, tek tohum). Image-level test metriği veri seti yapısı nedeniyle verilemez.
- Yeşil işaretçi bulgusu keşifseldir; **işaretçisiz eğitim seti ile yeniden eğitim** (3 GT × tohum), ikinci mimari, 20 epoch, AIForge-Doc v2 ve insan GT denetimi sonraki aşamadır. Bildiri hedefi: IEEE SIU 2027 / ICDAR 2027 WS / WIFS 2027.

## 5. Kaynaklar
Tam BibTeX: `literature/references.bib` (39 kayıt + bu sürümde eklenenler). Başlıcaları:
1. Wu J., Zhou Y., Xu M., et al. (2026). AIForge-Doc: A Benchmark for Detecting AI-Forged Tampering in Financial and Form Documents. arXiv:2602.20569.
2. Wu J., Zhou Y., Ng D.T., et al. (2026). When the Forger Is the Judge: GPT-Image-2 Cannot Recognize Its Own Faked Documents. arXiv:2604.25213.
3. Raj A., Zhang Y., Zewde K., et al. (2026). ChatGPT Images 2.5 on Forgery Tasks. arXiv:2609.13617.
4. Helali B. (2026). Beyond Zero-Shot on AIForge-Doc v2: A Trained Detector and a Process-Twin Control. Zenodo, doi:10.5281/zenodo.23917285 (hakemsiz).
5. Zhao Z., Xia W., Wei E., et al. (2026). DOCFORGE-BENCH. arXiv:2603.01433.
6. Qu C., Liu C., Liu Y., et al. (2023). Towards Robust Tampered Text Detection in Document Image. CVPR 2023. doi:10.1109/CVPR52729.2023.00575
7. Chen Z., et al. (2024). Enhancing Tampered Text Detection through Frequency Feature Fusion and Decomposition. ECCV 2024. doi:10.1007/978-3-031-73414-4_12
8. Wong K., et al. (2025). ADCD-Net. ICCV 2025. doi:10.1109/ICCV51701.2025.01792
9. Guillaro F., Cozzolino D., et al. (2023). TruFor. CVPR 2023. doi:10.1109/CVPR52729.2023.01974
10. Mareen H., et al. (2024). TGIF: Text-Guided Inpainting Forgery Dataset. IEEE WIFS 2024. doi:10.1109/WIFS61860.2024.10810690
11. Yermakov A., Cech J., Matas J., Fritz M. (2026). Deepfake Detection that Generalizes Across Benchmarks. WACV 2026. doi:10.1109/WACV61042.2026.00082
12. Geirhos R., et al. (2020). Shortcut Learning in Deep Neural Networks. Nature Machine Intelligence. doi:10.1038/s42256-020-00257-z
13. Nebioglu et al. (2026). INP-X. arXiv:2602.00192. · Mareen et al. (2026). TGIF2. arXiv:2603.28613. · Giakoumoglou et al. (2026). DiffusionPrint. arXiv:2604.12443.
14. He K., Zhang X., Ren S., Sun J. (2016). Deep Residual Learning for Image Recognition. CVPR 2016.
15. Ronneberger O., Fischer P., Brox T. (2015). U-Net. MICCAI 2015.
16. Ojala T., Pietikäinen M., Mäenpää T. (2002). LBP. IEEE TPAMI. · Haralick R.M., et al. (1973). Textural Features. IEEE TSMC. · Ahmed N., Natarajan T., Rao K.R. (1974). Discrete Cosine Transform. IEEE Trans. Computers.
17. Cortes C., Vapnik V. (1995). Support-Vector Networks. Machine Learning. · Breiman L. (2001). Random Forests. Machine Learning. · Milletari F., et al. (2016). V-Net / Dice loss. 3DV 2016.
18. Gonzalez R.C., Woods R.E. Digital Image Processing, 3. baskı, Pearson. · Sauvola J., Pietikäinen M. (2000). Adaptive document image binarization. Pattern Recognition.
19. Joya15 (erişim 9 Kasım 2026). Multi-Generator Deepfake Document Detection and Tamper Localization [GitHub, hakemsiz]. · leongyiquan/DocDet [GitHub, hakemsiz].

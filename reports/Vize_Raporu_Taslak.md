# Vize Proje Raporu (Taslak)

**Ders:** AIC515 Görüntü İşleme — Prof. Dr. Onur Osman, Topkapı Üniversitesi
**Öğrenci / Grup:** ………………………… (en fazla 3 kişi)
**Tarih:** Ekim 2026

## 1. Proje Adı

**Yapay Zekâ ile Üretilmiş Finansal Belge Sahteciliğinin Tespiti ve Manipüle Edilmiş Bölgelerin Segmentasyonu: AIForge-Doc Veri Setinde Klasik Görüntü İşleme, ResNet ve U-Net Karşılaştırması**
(İngilizce çalışma başlığı: *Detecting and Localizing AI-Inpainted Forgeries in Financial Document Images: Classical Image Processing, ResNet and U-Net Baselines on AIForge-Doc*)

Araştırma yönü (PHASE 1 raporundan): *What Do Document Forgery Localizers Learn from Box Masks? Pixel-Change Ground Truth and Shortcut Audits for AI-Inpainted Financial Documents.*

## 2. Literatür

### 2.1 Problem
Üretken yapay zekâ servisleri (Gemini 2.5 Flash Image, Ideogram v2 Edit, GPT-Image-2 vb.) ile bir fiş veya form görüntüsünde tek bir alanın (tutar, tarih, telefon) değiştirilmesi **inpainting** yoluyla, görsel olarak fark edilemeyen sahtecilikler üretmektedir. Problem iki düzeydedir: (i) **Image-level forgery detection** (görüntü sahte mi?) ve (ii) **Tampering localization** (hangi pikseller değiştirildi?). İkincisi bir ikili semantik segmentasyon problemidir ve bu dersin U-Net konusuyla doğrudan ilişkilidir.

### 2.2 Veri setiyle yapılmış çalışmalar
AIForge-Doc v1 veri setini tanıtan çalışma (Wu, Zhou, Xu ve ark., 2026, arXiv:2602.20569) CORD v2, WildReceipt, SROIE ve XFUND kaynaklı 4.061 belgede tek bir alanı iki ticari inpainting servisiyle değiştirmiş ve mevcut dedektörleri **sıfır-atış (zero-shot)** koşulunda değerlendirmiştir: genel amaçlı TruFor (Guillaro ve ark., CVPR 2023) image-AUC 0.751 ve pixel-IoU 0.358; belge-özel DocTamper/DTD (Qu ve ark., CVPR 2023) AUC 0.563 ve IoU 0.020; GPT-4o yargıcı AUC 0.509 (Tablo 2, s. 11). Aynı grubun AIForge-Doc v2 çalışması (Wu, Zhou, Ng ve ark., 2026, arXiv:2604.25213) GPT-Image-2 ile üretilen 3.066 sahtede insan 2AFC doğruluğunun 0.501, TruFor'un 0.599, DocTamper'ın 0.585 olduğunu; aynı dedektörlerin geleneksel splicing kalibrasyon setlerinde 0.962 / 0.852 AUC'ye ulaştığını raporlamıştır → AI-inpainting'e özgü 0.27–0.36 AUC açığı. Bu veri setiyle **eğitilmiş** (fine-tune edilmiş) bir model içeren yayın, literatür taramamızda (38 çalışma, 17 Ekim 2026 kesimi) bulunamamıştır. 27 Ekim 2026 güncellemesinde iki ek kaynak bulundu. (i) Aynı grubun üçüncü tur çalışması (Raj ve ark., 2026, arXiv:2609.13617) v3'te çizilen işaretçiyi alfa maskesiyle değiştirmiş ve TruFor haritalarının yapıştırılan kırpıntı dikdörtgenine karşı alan kutusu kadar iyi skorladığını bildirmiştir (piksel AUC 0.921 vs 0.918). Yani v3'te bir yapıştırma-dikişi kısayolu raporlanmıştır; bulgu zero-shot modeller ve v3 ile sınırlıdır. (ii) Hakemsiz bir GitHub deposu (Joya15) v1 ve v2 üzerinde SegFormer-Lite eğitip çapraz-veri piksel F1'ini 0.044 (v2→v1) ve 0.0001 (v1→v2) olarak raporlamaktadır; bu sonuçlar doğrulanmamıştır. Bu nedenle "AIForge-Doc üzerinde denetimli eğitim yapan ilk çalışma" iddiası kullanılmamalıdır. Savunulabilir iddia daha dardır: v1 üzerinde eğitim maskesinin geometrisini kontrollü biçimde değiştiren ve işaretçi kalıntısını ölçen bir çalışmaya rastlanmamıştır.

### 2.3 Belge sahteciliği lokalizasyonunda yöntemler ve sonuçlar
- **DocTamper / DTD (CVPR 2023):** 170k sentetik görüntü; RGB + 8×8 blok **DCT** katsayıları ve JPEG kuantizasyon tablosu (Frequency Perception Head), Swin omurgası, JPEG kalite müfredatı. DocTamper test IoU 0.828 / F 0.792.
- **FFDN (ECCV 2024):** Frekans bilgisinin RGB'yi bozmadan eklenmesi (sıfır-başlatılmış Visual Enhancement Module) ve Haar dalgacık ayrıştırması; DocTamper-T F 0.86, FCD 0.92; JPEG Q75'te 0.75.
- **ADCD-Net (ICCV 2025):** DCT ipucunun 8×8 hizalaması bozulunca (1 px kaydırma, %2 kırpma, %98 ölçekleme) DTD'nin F1'inin %68–73 düştüğünü ölçmüş; hizalama skoruyla adaptif DCT ağırlıklandırma önermiştir (ortalama F1 0.787, bozulmalarda TruFor'u ortalama %20.8 aşar). Kod MIT lisanslıdır.
- **DOCFORGE-BENCH (2026):** 14 yöntem × 8 belge seti sıfır-atış: hiçbir yöntem setlerin ≥6/8'inde Pixel-F1 ≥ 0.3'e ulaşamaz; kurcalanmış piksel oranı %0.27–4.17 olduğundan τ = 0.5 eşiği kalibrasyonsuzdur (yüksek Pixel-AUC, sıfıra yakın F1); 10 görüntüyle eşik kalibrasyonu açığın %39–55'ini kapatır.
- **Genel manipülasyon lokalizasyonu:** TruFor (Noiseprint++ gürültü artığı + RGB), CAT-Net (DCT/JPEG), MVSS-Net (gürültü + Sobel kenar görünümleri), PSCC-Net, IML-ViT, Mesorch, SAFIRE. TGIF (WIFS 2024) diffusion inpainting görüntü **tamamen yeniden üretildiğinde** bu yöntemlerin F1'inin ≤ 0.1'e düştüğünü göstermiştir.
- **Shortcut learning ve değerlendirme güvenilirliği:** Geirhos ve ark. (2020) çerçevesi; GenD (WACV 2026) eşleşmemiş gerçek–sahte eğitimde AUROC'un 90.0'dan 56.8'e düştüğünü; DocQT (2026) yalnızca JPEG kuantizasyon tablosu farkının FFDN'de 14.5 F1 puanı değiştirdiğini raporlar.

### 2.4 Projede kullanılan modellerin kaynakları
| Model / teknik | Sunulduğu çalışma |
|---|---|
| ResNet-18 (residual learning) | He, Zhang, Ren, Sun, "Deep Residual Learning for Image Recognition", CVPR 2016 |
| U-Net | Ronneberger, Fischer, Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation", MICCAI 2015 |
| ImageNet ön-eğitim / transfer öğrenme | Deng ve ark., CVPR 2009; Yosinski ve ark., NeurIPS 2014 |
| LBP | Ojala, Pietikäinen, Mäenpää, IEEE TPAMI 2002 |
| GLCM (Haralick dokusu) | Haralick, Shanmugam, Dinstein, IEEE TSMC 1973 |
| 2-B DCT | Ahmed, Natarajan, Rao, IEEE Trans. Computers 1974 |
| SVM | Cortes & Vapnik, Machine Learning 1995 |
| Random Forest | Breiman, Machine Learning 2001 |
| Dice kaybı | Milletari, Navab, Ahmadi, 3DV 2016 |
| Frekans domeni yüksek-geçiren filtreleme | Gonzalez & Woods, Digital Image Processing, 3. baskı, Bölüm 4 |

Ayrıntılı literatür matrisi (38 çalışma, 15 alan) `literature/literature_matrix.xlsx` dosyasındadır.

### 2.5 Araştırma boşluğu, katkı ve yenilik
| Boşluk | Bu projenin katkısı |
|---|---|
| AIForge-Doc'ta yalnız zero-shot sonuçlar var; kontrollü denetimli eğitim yok | 3 GT × 3 seed U-Net ve resmi test değerlendirmesi |
| Kutu maskesi "pixel-precise" kabul ediliyor; modele ne öğrettiği sorgulanmamış | Eşleşmiş authentic'lerden piksel-değişim ve glif GT'si; çift-GT değerlendirme (H1) |
| Üretim hattı artefaktlarının eğitilmiş modellerce kullanılıp kullanılmadığı bilinmiyor | Benign kontroller, sınır ablasyonları, işaretçi aktarım/onarım sondası |
| Sabit eşik (F1@0.5) ve tek koşu | Eşikten bağımsız metrikler, kalibre τ*, grup bootstrap, Holm |
| Kaynak belge sızıntısı ölçülmemiş | Grup-farkında doğrulama ayrımı; sızıntı %8.9 |

**Yeniliğin türü:** yeni bir mimari önerilmemektedir; katkı metodolojik ve ampiriktir. Bu katkı dört başlıkta toplanır: (1) yalnız eğitim maskesinin değiştirildiği kontrollü deney; (2) türetilmiş GT'ler; (3) v1 makalesinde (§3.3) çizildiği belirtilen yeşil işaretçinin yayımlanan sahtelerin %75.4'ünde kalıntı bıraktığının ve tüm modellerin onu kullandığının gösterilmesi; (4) yeniden kullanılabilir kısayol denetim protokolü. **Ar-Ge niteliği:** çalışma OECD Frascati (2015) ölçütlerini (yenilik, yaratıcılık, belirsizlik, sistematiklik, tekrarlanabilirlik) karşılar ve türü uygulamalı araştırmadır. Ayrıntı: `reports/Proje_Konumlandirma_Yenilik_OnIsleme.md`.

## 3. Materyal ve Metot

### 3.1 Veri tabanı: AIForge-Doc v1
- **Kaynak:** Hugging Face `Scam-AI/AIForge-Doc-v1` (gated; lisans beyanı CC BY-NC-SA 4.0 — ticari olmayan araştırma kullanımı). Yerel kopya: 12.187 dosya, 9.98 GB (`Vize Proje/data/AIForge-Doc-v1`).
- **İçerik (yerel audit ile doğrulandı):** 3.249 eğitim + 812 test sahte görüntü (RGB PNG) ve 0/255 ikili maskeler; her sahtenin kaynak (authentic) görüntüsü (eğitim tarafında 3.247/3.249 çift doğrulandı). Kaynak dağılımı: WildReceipt 1.696, CORD 1.000, SROIE 946, XFUND 419; belge tipi: fiş %89.7, form %10.3; 9 dil (EN 2.642, ID 1.000). Generator: Gemini 2.5 Flash Image 3.639 (%89.6), Ideogram v2 Edit 422 (%10.4).
- **Manipülasyon:** tek bir alanın değeri değiştirilir (fiyat ×U(1.15,3.0) vb.); bbox %50 genişletilmiş bir pencere inpaint edilir, yalnız alan bbox'ı geri yapıştırılır; maske = alan bbox'ı (tüm maskeler tek dikdörtgen).
- **İstatistikler:** çözünürlük 150×168 – 6016×7016 px (medyan 750×1296; 0.03–34.8 MP); kurcalanmış alan / görüntü alanı medyan **%0.98** (%50.7'si < %1); maske yüksekliği medyan 33 px → görüntü küçültülmeden **natif çözünürlükte crop** ile çalışılması zorunludur; piksel sınıf dengesizliği ≈ 1:100.
- **Audit bulguları (projeyi etkileyen):** (i) maskeler bbox düzeyindedir; eşleşen authentic ile fark alındığında bbox piksellerinin medyan %49'u gerçekten değişir; (ii) çiftlerin %33.5'inde bbox dışında da değişim vardır; (iii) resmî test authentic'leri test sahtelerinin aynası değildir (812'den 86'sı) → image-level değerlendirmede kullanılamaz; (iv) test sahtelerinin %8.9'u eğitimdeki bir kaynak belgeyi paylaşır. Ayrıntı: `dataset_audit/audit_report.md`.
- **Veri bölme (test setine dokunulmadı):** eğitim kümesi kaynak-belge gruplarına göre **train 2.762 / val 487**; vize için **smoke_train 400 / smoke_val 100** (alan dilimine tabakalı, kaynak×generator dengeli) — `dataset_audit/splits.json`.

![Örnek görüntü–maske çiftleri](dataset_audit/fig_examples_overlay.png)

![Kurcalanmış alan oranı dağılımı](dataset_audit/fig_tamper_ratio.png)

### 3.2 Genel işlem hattı
1. **Patch/crop çıkarımı (natif çözünürlük):** her sahte görüntüden maske bbox merkezli 256×256 (sınıflandırma) veya 512×512 (segmentasyon) crop; aynı görüntüden maskeyle kesişmeyen temiz crop; eşleşen authentic görüntüden aynı koordinatlarda temiz crop (içerik-kontrollü negatif).
2. **Klasik öznitelikler + SVM / Random Forest** (patch sınıflandırma).
3. **ResNet-18** patch sınıflandırma: (a) ImageNet-pretrained, dondurulmuş gövde + yeni tam bağlı katman; (b) tam fine-tune; (c) sıfırdan eğitim.
4. **U-Net** piksel segmentasyonu: (D1) RGB; (D2) RGB + FFT yüksek-geçiren kanal + Laplacian artık kanalı.
5. **Değerlendirme:** sınıflandırmada accuracy, balanced accuracy, AUROC, F1; segmentasyonda Pixel-F1@0.5, IoU, Dice, Pixel-AUPRC, Pixel-AUROC, Oracle-F1; alan dilimi ve generator kırılımı; JPEG Q75 ve 0.75× yeniden boyutlandırma dayanıklılık probu.

### 3.2.1 Ön işleme ve gerekçesi
Adli bilişimde aranan sinyal düşük seviyeli istatistiklerdir (gürültü artığı, yeniden örnekleme ve sıkıştırma izi, glif kenarı). Bu nedenle görüntü *iyileştirme* amaçlı ön işleme yapılmamış, yalnız izi koruyan ve sahte/authentic için simetrik adımlar uygulanmıştır:
- **Yapılanlar:** kayıpsız PNG okuma (BGR→RGB); yeniden boyutlandırma olmadan doğal çözünürlükte 256/512 px kırpma (küçük görüntülerde sıfır dolgu); ImageNet ortalama/std normalizasyonu; eğitimde hafif parlaklık/kontrast jitter'ı (±%8); geometrik artırım yok (metin yönü); çıkarımda 512 px pencere, 448 px adım ve örtüşme ortalaması. Klasik hatta gri seviye dönüşüm ve LBP/GLCM/Laplacian/FFT/DCT öznitelikleri; pilot U-Net'te FFT yüksek-geçiren ve Laplacian kanalları kullanıldı. Morfoloji ve Sauvola ikilileştirme yalnız GT türetiminde kullanıldı, model girdisinde kullanılmadı.
- **Yapılmayanlar ve kanıtı:** sabit boyuta küçültme, gürültü giderme ve JPEG dönüşümü. Bunların etkisi test zamanı bozulmalarıyla ölçüldü (doğrulama, kendi GT'si). Gauss σ=1 bulanıklık ΔAUPRC'yi bbox modelinde −0.190, piksel modelinde −0.361; 0.75× ölçekleme −0.112 / −0.245; JPEG Q75 −0.154 / −0.137 değiştirdi.
- **Gerekli olan veri temizliği:** yeşil işaretçi kalıntısının (sahtelerin %75.4'ü) onarılması ve GT'den çıkarılması. Kalıntı tek başına modeli kutuyu çizmeye yöneltiyor (authentic + işaretçi AP 0.906). Bu adım sondada uygulandı; işaretçisiz yeniden eğitim bir sonraki deneydir.

### 3.3 Klasik öznitelikler (Hafta 3–4, 9–11)
**2-B blok DCT** (8×8 blok, $u,v\in\{0,\dots,7\}$):
$$F(u,v)=\tfrac{1}{4}C(u)C(v)\sum_{x=0}^{7}\sum_{y=0}^{7} f(x,y)\cos\!\Big[\tfrac{(2x+1)u\pi}{16}\Big]\cos\!\Big[\tfrac{(2y+1)v\pi}{16}\Big],\quad C(0)=\tfrac{1}{\sqrt2},\ C(k>0)=1.$$
İlk 15 zigzag AC katsayısının blok-ortalama ve standart sapması ile düşük/orta/yüksek frekans enerji oranları öznitelik vektörüne eklenir.

**LBP** (P=8, R=1, uniform): $LBP_{P,R}=\sum_{p=0}^{P-1} s(g_p-g_c)\,2^p,\ s(z)=\mathbb{1}[z\ge 0]$; 10 kutulu histogram.

**GLCM** (Haralick): birlikte-oluşum matrisi $p(i,j)$ için kontrast $\sum_{i,j}(i-j)^2p(i,j)$, homojenlik $\sum_{i,j}\frac{p(i,j)}{1+|i-j|}$, enerji $\sum_{i,j}p(i,j)^2$, korelasyon $\sum_{i,j}\frac{(i-\mu_i)(j-\mu_j)p(i,j)}{\sigma_i\sigma_j}$; 4 yön ortalaması.

**Yüksek-geçiren artık (uzamsal):** $r = f * \nabla^2$, Laplacian çekirdeği $\begin{bmatrix}0&1&0\\1&-4&1\\0&1&0\end{bmatrix}$; $\mathrm{mean}|r|,\ \sigma_r,\ \mathrm{kurtosis}(r)$.

**FFT radyal spektrum:** $|\mathcal{F}(u,v)|$ üzerinde $D(u,v)=\sqrt{(u-P/2)^2+(v-Q/2)^2}$ bantlarında enerji oranları.

**Frekans domeninde Gaussian yüksek-geçiren filtre (U-Net D2 kanalı):** $H_{HP}(u,v)=1-e^{-D^2(u,v)/(2D_0^2)}$, $g=\mathcal{F}^{-1}\{H_{HP}\cdot\mathcal{F}\{f\}\}$, $D_0\approx 20$ px.

**SVM (RBF):** $\min_{w,b,\xi}\tfrac12\|w\|^2+C\sum_i\xi_i$ s.t. $y_i(w^\top\phi(x_i)+b)\ge 1-\xi_i$; $K(x,x')=\exp(-\gamma\|x-x'\|^2)$.
**Random Forest:** bootstrap örneklerle $T=500$ karar ağacının çoğunluk oyu; öznitelik önemi = ortalama Gini azalması.

Pseudo kod (klasik hat):
```
for each patch P (grayscale):
    f_dct  <- block_dct_stats(P, block=8, n_ac=15)
    f_lbp  <- hist(LBP(P, P=8, R=1, 'uniform'), bins=10)
    f_glcm <- mean over angles of [contrast, homogeneity, energy, correlation]
    f_hp   <- stats(P * Laplacian3x3)
    f_fft  <- radial_band_energy_ratios(|FFT2(P)|)
    x <- concat(f_dct, f_lbp, f_glcm, f_hp, f_fft)
X <- StandardScaler.fit_transform(X_train)
clf <- SVM_RBF(C grid) | RandomForest(n=500)
report accuracy, balanced accuracy, AUROC, F1 on validation patches
```

### 3.4 ResNet-18 ile patch sınıflandırma (Hafta 12)
Residual blok: $y = \mathcal{F}(x,\{W_i\}) + x$ (kısa yol bağlantısı); ResNet-18 = 4 aşama × 2 temel blok (3×3 conv–BN–ReLU), global ortalama havuzlama, tam bağlı katman. Üç ayar karşılaştırılır:
1. **Ön-eğitilmiş + yeni katman (linear probe):** ImageNet ağırlıkları dondurulur; yalnız yeni $\mathrm{FC}:512\to 1$ eğitilir.
2. **Tam fine-tune:** tüm ağırlıklar güncellenir (lr $10^{-4}$).
3. **Sıfırdan:** rastgele başlatma (lr $10^{-3}$).
Kayıp: ikili çapraz entropi $\mathcal{L}_{BCE}=-[y\log\hat p+(1-y)\log(1-\hat p)]$; AdamW; 12 epoch; batch 32; giriş: 256×256 natif crop'un 224×224 merkez kesiti (yeniden örnekleme izi eklememek için resize yapılmaz); yatay çevirme **kullanılmaz** (metin yönü); yalnız küçük parlaklık/kontrast jitter.
```
model <- resnet18(weights=IMAGENET1K_V1); model.fc <- Linear(512, 1)
if setting == 'probe': freeze all params except model.fc
for epoch in 1..E:
    for (x, y) in train_loader: loss <- BCEWithLogits(model(x), y); loss.backward(); opt.step()
    val_auroc <- AUROC(sigmoid(model(x_val)), y_val); keep best checkpoint
```

### 3.5 U-Net ile piksel segmentasyonu (Hafta 13)
Kodlayıcı (ResNet-18, ImageNet) – kod çözücü (4 kademe yukarı örnekleme) ve **atlama bağlantıları** (skip connections) ile aynı çözünürlükteki kodlayıcı özellikleri kod çözücüye birleştirilir: $d_l=\mathrm{Up}(d_{l+1})\oplus e_l$. Çıkış: piksel başına logit; $\hat p=\sigma(z)$.
Kayıp: $\mathcal{L}=\mathcal{L}_{BCE}+\mathcal{L}_{Dice}$, $\ \mathcal{L}_{Dice}=1-\frac{2\sum_i \hat p_i y_i+\epsilon}{\sum_i\hat p_i+\sum_i y_i+\epsilon}$ (küçük pozitif bölge ve 1:100 dengesizlik için).
D2 varyantında giriş 5 kanaldır: $[R,G,B,\ g_{HP}^{FFT},\ r_{Lap}]$; ilk konvolüsyonun ilk 3 kanalı ImageNet ağırlığını korur, ek 2 kanal ortalama ağırlıkla başlatılır.
Eğitim: 512×512 natif crop (%70 bbox içeren, %30 rastgele), AdamW lr $10^{-4}$, 15 epoch, batch 8, bf16 AMP (RTX 5080, 16 GB).
```
for epoch: for (x, m) in crops: z <- UNet(x); loss <- BCE(z,m) + Dice(sigmoid(z), m); step
val: p <- sigmoid(UNet(x_val));  F1@0.5, IoU@0.5, Dice, AUPRC, AUROC, Oracle-F1 (per-image best τ)
post-processing (Hafta 6): Otsu eşiği ve morfolojik açma/kapama ile ikili maske; en büyük bileşenin bbox'ı
```

### 3.6 Değerlendirme ölçütleri
$\mathrm{Precision}=\frac{TP}{TP+FP},\ \mathrm{Recall}=\frac{TP}{TP+FN},\ F_1=\frac{2PR}{P+R},\ \mathrm{IoU}=\frac{TP}{TP+FP+FN},\ \mathrm{Dice}=\frac{2TP}{2TP+FP+FN}$; AUROC/AUPRC eşikten bağımsızdır; **Oracle-F1** görüntü başına en iyi eşikteki F1'dir (ulaşılabilir üst sınır). Küçük pozitif alan nedeniyle sabit τ=0.5 F1'i ile AUPRC birlikte raporlanır (DOCFORGE-BENCH protokolü). Sonuçlar generator (Gemini/Ideogram) ve kurcalanmış alan dilimi (<%1, %1–5) bazında ayrıca verilir. Dayanıklılık: aynı val crop'ları JPEG Q75 ve 0.75× küçült–büyüt ile bozulup eğitim yapılmadan değerlendirilir.

### 3.7 Donanım ve yazılım
RTX 5080 (16 GB, CUDA 12.8), Python 3.11, PyTorch 2.11 (+cu128), torchvision 0.26, segmentation-models-pytorch 0.5, scikit-learn 1.9, scikit-image, OpenCV. Kurulum: `baselines/ENV_SETUP.md`. Tüm kodlar: `baselines/`.

### 3.8 Kullanılan ve kullanılmayan yöntemler
| Aile | Kullanıldı | Gerekçe |
|---|---|---|
| Klasik öznitelik + SVM/RF | Evet (pilot) | Ders yöntemleri, referans tabanı |
| ResNet-18 (probe / ince ayar / sıfırdan) | Evet | "Önceden eğitilmiş + yeni katman" ile "kendi model" sorusu |
| U-Net (ResNet-18 kodlayıcı) | Evet (ana model) | Küçük alanlar için atlama bağlantıları; düşük maliyet |
| TruFor, ADCD-Net | Zero-shot referans | Eşit koşullarda değil, ayrı raporlandı |
| SegFormer / IML-ViT | Hayır | GT etkisini tek mimariyle izole etmek; veri boyutu |
| DTD, CAT-Net, FFDN (RGB+DCT) | Hayır | Veri PNG; JPEG/DCT ipucu yok |
| YOLO / Faster R-CNN | Hayır | Görev piksel düzeyinde; kutu tespiti kutu kısayolunu doğrudan öğrenir |
| GPT-4o / VLM dedektör | Hayır (yalnız GT denetim aracı) | v1'de AUC 0.509 |
| Model topluluğu, çok görevli ağ | Hayır | Bütçe; gelecekteki çalışma |

Birlikte kullanım: ImageNet ResNet-18 ağırlıklarının U-Net kodlayıcısına aktarılması; klasik filtre çıktılarının U-Net'e kanal olarak verilmesi (pilot); segmentasyon haritasından görüntü skoru türetilmesi (en yüksek %1 piksel ortalaması); BCE + Dice kayıp füzyonu.

## 4. Sonuçlar (çok kısıtlı veri: smoke_train 400 / smoke_val 100)

Bu bölümün amacı sistemin uçtan uca çalıştığını göstermektir; değerler küçük alt kümeye aittir ve nihai performans iddiası taşımaz. Test setine dokunulmamıştır.

**Kurulum.** smoke_train 400 sahte görüntü (+ eşleşen authentic) → eğitim; smoke_val 100 → değerlendirme; seed 42; RTX 5080; toplam GPU eğitim süresi 23.5 dk; TestingSet'e dokunulmadı. Patch kümesi (256 px, natif çözünürlük): 1.587 eğitim / 296 val patch (tampered / clean-same-image / clean-paired-authentic). Segmentasyon: 512 px natif crop (%70 bbox içeren, %30 rastgele); val'de görüntü başına 1 bbox-içeren + 1 rastgele crop. **Tüm değerler tek seed ve küçük alt küme ile provizyoneldir** (Ideogram n=25, 5–20 % alan n=1).

### B) Klasik öznitelikler + ML
| Model | Acc | Bal.Acc | AUROC | AUPRC | F1@0.5 | AUROC tamp-vs-paired | AUROC tamp-vs-same | AUROC gemini | AUROC ideogram | AUROC <1% | AUROC 1-5% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SVM-RBF (C=0.5, γ=0.01) | 0.534 | 0.606 | 0.646 | 0.443 | 0.546 | 0.552 | 0.744 | 0.648 | 0.641 | 0.643 | 0.648 |
| RandomForest (500 ağaç) | 0.622 | 0.653 | 0.706 | 0.536 | 0.573 | 0.610 | 0.807 | 0.696 | 0.738 | 0.732 | 0.685 |

### C) ResNet18
| Ayar | best ep / çalışan | Acc | Bal.Acc | AUROC | AUPRC | F1@0.5 | AUROC tamp-vs-paired | AUROC tamp-vs-same | AUROC gemini | AUROC ideogram | AUROC <1% | AUROC 1-5% | TN/FP/FN/TP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ResNet18 probe (pretrained, dondurulmuş + yeni FC) | 10/12 | 0.639 | 0.693 | 0.726 | 0.554 | 0.616 | 0.635 | 0.820 | 0.735 | 0.698 | 0.680 | 0.767 | 103/93/14/86 |
| ResNet18 full fine-tune (pretrained) | 6/11 | 0.885 | 0.845 | 0.899 | 0.884 | 0.809 | 0.878 | 0.922 | 0.910 | 0.867 | 0.883 | 0.913 | 190/6/28/72 |
| ResNet18 scratch (rastgele init) | 4/9 | 0.503 | 0.610 | 0.679 | 0.497 | 0.561 | 0.573 | 0.790 | 0.670 | 0.708 | 0.656 | 0.707 | 55/141/6/94 |

### D) U-Net
| Varyant | best ep / çalışan | Pixel-F1@0.5 | IoU@0.5 | Oracle-F1 | Global-eşik F1 (eşik) | Pixel-AUPRC | Pixel-AUROC | F1@0.5 (yalnız bbox crop) | IoU (bbox crop) | Temiz crop FP piksel oranı | Temiz crop'ların >%1 FP'li payı |
|---|---|---|---|---|---|---|---|---|---|---|---|
| U-Net RGB (D1, 3 kanal) | 11/15 | 0.707 | 0.637 | 0.740 | 0.708 (0.45) | 0.947 | 0.948 | 0.724 | 0.653 | 0.26% | 8.5% |
| U-Net RGB+FFT-HP+Laplacian (D2, 5 kanal) | 12/15 | 0.674 | 0.613 | 0.711 | 0.674 (0.45) | 0.939 | 0.940 | 0.696 | 0.635 | 0.14% | 3.4% |

| Varyant | F1 gemini | F1 ideogram | F1 <1% | F1 1-5% | F1 5-20% | F1 cord | F1 sroie | F1 wildreceipt | F1 xfund |
|---|---|---|---|---|---|---|---|---|---|
| U-Net RGB (D1, 3 kanal) | 0.800 (n=75) | 0.497 (n=25) | 0.606 (n=47) | 0.830 (n=52) | 0.756 (n=1) | 0.478 | 0.869 | 0.765 | 0.601 |
| U-Net RGB+FFT-HP+Laplacian (D2, 5 kanal) | 0.779 (n=75) | 0.449 (n=25) | 0.589 (n=47) | 0.792 (n=52) | 0.788 (n=1) | 0.474 | 0.839 | 0.772 | 0.475 |

### E) Dayanıklılık
| Varyant | Koşul | Pixel-F1@0.5 | IoU@0.5 | Oracle-F1 | Pixel-AUPRC | Pixel-AUROC | Temiz crop FP | F1 gemini | F1 ideogram |
|---|---|---|---|---|---|---|---|---|---|
| U-Net RGB (D1, 3 kanal) | clean (PNG) | 0.708 | 0.637 | 0.740 | 0.947 | 0.948 | 0.26% | 0.800 | 0.497 |
| U-Net RGB (D1, 3 kanal) | JPEG Q75 | 0.636 | 0.565 | 0.680 | 0.929 | 0.932 | 0.18% | 0.728 | 0.444 |
| U-Net RGB (D1, 3 kanal) | rescale 0.75x→1x | 0.638 | 0.560 | 0.676 | 0.939 | 0.943 | 1.24% | 0.734 | 0.421 |
| U-Net RGB+FFT-HP+Laplacian (D2, 5 kanal) | clean (PNG) | 0.674 | 0.613 | 0.711 | 0.939 | 0.940 | 0.14% | 0.779 | 0.449 |
| U-Net RGB+FFT-HP+Laplacian (D2, 5 kanal) | JPEG Q75 | 0.604 | 0.540 | 0.640 | 0.912 | 0.917 | 0.13% | 0.715 | 0.328 |
| U-Net RGB+FFT-HP+Laplacian (D2, 5 kanal) | rescale 0.75x→1x | 0.633 | 0.566 | 0.674 | 0.924 | 0.929 | 0.42% | 0.755 | 0.353 |

### F) Kaynak
| Model | Eğitim süresi (s) | Peak GPU bellek (GiB) | Parametre | Çıkarım (ms/örnek) | Checkpoint (MB) | Not |
|---|---|---|---|---|---|---|
| svm_rbf | 7 | 0.00 | — | 27.9 | — | CPU; çıkarım süresi = öznitelik çıkarımı/patch |
| random_forest | 6 | 0.00 | — | 27.9 | — | CPU; çıkarım süresi = öznitelik çıkarımı/patch |
| resnet18_probe | 64 | 0.33 | 11.18 M | 8.2 | 42.7 | 12/12 epoch, 224 patch, batch 1 bf16 |
| resnet18_ft | 76 | 0.59 | 11.18 M | 7.3 | 42.7 | 11/12 epoch, 224 patch, batch 1 bf16 |
| resnet18_scratch | 65 | 0.59 | 11.18 M | 7.5 | 42.7 | 9/12 epoch, 224 patch, batch 1 bf16 |
| unet_rgb | 603 | 1.70 | 14.33 M | 15.8 | 54.8 | 15/15 epoch, 512 crop, batch 1 bf16 |
| unet_rgbfreq | 601 | 1.75 | 14.33 M | 18.8 | 54.8 | 15/15 epoch, 512 crop, batch 1 bf16 |


**Okuma.**
1. **Ön-eğitilmiş + yeni katman vs sıfırdan:** tam fine-tune (AUROC 0.899) ≫ dondurulmuş gövde + yeni FC (0.726) ≫ sıfırdan (0.679). 1.6k patch ile sıfırdan eğitim kararsızdır (en iyi epoch 4, early-stop 9). Soru "pretrained model daha iyi olmuş mu?" için küçük-veri cevabı: evet, ama yalnızca gövde de güncellendiğinde.
2. **İçerik-kontrollü negatif daha zor:** aynı koordinatlı authentic patch'e karşı AUROC her modelde daha düşüktür (ft 0.878 vs clean-same 0.922; RF 0.610 vs 0.807; SVM 0.552 ≈ şans) → modellerin bir kısmı içerik/konum ipuçlarından yararlanmaktadır; bu kontrol Faz 2'de zorunlu tutulacaktır.
3. **Klasik öznitelikler** şans üstü ama zayıf (RF AUROC 0.706; önemler düz 0.020–0.032; LBP, Laplacian artık istatistikleri ve GLCM kontrast önde). Veri PNG olduğundan JPEG/DCT izi yoktur; DCT öznitelikleri ayırt edici değildir (ders açısından beklenen negatif bulgu).
4. **U-Net RGB lokalizasyonu** küçük veriyle beklenenden güçlüdür: Pixel-F1@0.5 0.707, IoU 0.637, AUPRC 0.947, temiz crop'larda FP piksel oranı %0.26. Tahmin görselleri modelin **bbox dikdörtgen bandını bütünüyle** işaretlediğini göstermektedir — audit'e göre eps=0'da bbox piksellerinin %95'i, eps=8'de %49'u gerçekten değişmektedir; yani GT dikdörtgeni öğrenmek kolay ve metrikle ödüllendirilen bir çözümdür. Bu, seçilen araştırma yönünün (bbox-GT shortcut / piksel-değişim GT) ampirik motivasyonudur [HYPOTHESIS; sınır-ablasyon testi yapılmadı].
5. **Generator ve kaynak etkisi büyük:** F1 Gemini 0.800 (n=75) vs Ideogram 0.497 (n=25); SROIE 0.869 vs CORD 0.478; <1 % alan 0.606 vs 1–5 % 0.830 → küçük bölge ve generator (kaynakla confound'lu) eksenleri Faz 2 protokolünde ayrıştırılmalıdır.
6. **RGB + FFT yüksek-geçiren + Laplacian kanalı** RGB'yi geçmedi (F1 0.674 vs 0.707; AUPRC 0.939 vs 0.947); yalnızca temiz crop FP'si düşüktür (%0.14 vs %0.26). Tek seed ile fark ayırt edilemez; "frekans dalı eklemek" tek başına katkı değildir (Gap C değerlendirmesiyle tutarlı).
7. **Dayanıklılık probu (Gap C):** JPEG Q75 ve 0.75× yeniden ölçekleme F1'i ~7 puan düşürür (0.708 → 0.636 / 0.638), AUPRC ≥ 0.91 kalır; yeniden ölçeklemede temiz crop FP %0.26 → %1.24. Çöküş yoktur: öğrenilen iz salt yüksek-frekans gürültüsü değildir.
8. **Kaynak:** ResNet-18 eğitimi 64–76 s (peak 0.33–0.59 GiB); U-Net 10 dk (peak 1.7 GiB, 14.3 M parametre); çıkarım 7–8 ms/224-patch, 16–19 ms/512-crop → tam veri (2.762/487) ve 3 seed ile Faz 2 deneyleri tek GPU'da saatler mertebesindedir.

**Sınırlamalar.** Smoke alt küme, tek seed; GT = bbox maskesi (bbox dışı gerçek değişimler GT'de "temiz" sayılır); crop/patch düzeyi değerlendirme (tam-görüntü sliding-window ve image-level tespit metriği yok); 11 train / 4 val görüntüde maskeyle kesişmeyen 256 px pencere bulunamadığından clean-same patch üretilemedi; AUPRC/AUROC pikseller alt-örneklenerek hesaplandı; sandbox'ta multiprocessing kapalı olduğundan süreler tek işçiyle ölçüldü.

Kod ve tüm çıktılar: `baselines/` (README.md, `common/data.py`, `features_classical.py`, `train_classical.py`, `train_resnet.py`, `train_unet.py`, `evaluate.py`, `run_all.ps1`, `results/`, `figures/`, `logs/`), checkpoint'ler: `models/`.


![ResNet-18 eğitim eğrileri (smoke_train/val)](baselines/figures/training_curves_resnet.png)

![U-Net eğitim eğrileri](baselines/figures/training_curves_unet.png)

![U-Net tahminleri: forged crop / GT (bbox) / RGB / RGB+frekans](baselines/figures/unet_predictions.png)

![Random Forest öznitelik önemleri](baselines/figures/rf_feature_importance.png)



### 4.b Tam-veri sonuçları (PHASE 2; train 2.748 / val 487 / resmî test 812; 3 seed)
Vize teslimi sonrasında aynı kod hattı tam veriye genişletildi; ayrıntı `reports/PHASE2_Metodoloji_ve_Sonuclar.md`. Özet:
- **Resmî test (bbox-GT):** U-Net (ResNet-18, bbox-GT ile eğitim) Pixel-AUPRC **0.950 ± 0.011**, F1@τ* **0.846 ± 0.015**, IoU 0.789; temiz 740 alt kümede aynı (sızıntı etkisi yok). Zero-shot referanslar: TruFor F1@0.5 0.403 / IoU 0.328 (makale: 0.434 / 0.358), ADCD-Net 0.182 / 0.149.
- **Annotation etkisi (H1, doğrulandı):** aynı tahminler piksel-değişim GT'si ile skorlandığında AUPRC 0.923 → 0.374, F1 0.84 → 0.45 (ΔF1 0.398 [0.377, 0.418]); her GT kendi modelini birinci sıraya koyar (Kendall τ −0.22).
- **Model davranışı (H2, kısmen):** bbox-GT modeli dikdörtgen üretir (rectangularity medyan 0.87) ve alan içindeki glifleri bulanıklaştırmaya duyarsızdır; sınır halkası blur'u yalnız −1.5 % (plasebo 0). Piksel-GT modeli 0.5 eşiğinde 4× daha az authentic görüntüyü işaretler (kalibrasyon farkı; image-level AUROC iki modelde 0.892).
- **Dayanıklılık / transfer (H3, yanlışlandı):** piksel-GT eğitimi JPEG Q75 / 0.75× / blur altında veya görülmemiş generator'da (Ideogram) tutarlı bir avantaj sağlamaz.
- **Global iz (H4, yanlışlandı):** bbox dışındaki crop'lardan forged/authentic ayrımı AUROC 0.516 (pozitif kontrol 0.980).

| Eğitim GT \ Değerlendirme GT (val, AUPRC) | bbox | piksel | glif |
|---|---|---|---|
| bbox | **0.923** | 0.374 | 0.257 |
| piksel (eps8_morph) | 0.579 | **0.610** | 0.511 |
| glif (glyph_v2) | 0.351 | 0.379 | **0.676** |

![Aynı tahminler, üç GT (val 487)](phase2/experiments/figures/fig_dual_gt_metrics.png)

### 4.c Eğitim dinamiği ve yeni bulgu: yeşil işaretçi artefaktı (PHASE 2 güncellemesi)

**Checkpoint değerlendirmesi.** 12 tam-veri U-Net koşusunun tamamı 10 epoch tamamlandı; en iyi doğrulama epoch'u 7–10 arasında, son üç epoch'taki doğrulama AUPRC eğimi −0,001…+0,005 (plato). Eğitim kaybı bbox GT'de 0,24'e inerken piksel/glif GT'de 0,43–0,53'te kaldı: piksel-düzeyi hedef daha zor ve daha gürültülü. Tohumlar arası std ≤ 0,01; tepe GPU belleği 4,66 GiB; koşu başına 9,5–11 dk. Daha uzun eğitim mutlak farkları daraltabilir, sıralamayı değiştirmesi beklenmiyor (tüm sayılar `phase2/experiments/results/checkpoint_summary.csv`).

**Yeşil işaretçi kalıntısı.** GT doğrulama panellerinde fark edilen 1–3 px doygun yeşil çizgi, üretim hattının inpainting öncesi çizdiği dikdörtgen işaretçinin kalıntısıdır. Eşleşen 3.336 çiftin %75,4'ünde (Gemini %83,6, Ideogram %1,5), resmi test sahtelerinin %74,4'ünde var; 600 authentic görüntüde 0 yanlış alarm. Keşifsel sondada (368 doğrulama çifti, 9 model): yalnızca işaretçi pikselleri *authentic* görüntüye aktarıldığında bbox modeli kutuyu çiziyor (görüntü-başına AP 0,906 [0,887, 0,924], kutu kapsama 0,80); gerçek sahtelerde yalnızca işaretçi onarıldığında AP 0,988 → 0,763 (Δ −0,225). Piksel ve glif modelleri de ipucunu kullanıyor (işaretçi piksel-değişim GT'sinin parçası). Öğrenmesiz "yeşil çizgi → kutu" kuralı ise F1 0,05 veriyor: ipucu öğrenilmiş bir kısayoldur, sığ bir piksel kuralı değil. Sonuç: bu veri setinde raporlanan lokalizasyon skorlarının önemli bir kısmı adli kanıt değil, artefakt ölçümüdür; Gemini–Ideogram "üretici farkı" büyük ölçüde işaretçi farkıdır. Ayrıntı: `PHASE2_Metodoloji_ve_Sonuclar.md` Ek-C ve `Bildiri_Tam_Metin.md` §6.6-bis.

## 5. Tartışma ve Sonraki Adımlar
- Klasik öznitelikler ve ResNet patch sınıflandırması, "içerik-kontrollü" (aynı koordinatlı authentic) negatiflerle değerlendirildiğinde modelin gerçekten manipülasyon izini mi yoksa içerik/konum ipuçlarını mı öğrendiğini ayırt etmeye yarar.
- U-Net RGB vs RGB+frekans kanalı karşılaştırması, dersin 3–4. hafta tekniklerinin (uzamsal ve frekans domeni yüksek-geçiren filtreleme) ölçülebilir katkısını test eder; AIForge-Doc PNG olduğundan JPEG-DCT tabanlı literatür yöntemleri (DTD/FFDN/ADCD-Net) bu veri setinde yapısal dezavantajlıdır.
- PHASE 2 tamamlanmıştır (bkz. §4.b ve `reports/PHASE2_Metodoloji_ve_Sonuclar.md`); bildiri taslağı `reports/Bildiri_Taslagi_IEEE.docx`. Başlangıçta planlanan yön: eşleşen authentic'lerden türetilen **piksel-değişim ground-truth**'u ile bbox-GT'nin eğitim ve değerlendirmedeki etkisi, paste-back sınırı shortcut probeları, kaynak-eşlenmiş held-out generator ve degradasyon altında transfer; tam train/val (2.762/487) + temiz test alt kümesi (740); 3 seed, group-bootstrap güven aralıkları. Ayrıntı: `reports/PHASE1_Arastirma_Raporu.md`.

## 6. Kaynaklar
Tam BibTeX: `literature/references.bib` (38 kayıt). Başlıca kaynaklar:
1. Wu J., Zhou Y., Xu M., et al. (2026). *AIForge-Doc: A Benchmark for Detecting AI-Forged Tampering in Financial and Form Documents.* arXiv:2602.20569.
2. Wu J., Zhou Y., Ng D.T., et al. (2026). *When the Forger Is the Judge: GPT-Image-2 Cannot Recognize Its Own Faked Documents.* arXiv:2604.25213.
3. Zhao Z., Xia W., Wei E., et al. (2026). *DOCFORGE-BENCH: A Comprehensive 0-shot Benchmark for Document Forgery Detection and Analysis.* arXiv:2603.01433.
4. Qu C., Liu C., Liu Y., et al. (2023). *Towards Robust Tampered Text Detection in Document Image: New Dataset and New Solution.* CVPR 2023. doi:10.1109/CVPR52729.2023.00575
5. Chen Z., Chen S., Yao T., et al. (2024). *Enhancing Tampered Text Detection through Frequency Feature Fusion and Decomposition.* ECCV 2024. doi:10.1007/978-3-031-73414-4_12
6. Wong K., Zhou J., Wu H., Si Y.-W., Zhou J. (2025). *ADCD-Net: Robust Document Image Forgery Localization via Adaptive DCT Feature and Hierarchical Content Disentanglement.* ICCV 2025. doi:10.1109/ICCV51701.2025.01792
7. Guillaro F., Cozzolino D., et al. (2023). *TruFor: Leveraging All-Round Clues for Trustworthy Image Forgery Detection and Localization.* CVPR 2023. doi:10.1109/CVPR52729.2023.01974
8. Mareen H., et al. (2024). *TGIF: Text-Guided Inpainting Forgery Dataset.* IEEE WIFS 2024. doi:10.1109/WIFS61860.2024.10810690
9. Yermakov A., Cech J., Matas J. (2026). *Deepfake Detection that Generalizes Across Benchmarks.* WACV 2026. doi:10.1109/WACV61042.2026.00082
10. Geirhos R., et al. (2020). *Shortcut Learning in Deep Neural Networks.* Nature Machine Intelligence. doi:10.1038/s42256-020-00257-z
11. He K., Zhang X., Ren S., Sun J. (2016). *Deep Residual Learning for Image Recognition.* CVPR 2016.
12. Ronneberger O., Fischer P., Brox T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI 2015.
13. Ojala T., Pietikäinen M., Mäenpää T. (2002). *Multiresolution Gray-Scale and Rotation Invariant Texture Classification with Local Binary Patterns.* IEEE TPAMI.
14. Haralick R.M., Shanmugam K., Dinstein I. (1973). *Textural Features for Image Classification.* IEEE TSMC.
15. Gonzalez R.C., Woods R.E. *Digital Image Processing*, 3rd ed., Pearson.
16. Raj A., Zhang Y., Zewde K., et al. (2026). *ChatGPT Images 2.5 on Forgery Tasks: Testing Advertised Improvements Against Known Answers.* arXiv:2609.13617.
17. OECD (2015). *Frascati Manual 2015.* OECD Publishing, Paris. doi:10.1787/9789264239012-en.
18. Joya15 (t.y.; erişim 27 Ekim 2026). *Multi-Generator-Deepfake-Document-Detection-and-Tamper-Localization* [GitHub deposu, hakemsiz].

# Seed Makale İncelemesi — AI Destekli Finansal Belge Sahteciliği Tespiti

**Kapsam:** 6 seed makale; tam metin okundu, kimlik bilgileri CrossRef/arXiv API ile, kod-checkpoint-lisans durumu GitHub ve Hugging Face API'leri ile doğrulandı (erişim tarihi: 19 Ekim 2026).
**Kanıt etiketleri:** VERIFIED = birincil kaynaktan (makale tam metni / resmi repo / dataset kartı) doğrudan okundu · REPORTED = ikincil kaynak aktarıyor · HYPOTHESIS = bizim çıkarımımız · UNKNOWN = doğrulanamadı.
**Erişim özeti:** 6/6 makale **full-text**. Üç arXiv makalesi doğrudan arxiv.org PDF'inden; DocTamper CVF Open Access PDF'inden; FFDN ECVA Open Access PDF'inden; ADCD-Net arXiv 2507.16397 PDF'inden okundu. alphaXiv bağlayıcısı onaylanmadığı için kullanılmadı (gerekmedi). Erişilemeyen makale yok.

---

## Yönetici Özeti

Altı makale iki kampa ayrılıyor. **Dedektör makaleleri** (DocTamper/DTD, FFDN, ADCD-Net) tamamı DocTamper eğitim kümesinde (120k sentetik, JPEG-sıkıştırılmış, copy-move/splicing/dizgi "generation") eğitilir ve karar kanıtı olarak RGB + 8×8 blok DCT katsayılarını (JPEG Block Artifact Grid tutarsızlığı) kullanır. **Benchmark makaleleri** (AIForge-Doc v1, DOCFORGE-BENCH, AIForge-Doc v2) ise bu dedektörlerin sıfır-atış koşulunda çöktüğünü gösterir: diffusion/çok-modlu inpainting (Gemini 2.5 Flash Image, Ideogram v2, GPT-Image-2) JPEG geçmişi ve PRNU süreksizliği bırakmaz; ayrıca belge alanında kurcalanan piksel oranı %0.27–4 olduğundan τ=0.5 eşiği kalibrasyonsuz kalır. Bizim Gap A–D'miz açısından en kritik bulgular:

1. **Gap A (artifact-invariance / shortcut):** Üç dedektör de DCT-shortcut'a bağımlı; ADCD-Net bunu açıkça ölçer (DTD, 2% kırpma / 1 px kaydırma / %98 ölçekleme ile F1'de %68–73 düşüş; Fig. 5). AIForge-Doc v1/v2'de sahte görüntü ekstra bir yeniden-örnekleme/PNG döngüsünden geçer (v2 App. B "generation-asymmetry caveat") — bir dedektörün generatör izini değil bu asimetriyi öğrenmesi riski **HYPOTHESIS** olarak açık.
2. **Gap B (cross-generator):** v1 iki araç (Gemini %89.6 / Ideogram %10.4, dengesiz), v2 tek araç (GPT-Image-2) ve v1 ile spec-eşli. Hiçbir makale "bir generatörde eğit, diğerinde test et" protokolünü çalıştırmamış; v1+v2 çifti bunu ilk kez mümkün kılıyor.
3. **Gap C (küçük bölge & dayanıklılık):** AIForge-Doc'ta medyan kurcalanmış alan %0.92 (IQR %0.35–1.55); DOCFORGE-BENCH bunun doğrudan kalibrasyon çöküşüne yol açtığını gösterir (Pixel-AUC ≥0.76 iken Pixel-F1 ≈0). AIForge-Doc v1/v2'de **hiç degradasyon (JPEG/blur/noise/resize) değerlendirmesi yok**.
4. **Gap D (annotation / değerlendirme güvenilirliği):** AIForge-Doc maskeleri piksel-değişim maskesi değil, **alan bbox maskesidir** ("pixel-perfect" iddiası paste-back inşasına dayanır). DTD/FFDN/ADCD-Net optimal-eşik F1 raporlarken DOCFORGE-BENCH τ=0.5 kullanır. AIForge-Doc'un DocTamper için alıntıladığı "AUC 0.98 / IoU 0.71" değerleri CVPR makalesinde yoktur (makale AUC raporlamaz; Test IoU 0.828). HF kartlarında lisans ve generatör adlandırması makalelerle çelişir.

---

## 1. AIForge-Doc (v1) — arXiv:2602.20569

### Kimlik (VERIFIED, arXiv API + PDF)
| Alan | Değer |
|---|---|
| Başlık | AIForge-Doc: A Benchmark for Detecting AI-Forged Tampering in Financial and Form Documents |
| Yazarlar | Jiaqi Wu*, Yuchen Zhou*, Muduo Xu, Zisheng Liang, Simiao Ren†, Jiayu Xue, Meige Yang, Siying Chen, Jingheng Huan (Duke, NYU, UNC, Scam.ai, USC) |
| Tarih / sürüm | 24 Şub 2026, v1 (17 sayfa, 10 şekil) |
| Yayın durumu | **Preprint** (hakemli venue bulunamadı; web taraması 19 Eki 2026) |
| DOI | yok |
| Veri | HF `Scam-AI/AIForge-Doc-v1` — gated, lisans kartı **cc-by-nc-sa-4.0** (VERIFIED, HF API) |
| Kod | Makale "fully automated open-source pipeline" der; metinde URL yok. HF README bibtex'i `github.com/YOUR_ORG/aiforge-doc` yer tutucusu → kod URL'si **UNKNOWN** |
| Checkpoint | Yok (benchmark; TruFor/DocTamper resmi ağırlıkları kullanılmış) |

### Problem ve varsayımlar (VERIFIED)
Problem: diffusion/çok-modlu inpainting API'leriyle fiş/form belgelerinde tek bir sayısal alanın değiştirilmesinin tespiti ve piksel düzeyi lokalizasyonu; mevcut dedektörlerin bu tehdide kör olduğu iddiası (§1, §6). Varsayımlar: (i) tehdit modeli = tek alan, tek görüntü, tek araç ("single-field-per-image", §7 sınırlama); (ii) AI inpainting sıkıştırma dikişi, klonlama istatistiği ve gürültü-artığı süreksizliği bırakmaz (§4.4 — kısmen TruFor sonuçlarıyla doğrulanıyor); (iii) zero-shot değerlendirme benchmark için "doğru kapsam" (§5), fine-tune edilmemiş.

### Veri üretim pipeline'ı (§3, VERIFIED)
- **Kaynak setler (Tablo 1, s.7):** CORD v2 1,000 (id; price/total) · WildReceipt 1,696 (en; tel/addr/price) · SROIE 946 (en; total/date) · XFUND 419 (7 dil; numeric answer) → **4,061** sahte görüntü, 9 dil.
- **Alan seçimi:** öncelik: financial amount > date > document ID > quantity > other numeric; pratikte WildReceipt'te telefon (n=931) ve adres (n=457) baskın (§4.1).
- **Değer mutasyonu (§3.2):** parasal alanlar U(1.15,3.0) veya U(0.20,0.85) ile ölçeklenir, 30 denemeye kadar aynı karakter sayısı aranır, yoksa digit-flip; tarih ±1–5 yıl/±1–3 ay/±3–15 gün; ID 1–2 rakam flip; miktar 2–5×.
- **Context-window inpainting (§3.3):** bbox her yönde %50 genişletilir (min **150 px**; HF kartında **100 px** → tutarsızlık), ikili maske (255 alan / 0 bağlam), inpainting yalnız crop üzerinde, **sadece alan bbox'ı paste-back**, maske = tüm görüntü 0 + bbox'ta 255. Gemini maske tensörü kabul etmediğinden crop üzerine **3 px parlak yeşil dikdörtgen** çizilir ve crop **2× LANCZOS** büyütülüp sonuç küçültülür (§3.5).
- **Generatörler (§3.4):** 7 sistem denendi; Gemini 2.5 Flash Image (OpenRouter) **3,639 / %89.6** ve Ideogram v2 Edit (fal.ai) **422 / %10.4** konuşlandırıldı. FLUX Fill Pro, SD 3.5 Medium, GPT-Image-1, SD 1.5 Inpainting 320 denemelik prompt ablasyonunda (4 araç × 20 prompt × 4 referans) hiç kalite eşiğini geçemedi; FLUX Fill Dev 24 GB VRAM nedeniyle denenmedi. Ideogram içerik filtresi 83 isteği (≈%16.6) reddetti → 81 spec Gemini'ye yönlendirildi.
- **Authentic görüntü:** evet; test 812 sahte + 812 authentic = 1,624 (Tablo 2 başlığı); HF deposunda TrainingSet/authentic 3,249 ve TestingSet/authentic 812 dosya var (VERIFIED, HF API).
- **Generator metadata:** `metadata.jsonl` alanları: spec_id, image_id, source_dataset, doc_type, language, field_name, original_value, forged_value, bbox_xyxy, **assigned_tool (`gemini-nano` / `qwen-inpaint`)**, split, new_id, generated_at (HF README, VERIFIED). **Dikkat:** kartta `qwen-inpaint` = Ideogram v2 Edit olarak eşlenmiş; v2 makalesi v1 karşılıklarını "Gemini-nano 2,729 / qwen-inpaint 333" diye sayar. 422 "Ideogram" görüntüsünün gerçekten Ideogram ile mi üretildiği **UNKNOWN**; Gap B analizi için kritik.
- **Kalite kontrol (§3.6):** PaddleOCR token kontrolü → tüm görüntüler yazar incelemesi → forged≠original kontrolü.
- **Blending/compositing:** paste-back = hard paste (bbox sınırında blending yok); Gemini yolunda 2× yeniden örnekleme döngüsü var. Authentic görüntü bu döngüden geçirilmiyor (v2 makalesi bunu açıkça "generation-asymmetry caveat" olarak kabul ediyor).

### Maske prosedürü (Gap D, VERIFIED)
8-bit gri PNG, 0/255, **"Tampered region = the exact field bounding box (tight, no padding)"** (§4.2). Yani **bbox maskesi**; alan içindeki değişmeyen arka plan pikselleri de 255. "Pixel-perfect" ifadesi inşa yoluyla "bbox dışı hiç piksel değişmedi" anlamındadır, gerçek piksel-değişim maskesi değildir. Medyan kurcalanmış alan 5,589 px², %0.92 (IQR %0.35–1.55), görüntülerin %52'sinde <%1 (§4.3, Şekil 4).

### Train/test bağımsızlığı ve unseen-generator (VERIFIED)
Sabit 80/20 (3,249/812), kaynak set ve araç bazında tabakalı. Aynı kaynak görüntü iki split'e düşmez (bire bir eşleme, §4.1). Tüm baseline'lar zero-shot; **unseen generator testi yok** (her şey zaten unseen). Araç bazlı kırılım yalnız TruFor için: Gemini AUC 0.778 [0.754,0.802] n=727; Ideogram 0.521 [0.447,0.597] n=85 (Tablo 3, s.11) — Ideogram CI'si şansı kapsıyor.

### Degradasyon ve küçük bölge (VERIFIED)
Degradasyon değerlendirmesi **yok** (PNG, sıkıştırma yok; CAT-Net bu yüzden dışlanmış). Küçük bölge: istatistik veriliyor (%0.92), ancak alan boyutuna göre tabakalı metrik yok.

### Raporlanan metrikler (Tablo 2, s.11; zero-shot, 1,624 görüntü; eşik 0.5)
| Yöntem | AUC (img) [95% CI] | AP | Acc@opt | IoU | F1 | AUCpx |
|---|---|---|---|---|---|---|
| TruFor | 0.751 [.726,.776] | 0.709 | 0.731 | 0.358 | 0.434 | 0.916 |
| DocTamper (resmi ckpt) | 0.563 [.535,.591] | 0.564 | 0.558 | 0.020 | 0.030 | 0.675 |
| GPT-4o (gpt-4o-2024-11-20) | 0.509 [.481,.537] | 0.516 | 0.535 | – | – | – |
Tabloda "DocTamper (in-distribution)†: AUC 0.98, AP 0.97, Acc 0.94, IoU 0.71, F1 0.74, AUCpx 0.95 — Reported in [22]" satırı var. **CVPR 2023 DocTamper makalesinde AUC/AP raporlanmaz; Test IoU 0.828, F 0.792'dir (Tablo 3/4).** Bu satırın kaynağı doğrulanamadı → **UNKNOWN / olası yanlış alıntı** (Gap D).
Diğer tutarsızlık: §5.2 DocTamper'ı "Swin + Document Frequency Loss (DFL) head + Neighboring Feature Coupling (NFC)" diye tanımlar; CVPR makalesinde bu modüller yoktur (FPH, MID, CLTD vardır).

### Sınırlamalar (yazarların, §7 + bizim)
Yazarlar: alan tipi kaynak şemaya bağımlı (WildReceipt'te telefon/adres), tek alan/tek görüntü, iki araç, fine-tune çalışması yok, insan algı baseline'ı yok. Bizim eklediklerimiz: bbox maskesi; Ideogram alt kümesi n=85 ile istatistiksel güçsüz; authentic görüntülerin asimetrik işlem geçmişi (HYPOTHESIS: shortcut riski); kod URL'si ve HF kartı lisans metni iç çelişkili (başlık cc-by-nc-sa-4.0, gövde "for any purpose" + CC BY 4.0 bağlantısı).

### Gap ilişkisi
A: Yüksek — iki aracın farklı iz bıraktığı (ΔAUC 0.257) gösteriliyor; shortcut analizi yok. B: Yüksek — iki generatör ama dengesiz; cross-generator eğitim protokolü yok. C: Yüksek — %0.92 medyan alan; degradasyon yok. D: Yüksek — bbox maskesi, alıntı tutarsızlıkları.

---

## 2. DocTamper / DTD — CVPR 2023

### Kimlik (VERIFIED, CrossRef + CVF PDF)
| Alan | Değer |
|---|---|
| Başlık | Towards Robust Tampered Text Detection in Document Image: New Dataset and New Solution |
| Yazarlar | Chenfan Qu, Chongyu Liu, Yuliang Liu, Xinhong Chen, Dezhi Peng, Fengjun Guo, Lianwen Jin* (SCUT, HUST, IntSig) |
| Venue | CVPR 2023, s. 5937–5946 — **peer-reviewed** |
| DOI | 10.1109/CVPR52729.2023.00575 |
| Kod | github.com/qcf-568/DocTamper — inference kodu + `models/` dizini; eğitim kodu e-posta ile; veri sentez kodu `stg/` (README, VERIFIED). Repo'da LICENSE dosyası yok (GitHub API `license: null`) |
| Checkpoint | README "Inference models and code" → var (REPORTED; dosya indirilip doğrulanmadı). v2 makalesi `dtd doctamper.pth` kullanmış |
| Lisans | Veri: **non-commercial, üniversite/araştırma kurumlarına lisanslı, şifre başvurusu gerekli** (README); BaiduDrive + Kaggle aynası. Kod lisansı belirtilmemiş → UNKNOWN |

### Problem ve varsayımlar
Fotoğraflanmış belgelerde görsel olarak tutarlı metin kurcalamasının (splicing, copy-move, generation) piksel düzeyi segmentasyonu. Varsayım: kurcalama 8×8 DCT bloklarının BAG sürekliliğini bozar; görüntü JPEG geçmişi taşır; eğitim/test JPEG Q∈[75,100], 1–3 kez sıkıştırma (§5.2).

### Forensic evidence ve spatial/frequency kullanımı (§4, VERIFIED)
- **VPH:** 7 konv blok, RGB. **FPH:** Y-kanalı DCT katsayı haritası + JPEG **kuantizasyon tablosu** gömmesi, stride-8 konv (her piksel bir 8×8 blok), CoordConv, MobileConv. Füzyon: concat + scSE → Swin-small encoder → **MID** (çok görünümlü yinelemeli decoder). Kayıp: CE + Lovász. **CLTD:** JPEG kalite faktörü (B1,100) aralığından, B1∈(100−S/T,100), T=8192 — kolaydan zora müfredat.
- Kanıt türü: RGB + DCT (JPEG BAG). Gürültü artığı, OCR, FFT yok.

### Veri seti (Tablo 2, s.4, VERIFIED)
170k (EN 95k, ZH 75k); copy-move 60k, splicing 50k, generation 60k; **train 120k, test 30k, FCD 2k (Noisy Office kaynaklı), SCD 18k (HUAWEI Cloud kaynaklı)**; 50,562 kaynak belge. Sıkıştırılmamış saklanır; piksel düzeyi annotation. Sentez (§3.1): PaddleOCR/Tesseract bbox → SAUVOLA ile ön/arka plan rengi → "selective copy-paste" (benzer stil gruplarında) ve "selective generation" (OpenCV/G'MIC ile silme + benzer font ile yazdırma) → post-processing. **Generation = dizgi (typesetting), AI inpainting değil.**

### Maske prosedürü
Sentetik üretimde değiştirilen metin bölgesi için piksel düzeyi maske (metin bölgesi, karakter/kelime). Maskenin bbox mu glif mi olduğu makalede açık değil → **UNKNOWN** (Şekil 7 GT'leri kelime-bloğu biçimli görünüyor).

### Train/test bağımsızlığı, unseen generator
FCD/SCD kaynak görüntü dağılımı farklı (cross-domain). Kurcalama yöntemi aynı sentez pipeline'ı → **unseen manipulation method testi yok**. AI generatör yok.

### Degradasyon ve küçük bölge
Yalnız JPEG (Q75–Q90, Tablo 5/7). Küçük bölge: motivasyon (Şekil 1), tabakalı metrik yok.

### Raporlanan metrikler (VERIFIED)
Tablo 3/4 (s.7; test 1–3× JPEG, Q≥75; eşik 0.5): **DTD Test IoU 0.828 / P 0.814 / R 0.771 / F 0.792; FCD IoU 0.749 / F 0.816; SCD IoU 0.691 / F 0.754** (66M param). Karşılaştırma F: CAT-Net 0.700/0.553/0.631; CAT-Net+CLTD 0.721/0.741/0.670; MVSS-Net 0.431/0.424/0.414; ManTra-Net 0.153/0.209/0.157; Swin-Uper 0.638/0.546/0.574. Tablo 7 IoU (Test) Q75/80/85/90: DTD 0.83/0.85/0.87/0.89. Tablo 6 T-SROIE: DTD P 0.9923 / R 0.9930 / F 0.9927 (Wang et al. 0.9680).

### Sınırlamalar
DCT'ye bağımlılık (ADCD-Net ve FFDN tarafından ölçülmüş: hizalama bozulunca çöküş; PNG'de zayıflık); sentetik veri; AI inpainting kapsam dışı; AUC raporlanmaz; eğitim kodu kapalı; veri lisansı kısıtlı.

### Gap ilişkisi
A: Yüksek (DCT shortcut'ın kaynağı). B: Düşük–orta (generatör yok; cross-domain FCD/SCD var). C: Orta (JPEG dayanıklılığı var, diğer bozulmalar yok). D: Orta (maske tanımı belirsiz; IoU/F tutarlı raporlanıyor).

---

## 3. FFDN — ECCV 2024

### Kimlik (VERIFIED, CrossRef + ECVA PDF)
| Alan | Değer |
|---|---|
| Başlık | Enhancing Tampered Text Detection through Frequency Feature Fusion and Decomposition |
| Yazarlar | Zhongxi Chen†, Shen Chen†, Taiping Yao, Ke Sun, Shouhong Ding, Xianming Lin, Liujuan Cao, Rongrong Ji (Xiamen Univ., Tencent Youtu) |
| Venue | ECCV 2024, LNCS, s. 200–217 (ECVA paper id 04834) — **peer-reviewed** |
| DOI | 10.1007/978-3-031-73414-4_12 |
| Kod | Makalede resmi repo URL'si **yok** (yalnız MMSegmentation bağlantısı). ForensicHub (github.com/scu-zjz/ForensicHub, CC-BY-4.0) içinde `tasks/document/models/ffdn/ffdn.py` uygulaması mevcut (VERIFIED, GitHub code search) |
| Checkpoint | DOCFORGE-BENCH FFDN'i ForensicHub ağırlıklarıyla çalıştırmış (REPORTED); resmi yazar checkpoint'i UNKNOWN |
| Lisans | Orijinal kod lisansı UNKNOWN; ForensicHub uygulaması CC-BY-4.0 |

### Problem ve varsayımlar
DCT bilgisinin doğrudan füzyonunun RGB tespit yeteneğini bozduğu ve downsampling'in küçük kurcalama izlerini kaybettirdiği gözlemi; frekans bilgisinin "kontrollü ve adaptif" verilmesi gerektiği varsayımı (§1, §2.2). Yazarlar açıkça PNG veya çoklu post-processing'de DCT'nin zararlı olabildiğini belirtir.

### Forensic evidence ve spatial/frequency kullanımı (§3, VERIFIED)
- **VEM:** [F_v2, F_d] üzerinde kanal + uzamsal dikkat → **zero-initialized conv** → RGB özelliğine toplanır (RGB baskın başlangıç). F_d, DTD'nin FPH'sinden (DCT + kuantizasyon tablosu) gelir.
- **WFE:** çok seviyeli özelliklerin **Haar** LL/HH bileşenlerine öğrenilebilir wavelet-benzeri ayrıştırması (LWD), dikkat, FFM (GFA + WLM) ile füzyon; F1' = Conv([F̂^LF, F1]), F4' = Conv([F̂^HF, F4]); FPN head.
- Backbone ConvNeXtV2-base, 512×512, MMSeg, 4×V100, AdamW 1e-4, 100k iter, batch 16; kayıp CE + Lovász. Test: DTD pipeline'ı, JPEG Q 75–100.
- Kanıt türü: RGB (birincil) + DCT (yardımcı) + özellik-uzayı wavelet (yüksek/düşük frekans).

### Veri ve split
DocTamper: train 120k; DocTamper-T 30k; FCD 2k; SCD 18k. T-SROIE 986 görüntü — **yeniden eğitilerek** test edilmiş (§4.4), yani zero-shot cross-dataset değil.

### Degradasyon (VERIFIED)
Tablo 6 (s.14, DocTamper-T, F): Q100/Q90/Q75 → **Ours 0.96 / 0.91 / 0.75 (ort. düşüş %4.6 / %20.2)**; DTD* 0.95 / 0.87 / 0.65 (%6.8 / %28.5). Tablo 7 (F1, Q90): Gaussian noise / blur / resize 1.5× / resize 0.75× / color jitter → **DTD 0.71 / 0.69 / 0.78 / 0.69 / 0.71; Ours 0.84 / 0.79 / 0.85 / 0.78 / 0.81**. Küçük bölge: motivasyon; "especially for small targets" iddiası tabakalı metrikle desteklenmiyor.

### Raporlanan metrikler (Tablo 1, s.10; IoU/P/R/F)
**Ours: DocTamper-T 0.90/0.87/0.84/0.86; FCD 0.88/0.93/0.91/0.92; SCD 0.75/0.81/0.82/0.81.** DTD* (yeniden eğitim): 0.84/0.81/0.77/0.79; 0.79/0.88/0.82/0.85; 0.68/0.75/0.76/0.75. ConvNeXtV2 (salt RGB): 0.86/0.82/0.79/0.81; 0.65/0.79/0.75/0.77; 0.67/0.74/0.76/0.75. Ablasyon Tablo 2: FPH+VEM+WFE tam model; FPH'siz 0.83/—/—/0.78 (T). Tablo 8 T-SROIE (retrain): P/R/F 0.993/0.993/0.993.
Not: DOCFORGE-BENCH zero-shot FFDN: DocTamper F1 0.736 (in-domain), T-SROIE **0.043** (AUC 0.904), RTM 0.025 (AUC 0.743) — kalibrasyon çöküşü (REPORTED, DOCFORGE-BENCH Tablo 4).

### Sınırlamalar
Yalnız DocTamper ailesinde değerlendirme; cross-dataset zero-shot yok; AI inpainting yok; kod/ckpt yayını makalede yok; eşik politikası açıkça belirtilmemiş (DOCFORGE-BENCH "optimized threshold" der → REPORTED).

### Gap ilişkisi
A: Orta–yüksek (DCT'ye kontrollü bağımlılık fikri Gap A için doğrudan ilgili; ama PNG/AI koşulunda ölçüm yok). B: Düşük. C: Orta–yüksek (5 bozulma + 3 JPEG seviyesi). D: Orta (eşik belirsiz; retrain-based "generalization").

---

## 4. ADCD-Net — ICCV 2025

### Kimlik (VERIFIED, CrossRef + arXiv PDF)
| Alan | Değer |
|---|---|
| Başlık | ADCD-Net: Robust Document Image Forgery Localization via Adaptive DCT Feature and Hierarchical Content Disentanglement |
| Yazarlar | Kahim Wong, Jicheng Zhou, Haiwei Wu, Yain-Whar Si, Jiantao Zhou (Univ. of Macau; UESTC) |
| Venue | ICCV 2025, s. 19280–19289 — **peer-reviewed**; arXiv:2507.16397 (22 Tem 2025) |
| DOI | 10.1109/ICCV51701.2025.01792 |
| Kod | github.com/KAHIMWONG/ACDC-Net — **MIT lisansı** (GitHub API `MIT`, VERIFIED); repo adı "ACDC-Net" (makale metninde de yer yer "ACDC-Net" yazımı var) |
| Checkpoint | README: Google Drive arşivi `ADCDNet.pth` + `docres.pkl` + önceden üretilmiş OCR maskeleri (VERIFIED README; dosya indirilmedi). ForensicHub "Doc Protocol" ile 7 test setinde değerlendirme talimatı var |

### Problem ve varsayımlar
DCT izleri etkili ama 8×8 blok hizası bozulunca (kırpma, yeniden boyutlandırma, piksel kayması) kırılgan; metin–arka plan (text-BG) yoğunluk kontrastı özellik uzayında pristine metni kurcalanmış metne yaklaştırır; **arka planın büyük ölçüde pristine olduğu** varsayımı (PPE). Girdi JPEG varsayılır (DCT katsayıları + kuantizasyon tablosu JPEG başlığından, §3).

### Forensic evidence ve spatial/frequency kullanımı (§3, VERIFIED)
- **Adaptive RGB-DCT encoder:** F_fuse^{i+1} = f_fuse(F_rgb^i + ŝ_aln·F_dct^i); ŝ_aln∈(0,1) bir sınıflandırma başlığından (hizalı=1 / hizasız=0; hizasız örnekler random resize/crop/shift n mod 8≠0 ile üretilir). MoE benzeri yönlendirme.
- **HCD:** her ölçekte MLP ile içerik/sahtecilik ayrıştırma; sahtecilik özelliği uzamsal karıştırılıp (shuffle) içerikle birleştirilerek [X, X_dct] yeniden inşa edilir (L1).
- **PPE:** CRAFT OCR ile arka plan maskesi → sahtecilik özelliklerinin BG-ortalaması = pristine prototip → kosinüs benzerlik haritası → ölçek/bias modülasyonu.
- Kayıp: CE (hizalama) + L1 (rekonstrüksiyon) + CE+Lovász (lokalizasyon) + SupCon tabanlı **FOCAL** within-image kontrastif kayıp. Restormer encoder/decoder (DocRes init), FPH DTD'den, 23M param.
- Kanıt türü: RGB + DCT (adaptif ağırlıklı) + OCR tabanlı metin/BG önsel + pristine gürültü prototipi.

### Veri ve split
DocTamper train; Test 30k / FCD 2k / SCD 18k; ablasyon 3 setten 3,000 örnek; FAR için Tianchi 2023 "Detecting Tampered Text in Images" 1,000 pristine görüntü. Unseen generator: **yok**; unseen manipulation: yok (DocTamper sentezi).

### Degradasyon ve küçük bölge (VERIFIED)
5 bozulma (Gaussian noise, Gaussian blur, downscale, JPEG, multi-JPEG; Şekil 4) + DCT hizalama bozucu 3 pertürbasyon (crop %2, shift 1 px, scale %98; Şekil 5). **DTD temiz F1 0.935 → %67.95 / %73.34 / %69.28 düşüş; ADCD-Net %25.74 / %22.66 / %25.39.** DTD en yüksek gürültü/bulanıklıkta %95.50 / %71.96 düşer. ADCD-Net tüm bozulmalarda ikinci en iyi TruFor'u ortalama **%20.79** aşar. Küçük bölge: FOCAL/SupCon ile sınıf dengesizliği ele alınır; alan-tabakalı metrik yok.

### Raporlanan metrikler (Tablo 1, s.6; multi-JPEG Q≥75; P/R/F1)
**ADCD-Net: Test 0.789/0.823/0.806; FCD 0.866/0.770/0.815; SCD 0.690/0.799/0.740; AVG F1 0.787** (23M). DTD 0.814/0.771/0.792; 0.849/0.786/0.816; 0.745/0.762/0.754; AVG 0.778. PS-Net AVG 0.785. TruFor (yeniden eğitilmiş) AVG 0.655; TruFor* (resmi ckpt) AVG 0.205; CAT-Net 0.702; MVSS-Net 0.423. Tablo 2 (pristine 1,000 görüntü): **ADCD F1 0.808, FAR 0.012**; DTD 0.779 / 0.058; TruFor 0.655 / 0.027; MA-Net 0.445 / 0.029. Tablo 4 ablasyon ortalama F1: baseline 0.602 → tam model 0.750. Tablo 3: ŝ_aln=0 (DCT kapalı) beş bozulmada ŝ_aln=1'den ortalama %8.37 iyi.
Not: DOCFORGE-BENCH zero-shot ADCD-Net: DocTamper 0.762, T-SROIE 0.176 (AUC 0.899), RTM 0.064, FSTS 0.320; ortalama F1 0.23 (REPORTED).

### Sınırlamalar
JPEG girdi varsayımı (PNG/AI-inpainted PNG'de DCT yolu anlamsız); OCR bağımlılığı (CRAFT hatası PPE'yi bozar); eğitim yine sentetik DocTamper; AI inpainting ve cross-generator değerlendirme yok; metin dışı kurcalama (logo/imza) kapsam dışı.

### Gap ilişkisi
A: **Çok yüksek** — DCT shortcut'ı açıkça ölçen ve bir "gating" ile hafifleten tek makale; ablasyon tasarımı bizim artifact-invariance deneylerimiz için şablon. B: Düşük. C: Yüksek (en kapsamlı degradasyon seti + FAR). D: Orta (optimal-eşik F1 raporu; DOCFORGE-BENCH'e göre REPORTED).

---

## 5. DOCFORGE-BENCH — arXiv:2603.01433

### Kimlik (VERIFIED)
| Alan | Değer |
|---|---|
| Başlık | arXiv v2 kaydı: "DOCFORGE-BENCH: A Comprehensive **0-shot** Benchmark for Document Forgery Detection and Analysis"; PDF başlığı "…A Comprehensive Benchmark…" (tutarsızlık) |
| Yazarlar | Zengqi Zhao*, Weidi Xia*, En Wei*, Yan Zhang*, Jane Mo*, Tiannan Zhang*, Yuanqin Dai*, Zexi Chen*, Yiran Tao*, Simiao Ren† (UNC, UCI, WashU, Scam.ai, Duke, UC Davis, NYU, Georgetown) |
| Tarih | v1 2 Mar 2026, v2 10 Mar 2026 (24 sayfa) — **Preprint** |
| DOI | yok |
| Kod | Makale: github.com/BensonRen/document_forgery_benchmark → **19 Eki 2026'da GitHub API 404** (repo yok/özel) → toolkit erişimi **UNKNOWN** |
| Checkpoint | Yok (üçüncü taraf resmi ağırlıklar) |

### Problem ve varsayımlar
14 yöntem × 8 belge seti, sıfır-atış (yalnız yayımlanmış ağırlıklar, domain uyarlaması yok). Varsayım: pratisyen etiketli belge verisine sahip değildir; Pixel-F1@0.5 "deployment" metriğidir; authentic görüntüler piksel metriklerinden NaN kuralıyla dışlanır (yani **authentic görüntülerdeki yanlış pozitifler ölçülmez**).

### Setler (Tablo 1, s.7; VERIFIED)
DocTamper 1,000 örneklem (seed 42) · T-SROIE 360 · RTM 1,000 (tampered-only) · Tampered-IC13 188 tampered (bbox→raster) · ReceiptForgery 35 forged (bbox→raster; n küçük) · MixTamper 1,000 (RGB çok-etiketli maske → herhangi kanal) · FSTS-1.5k 1,488 · FantasyID 2,773 tampered. Metrik için görüntüler ≤2 MP'ye indirgenir. Hiçbir set AI üretimli değil (§3.4, §9.2).

### Yöntem envanteri ve eğitim verisi kökeni (§5, VERIFIED)
Genel: TruFor, ManTraNet, MVSS-Net, CAT-Net, PSCC-Net, IML-ViT, SAFIRE (hiçbiri belge görmemiş). Belge: "DocTamper (model)", DTD (ForensicHub), FFDN, CAFTB-Net, TIFDM, ASCFormer (RTM'de eğitilmiş), ADCD-Net. DocTamper/DTD/FFDN/CAFTB/ADCD DocTamperV1 train (~120k) ile eğitilmiş → DocTamper sonuçları in-domain. TIFDM özel korpus; DocTamper sonucu (0.742) in-domain olabilir. **Dikkat:** makale "DocTamper (model)"i SegFormer/Mask2Former tabanlı ve DTD'den ayrı bir yöntem olarak listeler ve "Qu et al. 2023a, arXiv:2311.18818" diye atıf yapar; resmi qcf-568 deposu DTD'dir. Bu ayrımın dayanağı **UNKNOWN** (arXiv 2311.18818 kaydı hız sınırı nedeniyle doğrulanamadı).

### Raporlanan metrikler (VERIFIED)
**Tablo 3 (s.10) — 7 set ortalaması Mean F1 / σ / Mean AUC:** CAFTB-Net 0.34/0.33/0.85 · DocTamper 0.30/0.31/0.81 · ASCFormer 0.27/0.21/0.63 · TIFDM 0.24/0.22/0.83 · ADCD-Net 0.23/0.23/0.79 · FFDN 0.17/0.22/0.84 · DTD 0.16/0.21/0.83 · TruFor 0.34/0.23/0.83 · CAT-Net 0.33/0.25/0.82 · PSCC-Net 0.13/0.16/0.73 · SAFIRE 0.10/0.12/0.58 · IML-ViT 0.08/0.10/0.67 · ManTraNet 0.07/0.06/0.58 · MVSS-Net 0.03/0.05/0.63.
**Tablo 4 (s.20) — Pixel-F1@0.5 (DocTamper / T-SROIE / RTM / IC13 / Receipt / FantasyID / MixTamper / FSTS):** DocTamper 0.914/0.045/0.035/0.185/0.015/0.107/0.604/0.475 · DTD 0.682/0.021/0.024/0.140/0.003/0.036/0.106/0.242 · FFDN 0.736/0.043/0.025/0.156/0.019/0.068/0.072/0.219 · CAFTB 0.893/0.048/0.039/0.170/0.023/0.213/0.676/0.671 · TIFDM 0.742/0.046/0.042/0.164/0.019/0.300/0.312/0.320 · ASCFormer 0.180/0.779/0.265/0.022/0.260/0.222/0.314/0.133 · ADCD-Net 0.762/0.176/0.064/0.115/0.043/0.057/0.331/0.320. TruFor: T-SROIE 0.664, MixTamper 0.689, FSTS 0.522, FantasyID 0.296; CAT-Net: DocTamper 0.672, MixTamper 0.695.
**Pixel-AUC örnekleri (s.10–11):** FFDN T-SROIE 0.904, RTM 0.743; ADCD-Net T-SROIE 0.899; ASCFormer T-SROIE 0.924; TruFor tüm setlerde ≥0.66.
**Kalibrasyon deneyi (§9.1):** 8/112 çift; PSCC-Net/FSTS F1 0.104→0.319 (N=200; N=10 ile 0.255); PSCC/DocTamper 0.024→0.110; FFDN/T-SROIE 0.043→0.102; kalibre τ* 0.02–0.15. Taban oranları: medyan kurcalanmış piksel %0.27 (ReceiptForgery), %0.45 (MixTamper), %0.71 (DocTamper), %0.97 (T-SROIE), %2.88–4.17 (FantasyID, FSTS).

### Sınırlamalar
AI üretimli set yok (yazarlar "50 AI-edited görüntü ile pilot" öneriyor); kalibrasyon 8 çiftle sınırlı; Oracle-F1 görüntü başına (ulaşılamaz üst sınır); image-level AUROC ertelenmiş; authentic görüntüler dışlandığından FPR ölçülmüyor; MixTamper kaynağı "Anonymous 2024"; IML-ViT yalnız sol-üst 1024² kırpma; toolkit repo'su yayında değil.

### Gap ilişkisi
A: Orta (DCT-ağırlıklı modellerin domain dışı çöküşü shortcut hipotezini destekler, ama nedensel analiz yok). B: Düşük–orta (generatör yok; dataset-arası genelleme var). C: **Çok yüksek** (küçük-bölge → kalibrasyon çöküşü mekanizması nicel). D: **Çok yüksek** (eşik politikası, Oracle vs fixed F1, bbox rasterizasyonu, NaN kuralı — değerlendirme güvenilirliği tartışmasının en kapsamlı kaynağı).

---

## 6. "When the Forger Is the Judge" / AIForge-Doc v2 — arXiv:2604.25213

**Not:** Görevde "AIForge-Doc v2 associated research" olarak verilen bu kimlik gerçekten AIForge-Doc v2 veri setini tanıtan makaledir; başlık farklıdır: *When the Forger Is the Judge: GPT-Image-2 Cannot Recognize Its Own Faked Documents*.

### Kimlik (VERIFIED)
| Alan | Değer |
|---|---|
| Yazarlar | Jiaqi Wu*, Yuchen Zhou*, Dennis Tsang Ng, Xingyu Shen, Kidus Zewde, Ankit Raj, Tommy Duong, Simiao Ren† |
| Tarih | 28 Nis 2026, v1 (13 sayfa) — **Preprint** |
| DOI | yok |
| Veri | HF `Scam-AI/AIForge-Doc-v2` — gated, kart lisansı **cc-by-nc-sa-4.0**; makale §7 "Dataset (CC-BY-4.0) at scam.ai/en/research" → **lisans çelişkisi** |
| Kod | "We release the dataset, pipeline, four-judge protocol, and calibration sets" — URL yalnız scam.ai; GitHub repo UNKNOWN |
| Checkpoint | yok |

### Problem ve varsayımlar
(i) GPT-Image-2 ile üretilen belge sahtelerini insan, genel dedektör (TruFor), belge dedektörü (DocTamper) ve **generatörün kendisi** ayırt edebilir mi? (ii) Dedektör çöküşü AI inpainting'e mi özgü, yoksa alan kaymasına mı? Varsayım: v1 spec'lerinin birebir yeniden kullanımı farkı generatöre atfetmeyi sağlar (§3.1). Ölçüm çalışması; yöntem önermez (App. A).

### Veri üretimi (§3, VERIFIED)
4,062 v1 spec'i (not: v1 makalesi 4,061 / XFUND 419 der; burada 4,062 / 420) → **3,066** üretildi (CORD 983, WildReceipt 1,336, SROIE 329, XFUND 418; Tablo 1 s.3). %24.5 deterministik red (≈%94 invalidAspectRatio [1:3,3:1], ≈%4 yükseklik sınırı). Ek ~%10 "providerInternalError" = belgesiz güvenlik filtresi, CORD finansal tutar düzenlemelerinde yoğun (§3.4). Pipeline: bbox %50 genişlet (min **100 px**), API boyut kısıtına "aspect-preserving size snap" (ör. 331×183→1104×608) sonra LANCZOS küçültme, **3 px yeşil kontur** işaretçi (v1'in "yarı saydam kırmızı dolgusu" renk sızdırıyormuş — v1 makalesi ise "3-pixel bright-green rectangle" der → tutarsızlık), beş maddeli prompt, **PNG metadata temizliği hem sahte hem authentic'e** uygulanır. Kalite: semantic check + yazar görsel kontrolü; PaddleOCR devre dışı (NumPy 2.0 uyumsuzluğu). Maske: v1 ile aynı bbox maskesi, medyan %0.92. HF: train 2,453 / test 613 image+mask; authentic'ler yalnız Google Drive "full package"ta.

### Değerlendirme protokolü (§4, VERIFIED)
Hesaplamalı yargıçlar **tüm eşli kümede** (3,066 + 3,066 = 6,132; train+test birlikte); insanlar 30 görüntülük tabakalı alt kümede 2AFC (CanUSpotAI.com, N=120, n=365). DocTamper görüntü skoru = 512×512, JPEG q=75 girdisinde FAKE öngörülen piksel oranı. Self-judge: `openai/gpt-5.4-image-2`, prompt "Is this image AI-edited? Answer with one word: yes or no." İkili çıktı → "AUC" aslında balanced accuracy (dipnot 1). **Piksel düzeyi metrik yok.**

### Raporlanan metrikler (VERIFIED)
**Tablo 2 (s.6):** Human 2AFC **0.501** [0.450,0.553]; TruFor AUC **0.599** [0.592,0.606] (TPR .594 / TNR .593 @0.5); DocTamper **0.585** [0.571,0.600]; GPT-Image-2 self-judge **0.532** [0.525,0.537] (TPR .153 / TNR .910).
**Tablo 5 (s.10) kaynak bazlı AUC (CORD / WildReceipt / SROIE / XFUND):** TruFor 0.498 / 0.791 / 0.513 / 0.500; DocTamper 0.524 / 0.654 / 0.710 / 0.500; self 0.562 / 0.510 / 0.512 / 0.538.
**Tablo 4 (s.8) kalibrasyon setleri:** TruFor cross-camera splicing (50 çift, PNG) AUC 0.962 [0.909,1.000] vs v2 0.599 → gap 0.363; DocTamper cross-document OCR-token splicing + 2-pass JPEG (Q85→Q75; 100 çift) 0.852 [0.794,0.904] vs 0.585 → gap 0.267.
**Tablo 3 (s.7) prompt ablasyonu (n=50+50):** P0 0.522, P1 CoT 0.453, P2 forensic role-play 0.586 (FPR %20.8), P3 0.538, P4 0.529. **Tablo 6:** AMBIG doldurma politikalarıyla 0.500–0.568. ρ(TruFor, DocTamper)=0.31; ρ(self, TruFor)=−0.19.

### Train/test, unseen generator (VERIFIED)
Fine-tune yok; v1–v2 spec eşleşmesi 3,062/3,066 (v1 karşılığı 2,729 "Gemini-nano", 333 "qwen-inpaint"). §3.3 "per-spec cross-generator comparison reported in §5" der, ancak §5'te böyle bir tablo yok → bu analiz makalede **eksik**. Cross-generator eğitim protokolü yok.

### Degradasyon, küçük bölge
Yok / yalnız istatistik. Authentic görüntü asimetrisi (sahte: GPT-Image-2 render + PNG yeniden kodlama; authentic: hayır) açıkça kabul edilir; simetrik yeniden kodlama yapılmamıştır (App. B).

### Sınırlamalar
İnsan deneyi 30 görüntü/365 oy (anonim web ziyaretçileri); self-judge tek deneme, ikili skor; piksel metriği yok; tek generatör; çoklu karşılaştırma düzeltmesi yok (App. D); lisans ve generatör adlandırma tutarsızlıkları; 29 Ağu 2026 sonrası GPT-Image-2 güvenlik politikası değişirse pipeline yeniden üretilemez.

### Gap ilişkisi
A: Yüksek (PRNU ve DCT kanallarının AI inpainting'de neden sustuğunu kalibrasyon setleriyle izole eder; asimetri uyarısı shortcut riskini adlandırır). B: **Çok yüksek** (v1 ile spec-eşli ikinci generatör → cross-generator protokolünün ham maddesi). C: Orta (yalnız alan istatistiği). D: Yüksek (bbox maskesi, AUC≠balanced accuracy, train+test birlikte değerlendirme).

---

## Yan Yana Karşılaştırma

| Boyut | AIForge-Doc v1 | DocTamper/DTD | FFDN | ADCD-Net | DOCFORGE-BENCH | AIForge-Doc v2 |
|---|---|---|---|---|---|---|
| Tür / yıl / venue | Benchmark · 2026 · arXiv (preprint) | Dataset+model · CVPR 2023 | Model · ECCV 2024 | Model · ICCV 2025 | Benchmark · 2026 · arXiv (preprint) | Dataset+ölçüm · 2026 · arXiv (preprint) |
| Kurcalama türü | Diffusion/çok-modlu inpainting (Gemini 2.5 Flash Image, Ideogram v2 Edit) | Copy-move, splicing, dizgi-generation (sentetik) | DocTamper | DocTamper | 8 set; AI üretimli yok | GPT-Image-2 inpainting |
| Boyut / split | 4,061 sahte (+4,061 authentic); 3,249/812 | 170k; 120k/30k + FCD 2k + SCD 18k | DocTamper | DocTamper | 14 yöntem × 8 set; ≤1,000 örnek/set | 3,066 sahte (+3,066 authentic); 2,453/613 |
| Forensic evidence | — (baseline: TruFor NoisePrint++/RGB, DTD RGB+DCT, GPT-4o) | RGB + DCT (JPEG BAG, kuantizasyon tablosu) | RGB (baskın) + DCT + Haar wavelet özellik ayrıştırma | RGB + adaptif DCT + OCR metin/BG önseli + pristine prototip + kontrastif | — (14 yöntem) | — (TruFor, DocTamper, insan, self-judge) |
| Spatial/frequency | n/a | Evet (FPH, scSE füzyon) | Evet (VEM zero-init, WFE) | Evet (ŝ_aln gating) | n/a | n/a |
| Train/test bağımsızlığı | Zero-shot; kaynak/araç tabakalı split | Cross-domain FCD/SCD | FCD/SCD; T-SROIE retrain | FCD/SCD; Tianchi pristine | Zero-shot; in-domain hücreler işaretli | Zero-shot; v1 spec-eşli |
| Unseen generator testi | Per-tool kırılım (TruFor) | Yok | Yok | Yok | Yok | Tek generatör; cross-gen tablo eksik |
| Maske | **bbox maskesi** (0/255) | Piksel-düzeyi metin bölgesi (ayrıntı UNKNOWN) | DocTamper | DocTamper | Karışık; bbox rasterize | **bbox maskesi** |
| Degradasyon | Yok | JPEG Q75–100 | JPEG + noise/blur/resize/jitter | 5 bozulma + crop/shift/scale | Yok (ham setler) | Yok |
| Küçük bölge | Medyan %0.92 | Motivasyon | Motivasyon | FOCAL/SupCon | Taban oranı %0.27–4.17 → kalibrasyon analizi | Medyan %0.92 |
| Ana sonuç | TruFor AUC 0.751, DocTamper 0.563 (IoU 0.020), GPT-4o 0.509 | DTD Test F 0.792 / IoU 0.828; T-SROIE F 0.9927 | FCD F 0.92 (DTD* 0.85); Q75 F 0.75 | AVG F1 0.787; +%20.79 dayanıklılık; FAR 0.012 | Hiçbir yöntem ≥6/8 sette F1≥0.3; CAFTB=TruFor mean F1 0.34 | İnsan 0.501; TruFor 0.599; DocTamper 0.585; self 0.532; kalibrasyon gap 0.27–0.36 |
| Kod / ckpt / lisans | Kod UNKNOWN; veri HF gated cc-by-nc-sa-4.0 | Inference kod+model; eğitim kodu e-posta; veri non-commercial şifreli; LICENSE yok | Resmi kod yok; ForensicHub uygulaması (CC-BY-4.0) | GitHub MIT; ckpt Google Drive | Repo 404 (UNKNOWN) | Veri HF gated cc-by-nc-sa-4.0 (makale CC-BY-4.0 der) |
| Gap A / B / C / D | Y / Y / Y / Y | Y / D-O / O / O | O-Y / D / O-Y / O | ÇY / D / Y / O | O / D-O / ÇY / ÇY | Y / ÇY / O / Y |

(Y = yüksek, O = orta, D = düşük, ÇY = çok yüksek ilgi.)

## Proje İçin Çıkarımlar
1. **Artifact-invariance deneyi için şablon ADCD-Net'tir:** DCT yolunu kapat/aç (ŝ_aln∈{0,1}) ve hizalama bozucu pertürbasyonlar (crop %2, shift 1 px, scale %98). AIForge-Doc PNG olduğundan DCT yolu baştan anlamsız; bunun yerine paste-back sınırındaki yeniden-örnekleme izi ve 2× LANCZOS döngüsü shortcut adayıdır (HYPOTHESIS; authentic görüntüleri aynı döngüden geçirerek kontrol edilmeli).
2. **Cross-generator protokolü:** v1 Gemini (3,639) ile eğit → v2 GPT-Image-2 (3,066, aynı spec'ler) ile test et; tersi; ve v1 "Ideogram/qwen-inpaint" alt kümesi (422) held-out. Önce `assigned_tool` alanının gerçek anlamı metadata.jsonl'dan teyit edilmeli.
3. **Küçük bölge / kalibrasyon:** DOCFORGE-BENCH'in Pixel-F1@0.5 + Pixel-AUC + Oracle-F1 üçlüsünü ve N=10 kalibrasyon protokolünü aynen benimse; alan boyutuna göre tabakalı raporla (<%0.5, %0.5–1, >%1).
4. **Annotation güvenilirliği:** bbox maskesi ile piksel-değişim maskesi (|forged − authentic| > ε) arasındaki farkı kendi audit'imizde ölç; IoU'nun bbox maskesine göre şişip şişmediğini raporla. Authentic görüntülerdeki FPR'yi DOCFORGE-BENCH'in aksine ölç.
5. **Alıntı hijyeni:** AIForge-Doc'un DocTamper için verdiği "AUC 0.98 / IoU 0.71" değerlerini ve "DFL/NFC" tanımını kullanma; DTD için CVPR Tablo 3/4 değerlerini (IoU 0.828 / F 0.792) kullan.

# PHASE 1 — Bilimsel Literatür Taraması, Gap Doğrulama ve Araştırma Yönü Seçimi

**Proje:** AIC515 Görüntü İşleme — Dönem Projesi (Topkapı Üniversitesi, Prof. Dr. Onur Osman)
**Araştırma alanı:** AI-Generated Document Forgery Detection and Tampering Localization
**Birincil veri seti:** AIForge-Doc v1 (Hugging Face `Scam-AI/AIForge-Doc-v1`, rev. `b8db5654`; arXiv:2602.20569)
**Research cutoff:** 17 Ekim 2026 · **Rapor tarihi:** 20 Ekim 2026
**Kanıt sınıfları:** **VERIFIED** (birincil kaynaktan / yerel veriden doğrulandı) · **REPORTED** (kaynak raporladı, bağımsız doğrulanmadı) · **HYPOTHESIS** (test edilecek varsayım) · **UNKNOWN** (doğrulanamadı)

Bu rapor şu alt belgelerle birlikte okunmalıdır (hepsi `Vize Proje/` altında):
`literature/seed_papers_review.md` (6 seed makale, tam metin), `literature/literature_matrix.{csv,xlsx,md}` (38 çalışma), `literature/search_strategy.md`, `literature/references.bib`, `dataset_audit/audit_report.md` (+ `audit_stats.json`, `splits.json`, figürler), `baselines/README.md` (küçük veri baseline sonuçları).

---

## A. Executive Research Summary

**Amaç.** Üretken yapay zekâ (diffusion / çok-modlu inpainting API'leri) ile finansal belge görüntülerinde (fiş, form) tek bir alanın değiştirilmesini piksel düzeyinde lokalize eden yöntemleri; genelleme, dayanıklılık, küçük-bölge doğruluğu, shortcut learning ve değerlendirme güvenilirliği açısından eleştirel biçimde incelemek ve deneysel olarak sınanabilir tek bir araştırma yönü seçmek.

**Literatürün durumu (38 çalışma; 27 tam metin).**
1. Belge-özel state-of-the-art (DTD/DocTamper CVPR 2023, FFDN ECCV 2024, ADCD-Net ICCV 2025, DiffNet 2026) tamamen **DocTamper sentetik eğitim kümesi** ve **RGB + 8×8 blok DCT (JPEG) kanıtı** üzerine kuruludur [VERIFIED]. Bu kanıt kanalı, AIForge-Doc'un **PNG** ve "tam yeniden kodlanmış" çıktılarında yapısal olarak sessizdir.
2. Benchmark çalışmaları bu yöntemlerin sıfır-atış koşulunda çöktüğünü göstermektedir: AIForge-Doc v1'de TruFor image-AUC 0.751 / pixel-IoU 0.358, DocTamper 0.563 / 0.020, GPT-4o 0.509 (Tablo 2, s.11) [VERIFIED]; AIForge-Doc v2'de (GPT-Image-2) insan 2AFC 0.501, TruFor 0.599, DocTamper 0.585 [VERIFIED]; DOCFORGE-BENCH'te 14 yöntemin hiçbiri 8 setin ≥6'sında Pixel-F1 ≥ 0.3'e ulaşamamıştır ve yüksek Pixel-AUC ile sıfıra yakın Pixel-F1 birlikte görülür (τ = 0.5 kalibrasyon çöküşü; kurcalanmış piksel taban oranı %0.27–4.17) [VERIFIED].
3. Shortcut learning kanıtı güçlüdür: ADCD-Net, DTD'nin 1 px kaydırma / %2 kırpma / %98 ölçekleme ile F1'inin %68–73 düştüğünü ölçer [VERIFIED]; DocQT yalnızca JPEG kuantizasyon tablosu uyumsuzluğunun FFDN'de 14.5 F1 puanı fark yarattığını gösterir [VERIFIED]; GenD (WACV 2026) eşleşmemiş gerçek–sahte eğitimde AUROC'un 90.0 → 56.8'e düştüğünü raporlar [VERIFIED]. **Hiçbir çalışma AI-inpainting belgelerinde paste-back / bbox-maske kaynaklı shortcut'ı ölçmemiştir** [UNKNOWN → gap].
4. Belge alanında AI-inpainting sahteciliği için **eğitilmiş ve yayımlanmış bir lokalizasyon yöntemi taramada bulunamadı** [UNKNOWN]; mevcut tüm sonuçlar zero-shot'tır.

**Veri seti audit'inin kritik bulguları (yerel, 12.187 dosya / 9.98 GB, kodla hesaplandı) [VERIFIED-LOCAL].**
- Maskeler %100 tek-dikdörtgen **alan bbox maskesidir**; eşleşen authentic görüntüyle karşılaştırıldığında bbox piksellerinin **medyan yalnızca %49'u** gerçekten değişmektedir (IoU(|forged−authentic|>8, maske) medyan 0.48). Makalenin "pixel-precise" ifadesi bbox-precise anlamındadır.
- Çiftlerin **%33.5'inde bbox dışında da değişim** vardır (Gemini %35.9 vs Ideogram %11.9); makalenin "yalnızca alan bbox'ı değişir" iddiası çürümüştür. Ayrıca çiftlerin ~%49'unda tüm görüntüye yayılan ±1–3 düzeyinde yeniden kodlama izi vardır.
- `TestingSet/authentic` test sahtelerinin aynası **değildir**: 812 dosyanın yalnızca 86'sı bir test sahtesinin kaynağıdır; 254'ü train-authentic ile bayt-özdeş, 535'i pHash ≤ 10 kopyasıdır; SROIE kaynaklı test-authentic yoktur → **bu veri setiyle eğitilen her model için resmî test setinin authentic yarısı image-level değerlendirmede geçersizdir.**
- Test sahtelerinin **%8.9'u (72/812)** bir train kaynak belgesini paylaşır (leakage); 12 metadata satırında bbox karışıktır; lisans beyanları çelişkilidir (cc-by-nc-sa-4.0 vs CC BY 4.0).
- Generator ↔ kaynak **confound**: WildReceipt'te hiç Ideogram yok, XFUND'da %45; Ideogram toplam 422 (test 85).

**Geçici bilimsel değerlendirme.** Dört başlangıç gap'inden ikisi (A: artifact/shortcut, D: annotation/evaluation güvenilirliği) hem literatürle hem yerel audit ile desteklenen **VALIDATED CANDIDATE**; B (cross-generator) **PARTIALLY SUPPORTED** (gerçek bir açık ama v1 tek başına confound'ludur); C (küçük bölge / dayanıklılık) yöntem olarak **LIKELY ALREADY COVERED**, değerlendirme ekseni olarak kısmen açıktır. Seçilen yön: **"Box mask'ten ne öğrenilir?" — eşleşmiş authentic'lerden türetilen piksel-değişim ground-truth'u ile bbox-GT'yi karşılaştıran, shortcut probelarıyla desteklenen annotation-aware lokalizasyon çalışması** (Bölüm H). Karar: **CONDITIONAL GO** — GO'ya dönüşmesi için gereken asgarî kanıtlar Bölüm H'de listelenmiştir; küçük-veri baseline sonuçları (Bölüm J) bu koşulların ilkini test eder.

**Kritik riskler.** (i) Piksel-değişim GT'si yalnızca train tarafında türetilebilir (test'te 723/812 sahtenin authentic eşi yok) → test değerlendirmesi bbox-GT + temiz-alt-küme ile sınırlı; (ii) küçük veri (3.249) ve ciddi sınıf dengesizliği (≈1:100 piksel); (iii) generator×kaynak confound cross-generator iddialarını zayıflatır; (iv) veri seti %90 fiş + form olduğundan "finansal belge" genellemesi (fatura, ekstre) savunulamaz; (v) preprint ağırlıklı literatür (AIForge-Doc v1/v2, DOCFORGE-BENCH hakemli değil).

---

## B. Source Verification & Evidence Quality

### B.1 Seed makaleler (6/6 tam metin; kimlik CrossRef/arXiv API ile doğrulandı)

| # | Çalışma | Venue / durum | Kimlik | Erişim | Kod / checkpoint / lisans |
|---|---|---|---|---|---|
| 1 | AIForge-Doc: A Benchmark for Detecting AI-Forged Tampering in Financial and Form Documents (Wu, Zhou, Xu, …, Ren, 2026) | **arXiv preprint** (hakemli venue bulunamadı, 19 Eki 2026) | arXiv:2602.20569 | full-text (arXiv PDF) | Kod URL'si yok (HF kartında `YOUR_ORG` yer tutucu) → UNKNOWN; veri HF gated; lisans çelişkili (cc-by-nc-sa-4.0 / CC BY 4.0) |
| 2 | Towards Robust Tampered Text Detection in Document Image: New Dataset and New Solution (Qu et al., CVPR 2023, s. 5937–5946) | **Peer-reviewed** | DOI 10.1109/CVPR52729.2023.00575 | full-text (CVF Open Access PDF) | github.com/qcf-568/DocTamper: inference kodu + modeller; eğitim kodu e-posta ile; veri non-commercial, şifreli; LICENSE dosyası yok |
| 3 | Enhancing Tampered Text Detection through Frequency Feature Fusion and Decomposition (Chen et al., ECCV 2024, LNCS 15091 s. 200–217) | **Peer-reviewed** | DOI 10.1007/978-3-031-73414-4_12 | full-text (ECVA PDF) | Resmî repo makalede yok; ForensicHub (CC-BY-4.0) içinde uygulama var; resmî checkpoint UNKNOWN |
| 4 | ADCD-Net: Robust Document Image Forgery Localization via Adaptive DCT Feature and Hierarchical Content Disentanglement (Wong et al., ICCV 2025, s. 19280–19289) | **Peer-reviewed** | DOI 10.1109/ICCV51701.2025.01792; arXiv:2507.16397 | full-text (arXiv PDF) | github.com/KAHIMWONG/ACDC-Net — **MIT**; checkpoint Google Drive (README'de; dosya indirilmedi) |
| 5 | DOCFORGE-BENCH: A Comprehensive (0-shot) Benchmark for Document Forgery Detection and Analysis (Zhao et al., 2026) | **arXiv preprint** (v2, 10 Mar 2026) | arXiv:2603.01433 | full-text (arXiv PDF) | Makaledeki repo (BensonRen/document_forgery_benchmark) 19 Eki 2026'da **404** → toolkit UNKNOWN |
| 6 | When the Forger Is the Judge: GPT-Image-2 Cannot Recognize Its Own Faked Documents (Wu, Zhou, Ng, …, Ren, 2026) — AIForge-Doc **v2**'yi tanıtan makale | **arXiv preprint** (28 Nis 2026) | arXiv:2604.25213 | full-text (arXiv PDF) | Veri HF gated (`Scam-AI/AIForge-Doc-v2`, kart cc-by-nc-sa-4.0; makale CC-BY-4.0 der); authentic'ler yalnızca Google Drive "full package"ta; kod repo UNKNOWN |

Not: Başlangıç notunda "AIForge-Doc v2 Associated Research" olarak verilen arXiv:2604.25213 kimliğinin başlığı farklıdır; içerik doğrulandıktan sonra v2 veri setinin birincil makalesi olduğu teyit edilmiştir.

### B.2 Kaynak kalitesi dağılımı (literatür matrisi, 38 çalışma)
- Hakemli: 21 · Preprint: 17 (AI-inpainting belge sahteciliği alt alanının tamamı 2026 preprint'lerinden oluşur).
- Erişim: 27 tam metin · 11 abstract/metadata (kapalı erişimli FFDN ve DocTamper tam metinleri seed inceleme hattında ayrıca okunmuştur; matris hattında abstract düzeyinde kalmıştır).
- Doğrulanamayan DOI/kimlik: 0 (48 adaydan birincil kaynağı doğrulanamayan CTAD, TTD, CatmullRom-Net, DocMMIR/DocLoc, T-SROIE orijinal makalesi matrise **alınmamıştır**).
- OpenAlex yalnızca atıf sayısı zenginleştirmede kullanılabildi (36/38); sistematik ileri-atıf genişlemesi yapılmadı (sınırlama).

### B.3 Önemli kaynak belirsizlikleri ve çelişkiler
1. AIForge-Doc v1 makalesinin DocTamper için alıntıladığı "in-distribution AUC 0.98 / IoU 0.71" değerleri ve "DFL/NFC modülleri" tanımı **CVPR 2023 makalesinde yoktur** (DTD Test IoU 0.828 / F 0.792; AUC raporlanmaz) → bu değerler kullanılmamalıdır [VERIFIED].
2. DOCFORGE-BENCH "DocTamper (model)" ile DTD'yi ayrı yöntemler olarak listeler ve arXiv:2311.18818'e atıf yapar; bu kaydın içeriği doğrulanamadı → ayrım UNKNOWN.
3. AIForge-Doc v2'deki "AUC" değerleri ikili çıktılı yargıçlar için aslında balanced accuracy'dir (makalenin kendi dipnotu) [VERIFIED].
4. Generator kod adları (`gemini-nano`, `qwen-inpaint`) ile kartın beyan ettiği ticari API'ler (Gemini 2.5 Flash Image, Ideogram v2 Edit) arasındaki eşleme yalnızca kart beyanına dayanır; makalede "Qwen" geçmez → gerçek API **REPORTED**.
5. Context padding: README "min 100 px" ↔ makale §3.3 "min 150 px"; işaretçi: v1 "3 px yeşil dikdörtgen" ↔ v2 makalesi v1 için "yarı saydam kırmızı dolgu" der.
6. Lisans: cc-by-nc-sa-4.0 (front-matter, gating) ↔ CC BY 4.0 (`dataset_info.json`, README bağlantısı). En kısıtlayıcı okuma (BY-NC-SA + gating) esas alınmalıdır; görüntüler yeniden dağıtılmamalı, yalnızca id listeleri paylaşılmalıdır.

---

## C. State-of-the-Art Literature Matrix

Tam matris (15 zorunlu alan + evidence_class/code/license/openalex_cited_by) `literature/literature_matrix.csv` ve `.xlsx` dosyalarındadır; okunabilir sürüm `literature/literature_matrix.md`. Aşağıda özet tablo verilmiştir (sonuç değerleri yalnızca birincil metinden okunanlardır; "n.e." = not extracted).

| ID | Çalışma (yıl, venue, durum) | Problem | Veri | Kanıt kanalı / teknik | Mimari | Ana raporlanan sonuç | Gap ilgisi |
|---|---|---|---|---|---|---|---|
| S01 | AIForge-Doc (2026, arXiv) | AI-inpainting alan sahteciliği benchmark'ı | 4.061 sahte + authentic; CORD/WildReceipt/SROIE/XFUND; Gemini 2.5 Flash Image %89.6 / Ideogram v2 %10.4 | Zero-shot TruFor (NoisePrint++), DTD (DCT), GPT-4o | — | TruFor AUC 0.751 / IoU 0.358; DTD 0.563 / 0.020; GPT-4o 0.509 | A,B,C,D |
| S02 | When the Forger Is the Judge / AIForge-Doc v2 (2026, arXiv) | GPT-Image-2 sahtelerinin tespiti; self-judge | 3.066 sahte (v1 spec-eşli) | TruFor, DocTamper, insan 2AFC, self-judge | — | İnsan 0.501; TruFor 0.599; DocTamper 0.585; self 0.532; kalibrasyon seti açığı 0.27–0.36 AUC | A,B,D |
| S03 | DOCFORGE-BENCH (2026, arXiv) | 14 yöntem × 8 set zero-shot | DocTamper, T-SROIE, RTM, IC13, Receipt, FantasyID, MixTamper, FSTS | Pixel-AUC vs Pixel-F1@0.5 vs Oracle-F1 | 14 model | Hiçbiri ≥6/8 sette F1≥0.3; CAFTB=TruFor mean F1 0.34; N=10 kalibrasyon açığın %39–55'ini kapatır | C,D |
| S04 | DocTamper / DTD (CVPR 2023) | Fotoğraflanmış belgede metin kurcalama segmentasyonu | 170k sentetik (120k train, 30k test, FCD 2k, SCD 18k) | RGB + DCT + kuantizasyon tablosu (FPH), CLTD müfredatı | Swin + MID | Test IoU 0.828 / F 0.792; FCD F 0.816; SCD F 0.754 | A,C |
| S05 | FFDN (ECCV 2024) | Frekansın RGB'yi bozmadan füzyonu | DocTamper | VEM (zero-init), Haar WFE | ConvNeXtV2-base + FPN | DocTamper-T F 0.86; FCD 0.92; SCD 0.81; Q75'te F 0.75 | A,C |
| S06 | ADCD-Net (ICCV 2025) | DCT hizalama bozulunca dayanıklılık | DocTamper + Tianchi pristine | Adaptif RGB-DCT gating, HCD, PPE, FOCAL | Restormer, 23M | AVG F1 0.787; crop/shift/scale'de DTD −%68–73 vs ADCD −%23–26; FAR 0.012 | A,C |
| S07 | Tampered-IC13 (ECCV 2022) | Tampered scene text detection | Tampered-IC13 | Doku gerçeklik dalı | Scene-text detektörü | n.e. | B,D |
| S08 | ICDAR 2023 DTT competition | Metin kurcalama sınıflandırma + piksel tespit | TTI 11.385 | Karışık | Karışık | n.e. | D |
| S09 | OSTF / DAF (AAAI 2025) | Görülmemiş metin-düzenleme modellerine açık-küme genelleme | OSTF (8 editör), IC13 | Texture-jitter pretext | Cascade R-CNN | mF 88.62 (seen) → 75.85 (unseen) | B,A |
| S10 | RTM (Pattern Recognition 2025) | Gerçek (manuel) metin kurcalama | RTM 14.250 | RGB + frekans/gürültü füzyon, kontrastif | CA Hub + GCNF | IC13 F1 98.97; RTM'de mevcut yöntemler zayıf | B,D |
| S11 | TextSleuth (2024, arXiv) | Açıklanabilir kurcalama tespiti | ETTD | LMM, iki aşamalı zoom | Qwen2-VL-7B | Ort. 97.2 vs 90.7; JPEG q50 96.3 | D,C |
| S12 | CTP-Net (2023, arXiv) | Karakter dokusu ile lokalizasyon | FCTM, DocTamper | OCR-güdümlü doku akışı | İki akışlı enc-dec | F1 0.45 / IoU 0.31 | C |
| S13 | Dong et al. (IEEE TCE 2024) | Tek-karakter kurcalama | n.e. | İz güçlendirme + çok ölçekli dikkat | — | n.e. (abstract) | C |
| S14 | DITL2 (ACM MM 2025) | Doğal görüntüden belgeye invariance transfer | n.e. | İki aşamalı transfer | — | n.e. (abstract) | B |
| S15 | DocQT (2026, arXiv) | JPEG kuantizasyon tablosu uyumsuzluğu | DocTamper, T-SROIE, Find-it | QT çeşitlendirme | FFDN vs Mesorch | FFDN +14.5 F1 (Real-QT) | A,C |
| S16 | DiffNet (2606.22285, arXiv) | Verimli cross-domain belge lokalizasyonu | ForensicHub Doc protokolü | Birleşik DCT-QT gömme | Hafif | Avg F1 0.515 vs ADCD 0.450 vs FFDN 0.301; RTM 0.173 | A,B |
| S17 | FSTS (2511.12658, arXiv) | Sentetik→gerçek açığı | FSTS, FSTS-1.5k | Fourier-serisi parametrik sentez | Çeşitli | FSTS eğitimi gerçek setlerde daha iyi (tablo n.e.) | B,D |
| S18 | Receipt Dataset / Find-it (ICDAR 2023 WS) | Gerçekçi fiş sahteciliği verisi | 988 SROIE + 163 sahte | — | — | n.e. | B |
| S19 | GPT4o-Receipt (2026, arXiv) | Tam-AI fiş tespiti (insan vs MLLM) | 1.235 fiş | Semantik/aritmetik tutarlılık | 5 MLLM | Claude Sonnet 4 F1 0.975; insan 0.852 | A |
| S20 | MLLM document manipulation (2025, arXiv) | Zero-shot MLLM sahtecilik tespiti | Find-it Again | Prompting; OH-JPEG baseline | MLLM'ler | OH-JPEG F1 0.80 | A |
| S21 | TextFake (2026, arXiv) | Metin-yoğun AIGI tespiti | 20k, 28 dil | Düşük seviye dedektör tanıları | 14 dedektör + 3 VLM | Ort. doğruluk 86.7 → 51.8 ("text density curse") | A,B |
| G01 | TruFor (CVPR 2023) | Genel sahtecilik tespit+lokalizasyon | PSCC sentetik; 8 test seti | Noiseprint++ + RGB, güven haritası | SegFormer | Avg pixel-F1 0.785 (best) / 0.696 (fixed); image AUC 0.857 | A |
| G02 | CAT-Net v2 (IJCV 2022) | JPEG artefaktıyla splicing | Sentetik JPEG | DCT hacmi + QT | HRNet + DCT akışı | GRIP p-F1 76.45 | A |
| G03 | MVSS-Net (TPAMI 2023) | Gürültü + kenar görünümleri | CASIAv2/DEFACTO | BayarConv + Sobel | Çift dal ResNet-50 | Com-F1 0.711 (Columbia) | A,D |
| G04 | PSCC-Net (TCSVT 2022) | Kabadan inceye lokalizasyon | Sentetik | Spatio-channel korelasyon | HRNet | n.e. (sütunlar belirsiz) | C |
| G05 | IML-ViT (2023, arXiv) | Yüksek çözünürlüklü ViT | CASIAv2 | Kenar denetimi, 1024 px | ViT-B MAE | Avg F1 0.480 vs NCL 0.396 | C |
| G06 | Mesorch (AAAI 2025) | Mikro+makro füzyon | CAT-Net protokolü | CNN + Transformer | ConvNeXt + SegFormer | Avg F1 0.677 / 0.726 (permute) | C |
| G07 | SAFIRE (AAAI 2025) | Kaynak-bölümleme | 5 set | Prompt tabanlı SAM | SAM | Avg F1-fixed 62.6 vs TruFor 56.3 | A |
| G08 | Omni-IML (2024, arXiv) | Doğal+belge+yüz birleşik | Karışık | Modal gate (RGB/frekans) | Gated enc + DWD | n.e. | A,D |
| I01 | TGIF (WIFS 2024) | Diffusion inpainting lokalizasyonu | 75k (SD2, SDXL, Firefly) | IFL yöntemleri | Çeşitli | Tam yeniden üretilmiş inpainting'de PSCC F1 0.05, MVSS++ ≤0.09 | A,B |
| I02 | COCO-Inpaint (2025, arXiv) | Inpainting benchmark, cross-model | 238k | IMDL yöntemleri | Çeşitli | IML-ViT in-domain 94.4/97.4/86.6 → en zor split 55.0/75.4/40.2 | B |
| E01 | Geirhos et al. (Nat. Mach. Intell. 2020) | Shortcut learning çerçevesi | — | — | — | Kavramsal | A,B |
| E02 | GenD (WACV 2026) | Benchmark-ötesi genelleme | 14 deepfake seti | LN-only fine-tune, eşleşmiş eğitim | CLIP-benzeri | AUROC 90.0; eşleşmemiş eğitimde 56.8 | A,B |
| E03 | Smeu et al. (CVPR 2025) | Veri seti shortcut'ı (sessizlik) | AV deepfake | Real-only | — | n.e. (abstract) | D |
| E04 | Frequency Bias Matters (2025, arXiv) | Frekans yanlılığı | GAN setleri | Spektral hizalama | — | n.e. | A,C |
| E05 | GIFTBench (2026, arXiv) | Çok eksenli genelleme tanısı | 115k + 12 set | Eksen-özel protokoller | Çeşitli | Asimetrik transfer (nitel) | B,D |
| E06 | Aoudi et al. (IEEE Access 2026) | "Metric Trap": sınır artefaktı bağımlılığı | 1M görüntü | Scene-disjoint + XAI | — | n.e. (abstract) | D,A |
| E07 | Zooming In on Fakes / BR-Gen (AAAI 2026) | Yerel AIGC sahteciliği | 150k | Gürültü-güdümlü güçlendirme | NFA-ViT | Avg 0.948 (in-domain) | C |

**Tema özetleri** (ayrıntı `literature_matrix.md` §Tema özetleri): belge-özel ilerleme büyük ölçüde sentetik-içi ve DCT-bağımlıdır; AI-inpainting (TGIF, AIForge-Doc v1/v2) splice dikişi ve çift-JPEG izini ortadan kaldırır; küçük bölge → sabit-eşik kalibrasyon çöküşü (DOCFORGE-BENCH); eşleşmiş gerçek–sahte eğitim shortcut'ı azaltır (GenD); metin yoğunluğu düşük-seviye dedektörleri bozar (TextFake); değerlendirme güvenilirliği (Metric Trap, GIFTBench) doğal görüntülerde tartışılmış, belgelerde yalnızca eşik düzeyinde (DOCFORGE-BENCH) ele alınmıştır.

### C.1 Kritik karşılaştırmalı inceleme (AIForge-Doc, DocTamper, FFDN, ADCD-Net, DOCFORGE-BENCH)

| Soru | AIForge-Doc v1 | DocTamper/DTD | FFDN | ADCD-Net | DOCFORGE-BENCH |
|---|---|---|---|---|---|
| Bilimsel problem | AI-inpainting alan sahteciliği benchmark'ı | Görsel olarak tutarlı metin kurcalama + JPEG dayanıklılığı | Frekans füzyonunun RGB'yi bozmadan yapılması | DCT hizalama kırılganlığı; metin/BG ayrıştırma | Sıfır-atış genelleme ve kalibrasyon |
| Temel varsayım | Tek alan / tek görüntü / tek araç; zero-shot "doğru kapsam" | Kurcalama 8×8 BAG sürekliliğini bozar; JPEG geçmişi var | DCT yardımcı, RGB birincil olmalı | Arka plan büyük ölçüde pristine; girdi JPEG | Pratisyenin etiketli belge verisi yok; τ=0.5 deployment |
| Forensic evidence | — (TruFor: gürültü artığı; DTD: DCT) | RGB + DCT + QT | RGB + DCT + Haar | RGB + adaptif DCT + OCR önseli + prototip | — |
| Spatial / frequency | n/a | FPH + scSE füzyon | VEM (zero-init) + WFE | ŝ_aln gating | n/a |
| Train/test bağımsızlığı | Kaynak/araç tabakalı 80/20; **kaynak-belge gruplaması yok** (audit: %8.9 leakage) | Cross-domain FCD/SCD; aynı sentez | FCD/SCD; T-SROIE **retrain** | FCD/SCD; Tianchi pristine | Zero-shot; in-domain hücreler işaretli |
| Unseen generator/method | Araç kırılımı yalnız TruFor (Ideogram n=85, CI şansı kapsar) | Yok | Yok | Yok | Yok (AI üretimli set yok) |
| Mask annotation etkisi | **bbox maskesi**; piksel-IoU tavanı (audit: bbox'ın %49'u değişiyor) | Piksel düzeyi (bbox mu glif mi UNKNOWN) | DocTamper | DocTamper | bbox→raster setler var; NaN kuralı authentic FPR'yi dışlar |
| Degradasyon / küçük bölge | **Yok** / medyan %0.92 | JPEG Q75–90 | JPEG + noise/blur/resize/jitter | 5 bozulma + crop/shift/scale | Yok / taban oranı analizi |
| Genelleme sınırlaması | İki araç, fine-tune yok | DCT shortcut; sentetik | PNG'de DCT zararlı (yazarlar kabul eder) | JPEG varsayımı; OCR bağımlılığı | 8/112 çiftte kalibrasyon; toolkit yok |
| Bizim fikirle örtüşme | Veri + leakage/annotation sorunları = Gap D malzemesi | DCT shortcut = Gap A'nın kaynağı | Kontrollü frekans = Gap A/C tasarım ilkesi | Gating ablasyonu = Gap A deney şablonu | Eşik protokolü = Gap C/D'nin zaten kapsanan kısmı |

---

## D. Existing Methods vs Research Gaps

### GAP A — Artifact-Invariant Forgery Localization
*Do document forgery localization models learn genuine manipulation evidence or dataset-generation artifacts?*

**Supporting evidence**
- DCT/JPEG shortcut nicel olarak gösterilmiştir: ADCD-Net Şekil 5 (DTD F1 0.935 → crop %2 / shift 1 px / scale %98 ile −%68–73) [VERIFIED]; DocQT (FFDN-Real 0.853 vs FFDN-Std 0.708; yalnız QT farkı) [VERIFIED]; DiffNet yazarları sentetik hat izlerinin öğrenildiğini kabul eder [VERIFIED].
- Tam yeniden üretilmiş inpainting'de klasik IFL yöntemleri çöker (TGIF: PSCC-Net F1 0.05) [VERIFIED] — API tabanlı belge düzenlemede splice dikişi oluşmayabilir.
- AIForge-Doc v2 App. B "generation-asymmetry caveat": sahte görüntüler ek yeniden-kodlama döngüsünden geçer, authentic'ler geçmez [VERIFIED]. **Yerel audit:** eps=0'da çiftlerin ~%49'unda tüm görüntüye yayılan ±1–3 düzeyinde fark; eps=8'de %33.5'inde bbox dışı değişim; Gemini'de bbox dışı sızıntı Ideogram'ın 3 katı (%35.9 vs %11.9) [VERIFIED-LOCAL] → image-level sınıflandırıcı için global yeniden-kodlama izi ve generator-ayrımcı bbox-dışı iz **shortcut adaylarıdır** [HYPOTHESIS].
- Maskeler %100 dikdörtgen; bbox piksellerinin yalnızca ~%49'u değişiyor [VERIFIED-LOCAL] → bbox-GT ile eğitilen bir segmentasyon ağı değişmeyen arka planı "tampered" öğrenmeye zorlanır; en kolay çözüm paste-back dikdörtgen sınırını öğrenmektir [HYPOTHESIS].
- GenD: eşleşmemiş gerçek–sahte eğitim AUROC 90.0 → 56.8 [VERIFIED]; TextFake: metin yoğunluğu düşük-seviye dedektörleri bozar [VERIFIED].

**Counter-evidence**
- Artefakta **adaptif** tasarımlar zaten vardır: ADCD-Net ŝ_aln gating, FFDN VEM zero-init, Omni-IML modal gate [VERIFIED]. Ancak üçü de JPEG/DCT artefaktı ve DocTamper sentezi içindir; PNG AI-inpainting'de ölçüm yoktur.
- Sınır-artefaktı bağımlılığı denetimi (Aoudi 2026 "Metric Trap", scene-disjoint + XAI) doğal görüntülerde önerilmiştir [REPORTED, abstract]; belgelerde uygulanmamıştır.
- SAFIRE'ın kaynak-tutarlılığı çerçevesi artefakt ezberlemeye alternatif sunar [VERIFIED]; belgelerde test edilmemiştir.

**Remaining unknowns**
- AIForge-Doc üzerinde eğitilen bir model paste-back sınırını mı, inpainting dokusunu mu öğreniyor? (sınır-ablasyonu, benign paste-back kontrolü, "yalnız sınır" girdisiyle lokalizasyon — hiçbiri ölçülmemiş.)
- Authentic görüntüler aynı yeniden-kodlama döngüsünden geçirildiğinde image-level performans korunuyor mu?
- Benign kontrol örneklerinin gerçek manipülasyondan ayrıldığı yönler (içerik değişimi yok ama yeniden örnekleme var) confounding yaratır; API'nin iç pipeline'ı bilinmediğinden (UNKNOWN) kontrolün sadakati sınırlıdır.

**Verdict: VALIDATED CANDIDATE** (AI-inpainting belge bağlamında, somut mekanizmalarla: paste-back/bbox sınırı, global yeniden-kodlama izi, generator-özgü bbox-dışı sızıntı). Genel "artifact-invariant learning via gating" fikri **PARTIALLY COVERED**; yalnızca gating/frekans dalı eklemek özgün sayılmaz.

### GAP B — Cross-Generator Generalization
*How reliably do forgery localization models generalize to unseen generative editing models?*

**Supporting evidence**
- AIForge-Doc v1'de araç bazlı fark büyüktür: TruFor Gemini AUC 0.778 [0.754, 0.802] vs Ideogram 0.521 [0.447, 0.597] [VERIFIED]; v2 tek generator (GPT-Image-2) ve v1 ile spec-eşlidir → doğal LOGO protokolü [VERIFIED].
- OSTF/DAF: görülmeyen metin editörlerinde ~13 puan mF düşüşü (scene text, box düzeyi) [VERIFIED, sütun eşlemesi provisional]; COCO-Inpaint cross-model split'te büyük düşüş [VERIFIED]; GenD [VERIFIED].
- Hiçbir belge çalışması "bir generator'da eğit, diğerinde test et" protokolünü çalıştırmamıştır [UNKNOWN → gap]; v2 makalesinin §3.3'te vaat ettiği per-spec cross-generator tablosu §5'te yoktur [VERIFIED].

**Counter-evidence**
- Açık-küme protokolü (OSTF, AAAI 2025) ve cross-model split'ler (COCO-Inpaint) zaten tanımlıdır; DITL2 invariance transfer önermektedir [REPORTED].

**Remaining unknowns / confound'lar [VERIFIED-LOCAL]**
- Generator × kaynak confound: WildReceipt'te Ideogram 0, XFUND'da %45; Ideogram'ın tamper-ratio medyanı %0.27 vs Gemini %1.07 → generator etkisi kaynak ve alan boyutu etkisinden ayrıştırılamaz.
- Ideogram n=422 (test 85): istatistiksel güç düşük.
- `qwen-inpaint` etiketinin gerçekten Ideogram v2 Edit olduğu yalnızca kart beyanıdır [REPORTED].
- v2 verisi gated; authentic'ler yalnızca Google Drive paketinde → erişim UNKNOWN.

**Verdict: PARTIALLY SUPPORTED** — gerçek ve ölçülmemiş bir açık; ancak AIForge-Doc v1 tek başına temiz bir cevap veremez. Kaynak-eşlenmiş alt kümeler (CORD/SROIE/XFUND içinde tool-dengeli) ve v2 erişimi gerektirir. Birincil katkı değil, **ikincil değerlendirme ekseni** olarak uygundur.

### GAP C — Robust Small-Region Localization
*Why do localization models fail on small manipulated regions and under realistic image degradations?*

**Supporting evidence**
- DOCFORGE-BENCH: taban oranı %0.27–4.17 → τ=0.5'te Pixel-F1 ≈ 0 iken Pixel-AUC ≥ 0.76; N=10 kalibrasyon açığın %39–55'ini kapatır [VERIFIED]. AIForge-Doc: medyan alan %0.98, maske yüksekliği medyan 33 px, çözünürlük 0.03–35 MP [VERIFIED-LOCAL] → 512'ye küçültme bölgeyi yok eder; tile/crop zorunlu.
- AIForge-Doc v1/v2'de **hiç degradasyon değerlendirmesi yoktur** [VERIFIED]; FFDN ve ADCD-Net'in dayanıklılık sonuçları yalnızca DocTamper sentezi içindir.

**Counter-evidence (yöntem olarak büyük ölçüde kapsanmış)**
- ADCD-Net (5 bozulma + hizalama pertürbasyonları, FOCAL kaybı), FFDN (5 bozulma + 3 JPEG seviyesi, WFE), IML-ViT (1024 px + kenar denetimi), Mesorch, NFA-ViT (gürültü güçlendirme), CTP-Net (OCR-güdümlü karakter dokusu) [VERIFIED]; DOCFORGE-BENCH eşik kalibrasyon protokolü [VERIFIED]. Frekans dalı / çok ölçekli füzyon / kontrastif kayıp eklemek özgün katkı değildir.

**Remaining unknowns**
- AI-inpainted PNG belgelerde JPEG/resize/blur altında herhangi bir modelin davranışı: sıfır veri.
- Alan dilimine (<%0.5, %0.5–1, >%1) göre tabakalı performans ve generator etkileşimi raporlanmamıştır.

**Verdict: LIKELY ALREADY COVERED** (yöntem düzeyinde); **PARTIALLY SUPPORTED** (değerlendirme düzeyinde). Projede zorunlu **değerlendirme ekseni** (tabakalı metrikler + JPEG Q75 / 0.75× resize probu), özgün iddia değil.

### GAP D — Annotation and Evaluation Reliability
*Do existing ground-truth annotations and evaluation protocols accurately measure forgery localization?*

**Supporting evidence [ağırlıkla VERIFIED-LOCAL]**
- GT = alan bbox maskesi; gerçek piksel değişimi bbox'ın medyan %49'u (IoU 0.48) → piksel-IoU/F1 tavanlıdır ve bbox-şekli bir shortcut'tır. Bbox dışı değişim çiftlerin %33.5'inde → GT'nin dışında kalan gerçek değişimler "false positive" sayılır.
- Resmî test authentic'i test sahtelerinin aynası değil (86/812); %66'sı train-authentic kopyası → image-level AUROC bu negatiflerle geçersiz. Test sahtelerinin %8.9'u train kaynak belgesini paylaşır (leakage).
- Literatür: AIForge-Doc'un DocTamper için yanlış alıntıladığı değerler; v2'de "AUC" = balanced accuracy; DTD/FFDN/ADCD-Net optimal-eşik F1 raporlarken DOCFORGE-BENCH τ=0.5 kullanır; DOCFORGE-BENCH NaN kuralı authentic FPR'yi dışlar [VERIFIED].

**Counter-evidence**
- DOCFORGE-BENCH eşik/kalibrasyon, Oracle-F1 ve bbox→raster sorunlarını zaten ele almıştır [VERIFIED]; Aoudi 2026 "Metric Trap" ve GIFTBench çok eksenli değerlendirme önerir [REPORTED/VERIFIED]; Smeu 2025 veri seti shortcut denetimi örneğidir [REPORTED]. → **Eşik ve genel protokol eleştirisi yeniden keşfedilmemelidir.**

**Remaining unknowns**
- Piksel-değişim GT'si ile bbox GT'si altında yöntem sıralaması değişir mi? Hangi GT ile eğitilen model daha iyi transfer eder?
- Bbox-GT'nin dedektör mimarisi tercihini (segmentasyon vs kutu tespiti) ne ölçüde yanlılaştırdığı.
- İnsan doğrulamalı piksel maskesi yoktur; türetilmiş D>ε maskesinin eşik duyarlılığı.

**Verdict: VALIDATED CANDIDATE** (AIForge-Doc'a özgü, ampirik olarak temellendirilmiş; eşik meselesi hariç literatürde kapsanmamış).

---

## E. AIForge-Doc Dataset Audit

Tam rapor: `dataset_audit/audit_report.md` (14 boyut + annotation semantics + split önerisi; tüm sayılar `audit_dataset.py` ile `audit_stats.json`'a yazılmıştır; yeniden üretim: `python audit_dataset.py --data <data> --out <out> --workers 24`, ~3 dk).

### E.1 Doğrulanmış gerçekler (Verified from source / local audit)
| Boyut | Bulgu | Etiket |
|---|---|---|
| 1 Yapı | `TrainingSet/{images,masks,authentic}` 3.249×3, `TestingSet/{images,masks,authentic}` 812×3, `metadata.jsonl` 4.061 satır; düz PNG (parquet yok); 9.98 GB; README'deki DATASHEET.md yok | VERIFIED-LOCAL |
| 2 Eşleşme | 0 eksik / 0 çift; maske boyutu = görüntü boyutu 4.061/4.061 | VERIFIED-LOCAL |
| 3 Maske | 0/255 ikili; tek bağlı bileşen; dikdörtgenlik 1.000 (hepsi); metadata bbox ↔ maske 4.047/4.061 birebir; 12 satır hatalı (spec_id çakışması) → **maske dosyası GT** | VERIFIED-LOCAL |
| 3b Semantik | Doğrulanmış 3.336 çift (train 3.247 / test 89): eps=8'de bbox-içi değişen oran medyan 0.49; IoU(D>8, maske) 0.48; bbox-dışı değişim %33.5 (Gemini %35.9 / Ideogram %11.9); eps=0'da ~%49 çiftte global ±1–3 iz | VERIFIED-LOCAL |
| 4 Kaynak | CORD 1.000 / WildReceipt 1.696 / SROIE 946 / XFUND 419; split her kaynakta %20 | VERIFIED-LOCAL (kartla uyumlu) |
| 5 Generator | gemini-nano 3.639 (test 727) / qwen-inpaint 422 (test 85); WildReceipt'te Ideogram 0, XFUND'da 188/419 | VERIFIED-LOCAL |
| 6 Adlandırma | Kod adları kartın ticari API eşlemesiyle çelişir; makalede "Qwen" yok | VERIFIED-SOURCE |
| 7 Belge tipi | receipt 3.642 (%89.7) / form 419; dil: en 2.642, id 1.000, 7 diğer; "invoice" yok | VERIFIED-LOCAL |
| 8 Alan | 38 alan adı; Telephone_key 931, text 880, Store_addr_key 457, answer 419, menu.price 404 … (SROIE'de serbest metin baskın; "numeric field" iddiasıyla kısmen çelişir) | VERIFIED-LOCAL |
| 9 Çözünürlük | W medyan 750 (150–6016), H medyan 1296 (168–7016); 0.03–34.8 MP; 4961×7016 görüntüler SROIE (makale XFUND der) | VERIFIED-LOCAL |
| 10 Tamper alanı | medyan %0.975 (p25 0.37, p75 1.62); <1%: %50.7, 1–5%: %49.0, 5–20%: %0.4; maske h medyan 33 px, w 191 px; tool: Gemini %1.07 vs Ideogram %0.27 | VERIFIED-LOCAL |
| 11 Duplicate | 307 MD5 grubu (12 forged–forged); 274 forged–forged pHash≤20 çifti; 41 authentic 2–3 forged'ın kaynağı → "one-to-one" içerik düzeyinde yanlış | VERIFIED-LOCAL |
| 12 Leakage | 581 grup iki split'e yayılır; test-forged 72/812 (%8.9) train'e pHash≤20 yakın (23 birebir) | VERIFIED-LOCAL |
| 13 Authentic | Train: 3.247/3.249 sahte için doğrulanmış kaynak (çift yapısı tam). Test: yalnız 86/812 kaynak; 254 bayt-özdeş + 535 pHash≤10 train kopyası; SROIE authentic yok | VERIFIED-LOCAL |
| 14 Lisans | front-matter cc-by-nc-sa-4.0 + gating "non-commercial" ↔ dataset_info "CC BY 4.0"; kaynak lisansları kart beyanı (CORD/XFUND CC BY 4.0, WildReceipt Apache 2.0, SROIE research use) | VERIFIED-SOURCE / UNKNOWN (kaynak metinleri açılmadı) |

### E.2 Çözülmemiş sorunlar ve tutarsızlıklar
13 paper↔kart↔metadata↔dosya tutarsızlığı listelenmiştir (audit_report §15); en önemlileri: authentic rolü ("mirror of test split" yanlış), "pixel-precise" ↔ bbox, context padding 100/150 px, generator adları, lisans, metadata şeması (README `image_id`/`final_*_path` yok), citation placeholder.

### E.3 Metadata gereksinimleri (projenin kullanacağı alanlar)
`new_id` (dosya kimliği), `source_dataset`, `doc_type`, `language`, `field_name`, `assigned_tool`; bbox için **maske dosyası**; authentic eşi için audit'in ürettiği `samples_forged.csv → pair_name, pair_accepted`; kaynak-belge grubu için `group`; leakage bayrağı için `nn_train_any_dist`.

### E.4 Leakage riskleri ve veri bölme kuralları (uygulandı)
- Resmî test (812) dokunulmadı. Train, kaynak-belge gruplarına (forged↔authentic + pHash≤20 bağlı bileşenleri) göre **train 2.762 / val 487** olarak bölündü (val–train grup kesişimi yok; test'e dokunan 581 train grubu val dışında tutuldu) → `dataset_audit/splits.json`.
- Küçük-veri çalışmaları için **smoke_train 400 / smoke_val 100** (alan dilimi tabakalı, kaynak×tool dengeli).
- Resmî test üzerinde raporlama: **"tam test" (812)** ve **"temiz test" (740 = train'e pHash≤20 yakın olmayan)** ayrı ayrı; image-level metrikler için resmî test-authentic kullanılmayacak, bunun yerine train-authentic'ten group-aware negatifler ve/veya train ile ilişkisiz 277 test-authentic alt kümesi (kaynak dağılımı eksik olduğu açıkça belirtilerek).

### E.5 Veri setinin projeyi kısıtladığı noktalar
Lokalizasyon hedefi bbox (piksel değil); küçük bölge + 1:100 piksel dengesizliği → tile/crop + Dice/focal; image-level için resmî negatifler güvenilmez; generator↔kaynak confound; %8.9 leakage; metadata bbox hataları; BY-NC-SA + gating (yeniden dağıtım yok); %90 fiş → "finansal belge" genellemesi sınırlı.

---

## F. Three Novel Research Candidates

Adaylar farklı bilimsel iddialar taşır: **C1** annotation (GT) etkisini, **C2** veri-üretim artefaktı etkisini, **C3** generator kaymasını hedefler.

### C1 — Annotation-Aware Localization: "Box mask'ten ne öğrenilir?"
- **Problem:** AI-inpainted belgelerde lokalizasyon ağlarını alan-bbox maskesiyle eğitmek/değerlendirmek, modelin kullandığı kanıtı (dikdörtgen sınır vs glif dokusu) ve ölçülen performansı sistematik biçimde bozuyor mu?
- **En yakın prior work:** DOCFORGE-BENCH (bbox→raster ve eşik; eğitim etkisini incelemez), Aoudi 2026 Metric Trap (doğal görüntü, XAI), AIForge-Doc v1 (bbox GT'yi tanımlar, sorgulamaz), MVSS-Net/IML-ViT (kenar denetimi — sınır artefaktını *kullanır*).
- **Çözülmemiş gap:** Eşleşmiş authentic'lerden türetilen piksel-değişim GT'si (D>ε) ile bbox GT'si altında (i) metrik ve yöntem sıralaması, (ii) öğrenilen kanıt, (iii) transfer/dayanıklılık karşılaştırması hiç yapılmamıştır.
- **Ana hipotez:** Bbox-GT ile eğitilen model paste-back dikdörtgen sınırına bağımlı bir çözüm öğrenir; piksel-değişim GT'si (veya glif-kısıtlı GT) ile eğitilen model sınır ablasyonuna daha dayanıklıdır ve görülmemiş generator/degradasyona daha iyi transfer eder; aynı tahminler iki GT altında farklı sıralama verir.
- **Metodolojik katkı:** (a) AIForge-Doc için türetilmiş ikinci GT (pixel-change + morfolojik/glif-kısıtlı varyantlar) ve eşik-duyarlılık analizi; (b) shortcut probe seti: sınır-ablasyonu, "rectangle-only" tahmin baseline'ı, benign paste-back kontrolü, "boundary-only" girdi; (c) çift-GT raporlama protokolü (bbox-IoU + pixel-AUPRC + alan-tabakalı Oracle/fixed F1 + authentic FPR).
- **Veri gereksinimi:** AIForge-Doc v1 train çiftleri (3.247 doğrulanmış) — **mevcut**; test için bbox-GT + temiz alt küme; opsiyonel 100 çiftlik insan doğrulamalı maske.
- **Baseline'lar:** RGB U-Net (ResNet-18/34), RGB+yüksek-geçiren U-Net, YOLO-seg/kutu dedektörü (bbox-GT'nin kutu formülasyonunu kayırıp kayırmadığını test eder), TruFor zero-shot, ADCD-Net (MIT; PNG'de DCT dalı kapalı/açık), klasik patch sınıflandırıcı.
- **Deneysel kontroller:** aynı mimari/augment/bütçe, 3 seed, group-aware bootstrap CI; eşik seçimi yalnızca val'de; test'e dokunulmaz.
- **Birincil metrikler:** Pixel-AUPRC/AUROC (eşiksiz), fixed-F1@0.5 ve Oracle-F1, bbox-IoU, alan-dilimi tabakalı; sınır-ablasyon ΔAUPRC; benign-kontrol FP alanı.
- **Geçerlilik tehditleri:** D>ε GT'si yeniden-kodlama gürültüsünü içerebilir (ε duyarlılığı raporlanmalı); bbox dışı gerçek değişimler GT'ye girer (bu tasarımın gereği); test'te piksel-GT yok (val ile sınırlı iddialar).
- **Başarısızlık koşulları:** İki GT ile eğitilen modeller aynı AUPRC/dayanıklılığa ulaşır ve sıralama değişmezse hipotez düşer — bu da "bbox GT yeterlidir" şeklinde yayımlanabilir negatif sonuçtur.
- **Yayın potansiyeli:** Orta-yüksek (benchmark-audit + metodoloji; IEEE/Springer konferansı için uygun; ICDAR/WIFS/ICIP tarzı).

### C2 — Paste-Back / Re-Encoding Artifact vs Generative Trace Disentanglement
- **Problem:** AIForge-Doc üzerinde eğitilen dedektörler inpainting izini mi, yoksa veri-üretim hattının compositing/yeniden-kodlama artefaktını mı öğreniyor; benign kontrol örnekleri bunu ayrıştırıp generator-bağımsız kanıt öğrenimini zorlayabilir mi?
- **En yakın prior work:** ADCD-Net (DCT gating + hizalama pertürbasyonları), GenD (eşleşmiş eğitim), TGIF (tam yeniden üretim), AIForge-Doc v2 App. B (asimetri uyarısı), Smeu 2025 (veri seti shortcut'ı).
- **Gap:** Benign paste-back (içerik değişmeden aynı crop→2× LANCZOS→paste-back) ve simetrik yeniden-kodlanmış authentic kontrolleriyle shortcut'ın nicel ölçümü ve kontrol-destekli eğitimin etkisi ölçülmemiştir.
- **Hipotez:** Bbox-GT/standart eğitimli modelin benign kontrollerde yüksek FP alanı ve authentic yeniden-kodlamada image-level çöküşü vardır; kontrol-destekli eğitim held-out generator'da AUPRC'yi artırır.
- **Katkı:** Kontrol seti inşa protokolü + artefakt-ayrışma ablasyonu; **risk:** API'lerin gerçek iç pipeline'ı bilinmiyor (UNKNOWN) → kontrol sadakati sınırlı; kısmen C1 ile örtüşür (probe olarak C1'e dahil edilebilir).
- **Veri:** Train çiftleri + sentetik kontroller (mevcut); **Baseline/kontrol/metrik:** C1 ile aynı + image-level AUROC (group-aware negatiflerle).
- **Tehditler:** Kontrol inşası generator yolunu tam taklit edemez; Gemini/Ideogram farkı kaynakla confound'lu.
- **Başarısızlık:** Model benign kontrollerde tepki vermiyorsa shortcut yok → C1'in sınır hipotezi de zayıflar (bilgi üretir).
- **Yayın:** Orta-yüksek; ancak "veri seti artefaktı" bulgusu tek başına metodolojik katkıdan çok benchmark eleştirisidir.

### C3 — Source-Matched Cross-Generator Generalization (v1 Gemini/Ideogram ↔ v2 GPT-Image-2)
- **Problem:** Belge inpainting lokalizasyonu görülmemiş generator'a ne kadar transfer eder; eşleşmiş-fark (siamese) veya consistency regularization açığı azaltır mı?
- **En yakın prior work:** OSTF/DAF (scene text, box), COCO-Inpaint (doğal görüntü cross-model), GenD, DITL2.
- **Gap:** Belgelerde LOGO protokolü yok; v1+v2 spec-eşli çift benzersiz fırsat.
- **Hipotez:** Gemini→Ideogram ve v1→v2 transferinde AUPRC önemli ölçüde düşer; düşüşün bir kısmı kaynak/alan-boyutu confound'undan gelir; eşleşmiş-fark denetimi düşüşü azaltır.
- **Veri gereksinimi:** v2 (gated; authentic Google Drive) **UNKNOWN**; v1 içinde kaynak-eşlenmiş alt kümeler (CORD 900/100, SROIE 812/134, XFUND 231/188) mevcut ama Ideogram n küçük.
- **Baseline'lar:** C1 ile aynı + domain generalization (consistency/augment) varyantları.
- **Tehditler:** Confound, düşük güç, generator kimliği REPORTED, v2 erişimi.
- **Başarısızlık:** Fark kaynak kontrolünden sonra kaybolursa "generator etkisi yok" — yine bilgi ama veri güçsüz.
- **Yayın:** Orta; veri erişimi ve güç sorunları hakem riskini artırır.

---

## G. Comparative Decision Matrix (1 = zayıf, 5 = güçlü; "Risk of Already Being Solved" için 5 = düşük risk)

| Ölçüt | C1 Annotation-aware | C2 Artifact disentanglement | C3 Cross-generator | Gerekçe (kanıt) |
|---|---|---|---|---|
| Novelty | 4 | 4 | 3 | C1/C2 belgelerde ölçülmemiş (UNKNOWN); C3'ün protokolü OSTF/COCO-Inpaint'te var |
| Scientific Value | 4 | 4 | 4 | Üçü de genelleme/güvenilirlik sorusuna cevap verir |
| Dataset Feasibility | 5 | 3 | 2 | C1: 3.247 çift yerelde doğrulandı; C2: API pipeline UNKNOWN; C3: v2 gated/UNKNOWN, confound, n=85 |
| Implementation Feasibility (tek GPU) | 4 | 3 | 3 | C1: U-Net + GT türetme; C2: kontrol seti inşası ek iş; C3: iki veri seti + DG yöntemleri |
| Experimental Verifiability | 5 | 3 | 3 | C1 hipotezleri aynı model/aynı tahminle test edilir; C2/C3 confound'lu |
| Publication Potential | 4 | 4 | 3 | Benchmark-audit + protokol katkısı hakemli konferanslarda değerli; C3 veri erişimine bağlı |
| Risk of Already Being Solved (5 = düşük) | 4 | 3 | 3 | C1 yalnız eşik kısmı (DOCFORGE-BENCH) kapsanmış; C2 ADCD-Net/GenD yakın; C3 OSTF/COCO-Inpaint yakın |
| **Toplam (35)** | **30** | **24** | **21** | |
| Implementation Complexity | Medium | Medium–High | Medium–High | |

Puanlar kanıta dayalı uzman değerlendirmesidir; ampirik başarı tahmini değildir.

---

## H. Selected Research Direction

**Seçim: C1 — Annotation-Aware Localization**, C2'nin benign paste-back kontrolü ve C3'ün kaynak-eşlenmiş held-out-Ideogram değerlendirmesi **ikincil probe/eksen** olarak C1'e gömülür.

**Önerilen başlık (geçici):** *What Do Document Forgery Localizers Learn from Box Masks? Pixel-Change Ground Truth and Shortcut Audits for AI-Inpainted Financial Documents*

**Tercih gerekçesi.** (1) Literatürde savunulabilir açık: belgelerde GT semantiğinin eğitim/değerlendirme etkisi ölçülmemiş (Bölüm D-Gap D); (2) deneysel doğrulanabilirlik: gerekli tüm veri (3.247 eşleşmiş çift) yerelde doğrulandı; (3) hipotezler aynı model/aynı tahminler üzerinde falsifiable; (4) tek GPU'da yürütülebilir (U-Net/ResNet ölçeği, crop tabanlı); (5) katkı yeni bir ağ adı değil, GT-protokol + shortcut-audit metodolojisidir; (6) baseline karşılaştırmaları adildir (aynı mimari, yalnız GT değişir); (7) negatif sonuç da bilgi üretir ("bbox GT yeterli").

**En güçlü karşı argüman.** "Bu bir yöntem değil, veri seti eleştirisidir; piksel-değişim GT'si test setinde olmadığından ana iddialar val ile sınırlı kalır ve DOCFORGE-BENCH benzeri benchmark çalışmaları bu alanı zaten doyurmuştur." **Yanıt:** katkı protokol + eğitim-sinyali etkisidir (yalnızca değerlendirme değil); test'te bbox-GT, temiz alt küme ve held-out generator üzerinde transfer iddiası doğrulanır; benchmark çalışmalarının hiçbiri eğitim GT'sinin öğrenilen kanıta etkisini ölçmemiştir. Ayrıca AIC515 kapsamıyla (eşikleme, morfoloji, filtreleme, U-Net) doğrudan ilişkilidir.

**Karar: CONDITIONAL GO.** GO'ya dönüşmesi için gereken asgarî kanıtlar:
1. Küçük-veri baseline'ında RGB U-Net'in smoke_val'de şans üstü piksel-AUPRC üretmesi ve kodların uçtan uca çalışması — **sağlandı** (Bölüm J.1: U-Net RGB Pixel-AUPRC 0.947, F1@0.5 0.707; ResNet-18 ft AUROC 0.899; tek seed, smoke alt küme).
2. Piksel-değişim GT'sinin ε-duyarlılığının (ε ∈ {4, 8, 16} + morfolojik açma) ve 100 çiftlik görsel insan doğrulamasının tamamlanması; GT'nin yeniden-kodlama gürültüsünden ayrışması.
3. Benign paste-back kontrol setinin inşası ve en az bir baseline'da FP davranışının ölçülmesi (shortcut hipotezinin ön testi).
4. Temiz test alt kümesi (740) ve group-aware negatif protokolünün sabitlenmesi (yapıldı: `splits.json`).

---

## I. Research Questions & Falsifiable Hypotheses

**Primary RQ.** AI-inpainted belge sahteciliği lokalizasyonunda, alan-bbox maskesi yerine eşleşmiş authentic'ten türetilen piksel-değişim maskesiyle eğitmek ve değerlendirmek, modelin kullandığı forensic kanıtı ve ölçülen performansı nasıl değiştirir?

**Supporting RQ1.** Aynı tahminler bbox-GT ve piksel-değişim-GT ile değerlendirildiğinde piksel metrikleri (F1, IoU, AUPRC) ve yöntem sıralaması ne kadar değişir?
**Supporting RQ2.** Bbox-GT ile eğitilen modeller paste-back dikdörtgen sınırına (shortcut) ne ölçüde bağımlıdır (sınır ablasyonu, rectangle-only baseline, benign paste-back kontrolü)?
**Supporting RQ3.** Hangi GT ile eğitim, görülmemiş generator (kaynak-eşlenmiş Ideogram alt kümesi) ve degradasyon (JPEG Q75, 0.75× resize) altında daha iyi transfer sağlar?

### H1 — Metrik ve sıralama bozulması
1. **İfade:** Aynı modelin aynı tahminleri için piksel-F1/IoU, piksel-değişim-GT'ye göre bbox-GT'ye göre olduğundan anlamlı ölçüde düşüktür ve en az dört baseline'dan ikisinin sıralaması iki GT arasında yer değiştirir.
2. **Bağımsız değişken:** Değerlendirme GT türü (bbox vs D>ε vs glif-kısıtlı).
3. **Bağımlı değişken:** Pixel-F1@0.5, IoU, AUPRC; sıralama (Kendall τ).
4. **Kontrol:** Aynı tahmin haritaları, aynı val görüntüleri, aynı eşik politikası.
5. **Baseline/kontrol grubu:** Rectangle-only "oracle-bbox" tahmini (üst sınır) ve rastgele tahmin.
6. **Metrik:** ΔF1, ΔIoU, Kendall τ; group-bootstrap %95 CI.
7. **Destekleyen kanıt:** bbox'ın %49'u değişiyor; IoU(D>8, maske) 0.48 [VERIFIED-LOCAL]; DOCFORGE-BENCH eşik duyarlılığı.
8. **Yanlışlama koşulu:** ΔF1 < 0.05 ve Kendall τ ≥ 0.9 (sıralama korunur).
9. **Alternatif açıklama:** D>ε GT'si yeniden-kodlama gürültüsünü içeriyorsa fark GT gürültüsünden kaynaklanır → ε-duyarlılık ve morfolojik temizleme kontrolü.

### H2 — Sınır (paste-back) shortcut'ı
1. **İfade:** Bbox-GT ile eğitilen U-Net'in piksel-AUPRC'si, forged görüntülerde bbox sınırı etrafındaki 5 px bandın yumuşatılmasıyla (sınır ablasyonu) göreli olarak piksel-değişim-GT ile eğitilen modelinkinden daha fazla düşer; ayrıca benign paste-back kontrollerinde anlamlı FP alanı üretir.
2. **Bağımsız değişken:** Eğitim GT türü; test-zamanı sınır ablasyonu (var/yok); benign kontrol (var/yok).
3. **Bağımlı değişken:** Göreli ΔAUPRC; benign kontrolde ortalama FP piksel oranı ve image-level yanlış alarm oranı.
4. **Kontrol:** Aynı mimari, augment, epoch, seed seti; ablasyon authentic görüntülere de uygulanır (yalnız sınır etkisi izole edilir).
5. **Baseline:** Ablasyonsuz performans; rastgele bölge yumuşatma (aynı alan, farklı konum) plasebo kontrolü.
6. **Metrik:** AUPRC (eşiksiz), FP oranı; 3 seed, CI.
7. **Destekleyen kanıt:** ADCD-Net hizalama pertürbasyon ablasyonu; MVSS-Net/IML-ViT kenar denetiminin sınır artefaktına dayanması; Aoudi 2026 [REPORTED]; maskelerin %100 dikdörtgen olması.
8. **Yanlışlama koşulu:** Her iki model de ablasyonda benzer (fark CI içinde) düşer ve benign kontrollerde FP ≈ authentic düzeyinde kalır.
9. **Alternatif açıklama:** Düşüş sınır değil, yumuşatmanın glif dokusunu da bozmasından olabilir → plasebo kontrolü ve iç-bölge yumuşatma varyantı.

### H3 — Transfer ve dayanıklılık
1. **İfade:** Piksel-değişim-GT (veya glif-kısıtlı GT) ile eğitilen model, kaynak-eşlenmiş held-out Ideogram alt kümesinde ve JPEG Q75 / 0.75× resize altında bbox-GT ile eğitilen modelden daha yüksek piksel-AUPRC elde eder.
2. **Bağımsız değişken:** Eğitim GT türü; test koşulu (in-generator / held-out generator / degradasyon).
3. **Bağımlı değişken:** Pixel-AUPRC, Oracle-F1, alan-tabakalı F1.
4. **Kontrol:** Eğitimde yalnız Gemini (CORD/SROIE/XFUND), test Ideogram aynı kaynaklar; aynı alan-dilimi dağılımına yeniden ağırlıklandırma (tamper-ratio confound'u için).
5. **Baseline:** Bbox-GT modeli; TruFor zero-shot (eşit koşul olmadığı açıkça belirtilerek).
6. **Metrik:** ΔAUPRC held-out − in-generator; group-bootstrap CI; 3 seed.
7. **Destekleyen kanıt:** TruFor Gemini 0.778 vs Ideogram 0.521; GenD eşleşmiş eğitim; FFDN/ADCD-Net degradasyon düşüşleri.
8. **Yanlışlama koşulu:** Fark yok veya ters yönde (CI'lar örtüşür).
9. **Alternatif açıklama:** Kazanım GT'den değil, piksel-değişim GT'sinin daha az pozitif piksel içermesi (sınıf dengesi) veya augment farkından olabilir → pozitif piksel sayısı eşlenmiş kontrol ve aynı augment.

### H4 (opsiyonel, image-level) — Global yeniden-kodlama shortcut'ı
1. **İfade:** Train-authentic negatiflerle eğitilen forged-vs-authentic sınıflandırıcısının AUROC'u, authentic görüntüler sahte yolundaki PNG yeniden-kodlama/2× LANCZOS döngüsünden geçirildiğinde anlamlı düşer.
2–9. IV: authentic ön-işleme (simetrik/asimetrik); DV: AUROC; kontrol: aynı model; baseline: asimetrik koşul; kanıt: eps=0'da %49 çiftte global iz, v2 App. B; yanlışlama: AUROC değişmez; alternatif: döngü glif dokusunu da değiştirir → yalnız yeniden-kodlama (ölçekleme yok) varyantı.

---

## J. Next-Stage Research Requirements (PHASE 2 öncesi)

### J.1 Küçük-veri baseline sonuçları (feasibility kanıtı — bu turda çalıştırıldı)

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


### J.2 Model geliştirmeden önce doğrulanması gerekenler
**Veri**
1. Piksel-değişim GT'sinin türetilmesi ve ε-duyarlılığı (ε ∈ {4, 8, 16}, morfolojik açma 3×3, bağlı-bileşen filtresi); 100 çiftlik görsel insan doğrulaması; glif-kısıtlı GT (Otsu/Sauvola ile metin pikselleri ∩ bbox) alternatifi.
2. Benign paste-back kontrol setinin inşası (train-authentic'ten: bbox crop → 2× LANCZOS up → down → paste-back; ayrıca yalnız PNG yeniden-kodlama) ve asimetri ölçümü.
3. Temiz test alt kümesi (740) ve group-aware negatif listesinin dondurulması; test'e hiçbir model seçimi için dokunulmaması.
4. AIForge-Doc v2 erişiminin (gated + Google Drive authentic) denenmesi; başarısızsa C3 ekseni v1-içi Ideogram ile sınırlı kalır (güç analizi: n=85 test / 422 toplam).
5. Lisans: BY-NC-SA + gating; türetilmiş maskeler yalnızca id + maske olarak (görüntüsüz) paylaşılabilir; aynı lisansla.

**Literatür**
6. FFDN ve DocTamper tam metinlerinin matris satırlarına işlenmesi (seed hattında okundu); DOCFORGE-BENCH toolkit ve arXiv:2311.18818 kimliğinin yeniden kontrolü; OpenAlex ileri-atıf taraması (AIForge-Doc'a atıf yapan 2026 çalışmaları — en yakın rakip takibi).
7. Aoudi 2026 (Metric Trap) ve DITL2 tam metinlerine erişim (kapalı erişim; kurumsal abonelik).

**Deneysel koşullar**
8. Literature baseline'ların kurulumu: TruFor (resmî ckpt; lisans kontrolü), ADCD-Net (MIT; PNG girdide DCT dalı için QT=1 veya ŝ_aln=0 koşulu), DTD (inference ckpt; LICENSE yok → yalnız değerlendirme), FFDN (ForensicHub uygulaması, ağırlık durumu REPORTED). Zero-shot ve fine-tuned sonuçlar **ayrı tablolarda** raporlanır.
9. İstatistik: 3 bağımsız seed, kaynak-belge grubu üzerinde bootstrap CI (1.000 örnek), çoklu karşılaştırma için Holm düzeltmesi; eşik yalnız val'de (fixed-F1@0.5, val-kalibre τ*, Oracle-F1 üst sınır olarak).
10. Kaynak bütçesi: RTX 5080 16 GB — 512×512 crop, batch 8–16, bf16; tam-görüntü çıkarım sliding-window (overlap 64 px); XFUND/SROIE 8–35 MP görüntüler için tile planı.

### J.3 AIC515 ders kapsamıyla ilişki (her teknik için "test edilebilir forensic avantaj sağlıyor mu?")

| Teknik (hafta) | Projedeki rol | Test edilebilir avantaj? | Karar |
|---|---|---|---|
| Yoğunluk dönüşümleri / kontrast germe (2) | Yalnız augmentation (brightness/contrast jitter) | Hayır (forensic kanıt değil) | Augment olarak kullan; katkı iddiası yok |
| Geometrik dönüşümler (2) | Dayanıklılık pertürbasyonu (0.75× resize, hizalama kaydırma) ve sınır-ablasyon kontrolleri | Evet (H3) | Kullan |
| Uzamsal filtreleme — keskinleştirme/yüksek-geçiren (3) | Laplacian artık kanalı (RGB+HPF U-Net varyantı); klasik öznitelik | Evet (RGB vs RGB+HPF ablasyonu) | Kullan, ölçülürse tut |
| Frekans domeni, 2D FFT (4) | FFT Gaussian yüksek-geçiren kanal; radyal spektrum öznitelikleri; **DCT PNG'de anlamsız** (negatif bulgu) | Evet (ablasyon) | Kullan; DCT'yi yalnız "neden çalışmaz" kontrolü olarak |
| Gürültü ve ortalama filtreleri / restorasyon (5) | Gürültü artığı = görüntü − medyan/Wiener süzgeç (NoisePrint benzeri el yapımı kanal) | Evet (kanal ablasyonu) | Opsiyonel |
| Eşikleme ve morfoloji (6) | Olasılık haritası son-işleme (Otsu vs sabit τ; açma/kapama); glif-kısıtlı GT üretimi (Sauvola) ; rectangle-fit shortcut probu | Evet (H1, H2) | Kullan — çalışmanın merkezinde |
| K-Means / Watershed (7) | Tahmin bölgesi içinde glif/arka plan ayrımı (piksel-GT'ye yaklaşım) | Kısmen | Opsiyonel |
| LBP / HOG / GLCM (9) | Klasik patch sınıflandırma baseline'ı | Evet (alt sınır) | Baseline olarak tut |
| SVM / Random Forest (10–11) | Klasik öznitelik sınıflandırıcıları | Evet | Baseline |
| ResNet (12) | Patch-düzeyi sınıflandırma; U-Net encoder | Evet (pretrained vs scratch) | Kullan |
| U-Net (13) | Ana lokalizasyon modeli (bbox-GT vs piksel-GT) | Evet | Ana model |
| YOLO (14) | Kutu-düzeyi tespit: bbox-GT'nin kutu formülasyonunu kayırıp kayırmadığı | Evet (H1'in mimari uzantısı) | PHASE 2'de opsiyonel |

### J.4 Research integrity & adversarial review (öz-denetim)
- **Problem gerçekten açık mı?** Belgelerde eğitim-GT semantiğinin öğrenilen kanıta etkisi: taramada (38 çalışma + 6 seed tam metin) ölçen çalışma yok → açık (UNKNOWN düzeyinde, kesin yokluk değil; ileri-atıf taraması eksik).
- **Çürüten yayın var mı?** DOCFORGE-BENCH eşik meselesini çözmüştür; bu nedenle eşik katkısı iddia edilmez. Aoudi 2026 benzer "metric trap" tezini doğal görüntülerde savunur — bizim katkımız belge + eğitim-sinyali boyutudur.
- **En yakın prior work:** DOCFORGE-BENCH (değerlendirme), ADCD-Net (shortcut ablasyonu), GenD (eşleşmiş eğitim). Ölçülebilir fark: aynı tahminlerde çift-GT metrikleri, sınır-ablasyon ΔAUPRC, benign kontrol FP.
- **Veri seçimi yapay avantaj yaratıyor mu?** Evet riski var: eşleşmiş çiftler yalnızca AIForge-Doc'ta mevcut; bulgular bbox-GT'li diğer setlere (Tampered-IC13, Receipt) genellenemez → sınırlama olarak yazılır.
- **Shortcut/leakage yanlış sonuç verebilir mi?** %8.9 leakage ve test-authentic sorunu için temiz alt küme + group-aware negatifler; global yeniden-kodlama izi H4 ile açıkça test edilir.
- **Yanlış annotation semantiği sonucu geçersiz kılar mı?** Bu çalışmanın konusu tam olarak budur; türetilmiş GT'nin kendi hataları (ε) duyarlılık analiziyle raporlanır.
- **Küçük veri / dengesizlik:** 3.249 görüntü, 1:100 piksel; crop örnekleme, Dice/focal, 3 seed + bootstrap CI; etki büyüklükleri CI ile verilir.
- **Gerçek finansal doğrulamaya genelleme:** Savunulamaz — %90 fiş, tek alan/tek görüntü tehdit modeli, iki (üç) generator; sonuçlar "AIForge-Doc benzeri fiş/form" ile sınırlı ifade edilir.
- **Başarısızlık koşulları:** H1–H3 yanlışlanırsa sonuç "bbox-GT pratikte yeterlidir ve sınır shortcut'ı yoktur" şeklinde yayımlanabilir negatif bulgu; GT türetimi çok gürültülüyse (insan doğrulaması < %80 uyum) C1 düşer ve C2'ye geçilir.

### J.5 Arama stratejisi özeti
`literature/search_strategy.md`: arXiv API (13 tema sorgusu, 439 kayıt / 395 tekil), Consensus (8 sorgu, 78), hedefli isim aramaları; CrossRef/arXiv kimlik doğrulaması (48 aday → 38 dahil); inclusion: 2023–Ekim 2026 (seminal eski çalışmalar hariç), belge/metin/inpainting lokalizasyonu veya genelleme/shortcut/değerlendirme metodolojisi; exclusion: birincil kaynağı doğrulanamayan, yalnız video/ses, yalnız tam-görüntü AIGI (TextFake hariç).

---

## Ekler
- **Ek A — Seed makale tam inceleme:** `literature/seed_papers_review.md`, `literature/seed_papers.json`
- **Ek B — Literatür matrisi:** `literature/literature_matrix.{csv,xlsx,md}`, `literature/references.bib`, `literature/search_strategy.md`
- **Ek C — Veri seti audit:** `dataset_audit/audit_report.md`, `audit_stats.json`, `splits.json`, `samples_forged.csv`, figürler (`fig_resolution.png`, `fig_tamper_ratio.png`, `fig_examples_overlay.png`, `fig_pixel_diff_quads.png`)
- **Ek D — Baseline kodları ve sonuçları:** `baselines/` (README.md, kod, results/, figures/, logs/), `models/`
- **Ek E — Ortam:** `baselines/ENV_SETUP.md`, `baselines/requirements_freeze.txt`

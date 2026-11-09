# Literatür Güncellemesi v3: AIForge-Doc Sahtecilik Lokalizasyonu (Faz 3)

**Tarih:** 9 Kasım 2026
**Yöntem:** `labrat:literature-research` ve `labrat:paper-check` (5. kontrol: atıf/BibTeX doğrulaması)
**Çıktılar:**
- `/home/claude/lit_update/references_v3.bib`: 77 kayıt (39 mevcut + 38 yeni). Mevcut anahtarların hiçbiri değiştirilmedi.
- `/home/claude/lit_update/related_work_en.tex`: 486 kelime, 3 alt bölüm.
- Bu rapor.

**Derleme testi:** `related_work_en.tex` ve `\nocite{*}`, IEEEtran.cls/IEEEtran.bst ile pdflatex→bibtex→pdflatex×2 akışında derlendi. Tanımsız atıf, BibTeX uyarısı ya da LaTeX hatası çıkmadı.

---

## 1. Ne arandı, nasıl doğrulandı

| Kanal | Kullanım | Not |
|---|---|---|
| WebSearch | YOLO/detektör tabanlı kurcalama tespiti, T-SROIE/T-IC13, CAFTB-Net, SAHI, YOLO11, DeepLabV3+, GT semantiği ve kısayol çalışmaları (2025–26), AIForge-Doc takip çalışmaları, GenText-Forensics/RealText-V2, yayımlanmış sürümler (FSTS, Omni-IML, IML-ViT, CTP-Net, TextSleuth) | Her konu için en az 2 farklı ifadeyle arandı |
| arXiv abs sayfaları (WebFetch) | Mevcut 17 arXiv kaydının ve yeni 8 arXiv kaydının başlık, yazar, tarih ve journal-ref alanları | Tüm kimlikler çözüldü |
| Crossref REST API (WebFetch) | Mevcut 22 DOI ve yeni 24 DOI: başlık, yazar sırası, mekân, yıl, sayfa | Tamamı eşleşti |
| Yayıncı ve mekân sayfaları | NeurIPS 2025 bildiri sayfası (FSTS), ICLR 2026 poster sayfası (Omni-IML), AAAI OJS (TextShield-R1) | Preprint kayıtları yayımlanmış sürüme güncellendi |
| Hugging Face Hub (MCP) | `Scam-AI` veri setleri, AIForge-Doc v1/v2 kartları, HF Papers meta verisi, RealText-V2 | §5'e bakın |
| Semantic Scholar citations API | v1 (2602.20569) için 9, v2 (2604.25213) için 5 atıf yapan çalışma | Hiçbiri v1 üzerinde lokalizatör eğitmiyor |
| Zenodo, DOAJ, GitHub | Helali (2026), T-SROIE kaynağı, DocDet | |

**Erişim kısıtları:** DBLP (`dblp.org`) robots.txt nedeniyle WebFetch'e kapalıydı; konteynerden doğrudan çıkış da proxy tarafından reddedildi. Bu yüzden `paper-search` CLI kullanılamadı ve DBLP doğrulaması yapılamadı. Yerine birincil kaynaklar kullanıldı: Crossref, arXiv ve mekânın kendi sayfası. alphaXiv MCP'de makale arama aracı yoktu, yalnız araştırmacı araçları vardı. Consensus kullanılmadı.

---

## 2. Yeni eklenen çalışmalar (38)

Konum kısaltmaları: **RW** Related Work · **M** Method · **D** Discussion · **E** Experiments

### 2a. Belge/metin sahteciliği, inpainting adli bilişimi, GT semantiği (16)

| Anahtar | Doğrulama | Tek satırlık ilgi | Nerede |
|---|---|---|---|
| `mareen2026tgif2` | arXiv 2603.28613 | TGIF genişletmesi (FLUX.1, rastgele maskeler). İnce ayarlı lokalizatörlerin nesne yanlılığı edindiğini gösteriyor; maske semantiği kısayolu ile paralel | RW §2.2, D |
| `nebioglu2026inpx` | arXiv 2602.00192 | INP-X: dedektörler düzenlenen bölgeye değil, VAE kaynaklı global spektral kaymaya yaslanıyor (özette doğruluk %91'den %55'e düşüyor). H4 ve işaretçi bulgusuyla paralel | RW §2.2, D (H4) |
| `giakoumoglou2026diffusionprint` | arXiv 2604.12443 | Difüzyon inpainting için üreteç parmak izi temelli lokalizasyon | RW §2.2 |
| `pandolfini2025localized` | arXiv 2512.16688 | Sentetik görüntü dedektörleri orta ve büyük alanlı inpainting'i yakalıyor; küçük alan zor kalıyor | RW §2.2, D (küçük alan) |
| `qu2026textshieldr1` | AAAI 2026, DOI 10.1609/aaai.v40i10.37814 | MLLM ve RL ile açıklanabilir kurcalanmış metin tespiti; TFR benchmark'ı | RW §2.1 |
| `zeng2026docshield` | arXiv 2604.02694 | Tespit, lokalizasyon ve açıklamayı birleştiren ajan tabanlı belge güvenliği | RW §2.1 |
| `zhang2026scrf` | arXiv 2608.02258 | Açık-küme görsel metin adli analizi (Rectified Flow, restorasyon tabanlı) | RW §2.1 |
| `guangjian2026defakerone` | arXiv 2605.14091 | Birleşik tespit ve lokalizasyon temel modeli (InternVL2 ve SAM2). Görüntü düzeyi ve piksel düzeyi birlikte | RW §2.3 |
| `helali2026processtwin` | Zenodo 10.5281/zenodo.23917285 (yazılım arşivi, hakemsiz) | AIForge-Doc v2'de eğitilmiş FCN ve "process-twin" kontrolü. Eş zamanlı çalışma; benign kontrollerimizin görüntü düzeyi karşılığı | RW §2.1, D |
| `leong2026docdet` | GitHub (arşivsel değil) | v1, v2 ve GPT4o-Receipt üzerinde karo tabanlı ResNet-18 ile sayfa sınıflandırması; piksel lokalizasyonu yok | RW §2.1 (tek cümle) |
| `song2024caftbnet` | Crossref 10.1145/3686158 (ACM TOMM) | CAFTB-Net: DOCFORGE-BENCH/ForensicHub'da adı geçen belge lokalizatörü | RW §2.1 |
| `wang2022tsroie` | DOAJ kaydı (DOI 10.11959/j.issn.2096-109x.2022035) | T-SROIE'yi tanıtan çalışma (986 görüntü, Çince) | RW §2.1 |
| `xu2026realitygap` | Crossref 10.1109/TIFS.2026.3705334 | İnsan eliyle hazırlanmış gerçek dünya kurcalanmış metin veri seti (TIFS 2026) | RW §2.1 |
| `shang2026maskspixels` | arXiv 2603.20193 | Nesne maskeleri değişmemiş pikselleri etiketliyor ve gerçek düzenlemeleri kaçırıyor; yalnız maskeye dayalı metrikler dedektörleri yanlış sıralıyor. **Bbox ve piksel GT tezimize en yakın çalışma** | RW §2.2, D §7.1 |
| `gentextforensics2026` | Challenge web sitesi ve HF `vankey/RealText-V2` (arşivsel değil) | ACM MM 2026 GenText-Forensics: tespit, lokalizasyon (mIoU) ve açıklama | RW §2.1 |
| `zhou2018rgbn` | Crossref 10.1109/CVPR.2018.00116 | RGB-N: Faster R-CNN ile kutu düzeyi manipülasyon tespiti, yani detektör formülasyonunun öncülü | RW §2.3, M (YOLO) |

### 2b. Tespit ve segmentasyon, küçük nesne, kutu denetimi, metrik (10)

| Anahtar | Doğrulama | Tek satırlık ilgi | Nerede |
|---|---|---|---|
| `redmon2016yolo` | Crossref 10.1109/CVPR.2016.91 | Tek aşamalı detektör ailesinin kaynağı | RW §2.3, M |
| `jocher2024yolo11` | Ultralytics resmî dokümanı (yazılım, v11.0.0, AGPL-3.0; makalesi yok, DOI "pending") | Kullanılan YOLO11 sürümü | M |
| `akyon2022sahi` | Crossref 10.1109/ICIP46576.2022.9897990 | Natif çözünürlükte karolama (slicing) ile küçük nesne çıkarımı | M (karolama), RW §2.3 |
| `cheng2023smallobject` | Crossref 10.1109/TPAMI.2023.3290594 | Küçük nesne tespiti üzerine derleme ve benchmark (medyan 32 px yükseklik bağlamı) | RW §2.3, D |
| `chen2018deeplabv3plus` | Crossref 10.1007/978-3-030-01234-2_49 | İkinci mimari (DeepLabV3+) | M |
| `he2017maskrcnn` | Crossref 10.1109/ICCV.2017.322 | Kutu ve maskeyi birlikte öğrenme; ortak piksel+bbox kaybı için emsal | M (joint loss), RW §2.3 |
| `lin2014coco` | Crossref 10.1007/978-3-319-10602-1_48 | COCO tarzı AP/mAP@IoU tanımı (YOLO değerlendirmesi) | M (metrikler) |
| `dai2015boxsup` | Crossref 10.1109/ICCV.2015.191 | Kutu etiketiyle segmentasyon denetimi | RW §2.2, D |
| `tian2021boxinst` | Crossref 10.1109/CVPR46437.2021.00540 | Yalnız kutu etiketiyle örnek segmentasyonu | RW §2.2, D |
| `maierhein2024metrics` | Crossref 10.1038/s41592-023-02151-z | Metrik seçiminin hedef yapıya göre yapılması (küçük nesne, eşik) | M (metrikler), D |

### 2c. Metinde zaten atıf yapılan ama bib'de olmayan klasik kaynaklar (12)

Bildiri kaynakçasındaki [40]–[51] maddelerinin karşılığıdır; her biri Crossref ile doğrulandı.
`ahmed1974dct`, `ojala2002lbp`, `haralick1973glcm`, `sauvola2000binarization`, `cortes1995svm`, `breiman2001rf`, `he2016resnet`, `deng2009imagenet`, `ronneberger2015unet`, `milletari2016vnet`, `efron1979bootstrap`, `kendall1938tau` → **M** (§5.2–5.6).
Not: Bildiri [41]'de Ojala başlığı kesik ("…with local binary"). Doğrusu "…with local binary patterns" (bib'de düzeltildi).

---

## 3. Mevcut kayıtlarda yapılan düzeltmeler

39 kaydın 39'u yeniden doğrulandı: 22'si Crossref DOI ile, 15'i arXiv abs sayfasıyla, 2'si yayımlanmış sürümün mekân sayfasıyla. Hepsi gerçek; uydurma kayıt yok. Esasa ilişkin düzeltmeler:

| # | Anahtar | Sorun | Düzeltme | Kaynak |
|---|---|---|---|---|
| 1 | `fsts2025` | arXiv preprint olarak kayıtlıydı | **NeurIPS 2025 Datasets & Benchmarks Track** (`@inproceedings`, cilt 38). Yazar adları bildiri sayfasındaki gibi: *Jian Jun Zhang*, *Wenkang Su* (arXiv'de "Jian Zhang", "Wenkan Su") | papers.neurips.cc |
| 2 | `qu2024omniiml` | arXiv preprint olarak kayıtlıydı | **ICLR 2026**. Yayımlanan başlık: "Omni-IML: Towards Unified **Interpretable** Image Manipulation Localization" | iclr.cc poster sayfası |
| 3 | `martineztornes2023receipt` | booktitle "ICDAR 2023 **Workshops**" | Crossref'e göre ana konferans kitabı: "Document Analysis and Recognition – ICDAR 2023" (s. 454–469) | Crossref |
| 4 | `li2025ditl2` | Başlık "DITL 2 :" | "DITL$^{2}$:" | Crossref ("DITL²") |
| 5 | `zhu2025mesorch` | Başlıkta HTML varlığı `&amp;` | `\&` | Crossref |

Biçimsel tamamlamalar (hata değil): Crossref'teki sayfa, cilt ve sayı bilgileri DOI'li 22 kayda eklendi. DOI'ler kanonik büyük harfle yazıldı. Ä/é/ç gibi karakterler IEEEtran/BibTeX güvenliği için LaTeX kaçışlarına çevrildi. `wang2022tamperedic13` için LNCS cilt numarası Crossref'te olmadığından eklenmedi; ECVA dosya kimliği 13688'e işaret ediyor, kamera-hazır öncesinde teyit edilmeli.

**GenD (Yermakov+ WACV 2026) notu:** bib kaydında **Mario Fritz zaten var** ve Crossref ile 4 yazar birebir tutuyor. "Fritz eksik" sorunu yalnız Vize raporundaki düz metin kaynakçasında. Ayrıca Bildiri §2'deki iddianın yönü ("unpaired … from 56.8 to 90.0") önceki incelemeye göre ters. Bu metin düzeltmesi bib kapsamı dışında, yapılması gerekiyor.

---

## 4. Doğrulanamayan veya kısmen doğrulanan öğeler

| Öğe | Durum | Karar |
|---|---|---|
| GenText-Forensics / RealText-V2 | Challenge sitesi ve HF veri seti var. **Genel bakış makalesi ya da BibTeX yok**; organizatörler açılış sayfasında adlandırılmamış | Arşivsel olmayan `@misc` olarak eklendi; yayımlanmış bir makale çıkarsa değiştirilmeli. Alternatif: 2. olan çözümün makalesi arXiv 2609.02097 (Liu+ 2026; doğrulandı, eklenmedi) |
| `wang2022tsroie` | DOAJ kaydı başlığı, 4 yazarı ve DOI'yi veriyor. DOI **Crossref'te çözülmüyor**; cilt, sayı ve sayfa yok | Eklendi; dergi sayfasından cilt ve sayfa tamamlanmalı |
| `leong2026docdet` | Depo mevcut ve içeriği doğrulandı; **makale, DOI ve lisans yok**; yıl, depo etkinliğine göre 2026 kabul edildi | Eklendi (arşivsel değil); IEEE hakemi zayıf bulabilir, dipnot olarak da verilebilir |
| `jocher2024yolo11` | Resmî atıf bloğu mevcut; **hakemli makalesi yok** | Ultralytics'in önerdiği biçimde eklendi |
| `efron1979bootstrap` | Crossref kaydında sayfa yok | Sayfa alanı boş bırakıldı (uydurulmadı) |
| `cheng2023smallobject` | Crossref kaydı erken erişim (yıl 2023), nihai cilt ve sayfa yok | Yalnız DOI ve yıl verildi |
| `maierhein2024metrics` | 73 yazar | İlk 5 yazar ve "others" |
| Venus-DeFakerOne | arXiv yazar alanı kurumsal: "GuangJian Team" | `{{GuangJian Team}}` olarak verildi |
| DBLP | Erişilemedi (robots) | Tüm kayıtlar Crossref, arXiv ve mekân sayfası ile doğrulandı. Hiçbir kayıt DBLP'ye karşı kontrol edilmedi |
| Joya15 (GitHub) | Bu görevde yeniden kontrol edilmedi | Eklenmedi |

---

## 5. AIForge-Doc v2 erişim durumu

- **HF repo kimliği:** `Scam-AI/AIForge-Doc-v2` (https://huggingface.co/datasets/Scam-AI/AIForge-Doc-v2). Son güncelleme 28 Eylül 2026.
- **Erişim:** *Gated, otomatik onay.* Ad, e-posta ve bir onay kutusu ("yalnız ticari olmayan araştırma; yayın yapılırsa makaleye atıf") doldurulduktan sonra erişim **otomatik** açılıyor. Lisans **CC BY-NC-SA 4.0**; kaynak veri setlerinin (CORD v2 CC BY 4.0, WildReceipt Apache 2.0, SROIE ICDAR araştırma kullanımı, XFUND CC BY 4.0) lisansları da geçerli.
- **İçerik:** 3.066 GPT-Image-2 sahtesi (Train 2.453 / Test 613), DocTamper uyumlu ikili maskeler ve `metadata.jsonl` (bbox, `v1_assigned_tool`). **Eşleşmiş authentic görüntüler HF reposunda değil**, kartta bağlantısı verilen "full package" zip'inde (Google Drive). Kart, authentic görüntülerin maske dışında sahteyle piksel düzeyinde aynı olduğunu söylüyor.
- **v1 ile eşleşme:** 3.066 spec'in 3.062'sinin v1'de karşılığı var (2.729 Gemini, 333 Ideogram). Bu, held-out generator deneyini spec-for-spec yapmaya olanak veriyor.
- **Maske semantiği:** Kart v2 maskesini "pixel-precise" diye tanımlıyor; ancak 4. adımda "yalnız tam alan bbox'ı geri yapıştırılır ve GT bu pikselleri işaretler" deniyor. Yani **v2 GT'si de alan kutusu**; bbox ve piksel GT bulgusu v2'ye doğrudan taşınabilir.
- **⚠ İşaretçi hakkında tutarsızlık:** v2 kartı 3. adımda 3 px **yeşil çerçeve** kullanıldığını ve "v1'in **kırmızı dolgusunun** renk taşmasına yol açtığını" yazıyor. Bildiri ve önceki inceleme ise v1 makalesinin §3.4'ünü "3-pixel bright-green rectangle" diye aktarıyor. Bildiride v1 işaretçisinin rengi ve biçimi atfedilmeden önce v1 makalesi §3.4 ile v1 HF kartı yeniden okunmalı. Bulgunuz verideki yeşil kalıntıdır ve ampirik olarak geçerli kalır; tutarsızlık yalnız kaynağa atıf cümlesini etkiler.
- **v3 ve diğer takip çalışmaları:** HF'de `Scam-AI` altında yalnız v1 ve v2 belge setleri var; "v3" (raj2026images25'in turu) için ayrı bir veri seti bulunamadı. Semantic Scholar'daki takip çalışmalarının hiçbiri v1 üzerinde piksel lokalizatörü eğitmiyor. Hakemsiz eğitimli çalışmalar Helali (v2) ve DocDet (v1+v2, sayfa sınıflandırma). Diğer atıflar zero-shot, insan çalışması ya da ilgisiz konular: pay-stub, kredi belgesi, ajan tabanlı PDF sahteciliği, 2606.19259.

---

## 6. "Box-level formulation" için ilgili çalışma paragrafı (İngilizce, 136 kelime)

> Manipulation localization was cast as box detection early on, when RGB-N attached a noise stream to a region-proposal detector \cite{zhou2018rgbn}, and tampered scene text is likewise scored per detected text instance \cite{wang2022tamperedic13,qu2025ostf}. Single-stage detectors \cite{redmon2016yolo}, here YOLO11 \cite{jocher2024yolo11}, regress boxes directly and are evaluated with COCO-style average precision \cite{lin2014coco}; for small targets on large pages, overlapping-tile inference at native resolution is the usual remedy \cite{akyon2022sahi,cheng2023smallobject}. Pixel segmenters such as U-Net and DeepLabV3+ \cite{ronneberger2015unet,chen2018deeplabv3plus}, by contrast, must turn a box label into a dense map, a gap that box-supervised segmentation handles explicitly \cite{dai2015boxsup,tian2021boxinst}. Joint box-and-mask training \cite{he2017maskrcnn} sits between the two. Because AIForge-Doc annotates the pasted field box \cite{wu2026aiforgedoc}, a detector matches the label geometry by design; we therefore treat the detector as a reference for what box ground truth can reward, not as evidence of pixel-level localization.

---

## 7. Bildiriye işlenmesi önerilenler (kısa)

1. Related Work bölümünü `related_work_en.tex` ile değiştirin. Bildiri §2'deki eski "Generic manipulation localization" paragrafındaki anahtarlar (TruFor, CAT-Net, MVSS, PSCC, IML-ViT, Mesorch, SAFIRE, Omni-IML) yeni metinde yok; sayfa sınırı izin verirse bu paragraf ayrıca korunabilir.
2. Yenilik iddiası şöyle sınırlanmalı: "bbox ve piksel GT ile eğitim ve puanlama, aynı benchmark'ta detektör ile segmentere karşı." En yakın çalışmalar `shang2026maskspixels` (doğal görüntüler, VLM düzenlemeleri, eğitim yok) ve `helali2026processtwin` (v2, görüntü düzeyi). İkisi de mutlaka anılmalı.
3. §5 Method'a `jocher2024yolo11`, `akyon2022sahi`, `chen2018deeplabv3plus`, `he2017maskrcnn` (ortak kayıp) ve `lin2014coco` (mAP) eklenmeli.
4. §7 Discussion için H4 ve işaretçi bulgusu `nebioglu2026inpx` ve `mareen2026tgif2` ile ilişkilendirilmeli; küçük alan tartışması `pandolfini2025localized` ve `cheng2023smallobject` ile desteklenmeli.
5. Kaynak sayısı 77'ye çıktı. Konferans sürümünde `\nocite{*}` kullanmayın, yalnız atıf yapılanlar listelensin.

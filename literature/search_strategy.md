# Arama Stratejisi — Literature Review Matrix (AIC515, AI-generated financial document forgery)

**Tarih:** 19 Ekim 2026 · **Araştırma kesim tarihi:** 17 Ekim 2026 · **Öncelikli pencere:** 2023-01-01 – 2026-10-17 (seminal eski çalışmalar: CAT-Net 2021/22, MVSS-Net 2021/23, PSCC-Net 2022, Tampered-IC13 2022, Geirhos 2020 ayrıca dahil edildi).

## 1. Kaynaklar ve erişim durumu

| Kaynak | Kullanım | Durum |
|---|---|---|
| arXiv API (export.arxiv.org) | 13 tema sorgusu, metadata + abstract | Kullanıldı (439 kayıt, 395 tekil) |
| Consensus (MCP connector) | 8 doğal dil sorgusu, hakemli venue/atıf bilgisi | Kullanıldı (78 kayıt) |
| alphaXiv (MCP connector) | 32 arXiv makalesinin tam metni | Kullanıldı |
| CrossRef API | DOI doğrulama (title query + DOI resolve), BibTeX metadata | Kullanıldı (tüm DOI'ler çözümlendi) |
| fetch_article_fulltext (Unpaywall/Springer/Elsevier) | arXiv dışı 5 hakemli makale | 1/5 tam metin (Receipt Dataset, HAL OA); DocTamper abstract; FFDN/Tampered-IC13/ICDAR23 kapalı erişim → abstract düzeyi |
| web_search | FFDN'nin tam adının/DOI'sinin tespiti, ICDAR 2023 yarışma özeti | Kullanıldı (2 sorgu) |
| OpenAlex | Oturum sonunda credential onaylandı → yalnızca atıf sayısı zenginleştirme (36/38 eşleşme, `openalex_cited_by` sütunu) | Arama aşamasında kullanılamadı; tarama sonrası tamamlayıcı |
| Elicit (MCP) | — | Kullanılmadı (Consensus + arXiv yeterli kapsama sağladı) |

## 2. Anahtar kelime grupları (arXiv sorguları, cat:cs.CV ağırlıklı)

| Tema | Sorgu (özet) | Kayıt | 2023+ |
|---|---|---|---|
| T1 Document forgery detection/localization | document AND (tampering OR forgery) AND (localization OR detection) | 41 | 31 |
| T2 Image manipulation/forgery localization | "image manipulation localization" OR "image forgery localization" | 60 | 50 |
| T3 AI-generated document tampering | document AND (AI-generated OR diffusion) AND (tamper OR forg) | 5 | 5 |
| T4 Generative inpainting forensics | inpainting AND (detection OR forensics OR localization) | 60 | 41 |
| T5 Frequency-domain forgery detection | frequency AND (forgery OR manipulation) AND (detection OR localization) | 60 | 43 |
| T6 Robust image forensics | robust AND "image forensics" AND (JPEG OR compression OR resizing) | 16 | 10 |
| T7 Domain generalization / cross-generator | (generalization OR cross-generator OR unseen generators) AND (AI-generated image detection OR deepfake detection) | 60 | 58 |
| T8 Shortcut learning | "shortcut learning" AND (forgery OR deepfake OR forensics OR AI-generated) | 13 | 12 |
| T9 Boundary artifact detection | (boundary artifact OR edge artifact) AND (forgery OR manipulation OR splicing) | 15 | 10 |
| T10 Pixel-level forgery segmentation | pixel-level AND (forgery OR tampering) AND segmentation | 18 | 15 |
| T11 Annotation bias & benchmark reliability | (annotation OR benchmark) AND (bias OR reliability OR flaws) AND (forgery OR manipulation detection OR forensics) | 60 | 52 |
| T12 Small-object / small-region segmentation | (small object OR small region) AND segmentation AND (forgery OR tampering OR text) | 11 | 9 |
| T13 Tampered text detection | (tampered text OR text tampering OR text manipulation) AND (detection OR localization) | 20 | 19 |

Consensus sorguları: document image tampering localization; tampered text detection dataset; IML noise residual transformer; generative inpainting detection diffusion; shortcut learning deepfake generalization; benchmark annotation errors forgery localization reliability; receipt invoice forgery AI-generated; JPEG compression robust forgery localization DCT.

Hedefli isim aramaları (arXiv `ti:`/CrossRef `query.title`): DocTamper, FFDN, ADCD-Net, Tampered-IC13, T-SROIE, OSTF, RTM, TextSleuth, CTP-Net, DITL2, DocQT, Find-it/Receipt dataset, ICDAR 2023 DTTI, GPT4o-Receipt, TruFor, CAT-Net, MVSS-Net, IML-ViT, PSCC-Net, Mesorch, SAFIRE, Omni-IML, TGIF, COCO-Inpaint, GenD (Yermakov), Geirhos, Smeu, GIFTBench, Aoudi, BR-Gen.

## 3. Dahil etme / hariç tutma kriterleri

**Dahil:** (i) belge/metin görüntüsü sahtecilik tespiti veya lokalizasyonu; **veya** (ii) genel image manipulation localization'da yaygın baz çizgi (TruFor, CAT-Net, MVSS-Net, PSCC-Net, IML-ViT, Mesorch, SAFIRE, Omni-IML); **veya** (iii) üretken inpainting forensics; **veya** (iv) generalization / shortcut learning / benchmark-annotation güvenilirliği çalışması doğrudan görüntü forensics'e uygulanabilir; (v) kimliği CrossRef veya arXiv'den doğrulanmış; (vi) en az abstract'a erişilmiş.

**Hariç:** video/ses/konuşma deepfake (T8–T9'daki çoğu kayıt), yüz deepfake mimari çalışmaları (generalization için yalnızca Yermakov 2026 ve Smeu 2025 tutuldu), watermarking, OCR/doküman anlama, kopya-taşıma (copy-move) özel yöntemler, tıbbi/uydu görüntüleri, doğrulanamayan DOI, yalnızca başlıktan ilgisi anlaşılamayan kayıtlar, 2 düşük kaliteli/venue'su belirsiz ön baskı (örn. 2402.13545 iki aşamalı T-SROIE raporu), metodolojik tekrar (RITA 2509.20006 — matrise alınmadı, GIFTBench tercih edildi).

## 4. PRISMA tarzı akış

| Aşama | Sayı |
|---|---|
| Tanımlanan kayıtlar (arXiv 439 + Consensus 78 + hedefli/web 10) | 527 |
| Tekilleştirme sonrası (arXiv 395 tekil + Consensus/hedefli tekil ~70) | ≈465 |
| Başlık/abstract taramasından geçen (aday listesi) | 48 |
| Kimlik doğrulaması yapılan (CrossRef/arXiv) | 48 (48 doğrulandı, 0 UNVERIFIED) |
| Tam değerlendirme (abstract veya tam metin okunan) | 42 |
| Hariç tutulan (düşük kalite 1, metodolojik tekrar 1, kapsam dışı 2) | 4 |
| **Matrise dahil edilen** | **38** |
| — tam metin düzeyinde incelenen | 27 (26 arXiv tam metin + 1 OA PDF) |
| — abstract/metadata düzeyinde | 11 (DocTamper, FFDN, Tampered-IC13, ICDAR23, Dong TCE24, DITL2, Smeu CVPR25, Aoudi 2026 + 3 seed: AIForge-Doc v1/v2, DocForge-Bench, ADCD-Net hafif okuma) |

Not: Seed makaleler (AIForge-Doc v1/v2, DocTamper, FFDN, ADCD-Net, DOCFORGE-BENCH) başka bir alt-ajan tarafından derinlemesine incelendiği için burada abstract/metadata düzeyinde tutuldu.

## 5. Kanıt sınıflandırması

- **VERIFIED**: DOI/arXiv kimliği CrossRef/arXiv'den çözümlendi ve içerik birincil kaynaktan okundu.
- **REPORTED**: değer ikincil kaynaktan (örn. DocForge-Bench'in FFDN için bildirdiği F1) veya yalnızca abstract'tan alındı; matriste açıkça işaretlendi.
- **not extracted**: tam metin mevcut olsa da sütun başlıkları güvenle eşleştirilemeyen tablo değerleri raporlanmadı; "columns inferred" ibaresi geçici (provisional) okuma anlamına gelir.
- **UNKNOWN**: lisans/kod bilgisi birincil metinde bulunamadı.

## 6. Bilinen sınırlamalar

1. OpenAlex arama aşamasında kullanılamadı (credential onayı tarama bittikten sonra geldi); bu nedenle ileri atıf (cited-by) genişlemesi sistematik yapılmadı ve ileri/geri atıf takibi Consensus ve makale referans listeleri üzerinden kısmen yapıldı. Oturum sonunda OpenAlex yalnızca atıf sayısı zenginleştirmesi için kullanıldı (36/38 eşleşme, `openalex_cited_by` sütunu; DITL2 ve MLLM-doc eşleşmedi).
2. Kapalı erişimli 4 hakemli makalenin (FFDN, DocTamper, Tampered-IC13, ICDAR23) sonuç tabloları okunmadı; FFDN sonuçları DocForge-Bench'ten REPORTED olarak alındı.
3. T-SROIE veri setinin birincil kaynağı (Wang et al. 2022, Chinese Journal of Network and Information Security) doğrulanamadı; matrise alınmadı, yalnızca DocForge-Bench/DiffNet üzerinden anıldı. CTAD, TTD, CatmullRom-Net, DocMMIR/DocLoc adları için arXiv/CrossRef/Consensus'ta eşleşen birincil kayıt bulunamadı → UNKNOWN.

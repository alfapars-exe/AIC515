# Literatür Doğrulama Raporu — AIForge-Doc v1 Projesi (AIC515)

*Tarih: 9 Kasım 2026 · Kapsam: `Vize_Raporu_Taslak.md` §2 ve §2.5, `literature/*` · Yöntem: labrat literature-research prosedürü (mevcut işin haritalanması → web/Consensus/HF/GitHub araması ile boşluk doğrulama → sayıların birincil kaynaktan teyidi). Proje dosyaları değiştirilmedi.*

## Özet

- **Gap (a) — "v1 üzerinde eğitilmiş model yalnız Joya15'te var"** kısmen **çürütüldü**: 3 Kasım 2026'da Zenodo/GitHub'da yayımlanan **Helali (2026)** çalışması AIForge-Doc **v2** üzerinde denetimli bir dedektör eğitmiş ve "process-twin" (içerik değişmeden yeniden örnekle-yapıştır hattından geçirilmiş orijinal) kontrolü ile **üretim hattı artefaktı kısayolunu** bağımsız olarak ölçmüştür. Ayrıca GitHub'da v1+v2 ile eğitilmiş en az iki hobi deposu daha var (DocDet, Joya15). Hiçbiri hakemli değil; hiçbiri **v1 üzerinde piksel lokalizasyonu + GT geometrisi + işaretçi analizi** yapmıyor.
- **Gap (b) — bbox-vs-piksel GT geometrisi ve yeşil işaretçi kalıntısı**: **doğrulandı**. v1 makalesi §3.4'te "3 piksellik parlak yeşil dikdörtgen çizildiğini" ve §3.3/§4.2'de GT maskesinin **alan bbox'ı** olduğunu açıkça yazıyor; ancak bulduğumuz hiçbir kaynak bu kalıntıyı ölçmüyor ya da GT geometrisini değiştirerek eğitmiyor. Güven: **ORTA-YÜKSEK** (3+ ifade ile arandı, GitHub/HF/Consensus/arXiv tarandı; Semantic Scholar'a göre v1'e 9 atıf var ve hiçbiri eğitim yapmıyor).
- Benchmark tablosundaki DTD, FFDN, ADCD-Net, TruFor, AIForge-Doc v1/v2 ve DOCFORGE-BENCH değerleri birincil kaynaktan **teyit edildi**; v3 (Raj ve ark.) ayrıntıları arXiv hız sınırı nedeniyle bu oturumda yeniden okunamadı.

## Gap doğrulaması (bulgular + URL'ler)

| # | Kaynak | Tür / tarih | AIForge-Doc kullanımı | Raporladığı | Gap'e etkisi |
|---|---|---|---|---|---|
| 1 | Helali, B. "Beyond Zero-Shot on AIForge-Doc v2: A Trained Detector and a Process-Twin Control" — https://github.com/bilLkarkariy/aiforge-doc-v2-process-twins · DOI 10.5281/zenodo.23917285 | Hakemsiz ön baskı + kod/ağırlık, 3 Kas 2026 | **v2** resmî eğitim bölümünde 371k parametreli FCN eğitimi; 612 test çifti | Görüntü düzeyi AUC 0.989 (sahte vs orijinal); **process-twin'lerin %37.7'si işaretleniyor** (≤%10 ön-kayıtlı tavan aşıldı), twin-vs-orijinal AUC 0.880; sahte > kendi twin'i 602/612. Piksel F1/IoU **yok**, v1 **yok**, işaretçi **yok** | (a)'yı daraltır: "AIForge-Doc'ta denetimli eğitim yok" artık söylenemez; "hat artefaktı kısayolu" fikri v2'de bağımsız olarak gösterildi. Rapor bunu **önceli (concurrent) çalışma** olarak anmalı. |
| 2 | Joya15, *Multi-Generator-Deepfake-Document-Detection-and-Tamper-Localization* — https://github.com/Joya15/Multi-Generator-Deepfake-Document-Detection-and-Tamper-Localization | GitHub, hakemsiz, 2026 | v1 + v2 + GPT4o-Receipt (11.962 örnek) | Ana model **Tiny U-Net** (2 epoch): piksel F1 0.322 / IoU 0.192 / px-AUC 0.883 (birleşik test). SegFormer-lite çapraz: v2→v1 F1 0.044, v1→v2 0.0001 | Raporda "SegFormer-Lite eğitip" ifadesi düzeltilmeli (ana model Tiny U-Net; SegFormer-lite yalnız çapraz deney). |
| 3 | leongyiquan/DocDet — https://github.com/leongyiquan/DocDet | GitHub, hakemsiz, 2026 | v1 + v2 + GPT4o-Receipt ile eğitim/test (850 sayfa) | Karo (tile) sınıflayıcı ResNet-18 + yüksek-frekans başlığı; sayfa AUC 0.80; lokalizasyon yok | Görüntü düzeyi; GT geometrisine dokunmuyor. |
| 4 | Raj ve ark. 2026, arXiv:2609.13617 (v3) — https://arxiv.org/abs/2609.13617 | Ön baskı, Eyl 2026 | v3 (ChatGPT Images 2.5) zero-shot | Abstract: "localisation remains weak"; dedektör kontrollü sette %68.6, çevrimiçi görüntülerde %35.9 | Raporun aktardığı "alfa maskesi / 0.921 vs 0.918" ayrıntısı **bu oturumda teyit edilemedi** (arXiv HTML/PDF hız sınırı); sayfa/tablo numarası eklenerek yeniden kontrol edilmeli. |
| 5 | Venus-DeFakerOne, arXiv:2605.14091 — https://arxiv.org/abs/2605.14091 | Ön baskı, May 2026 | **Kullanmıyor** (belge seti: DocTamper, T-SROIE, RTM, OSTF, SACP, RIFLC + 2.3M özel) | Belge lokalizasyon F1: DocTamper FCD 80.8 / SCD 74.9 / Test 73.3, T-SROIE 81.7, IC13 90.5, OSTF 90.6, RTM 59.1 (ort. 78.7) | AIForge-Doc'ta eğitim/değerlendirme yok → gap sürüyor. |
| 6 | DocShield arXiv:2604.02694; SEED arXiv:2606.21138; GenText-Forensics (ACM MM 2026, RealText-V2) — https://gentext-forensics-acm-mm-2026.github.io/datasets.html | Ön baskılar / yarışma | AIForge-Doc'u yalnız **ilgili çalışma** olarak anıyor (DocShield'da hiç geçmiyor) | — | Gap'i etkilemez. |
| 7 | Scam AI araştırma sayfası — https://scam.ai/research ; HF org — https://huggingface.co/Scam-AI | Veri sahibi | v1 (4.061), v2 (3.066), v3 (giriş gerekli); hepsi CC BY-NC-SA 4.0, e-posta kapılı | Fine-tuning veya işaretçi artefaktı hakkında **hiçbir ifade yok** | Veri sahibinin kendisi de artefaktı raporlamamış. |

Ek taramalar: Semantic Scholar (Consensus üzerinden) v1 için 9, v2 için 5 atıf gösteriyor; tespit edilen atıf yapanlar (v2, v3, DOCFORGE-BENCH, GPT4o-Receipt, AgentForge-Bench, kimlik belgesi anketi Das 2026, SEED) zero-shot ya da yalnız anma düzeyinde. Kaggle'da sonuç çıkmadı. GitHub kod aramasında kalan eşleşmeler (itsfraud, DocForge-VLM, sigil, Vellum-AI, techjam…) veri setini yalnız README'de listeleyen ya da başka veri kullanan hobi projeleri.

**Sonuç:** Savunulabilir iddia şu biçimde yazılmalı: *"AIForge-Doc v1 üzerinde eğitim maskesinin geometrisini kontrollü değiştiren, piksel-değişim/glif GT'si türeten ve v1 makalesinde (§3.4) tarif edilen yeşil işaretçinin kalıntısını ölçen hakemli ya da hakemsiz bir çalışmaya rastlanmamıştır; v2'de hat-artefaktı kontrolü yapan eş zamanlı hakemsiz bir çalışma (Helali 2026) vardır."*

## Benchmark tablosu

Metrikler piksel düzeyi F1 (ayrı belirtilmedikçe); ✔ = bu oturumda birincil kaynaktan teyit edildi; (R) = ikincil kaynaktan/önceki matristen aktarıldı, yeniden doğrulanmadı.

**A. Denetimli (in-domain) belge sahteciliği lokalizasyonu**

| Yöntem | DocTamper-Test | DocTamper-FCD | DocTamper-SCD | T-SROIE | Kaynak |
|---|---|---|---|---|---|
| DTD (CVPR 2023) | F 0.792 / IoU 0.828 ✔ | F 0.816 / IoU 0.749 ✔ | F 0.754 / IoU 0.691 ✔ | F 0.9927 ✔ | Qu ve ark., Tablo 3/4/6 |
| FFDN (ECCV 2024) | F 0.86 / IoU 0.90 ✔ (DTD* 0.79/0.84) | F 0.92 / IoU 0.88 ✔ | F 0.81 / IoU 0.75 ✔ | F 0.993 ✔ | Chen ve ark., Tablo 1/8; JPEG Q75'te F 0.75 ✔ |
| ADCD-Net (ICCV 2025) | F1 0.806 ✔ | 0.815 ✔ | 0.740 ✔ (ort. 0.787) | — | Wong ve ark., Tablo 1; pristine FAR 0.012 ✔; DTD'nin 1 px kaydırmada F1 düşüşü %73.3 ✔ |
| CAT-Net | 0.700 ✔ (DTD makalesi) / 0.721 (ADCD) ✔ | 0.553 / 0.741 ✔ | 0.631 / 0.670 ✔ | — | |
| MVSS-Net | 0.431 ✔ | 0.424 ✔ | 0.414 ✔ | — | |
| TruFor (yeniden eğitilmiş / resmî ckpt) | 0.691 / 0.168 ✔ | 0.667 / 0.109 ✔ | 0.594 / 0.281 ✔ | — | ADCD-Net Tablo 1 |
| DiffNet (arXiv:2606.22285) | ForensicHub Doc protokolü ort. F1 0.515 vs ADCD 0.450 vs FFDN 0.301 (R) | | | | |
| DeFakerOne (arXiv:2605.14091) | 73.3 ✔ | 80.8 ✔ | 74.9 ✔ | 81.7 ✔ (IC13 90.5, OSTF 90.6, RTM 59.1) | 2.3M özel örnekle eğitim |

**B. Sıfır-atış (zero-shot) belge setleri — DOCFORGE-BENCH Pixel-F1@0.5 (Ek Tablo 4 ✔)**

| Yöntem | DocTamper | T-SROIE | RTM | T-IC13 | Receipt | FantasyID | MixTamper | FSTS | Ort. F1 / AUC (Tablo 3 ✔) |
|---|---|---|---|---|---|---|---|---|---|
| TruFor | 0.154 | 0.664 | 0.095 | 0.108 | 0.203 | 0.296 | 0.689 | 0.522 | 0.34 / 0.83 |
| CAT-Net | 0.672 | 0.274 | 0.042 | 0.038 | 0.234 | 0.155 | 0.695 | 0.512 | 0.33 / 0.82 |
| DTD | 0.682† | 0.021 | 0.024 | 0.140 | 0.003 | 0.036 | 0.106 | 0.242 | 0.16 / 0.83 |
| FFDN | 0.736† | 0.043 | 0.025 | 0.156 | 0.019 | 0.068 | 0.072 | 0.219 | 0.17 / 0.84 |
| ADCD-Net | 0.762† | 0.176 | 0.064 | 0.115 | 0.043 | 0.057 | 0.331 | 0.320 | 0.23 / 0.79 |
| PSCC-Net | 0.134 | 0.002 | 0.004 | 0.046 | 0.015 | 0.046 | 0.470 | 0.316 | 0.13 / 0.73 |
| IML-ViT | 0.071 | 0.015 | 0.025 | 0.040 | 0.010 | 0.023 | 0.340 | 0.130 | 0.08 / 0.67 |
| SAFIRE | 0.102 | 0.016 | 0.021 | 0.093 | 0.009 | 0.044 | 0.387 | 0.124 | 0.10 / 0.58 |
| MVSS-Net | 0.012 | 0.002 | 0.004 | 0.059 | 0.002 | 0.002 | 0.142 | 0.025 | 0.03 / 0.63 |

† in-domain (eğitim verisi DocTamper). Mesorch DOCFORGE-BENCH metninde yer almıyor; doğal görüntü ortalaması F1 0.677 (R, AAAI 2025). TruFor doğal görüntü ortalaması F1 0.785 (en iyi eşik) / 0.696 (sabit), görüntü AUC 0.857 ✔ (CVPR 2023 Tablo 1–2).

**C. AI-inpainting belge setleri**

| Set | Yöntem | Protokol | Sonuç | Kaynak |
|---|---|---|---|---|
| AIForge-Doc v1 (test 812) | TruFor | zero-shot | img-AUC 0.751 [.726,.776], AP 0.709, **IoU 0.358, F1 0.434**, px-AUC 0.916 ✔ | Wu ve ark. Tablo 2 |
| | DocTamper (DTD) | zero-shot | AUC 0.563, IoU 0.020, F1 0.030, px-AUC 0.675 ✔ | |
| | GPT-4o yargıç | zero-shot | AUC 0.509 ✔ | |
| | Tiny U-Net (Joya15) | eğitimli, hakemsiz | px F1 0.322 / IoU 0.192 (v1+v2 birleşik test) | GitHub |
| | **Bu proje (U-Net, bbox-GT)** | eğitimli, 3 seed | AUPRC 0.950±0.011, F1@τ* 0.846±0.015, IoU 0.789 (resmî test); piksel-GT ile aynı tahminler F1 0.45 | Vize raporu §4.b |
| AIForge-Doc v2 (GPT-Image-2) | İnsan 2AFC / TruFor / DocTamper / öz-yargıç | zero-shot | 0.501 / AUC 0.599 / 0.585 / 0.532 ✔; kalibrasyon setlerinde 0.962 / 0.852 ✔ | Wu ve ark. abstract |
| | Helali FCN | eğitimli, hakemsiz | img-AUC 0.989; process-twin %37.7 işaretlendi; piksel metrik yok | Zenodo |
| TGIF (WIFS 2024) | PSCC-Net, MVSS-Net++, CAT-Net, TruFor… | zero-shot | tam yeniden üretilmiş görüntülerde F1 ≤ 0.1 (R) | Mareen ve ark. |

## 2025–2026 yeni çalışmalar (matris 17 Ekim 2026 kesiminde yok)

1. **Helali (2026)**, *Beyond Zero-Shot on AIForge-Doc v2* — Zenodo 10.5281/zenodo.23917285 (3 Kas 2026). v2'de eğitimli dedektör + process-twin kontrolü; projenin benign kontrolleriyle doğrudan karşılaştırılmalı.
2. **TGIF2** — Mareen ve ark., arXiv:2603.28613 / EURASIP J. Inf. Security 2026. FLUX.1 düzenlemeleri ve rastgele (semantik olmayan) maskeler; FR görüntülerde fine-tuning'in etkisini ve maske-nesne yanlılığını ölçer → maske geometrisi/yanlılık tartışmasına destek.
3. **INP-X: AI-Generated Image Detectors Overrely on Global Artifacts** — Nebioglu ve ark., arXiv:2602.00192. İnpaint bölgesi dışındaki pikseller geri konunca ticari dedektörler %91→%55; "global hat artefaktı" kısayolunun doğrudan kanıtı (projenin H4/işaretçi bulgusuyla paralel).
4. **DiffusionPrint** — Giakoumoglou ve ark., arXiv:2604.12443. Difüzyon inpainting için öğrenilmiş üretici parmak izi; TruFor'a ikinci modalite olarak eklenince görülmemiş maske türlerinde +%28.
5. **Detecting Localized Deepfakes** — Pandolfini ve ark., arXiv:2512.16688. Tam-görüntü dedektörleri <%20 alanlı düzenlemelerde çöküyor; küçük-bölge zorluğunun bağımsız teyidi.
6. **TextShield-R1** — Qu ve ark., AAAI 2026, arXiv:2602.19828. RL tabanlı MLLM + 45k görüntülük TFR benchmark'ı (16 dil, 10 teknik); belge metni için yeni geniş test seti.
7. **DocShield / RealText-V1** — Zeng ve ark., arXiv:2604.02694. Görsel + anlamsal ipuçlarını birleştiren ajan çerçevesi; T-IC13'te GPT-4o'ya karşı +%23.4 F1.
8. **GenText-Forensics Challenge (ACM MM 2026) / RealText-V2** — 20k+ çok dilli örnek, piksel maskesi + açıklama; SEED (arXiv:2606.21138) ve MSU raporu (DTD + Qwen3-VL) tipik çözümler.
9. **Open-Set Visual Text Forensics via SC-RF** — Zhang ve ark., arXiv:2608.02258. Üretici tabanlı "onarım maliyeti" ile lokalizasyon; görülmemiş metin düzenleme kalıplarında sıfır-atış.
10. **Unmask Tampering (CCS 2025)** ve **Shao ve ark. 2025 (doğal bozulmalara dayanıklılık, TSorie-CRP)** — yeniden çekim/bozulma altında belge lokalizasyonu; H3 dayanıklılık tartışması için bağlam. Ayrıca **Das ve ark. 2026** kimlik belgesi anketi (SDGI: Latin-dışı yazıda inpainting kararsızlığı) XFUND alt kümesi için ilgili.

## Veri seti seçiminin gerekçesi

Ders listesi (Oxford Pets, ISIC 2018, Kvasir-SEG, MVTec AD…) "derin öğrenmeyle sınıflandırma ve/veya segmentasyon" şablonunu örnekler; AIForge-Doc v1 aynı şablona birebir oturur: HF kartı görevleri `image-classification` + `image-segmentation` olarak etiketler, veri 0/255 ikili maskeli 4.061 görüntüdür ve proje hem ResNet-18 sınıflandırma hem U-Net segmentasyon içerir. Kvasir-SEG/ISIC ile aynı "küçük ön plan, ikili maske" yapısına sahip olup (medyan kurcalanmış alan %0.98) daha ağır sınıf dengesizliği ve MVTec-benzeri bir "anomali lokalizasyonu" karakteri taşır. Erişim e-posta kapılı ancak otomatik onaylı, lisans CC BY-NC-SA 4.0 (akademik kullanım serbest; veri yeniden dağıtılmaz, yalnız türetilmiş maskeler/kod paylaşılır) ve makalede yayımlanmış zero-shot referans sayıları (TruFor IoU 0.358 / F1 0.434) karşılaştırma tabanı sağlar; raporda bu üç nokta ve dersi veren hocanın onayının alındığı bir cümleyle belirtilmelidir.

## Rapora yansıtılması gereken düzeltmeler

1. **§2.2 ve §2.5 gap ifadesi**: Helali (2026) eklenmeli; "AIForge-Doc'ta yalnız zero-shot sonuçlar var" satırı "v1'de kontrollü GT deneyi yok; v2'de görüntü düzeyi eğitimli bir dedektör ve hat-artefaktı kontrolü (Helali 2026, hakemsiz) var" olarak yeniden yazılmalı. DocDet deposu da dipnotta anılabilir.
2. **Joya15 tanımı**: "SegFormer-Lite eğitip" → "Tiny U-Net çok görevli model (piksel F1 0.322) eğitip, SegFormer-lite ile çapraz-veri deneyi yapıp".
3. **Yeşil işaretçi atfı**: v1 makalesinde ifade §3.3 değil **§3.4 (Deployed tools, Gemini maddesi)**: "we draw a 3-pixel bright-green rectangle on the context crop to mark the edit region". Bu, kalıntının Gemini'de %83.6 / Ideogram'da %1.5 olmasını da açıklar; rapora eklenmeli.
4. **bbox-GT alıntısı**: "pixel-precise" iddiasına karşı makalenin kendi tanımı doğrudan alıntılanmalı: §3.3 adım 5 "255 at the exact field bbox" ve §4.2 "Tampered region = the exact field bounding box (tight, no padding)".
5. **v3 (Raj ve ark.) ayrıntıları** (alfa maskesi, px-AUC 0.921 vs 0.918): sayfa/tablo numarasıyla yeniden teyit edilmeli; bu oturumda doğrulanamadı.
6. **DOCFORGE-BENCH**: makalenin kendi ifadesi "all eight datasets predate the era of generative AI editing" gap gerekçesine eklenebilir; Mesorch'un bu benchmark'ta yer almadığı not edilmeli (§2.3'teki liste Mesorch'u DOCFORGE-BENCH ile ilişkilendirmiyor ama matriste S03 satırı Mesorch'u sayıyor → düzeltilmeli).
7. **Literatür matrisi**: yukarıdaki 10 çalışma (özellikle TGIF2, INP-X, DiffusionPrint, TextShield-R1, DocShield — hepsi kesim tarihinden önce yayımlanmış) eklenmeli; references.bib'e Helali için Zenodo DOI'li kayıt girilmeli.
8. **Teyit edilen sayılar** (değişiklik gerekmez): DTD 0.792/0.828; FFDN 0.86/0.92/Q75 0.75; ADCD-Net 0.787 / %20.79 / FAR 0.012; TruFor v1 0.751/0.358/0.434; v2 0.599/0.585/0.501; DOCFORGE-BENCH Tablo 3–4.

## Kaynaklar

- Wu ve ark. 2026, AIForge-Doc v1 — https://arxiv.org/abs/2602.20569 · HF: https://huggingface.co/datasets/Scam-AI/AIForge-Doc-v1
- Wu ve ark. 2026, AIForge-Doc v2 — https://arxiv.org/abs/2604.25213 · HF: https://huggingface.co/datasets/Scam-AI/AIForge-Doc-v2
- Raj ve ark. 2026, ChatGPT Images 2.5 on Forgery Tasks (v3) — https://arxiv.org/abs/2609.13617
- Scam AI araştırma listesi — https://scam.ai/research · https://huggingface.co/Scam-AI
- Helali 2026, process-twin — https://github.com/bilLkarkariy/aiforge-doc-v2-process-twins · https://zenodo.org/records/23917285
- Joya15 — https://github.com/Joya15/Multi-Generator-Deepfake-Document-Detection-and-Tamper-Localization
- DocDet — https://github.com/leongyiquan/DocDet
- Zhao ve ark. 2026, DOCFORGE-BENCH — https://arxiv.org/abs/2603.01433
- Qu ve ark. 2023, DocTamper/DTD — https://openaccess.thecvf.com/content/CVPR2023/papers/Qu_Towards_Robust_Tampered_Text_Detection_in_Document_Image_New_Dataset_CVPR_2023_paper.pdf
- Chen ve ark. 2024, FFDN — https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04834.pdf
- Wong ve ark. 2025, ADCD-Net — https://arxiv.org/abs/2507.16397 · https://openaccess.thecvf.com/content/ICCV2025/papers/Wong_ADCD-Net_Robust_Document_Image_Forgery_Localization_via_Adaptive_DCT_Feature_ICCV_2025_paper.pdf
- Guillaro ve ark. 2023, TruFor — https://openaccess.thecvf.com/content/CVPR2023/papers/Guillaro_TruFor_Leveraging_All-Round_Clues_for_Trustworthy_Image_Forgery_Detection_and_CVPR_2023_paper.pdf
- Venus-DeFakerOne — https://arxiv.org/abs/2605.14091
- Dhouib ve ark. 2026, DiffNet — https://arxiv.org/abs/2606.22285
- Mareen ve ark. 2026, TGIF2 — https://arxiv.org/abs/2603.28613
- Nebioglu ve ark. 2026, INP-X — https://arxiv.org/abs/2602.00192
- Giakoumoglou ve ark. 2026, DiffusionPrint — https://arxiv.org/abs/2604.12443
- Pandolfini ve ark. 2025, Detecting Localized Deepfakes — https://arxiv.org/abs/2512.16688
- Qu ve ark. 2026, TextShield-R1 — https://arxiv.org/abs/2602.19828
- Zeng ve ark. 2026, DocShield — https://arxiv.org/abs/2604.02694
- Wong ve ark. 2026, SEED — https://arxiv.org/abs/2606.21138 · GenText-Forensics — https://gentext-forensics-acm-mm-2026.github.io/
- Zhang ve ark. 2026, SC-RF — https://arxiv.org/abs/2608.02258
- Ren ve ark. 2026, Can Loan Processors Spot AI-Edited Statements — https://arxiv.org/abs/2610.30758 (AIForge-Doc kullanmıyor; insan çalışması)

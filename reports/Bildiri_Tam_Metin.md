# What Do Document Forgery Localizers Learn from Box Masks? Pixel-Change Ground Truth, Shortcut Audits and a Hidden Marker Artefact in AI-Inpainted Financial Documents

> **ERRATA (9 Kasım 2026, Faz 3 incelemesi — bu belge tarihsel kayıttır; güncel ve bağlayıcı metin `paper/` altındaki IEEE bildirisi ve `phase3/results/tables_p3.md`'dir):**
> 1. **Faz 1 Pixel-AUPRC şişkin:** crop başına ≤20k negatif alt-örneklemeyle hesaplanmıştı (0.947 / 0.939). Tüm piksellerle kesin değer **0.805 / 0.792** (`phase3/results/phase1_exact_auprc.csv`). Faz 2 sayıları zaten kesindir.
> 2. **Tek tohum etiketleri:** "box modeli authentic val görüntülerinin %44'ünü işaretler" değeri yalnız s44'e aittir; 3 tohum ortalaması **%26** (>0.1 % piksel @0.5). Görüntü-düzeyi AUROC'lar (0.892 / 0.892 / 0.799) 3 tohum ortalamasıdır ve doğrudur. Tablo 9d'deki değerler 3 tohum ortalamasıyla değiştirilmelidir.
> 3. **GenD (Yermakov ve ark., WACV 2026):** iddianın yönü ters aktarılmıştı; 56.8 → 90.0 sayıları kaynağa yer gösterilerek doğrulanamadığı için kullanılmamalıdır.
> 4. **"İnsan doğrulaması":** İlk GT denetimi VLM destekli tutarlılık denetimidir. 10 Kasım 2026'da aynı 100 çift yazar tarafından elle denetlenmiştir (tek değerlendirici): piksel-değişim GT 36 uygun / 64 kısmen uygun / 0 hatalı, resmî kutu 99/100 uygun (`phase2/gt/human_verification_summary.md`). Bu belgedeki "insan doğrulaması bekliyor / yapılmadı" ifadeleri bu tarihten öncesine aittir.
> 5. **Kaynakça:** doğrulanmış güncel liste `literature/references_v3.bib` (77 kayıt; Helali 2026, DocDet, Shang ve ark. 2026 ve klasik kaynaklar eklendi; 5 kayıt düzeltildi).
> 6. **İşaretçi atfı:** AIForge-Doc v1 makalesi §3.4 (Gemini: 3 px parlak yeşil dikdörtgen) — doğrulandı.


*[AUTHOR NAMES], AIC515 Image Processing, Topkapı University — technical manuscript, 20 October 2026. Conference version to be trimmed (see Sec. 9).*

**Abstract** — Public benchmarks of AI-inpainted financial documents annotate forgeries with the bounding box of the edited field. Auditing AIForge-Doc v1 (4,061 forgeries, Gemini 2.5 Flash Image and Ideogram v2 Edit) we find that all masks are rectangles, that only a median 49 % of box pixels actually differ from the released authentic source, that one third of forgeries also change pixels outside the box, and that the released test "authentic" set is not the mirror of the test forgeries. Using the 3,247 verified training pairs we derive pixel-change and glyph-level ground truths and train the same ResNet-18 U-Net under three supervision geometries (three seeds each). Evaluated against the box ground truth the box-trained model reaches 0.95 pooled pixel-AUPRC on the official test split (zero-shot TruFor: F1 0.40), but the same predictions score 0.37 against the pixel-change ground truth, and each ground truth ranks its own model first (Kendall τ −0.22). Box-trained predictions are rectangles (median rectangularity 0.87) that are insensitive to blurring the glyphs inside the field and, at the default threshold, flag 44 % of authentic images (pixel-trained: 10 %) with equal image-level AUROC (0.89). Two pre-registered shortcut hypotheses were falsified: pixel-change supervision does not transfer better to an unseen generator or resist degradations, and no usable image-wide trace exists outside the field (AUROC 0.52). The audit, however, revealed an unreported pipeline artefact: a residue of the green rectangle marker drawn before inpainting lies along the inside edge of the box in 75 % of forgeries (83 % of Gemini, 2 % of Ideogram, 74 % of the official test split). Transferring only this residue onto unedited authentic images makes the box-trained model predict the box (per-image AP 0.91, box coverage 0.80), repairing it in real forgeries removes 0.22 AP, and pixel- and glyph-trained models inherit the cue because the line is itself a pixel change. We conclude that localization accuracy reported on this benchmark is largely a measure of annotation geometry and of a marker artefact rather than of forensic evidence, release derived masks, controls, probes and code, and recommend dual-ground-truth, threshold-free reporting together with cue-transfer audits for box-annotated forensic datasets.

*Index Terms* — document forensics, tampering localization, generative inpainting, shortcut learning, dataset artefacts, ground-truth annotation, U-Net.

## 1. Introduction

Commercial image-editing services built on diffusion and multimodal generators (Gemini 2.5 Flash Image, Ideogram v2 Edit, GPT-Image-2) can rewrite a single numeric field of a receipt or a form while leaving no visible seam. Recent benchmarks show that forensic detectors trained on classical splicing or on synthetic document tampering transfer poorly to this threat. On AIForge-Doc v1 the generic localizer TruFor reaches an image-level AUC of 0.751 and a pixel IoU of 0.358 zero-shot, while the document-specific DocTamper/DTD model drops to an AUC of 0.563 and an IoU of 0.020 [1]; on AIForge-Doc v2 (GPT-Image-2) human observers are at chance (2AFC accuracy 0.501) and TruFor reaches only 0.599 [2]; across eight document test sets none of fourteen published localizers reaches a pixel-F1 of 0.3 on six or more sets without adaptation [3]. The natural response is to fine-tune a localizer on the new datasets. This paper asks a prior question: **what does the ground truth of such a dataset actually teach a localizer, and what does the reported accuracy actually measure?**

AIForge-Doc annotates every forgery with the *bounding box of the edited field*, not with the pixels that changed. Auditing the released files we found that these masks are rectangles without exception, that at a difference threshold of 8 grey levels only a median 49 % of the pixels inside the box differ from the released authentic image, and that one third of the forgeries also change pixels *outside* the box. A segmentation network trained on such masks is rewarded for predicting a rectangle around any detectable glyph change, so the paste-back geometry can become a *shortcut* in the sense of Geirhos et al. [4] — a decision rule that fits the training distribution without capturing the intended evidence. Shortcut reliance of document localizers is well documented for the JPEG 8×8 grid: ADCD-Net reports a 68–73 % F1 drop of DTD under a one-pixel shift [5] and DocQT a 14.5-point F1 effect of the quantization table alone [6]. No study has examined the effect of the *annotation geometry* on what is learned, nor measured any of these effects on AI-inpainted documents. During our audit we additionally found that three quarters of the Gemini-generated forgeries carry a thin saturated-green line along the inside edge of the annotated box — a residue of the rectangle marker the dataset pipeline draws before inpainting — which turns out to be a near-deterministic cue for the box location.

AIForge-Doc is unusual in that the authentic source image is released for every training forgery. This lets us derive a second, *pixel-change* ground truth from paired differences and run a controlled comparison: the same architecture, data, augmentation and budget, with only the training mask changed. The study was conducted in two pre-registered phases (a literature and gap analysis with falsifiable hypotheses, then the experiments), and all code, derived masks and result tables are released. Our contributions are:

1. **A reproducible audit of AIForge-Doc v1** (12,187 files) quantifying mask semantics, outside-box edits, a global re-encoding trace, source-document leakage between official splits (8.9 % of test forgeries), the invalidity of the released test "authentic" set for image-level evaluation, and a previously unreported green marker residue present in 75 % of forged–authentic pairs (Sec. 3).
2. **Derived pixel-change and glyph ground truths** for all 3,247 verified training pairs with an ε-sensitivity analysis, a benign-control set that reproduces the pipeline's compositing step without content change, and a 100-pair verification sheet audited with a vision-language model (Sec. 4).
3. **A shortcut-audit protocol** for box-annotated localizers: dual-ground-truth evaluation, boundary ablation with placebo and interior controls, rectangle-fit post-processing, benign controls, a global-trace probe on clean crops, and a marker-transfer/marker-repair probe (Sec. 5.7).
4. **Full-data experiments**: nine ResNet-18 U-Nets (three ground truths × three seeds) plus Gemini-only runs, evaluated on the official test split, under degradations and on a held-out generator, with group-bootstrap confidence intervals and Holm-corrected decisions for four pre-registered hypotheses, alongside zero-shot TruFor and ADCD-Net references under identical evaluation code (Sec. 6).

Section 2 reviews related work, Section 3 the dataset audit, Section 4 the derived ground truths and controls, Section 5 the methods and the reasons for choosing them, Section 6 the results, Section 7 the discussion and threats to validity, Section 8 what was solved and what remains open, and Section 9 the reproducibility package.

## 2. Related Work

**Document tampering localization.** DocTamper/DTD [7] introduced a 170k-image synthetic benchmark (copy-move, splicing and typeset "generation" of text) and a dual-stream network that fuses RGB with 8×8 block-DCT coefficients and the JPEG quantization table, trained with a compression-quality curriculum. FFDN [8] injects the same frequency cues through a zero-initialised module and a wavelet decomposition so that they cannot dominate the RGB stream; ADCD-Net [5] gates the DCT stream with a predicted block-alignment score and adds OCR-based text/background disentanglement; DiffNet [9] makes the DCT backbone efficient and reports the best cross-domain average on the ForensicHub document protocol. All of these train on the DocTamper synthesis and rely on JPEG evidence. DOCFORGE-BENCH [3] evaluates fourteen detectors zero-shot on eight document sets and attributes most failures to threshold calibration when tampered pixels are 0.27–4 % of the image; DocQT [6] and DiffNet explicitly name compression-profile and synthesis-pipeline shortcuts. Earlier and parallel work on tampered scene and e-commerce text (Tampered-IC13 [10], the ICDAR 2023 competition [11], RTM [12], OSTF/DAF with its open-set generator protocol [13], CTP-Net [14], TextSleuth [15], FSTS [16], DITL² [17], forgery-trace enhancement [18]) uses box- or pixel-level labels of human or programmatic edits; none addresses API inpainting.

**AI-generated document forgeries.** AIForge-Doc v1 [1] and v2 [2] are, to our knowledge, the only public pixel-annotated benchmarks of API-inpainted financial documents; both evaluate detectors zero-shot only and both describe a context-window inpainting pipeline in which a 3 px green rectangle marker is drawn on the crop and only the field box is pasted back. A concurrent study on a third round [19] replaces the drawn marker with an alpha mask, pastes back the whole context crop, and reports that TruFor's maps score as well against the pasted crop rectangle as against the field box (pixel AUC 0.921 vs 0.918 on 200 rows), concluding that a splice localizer can score well on that round without localizing the edit. That finding concerns a paste-back seam in round v3 and zero-shot detectors; it does not examine v1, trained localizers, the box-versus-pixel ground truth or the marker residue studied here. The Receipt/Find-it dataset [20] provides human receipt forgeries; GPT4o-Receipt [21] and the MLLM study of [22] show that fully generated receipts are often detectable from arithmetic inconsistency rather than pixels; TextFake [23] documents a "text-density curse" for low-level detectors. TGIF [24] shows that when an inpainting model re-decodes the whole image, classical splicing localizers collapse (F1 ≤ 0.1), and COCO-Inpaint [25] provides cross-generator splits for natural images.

**Generic manipulation localization.** TruFor [26] fuses a learned noise residual (Noiseprint++) with RGB and outputs a confidence map; CAT-Net [27] learns double-JPEG statistics; MVSS-Net [28] combines a noise view with a Sobel edge view; PSCC-Net [29], IML-ViT [30], Mesorch [31], SAFIRE [32] and Omni-IML [33] represent progressive, transformer, multi-scale, promptable and unified designs. Several of these supervise edges explicitly, i.e., they *use* boundary artefacts.

**Shortcuts, generalization and evaluation reliability.** Geirhos et al. [4] define shortcut learning and recommend out-of-distribution testing by default; GenD [34] shows that unpaired real/fake training alone can move cross-dataset deepfake AUROC from 56.8 to 90.0; Smeu et al. [35] expose a trivial dataset artefact (leading silence) in audio-visual deepfake benchmarks; frequency bias is identified as a root cause of poor generalization in [36]; Aoudi et al. [37] describe a "metric trap" in natural-image forgery localization where pixel agreement hides reliance on boundary artefacts; GIFTBench [38] separates source, target and operation shifts; BR-Gen/NFA-ViT [39] targets small localized AI edits. Our work transfers these concerns to the geometry of document annotations and to a concrete pipeline artefact.

**Classical image processing and learning components.** The pipeline follows the course material: 2-D block DCT [40], local binary patterns [41], grey-level co-occurrence features [42], Sauvola adaptive binarization [43], support vector machines [44] and random forests [45], residual networks [46] pre-trained on ImageNet [47], U-Net [48], the Dice loss [49], bootstrap resampling [50] and Kendall's rank correlation [51].

## 3. Dataset and Audit

### 3.1 AIForge-Doc v1
Table 1 summarises the dataset as recomputed from the released files (`audit_dataset.py`, about three minutes on 32 cores). Each forged image has exactly one edited field; the generator label in the metadata is `gemini-nano` (Gemini 2.5 Flash Image according to the dataset card) for 3,639 images and `qwen-inpaint` (Ideogram v2 Edit) for 422. Masks are 8-bit PNGs with values {0, 255}.

**Table 1.** AIForge-Doc v1 as found in the released files (all values computed by the audit script).

| Property | Value (computed from files) |
|---|---|
| Forged images (train / test) | 3249 / 812 |
| Authentic files (train / test) | 3,249 / 812 (test set is not the mirror of the test forgeries: 86/812 are sources of a test forgery) |
| Source datasets | CORD 1,000 · WildReceipt 1,696 · SROIE 946 · XFUND 419 |
| Document type / languages | receipt 89.7 %, form 10.3 %; 9 languages (en 2,642, id 1,000) |
| Generator (metadata label) | Gemini 2.5 Flash Image (`gemini-nano`) 3,639 · Ideogram v2 Edit (`qwen-inpaint`) 422 |
| Mask format | 8-bit PNG {0,255}; 100 % single rectangles (rectangularity 1.00) |
| Metadata box = mask box | 4,047 / 4,061 (12 rows with swapped provenance) |
| Image size | W median 750 (150–6,016), H median 1,296 (168–7,016); 0.03–34.8 MP |
| Tampered area / image | median 0.98 % (IQR 0.37–1.62 %); < 1 %: 50.7 %, 1–5 %: 49.0 %, > 5 %: 0.4 % |
| Mask height / width (median) | 33 px / 191 px |
| Verified forged–authentic pairs | 3,247 / 3,249 (train), 89 / 812 (test) |
| Box pixels that actually change (D > 8) | median 49 % (IQR 37–62 %); IoU({D>8}, box) median 0.48 |
| Pairs with changes outside the box (D > 8) | 33.5 % (Gemini 35.9 %, Ideogram 11.9 %) |
| Pairs with a global ±1–3 re-encoding trace (D > 0) | 48.6 % (vanishes for ε ≥ 4) |
| Test forgeries sharing a source document with training (pHash ≤ 20) | 72 / 812 (8.9 %) |
| Licence | card: CC BY-NC-SA 4.0 + gated; `dataset_info.json`: CC BY 4.0 (conflict) |

### 3.2 Audit procedure
Forged–authentic pairs were matched by 256-bit perceptual hash and verified by pixel difference (mean outside-mask difference below 25 grey levels and either hash distance ≤ 10 or fewer than 5 % of outside-mask pixels changed by more than 8), yielding 3,247/3,249 verified training pairs but only 89/812 test pairs. For every verified pair we computed the per-pixel maximum-channel absolute difference D = max_c |forged − authentic|, connected-component statistics of the mask, perceptual-hash neighbourhoods across splits, and — motivated by a visual inspection during the ground-truth verification (Sec. 4.3) — a *green-residue* detector: pixels with G − max(R, B) > 50 and G > 110 that are present in the forged image but not in its authentic source, counted inside a ±6 px ring around the box boundary.

### 3.3 Findings
(i) *Mask geometry.* All 4,061 masks are single rectangles (rectangularity 1.00); the metadata box equals the mask box in 4,047 cases, the remaining 12 rows have swapped provenance, so masks rather than metadata are used as ground truth throughout.
(ii) *Mask semantics.* The median fraction of box pixels with D > 8 is 0.49 (IQR 0.37–0.62) and the median IoU between {D > 8} and the box is 0.48 (Fig. 1); the paper's "pixel-precise" wording describes box precision.
(iii) *Outside-box edits.* 33.5 % of pairs have D > 8 pixels outside the box (Gemini 35.9 %, Ideogram 11.9 %; WildReceipt 60 %, CORD 0.4 %), typically the whole text line being re-synthesised; 48.6 % of pairs show a global ±1–3 difference over the entire image at D > 0 that vanishes for ε ≥ 4 (Sec. 4.1).
(iv) *Scale.* The median tampered area is 0.98 % of the image, the median mask height 33 px, and images range from 0.03 to 34.8 MP (Fig. A1), so native-resolution tiling is mandatory: resizing a 750×1,296 receipt to 512×512 leaves the median field 13 px high.
(v) *Leakage and negatives.* 72/812 test forgeries (8.9 %) share a source document with a training image (pHash ≤ 20); the released `TestingSet/authentic` folder mirrors the test forgeries for only 86 images, 254 of its files are byte-identical to training authentics and no SROIE authentic is included, so it cannot serve as the negative class for image-level evaluation.
(vi) *Confounds.* Generator and source are entangled (no Ideogram in WildReceipt, 45 % in XFUND) and Ideogram edits are smaller (median 0.27 % vs 1.07 % of the image).
(vii) *Green marker residue.* 2,514 of 3,336 verified pairs (75.4 %) contain at least 20 new saturated-green pixels in the ring around the box, and in 2,509 of them the residue forms a line-like component of 15 px or more. The residue is almost entirely *inside* the box (median 176 px inside, 0 outside; outside pixels in 0.3 % of affected pairs), and it is generator-specific: 83.6 % of Gemini pairs versus 1.5 % of Ideogram pairs (CORD 50 %, SROIE 86 %, WildReceipt 90 %, XFUND 51 %). It affects 93 % of the 89 verified test pairs. The pattern matches the pipeline described by the authors [1] — a 3 px green rectangle drawn on the context crop to tell the editor where the field is, followed by paste-back of the field box only — with the inner edge of the drawn rectangle falling inside the pasted region. We verified visually that the line is absent from the authentic sources (Fig. 2).
(viii) *Licence.* The card states CC BY-NC-SA 4.0 with gating while `dataset_info.json` says CC BY 4.0; we treat the restrictive reading as binding and redistribute only identifier lists and derived masks.

![Fig. 1. Forged crop, authentic crop, official box mask and |forged − authentic| > 8 for four pairs. The box contains unchanged background and, in row 2, the generator re-synthesised the whole line outside the box.](dataset_audit/fig_pixel_diff_quads.png)

![Fig. 2. Verification panels (forged A, authentic B, amplified difference C, box mask D, pixel-change mask E, glyph mask F). A thin green line along the inside edge of the box is visible in the forged crops of the Gemini pairs (rows 2–5) and absent in the authentic crops.](phase2/gt/verification_grid_page1.png)

### 3.4 Splits
The official test split (812) is untouched. Training images are divided by source-document groups — connected components of verified pair edges and perceptual-hash near-duplicate edges (Hamming ≤ 20) — into 2,762 train and 487 validation images with no group overlap; the 581 training groups that touch the test split are kept in training. Test results are reported on all 812 images and on the 740 "clean" images without a near-duplicate in training. A stratified pilot subset (400 train / 100 validation) was used for the mid-term feasibility study (Sec. 6.1).

## 4. Derived Ground Truths and Controls

### 4.1 Pixel-change and glyph ground truths
From D we derive (Table 2): **pixel_ε** = {D > ε} for ε ∈ {4, 8, 16}; **pixel_8_morph** = {D > 8} followed by a 3×3 opening, removal of components smaller than 8 px and a 3×3 closing; **glyph** = (Sauvola text pixels of forged ∪ authentic, window 25, k = 0.2) ∩ dilate({D > 8}, 7×7), with the stroke polarity decided per box (dark-on-light unless the Sauvola-dark ratio inside the box exceeds 0.5 — the naive median-brightness rule mis-fires on 199 dim photographs; the corrected rule, `glyph_v2`, inverts only 21 boxes); and **bbox ∩ pixel_8** as a conservative variant. The ε-sensitivity curve (Fig. 3) separates two phenomena: the share of pairs whose outside-box pixels change by more than 1 % of the box falls from 47.8 % at ε = 0 to 4.7 % at ε = 4 and 1.3 % at ε = 8 — the global re-encoding trace of the Gemini path — whereas the share with at least one outside-box pixel is still 33.5 % at ε = 8 and 12 % at ε = 16 — genuine local edits outside the box. Morphological cleaning reduces the number of pairs with outside-box components from 1,117 to 156. Ten pairs with D_max ≤ 4 (forged ≈ authentic) and five pairs whose only changes lie outside the box are excluded from training under every condition; the training conditions use **bbox**, **pixel_8_morph** ("pixel") and **glyph_v2** ("glyph").

**Table 2.** Derived ground-truth variants over 3,336 verified pairs (medians unless stated).

| Variant | Box coverage (median) | IoU with box (median) | Positive pixels, % of image (median) | Pairs with ≥1 outside-box pixel | Pairs with >1 % outside share | Components (median) |
|---|---|---|---|---|---|---|
| pixel_eps4 | 0.666 | 0.620 | 0.66 | 1538 | 1238 | 67 |
| pixel_eps8 | 0.492 | 0.481 | 0.44 | 1117 | 602 | 51 |
| pixel_eps16 | 0.354 | 0.351 | 0.27 | 403 | 193 | 33 |
| pixel_eps8_morph | 0.351 | 0.350 | 0.23 | 161 | 126 | 8 |
| glyph | 0.293 | 0.276 | 0.32 | 2281 | 1863 | 27 |
| glyph_v2 | 0.286 | 0.269 | 0.31 | 2216 | 1782 | 28 |
| bbox_and_eps8 | 0.492 | 0.492 | 0.42 | 0 | 0 | 39 |

![Fig. 3. ε-sensitivity of the pixel-change ground truth: (a) positive area, (b) share of pairs with outside-box change, (c) by generator.](phase2/gt/fig_eps_sensitivity_pub.png)

### 4.2 Benign controls
From the authentic image of every validation forgery we construct (Table 3): (a) **benign paste-back** — the box expanded by 50 % per side (≥ 100 px) is up-sampled 2× with Lanczos, down-sampled back and only the box region is pasted onto the authentic copy, i.e., the pipeline's compositing step without content change and without the green marker; (b) **full-image resampling** — the whole image through the same 2× cycle; (c) PNG re-encoding (pixel-identical, kept as a null control). Benign paste-back leaves a median of 72 pixels with D > 8 (real forgeries: 3,020) confined to the box, whereas full-image resampling produces a comparable *count* of D > 8 pixels (median 2,627) spread over all text edges at much lower amplitude.

**Table 3.** Difference statistics of the control sets against their authentic source (n = 487 each).

| Image set (n = 487 each) | D_max median [IQR] | Pixels with D > 8, median [IQR] | Outside-box D > 8 pixels (median) |
|---|---|---|---|
| Real forgery (reference) | 180 [145–232] | 3020 [1464–6936] | 0 |
| Benign paste-back | 17 [10–28] | 72 [3–348] | 0 |
| Full-image 2× resample | 27 [17–42] | 2627 [331–12166] | 2466 |
| PNG re-encode | 0 [0–0] | 0 [0–0] | 0 |

### 4.3 VLM-assisted verification of the derived masks
The protocol called for a 100-pair human verification. Because no annotator time was available before this version, we audited the stratified 100-pair sheet (7 source × generator strata) with a vision-language model (claude-sonnet-5 via the platform API) that received the six-panel image of Fig. 2 and the metadata values and returned a structured verdict. Of 96 parsed answers, the pixel-change mask was judged OK in 69 and PARTIAL in 26 cases (WRONG in 1), the glyph mask OK in 55 and PARTIAL in 41; the model judged that the box contained substantial unchanged area in 96/96 cases; it reported green-marked pixels outside the box in 21 cases and could confirm a visible change there in only 4 (15 unclear). Asked independently about a thin green line near the box edge, it answered yes for 40, no for 24 and unsure for 32 pairs; against the objective residue detector its yes/no answers agree in 45 of 64 decided cases (37 true positives, 16 misses, 3 false alarms), i.e., it under-reports the artefact. A repeat pass on 18 pairs agreed with the first on 83 % (pixel verdict), 67 % (glyph verdict), 83 % (outside-box) and 78 % (green line) of items, and in 68 % of 78 checkable cases the model's free-text description of the change contained the forged value from the metadata. We treat this as a consistency check of the derived masks, not as a substitute for human annotation; the sheet with both the automatic flags and the VLM columns is released for human completion.

## 5. Methods — and why they were chosen

This section is written so that each design decision can be traced to a property of the data established in Sections 3–4.

### 5.1 Problem formulation
Given a document image **x** ∈ ℝ^{H×W×3}, tampering localization predicts a map **p** ∈ [0,1]^{H×W}, the probability that each pixel was altered; thresholding **p** gives a binary mask. Image-level detection reduces **p** to a scalar (we use the mean of the top 1 % of pixel scores) or trains a classifier on crops. Because the dataset's masks are field boxes, the *target* of localization is itself a design choice; we train and evaluate under three targets (Sec. 4.1).

### 5.2 Classical pipeline (feature extraction + shallow classifiers)
*Why.* A hand-crafted baseline establishes whether low-level statistics alone separate edited from clean regions, and it connects the study to the course material. Features are computed on 256×256 grey-scale patches: statistics of the first 15 zig-zag AC coefficients of the 8×8 block DCT [40] (sensitive to JPEG block structure — expected *not* to help on PNG data, which the result confirms), a 10-bin uniform LBP histogram [41] (micro-texture of rendered vs photographed glyphs), four GLCM statistics [42] averaged over four directions (contrast, homogeneity, energy, correlation), Laplacian high-pass residual statistics (mean |r|, σ, kurtosis — the spatial-domain analogue of a noise residual), and radial FFT band-energy ratios (frequency-domain high-pass filtering in the sense of Gonzalez & Woods). Classifiers: RBF-SVM [44] (standardised features, small C/γ grid) and a 500-tree random forest [45], which also yields feature importances.

### 5.3 Patch classifier: ResNet-18 in three regimes
*Why.* Before investing in segmentation we asked whether a pre-trained network with a new head ("transfer learning") beats training from scratch on 1.6k patches — the course's "pretrained model + new layer" question. ResNet-18 [46] with ImageNet weights [47] is used (a) frozen with a new linear head (linear probe), (b) fully fine-tuned, (c) randomly initialised. Input: the 224×224 centre of a 256 native crop (no resizing, which would add a resampling trace of its own); no horizontal flips (text orientation); mild brightness/contrast jitter; binary cross-entropy; AdamW. Negatives are (i) clean crops of the same forged image and (ii) the *same-coordinate crop of the paired authentic image* — the latter is the strongest control because content and position are identical and only the manipulation differs.

### 5.4 Localizer: U-Net with a ResNet-18 encoder
*Why U-Net.* Fields are 33 px high on average; a decoder with skip connections [48] restores the spatial resolution lost in the encoder, which is what small-region segmentation needs. *Why a ResNet-18 encoder.* With 2.7k training images, ImageNet initialisation shortens convergence and regularises; ResNet-18 (14.3 M parameters in the full U-Net) trains in 10 minutes per run on a 16 GB GPU, which made nine seeds × conditions affordable. *Why native-resolution crops.* Resizing destroys the signal (Sec. 3.3-iv); we train on 512×512 crops at native scale, 70 % centred on a randomly chosen ground-truth pixel (±128 px jitter, a quarter of the crop) and 30 % at random, two crops per image per epoch. *Why BCE + Dice.* Tampered pixels are ≈ 1 % of each image; Dice [49] is insensitive to the negative majority and complements the per-pixel cross-entropy. *Why RGB only.* A pilot variant with two extra channels (FFT Gaussian high-pass, D₀ ≈ 20 px, and a 3×3 Laplacian residual) did not improve over RGB (Sec. 6.1), consistent with the literature's warning that frequency cues can hurt on PNG data [8]; adding them would also have confounded the ground-truth comparison. *Inference.* Full-image sliding window (512 px, stride 448, averaged overlaps); images with a longest side above 4,096 px (15 validation, 29 test) are scaled to 4,096 and the map resized back.

### 5.5 Training protocol and checkpoint selection
All conditions share AdamW (lr 1e-4), batch 16, bf16 mixed precision, 10 epochs (fixed before the runs), seeds 42–44, and early stopping on the validation AUPRC of crops *under the training ground truth*; only the best checkpoint is kept. The number of epochs was chosen from a one-epoch dry run (≈ 60 s per epoch including validation) to keep the 12-run grid under 2.5 GPU-hours. Section 6.2 reports convergence diagnostics (best epoch, last-three-epoch slope, train loss) for every run so that the reader can judge whether 10 epochs sufficed.

### 5.6 Evaluation: metrics and statistics
*Why threshold-free metrics first.* With a 1:100 class ratio the pixel AUROC is dominated by easy negatives and stays above 0.9 for almost any sensible model (Table 5); the area under the precision–recall curve (AUPRC, pooled over all pixels and computed exactly from 256-bin histograms) and the per-image average precision (AP) are the informative threshold-free summaries. *Thresholded metrics.* F1 and IoU at 0.5, at the validation-calibrated global threshold τ* (grid 0.05–0.95), and the per-image Oracle-F1 (best threshold per image, an unattainable upper bound), following DOCFORGE-BENCH. *Dual ground truth.* Every model is scored against all three ground truths on validation; on the official test split only the box ground truth exists for 723 of 812 images. *Uncertainty.* Seeds are treated as a fixed factor (three-seed mean per image); confidence intervals come from 1,000 bootstrap resamples over source-document groups [50], paired across conditions; p-values are two-sided bootstrap p-values with Holm correction within each hypothesis family; ranking stability uses Kendall's τ [51]. The falsification criteria were fixed in the Phase-1 report before any full-data run.

### 5.7 Shortcut probes
Each probe isolates one candidate cue and includes a control that should *not* move a model that ignores that cue.
1. **Boundary ablation.** A 5 px ring around the box boundary (dilate 3 − erode 2) is Gaussian-blurred (σ = 1.5). *Placebo:* the same ring shape at a random non-overlapping position (should change nothing). *Interior:* erode(box, 3) is blurred and the boundary preserved (tests reliance on glyph content). Reported as relative ΔAUPRC/ΔAP on the same images.
2. **Rectangle fit.** The largest connected component of the thresholded prediction is replaced by its bounding box. If a metric does not punish this, it rewards rectangular shape rather than pixel accuracy. We also report the rectangularity (mask area / area of its bounding box) and component count of raw predictions.
3. **Benign controls.** False-positive pixel rate, share of images with > 0.1 % positive pixels and image-level AUROC (forged vs control) on authentic, benign paste-back and full-image-resampled validation images.
4. **Global-trace probe (H4).** A ResNet-18 trained to separate 224 px crops that do *not* touch the mask of a forged image from the same-coordinate crops of the authentic source; an AUROC above chance indicates image-wide evidence outside the field. *Positive control:* box-centred crops. *Symmetric control:* negatives replaced by full-image-resampled authentics.
5. **Marker probe (new).** For every validation pair the green residue (Sec. 3.3-vii, dilated by 1 px) is either *repaired* in the forged image by copying the authentic pixels (content of the forgery unchanged, marker removed) or *transferred* onto the authentic image (marker present, content unchanged). A model that uses the marker loses accuracy on the repaired forgery and predicts the box on the authentic-plus-marker image, where nothing was edited.

### 5.8 Zero-shot references
TruFor (official weights, ≤ 2 MP protocol of DOCFORGE-BENCH) and ADCD-Net (MIT licence; images re-encoded as JPEG Q100 and scaled to a 512 px long side as in the repository's cross-domain protocol, with an Otsu-based text mask replacing PaddleOCR) are run with the same evaluation code. They are *not* trained on AIForge-Doc and are reported for context in a separate table; the comparison is not on equal footing, which is stated wherever they appear.


## 6. Experiments and Results

Unless stated otherwise, values are means ± standard deviation over three seeds on the 487 validation images with full-image inference; "AUPRC" is the pooled pixel AUPRC, "AP" the mean per-image average precision, F1@τ* uses the validation-calibrated threshold. Confidence intervals in brackets are 95 % group-bootstrap intervals (Sec. 5.6). All numbers are read from the result files listed in Sec. 9.

### 6.1 Pilot study (mid-term): does the pipeline work on little data?
On the 400/100 pilot subset the patch classifiers and the segmenter were run end to end (Tables 4a–4c). Three observations shaped the main study: (i) the fully fine-tuned ResNet-18 (AUROC 0.899) clearly beats both the frozen-backbone linear probe (0.726) and training from scratch (0.679) — pre-training is necessary but a new head alone is not sufficient; (ii) every model finds the *paired-authentic* negative harder than a clean crop of the same forged image (fine-tuned ResNet 0.878 vs 0.922; random forest 0.610 vs 0.807; SVM 0.552 ≈ chance), so content and position cues contribute to the easier numbers; (iii) the RGB U-Net already reached a pixel-F1 of 0.707 at crop level on 400 images, and its predictions were rectangles — the observation that motivated the ground-truth study. DCT features were uninformative (feature importances flat), as expected for PNG data, and the frequency-augmented U-Net did not beat RGB (0.674 vs 0.707).

**Table 4a.** Pilot patch classification (296 validation patches; single seed).

| Model | n val patches | Accuracy | Balanced acc. | AUROC | AUPRC | F1@0.5 | AUROC tampered vs paired-authentic |
|---|---|---|---|---|---|---|---|
| svm_rbf | 296 | 0.534 | 0.606 | 0.646 | 0.443 | 0.546 | 0.552 |
| random_forest | 296 | 0.622 | 0.653 | 0.706 | 0.536 | 0.573 | 0.610 |
| ResNet-18 probe | 296 | 0.639 | 0.693 | 0.726 | 0.554 | 0.616 | 0.635 |
| ResNet-18 ft | 296 | 0.885 | 0.845 | 0.899 | 0.884 | 0.809 | 0.878 |
| ResNet-18 scratch | 296 | 0.503 | 0.610 | 0.679 | 0.497 | 0.561 | 0.573 |

**Table 4b.** Pilot segmentation at crop level (smoke_val; single seed; subsampled AUPRC).

| Variant | Pixel-F1@0.5 | IoU@0.5 | Oracle-F1 | AUPRC (subsampled) | Clean-crop FP rate |
|---|---|---|---|---|---|
| rgb | 0.707 | 0.637 | 0.740 | 0.947 | 0.003 |
| rgbfreq | 0.674 | 0.613 | 0.711 | 0.939 | 0.001 |

**Table 4c.** Pilot robustness probe (crop level; no retraining).

| Variant | Condition | Pixel-F1@0.5 | IoU@0.5 | AUPRC | F1 Gemini | F1 Ideogram |
|---|---|---|---|---|---|---|
| rgb | clean (PNG) | 0.708 | 0.637 | 0.947 | 0.800 | 0.497 |
| rgb | JPEG Q75 | 0.636 | 0.565 | 0.929 | 0.728 | 0.444 |
| rgb | rescale 0.75x→1x | 0.638 | 0.560 | 0.939 | 0.734 | 0.421 |
| rgbfreq | clean (PNG) | 0.674 | 0.613 | 0.939 | 0.779 | 0.449 |
| rgbfreq | JPEG Q75 | 0.604 | 0.540 | 0.912 | 0.715 | 0.328 |
| rgbfreq | rescale 0.75x→1x | 0.633 | 0.566 | 0.924 | 0.755 | 0.353 |

![Fig. 4. Pilot U-Net predictions (forged crop, box GT, RGB model, RGB+frequency model): the predicted masks are rectangles.](baselines/figures/unet_predictions.png)

### 6.2 Training dynamics and checkpoint selection
Table 5 lists every full-data run. All twelve runs completed the planned 10 epochs; the best validation-crop AUPRC was reached at epoch 7–10, the slope of the validation metric over the last three epochs is between −0.0014 and +0.0050 per epoch, and the training loss at the last epoch is 0.24 for box supervision but 0.51–0.53 for pixel and 0.43–0.45 for glyph supervision: the box target is markedly easier to fit. Validation-crop AUPRC (computed on crops that contain positives, so not comparable to full-image numbers) saturates at 0.986–0.988 for box models and at 0.915–0.928 for pixel/glyph models. Seed-to-seed spread is small (Fig. 5). We therefore consider the box runs converged and the pixel/glyph runs close to convergence; the possibility that longer training would narrow some gaps is discussed in Sec. 7. Peak GPU memory was 4.66 GiB and a run took 9.5–11.1 minutes (124 minutes for the grid).

**Table 5.** Convergence and checkpoint diagnostics for the twelve full-data runs (val-crop AUPRC = early-stopping criterion under the training ground truth).

| Run | n train | Epochs | Best epoch | Best val-crop AUPRC | Last-epoch AUPRC | Slope last 3 ep. | Train loss (last) | Time (min) | Peak GPU (GiB) |
|---|---|---|---|---|---|---|---|---|---|
| unet_bbox_geminionly_s42 | 2454 | 10 | 9 | 0.988 | 0.988 | +0.0000 | 0.231 | 9.900 | 4.660 |
| unet_bbox_s42 | 2748 | 10 | 9 | 0.986 | 0.986 | +0.0005 | 0.244 | 10.700 | 4.660 |
| unet_bbox_s43 | 2748 | 10 | 7 | 0.986 | 0.986 | -0.0001 | 0.237 | 11.000 | 4.660 |
| unet_bbox_s44 | 2748 | 10 | 7 | 0.987 | 0.985 | +0.0007 | 0.244 | 9.500 | 4.660 |
| unet_glyph_v2_geminionly_s42 | 2454 | 10 | 8 | 0.920 | 0.918 | -0.0014 | 0.446 | 10.100 | 4.660 |
| unet_glyph_v2_s42 | 2748 | 10 | 10 | 0.923 | 0.923 | +0.0018 | 0.437 | 10.100 | 4.660 |
| unet_glyph_v2_s43 | 2748 | 10 | 8 | 0.917 | 0.916 | -0.0007 | 0.429 | 10.500 | 4.660 |
| unet_glyph_v2_s44 | 2748 | 10 | 10 | 0.915 | 0.915 | +0.0036 | 0.435 | 11.100 | 4.660 |
| unet_pixel_eps8_morph_geminionly_s42 | 2454 | 10 | 8 | 0.928 | 0.927 | -0.0006 | 0.522 | 10.100 | 4.660 |
| unet_pixel_eps8_morph_s42 | 2748 | 10 | 8 | 0.925 | 0.924 | -0.0003 | 0.515 | 10.300 | 4.660 |
| unet_pixel_eps8_morph_s43 | 2748 | 10 | 10 | 0.924 | 0.924 | +0.0017 | 0.508 | 10.500 | 4.660 |
| unet_pixel_eps8_morph_s44 | 2748 | 10 | 10 | 0.924 | 0.924 | +0.0050 | 0.526 | 10.600 | 4.660 |

![Fig. 5. Training curves of the nine main runs (train loss, validation-crop AUPRC, clean-crop false-positive rate).](phase2/experiments/figures/fig_training_curves.png)

### 6.3 Zero-shot references
Table 6 gives the zero-shot results of TruFor and ADCD-Net on the official test split under the box ground truth. TruFor reaches F1@0.5 0.403 / IoU 0.328 (the dataset paper reports 0.434 / 0.358 with a different resizing protocol); ADCD-Net 0.182 / 0.149 but a higher pooled AUPRC (0.130 vs 0.080), i.e., it ranks pixels well but is strongly under-confident; its DCT branch still contributes after synthetic JPEG re-encoding (disabling it lowers F1 to 0.100). Neither model's performance changes on the leakage-free subset.

**Table 6.** Zero-shot references on the official test split (box ground truth; not trained on AIForge-Doc).

| Condition | Test subset | n | F1@0.5 | IoU@0.5 | Oracle-F1 | F1@τ*(val) | Pooled AUPRC | Pooled AUROC | Per-image AUROC (mean) | Per-image AP (mean) |
|---|---|---|---|---|---|---|---|---|---|---|
| trufor | full | 812 | 0.403 | 0.328 | 0.456 | 0.405 | 0.080 | 0.822 | 0.903 | 0.443 |
| trufor | clean | 740 | 0.401 | 0.326 | 0.455 | 0.403 | 0.084 | 0.828 | 0.905 | 0.442 |
| trufor_1024pad | full | 812 | 0.134 | 0.100 | 0.216 | 0.137 | 0.028 | 0.749 | 0.779 | 0.193 |
| trufor_1024pad | clean | 740 | 0.133 | 0.099 | 0.218 | 0.137 | 0.030 | 0.758 | 0.782 | 0.195 |
| adcd_q100_ocrproxy | full | 812 | 0.182 | 0.149 | 0.328 | 0.244 | 0.130 | 0.860 | 0.868 | 0.283 |
| adcd_q100_ocrproxy | clean | 740 | 0.189 | 0.155 | 0.333 | 0.250 | 0.133 | 0.855 | 0.869 | 0.291 |
| adcd_q95_ocrproxy | full | 812 | 0.114 | 0.089 | 0.269 | 0.177 | 0.090 | 0.859 | 0.851 | 0.212 |
| adcd_q95_ocrproxy | clean | 740 | 0.117 | 0.091 | 0.272 | 0.179 | 0.092 | 0.855 | 0.850 | 0.215 |
| adcd_q100_noaln | full | 812 | 0.100 | 0.079 | 0.247 | 0.155 | 0.114 | 0.830 | 0.834 | 0.194 |
| adcd_q100_noaln | clean | 740 | 0.101 | 0.079 | 0.246 | 0.156 | 0.123 | 0.826 | 0.834 | 0.194 |
| adcd_q100_emptyocr | full | 812 | 0.179 | 0.147 | 0.328 | 0.243 | 0.131 | 0.855 | 0.864 | 0.283 |
| adcd_q100_emptyocr | clean | 740 | 0.186 | 0.153 | 0.333 | 0.250 | 0.135 | 0.850 | 0.865 | 0.290 |

### 6.4 The same predictions, three ground truths (H1)
Table 7 crosses training ground truth with evaluation ground truth and Fig. 6 shows the three threshold-free summaries. The box-trained model reaches AUPRC 0.923 ± 0.015 and F1@τ* 0.842 ± 0.023 against the box ground truth, but the *same probability maps* score 0.374 ± 0.008 against the pixel-change ground truth and 0.257 ± 0.007 against the glyph ground truth. Every ground truth ranks its own model first; with group-bootstrap intervals the box model's ΔF1@τ* between box and pixel evaluation is 0.398 [0.377, 0.418] and its ΔAP 0.522 [0.500, 0.544] (Holm p = 0.012), whereas the pixel-trained model's ΔAP between the two ground truths is only 0.012 [−0.006, 0.028] — the inflation is specific to box training. Kendall's τ between the nine-model rankings under box and pixel ground truth is −0.22 [−0.28, 0.06] (box vs glyph −0.72), and the probability of τ ≥ 0.9 is 0.001; all nine models change rank. Pixel AUROC stays above 0.9 in most cells (0.988 for the box model under the pixel ground truth), confirming that AUROC is uninformative at this class ratio. **H1 is confirmed** against the pre-registered criterion (ΔF1 < 0.05 and τ ≥ 0.9 would have falsified it).

**Table 7.** Validation (n = 487), full-image inference, three seeds: training ground truth × evaluation ground truth.

| Train GT | Eval GT | AUPRC | AUROC | AP | F1@0.5 | F1@τ* | IoU@τ* | Oracle-F1 | τ* |
|---|---|---|---|---|---|---|---|---|---|
| bbox | bbox | 0.923 ± 0.015 | 0.986 ± 0.001 | 0.896 ± 0.007 | 0.836 ± 0.029 | 0.842 ± 0.023 | 0.785 ± 0.027 | 0.873 ± 0.013 | 0.62 |
| bbox | pixel | 0.374 ± 0.008 | 0.988 ± 0.000 | 0.377 ± 0.005 | 0.432 ± 0.022 | 0.447 ± 0.011 | 0.321 ± 0.010 | 0.466 ± 0.010 | 0.95 |
| bbox | glyph | 0.257 ± 0.007 | 0.929 ± 0.011 | 0.282 ± 0.003 | 0.371 ± 0.020 | 0.387 ± 0.009 | 0.251 ± 0.007 | 0.402 ± 0.009 | 0.95 |
| pixel | bbox | 0.579 ± 0.014 | 0.873 ± 0.019 | 0.588 ± 0.013 | 0.525 ± 0.015 | 0.600 ± 0.014 | 0.465 ± 0.015 | 0.606 ± 0.014 | 0.05 |
| pixel | pixel | 0.610 ± 0.012 | 0.960 ± 0.011 | 0.579 ± 0.003 | 0.572 ± 0.005 | 0.577 ± 0.007 | 0.446 ± 0.007 | 0.608 ± 0.005 | 0.45 |
| pixel | glyph | 0.511 ± 0.010 | 0.908 ± 0.013 | 0.536 ± 0.005 | 0.526 ± 0.007 | 0.532 ± 0.008 | 0.387 ± 0.004 | 0.570 ± 0.006 | 0.60 |
| glyph | bbox | 0.351 ± 0.015 | 0.796 ± 0.028 | 0.373 ± 0.011 | 0.344 ± 0.003 | 0.401 ± 0.007 | 0.264 ± 0.005 | 0.403 ± 0.007 | 0.05 |
| glyph | pixel | 0.379 ± 0.012 | 0.912 ± 0.017 | 0.397 ± 0.006 | 0.435 ± 0.002 | 0.457 ± 0.006 | 0.319 ± 0.005 | 0.473 ± 0.004 | 0.05 |
| glyph | glyph | 0.676 ± 0.019 | 0.945 ± 0.010 | 0.695 ± 0.012 | 0.672 ± 0.010 | 0.674 ± 0.011 | 0.544 ± 0.013 | 0.694 ± 0.009 | 0.45 |

![Fig. 6. Pooled AUPRC, F1@τ* and per-image AP of the three model families under the three ground truths (validation, 3 seeds).](phase2/experiments/figures/fig_dual_gt_metrics.png)

![Fig. 7. Example predictions: forged image, three ground truths, and the probability maps of the box-, pixel- and glyph-trained models.](phase2/experiments/figures/fig_examples_predictions.png)

### 6.5 Official test split
On the official test split (box ground truth, τ* from validation) the box-trained U-Net obtains AUPRC 0.950 ± 0.011, F1@τ* 0.846 ± 0.015 and IoU 0.789 ± 0.018 on all 812 images and 0.949 / 0.845 / 0.788 on the 740 leakage-free images (ΔAP 0.0012 [−0.003, 0.005]) — the 8.9 % source overlap does not inflate the score. On the 89 test images with a released authentic pair the box model scores AUPRC 0.963 (box GT) but 0.427 (pixel GT), and the pixel model 0.739 (box GT) and 0.654 (pixel GT), reproducing the validation pattern (Table 8). The box model exceeds zero-shot TruFor by ΔAP 0.465 [0.436, 0.492] and ADCD-Net by 0.625 [0.598, 0.652] under the box ground truth — a comparison that, by Sec. 6.4, is itself biased toward the box-trained model.

**Table 8.** Official test split (τ* from validation; three seeds).

| Train GT | Eval GT | Subset | n | AUPRC | AP | F1@0.5 | F1@τ* | IoU@τ* | Oracle-F1 |
|---|---|---|---|---|---|---|---|---|---|
| bbox | bbox | all | 812 | 0.950 ± 0.011 | 0.908 ± 0.005 | 0.837 ± 0.027 | 0.846 ± 0.015 | 0.789 ± 0.018 | 0.879 ± 0.011 |
| bbox | pixel | all | 88 | 0.427 ± 0.002 | 0.370 ± 0.002 | 0.446 ± 0.013 | 0.467 ± 0.004 | 0.333 ± 0.005 | 0.472 ± 0.005 |
| bbox | glyph | all | 88 | 0.183 ± 0.004 | 0.285 ± 0.005 | 0.382 ± 0.012 | 0.401 ± 0.003 | 0.259 ± 0.002 | 0.404 ± 0.005 |
| bbox | bbox | clean_test | 740 | 0.949 ± 0.012 | 0.907 ± 0.006 | 0.836 ± 0.027 | 0.845 ± 0.015 | 0.788 ± 0.018 | 0.878 ± 0.012 |
| bbox | pixel | clean_test | 79 | 0.436 ± 0.003 | 0.380 ± 0.002 | 0.456 ± 0.012 | 0.477 ± 0.005 | 0.341 ± 0.005 | 0.482 ± 0.005 |
| bbox | glyph | clean_test | 79 | 0.185 ± 0.004 | 0.292 ± 0.004 | 0.393 ± 0.011 | 0.412 ± 0.002 | 0.267 ± 0.002 | 0.415 ± 0.005 |
| bbox | bbox | paired_89 | 89 | 0.963 ± 0.014 | 0.949 ± 0.003 | 0.894 ± 0.020 | 0.903 ± 0.006 | 0.852 ± 0.011 | 0.925 ± 0.008 |
| bbox | pixel | paired_89 | 88 | 0.427 ± 0.002 | 0.370 ± 0.002 | 0.446 ± 0.013 | 0.467 ± 0.004 | 0.333 ± 0.005 | 0.472 ± 0.005 |
| bbox | glyph | paired_89 | 88 | 0.183 ± 0.004 | 0.285 ± 0.005 | 0.382 ± 0.012 | 0.401 ± 0.003 | 0.259 ± 0.002 | 0.404 ± 0.005 |
| pixel | bbox | all | 812 | 0.638 ± 0.015 | 0.608 ± 0.016 | 0.545 ± 0.016 | 0.616 ± 0.013 | 0.480 ± 0.015 | 0.624 ± 0.015 |
| pixel | pixel | all | 88 | 0.654 ± 0.016 | 0.585 ± 0.003 | 0.579 ± 0.003 | 0.582 ± 0.004 | 0.445 ± 0.003 | 0.608 ± 0.005 |
| pixel | glyph | all | 88 | 0.376 ± 0.013 | 0.554 ± 0.005 | 0.549 ± 0.006 | 0.554 ± 0.004 | 0.401 ± 0.003 | 0.587 ± 0.005 |
| pixel | bbox | clean_test | 740 | 0.640 ± 0.014 | 0.607 ± 0.016 | 0.545 ± 0.016 | 0.614 ± 0.013 | 0.479 ± 0.015 | 0.623 ± 0.015 |
| pixel | pixel | clean_test | 79 | 0.665 ± 0.016 | 0.599 ± 0.003 | 0.591 ± 0.004 | 0.595 ± 0.004 | 0.455 ± 0.003 | 0.621 ± 0.005 |
| pixel | glyph | clean_test | 79 | 0.374 ± 0.014 | 0.560 ± 0.005 | 0.562 ± 0.006 | 0.567 ± 0.004 | 0.410 ± 0.003 | 0.600 ± 0.005 |
| pixel | bbox | paired_89 | 89 | 0.739 ± 0.031 | 0.673 ± 0.021 | 0.584 ± 0.015 | 0.660 ± 0.016 | 0.519 ± 0.019 | 0.666 ± 0.018 |
| pixel | pixel | paired_89 | 88 | 0.654 ± 0.016 | 0.585 ± 0.003 | 0.579 ± 0.003 | 0.582 ± 0.004 | 0.445 ± 0.003 | 0.608 ± 0.005 |
| pixel | glyph | paired_89 | 88 | 0.376 ± 0.013 | 0.554 ± 0.005 | 0.549 ± 0.006 | 0.554 ± 0.004 | 0.401 ± 0.003 | 0.587 ± 0.005 |
| glyph | bbox | all | 812 | 0.403 ± 0.011 | 0.377 ± 0.012 | 0.351 ± 0.003 | 0.405 ± 0.007 | 0.266 ± 0.006 | 0.410 ± 0.008 |
| glyph | pixel | all | 88 | 0.410 ± 0.017 | 0.369 ± 0.003 | 0.418 ± 0.002 | 0.442 ± 0.003 | 0.306 ± 0.002 | 0.456 ± 0.004 |
| glyph | glyph | all | 88 | 0.402 ± 0.017 | 0.678 ± 0.014 | 0.644 ± 0.009 | 0.643 ± 0.010 | 0.512 ± 0.012 | 0.670 ± 0.012 |
| glyph | bbox | clean_test | 740 | 0.403 ± 0.010 | 0.376 ± 0.011 | 0.352 ± 0.003 | 0.406 ± 0.006 | 0.267 ± 0.005 | 0.411 ± 0.008 |
| glyph | pixel | clean_test | 79 | 0.435 ± 0.015 | 0.379 ± 0.004 | 0.429 ± 0.002 | 0.455 ± 0.002 | 0.316 ± 0.002 | 0.468 ± 0.003 |
| glyph | glyph | clean_test | 79 | 0.414 ± 0.017 | 0.700 ± 0.014 | 0.663 ± 0.009 | 0.663 ± 0.010 | 0.528 ± 0.012 | 0.690 ± 0.013 |
| glyph | bbox | paired_89 | 89 | 0.386 ± 0.009 | 0.387 ± 0.005 | 0.362 ± 0.004 | 0.420 ± 0.008 | 0.279 ± 0.007 | 0.423 ± 0.008 |
| glyph | pixel | paired_89 | 88 | 0.410 ± 0.017 | 0.369 ± 0.003 | 0.418 ± 0.002 | 0.442 ± 0.003 | 0.306 ± 0.002 | 0.456 ± 0.004 |
| glyph | glyph | paired_89 | 88 | 0.402 ± 0.017 | 0.678 ± 0.014 | 0.644 ± 0.009 | 0.643 ± 0.010 | 0.512 ± 0.012 | 0.670 ± 0.012 |

### 6.6 What the box model predicts (H2)
*Shape.* Predictions of the box-trained model are rectangles: mean rectangularity 0.62–0.66 (median 0.87) with a single connected component, against 0.26–0.32 for the pixel model and 0.10–0.14 for the glyph model (Table 9a).
*Content vs boundary.* Blurring the interior of the field leaves the box model unchanged (+1.2 ± 0.3 % AUPRC, +0.1 % [−0.9, 1.0] AP) while the same manipulation costs the glyph model 10.5 % AP and the pixel model 7.2 % — the box model does not rely on glyph texture. Blurring the 5 px boundary ring costs the box model 2.0 ± 0.4 % AUPRC / 1.5 % [1.1, 2.0] AP, the placebo ring 0.0 %, but the pixel (−3.0 %) and glyph (−3.4 %) models lose *more*, so the pre-registered boundary-shortcut mechanism is reversed (difference +1.4 % [0.5, 2.4], Holm p = 0.024) (Table 9b, Fig. 8).
*Metric artefacts.* Interior blur *raises* the pixel model's score under the box ground truth (relΔ pooled AUPRC +10.9 ± 5.6 %, relΔ AP +4.5 ± 2.7 %) while lowering it under the pixel ground truth by 8.3 % — the box metric rewards erasing content. Replacing predictions by the bounding box of their largest component is harmless for the box model under the box ground truth (ΔF1 −0.019 ± 0.039, within seed noise) but destroys the pixel and glyph models under their own ground truths (−0.18 and −0.48; Table 9c).
*Benign controls.* At the default threshold 0.5 the box model flags 44 % of authentic validation images (> 0.1 % positive pixels), 50 % of benign paste-backs and 53 % of full-image-resampled images; the pixel model 10 %, 13 % and 14 %. At the validation-calibrated thresholds the per-pixel false-positive rates are similar (1.65 ‰ vs 1.99 ‰ on authentic images), so this difference is largely one of calibration. Benign paste-back raises the false-positive rate of every model by a small but significant amount (box: +0.50 ‰ [0.36, 0.66]). Image-level discrimination (top-1 % score, forged vs authentic) is 0.892 [0.876, 0.907] for the box model, 0.892 [0.874, 0.909] for the pixel model and 0.799 for the glyph model, dropping by 1.6–3.0 points against paste-back or resampled negatives (Table 9d).

**Table 9a.** Shape of raw predictions on validation (three-seed mean).

| Train GT | Rectangularity @0.5 (mean) | Rectangularity @τ* (mean) | Components @0.5 (mean) |
|---|---|---|---|
| bbox | 0.590 | 0.624 | 3.400 |
| pixel | — | — | — |
| glyph | — | — | — |

**Table 9b.** Boundary / placebo / interior ablation (each model under its own ground truth; relative change on the same images; three seeds).

| Train GT (own GT) | Ablation | relΔ pooled AUPRC | relΔ per-image AP | relΔ F1@τ* |
|---|---|---|---|---|
| bbox | boundary_blur | -2.0 ± 0.4 % | -1.5 ± 0.2 % | -2.2 ± 0.3 % |
| bbox | inner_blur | +1.2 ± 0.3 % | +0.1 ± 0.4 % | -0.1 ± 0.7 % |
| bbox | placebo_ring | -0.0 ± 0.0 % | +0.0 ± 0.0 % | -0.0 ± 0.0 % |
| pixel | boundary_blur | -6.4 ± 1.0 % | -3.0 ± 0.7 % | -3.1 ± 0.9 % |
| pixel | inner_blur | -8.3 ± 2.5 % | -7.2 ± 1.3 % | -10.0 ± 0.8 % |
| pixel | placebo_ring | +0.1 ± 0.0 % | +0.0 ± 0.0 % | +0.0 ± 0.0 % |
| glyph | boundary_blur | -2.8 ± 0.3 % | -3.4 ± 0.6 % | -2.9 ± 0.4 % |
| glyph | inner_blur | -5.6 ± 0.5 % | -10.5 ± 0.8 % | -12.3 ± 0.7 % |
| glyph | placebo_ring | -0.1 ± 0.0 % | -0.0 ± 0.0 % | -0.0 ± 0.0 % |

**Table 9c.** Rectangle-fit post-processing (F1@τ*; three-seed mean).

| Train GT | Eval GT | F1@τ* raw | F1 rectangle-fit | ΔF1 | ΔIoU |
|---|---|---|---|---|---|
| bbox | bbox | 0.842 | 0.823 | -0.019 ± 0.039 | -0.022 |
| bbox | pixel | 0.447 | 0.411 | -0.036 ± 0.004 | -0.033 |
| bbox | glyph | 0.387 | 0.361 | -0.026 ± 0.003 | -0.020 |
| pixel | bbox | 0.600 | 0.501 | -0.099 ± 0.009 | -0.076 |
| pixel | pixel | 0.577 | 0.393 | -0.183 ± 0.007 | -0.171 |
| pixel | glyph | 0.532 | 0.315 | -0.217 ± 0.017 | -0.185 |
| glyph | bbox | 0.401 | 0.204 | -0.197 ± 0.004 | -0.126 |
| glyph | pixel | 0.457 | 0.222 | -0.234 ± 0.002 | -0.175 |
| glyph | glyph | 0.674 | 0.192 | -0.482 ± 0.007 | -0.428 |

**Table 9d.** Benign controls (validation; three-seed mean).

| Train GT | Image set | FP pixel rate @0.5 | FP pixel rate @τ* | Images > 0.1 % positive @0.5 | @τ* | Image AUROC forged vs set (top-1 %) | Gemini | Ideogram |
|---|---|---|---|---|---|---|---|---|
| bbox | authentic | 4.46 ‰ | 1.62 ‰ | 44.1 % | 22.4 % | 0.838 | 0.857 | 0.618 |
| bbox | benign_pasteback | 5.30 ‰ | 2.12 ‰ | 50.1 % | 28.1 % | 0.818 | 0.837 | 0.590 |
| bbox | forged_val | 13.69 ‰ | nan ‰ | 91.0 % | nan % | — | — | — |
| bbox | fullimage_resample | 6.25 ‰ | 2.16 ‰ | 53.0 % | 27.9 % | 0.806 | 0.829 | 0.496 |
| pixel | authentic | 0.82 ‰ | 1.80 ‰ | 10.1 % | 23.4 % | 0.896 | 0.906 | 0.751 |
| pixel | benign_pasteback | 0.95 ‰ | 2.01 ‰ | 12.7 % | 26.7 % | 0.881 | 0.892 | 0.726 |
| pixel | forged_val | 5.12 ‰ | nan ‰ | 77.4 % | nan % | — | — | — |
| pixel | fullimage_resample | 1.01 ‰ | 2.25 ‰ | 14.4 % | 33.3 % | 0.870 | 0.887 | 0.558 |
| glyph | authentic | 1.86 ‰ | 3.18 ‰ | 19.5 % | 28.5 % | 0.814 | 0.823 | 0.720 |
| glyph | benign_pasteback | 1.98 ‰ | 3.35 ‰ | 23.4 % | 33.9 % | 0.793 | 0.804 | 0.667 |
| glyph | forged_val | 4.01 ‰ | nan ‰ | 68.8 % | nan % | — | — | — |
| glyph | fullimage_resample | 2.00 ‰ | 3.44 ‰ | 22.2 % | 33.7 % | 0.791 | 0.806 | 0.538 |

![Fig. 8. Boundary, placebo and interior ablation with group-bootstrap intervals.](phase2/stats/figures/fig_H2_ablation_ci.png)

### 6.6-bis The green marker: a pipeline artefact that every model exploits (exploratory probe)
This probe was not pre-registered; it was designed after the artefact was noticed during the ground-truth verification (Sec. 3.3-vii) and is reported as exploratory. In the validation split 368 of 487 pairs (75.6 %; Gemini 82.3 %, Ideogram 2.4 %) carry the residue. Table 14 and Fig. 10 give the four conditions of Sec. 5.7-5 on these 368 images and, for comparison, on the 119 residue-free images.

Three results follow. **(a) The marker alone triggers the box.** On authentic images onto which only the marker pixels were transferred — no character was changed — the box-trained model reaches a per-image AP of 0.906 [0.887, 0.924] against the (hypothetical) box ground truth, covers 80 % of the box at its calibrated threshold and raises its image-level score from 0.10 to 0.53; on the same authentic images without the marker AP is 0.178 and coverage 0.04. **(b) Removing the marker removes a quarter of the performance.** Repairing the residue in real forgeries (content unchanged) lowers the box model's AP from 0.988 to 0.763 (paired Δ −0.225 [−0.262, −0.189]) and its box coverage from 0.94 to 0.59; on residue-free forgeries the box model scores 0.747 — essentially the repaired level. The "generator gap" of Secs. 6.4 and 6.7 is therefore largely a marker gap. **(c) The pixel and glyph models inherit the cue.** Because the marker line differs from the authentic image, it is part of the derived pixel-change ground truth (median 10 % of the positive pixels of `pixel_8_morph` in affected pairs, p75 19 %); the pixel model's AP under its own ground truth changes by −0.266 [−0.295, −0.238] when the marker is repaired, and the marker transferred onto authentic images raises its box-AP from 0.14 to 0.64 and the glyph model's from 0.13 to 0.47. The effect is weaker than for the box model, which not only detects the line but completes the rectangle (IoU between prediction and box 0.74 vs 0.39 and 0.24).

Two further checks bound the finding. A hand-written rule that detects saturated-green line segments anywhere in the image and predicts their bounding box fires on 53 % of test images and reaches a mean F1 of only 0.05 (Table 15): the residue typically marks one or two edges of the field, so the box cannot be read off the marker without learning; the networks combine the line with the appearance of the adjacent text field. The detector used to flag the residue has a false-alarm rate of 0/600 on authentic images, and the residue is present in 604 of the 812 official test forgeries (74.4 %; Gemini 82.8 %), so the test numbers of Sec. 6.5 are obtained on a split in which three quarters of the images carry the cue. Finally, this explains why the pre-registered boundary ablation (Sec. 6.6) missed the shortcut: a Gaussian blur of σ = 1.5 smears a 1–3 px saturated-green line but does not remove its chroma, whereas the placebo and interior conditions never touched it.

**Table 14.** Green-marker probe on the 368 residue-carrying validation pairs (three-seed means, group-bootstrap 95 % CI; "box GT" for authentic inputs is the box of the paired forgery).

| Model family | Input | n | AP (box GT) | AP (pixel GT) | Box coverage @τ* | IoU(pred, box) @τ* | Positive fraction @τ* | Top-1 % score |
|---|---|---|---|---|---|---|---|---|
| bbox | forged | 368 | 0.988 [0.982, 0.992] | 0.622 [0.594, 0.647] | 0.940 [0.930, 0.948] | 0.878 [0.867, 0.890] | 0.014 [0.013, 0.015] | 0.634 [0.586, 0.679] |
| bbox | forged_repaired | 368 | 0.763 [0.726, 0.800] | 0.512 [0.475, 0.546] | 0.592 [0.552, 0.631] | 0.534 [0.498, 0.569] | 0.009 [0.008, 0.010] | 0.450 [0.404, 0.497] |
| bbox | authentic | 368 | 0.178 [0.155, 0.201] | 0.089 [0.074, 0.107] | 0.042 [0.026, 0.059] | 0.033 [0.020, 0.047] | 0.002 [0.001, 0.003] | 0.098 [0.079, 0.120] |
| bbox | authentic_plus_marker | 368 | 0.906 [0.887, 0.924] | 0.562 [0.531, 0.591] | 0.799 [0.773, 0.825] | 0.737 [0.710, 0.763] | 0.012 [0.011, 0.013] | 0.532 [0.484, 0.579] |
| pixel | forged | 368 | 0.749 [0.732, 0.766] | 0.761 [0.737, 0.782] | 0.539 [0.520, 0.560] | 0.499 [0.480, 0.519] | 0.009 [0.008, 0.010] | 0.195 [0.161, 0.228] |
| pixel | forged_repaired | 368 | 0.499 [0.468, 0.531] | 0.495 [0.461, 0.529] | 0.280 [0.256, 0.305] | 0.253 [0.231, 0.276] | 0.005 [0.004, 0.006] | 0.085 [0.063, 0.108] |
| pixel | authentic | 368 | 0.138 [0.118, 0.159] | 0.085 [0.069, 0.102] | 0.032 [0.020, 0.046] | 0.027 [0.017, 0.039] | 0.002 [0.002, 0.003] | 0.035 [0.022, 0.051] |
| pixel | authentic_plus_marker | 368 | 0.641 [0.617, 0.665] | 0.634 [0.603, 0.663] | 0.426 [0.404, 0.451] | 0.391 [0.369, 0.413] | 0.007 [0.006, 0.008] | 0.126 [0.099, 0.153] |
| glyph | forged | 368 | 0.524 [0.506, 0.542] | 0.586 [0.563, 0.609] | 0.325 [0.314, 0.336] | 0.284 [0.274, 0.296] | 0.007 [0.006, 0.008] | 0.089 [0.066, 0.114] |
| glyph | forged_repaired | 368 | 0.400 [0.374, 0.426] | 0.446 [0.413, 0.477] | 0.193 [0.178, 0.206] | 0.172 [0.159, 0.184] | 0.005 [0.005, 0.006] | 0.064 [0.044, 0.085] |
| glyph | authentic | 368 | 0.127 [0.110, 0.144] | 0.080 [0.066, 0.096] | 0.023 [0.017, 0.031] | 0.016 [0.011, 0.022] | 0.004 [0.003, 0.006] | 0.068 [0.046, 0.092] |
| glyph | authentic_plus_marker | 368 | 0.465 [0.446, 0.486] | 0.496 [0.469, 0.521] | 0.284 [0.271, 0.298] | 0.241 [0.229, 0.255] | 0.007 [0.006, 0.009] | 0.102 [0.076, 0.128] |

**Table 14b.** Paired contrasts (residue images).

| Model family | Paired contrast (residue images) | n | Estimate [95 % CI] |
|---|---|---|---|
| bbox | repaired − forged (ΔAP box GT) | 368 | -0.225 [-0.262, -0.189] |
| bbox | repaired − forged (ΔAP pixel GT) | 368 | -0.110 [-0.133, -0.086] |
| bbox | authentic+marker − authentic (Δ positive fraction @τ*) | 368 | +0.010 [+0.009, +0.011] |
| bbox | authentic+marker − authentic (Δ IoU(pred, box)) | 368 | +0.703 [+0.677, +0.731] |
| bbox | authentic+marker − authentic (Δ box coverage) | 368 | +0.757 [+0.730, +0.787] |
| bbox | authentic+marker − authentic (Δ top-1 % score) | 368 | +0.434 [+0.388, +0.479] |
| glyph | repaired − forged (ΔAP box GT) | 368 | -0.124 [-0.146, -0.103] |
| glyph | repaired − forged (ΔAP pixel GT) | 368 | -0.140 [-0.164, -0.119] |
| glyph | authentic+marker − authentic (Δ positive fraction @τ*) | 368 | +0.003 [+0.003, +0.004] |
| glyph | authentic+marker − authentic (Δ IoU(pred, box)) | 368 | +0.225 [+0.213, +0.239] |
| glyph | authentic+marker − authentic (Δ box coverage) | 368 | +0.261 [+0.247, +0.276] |
| glyph | authentic+marker − authentic (Δ top-1 % score) | 368 | +0.033 [+0.022, +0.044] |
| pixel | repaired − forged (ΔAP box GT) | 368 | -0.251 [-0.281, -0.219] |
| pixel | repaired − forged (ΔAP pixel GT) | 368 | -0.266 [-0.295, -0.238] |
| pixel | authentic+marker − authentic (Δ positive fraction @τ*) | 368 | +0.005 [+0.004, +0.005] |
| pixel | authentic+marker − authentic (Δ IoU(pred, box)) | 368 | +0.364 [+0.343, +0.387] |
| pixel | authentic+marker − authentic (Δ box coverage) | 368 | +0.394 [+0.372, +0.419] |
| pixel | authentic+marker − authentic (Δ top-1 % score) | 368 | +0.091 [+0.068, +0.115] |

**Table 15.** No-learning green-line rule on the official test split and on authentic images.

| Input | n | Fires | Mean F1 (box GT) | Mean IoU | IoU ≥ 0.5 |
|---|---|---|---|---|---|
| Test forgeries, all | 812 | 52.8 % | 0.054 | 0.040 | 3.0 % |
| Test forgeries, Gemini | 727 | 58.9 % | 0.061 | 0.045 | — |
| Test forgeries, Ideogram | 85 | 1.2 % | 0.000 | 0.000 | — |
| Authentic (validation sources) | 600 | 0.0 % | — | — | — |

![Fig. 10. Green-marker probe: per-image AP against the box, box coverage at τ* and image-level score for forged, marker-repaired forged, authentic and authentic-plus-marker inputs (368 residue pairs; 3 seeds; group-bootstrap CI).](phase2/experiments/figures/fig_green_marker.png)


### 6.7 Transfer and robustness (H3)
Table 10 (own ground truth) shows that pixel- and glyph-trained models are not more robust. Under pooled AUPRC the pixel model loses more than the box model under Gaussian blur σ = 1 (−0.361 ± 0.032 vs −0.190 ± 0.058) and 0.75× rescaling (−0.245 vs −0.112), and about the same under JPEG Q75 (−0.137 vs −0.154). Under per-image AP the differences are small and mixed in sign: JPEG favours the pixel model (+0.021 [0.007, 0.034]), blur the box model (−0.016 [−0.029, −0.002], Holm p = 0.06), and rescaling is indistinguishable (−0.007 [−0.018, 0.003]). On the held-out generator (Table 11; models trained on Gemini only, single seed) the box model keeps AP 0.395 [0.325, 0.468] on the source-matched Ideogram subset against 0.908 on Gemini (gap 0.513), the pixel model 0.279 [0.175, 0.373] against 0.582 (gap 0.303) under the pixel ground truth; under the common pixel ground truth the Ideogram difference between the two is +0.025 [−0.039, 0.088] (Holm p = 0.86). Models that saw Ideogram during training reach AP 0.629 (box) and 0.454 (pixel, pixel GT) on it. **H3 is falsified**: pixel-change supervision gives no consistent robustness or transfer advantage; where it loses (blur, rescaling), the likely reason is its reliance on high-frequency glyph edges, which are exactly what these degradations remove.

**Table 10.** Degradations applied at test time (validation; each model under its own ground truth; three seeds).

| Train GT (own GT) | Condition | AUPRC | ΔAUPRC vs clean | AP | ΔAP vs clean | F1@τ* | ΔF1@τ* |
|---|---|---|---|---|---|---|---|
| bbox | blur1 | 0.733 | -0.190 ± 0.058 | 0.801 | -0.095 ± 0.020 | 0.696 | -0.145 |
| bbox | jpeg75 | 0.769 | -0.154 ± 0.015 | 0.770 | -0.126 ± 0.013 | 0.714 | -0.128 |
| bbox | rescale075 | 0.811 | -0.112 ± 0.037 | 0.833 | -0.063 ± 0.019 | 0.735 | -0.107 |
| pixel | blur1 | 0.249 | -0.361 ± 0.032 | 0.468 | -0.111 ± 0.016 | 0.450 | -0.127 |
| pixel | jpeg75 | 0.473 | -0.137 ± 0.012 | 0.474 | -0.105 ± 0.003 | 0.472 | -0.105 |
| pixel | rescale075 | 0.364 | -0.245 ± 0.011 | 0.508 | -0.071 ± 0.010 | 0.492 | -0.085 |
| glyph | blur1 | 0.278 | -0.398 ± 0.046 | 0.538 | -0.157 ± 0.031 | 0.522 | -0.152 |
| glyph | jpeg75 | 0.554 | -0.122 ± 0.005 | 0.592 | -0.103 ± 0.004 | 0.578 | -0.095 |
| glyph | rescale075 | 0.387 | -0.289 ± 0.036 | 0.589 | -0.105 ± 0.025 | 0.572 | -0.101 |

**Table 11.** Held-out generator: Ideogram (source-matched to CORD/SROIE/XFUND) versus Gemini.

| Training set | Train GT | Eval GT | seeds | n Ideogram | Ideogram AP | Ideogram F1@0.5 | Gemini (source-matched) AP | Gemini F1@0.5 | Gap AP |
|---|---|---|---|---|---|---|---|---|---|
| All tools | bbox | bbox | 3 | 126 | 0.629 | 0.544 | 0.912 | 0.831 | 0.283 |
| All tools | bbox | pixel | 3 | 41 | 0.374 | 0.352 | 0.359 | 0.401 | -0.016 |
| All tools | bbox | glyph | 3 | 41 | 0.277 | 0.288 | 0.297 | 0.371 | 0.021 |
| All tools | pixel | bbox | 3 | 126 | 0.423 | 0.408 | 0.574 | 0.515 | 0.151 |
| All tools | pixel | pixel | 3 | 41 | 0.454 | 0.452 | 0.571 | 0.552 | 0.117 |
| All tools | pixel | glyph | 3 | 41 | 0.401 | 0.391 | 0.584 | 0.531 | 0.183 |
| All tools | glyph | bbox | 3 | 126 | 0.312 | 0.268 | 0.369 | 0.335 | 0.057 |
| All tools | glyph | pixel | 3 | 41 | 0.365 | 0.378 | 0.389 | 0.420 | 0.024 |
| All tools | glyph | glyph | 3 | 41 | 0.442 | 0.454 | 0.704 | 0.677 | 0.262 |
| Gemini only | bbox | bbox | 1 | 126 | 0.395 | 0.338 | 0.908 | 0.861 | 0.513 |
| Gemini only | bbox | pixel | 1 | 41 | 0.254 | 0.242 | 0.347 | 0.412 | 0.092 |
| Gemini only | bbox | glyph | 1 | 41 | 0.170 | 0.183 | 0.291 | 0.386 | 0.121 |
| Gemini only | pixel | bbox | 1 | 126 | 0.255 | 0.224 | 0.559 | 0.509 | 0.304 |
| Gemini only | pixel | pixel | 1 | 41 | 0.279 | 0.254 | 0.582 | 0.566 | 0.303 |
| Gemini only | pixel | glyph | 1 | 41 | 0.233 | 0.219 | 0.574 | 0.541 | 0.342 |
| Gemini only | glyph | bbox | 1 | 126 | 0.199 | 0.151 | 0.394 | 0.337 | 0.195 |
| Gemini only | glyph | pixel | 1 | 41 | 0.233 | 0.193 | 0.406 | 0.428 | 0.173 |
| Gemini only | glyph | glyph | 1 | 41 | 0.295 | 0.257 | 0.717 | 0.679 | 0.422 |

![Fig. 9. Degradation and held-out-generator contrasts with group-bootstrap intervals.](phase2/stats/figures/fig_H3_transfer_ci.png)

### 6.8 No usable trace outside the edited field (H4)
A ResNet-18 trained to separate 224 px crops that do not touch the mask of a forged image from the same-coordinate crops of the authentic source reaches AUROC 0.516 [0.513, 0.519] (457 validation pairs; Gemini 0.516, Ideogram 0.512), and 0.498 [0.492, 0.503] when the negatives are full-image-resampled authentics; the same pipeline on box-centred crops reaches 0.980 [0.972, 0.988] (Table 12). The interval excludes 0.5, but the effect is practically nil; a post-hoc look explains it: in 260/457 pairs the outside-box crops are pixel-identical, and in the remaining 197 pairs the classifier orders forged above authentic 92 % of the time but only 48 % against the resampled authentic. A weak re-encoding signature exists where the generator touched pixels outside the box (mostly WildReceipt/Gemini); it is a resampling trace, not forgery evidence, and symmetric preprocessing removes it. **H4 is falsified in its practical form.**

**Table 12.** Global-trace probe (image-level AUROC of a crop classifier).

| Condition | Subset | Pairs | AUROC | Pairwise accuracy |
|---|---|---|---|---|
| forged_vs_authentic | all | 457 | 0.516 | 0.396 |
| forged_vs_authentic | gemini | 416 | 0.516 | 0.425 |
| forged_vs_authentic | ideogram | 41 | 0.512 | 0.098 |
| forged_vs_fullimage_resample | all | 457 | 0.498 | 0.449 |
| forged_vs_fullimage_resample | gemini | 416 | 0.498 | 0.450 |
| forged_vs_fullimage_resample | ideogram | 41 | 0.482 | 0.439 |

### 6.9 Pre-registered decisions
Table 13 lists the decision contrasts with group-bootstrap intervals and Holm-corrected p-values; the complete ledger of 463 contrasts is in `phase2/stats/stats_summary.csv`.

| Hypothesis | Falsification criterion (fixed in Phase 1) | Observed (95 % CI) | Decision |
|---|---|---|---|
| H1 metric / ranking | ΔF1 < 0.05 and Kendall τ ≥ 0.9 | ΔF1@τ* 0.398 [0.377, 0.418]; ΔAP 0.522 [0.500, 0.544]; τ −0.22 [−0.28, 0.06], P(τ ≥ 0.9) = 0.001 | **Confirmed** |
| H2 boundary shortcut | Both models drop alike and benign FP ≈ authentic | Boundary relΔAP box −1.5 % [−2.0, −1.1] vs pixel −3.0 % [−3.8, −2.0] (reversed); placebo 0.0 %; interior blur box +0.1 % [−0.9, 1.0]; benign paste-back FP +0.50 ‰ [0.36, 0.66]; exploratory marker probe: authentic + marker AP 0.906 [0.887, 0.924], marker repair ΔAP −0.225 [−0.262, −0.189] | **Partial** — shape and content-insensitivity confirmed, boundary-ring mechanism falsified, a different boundary cue (the green marker residue) confirmed post hoc |
| H3 transfer / robustness | No difference or reversed | Ideogram ΔAP pixel − box (pixel GT) +0.025 [−0.039, 0.088], Holm p 0.86; degradation differences mixed in sign | **Falsified** |
| H4 global trace | AUROC ≈ 0.5 | 0.516 [0.513, 0.519]; resampled negatives 0.498 [0.492, 0.503]; positive control 0.980 [0.972, 0.988] | **Falsified** (practically) |

**Table 13.** Decision contrasts (group bootstrap, B = 1,000; Holm within family).

| Hypothesis | Contrast | Metric | Estimate [95 % CI] | p (boot) | p (Holm) |
|---|---|---|---|---|---|
| H1 | Δ[bbox-GT-trained] bbox-GT − pixel-GT | ap_img | 0.522 [0.500, 0.544] | 0.001 | 0.012 |
| H1 | Δ[bbox-GT-trained] bbox-GT − pixel-GT | f1_tau | 0.398 [0.377, 0.418] | 0.001 | 0.012 |
| H1 | Δ[bbox-GT-trained] bbox-GT − glyph-GT | ap_img | 0.617 [0.599, 0.635] | 0.001 | 0.012 |
| H1 | Δ[bbox-GT-trained] bbox-GT − glyph-GT | f1_tau | 0.458 [0.443, 0.474] | 0.001 | 0.012 |
| H1 | Δ[pixel-GT-trained] bbox-GT − pixel-GT | ap_img | 0.012 [-0.006, 0.028] | 0.202 | 0.202 |
| H1 | Δ[pixel-GT-trained] bbox-GT − pixel-GT | f1_tau | 0.026 [0.010, 0.041] | 0.001 | 0.012 |
| H1 | Δ[pixel-GT-trained] bbox-GT − glyph-GT | ap_img | 0.054 [0.035, 0.074] | 0.001 | 0.012 |
| H1 | Δ[pixel-GT-trained] bbox-GT − glyph-GT | f1_tau | 0.071 [0.055, 0.086] | 0.001 | 0.012 |
| H1 | Δ[glyph-GT-trained] bbox-GT − pixel-GT | ap_img | -0.023 [-0.037, -0.009] | 0.001 | 0.012 |
| H1 | Δ[glyph-GT-trained] bbox-GT − pixel-GT | f1_tau | -0.054 [-0.068, -0.041] | 0.001 | 0.012 |
| H1 | Δ[glyph-GT-trained] bbox-GT − glyph-GT | ap_img | -0.320 [-0.337, -0.305] | 0.001 | 0.012 |
| H1 | Δ[glyph-GT-trained] bbox-GT − glyph-GT | f1_tau | -0.271 [-0.285, -0.257] | 0.001 | 0.012 |
| H2 | relΔAP[bbox-GT-trained | bbox GT] boundary_blur vs clean | ap_img | -0.015 [-0.020, -0.011] | 0.001 | 0.021 |
| H2 | relΔAP[bbox-GT-trained | bbox GT] placebo_ring vs clean | ap_img | 0.000 [-0.001, 0.001] | 0.802 | 0.802 |
| H2 | relΔAP[pixel-GT-trained | pixel-change GT] boundary_blur vs clean | ap_img | -0.030 [-0.038, -0.020] | 0.001 | 0.021 |
| H2 | relΔAP[pixel-GT-trained | pixel-change GT] placebo_ring vs clean | ap_img | 0.000 [-0.000, 0.001] | 0.314 | 0.678 |
| H2 | relΔAP[glyph-GT-trained | glyph GT] boundary_blur vs clean | ap_img | -0.034 [-0.040, -0.028] | 0.001 | 0.021 |
| H2 | relΔAP[glyph-GT-trained | glyph GT] placebo_ring vs clean | ap_img | -0.000 [-0.001, 0.000] | 0.226 | 0.678 |
| H2 | relΔAP[boundary_blur] bbox-model(own GT) − pixel-model(own GT) | ap_img | 0.014 [0.005, 0.024] | 0.006 | 0.024 |
| H2 | relΔAP[boundary_blur] bbox-model − pixel-model (both bbox-GT) | ap_img | 0.046 [0.036, 0.055] | 0.001 | 0.021 |
| H2 | relΔAP[boundary_blur] bbox-model(own GT) − glyph-model(own GT) | ap_img | 0.019 [0.014, 0.024] | 0.001 | 0.021 |
| H2 | relΔAP[boundary_blur] bbox-model − glyph-model (both bbox-GT) | ap_img | 0.039 [0.030, 0.048] | 0.001 | 0.021 |
| H2 | ΔF1τ rect-fit − raw [bbox-GT-trained | bbox GT] | f1_tau | -0.019 [-0.026, -0.012] | 0.001 | 0.021 |
| H2 | ΔF1τ rect-fit − raw [bbox-GT-trained | pixel-change GT] | f1_tau | -0.036 [-0.042, -0.030] | 0.001 | 0.021 |
| H2 | ΔFP pixel rate [bbox-GT-trained] benign_pasteback − authentic @tau | fp_tau | 0.001 [0.000, 0.001] | 0.001 | 0.021 |
| H2 | ΔFP pixel rate [bbox-GT-trained] fullimage_resample − authentic @tau | fp_tau | 0.001 [0.000, 0.001] | 0.001 | 0.021 |
| H2 | ΔFP pixel rate [pixel-GT-trained] benign_pasteback − authentic @tau | fp_tau | 0.000 [0.000, 0.000] | 0.001 | 0.021 |
| H2 | ΔFP pixel rate [pixel-GT-trained] fullimage_resample − authentic @tau | fp_tau | 0.000 [0.000, 0.001] | 0.001 | 0.021 |
| H2 | ΔFP pixel rate [glyph-GT-trained] benign_pasteback − authentic @tau | fp_tau | 0.000 [0.000, 0.000] | 0.001 | 0.021 |
| H2 | ΔFP pixel rate [glyph-GT-trained] fullimage_resample − authentic @tau | fp_tau | 0.000 [0.000, 0.000] | 0.001 | 0.021 |
| H2 | image-level AUROC forged vs authentic [bbox-GT-trained] (top-1% score) | auroc_img_top1 | 0.892 [0.876, 0.907] | 0.001 | 0.021 |
| H2 | image-level AUROC forged vs authentic [pixel-GT-trained] (top-1% score) | auroc_img_top1 | 0.892 [0.874, 0.909] | 0.001 | 0.021 |
| H2 | image-level AUROC forged vs authentic [glyph-GT-trained] (top-1% score) | auroc_img_top1 | 0.799 [0.778, 0.823] | 0.001 | 0.021 |
| H3 | ΔAP drop difference [jpeg75] pixel-model(own GT) − bbox-model(own GT) | ap_img | 0.021 [0.007, 0.034] | 0.004 | 0.020 |
| H3 | ΔAP drop difference [jpeg75] pixel-model − bbox-model (both pixel-GT) | ap_img | -0.063 [-0.074, -0.053] | 0.001 | 0.014 |
| H3 | ΔAP drop difference [rescale075] pixel-model(own GT) − bbox-model(own GT) | ap_img | -0.007 [-0.018, 0.003] | 0.144 | 0.432 |
| H3 | ΔAP drop difference [rescale075] pixel-model − bbox-model (both pixel-GT) | ap_img | -0.044 [-0.055, -0.034] | 0.001 | 0.014 |
| H3 | ΔAP drop difference [blur1] pixel-model(own GT) − bbox-model(own GT) | ap_img | -0.016 [-0.029, -0.002] | 0.016 | 0.064 |
| H3 | ΔAP drop difference [blur1] pixel-model − bbox-model (both pixel-GT) | ap_img | -0.071 [-0.084, -0.058] | 0.001 | 0.014 |
| H3 | ΔAP drop difference [jpeg75] glyph-model(own GT) − bbox-model(own GT) | ap_img | 0.023 [0.013, 0.034] | 0.001 | 0.014 |
| H3 | ΔAP drop difference [jpeg75] glyph-model − bbox-model (both pixel-GT) | ap_img | -0.019 [-0.028, -0.010] | 0.001 | 0.014 |
| H3 | ΔAP drop difference [rescale075] glyph-model(own GT) − bbox-model(own GT) | ap_img | -0.042 [-0.054, -0.031] | 0.001 | 0.014 |
| H3 | ΔAP drop difference [rescale075] glyph-model − bbox-model (both pixel-GT) | ap_img | -0.023 [-0.033, -0.013] | 0.001 | 0.014 |
| H3 | ΔAP drop difference [blur1] glyph-model(own GT) − bbox-model(own GT) | ap_img | -0.062 [-0.075, -0.048] | 0.001 | 0.014 |
| H3 | ΔAP drop difference [blur1] glyph-model − bbox-model (both pixel-GT) | ap_img | -0.029 [-0.040, -0.018] | 0.001 | 0.014 |
| H3 | held-out Ideogram ΔAP pixel-model − bbox-model (gemini-only; both pixel-GT) | ap_img | 0.025 [-0.039, 0.088] | 0.432 | 0.864 |
| H3 | held-out Ideogram ΔAP glyph-model − bbox-model (gemini-only; both pixel-GT) | ap_img | -0.021 [-0.091, 0.047] | 0.558 | 0.864 |
| H4 | AUROC forged vs authentic [outside-bbox crops (H4) | all] | auroc_pair | 0.516 [0.513, 0.519] | 0.001 | 0.003 |
| H4 | AUROC forged vs fullimage_resample [outside-bbox crops (H4) | all] | auroc_pair | 0.498 [0.492, 0.503] | 0.336 | 0.336 |
| H4 | ΔAUROC [outside-bbox crops (H4)] authentic − resample | auroc_pair | 0.018 [0.014, 0.023] | 0.001 | 0.003 |
| leakage | test Δ[bbox-GT-trained | bbox GT] all 812 − clean 740 | ap_img | 0.001 [-0.003, 0.005] | 0.524 | 1.000 |
| leakage | test Δ[bbox-GT-trained | bbox GT] non-clean 72 − clean 740 | ap_img | 0.014 [-0.032, 0.055] | 0.524 | 1.000 |
| leakage | test Δ[bbox-GT-trained | bbox GT] all 812 − clean 740 | f1_tau | 0.001 [-0.004, 0.005] | 0.692 | 1.000 |
| leakage | test Δ[bbox-GT-trained | bbox GT] non-clean 72 − clean 740 | f1_tau | 0.009 [-0.050, 0.062] | 0.692 | 1.000 |
| zero-shot | Δ bbox-model − zero-shot trufor (test, bbox GT) | ap_img | 0.465 [0.436, 0.492] | 0.001 | 0.008 |
| zero-shot | Δ bbox-model − zero-shot trufor (test, bbox GT) | f1_0p5 | 0.435 [0.409, 0.459] | 0.001 | 0.008 |
| zero-shot | Δ bbox-model − zero-shot adcd_q100_ocrproxy (test, bbox GT) | ap_img | 0.625 [0.598, 0.652] | 0.001 | 0.008 |
| zero-shot | Δ bbox-model − zero-shot adcd_q100_ocrproxy (test, bbox GT) | f1_0p5 | 0.656 [0.630, 0.682] | 0.001 | 0.008 |


## 7. Discussion

### 7.1 What the reported numbers measure
A localizer trained and scored on field boxes reaches 0.95 pooled AUPRC on the official AIForge-Doc test split. Three findings qualify this number. First, it measures *field localization*, not *pixel-change localization*: half of the "correct" pixels never changed, and the same probability maps drop to 0.37–0.43 when scored against the pixels that did (H1). Second, the box model learns the geometry it was taught: its predictions are rectangles whose score does not move when the glyphs inside are blurred, and a metric that accepts a filled rectangle rewards content erasure (H2, shape part). Third — and this was not anticipated in the pre-registration — a residue of the dataset's own rectangle marker lies inside 75 % of the boxes, the trained networks use it, and removing it costs the box model 0.22 AP while transferring it onto an unedited image makes the model draw the box. Taken together, a large part of what a localizer "learns" on this benchmark is the annotation geometry plus a pipeline artefact, and the apparent difficulty of the Ideogram subset is mostly the absence of that artefact rather than a property of the generator.

### 7.2 Why the pre-registered boundary test failed and the marker test succeeded
The boundary-blur ablation was designed for a *resampling seam* — a subtle discontinuity that blurring would erase. The actual cue is a saturated colour line that survives blurring. The lesson for shortcut audits is methodological: ablations should be designed after looking at the data at pixel level, and a cue-specific transfer test (put the suspected cue on a clean image) is more diagnostic than a generic degradation. We therefore keep the pre-registered decision (boundary-blur mechanism falsified) and add the exploratory decision (marker shortcut confirmed) side by side rather than silently replacing one with the other.

### 7.3 Which ground truth to train on
Pixel-change supervision yields more precise and better-calibrated masks (4.5× fewer authentic images flagged at the default threshold), equal image-level discrimination, and a smaller marker dependence, but no robustness or transfer gain; it also inherits the marker because the marker *is* a pixel change. A marker-free ground truth (subtracting detected residue from the mask) and marker-free training images (inpainting or authentic copy of the residue) are the obvious next experiments; the released code produces both inputs. Box supervision transfers better in absolute terms to Ideogram but over-segments and is poorly calibrated at 0.5. For the deployment question "which pixels were edited?", neither current ground truth is adequate without the marker correction.

### 7.4 Negative results and what they rule out
Two pre-registered hypotheses were falsified: pixel-change supervision does not transfer or degrade better (H3), and there is no learnable image-wide trace outside the edited field (H4; the weak 0.516 AUROC is a resampling signature removed by symmetric preprocessing). The benign paste-back control — the pipeline's compositing step *without* the marker — raises false alarms only marginally, which is consistent with the marker rather than the resampling being the exploited artefact. These results narrow the open question (Sec. 8) to the appearance of API-rendered text itself.

### 7.5 Threats to validity
(1) *Derived ground truth.* The pixel-change mask is an estimate (ε = 8, morphology); its ε-sensitivity is reported and a vision-language-model audit of 100 pairs judged it adequate in 69 % and partially adequate in 26 % of cases, but no human verification has been completed. (2) *Test-set ground truth.* Pixel ground truth exists only where authentic pairs are released (training split and 89 test images); test conclusions rest on the box ground truth and on a split in which 74 % of images carry the marker. (3) *Training budget.* Ten epochs were fixed a priori; convergence diagnostics (Table 5) show plateaued validation metrics but still-decreasing training loss for pixel/glyph supervision, so absolute gaps may shrink with longer training; the ranking was stable across seeds. (4) *Confounds.* Generator, source, edit size and marker presence are entangled; the held-out-generator runs use one seed and 41–126 Ideogram images. (5) *Single architecture.* All trained models are ResNet-18 U-Nets; zero-shot references are not on equal footing. (6) *Exploratory analyses.* The marker probe, the heuristic rule and the post-hoc decomposition of H4 were not pre-registered and should be replicated on AIForge-Doc v2 (GPT-Image-2), whose pipeline uses the same 3 px green outline by design [2]; the concurrent round v3 removed the drawn marker but introduced a crop-rectangle paste-back seam that zero-shot TruFor exploits [19]. (7) *Scope.* 90 % receipts, one edited field per image, nine languages but 90 % English/Indonesian; nothing here should be read as a statement about invoices, bank statements or multi-field forgeries.

## 8. Conclusion: what was solved, what was not, and what remains open

**Solved (supported by the pre-registered or exploratory evidence).**
1. *The annotation geometry of AIForge-Doc determines what a localizer learns and how it scores.* Box masks cover a median 49 % unchanged pixels; the same predictions score 0.92 against the box and 0.37 against the pixel change; every ground truth ranks its own model first (H1 confirmed).
2. *Box-trained models predict rectangles and are insensitive to the glyphs inside the field.* Interior blur does not move them; rectangle-fit post-processing does not hurt them; the box metric rewards content erasure (H2, shape part).
3. *A pipeline artefact — the green marker residue — is present in 75 % of forgeries and is a decisive cue for every trained model.* Marker transfer onto clean images triggers the box (AP 0.91); marker repair removes 0.22 AP; the Ideogram "gap" largely disappears once residue-free images are compared with residue-free images (exploratory, large effect, tight intervals).
4. *Two suspected shortcuts are not operative.* The paste-back resampling seam (benign paste-back control, boundary blur) and a global re-encoding trace outside the field (H4) contribute little or nothing.
5. *Dataset hygiene facts that future users need:* source-document leakage between official splits (8.9 %) has no measurable effect on localization scores, but the released test "authentic" set cannot be used as the negative class; 12 metadata boxes are wrong; the licence statements conflict.

**Not solved (hypotheses falsified or evidence insufficient).**
1. *Pixel-change supervision does not buy robustness or cross-generator transfer* (H3 falsified); where it differs, it is more fragile to blur and rescaling.
2. *The pre-registered boundary mechanism was the wrong mechanism* — the audit identified the real cue only post hoc, so the H2 decision is "partial" on pre-registered grounds and "confirmed" on exploratory grounds.
3. *How much performance remains once the marker is removed from both training and test images* is unknown: all models here were trained with the marker present. The repaired-input numbers (box model AP 0.76) are a lower bound on what a marker-free training would achieve and an upper bound on what the marker-trained models know about the edit itself.
4. *Human verification of the derived masks* was replaced by a vision-language-model audit; it is a consistency check, not ground truth.

**Open research questions.**
- RQ-A. After marker-free retraining (residue removed from images and from the pixel ground truth), what is the achievable pixel-change AUPRC on AIForge-Doc, and does the box/pixel ranking reversal of H1 persist?
- RQ-B. Does a combined objective (pixel-change mask with a box-level auxiliary term, or a glyph-weighted box loss) obtain the calibration of pixel supervision with the robustness of box supervision?
- RQ-C. Is the residual cue after marker removal the *appearance of API-rendered text* (font rendering, anti-aliasing, local contrast), and can it be isolated with a control that re-renders authentic fields without changing their content?
- RQ-D. Does the marker artefact exist in AIForge-Doc v2 (GPT-Image-2, 3 px green outline by design) and in other API-inpainting benchmarks, and how much of the published zero-shot gaps between generators is explained by it?
- RQ-E. What evaluation protocol should box-annotated document benchmarks adopt — we propose dual-ground-truth, threshold-free reporting plus a cue-transfer audit — and how should such datasets document their marker and paste-back steps?

In sum, on AIForge-Doc v1 the first-order determinants of a document forgery localizer's behaviour are the ground-truth geometry and a marker left by the generation pipeline, not the generator; benchmark scores obtained under box annotation should not be read as pixel-level forensic accuracy until both are corrected.

## 9. Reproducibility and release
All code, derived masks (identifier lists plus mask PNGs; no dataset images), controls, result tables (`phase2/experiments/results`, `phase2/stats`), figures and the audit of the VLM verification are in the project package (`TESLIM_PAKETI/`, mirror of the project tree; `run_everything.ps1` reproduces the pipeline end to end on one 16 GB GPU in about six hours excluding the dataset download; `tools/build_manuscript.py` regenerates every table in this document from the result files). Model checkpoints (5 pilot, 12 full-data U-Nets, 2 probe classifiers) are included; third-party weights (TruFor, ADCD-Net) are kept with their licence files. The dataset itself is gated (CC BY-NC-SA 4.0 per its card) and must be downloaded by the user. *Conference version:* Sections 3–4 and 6.1–6.3 can be condensed to one page each and Tables 4, 5, 9c–d, 13 moved to supplementary material; the pre-registered/exploratory distinction and Table 14 should be retained.

## References
[1] Jiaqi Wu, Yuchen Zhou, Muduo Xu, Zisheng Liang, Simiao Ren, Jiayu Xue, Meige Yang, Siying Chen, Jingheng Huan. "AIForge-Doc: A Benchmark for Detecting AI-Forged Tampering in Financial and Form Documents." arXiv preprint, 2026. arXiv:2602.20569
[2] Jiaqi Wu, Yuchen Zhou, Dennis Tsang Ng, Xingyu Shen, Kidus Zewde, Ankit Raj, Tommy Duong, Simiao Ren. "When the Forger Is the Judge: GPT-Image-2 Cannot Recognize Its Own Faked Documents." arXiv preprint, 2026. arXiv:2604.25213
[3] Zengqi Zhao, Weidi Xia, En Wei, Yan Zhang, Jane Mo, Tiannan Zhang, Yuanqin Dai, Zexi Chen, Yiran Tao, Simiao Ren. "DOCFORGE-BENCH: A Comprehensive 0-shot Benchmark for Document Forgery Detection and Analysis." arXiv preprint, 2026. arXiv:2603.01433
[4] Robert Geirhos, Jörn-Henrik Jacobsen, Claudio Michaelis, Richard Zemel, Wieland Brendel, Matthias Bethge, Felix A. Wichmann. "Shortcut learning in deep neural networks." Nature Machine Intelligence, 2020. doi:10.1038/s42256-020-00257-z
[5] Kahim Wong, Jicheng Zhou, Haiwei Wu, Yain-Whar Si, Jiantao Zhou. "ADCD-Net: Robust Document Image Forgery Localization via Adaptive DCT Feature and Hierarchical Content Disentanglement." Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025. doi:10.1109/iccv51701.2025.01792
[6] Kylian Ronfleux-Corail, Guillaume Bernard, Mickaël Coustaty, Nicolas Sidère. "DocQT: Improving Document Forgery Localization Robustness via Diverse JPEG Quantization Tables." arXiv preprint, 2026. arXiv:2605.19688
[7] Chenfan Qu, Chongyu Liu, Yuliang Liu, Xinhong Chen, Dezhi Peng, Fengjun Guo, Lianwen Jin. "Towards Robust Tampered Text Detection in Document Image: New Dataset and New Solution." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023. doi:10.1109/cvpr52729.2023.00575
[8] Zhongxi Chen, Shen Chen, Taiping Yao, Ke Sun, Shouhong Ding, Xianming Lin, Liujuan Cao, Rongrong Ji. "Enhancing Tampered Text Detection Through Frequency Feature Fusion and Decomposition." Computer Vision -- ECCV 2024 (LNCS 15091), 2024. doi:10.1007/978-3-031-73414-4_12
[9] Mohamed Dhouib, Ye Zhu, Sonia Vanier, Aymen Shabou. "Efficient Document Tampering Localization with Multi-Level Discrepancy Features and Unified DCT-Quantization Embedding." arXiv preprint, 2026. arXiv:2606.22285
[10] Yuxin Wang, Hongtao Xie, Mengting Xing, Jing Wang, Shenggao Zhu, Yongdong Zhang. "Detecting Tampered Scene Text in the Wild." Computer Vision -- ECCV 2022 (LNCS), 2022. doi:10.1007/978-3-031-19815-1_13
[11] Dongliang Luo, Yu Zhou, Rui Yang, Yuliang Liu, Xianjin Liu, Jishen Zeng, Enming Zhang, Biao Yang, Ziming Huang, Lianwen Jin, Xiang Bai. "ICDAR 2023 Competition on Detecting Tampered Text in Images." Document Analysis and Recognition -- ICDAR 2023 (LNCS 14188), 2023. doi:10.1007/978-3-031-41679-8_36
[12] Dongliang Luo, Yuliang Liu, Rui Yang, Xianjin Liu, Jishen Zeng, Yu Zhou, Xiang Bai. "Toward real text manipulation detection: New dataset and new solution." Pattern Recognition, 2025. doi:10.1016/j.patcog.2024.110828
[13] Chenfan Qu, Yiwu Zhong, Fengjun Guo, Lianwen Jin. "Revisiting Tampered Scene Text Detection in the Era of Generative AI." Proceedings of the AAAI Conference on Artificial Intelligence, 2025. doi:10.1609/aaai.v39i1.32051
[14] Xin Liao, Siliang Chen, Jiaxin Chen, Tianyi Wang, Xiehua Li. "CTP-Net: Character Texture Perception Network for Document Image Forgery Localization." arXiv preprint, 2023. arXiv:2308.02158
[15] Chenfan Qu, Jian Liu, Haoxing Chen, Baihan Yu, Jingjing Liu, Weiqiang Wang, Lianwen Jin. "TextSleuth: Towards Explainable Tampered Text Detection." arXiv preprint, 2024. arXiv:2412.14816
[16] Zeqin Yu, Haotao Xie, Jian Zhang, Jiangqun Ni, Wenkan Su, Jiwu Huang. "Toward Real-world Text Image Forgery Localization: Structured and Interpretable Data Synthesis." arXiv preprint, 2025. arXiv:2511.12658
[17] Songze Li, Yunfei Guo, Shen Chen, Bin Li, Kaiqing Lin, Changsheng Chen, Haodong Li, Taiping Yao, Shouhong Ding. "DITL 2 : Dual-Stage Invariance Transfer Learning for Generalizable Document Image Tampering Localization." Proceedings of the 33rd ACM International Conference on Multimedia, 2025. doi:10.1145/3746027.3754857
[18] Li Dong, Weipeng Liang, Rangding Wang. "Robust Text Image Tampering Localization via Forgery Traces Enhancement and Multiscale Attention." IEEE Transactions on Consumer Electronics, 2024. doi:10.1109/tce.2024.3367947
[19] Ankit Raj, Yuxin Zhang, Kidus Zewde, Tommy Duong, Jiaqi Gan, Xingyu Shen, Yuchen Zhou, Huaiyu Guo, Siyu Zhang, Simiao Ren. "ChatGPT Images 2.5 on Forgery Tasks: Testing Advertised Improvements Against Known Answers." arXiv preprint, 2026. arXiv:2609.13617
[20] Beatriz Martínez Tornés, Théo Taburet, Emanuela Boros, Kais Rouis, Antoine Doucet, Petra Gomez-Krämer, Nicolas Sidere, Vincent Poulain d’Andecy. "Receipt Dataset for Document Forgery Detection." Document Analysis and Recognition -- ICDAR 2023 Workshops (LNCS), 2023. doi:10.1007/978-3-031-41682-8_28
[21] Yan Zhang, Simiao Ren, Ankit Raj, En Wei, Dennis Ng, Alex Shen, Jiayu Xue, Yuxin Zhang, Evelyn Marotta. "GPT4o-Receipt: A Dataset and Human Study for AI-Generated Document Forensics." arXiv preprint, 2026. arXiv:2603.11442
[22] Zisheng Liang, Kidus Zewde, Rudra Pratap Singh, Disha Patil, Zexi Chen, Jiayu Xue, Yao Yao, Yifei Chen, Qinzhe Liu, Simiao Ren. "Can Multi-modal (reasoning) LLMs detect document manipulation?." arXiv preprint, 2025. arXiv:2508.11021
[23] Yuning Zhang, Changtao Miao, Mingyu Liao, Tingyu Liu, Xinghao Wang, Tao Gong, Qi Chu, Nenghai Yu. "TextFake: Benchmarking AI-Generated Image Detection on Text-Rich Images." arXiv preprint, 2026. arXiv:2606.01050
[24] Hannes Mareen, Dimitrios Karageorgiou, Glenn Van Wallendael, Peter Lambert, Symeon Papadopoulos. "TGIF: Text-Guided Inpainting Forgery Dataset." IEEE International Workshop on Information Forensics and Security (WIFS), 2024. doi:10.1109/wifs61860.2024.10810690
[25] Haozhen Yan, Yan Hong, Jiahui Zhan, Suning Lang, Yikun Ji, Huijia Zhu, Jun Lan, Jianfu Zhang. "COCO-Inpaint: A Benchmark for Detecting and Localizing Inpainting-Based Image Manipulations." arXiv preprint, 2025. arXiv:2504.18361
[26] Fabrizio Guillaro, Davide Cozzolino, Avneesh Sud, Nicholas Dufour, Luisa Verdoliva. "TruFor: Leveraging All-Round Clues for Trustworthy Image Forgery Detection and Localization." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023. doi:10.1109/cvpr52729.2023.01974
[27] Myung-Joon Kwon, Seung-Hun Nam, In-Jae Yu, Heung-Kyu Lee, Changick Kim. "Learning JPEG Compression Artifacts for Image Manipulation Detection and Localization." International Journal of Computer Vision, 2022. doi:10.1007/s11263-022-01617-5
[28] Chengbo Dong, Xinru Chen, Ruohan Hu, Juan Cao, Xirong Li. "MVSS-Net: Multi-View Multi-Scale Supervised Networks for Image Manipulation Detection." IEEE Transactions on Pattern Analysis and Machine Intelligence, 2023. doi:10.1109/tpami.2022.3180556
[29] Xiaohong Liu, Yaojie Liu, Jun Chen, Xiaoming Liu. "PSCC-Net: Progressive Spatio-Channel Correlation Network for Image Manipulation Detection and Localization." IEEE Transactions on Circuits and Systems for Video Technology, 2022. doi:10.1109/tcsvt.2022.3189545
[30] Xiaochen Ma, Bo Du, Zhuohang Jiang, Xia Du, Ahmed Y. Al Hammadi, Jizhe Zhou. "IML-ViT: Benchmarking Image Manipulation Localization by Vision Transformer." arXiv preprint, 2023. arXiv:2307.14863
[31] Xuekang Zhu, Xiaochen Ma, Lei Su, Zhuohang Jiang, Bo Du, Xiwen Wang, Zeyu Lei, Wentao Feng, Chi-Man Pun, Ji-Zhe Zhou. "Mesoscopic Insights: Orchestrating Multi-Scale &amp; Hybrid Architecture for Image Manipulation Localization." Proceedings of the AAAI Conference on Artificial Intelligence, 2025. doi:10.1609/aaai.v39i10.33198
[32] Myung-Joon Kwon, Wonjun Lee, Seung-Hun Nam, Minji Son, Changick Kim. "SAFIRE: Segment Any Forged Image Region." Proceedings of the AAAI Conference on Artificial Intelligence, 2025. doi:10.1609/aaai.v39i4.32467
[33] Chenfan Qu, Yiwu Zhong, Fengjun Guo, Lianwen Jin. "Omni-IML: Towards Unified Image Manipulation Localization." arXiv preprint, 2024. arXiv:2411.14823
[34] Andrii Yermakov, Jan Cech, Jiri Matas, Mario Fritz. "Deepfake Detection that Generalizes Across Benchmarks." IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2026. doi:10.1109/wacv61042.2026.00082
[35] Stefan Smeu, Dragos-Alexandru Boldisor, Dan Oneata, Elisabeta Oneata. "Circumventing shortcuts in audio-visual deepfake detection datasets with unsupervised learning." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025. doi:10.1109/cvpr52734.2025.01753
[36] Chi Liu, Tianqing Zhu, Wanlei Zhou, Wei Zhao. "Frequency Bias Matters: Diving into Robust and Generalized Deep Image Forgery Detection." arXiv preprint, 2025. arXiv:2511.19886
[37] Samer Aoudi, Saleel A. P., Nedaa Al Barghuthi, Omar Alamir. "Beyond Accuracy: Auditing Image Forgery Localization via Scene-Disjoint Evaluation and XAI Faithfulness." IEEE Access, 2026. doi:10.1109/access.2026.3680143
[38] Baoke Dou, Ziye Wang, Hao Wang, Guoqing Cai, Wende Tan, Chenyang Si, Liucheng Guo, Yueming Lyu. "GIFTBench: Diagnosing Generalization in Image Forgery Localization and Informing Model Design." arXiv preprint, 2026. arXiv:2610.01778
[39] Lvpan Cai, Haowei Wang, Jiayi Ji, Yanshu Zhoumen, Shen Chen, Taiping Yao, Xiaoshuai Sun. "Zooming In on Fakes: A Novel Dataset for Localized AI-Generated Image Detection with Forgery Amplification Approach." Proceedings of the AAAI Conference on Artificial Intelligence, 2026. doi:10.1609/aaai.v40i4.37240
[40] Ahmed N., Natarajan T., Rao K. R. "Discrete Cosine Transform." IEEE Transactions on Computers, 1974. doi:10.1109/T-C.1974.223784
[41] Ojala T., Pietikäinen M., Mäenpää T. "Multiresolution gray-scale and rotation invariant texture classification with local binary." IEEE Transactions on Pattern Analysis and Machine Intelligen, 2002. doi:10.1109/TPAMI.2002.1017623
[42] Haralick R. M., Shanmugam K., Dinstein I. "Textural Features for Image Classification." IEEE Transactions on Systems, Man, and Cybernetics, 1973. doi:10.1109/TSMC.1973.4309314
[43] Sauvola J., Pietikäinen M. "Adaptive document image binarization." Pattern Recognition, 2000. doi:10.1016/S0031-3203(99)00055-2
[44] Cortes C., Vapnik V. "Support-vector networks." Machine Learning, 1995. doi:10.1007/BF00994018
[45] Breiman L. "Random Forests." Machine Learning, 2001. doi:10.1023/A:1010933404324
[46] He K., Zhang X., Ren S., Sun J. "Deep Residual Learning for Image Recognition." 2016 IEEE Conference on Computer Vision and Pattern Recognit, 2016. doi:10.1109/CVPR.2016.90
[47] Deng J., Dong W., Socher R., Li L.-J., Li K., Fei-Fei L. "ImageNet: A large-scale hierarchical image database." 2009 IEEE Conference on Computer Vision and Pattern Recognit, 2009. doi:10.1109/CVPR.2009.5206848
[48] Ronneberger O., Fischer P., Brox T. "U-Net: Convolutional Networks for Biomedical Image Segmentation." Lecture Notes in Computer Science, 2015. doi:10.1007/978-3-319-24574-4_28
[49] Milletari F., Navab N., Ahmadi S.-A. "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation." 2016 Fourth International Conference on 3D Vision (3DV), 2016. doi:10.1109/3DV.2016.79
[50] Efron B. "Bootstrap Methods: Another Look at the Jackknife." The Annals of Statistics, 1979. doi:10.1214/aos/1176344552
[51] Kendall M. G. "A NEW MEASURE OF RANK CORRELATION." Biometrika, 1938. doi:10.1093/biomet/30.1-2.81

## Appendix A — Additional figures

![Fig. A1. Image resolution distribution of AIForge-Doc v1.](dataset_audit/fig_resolution.png)

![Fig. A2. Tampered-area ratio distribution.](dataset_audit/fig_tamper_ratio.png)

![Fig. A3. Example images with box masks.](dataset_audit/fig_examples_overlay.png)

![Fig. A4. Pairwise IoU between derived ground-truth variants.](phase2/gt/fig_variant_iou_matrix.png)

![Fig. A5. Distribution of box coverage and outside-box share per variant.](phase2/gt/fig_variant_boxplots.png)

![Fig. A6. Difference statistics of the benign controls versus real forgeries.](phase2/controls/fig_controls_asymmetry.png)

![Fig. A7. Zero-shot TruFor / ADCD-Net F1 by tampered-area bin and generator.](phase2/literature_baselines/figures/fig_zero_shot_f1_breakdown.png)

![Fig. A8. Held-out generator contrasts.](phase2/experiments/figures/fig_heldout_generator.png)

![Fig. A9. Robustness curves.](phase2/experiments/figures/fig_robustness.png)

![Fig. A10. Benign-control score distributions.](phase2/experiments/figures/fig_controls_scores.png)

![Fig. A11. H4 crop-classifier score distributions.](phase2/experiments/figures/fig_h4_scores.png)

![Fig. A12. H1 dual-ground-truth contrasts with group-bootstrap CIs.](phase2/stats/figures/fig_H1_dualgt_ci.png)

![Fig. A13. H2 control contrasts with CIs.](phase2/stats/figures/fig_H2_controls_ci.png)

![Fig. A14. H4 AUROC with CIs.](phase2/stats/figures/fig_H4_auroc.png)

![Fig. A15. Pilot: ResNet-18 training curves.](baselines/figures/training_curves_resnet.png)

![Fig. A16. Pilot: U-Net training curves.](baselines/figures/training_curves_unet.png)

![Fig. A17. Pilot: random-forest feature importances (flat; DCT features uninformative on PNG data).](baselines/figures/rf_feature_importance.png)

![Fig. A18. Pilot: ResNet-18 confusion matrices.](baselines/figures/confusion_resnet.png)

## Appendix B — Mapping to the AIC515 syllabus
| Technique (week) | Role in this study | Measured contribution |
|---|---|---|
| Intensity transforms / contrast (2) | Augmentation only | none claimed |
| Geometric transforms (2) | Degradation (0.75× rescale), window tiling | Table 10 |
| Spatial high-pass filtering (3) | Laplacian residual features; pilot U-Net channel | Tables 4a, 4b (no gain over RGB) |
| Frequency domain, 2-D FFT (4) | FFT band features; Gaussian high-pass channel; DCT shown uninformative on PNG | Tables 4a, 4b |
| Noise / restoration (5) | Residual statistics; blur degradation | Table 10 |
| Thresholding and morphology (6) | Derived ground truths (Sauvola, opening/closing, component filter); τ* calibration; rectangle fit | Tables 2, 9c |
| Segmentation (7) | Connected components in GT derivation and rectangularity | Table 9a |
| Feature extraction, SVM / RF (9–11) | Classical baseline | Table 4a |
| ResNet (12) | Patch classifier (3 regimes); U-Net encoder; H4 probe | Tables 4a, 12 |
| U-Net (13) | Main localizer under three ground truths | Tables 7–11, 14 |
| YOLO (14) | Not used (box-level formulation left as future work, RQ-B) | — |

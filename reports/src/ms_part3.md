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
(1) *Derived ground truth.* The pixel-change mask is an estimate (ε = 8, morphology); its ε-sensitivity is reported and a vision-language-model audit of 100 pairs judged it adequate in 69 % and partially adequate in 26 % of cases, but no human verification has been completed. (2) *Test-set ground truth.* Pixel ground truth exists only where authentic pairs are released (training split and 89 test images); test conclusions rest on the box ground truth and on a split in which 74 % of images carry the marker. (3) *Training budget.* Ten epochs were fixed a priori; convergence diagnostics (Table 5) show plateaued validation metrics but still-decreasing training loss for pixel/glyph supervision, so absolute gaps may shrink with longer training; the ranking was stable across seeds. (4) *Confounds.* Generator, source, edit size and marker presence are entangled; the held-out-generator runs use one seed and 41–126 Ideogram images. (5) *Single architecture.* All trained models are ResNet-18 U-Nets; zero-shot references are not on equal footing. (6) *Exploratory analyses.* The marker probe, the heuristic rule and the post-hoc decomposition of H4 were not pre-registered and should be replicated on AIForge-Doc v2 (GPT-Image-2), whose pipeline uses the same 3 px green outline by design [@wu2026aiforgedocv2]; the concurrent round v3 removed the drawn marker but introduced a crop-rectangle paste-back seam that zero-shot TruFor exploits [@raj2026images25]. (7) *Scope.* 90 % receipts, one edited field per image, nine languages but 90 % English/Indonesian; nothing here should be read as a statement about invoices, bank statements or multi-field forgeries.

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
{{REFERENCES}}

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

## 6. Experiments and Results

Unless stated otherwise, values are means ± standard deviation over three seeds on the 487 validation images with full-image inference; "AUPRC" is the pooled pixel AUPRC, "AP" the mean per-image average precision, F1@τ* uses the validation-calibrated threshold. Confidence intervals in brackets are 95 % group-bootstrap intervals (Sec. 5.6). All numbers are read from the result files listed in Sec. 9.

### 6.1 Pilot study (mid-term): does the pipeline work on little data?
On the 400/100 pilot subset the patch classifiers and the segmenter were run end to end (Tables 4a–4c). Three observations shaped the main study: (i) the fully fine-tuned ResNet-18 (AUROC 0.899) clearly beats both the frozen-backbone linear probe (0.726) and training from scratch (0.679) — pre-training is necessary but a new head alone is not sufficient; (ii) every model finds the *paired-authentic* negative harder than a clean crop of the same forged image (fine-tuned ResNet 0.878 vs 0.922; random forest 0.610 vs 0.807; SVM 0.552 ≈ chance), so content and position cues contribute to the easier numbers; (iii) the RGB U-Net already reached a pixel-F1 of 0.707 at crop level on 400 images, and its predictions were rectangles — the observation that motivated the ground-truth study. DCT features were uninformative (feature importances flat), as expected for PNG data, and the frequency-augmented U-Net did not beat RGB (0.674 vs 0.707).

**Table 4a.** Pilot patch classification (296 validation patches; single seed).

{{T:pilot_cls}}

**Table 4b.** Pilot segmentation at crop level (smoke_val; single seed; subsampled AUPRC).

{{T:pilot_seg}}

**Table 4c.** Pilot robustness probe (crop level; no retraining).

{{T:pilot_rob}}

![Fig. 4. Pilot U-Net predictions (forged crop, box GT, RGB model, RGB+frequency model): the predicted masks are rectangles.](baselines/figures/unet_predictions.png)

### 6.2 Training dynamics and checkpoint selection
Table 5 lists every full-data run. All twelve runs completed the planned 10 epochs; the best validation-crop AUPRC was reached at epoch 7–10, the slope of the validation metric over the last three epochs is between −0.0014 and +0.0050 per epoch, and the training loss at the last epoch is 0.24 for box supervision but 0.51–0.53 for pixel and 0.43–0.45 for glyph supervision: the box target is markedly easier to fit. Validation-crop AUPRC (computed on crops that contain positives, so not comparable to full-image numbers) saturates at 0.986–0.988 for box models and at 0.915–0.928 for pixel/glyph models. Seed-to-seed spread is small (Fig. 5). We therefore consider the box runs converged and the pixel/glyph runs close to convergence; the possibility that longer training would narrow some gaps is discussed in Sec. 7. Peak GPU memory was 4.66 GiB and a run took 9.5–11.1 minutes (124 minutes for the grid).

**Table 5.** Convergence and checkpoint diagnostics for the twelve full-data runs (val-crop AUPRC = early-stopping criterion under the training ground truth).

{{T:ckpt}}

![Fig. 5. Training curves of the nine main runs (train loss, validation-crop AUPRC, clean-crop false-positive rate).](phase2/experiments/figures/fig_training_curves.png)

### 6.3 Zero-shot references
Table 6 gives the zero-shot results of TruFor and ADCD-Net on the official test split under the box ground truth. TruFor reaches F1@0.5 0.403 / IoU 0.328 (the dataset paper reports 0.434 / 0.358 with a different resizing protocol); ADCD-Net 0.182 / 0.149 but a higher pooled AUPRC (0.130 vs 0.080), i.e., it ranks pixels well but is strongly under-confident; its DCT branch still contributes after synthetic JPEG re-encoding (disabling it lowers F1 to 0.100). Neither model's performance changes on the leakage-free subset.

**Table 6.** Zero-shot references on the official test split (box ground truth; not trained on AIForge-Doc).

{{T:zeroshot}}

### 6.4 The same predictions, three ground truths (H1)
Table 7 crosses training ground truth with evaluation ground truth and Fig. 6 shows the three threshold-free summaries. The box-trained model reaches AUPRC 0.923 ± 0.015 and F1@τ* 0.842 ± 0.023 against the box ground truth, but the *same probability maps* score 0.374 ± 0.008 against the pixel-change ground truth and 0.257 ± 0.007 against the glyph ground truth. Every ground truth ranks its own model first; with group-bootstrap intervals the box model's ΔF1@τ* between box and pixel evaluation is 0.398 [0.377, 0.418] and its ΔAP 0.522 [0.500, 0.544] (Holm p = 0.012), whereas the pixel-trained model's ΔAP between the two ground truths is only 0.012 [−0.006, 0.028] — the inflation is specific to box training. Kendall's τ between the nine-model rankings under box and pixel ground truth is −0.22 [−0.28, 0.06] (box vs glyph −0.72), and the probability of τ ≥ 0.9 is 0.001; all nine models change rank. Pixel AUROC stays above 0.9 in most cells (0.988 for the box model under the pixel ground truth), confirming that AUROC is uninformative at this class ratio. **H1 is confirmed** against the pre-registered criterion (ΔF1 < 0.05 and τ ≥ 0.9 would have falsified it).

**Table 7.** Validation (n = 487), full-image inference, three seeds: training ground truth × evaluation ground truth.

{{T:dualgt}}

![Fig. 6. Pooled AUPRC, F1@τ* and per-image AP of the three model families under the three ground truths (validation, 3 seeds).](phase2/experiments/figures/fig_dual_gt_metrics.png)

![Fig. 7. Example predictions: forged image, three ground truths, and the probability maps of the box-, pixel- and glyph-trained models.](phase2/experiments/figures/fig_examples_predictions.png)

### 6.5 Official test split
On the official test split (box ground truth, τ* from validation) the box-trained U-Net obtains AUPRC 0.950 ± 0.011, F1@τ* 0.846 ± 0.015 and IoU 0.789 ± 0.018 on all 812 images and 0.949 / 0.845 / 0.788 on the 740 leakage-free images (ΔAP 0.0012 [−0.003, 0.005]) — the 8.9 % source overlap does not inflate the score. On the 89 test images with a released authentic pair the box model scores AUPRC 0.963 (box GT) but 0.427 (pixel GT), and the pixel model 0.739 (box GT) and 0.654 (pixel GT), reproducing the validation pattern (Table 8). The box model exceeds zero-shot TruFor by ΔAP 0.465 [0.436, 0.492] and ADCD-Net by 0.625 [0.598, 0.652] under the box ground truth — a comparison that, by Sec. 6.4, is itself biased toward the box-trained model.

**Table 8.** Official test split (τ* from validation; three seeds).

{{T:test}}

### 6.6 What the box model predicts (H2)
*Shape.* Predictions of the box-trained model are rectangles: mean rectangularity 0.62–0.66 (median 0.87) with a single connected component, against 0.26–0.32 for the pixel model and 0.10–0.14 for the glyph model (Table 9a).
*Content vs boundary.* Blurring the interior of the field leaves the box model unchanged (+1.2 ± 0.3 % AUPRC, +0.1 % [−0.9, 1.0] AP) while the same manipulation costs the glyph model 10.5 % AP and the pixel model 7.2 % — the box model does not rely on glyph texture. Blurring the 5 px boundary ring costs the box model 2.0 ± 0.4 % AUPRC / 1.5 % [1.1, 2.0] AP, the placebo ring 0.0 %, but the pixel (−3.0 %) and glyph (−3.4 %) models lose *more*, so the pre-registered boundary-shortcut mechanism is reversed (difference +1.4 % [0.5, 2.4], Holm p = 0.024) (Table 9b, Fig. 8).
*Metric artefacts.* Interior blur *raises* the pixel model's score under the box ground truth ({{T:inner_cross}}) while lowering it under the pixel ground truth by 8.3 % — the box metric rewards erasing content. Replacing predictions by the bounding box of their largest component is harmless for the box model under the box ground truth (ΔF1 −0.019 ± 0.039, within seed noise) but destroys the pixel and glyph models under their own ground truths (−0.18 and −0.48; Table 9c).
*Benign controls.* At the default threshold 0.5 the box model flags 44 % of authentic validation images (> 0.1 % positive pixels), 50 % of benign paste-backs and 53 % of full-image-resampled images; the pixel model 10 %, 13 % and 14 %. At the validation-calibrated thresholds the per-pixel false-positive rates are similar (1.65 ‰ vs 1.99 ‰ on authentic images), so this difference is largely one of calibration. Benign paste-back raises the false-positive rate of every model by a small but significant amount (box: +0.50 ‰ [0.36, 0.66]). Image-level discrimination (top-1 % score, forged vs authentic) is 0.892 [0.876, 0.907] for the box model, 0.892 [0.874, 0.909] for the pixel model and 0.799 for the glyph model, dropping by 1.6–3.0 points against paste-back or resampled negatives (Table 9d).

**Table 9a.** Shape of raw predictions on validation (three-seed mean).

{{T:rect}}

**Table 9b.** Boundary / placebo / interior ablation (each model under its own ground truth; relative change on the same images; three seeds).

{{T:boundary}}

**Table 9c.** Rectangle-fit post-processing (F1@τ*; three-seed mean).

{{T:rectfit}}

**Table 9d.** Benign controls (validation; three-seed mean).

{{T:controls_probe}}

![Fig. 8. Boundary, placebo and interior ablation with group-bootstrap intervals.](phase2/stats/figures/fig_H2_ablation_ci.png)

{{GREEN_RESULTS}}

### 6.7 Transfer and robustness (H3)
Table 10 (own ground truth) shows that pixel- and glyph-trained models are not more robust. Under pooled AUPRC the pixel model loses more than the box model under Gaussian blur σ = 1 (−0.361 ± 0.032 vs −0.190 ± 0.058) and 0.75× rescaling (−0.245 vs −0.112), and about the same under JPEG Q75 (−0.137 vs −0.154). Under per-image AP the differences are small and mixed in sign: JPEG favours the pixel model (+0.021 [0.007, 0.034]), blur the box model (−0.016 [−0.029, −0.002], Holm p = 0.06), and rescaling is indistinguishable (−0.007 [−0.018, 0.003]). On the held-out generator (Table 11; models trained on Gemini only, single seed) the box model keeps AP 0.395 [0.325, 0.468] on the source-matched Ideogram subset against 0.908 on Gemini (gap 0.513), the pixel model 0.279 [0.175, 0.373] against 0.582 (gap 0.303) under the pixel ground truth; under the common pixel ground truth the Ideogram difference between the two is +0.025 [−0.039, 0.088] (Holm p = 0.86). Models that saw Ideogram during training reach AP 0.629 (box) and 0.454 (pixel, pixel GT) on it. **H3 is falsified**: pixel-change supervision gives no consistent robustness or transfer advantage; where it loses (blur, rescaling), the likely reason is its reliance on high-frequency glyph edges, which are exactly what these degradations remove.

**Table 10.** Degradations applied at test time (validation; each model under its own ground truth; three seeds).

{{T:robust}}

**Table 11.** Held-out generator: Ideogram (source-matched to CORD/SROIE/XFUND) versus Gemini.

{{T:heldout}}

![Fig. 9. Degradation and held-out-generator contrasts with group-bootstrap intervals.](phase2/stats/figures/fig_H3_transfer_ci.png)

### 6.8 No usable trace outside the edited field (H4)
A ResNet-18 trained to separate 224 px crops that do not touch the mask of a forged image from the same-coordinate crops of the authentic source reaches AUROC 0.516 [0.513, 0.519] (457 validation pairs; Gemini 0.516, Ideogram 0.512), and 0.498 [0.492, 0.503] when the negatives are full-image-resampled authentics; the same pipeline on box-centred crops reaches 0.980 [0.972, 0.988] (Table 12). The interval excludes 0.5, but the effect is practically nil; a post-hoc look explains it: in 260/457 pairs the outside-box crops are pixel-identical, and in the remaining 197 pairs the classifier orders forged above authentic 92 % of the time but only 48 % against the resampled authentic. A weak re-encoding signature exists where the generator touched pixels outside the box (mostly WildReceipt/Gemini); it is a resampling trace, not forgery evidence, and symmetric preprocessing removes it. **H4 is falsified in its practical form.**

**Table 12.** Global-trace probe (image-level AUROC of a crop classifier).

{{T:h4}}

### 6.9 Pre-registered decisions
Table 13 lists the decision contrasts with group-bootstrap intervals and Holm-corrected p-values; the complete ledger of 463 contrasts is in `phase2/stats/stats_summary.csv`.

| Hypothesis | Falsification criterion (fixed in Phase 1) | Observed (95 % CI) | Decision |
|---|---|---|---|
| H1 metric / ranking | ΔF1 < 0.05 and Kendall τ ≥ 0.9 | ΔF1@τ* 0.398 [0.377, 0.418]; ΔAP 0.522 [0.500, 0.544]; τ −0.22 [−0.28, 0.06], P(τ ≥ 0.9) = 0.001 | **Confirmed** |
| H2 boundary shortcut | Both models drop alike and benign FP ≈ authentic | Boundary relΔAP box −1.5 % [−2.0, −1.1] vs pixel −3.0 % [−3.8, −2.0] (reversed); placebo 0.0 %; interior blur box +0.1 % [−0.9, 1.0]; benign paste-back FP +0.50 ‰ [0.36, 0.66]; exploratory marker probe: authentic + marker AP 0.906 [0.887, 0.924], marker repair ΔAP −0.225 [−0.262, −0.189] | **Partial** — shape and content-insensitivity confirmed, boundary-ring mechanism falsified, a different boundary cue (the green marker residue) confirmed post hoc |
| H3 transfer / robustness | No difference or reversed | Ideogram ΔAP pixel − box (pixel GT) +0.025 [−0.039, 0.088], Holm p 0.86; degradation differences mixed in sign | **Falsified** |
| H4 global trace | AUROC ≈ 0.5 | 0.516 [0.513, 0.519]; resampled negatives 0.498 [0.492, 0.503]; positive control 0.980 [0.972, 0.988] | **Falsified** (practically) |

**Table 13.** Decision contrasts (group bootstrap, B = 1,000; Holm within family).

{{T:stats}}

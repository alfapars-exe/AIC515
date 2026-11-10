## Paper Check Report

Paper: `/home/claude/paper/main.tex` + `sec/{abstract,intro,related,data,method,results,discussion}.tex`, `refs.bib`
Sources of truth: `/mnt/user-data/uploads/AI 515/Vize Proje/` (abbreviated `VP/` below)
Checks run: 1 (Numbers), 2 (Terms), 3 (Code vs paper), 5 (Citations: key existence and AIForge-Doc v1/TruFor claims only, no DBLP), 6 (Eval integrity, by reading code). Check 4 skipped (no equations beyond definitions).
Compile: `latexmk -pdf` exits 0; 8 pages; no undefined references, no BibTeX warnings, no overfull boxes.

### Summary
- Files checked: 8 .tex files, 5 code files (`phase3/train_unet_p3.py`, `phase3/probes_p3.py`, `phase2/experiments/train_unet_p2.py`, `phase3_yolo/train_yolo.py`, `phase3_yolo/convert_to_yolo.py`), 22 result/report files
- Issues found: 6 HIGH, 17 MEDIUM, 20 LOW

Seed provenance used for this audit: `T3_probes.csv` has `seeds=3` for `U-Net|main|{bbox,glyph_v2,pixel_eps8_morph}` and `seeds=1` for every other row (including `YOLO11s|main|bbox`). `T4_heldout.csv`: YOLO11s Gemini-only is `seeds=1`. `T6_robustness.csv`: only U-Net main has ±sd (3 seeds), U-Net main has no gray/CLAHE values. `T7_controls.csv`: U-Net main 3 seeds, all others 1, no YOLO rows.

---

### HIGH

1. **[Numbers / seed labelling] Table `tab:marker` U-Net rows do not match the source and are labelled seed 42, but the source rows are three-seed means**
   - Paper says: caption "Marker probes on 380 validation pairs with residue (seed 42)"; text "(seed 42 for every row)" (results.tex:47, :52). Rows (results.tex:62-63):
     `U-Net box 0.988 | 0.96 | 0.764 | -0.224 | 0.81 | 0.902` and `U-Net pix 0.739 | 0.57 | 0.497 | -0.242 | 0.44 | 0.828`.
   - Source shows (`VP/phase3/results/T3_probes.csv`, rows `U-Net|main|bbox` and `U-Net|main|pixel_eps8_morph`, `seeds=3`):
     box: APbox_forged 0.988, cover_forged **0.939**, APbox_forged_repaired **0.766**, dAP_repair **-0.222 [-0.256, -0.190]**, cover_authentic_plus_marker **0.795**, AUROC_authentic_plus_marker **0.903**, APbox_authentic_plus_marker **0.905**.
     pix: **0.747 | 0.538 | 0.500 | -0.247 | 0.420 | 0.858**.
     DeepLabV3+, U-Net+aux, repaired rows match the source (all `seeds=1`).
   - The same stale values propagate to: abstract.tex:2 "from 0.99 to 0.76"; results.tex:47 "0.988 to 0.764 (Δ=-0.224 [-0.261,-0.190])", "cover 81 % ... (AP 0.895)", "Δ ... -0.242"; results.tex:49 "instead of 81 % (11 % instead of 44 % ...)"; results.tex:137 "(U-Net 0.224)".
   - Suggested fix (if T3 is the intended source):
     - Rows: `U-Net \GTb & 0.988 & 0.94 & 0.766 & $-0.222$ & 0.80 & 0.903 \\` and `U-Net \GTp & 0.747 & 0.54 & 0.500 & $-0.247$ & 0.42 & 0.858 \\`
     - Caption: "Marker probes on 380 validation pairs with residue (standard U-Net rows: mean of three seeds; all other rows: seed 42). ..."
     - results.tex:47: "(three-seed means for the standard U-Nets, seed 42 for all other rows). For the standard box U-Net, copying only the residue pixels back from the authentic image lowers per-image AP from 0.988 to 0.766 ($\Delta{=}-0.222$ \ci{-0.256}{-0.190}), and pasting only the residue onto the authentic image makes the model cover 80\,\% of the box at $\tau^*$ (AP 0.905). DeepLabV3+ and the pixel-trained U-Net show the same dependence ($\Delta$ $-0.174$ and $-0.247$)."
     - results.tex:49: "the residue alone now covers 42\,\% of the box instead of 80\,\% (11\,\% instead of 42\,\% for the pixel model)"
     - abstract.tex:2: "removing it lowers the box model's per-image AP from 0.99 to 0.77"
     - results.tex:137: "removing the residue costs the detector only 0.062 AP (U-Net 0.222)"
     - If the paper's numbers instead come from a seed-42-only probe file, that file is not among the staged sources; then T3 must be regenerated or the caption must cite that file. Either way the current text and T3 disagree.

2. **[Numbers / seed labelling] Table `tab:render` U-Net rows do not match the source and are mislabelled seed 42**
   - Paper says: caption "(all validation pairs, seed 42)" (results.tex:86); rows (results.tex:96-97) `U-Net box 0.87 0.22 0.34 | 0.972 0.631 0.728`, `U-Net pix 0.43 0.06 0.17 | 0.925 0.558 0.681`; text results.tex:83 "falls from 0.87 (forged) to 0.22 ... 0.34 ..., AUROC from 0.972 to 0.631 and 0.728"; results.tex:137 "AUROC 0.865 (U-Net 0.631)".
   - Source shows (`T3_probes.csv`, `seeds=3`): box inbox_forged/render/content **0.864 / 0.237 / 0.348**, AUROC_forged/render/content **0.971 / 0.637 / 0.733**; pix **0.406 / 0.045 / 0.150**, **0.942 / 0.554 / 0.681**. DeepLabV3+ and repaired rows match.
   - Suggested fix: rows `U-Net \GTb & 0.86 & 0.24 & 0.35 & 0.971 & 0.637 & 0.733 \\`, `U-Net \GTp & 0.41 & 0.05 & 0.15 & 0.942 & 0.554 & 0.681 \\`; caption "(all validation pairs; standard U-Net rows: mean of three seeds, other rows seed 42)"; results.tex:83 "falls from 0.86 (forged) to 0.24 (render-only) and 0.35 (content-only), and forged-vs-authentic AUROC from 0.971 to 0.637 and 0.733"; results.tex:137 "AUROC 0.865 (U-Net 0.637)".

3. **[Numbers / claim] "reduce image-level AUROC by only 2--3 points for all models" is false**
   - Paper says: results.tex:43 "Benign paste-back and full-image resampling ... reduce image-level AUROC by only 2--3 points for all models".
   - Source shows (`T7_controls.csv`, auroc_top1, authentic minus control): three-seed U-Nets 1.6-3.0; DeepLabV3+ pix resample **4.5**; U-Net+aux 1.8-3.2; marker-repaired box **6.5 / 3.5**, repaired glyph **5.0 / 4.2**, repaired pix **6.5 / 6.1**. No YOLO rows exist in T7.
   - Suggested fix: "reduce image-level AUROC by 1.6--3.0 points for the three-seed U-Nets and by at most 4.5 points for the other standard segmenters, but by 3.5--6.5 points for the marker-repaired models (one seed); the detector was not evaluated on these controls."

4. **[Claim without evidence] Joint supervision was never tested on the held-out generator**
   - Paper says: discussion.tex:13 "Pixel-change supervision gives the most faithful masks and the fewest false alarms, but neither it nor joint supervision closes the generator gap."
   - Source shows: `T4_heldout.csv` contains only `U-Net main/geminionly` and `YOLO11s main/geminionly`; there is no `U-Net+box-aux` row and no Gemini-only joint model in any results file.
   - Suggested fix: "Pixel-change supervision gives the most faithful masks and the fewest false alarms, but it does not close the generator gap (joint supervision was not tested on the held-out generator)."

5. **[Claim without evidence] "the marker-repaired models ... bound that part" of the held-out gap**
   - Paper says: results.tex:108 "part of this gap is a marker gap; the marker-repaired models in Sec.~\ref{sec:res-marker} bound that part."
   - Source shows: the marker-repaired models were trained on all tools (one seed); no Gemini-only marker-repaired model exists (`T4_heldout.csv`), so they say nothing about the Gemini-to-Ideogram gap.
   - Suggested fix: "part of this gap may be a marker gap, which this design cannot separate: we did not train Gemini-only models on marker-repaired images."

6. **[Claim] Joint supervision "improves the pixel model on every axis we measured" is contradicted by the source**
   - Paper says: results.tex:131.
   - Source shows: test pixel-GT AUPRC (n=89) is **0.650** for `U-Net+box-aux|pix` vs **0.654 ± 0.016** for `U-Net|main|pix` (`T2_test.csv`, AUPRC_pix_89). The other quoted values (0.610 to 0.622, 0.579 to 0.647, 0.638 to 0.682, -59 % to -43 %, 0.897 vs 0.892) match.
   - Suggested fix: "\emph{Joint supervision} ... is the only variant that improves the pixel model on almost every axis we measured (the 89-pair test pixel AUPRC is unchanged, 0.650 vs.\ 0.654): ..." Also the blur numbers are in `T6`, not in Table~\ref{tab:dualgt}: change "(Table~\ref{tab:dualgt})" to "(Table~\ref{tab:dualgt}; blur from the robustness runs)".

---

### MEDIUM

1. **[Seed labelling] Single-seed YOLO results are placed under "Over three seeds" and compared with three-seed U-Nets without a label**
   - Paper says: results.tex:137 "Over three seeds it reaches ... trained on Gemini only it keeps 0.752 AP on Ideogram (U-Net 0.412; Gemini 0.938 vs. 0.909) ... The probes explain part of this: removing the residue costs the detector only 0.062 AP ... AUROC 0.865". abstract.tex:2 "transfers better from Gemini to Ideogram".
   - Source shows: `T4_heldout.csv` YOLO11s geminionly `seeds=1`; `T3_probes.csv` YOLO11s `seeds=1`. U-Net comparators are three-seed.
   - Suggested fix: "trained on Gemini only (one seed) it keeps 0.752 AP on Ideogram (three-seed Gemini-only U-Net 0.412; Gemini 0.938 vs.\ 0.909)"; "The probes (seed 42 for the detector) explain part of this". Abstract: "depends less on the marker and transfers better to Ideogram edits (0.87 vs.\ 0.63 AP, three seeds)" (the all-tools comparison 0.872 vs 0.629 is three-seed for both).

2. **[Seed labelling] Marker-repaired comparisons presented as gains/established without the one-seed caveat**
   - Paper says: discussion.tex:5 "gains slightly on residue-free Gemini images (0.756 vs. 0.722)"; results.tex:49 "on residue-free Gemini images rises from 0.722 to 0.756"; intro.tex:10 "we retrain every condition on marker-repaired images"; abstract "Retraining on marker-repaired images removes this dependence".
   - Source shows: `T3b_test_marker_subsets.csv` repaired box (seeds=1) 0.756 [0.690, 0.816] vs U-Net main box (seeds=3) 0.722 [0.657, 0.788]: overlapping CIs, single seed vs three-seed mean.
   - Suggested fix: discussion "and scores similarly on residue-free Gemini test images (0.756 vs.\ 0.722; one seed, overlapping CIs)"; results.tex:49 "and on residue-free Gemini images is 0.756 (0.722 before; overlapping CIs)"; intro "we retrain every condition on marker-repaired images (one seed)"; abstract "Retraining on marker-repaired images (one seed) removes this dependence".

3. **[Numbers / Terms] Marker prevalence is reported with three different denominators, and the discussion phrasing is wrong**
   - Paper says: data.tex:9 "83.5 % of Gemini training forgeries and 1.4 % of Ideogram ones"; results.tex:108 "(2.1 % vs. 85.9 %)"; discussion.tex:5 "Since the marker is 86 % Gemini and 2 % Ideogram".
   - Source shows: `dataset_curation.md` tool × marker over all 4,061 images: Gemini 85.9 %, Ideogram 2.1 %. 83.5 % / 1.4 % is not in any staged primary file (`reports/Proje_Inceleme_LabRat.md` quotes 83.6 % / 1.5 %; the source `green_marker_summary.csv` is not staged). "The marker is 86 % Gemini" is wrong: about 99.7 % of marker-bearing images are Gemini (3,126 of 3,135).
   - Suggested fix: discussion "Since the residue is present in 86\,\% of Gemini and 2\,\% of Ideogram forgeries,"; results "(2.1\,\% vs.\ 85.9\,\% of all 4{,}061 forgeries)"; data: keep 83.5 %/1.4 % only if it is traced to a file, and say "of Gemini training forgeries (85.9\,\% over all splits)".

4. **[Method / Eval integrity] The marker rule needs the authentic image, but test marker subsets cover all 812 test images**
   - Paper says: data.tex:9 the rule requires the residue to be "absent from the authentic image"; results.tex:47 and Table `tab:yolo` split the test set into 627 with / 185 without residue.
   - Source shows: only 89 of 812 test forgeries have a verified authentic pair (`audit_report.md` §3b), yet `T3b_test_marker_subsets.csv` labels all 812 (627 + 185), and `dataset_curation.md` labels all 4,061. The rule used for unpaired images is undescribed.
   - Suggested fix: add to data.tex: "For the 723 test forgeries without a paired authentic image, residue is detected by the colour rule alone within the ring (no authentic comparison); on paired images this variant agrees with the paired rule in X\,\% of cases." (fill X from the audit script).

5. **[Numbers] χ² bound is wrong**
   - Paper says: data.tex:11 "($\chi^2$ $p\ge0.31$)".
   - Source shows: `dataset_curation.md` §3: tool p = **0.308**, area 0.422, marker 0.892, source 0.997.
   - Suggested fix: "($\chi^2$ $p\ge0.30$)" or "($\chi^2$ $p>0.3$)".

6. **[Numbers] Median box height is 33 px, not 32 px**
   - Paper says: data.tex:7 "the median box height is 32 px"; method.tex:12 "Because the median field is 32 px high".
   - Source shows: `audit_report.md` §10: mask bbox height median **33** px (bbox 191×33). 32 matches `y1-y0` of the metadata bbox, which is inclusive (audit §3: "x2,y2 inclusive"), so 32 is an off-by-one.
   - Suggested fix: "33\,px" in both places.

7. **[Code vs paper] YOLO is not trained on "the same \GTb{} boxes"**
   - Paper says: method.tex:12 "YOLO11 (single class) is trained on the same \GTb{} boxes".
   - Code shows: `convert_to_yolo.py:37-42, 84, 132` take boxes from metadata `bbox_xyxy`, not from the mask, and use `x1-x0` widths. The audit (`audit_report.md` §3) finds that metadata is inclusive (so every label is 1 px short in width and height) and that 12 SROIE rows have a metadata bbox belonging to another image (IoU < 0.9), with the recommendation "use the mask, not the metadata bbox".
   - Suggested fix: derive labels from the mask (`mask_bbox(read_mask(...))`, +1 on x1/y1) and retrain; or state: "YOLO labels are taken from the metadata boxes, which coincide with the mask boxes except for 14 of 4,061 images (12 with corrupted metadata)."

8. **[Code vs paper] YOLO checkpoint selection is not by mAP@0.5**
   - Paper says: method.tex:12 "selection by validation-tile mAP@0.5".
   - Code shows: `train_yolo.py:58-61` uses Ultralytics `model.train`, which saves `best.pt` by its fitness, 0.1·mAP50 + 0.9·mAP50-95 by default (Ultralytics forum: default weights [P, R, mAP50, mAP50-95] = [0, 0, 0.1, 0.9]). `train_yolo.py:66` computes `best_epoch` as the mAP50 argmax for logging only, so `T5` `best_epoch` may not be the epoch of the `best.pt` used in `probes_p3.py:59`.
   - Suggested fix: "selection by the Ultralytics validation-tile fitness ($0.1\,\mathrm{AP}_{50}+0.9\,\mathrm{AP}_{50:95}$)", and recompute `best_epoch` from fitness.

9. **[Code vs paper / Eval integrity] The "negative" tile can contain up to 49 % of the forged field labelled empty**
   - Paper says: method.tex:12 "all tiles holding ≥50 % of the box plus one random negative tile per image".
   - Code shows: `convert_to_yolo.py:74, 78, 84`: every tile with an overlap fraction < 0.5 goes into `neg` and gets an empty label, including tiles that contain part of the box. With 1024/768 tiling and boxes up to 2,726 px wide, a "negative" can show a partly forged field with no label.
   - Suggested fix: change code to `(pos if r >= 0.5 else neg if inter == 0 else skip)` and retrain, or describe it: "plus one random tile holding less than half of the box, labelled empty."

10. **[Code vs paper] Large images are not processed at native resolution**
    - Paper says: data.tex:7 "all models run at native resolution with tiling"; method.tex:10 "Inference is full-image sliding window".
    - Source shows: `phase2/experiments/README.md` §2 and §7: images with longest side > 4096 (15 val, 29 test) are downscaled to 4096 and the map is upsampled bilinearly.
    - Suggested fix: method.tex:10 add "images whose longest side exceeds 4{,}096\,px (15 validation, 29 test) are downscaled to 4{,}096 for inference and the map is upsampled back."

11. **[Method / Terms] Image-level score definition does not match what is reported**
    - Paper says: method.tex:14 "Image-level discrimination uses the maximum score".
    - Source shows: control AUROCs quoted in results.tex:43 and :131 (2-3 points, 0.897 vs 0.892) are `auroc_top1` (99th percentile, `T7_controls.csv`; `hypothesis_decisions.md` "top-1 % skor"); `auroc_max` differs (e.g. U-Net pix 0.9176). Probe AUROCs use `score_max` (`probes_p3.py:119-124`).
    - Suggested fix: "Image-level discrimination uses the 99th-percentile score (benign controls) or the maximum score (input probes, Tables~\ref{tab:marker} and~\ref{tab:render})."

12. **[Claim] Grayscale "every model" includes models never tested**
    - Paper says: results.tex:133 "costing 47--67 % of the AUPRC of every model including the marker-repaired ones".
    - Source shows: `T6_robustness.csv` has "—" for gray and CLAHE on all three U-Net main (three-seed) rows; range over measured models -46.7 % to -66.7 % is correct.
    - Suggested fix: "costing 47--67\,\% of the AUPRC of every model for which it was measured (all single-seed models, including the marker-repaired ones, and the detector; the three-seed U-Nets were not tested)". Same caveat for "CLAHE changes AUPRC by at most 3 %".

13. **[Claim] Tile-border explanation is not supported by the box-count statistic cited**
    - Paper says: results.tex:139 "tiled inference leaves fields cut at tile borders as several partial boxes (about six per image) that NMS does not merge".
    - Source shows: `detection_test.csv` mean boxes/image: tiled s42 5.98 vs untiled 1280 **4.93**; `tables_p3.md` T5c val: tiled 5.82 vs untiled **5.85**; tiles_authneg 2.92 boxes/image yet AP50 0.874 (lower than 0.882). Mean box count does not measure fragmentation of one field, and fewer boxes do not raise AP50.
    - Suggested fix: "... but \emph{raises} box AP@0.5 from 0.88 to 0.96 (0.93 on fields below 1\,\% of the image). We suspect that fields cut at tile borders yield partial boxes that NMS does not merge, but we did not measure this directly; ..."

14. **[Internal contradiction] "box detection delivers it cheaply"**
    - Paper says: discussion.tex:3. Results say detection "was not cheaper" (results.tex:139: YOLO11s 98 ms vs U-Net 81 ms).
    - Suggested fix: "and a box detector delivers it with the highest box scores, but it should not be reported as pixel-level tamper localization."

15. **[Numbers] Abstract mixes test and validation for YOLO without saying so**
    - Paper says: abstract.tex:2 "(0.97 test AUPRC) ... yet scores 0.37 under the pixel-change truth".
    - Source shows: 0.37 is validation (`T1` 0.368 ± 0.015); on the 89 test pairs YOLO scores 0.404 ± 0.042 (`T2`).
    - Suggested fix: "yet scores 0.37 validation AUPRC under the pixel-change truth".

16. **[Terms] Internal project label "Phase-2" used in the paper**
    - Paper says: results.tex:108 "confirms the Phase-2 decision with three seeds"; results.tex:132 "the Phase-2 concern that ten epochs were too few".
    - Suggested fix: "confirms, with three seeds, the single-seed result of our preliminary runs"; "the concern that ten epochs were too few is not supported".

17. **[Terms] H1-H4 are used but never stated, nor their falsification criteria**
    - Paper says: method.tex:14 "H1--H4 and their falsification criteria were fixed before the full-data runs"; results use "H2 partly supported", "H4 rejected".
    - Source: `phase2/stats/hypothesis_decisions.md` defines them.
    - Suggested fix: add to method.tex: "H1: the training mask changes metric values and model ranking (falsified if $\Delta$F1$<0.05$ and Kendall $\tau\ge0.9$); H2: the box model relies on the box boundary; H3: pixel/glyph supervision transfers better across generators and degradations; H4: a learnable global trace exists outside the box (falsified if AUROC$\approx0.5$)."

---

### LOW

1. [Numbers] results.tex:6 "($n{=}89$ pairs)": 88 of the 89 have a non-empty pixel GT (`hypothesis_decisions.md` exclusions). Suggest "($n{=}88$ pairs with non-empty \GTp)".
2. [Numbers] Table `tab:heldout` caption "(val+test ...)": \GTp-scored rows use validation only (n=41; test pairs contain no Ideogram, README §7). Suggest "(\GTb: val+test, $n{=}126$; \GTp: val, $n{=}41$; mean over three seeds; CIs exclude seed variance)".
3. [Numbers] Table `tab:marker` caption mentions "95 % group-bootstrap CI for the repair contrast" but no CI is shown in the table.
4. [Numbers] data.tex:5 statistics (0.49, IQR 0.37-0.62, 33.5 %) are over 3,336 verified pairs (3,247 train + 89 test, audit §3b), not training pairs only. Suggest "over all 3{,}336 verified pairs".
5. [Numbers] data.tex:5 "Half of the Gemini outputs carry a global ±1-3 re-encoding difference": audit gives 48.6 % of all pairs (1,622/3,336), not a Gemini-specific share. Suggest "About half of all pairs (48.6\,\%) carry ...".
6. [Numbers] data.tex:11 "Fourteen degenerate training pairs (forged ≈ authentic, or all changes outside the box)": README §2 says the 14 include 2 unverified pairs (`pair_accepted=False`). Suggest "(forged ≈ authentic, all changes outside the box, or pair not verified)".
7. [Code vs paper] Undocumented settings. U-Net: photometric jitter ±8 % brightness/contrast (`train_unet_p3.py:45`), weight decay 1e-4, early-stop patience 4, selection on fixed validation crops with ≤20k sampled negatives per crop (`train_unet_p2.py:38-61`). YOLO: cosine LR, batch 8, hsv_s 0.1, hsv_v 0.08, translate 0.1, fliplr 0.5 (`train_yolo.py:25-27, 58-61`). The `--epochs` default is 40 but 30 were run (`checkpoint_summary_yolo.csv`); document the command line.
8. [Code vs paper] YOLO "conservative" profile keeps `translate=0.1`. Ultralytics applies it with `cv2.warpAffine`, which as far as I know shifts by fractional pixels with bilinear interpolation, i.e. resamples. This conflicts with the stated rationale "no ... scale jitter, which would add resampling traces". Verify against the installed version.
9. [Code vs paper] data.tex:9 "a 6-px ring around the box": code `probes_p3.py:78` is dilation(6) minus erosion(6), a band about 12 px wide straddling the box edge. Suggest "a band of ±6\,px around the box edge".
10. [Numbers] Table `tab:yolo` ms column: U-Net \GTp{} (3) and repaired rows reuse `unet_bbox_s42` timing (only unet_bbox, dlv3p, unetjoint were benchmarked, `speed_bench.csv`). Add "(same architecture as U-Net \GTb)". The causal claim "because per-tile pre- and post-processing dominates" (results.tex:139) has no profiling data behind it, and YOLO11m (76 ms) being faster than YOLO11n (81 ms) suggests timing noise. Soften to "possibly because ...".
11. [Numbers] Table `tab:yolo` YOLO11m AP50 0.898: source 0.8975 (`detection_test.csv`), shown as 0.897 in `T2`/`T5`. Pick one rounding.
12. [Eval] Bootstrap CIs are computed after averaging seeds per image, so they exclude seed variance (`hypothesis_decisions.md`). State this once in Method.
13. [Claim] results.tex:108 "Every Gemini-only model collapses on Ideogram": for the \GTb{} model scored under \GTp{}, 0.259 [0.168, 0.350] vs 0.349 [0.322, 0.377] overlaps. Suggest "drops on Ideogram".
14. [Terms] results.tex:108 "under its own ground truth the pixel model is no better on Ideogram than the box model (0.279 vs 0.259)": both are scored under \GTp, which is not the box model's own GT. Suggest "under the pixel-change ground truth ...".
15. [Claim] results.tex:43 "so no global re-encoding trace is exploitable": `hypothesis_decisions.md` post-hoc finds a within-pair trace (forged scored higher in 92 % of the 197 differing pairs) that vanishes against the symmetric resample control. Suggest "no image-level global trace is exploitable".
16. [Terms] The joint model has four names: "U-Net + box aux.", "U-Net+aux", "joint model", "Joint supervision". Pick one.
17. [Method] `render_only` reverts only the morphologically cleaned \GTp{} region dilated by 2 px (`probes_p3.py:81-82`). Isolated D>8 pixels removed by the 3×3 opening stay forged, so "no character changes" (method.tex:7) is approximate.
18. [Eval] Per-image AP in probes subsamples up to 300k negatives per image (`probes_p3.py:30-36`). It is approximate, unlike the "exact" pooled AUPRC.
19. [Eval] Validation is used for checkpoint selection, τ* calibration and the Table 1 numbers. Table 1 is threshold-free, but the val numbers are not held-out; worth one sentence.
20. [Citations / other] 17 unused entries in `refs.bib` (harmless with BibTeX). The author block reads "AIC515 Image Processing" while the project is titled "AI 515"; check the official course code.

---

### Numerical Audit
| Paper claim | Location (.tex) | Source file/code | Source value | Match? |
|---|---|---|---|---|
| 4,061 (3,249/812) | data:3 | audit_report §0 | 3249/812=4061 | Yes |
| receipts 89.7 %, forms 10.3 % | data:3 | dataset_curation §1 | 89.7/10.3 | Yes |
| Gemini 89.6 %, Ideogram 10.4 % | data:3 | dataset_curation §1 | 89.6/10.4 | Yes |
| 3,247 of 3,249 verified | data:5 | audit §3b | 3247/3249 | Yes |
| median 0.49, IQR 0.37-0.62 | data:5, intro:5 | audit §3b | 0.492 [0.374-0.620] | Yes (over 3,336 pairs, LOW-4) |
| 33.5 % outside (Gemini 35.9, Ideogram 11.9) | data:5, intro:5 | audit §3b | 33.5/35.9/11.9 | Yes |
| half of Gemini outputs ±1-3 trace | data:5 | audit §3b | ~48.6 % of all pairs | Partly (LOW-5) |
| median area 0.98 %, 50.7 % < 1 % | data:7 | audit §10 | 0.975 %, 50.7 % | Yes |
| median box height 32 px | data:7, method:12 | audit §10 | 33 px | **No** (MED-6) |
| 0.03-35 MP | data:7 | audit §9 | 0.03-34.81 | Yes |
| marker 83.5 % Gemini / 1.4 % Ideogram | data:9 | not in staged primary files (review: 83.6/1.5) | — | Unverified (MED-3) |
| Cramér's V = 0.61 (marker×tool), 0.43 (tool×source) | data:9 | dataset_curation §2 | 0.608, 0.429 | Yes |
| 2,762 / 487 | data:11 | audit §16 | 2762/487 | Yes |
| χ² p ≥ 0.31 | data:11 | dataset_curation §3 | min 0.308 | **No** (MED-5) |
| 8.9 % leakage, 740 clean | data:11 | audit §12 | 72/812=8.9 %, 740 | Yes |
| 86 matches | data:11 | audit §13 | 86/812 | Yes |
| 2,748 after 14 exclusions | data:11 | README §2/§4 | n_train 2748 | Yes |
| median IoU 0.35 (pix), 0.27 (glyph) | method:4 | Bildiri_Tam_Metin Table 2 | 0.350 / 0.269 | Yes (secondary) |
| 14.3 M params | method:10 | speed_bench | 14.33 | Yes |
| 512 crops, 70/30, BCE+Dice, AdamW 1e-4 cosine, bs16, bf16, 10 ep | method:10 | train_unet_p2/p3 | same | Yes |
| sliding window 512 / 448 | method:10 | README §2 | 512/448 | Yes (but >4096 downscale, MED-10) |
| joint loss L_pix + 0.5 L_box | method:10 | train_unet_p3.py:111-112 | (BCE+Dice)+W(BCE+Dice), W=0.5 | Yes |
| tiles 1024 / 768, ≥50 % + 1 neg | method:12 | convert_to_yolo.py:72-78 | same | Yes (neg labelling, MED-9) |
| AdamW 1e-3, 30 ep, patience 10 | method:12 | train_yolo.py; checkpoint_summary | lr0 1e-3, 30 ep, patience 10 | Yes |
| selection by val-tile mAP@0.5 | method:12 | train_yolo.py (Ultralytics fitness) | 0.1 mAP50 + 0.9 mAP50-95 | **No** (MED-8) |
| 487 val / 812 test / n=89 | method:14 | T1/T2 | 487/812/89 (88 non-empty) | Yes (LOW-1) |
| 0.92 → 0.37; 0.923 / 0.374 / 0.257 | abstract, intro:9, results:6 | T1 | 0.923/0.374/0.257 | Yes |
| test 0.950 / 0.427 | results:6 | T2 | 0.950 / 0.427 | Yes |
| DLV3+ 0.939 → 0.382 | results:6 | T1 | 0.939 / 0.382 | Yes |
| Table dualgt (all 36 cells) | results:17-36 | T1 | all match | Yes |
| rectangularity 0.87/0.32/0.05 | results:43 | README T6 | 0.873/0.315/0.051 | Yes |
| inner +1.2 %, rim -2.0 %, placebo 0.0 % (3 seeds) | results:43 | README T5 | +0.012/-0.020/-0.000 | Yes |
| H4 AUROC 0.516 / 0.980 | results:43 | README T8 | 0.516 / 0.980 | Yes |
| controls 2-3 points for all models | results:43 | T7 | 1.6-6.5 | **No** (HIGH-3) |
| FP 12 % vs 26 % (3 seeds) | results:43 | T7 flag05 | 0.1157 / 0.2601 | Yes |
| 380 residue pairs | results:47 | T3 n_res | 380 | Yes |
| 0.988 → 0.764, Δ -0.224 [-0.261, -0.190] | results:47 | T3 (3 seeds) | 0.988 → 0.766, -0.222 [-0.256, -0.190] | **No** (HIGH-1) |
| cover 81 %, AP 0.895 | results:47 | T3 | 0.795, 0.905 | **No** (HIGH-1) |
| Δ -0.174 (DLV3+), -0.242 (pix) | results:47 | T3 | -0.174, -0.247 | Partly (HIGH-1) |
| test AP 0.972 (627) / 0.690 (185) / 0.722 (102) | results:47, disc:5 | T3b | 0.972/0.690/0.722 | Yes |
| repaired Δ +0.019 [0.002, 0.036] | results:49 | T3 | +0.019 [+0.002, +0.036] | Yes |
| repaired cover 42 % vs 81 %; 11 % vs 44 % | results:49 | T3 | 0.424 vs 0.795; 0.108 vs 0.420 | **No** for 81/44 (HIGH-1) |
| repaired test AUPRC 0.939 (0.950) | results:49 | T2 | 0.939 / 0.950 | Yes |
| repaired 0.920 (marker), 0.756 (no-mk Gemini) | results:49 | T3b | 0.920 / 0.756 | Yes |
| Table marker DLV3+, aux, repaired rows | results:64-68 | T3 | match | Yes |
| Table marker U-Net rows | results:62-63 | T3 (3 seeds) | see HIGH-1 | **No** |
| Table render DLV3+, repaired rows | results:98-100 | T3 | match | Yes |
| Table render U-Net rows; text 0.87/0.22/0.34, 0.972/0.631/0.728 | results:83, 96-97 | T3 (3 seeds) | 0.864/0.237/0.348, 0.971/0.637/0.733 | **No** (HIGH-2) |
| content-only 0.872 / 0.834 (0.681), render 0.713 / 0.638 | results:83 | T3 | 0.872/0.834/0.681; 0.713/0.638 | Yes |
| held-out n = 126 / 41 | results:108, disc:9 | T4; README §7 | 126 (41+85) / 41 (val only) | Yes |
| 0.909 → 0.412; 0.567 → 0.264; 0.279 vs 0.259 | results:108 | T4 | match | Yes |
| Table heldout (all 12 cells) | results:119-124 | T4 | match | Yes |
| 2.1 % vs 85.9 % | results:108 | dataset_curation tool×marker | 2.1 / 85.9 | Yes (different denominator from data:9) |
| joint 0.610→0.622, 0.579→0.647, 0.638→0.682, -59→-43 %, 0.897 vs 0.892 | results:131 | T1/T2/T6/T7 | match | Yes ("every axis" false, HIGH-6) |
| 20 ep 0.926/0.594/0.645 | results:132 | T1 | match | Yes |
| gray 47-67 %, CLAHE ≤3 % | results:133 | T6 | 46.7-66.7 %, ≤3.0 % | Yes for measured models (MED-12) |
| blur -59 % vs -21 %; repaired JPEG -32 to -50 % | results:133 | T6 | -59.1/-20.7; -31.6/-32.5/-50.3 | Yes |
| YOLO 0.965 val / 0.969 test; U-Net 0.923 / 0.950 | results:137 | T1/T2 | match | Yes |
| F1 0.946 / IoU 0.918; U-Net 0.846 / 0.789 | results:137 | T2 | match | Yes |
| AP50 0.880; U-Net 0.873 (s42/43), 0.391 (s44) | results:137, tab:yolo | detection_test | 0.8799; (0.8742+0.8724)/2=0.8733; 0.3905 | Yes |
| YOLO pix 0.368 (U-Net 0.374) | results:137 | T1 | 0.368 / 0.374 | Yes |
| no-mk 0.878 (0.690), Gemini no-mk 0.886 (0.722) | results:137 | T3b | match | Yes |
| Gemini-only YOLO 0.752, Gemini 0.938 (U-Net 0.412 / 0.909) | results:137 | T4 | match (YOLO 1 seed) | Yes (MED-1 label) |
| all-tools Ideogram 0.872 (U-Net 0.629) | results:137 | T4 | match (both 3 seeds) | Yes |
| YOLO probe -0.062, render AUROC 0.865 | results:137 | T3 (1 seed) | -0.062 / 0.865 | Yes |
| U-Net comparators 0.224 / 0.631 | results:137 | T3 (3 seeds) | 0.222 / 0.637 | **No** (HIGH-1/2) |
| 1280: -0.037 AUPRC, -0.122 AP, AP50 0.88 → 0.96 (0.93 < 1 %) | results:139 | T2/T3b/T5 | 0.969-0.932, 0.878-0.756, 0.960, 0.933 | Yes |
| n/s/m 0.963/0.969/0.969; defaug 0.971; authneg 0.971 | results:139 | T2 | match | Yes |
| YOLO robustness ≤1.4 %, JPEG 49 %, gray 59 % | results:139 | T6 | -1.4/-1.2/-0.7; -48.5; -58.6 | Yes |
| speeds 76-98, 303, 81, 122 ms; 3.1 MP | results:139, tab:yolo | speed_bench | 76.3/80.5/98.1; 303.0; 80.9; 122.0; 3.14 | Yes |
| Table yolo (all other cells) | results:150-158 | T2/T3b/detection_test/speed | match | Yes (LOW-10, LOW-11) |
| TruFor F1@0.5 0.403 | tab:yolo footnote | zero_shot_results (trufor test full) | 0.4025 | Yes |
| 0.55 AUPRC loss | disc:3 | T1 | 0.923-0.374 = 0.549 | Yes |
| repaired 0.916 (0.923) | disc:5 | T1 | 0.916 / 0.923 | Yes |
| VLM 69/26/1 of 96 | disc:9 | Proje_Inceleme W1; Bildiri_Tam_Metin | 96 pairs, 69 %/26 % | Consistent (secondary) |
| TruFor IoU 0.36 zero-shot (AIForge-Doc v1) | intro:3 | reports/inceleme/paper_check.md (v1 Table 2: IoU 0.358) | 0.358 | Yes (secondary) |

### Unverified Claims
| .tex location | Claim | Cited paper | Status | Suggestion |
|---|---|---|---|---|
| data:9 | 83.5 % Gemini / 1.4 % Ideogram training forgeries; no false alarm on 600 authentic images | — | Source `green_marker_summary.csv` not staged; a review quotes 83.6/1.5 | Re-export and cite the file; reconcile with 85.9/2.1 |
| data:9 | dataset paper draws a 3-px bright-green rectangle (Sec. 3.4) | wu2026aiforgedoc | Quoted verbatim in `reports/Proje_Inceleme_LabRat.md` | OK |
| intro:3 | TruFor pixel IoU 0.36 zero-shot; document-specific models fall further | wu2026aiforgedoc, guillaro2023trufor | Prior check records v1 Table 2: TruFor IoU 0.358, DTD IoU 0.020 | OK |
| intro:3 | humans perform at chance on v2 | wu2026aiforgedocv2 | Prior check: 2AFC 0.501 | OK |
| abstract, fig caption | residue is 1-3 px | — | No measurement file staged | Cite the residue-width statistic or drop "1--3 px" |
| results:139 | per-tile pre/post-processing dominates runtime | — | No profiling data | Soften (LOW-10) |
| results:139 | partial boxes at tile borders explain the lower AP50 | — | Box counts do not support it (MED-13) | Hedge |
| method:12, 14 | cross-tile NMS; segmenter boxes from connected components at τ* | — | `infer_to_maps.py` / box-eval code not staged; probes call it with iou=0.6, conf=0.001 | Verify against the code |
| discussion:9 | TruFor training code not released | guillaro2023trufor | Not checked | Keep, or cite the repository |

### Corrected BibTeX Entries
None. DBLP verification was skipped as requested. All 62 distinct `\cite` keys exist in `refs.bib`; 17 entries are unused.

### Verified Claims
- Every cite key resolves; the PDF compiles cleanly (8 pages).
- All Table 1 (dual-GT) cells, all Table `tab:heldout` cells, and all Table `tab:yolo` cells except YOLO11m AP50 rounding match `T1`/`T2`/`T3b`/`T4`/`detection_test`/`speed_bench`.
- Dataset statistics (4,061; splits; tool/source/doc-type shares; 0.49 box-change fraction; 33.5 % outside change; 0.98 % area; 8.9 % leakage; 740; 86; 2,762/487; Cramér's V 0.61 and 0.43) match `audit_report.md` and `dataset_curation.md`.
- Held-out counts n=126 (val 41 + test 85, box) and n=41 (val only, pixel) are correct.
- Three-seed labels are correct for: U-Net main grid (T1, T2, T6 blur/JPEG/rescale, T7), held-out Gemini-only U-Nets (T4), YOLO11s tiles main (T1, T2, T3b, T4 all-tools). The Limitations seed list (discussion:9) is correct.
- U-Net/YOLO hyperparameters in Method (crop sampling 70/30, BCE+Dice, AdamW 1e-4 cosine, batch 16, bf16, 10 epochs, sliding window 512/448, joint loss weight 0.5, tiles 1024/768 with the ≥50 % rule, AdamW 1e-3, 30 epochs, patience 10) match the code.
- Gemini-only U-Nets are selected on Gemini-only validation (`train_unet_p2.py:87-88`) and Gemini-only YOLO validation tiles exclude Ideogram (`convert_to_yolo.py:113-114`): no held-out leakage into checkpoint selection.
- Test τ* comes from validation; test-leakage subsets (740) are reported; image-level detection uses validation pairs only, as stated.
- Speeds in ms (76/81/98, 303, 81, 122; 3.1 MP; idle RTX 5080) match `speed_bench.csv`.

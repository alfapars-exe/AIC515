# Faz 3 — sentez tabloları (otomatik; synthesize_p3.py)

## T1. Dual-GT (val 487, temiz) — tüm aileler

| family | variant | train_gt | seeds | AUPRC_box | F1tau_box | AUPRC_glyph | F1tau_glyph | AUPRC_pix | F1tau_pix |
|---|---|---|---|---|---|---|---|---|---|
| U-Net | main | box | 3 | 0.923 ± 0.015 | 0.842 ± 0.023 | 0.257 ± 0.007 | 0.387 ± 0.009 | 0.374 ± 0.008 | 0.447 ± 0.011 |
| U-Net | main | glyph | 3 | 0.351 ± 0.015 | 0.401 ± 0.007 | 0.676 ± 0.019 | 0.674 ± 0.011 | 0.379 ± 0.012 | 0.457 ± 0.006 |
| U-Net | main | pix | 3 | 0.579 ± 0.014 | 0.600 ± 0.014 | 0.511 ± 0.010 | 0.532 ± 0.008 | 0.610 ± 0.012 | 0.577 ± 0.007 |

## T2. Resmî test (box GT; piksel GT 89 çift) + tespit AP

| family | variant | train_gt | seeds | AUPRC_box_812 | AUPRC_box_740 | F1tau_box | IoUtau_box | AUPRC_pix_89 |
|---|---|---|---|---|---|---|---|---|
| U-Net | main | box | 3 | 0.950 ± 0.011 | 0.949 ± 0.012 | 0.846 ± 0.015 | 0.789 ± 0.018 | 0.427 ± 0.002 |
| U-Net | main | glyph | 3 | 0.403 ± 0.011 | 0.403 ± 0.010 | 0.405 ± 0.007 | 0.266 ± 0.006 | 0.410 ± 0.017 |
| U-Net | main | pix | 3 | 0.638 ± 0.015 | 0.640 ± 0.014 | 0.616 ± 0.013 | 0.480 ± 0.015 | 0.654 ± 0.016 |

## T3b. Test: işaretçili vs işaretçisiz görüntüler (box GT, görüntü-AP)

| family | variant | train_gt | seeds | AP_with_marker | n_with_marker | AP_no_marker | n_no_marker | AP_no_marker_gemini | n_nm_gem |
|---|---|---|---|---|---|---|---|---|---|
| U-Net | main | box | 3 | 0.972 [0.966, 0.977] | 627 | 0.690 [0.640, 0.739] | 185 | 0.722 [0.657, 0.788] | 102 |
| U-Net | main | glyph | 3 | 0.388 [0.377, 0.399] | 627 | 0.339 [0.307, 0.374] | 185 | 0.361 [0.317, 0.405] | 102 |
| U-Net | main | pix | 3 | 0.639 [0.624, 0.652] | 627 | 0.503 [0.455, 0.550] | 185 | 0.559 [0.504, 0.622] | 102 |

## T4. Held-out üretici (val+test; Ideogram vs kaynak-eşlenmiş Gemini; görüntü-AP)

| family | variant | train_gt | gemini_only | seeds | eval_gt | n_ideogram | AP_ideogram | AP_gemini_matched | gap |
|---|---|---|---|---|---|---|---|---|---|
| U-Net | geminionly | box | True | 1 | box | 126 | 0.395 [0.324, 0.471] | 0.908 [0.890, 0.927] | 0.513 |
| U-Net | geminionly | box | True | 1 | pix | 41 | 0.254 [0.161, 0.345] | 0.347 [0.319, 0.375] | 0.092 |
| U-Net | geminionly | glyph | True | 1 | box | 126 | 0.199 [0.160, 0.241] | 0.394 [0.380, 0.410] | 0.195 |
| U-Net | geminionly | glyph | True | 1 | pix | 41 | 0.233 [0.162, 0.311] | 0.406 [0.377, 0.434] | 0.173 |
| U-Net | geminionly | pix | True | 1 | box | 126 | 0.255 [0.204, 0.305] | 0.559 [0.542, 0.576] | 0.304 |
| U-Net | geminionly | pix | True | 1 | pix | 41 | 0.279 [0.181, 0.386] | 0.582 [0.552, 0.614] | 0.303 |
| U-Net | main | box | False | 3 | box | 126 | 0.629 [0.569, 0.692] | 0.912 [0.896, 0.928] | 0.283 |
| U-Net | main | box | False | 3 | pix | 41 | 0.374 [0.286, 0.458] | 0.359 [0.332, 0.386] | -0.016 |
| U-Net | main | glyph | False | 3 | box | 126 | 0.312 [0.269, 0.357] | 0.369 [0.356, 0.382] | 0.057 |
| U-Net | main | glyph | False | 3 | pix | 41 | 0.365 [0.276, 0.440] | 0.389 [0.364, 0.414] | 0.024 |
| U-Net | main | pix | False | 3 | box | 126 | 0.423 [0.367, 0.481] | 0.574 [0.557, 0.590] | 0.151 |
| U-Net | main | pix | False | 3 | pix | 41 | 0.454 [0.350, 0.550] | 0.571 [0.540, 0.603] | 0.117 |

## T5. YOLO ablasyonları ve maliyet

|  |
|

## T6. Dayanıklılık (relΔ pooled AUPRC, kendi GT'sinde)

| family | variant | train_gt | blur1 | jpeg75 | rescale075 |
|---|---|---|---|---|---|
| U-Net | main | box | -20.7% ± 6.7 | -16.6% ± 1.4 | -12.2% ± 4.3 |
| U-Net | main | glyph | -59.0% ± 6.8 | -18.0% ± 0.8 | -42.7% ± 5.4 |
| U-Net | main | pix | -59.1% ± 4.5 | -22.5% ± 2.3 | -40.3% ± 1.5 |

## T7. Kontroller (val; image-level AUROC forged vs kontrol; bayrak oranı @0.5 (>0.1 % piksel); FP@τ*)

| family | variant | train_gt | source | seeds | auroc_top1 | auroc_top1_sd | auroc_max | flag05 | fp_tau |
|---|---|---|---|---|---|---|---|---|---|
| U-Net | main | box | authentic | 3 | 0.8917 | 0.0465 | 0.9224 | 0.2601 | 0.0016 |
| U-Net | main | box | benign_pasteback | 3 | 0.872 | 0.0467 | 0.8993 | 0.3128 | 0.0021 |
| U-Net | main | box | fullimage_resample | 3 | 0.8664 | 0.0526 | 0.9 | 0.3381 | 0.0022 |
| U-Net | main | glyph | authentic | 3 | 0.7991 | 0.0138 | 0.7496 | 0.2286 | 0.0038 |
| U-Net | main | glyph | benign_pasteback | 3 | 0.7787 | 0.0129 | 0.7242 | 0.2704 | 0.004 |
| U-Net | main | glyph | fullimage_resample | 3 | 0.7692 | 0.0196 | 0.7132 | 0.2704 | 0.0042 |
| U-Net | main | pix | authentic | 3 | 0.8916 | 0.0057 | 0.9176 | 0.1157 | 0.002 |
| U-Net | main | pix | benign_pasteback | 3 | 0.8758 | 0.0097 | 0.9031 | 0.1465 | 0.0022 |
| U-Net | main | pix | fullimage_resample | 3 | 0.8634 | 0.0121 | 0.8868 | 0.1663 | 0.0025 |

## T8. Faz 1 AUPRC düzeltmesi (W6)

| variant | reported_phase1_auprc | reproduced_subsampled_auprc | exact_auprc_all_pixels | exact_auroc | pos_rate | n_crops | n_pixels |
|---|---|---|---|---|---|---|---|
| rgb | 0.947 | 0.9423 | 0.8053 | 0.9344 | 0.03306 | 200 | 52428800 |
| rgbfreq | 0.939 | 0.9331 | 0.7922 | 0.9255 | 0.03306 | 200 | 52428800 |

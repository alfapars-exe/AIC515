# Faz 3 — sentez tabloları (otomatik; synthesize_p3.py)

## T1. Dual-GT (val 487, temiz) — tüm aileler

| family | variant | train_gt | seeds | AUPRC_box | F1tau_box | AUPRC_glyph | F1tau_glyph | AUPRC_pix | F1tau_pix |
|---|---|---|---|---|---|---|---|---|---|
| DeepLabV3+ | main | box | 1 | 0.939 | 0.839 | 0.256 | 0.402 | 0.382 | 0.464 |
| DeepLabV3+ | main | glyph | 1 | 0.345 | 0.498 | 0.549 | 0.560 | 0.401 | 0.494 |
| DeepLabV3+ | main | pix | 1 | 0.537 | 0.579 | 0.432 | 0.479 | 0.583 | 0.542 |
| U-Net | 20ep | box | 1 | 0.926 | 0.851 | 0.259 | 0.397 | 0.373 | 0.460 |
| U-Net | 20ep | glyph | 1 | 0.398 | 0.394 | 0.645 | 0.651 | 0.390 | 0.452 |
| U-Net | 20ep | pix | 1 | 0.568 | 0.587 | 0.488 | 0.529 | 0.594 | 0.576 |
| U-Net | main | box | 3 | 0.923 ± 0.015 | 0.842 ± 0.023 | 0.257 ± 0.007 | 0.387 ± 0.009 | 0.374 ± 0.008 | 0.447 ± 0.011 |
| U-Net | main | glyph | 3 | 0.351 ± 0.015 | 0.401 ± 0.007 | 0.676 ± 0.019 | 0.674 ± 0.011 | 0.379 ± 0.012 | 0.457 ± 0.006 |
| U-Net | main | pix | 3 | 0.579 ± 0.014 | 0.600 ± 0.014 | 0.511 ± 0.010 | 0.532 ± 0.008 | 0.610 ± 0.012 | 0.577 ± 0.007 |
| U-Net | marker-repaired | box | 1 | 0.916 | 0.797 | 0.266 | 0.379 | 0.389 | 0.429 |
| U-Net | marker-repaired+nm-GT | glyph | 1 | 0.348 | 0.401 | 0.661 | 0.662 | 0.371 | 0.456 |
| U-Net | marker-repaired+nm-GT | pix | 1 | 0.513 | 0.524 | 0.455 | 0.479 | 0.544 | 0.502 |
| U-Net+box-aux | main | glyph | 1 | 0.370 | 0.441 | 0.636 | 0.638 | 0.391 | 0.466 |
| U-Net+box-aux | main | pix | 1 | 0.647 | 0.642 | 0.518 | 0.544 | 0.622 | 0.580 |

## T2. Resmî test (box GT; piksel GT 89 çift) + tespit AP

| family | variant | train_gt | seeds | AUPRC_box_812 | AUPRC_box_740 | F1tau_box | IoUtau_box | AUPRC_pix_89 |
|---|---|---|---|---|---|---|---|---|
| DeepLabV3+ | main | box | 1 | 0.952 | 0.952 | 0.833 | 0.765 | 0.408 |
| DeepLabV3+ | main | glyph | 1 | 0.368 | 0.365 | 0.494 | 0.353 | 0.389 |
| DeepLabV3+ | main | pix | 1 | 0.585 | 0.583 | 0.590 | 0.457 | 0.667 |
| U-Net | 20ep | box | 1 | 0.945 | 0.946 | 0.858 | 0.799 | 0.422 |
| U-Net | 20ep | glyph | 1 | 0.480 | 0.475 | 0.403 | 0.265 | 0.477 |
| U-Net | 20ep | pix | 1 | 0.609 | 0.606 | 0.602 | 0.466 | 0.638 |
| U-Net | main | box | 3 | 0.950 ± 0.011 | 0.949 ± 0.012 | 0.846 ± 0.015 | 0.789 ± 0.018 | 0.427 ± 0.002 |
| U-Net | main | glyph | 3 | 0.403 ± 0.011 | 0.403 ± 0.010 | 0.405 ± 0.007 | 0.266 ± 0.006 | 0.410 ± 0.017 |
| U-Net | main | pix | 3 | 0.638 ± 0.015 | 0.640 ± 0.014 | 0.616 ± 0.013 | 0.480 ± 0.015 | 0.654 ± 0.016 |
| U-Net | marker-repaired | box | 1 | 0.939 | 0.939 | 0.797 | 0.726 | 0.425 |
| U-Net | marker-repaired+nm-GT | glyph | 1 | 0.402 | 0.404 | 0.405 | 0.266 | 0.354 |
| U-Net | marker-repaired+nm-GT | pix | 1 | 0.564 | 0.562 | 0.537 | 0.405 | 0.570 |
| U-Net+box-aux | main | glyph | 1 | 0.432 | 0.433 | 0.448 | 0.304 | 0.460 |
| U-Net+box-aux | main | pix | 1 | 0.682 | 0.680 | 0.654 | 0.522 | 0.650 |

## T3. Sondalar (val çiftleri; işaretçili alt küme için AP; seed-ortalaması + grup bootstrap CI)

| model | seeds | n_res | APbox_forged | cover_forged | APbox_forged_repaired | cover_forged_repaired | APbox_authentic_plus_marker | cover_authentic_plus_marker | APbox_authentic | cover_authentic | dAP_repair | AUROC_forged | AUROC_forged_repaired | AUROC_authentic_plus_marker | AUROC_render_only | AUROC_content_only | inbox_render_only | APbox_all_render_only | inbox_content_only | APbox_all_content_only | inbox_forged | APbox_all_forged | dAPbox_clahe | dAPbox_gray |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DeepLabV3+|main|bbox | 1 | 380 | 0.982 [0.974, 0.988] | 0.920 | 0.807 [0.774, 0.837] | 0.641 | 0.928 [0.913, 0.941] | 0.775 | 0.184 [0.162, 0.210] | 0.040 | -0.174 [-0.206, -0.145] | 0.972 | 0.877 | 0.899 | 0.644 | 0.755 | 0.281 | 0.452 [0.415, 0.488] | 0.416 | 0.606 [0.572, 0.638] | 0.869 | 0.934 [0.918, 0.949] | -0.006 [-0.014, +0.001] | -0.343 [-0.376, -0.308] |
| DeepLabV3+|main|glyph_v2 | 1 | 380 | 0.627 [0.611, 0.643] | 0.477 | 0.503 [0.477, 0.528] | 0.352 | 0.569 [0.548, 0.587] | 0.424 | 0.147 [0.130, 0.169] | 0.051 | -0.124 [-0.149, -0.101] | 0.908 | 0.769 | 0.827 | 0.581 | 0.684 | 0.090 | 0.279 [0.256, 0.302] | 0.174 | 0.392 [0.367, 0.417] | 0.350 | 0.595 [0.576, 0.613] | +0.029 [+0.023, +0.035] | -0.191 [-0.220, -0.162] |
| DeepLabV3+|main|pixel_eps8_morph | 1 | 380 | 0.730 [0.711, 0.749] | 0.597 | 0.555 [0.523, 0.585] | 0.400 | 0.628 [0.604, 0.651] | 0.484 | 0.172 [0.151, 0.195] | 0.072 | -0.175 [-0.208, -0.146] | 0.934 | 0.763 | 0.828 | 0.548 | 0.677 | 0.071 | 0.277 [0.251, 0.303] | 0.189 | 0.437 [0.407, 0.466] | 0.420 | 0.700 [0.680, 0.720] | +0.026 [+0.018, +0.033] | -0.258 [-0.286, -0.228] |
| U-Net+box-aux|main|glyph_v2 | 1 | 380 | 0.622 [0.605, 0.640] | 0.381 | 0.470 [0.443, 0.496] | 0.245 | 0.546 [0.525, 0.567] | 0.338 | 0.111 [0.096, 0.127] | 0.030 | -0.153 [-0.181, -0.126] | 0.880 | 0.753 | 0.811 | 0.577 | 0.666 | 0.062 | 0.237 [0.215, 0.258] | 0.114 | 0.333 [0.306, 0.360] | 0.270 | 0.595 [0.576, 0.614] | +0.017 [+0.010, +0.023] | -0.278 [-0.308, -0.248] |
| U-Net+box-aux|main|pixel_eps8_morph | 1 | 380 | 0.771 [0.757, 0.785] | 0.589 | 0.597 [0.564, 0.628] | 0.377 | 0.668 [0.647, 0.688] | 0.482 | 0.142 [0.122, 0.164] | 0.035 | -0.174 [-0.206, -0.145] | 0.952 | 0.789 | 0.873 | 0.574 | 0.699 | 0.067 | 0.274 [0.245, 0.302] | 0.172 | 0.421 [0.390, 0.452] | 0.436 | 0.734 [0.715, 0.752] | +0.036 [+0.028, +0.043] | -0.304 [-0.334, -0.273] |
| U-Net|main|bbox | 1 | 380 | 0.988 [0.982, 0.992] | 0.958 | 0.764 [0.725, 0.797] | 0.635 | 0.895 [0.874, 0.913] | 0.806 | 0.154 [0.134, 0.179] | 0.041 | -0.224 [-0.261, -0.190] | 0.972 | 0.833 | 0.902 | 0.631 | 0.728 | 0.220 | 0.391 [0.356, 0.426] | 0.338 | 0.552 [0.517, 0.587] | 0.870 | 0.940 [0.922, 0.956] | -0.006 [-0.014, +0.001] | -0.430 [-0.465, -0.395] |
| U-Net|main|glyph_v2 | 1 | 380 | 0.527 [0.510, 0.546] | 0.337 | 0.409 [0.380, 0.435] | 0.208 | 0.456 [0.436, 0.478] | 0.291 | 0.116 [0.100, 0.135] | 0.024 | -0.118 [-0.143, -0.095] | 0.886 | 0.752 | 0.821 | 0.577 | 0.681 | 0.051 | 0.208 [0.188, 0.229] | 0.104 | 0.328 [0.301, 0.354] | 0.244 | 0.511 [0.493, 0.530] | +0.039 [+0.033, +0.046] | -0.198 [-0.222, -0.175] |
| U-Net|main|pixel_eps8_morph | 1 | 380 | 0.739 [0.722, 0.756] | 0.565 | 0.497 [0.464, 0.527] | 0.318 | 0.618 [0.592, 0.640] | 0.442 | 0.123 [0.104, 0.144] | 0.037 | -0.242 [-0.274, -0.210] | 0.925 | 0.759 | 0.828 | 0.558 | 0.681 | 0.055 | 0.201 [0.178, 0.223] | 0.170 | 0.366 [0.337, 0.396] | 0.426 | 0.697 [0.677, 0.717] | +0.030 [+0.023, +0.036] | -0.373 [-0.403, -0.342] |
| U-Net|marker-repaired+nm-GT|glyph_v2 | 1 | 380 | 0.510 [0.491, 0.530] | 0.321 | 0.510 [0.491, 0.531] | 0.314 | 0.345 [0.321, 0.369] | 0.186 | 0.119 [0.103, 0.138] | 0.029 | +0.001 [-0.008, +0.009] | 0.876 | 0.875 | 0.688 | 0.660 | 0.822 | 0.081 | 0.265 [0.244, 0.286] | 0.185 | 0.422 [0.399, 0.445] | 0.237 | 0.501 [0.482, 0.522] | +0.036 [+0.029, +0.043] | -0.185 [-0.209, -0.161] |
| U-Net|marker-repaired+nm-GT|pixel_eps8_morph | 1 | 380 | 0.648 [0.625, 0.670] | 0.444 | 0.686 [0.667, 0.705] | 0.503 | 0.256 [0.235, 0.281] | 0.108 | 0.131 [0.113, 0.152] | 0.033 | +0.038 [+0.021, +0.057] | 0.869 | 0.899 | 0.592 | 0.638 | 0.834 | 0.096 | 0.315 [0.288, 0.340] | 0.289 | 0.530 [0.506, 0.555] | 0.338 | 0.627 [0.606, 0.650] | +0.012 [+0.003, +0.021] | -0.225 [-0.248, -0.202] |
| U-Net|marker-repaired|bbox | 1 | 380 | 0.939 [0.920, 0.957] | 0.889 | 0.958 [0.942, 0.970] | 0.894 | 0.603 [0.563, 0.640] | 0.424 | 0.162 [0.138, 0.185] | 0.050 | +0.019 [+0.002, +0.036] | 0.947 | 0.957 | 0.695 | 0.713 | 0.872 | 0.356 | 0.542 [0.506, 0.579] | 0.590 | 0.752 [0.724, 0.780] | 0.828 | 0.899 [0.877, 0.921] | -0.017 [-0.029, -0.006] | -0.297 [-0.332, -0.265] |

## T3b. Test: işaretçili vs işaretçisiz görüntüler (box GT, görüntü-AP)

| family | variant | train_gt | seeds | AP_with_marker | n_with_marker | AP_no_marker | n_no_marker | AP_no_marker_gemini | n_nm_gem |
|---|---|---|---|---|---|---|---|---|---|
| DeepLabV3+ | main | box | 1 | 0.962 [0.954, 0.969] | 627 | 0.702 [0.650, 0.750] | 185 | 0.743 [0.677, 0.806] | 102 |
| DeepLabV3+ | main | glyph | 1 | 0.510 [0.496, 0.525] | 627 | 0.405 [0.368, 0.443] | 185 | 0.466 [0.418, 0.518] | 102 |
| DeepLabV3+ | main | pix | 1 | 0.637 [0.621, 0.653] | 627 | 0.542 [0.497, 0.589] | 185 | 0.633 [0.579, 0.696] | 102 |
| U-Net | 20ep | box | 1 | 0.973 [0.966, 0.979] | 627 | 0.713 [0.664, 0.763] | 185 | 0.738 [0.668, 0.801] | 102 |
| U-Net | 20ep | glyph | 1 | 0.427 [0.414, 0.439] | 627 | 0.348 [0.305, 0.392] | 185 | 0.393 [0.336, 0.451] | 102 |
| U-Net | 20ep | pix | 1 | 0.633 [0.616, 0.647] | 627 | 0.515 [0.466, 0.559] | 185 | 0.578 [0.517, 0.642] | 102 |
| U-Net | main | box | 3 | 0.972 [0.966, 0.977] | 627 | 0.690 [0.640, 0.739] | 185 | 0.722 [0.657, 0.788] | 102 |
| U-Net | main | glyph | 3 | 0.388 [0.377, 0.399] | 627 | 0.339 [0.307, 0.374] | 185 | 0.361 [0.317, 0.405] | 102 |
| U-Net | main | pix | 3 | 0.639 [0.624, 0.652] | 627 | 0.503 [0.455, 0.550] | 185 | 0.559 [0.504, 0.622] | 102 |
| U-Net | marker-repaired | box | 1 | 0.920 [0.904, 0.934] | 627 | 0.701 [0.649, 0.753] | 185 | 0.756 [0.690, 0.816] | 102 |
| U-Net | marker-repaired+nm-GT | glyph | 1 | 0.379 [0.367, 0.392] | 627 | 0.355 [0.320, 0.391] | 185 | 0.371 [0.327, 0.417] | 102 |
| U-Net | marker-repaired+nm-GT | pix | 1 | 0.526 [0.507, 0.544] | 627 | 0.517 [0.473, 0.560] | 185 | 0.579 [0.523, 0.634] | 102 |
| U-Net+box-aux | main | glyph | 1 | 0.468 [0.455, 0.480] | 627 | 0.383 [0.346, 0.422] | 185 | 0.423 [0.375, 0.474] | 102 |
| U-Net+box-aux | main | pix | 1 | 0.689 [0.676, 0.701] | 627 | 0.538 [0.488, 0.586] | 185 | 0.595 [0.532, 0.662] | 102 |

## T4. Held-out üretici (val+test; Ideogram vs kaynak-eşlenmiş Gemini; görüntü-AP)

| family | variant | train_gt | gemini_only | seeds | eval_gt | n_ideogram | AP_ideogram | AP_gemini_matched | gap |
|---|---|---|---|---|---|---|---|---|---|
| U-Net | geminionly | box | True | 3 | box | 126 | 0.412 [0.341, 0.481] | 0.909 [0.892, 0.927] | 0.497 |
| U-Net | geminionly | box | True | 3 | pix | 41 | 0.259 [0.168, 0.350] | 0.349 [0.322, 0.377] | 0.090 |
| U-Net | geminionly | glyph | True | 3 | box | 126 | 0.166 [0.134, 0.201] | 0.371 [0.359, 0.383] | 0.205 |
| U-Net | geminionly | glyph | True | 3 | pix | 41 | 0.182 [0.121, 0.246] | 0.388 [0.363, 0.413] | 0.205 |
| U-Net | geminionly | pix | True | 3 | box | 126 | 0.264 [0.213, 0.313] | 0.567 [0.550, 0.583] | 0.303 |
| U-Net | geminionly | pix | True | 3 | pix | 41 | 0.279 [0.182, 0.382] | 0.577 [0.546, 0.608] | 0.297 |
| U-Net | main | box | False | 3 | box | 126 | 0.629 [0.569, 0.692] | 0.912 [0.896, 0.928] | 0.283 |
| U-Net | main | box | False | 3 | pix | 41 | 0.374 [0.286, 0.458] | 0.359 [0.332, 0.386] | -0.016 |
| U-Net | main | glyph | False | 3 | box | 126 | 0.312 [0.269, 0.357] | 0.369 [0.356, 0.382] | 0.057 |
| U-Net | main | glyph | False | 3 | pix | 41 | 0.365 [0.276, 0.440] | 0.389 [0.364, 0.414] | 0.024 |
| U-Net | main | pix | False | 3 | box | 126 | 0.423 [0.367, 0.481] | 0.574 [0.557, 0.590] | 0.151 |
| U-Net | main | pix | False | 3 | pix | 41 | 0.454 [0.350, 0.550] | 0.571 [0.540, 0.603] | 0.117 |

## T5. YOLO ablasyonları ve maliyet

| run | epochs | best_epoch | train_min | params_M | val_tile_mAP50 |
|---|---|---|---|---|---|
| aiforge-tiles-yolo11s-img1024-conserv-s42 | 30 | 26 | 57.0 | 9.43 | 0.9812 |
| aiforge-tiles-yolo11s-img1024-conserv-s43 | 30 | 26 | 59.3 | 9.43 | 0.9876 |
| aiforge-tiles-yolo11s-img1024-conserv-s44 | 30 | 23 | 53.4 | 9.43 | 0.9882 |
| aiforge-full-yolo11s-img1280-conserv-s42 | 30 | 27 | 33.1 | 9.43 | 0.9601 |
| aiforge-tiles-yolo11s-img1024-default-s42 | 30 | 28 | 43.4 | 9.43 | 0.9859 |
| aiforge-tiles_authneg-yolo11s-img1024-conserv-s42 | 30 | 22 | 53.5 | 9.43 | 0.9875 |
| aiforge-tiles-yolo11n-img1024-conserv-s42 | 30 | 28 | 31.6 | 2.6 | 0.9599 |

## T6. Dayanıklılık (relΔ pooled AUPRC, kendi GT'sinde)

| family | variant | train_gt | blur1 | clahe | gray | jpeg75 | rescale075 |
|---|---|---|---|---|---|---|---|
| DeepLabV3+ | main | box | -13.6% | -1.3% | -46.7% | -11.9% | -7.1% |
| DeepLabV3+ | main | glyph | -34.2% | -0.2% | -60.9% | -16.8% | -22.7% |
| DeepLabV3+ | main | pix | -53.6% | +0.8% | -58.5% | -20.2% | -38.7% |
| U-Net | main | box | -20.7% ± 6.7 | — | — | -16.6% ± 1.4 | -12.2% ± 4.3 |
| U-Net | main | glyph | -59.0% ± 6.8 | — | — | -18.0% ± 0.8 | -42.7% ± 5.4 |
| U-Net | main | pix | -59.1% ± 4.5 | — | — | -22.5% ± 2.3 | -40.3% ± 1.5 |
| U-Net | marker-repaired | box | -27.8% | -3.0% | -61.5% | -31.6% | -11.6% |
| U-Net | marker-repaired+nm-GT | glyph | -44.4% | -2.8% | -66.7% | -32.5% | -28.9% |
| U-Net | marker-repaired+nm-GT | pix | -75.7% | -0.7% | -53.8% | -50.3% | -51.5% |
| U-Net+box-aux | main | glyph | -31.4% | -1.2% | -62.3% | -18.3% | -20.1% |
| U-Net+box-aux | main | pix | -42.7% | -0.6% | -63.5% | -19.8% | -31.6% |

## T7. Kontroller (val; image-level AUROC forged vs kontrol; bayrak oranı @0.5 (>0.1 % piksel); FP@τ*)

| family | variant | train_gt | source | seeds | auroc_top1 | auroc_top1_sd | auroc_max | flag05 | fp_tau |
|---|---|---|---|---|---|---|---|---|---|
| DeepLabV3+ | main | box | authentic | 1 | 0.8405 | nan | 0.9125 | 0.4579 | 0.0023 |
| DeepLabV3+ | main | box | benign_pasteback | 1 | 0.8156 | nan | 0.8956 | 0.5195 | 0.003 |
| DeepLabV3+ | main | box | fullimage_resample | 1 | 0.8199 | nan | 0.8952 | 0.5236 | 0.0028 |
| DeepLabV3+ | main | glyph | authentic | 1 | 0.7241 | nan | 0.8253 | 0.4066 | 0.0097 |
| DeepLabV3+ | main | glyph | benign_pasteback | 1 | 0.7017 | nan | 0.7972 | 0.4435 | 0.0101 |
| DeepLabV3+ | main | glyph | fullimage_resample | 1 | 0.6956 | nan | 0.8061 | 0.4538 | 0.0104 |
| DeepLabV3+ | main | pix | authentic | 1 | 0.8273 | nan | 0.9047 | 0.2423 | 0.0062 |
| DeepLabV3+ | main | pix | benign_pasteback | 1 | 0.8068 | nan | 0.8884 | 0.2875 | 0.0066 |
| DeepLabV3+ | main | pix | fullimage_resample | 1 | 0.782 | nan | 0.8863 | 0.3326 | 0.0074 |
| U-Net | main | box | authentic | 3 | 0.8917 | 0.0465 | 0.9224 | 0.2601 | 0.0016 |
| U-Net | main | box | benign_pasteback | 3 | 0.872 | 0.0467 | 0.8993 | 0.3128 | 0.0021 |
| U-Net | main | box | fullimage_resample | 3 | 0.8664 | 0.0526 | 0.9 | 0.3381 | 0.0022 |
| U-Net | main | glyph | authentic | 3 | 0.7991 | 0.0138 | 0.7496 | 0.2286 | 0.0038 |
| U-Net | main | glyph | benign_pasteback | 3 | 0.7787 | 0.0129 | 0.7242 | 0.2704 | 0.004 |
| U-Net | main | glyph | fullimage_resample | 3 | 0.7692 | 0.0196 | 0.7132 | 0.2704 | 0.0042 |
| U-Net | main | pix | authentic | 3 | 0.8916 | 0.0057 | 0.9176 | 0.1157 | 0.002 |
| U-Net | main | pix | benign_pasteback | 3 | 0.8758 | 0.0097 | 0.9031 | 0.1465 | 0.0022 |
| U-Net | main | pix | fullimage_resample | 3 | 0.8634 | 0.0121 | 0.8868 | 0.1663 | 0.0025 |
| U-Net | marker-repaired | box | authentic | 1 | 0.8901 | nan | 0.8864 | 0.2608 | 0.0024 |
| U-Net | marker-repaired | box | benign_pasteback | 1 | 0.8256 | nan | 0.826 | 0.4025 | 0.0041 |
| U-Net | marker-repaired | box | fullimage_resample | 1 | 0.8553 | nan | 0.8617 | 0.3634 | 0.0032 |
| U-Net | marker-repaired+nm-GT | glyph | authentic | 1 | 0.7834 | nan | 0.7301 | 0.271 | 0.0044 |
| U-Net | marker-repaired+nm-GT | glyph | benign_pasteback | 1 | 0.7329 | nan | 0.6666 | 0.3676 | 0.005 |
| U-Net | marker-repaired+nm-GT | glyph | fullimage_resample | 1 | 0.7409 | nan | 0.6767 | 0.3306 | 0.0049 |
| U-Net | marker-repaired+nm-GT | pix | authentic | 1 | 0.8437 | nan | 0.8365 | 0.1848 | 0.0024 |
| U-Net | marker-repaired+nm-GT | pix | benign_pasteback | 1 | 0.779 | nan | 0.7772 | 0.3018 | 0.0032 |
| U-Net | marker-repaired+nm-GT | pix | fullimage_resample | 1 | 0.783 | nan | 0.7848 | 0.2936 | 0.0033 |
| U-Net+box-aux | main | glyph | authentic | 1 | 0.8013 | nan | 0.7338 | 0.2752 | 0.0043 |
| U-Net+box-aux | main | glyph | benign_pasteback | 1 | 0.7828 | nan | 0.7109 | 0.2998 | 0.0045 |
| U-Net+box-aux | main | glyph | fullimage_resample | 1 | 0.7697 | nan | 0.6997 | 0.3203 | 0.0047 |
| U-Net+box-aux | main | pix | authentic | 1 | 0.8973 | nan | 0.921 | 0.1211 | 0.0022 |
| U-Net+box-aux | main | pix | benign_pasteback | 1 | 0.8785 | nan | 0.9031 | 0.1622 | 0.0025 |
| U-Net+box-aux | main | pix | fullimage_resample | 1 | 0.867 | nan | 0.8973 | 0.1828 | 0.0028 |

## T8. Faz 1 AUPRC düzeltmesi (W6)

| variant | reported_phase1_auprc | reproduced_subsampled_auprc | exact_auprc_all_pixels | exact_auroc | pos_rate | n_crops | n_pixels |
|---|---|---|---|---|---|---|---|
| rgb | 0.947 | 0.9423 | 0.8053 | 0.9344 | 0.03306 | 200 | 52428800 |
| rgbfreq | 0.939 | 0.9331 | 0.7922 | 0.9255 | 0.03306 | 200 | 52428800 |

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
| U-Net | marker-repaired | box | 3 | 0.912 ± 0.006 | 0.797 ± 0.006 | 0.265 ± 0.001 | 0.377 ± 0.004 | 0.373 ± 0.013 | 0.427 ± 0.006 |
| U-Net | marker-repaired+nm-GT | glyph | 1 | 0.348 | 0.401 | 0.661 | 0.662 | 0.371 | 0.456 |
| U-Net | marker-repaired+nm-GT | pix | 1 | 0.513 | 0.524 | 0.455 | 0.479 | 0.544 | 0.502 |
| U-Net+box-aux | main | glyph | 1 | 0.370 | 0.441 | 0.636 | 0.638 | 0.391 | 0.466 |
| U-Net+box-aux | main | pix | 3 | 0.668 ± 0.019 | 0.629 ± 0.013 | 0.523 ± 0.009 | 0.551 ± 0.006 | 0.633 ± 0.013 | 0.588 ± 0.007 |
| YOLO11m | main | box | 1 | 0.960 | 0.942 | 0.224 | 0.418 | 0.356 | 0.493 |
| YOLO11n | main | box | 1 | 0.944 | 0.907 | 0.219 | 0.404 | 0.366 | 0.468 |
| YOLO11s | authneg | box | 1 | 0.967 | 0.948 | 0.223 | 0.421 | 0.361 | 0.495 |
| YOLO11s | defaug | box | 1 | 0.967 | 0.945 | 0.221 | 0.421 | 0.373 | 0.495 |
| YOLO11s | full1280 | box | 1 | 0.922 | 0.902 | 0.203 | 0.395 | 0.334 | 0.463 |
| YOLO11s | main | box | 3 | 0.965 ± 0.003 | 0.947 ± 0.005 | 0.221 ± 0.003 | 0.420 ± 0.002 | 0.368 ± 0.015 | 0.494 ± 0.003 |

## T2. Resmî test (box GT; piksel GT 89 çift) + tespit AP

| family | variant | train_gt | seeds | AUPRC_box_812 | AUPRC_box_740 | F1tau_box | IoUtau_box | AUPRC_pix_89 | AP50 | AP50_95 |
|---|---|---|---|---|---|---|---|---|---|---|
| DeepLabV3+ | main | box | 1 | 0.952 | 0.952 | 0.833 | 0.765 | 0.408 | 0.833 | 0.577 |
| DeepLabV3+ | main | glyph | 1 | 0.368 | 0.365 | 0.494 | 0.353 | 0.389 | — | — |
| DeepLabV3+ | main | pix | 1 | 0.585 | 0.583 | 0.590 | 0.457 | 0.667 | — | — |
| U-Net | 20ep | box | 1 | 0.945 | 0.946 | 0.858 | 0.799 | 0.422 | — | — |
| U-Net | 20ep | glyph | 1 | 0.480 | 0.475 | 0.403 | 0.265 | 0.477 | — | — |
| U-Net | 20ep | pix | 1 | 0.609 | 0.606 | 0.602 | 0.466 | 0.638 | — | — |
| U-Net | main | box | 3 | 0.950 ± 0.011 | 0.949 ± 0.012 | 0.846 ± 0.015 | 0.789 ± 0.018 | 0.427 ± 0.002 | 0.712 ± 0.279 | 0.453 ± 0.224 |
| U-Net | main | glyph | 3 | 0.403 ± 0.011 | 0.403 ± 0.010 | 0.405 ± 0.007 | 0.266 ± 0.006 | 0.410 ± 0.017 | 0.011 | 0.002 |
| U-Net | main | pix | 3 | 0.638 ± 0.015 | 0.640 ± 0.014 | 0.616 ± 0.013 | 0.480 ± 0.015 | 0.654 ± 0.016 | 0.107 ± 0.027 | 0.053 ± 0.014 |
| U-Net | marker-repaired | box | 3 | 0.940 ± 0.003 | 0.940 ± 0.002 | 0.799 ± 0.016 | 0.727 ± 0.015 | 0.395 ± 0.027 | 0.804 ± 0.005 | 0.486 ± 0.007 |
| U-Net | marker-repaired+nm-GT | glyph | 1 | 0.402 | 0.404 | 0.405 | 0.266 | 0.354 | — | — |
| U-Net | marker-repaired+nm-GT | pix | 1 | 0.564 | 0.562 | 0.537 | 0.405 | 0.570 | — | — |
| U-Net+box-aux | main | glyph | 1 | 0.432 | 0.433 | 0.448 | 0.304 | 0.460 | — | — |
| U-Net+box-aux | main | pix | 3 | 0.702 ± 0.020 | 0.702 ± 0.021 | 0.641 ± 0.012 | 0.506 ± 0.015 | 0.652 ± 0.003 | 0.125 ± 0.020 | 0.064 ± 0.010 |
| YOLO11m | main | box | 1 | 0.969 | 0.969 | 0.947 | 0.920 | 0.418 | 0.897 | 0.800 |
| YOLO11n | main | box | 1 | 0.963 | 0.964 | 0.926 | 0.891 | 0.413 | 0.871 | 0.738 |
| YOLO11s | authneg | box | 1 | 0.971 | 0.971 | 0.943 | 0.913 | 0.425 | 0.874 | 0.774 |
| YOLO11s | defaug | box | 1 | 0.971 | 0.972 | 0.951 | 0.922 | 0.397 | 0.864 | 0.765 |
| YOLO11s | full1280 | box | 1 | 0.932 | 0.932 | 0.909 | 0.872 | 0.430 | 0.960 | 0.881 |
| YOLO11s | main | box | 3 | 0.969 ± 0.006 | 0.969 ± 0.007 | 0.946 ± 0.007 | 0.918 ± 0.006 | 0.404 ± 0.042 | 0.880 ± 0.002 | 0.780 ± 0.003 |

## T3. Sondalar (val çiftleri; işaretçili alt küme için AP; seed-ortalaması + grup bootstrap CI)

| model | seeds | n_res | APbox_forged | cover_forged | APbox_forged_repaired | cover_forged_repaired | APbox_authentic_plus_marker | cover_authentic_plus_marker | APbox_authentic | cover_authentic | dAP_repair | AUROC_forged | AUROC_forged_repaired | AUROC_authentic_plus_marker | AUROC_render_only | AUROC_content_only | inbox_render_only | APbox_all_render_only | inbox_content_only | APbox_all_content_only | inbox_forged | APbox_all_forged | dAPbox_clahe | dAPbox_gray |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DeepLabV3+|main|bbox | 1 | 380 | 0.982 [0.974, 0.988] | 0.920 | 0.807 [0.774, 0.837] | 0.641 | 0.928 [0.913, 0.941] | 0.775 | 0.184 [0.162, 0.210] | 0.040 | -0.174 [-0.206, -0.145] | 0.972 | 0.877 | 0.899 | 0.644 | 0.755 | 0.281 | 0.452 [0.415, 0.488] | 0.416 | 0.606 [0.572, 0.638] | 0.869 | 0.934 [0.918, 0.949] | -0.006 [-0.014, +0.001] | -0.343 [-0.376, -0.308] |
| DeepLabV3+|main|glyph_v2 | 1 | 380 | 0.627 [0.611, 0.643] | 0.477 | 0.503 [0.477, 0.528] | 0.352 | 0.569 [0.548, 0.587] | 0.424 | 0.147 [0.130, 0.169] | 0.051 | -0.124 [-0.149, -0.101] | 0.908 | 0.769 | 0.827 | 0.581 | 0.684 | 0.090 | 0.279 [0.256, 0.302] | 0.174 | 0.392 [0.367, 0.417] | 0.350 | 0.595 [0.576, 0.613] | +0.029 [+0.023, +0.035] | -0.191 [-0.220, -0.162] |
| DeepLabV3+|main|pixel_eps8_morph | 1 | 380 | 0.730 [0.711, 0.749] | 0.597 | 0.555 [0.523, 0.585] | 0.400 | 0.628 [0.604, 0.651] | 0.484 | 0.172 [0.151, 0.195] | 0.072 | -0.175 [-0.208, -0.146] | 0.934 | 0.763 | 0.828 | 0.548 | 0.677 | 0.071 | 0.277 [0.251, 0.303] | 0.189 | 0.437 [0.407, 0.466] | 0.420 | 0.700 [0.680, 0.720] | +0.026 [+0.018, +0.033] | -0.258 [-0.286, -0.228] |
| U-Net+box-aux|main|glyph_v2 | 1 | 380 | 0.622 [0.605, 0.640] | 0.381 | 0.470 [0.443, 0.496] | 0.245 | 0.546 [0.525, 0.567] | 0.338 | 0.111 [0.096, 0.127] | 0.030 | -0.153 [-0.181, -0.126] | 0.880 | 0.753 | 0.811 | 0.577 | 0.666 | 0.062 | 0.237 [0.215, 0.258] | 0.114 | 0.333 [0.306, 0.360] | 0.270 | 0.595 [0.576, 0.614] | +0.017 [+0.010, +0.023] | -0.278 [-0.308, -0.248] |
| U-Net+box-aux|main|pixel_eps8_morph | 3 | 380 | 0.795 [0.782, 0.807] | 0.558 | 0.600 [0.566, 0.631] | 0.343 | 0.700 [0.680, 0.719] | 0.457 | 0.148 [0.128, 0.171] | 0.033 | -0.194 [-0.226, -0.164] | 0.961 | 0.805 | 0.887 | 0.593 | 0.716 | 0.061 | 0.294 [0.266, 0.320] | 0.163 | 0.441 [0.409, 0.470] | 0.413 | 0.757 [0.739, 0.775] | +0.035 [+0.029, +0.041] | -0.338 [-0.367, -0.309] |
| U-Net|main|bbox | 3 | 380 | 0.988 [0.982, 0.992] | 0.939 | 0.766 [0.731, 0.797] | 0.595 | 0.905 [0.886, 0.922] | 0.795 | 0.176 [0.155, 0.201] | 0.041 | -0.222 [-0.256, -0.190] | 0.971 | 0.837 | 0.903 | 0.637 | 0.733 | 0.237 | 0.415 [0.381, 0.447] | 0.348 | 0.557 [0.521, 0.589] | 0.864 | 0.933 [0.915, 0.950] | +0.000 [-0.004, +0.005] | -0.388 [-0.421, -0.355] |
| U-Net|main|glyph_v2 | 3 | 380 | 0.522 [0.505, 0.541] | 0.325 | 0.398 [0.372, 0.425] | 0.195 | 0.461 [0.442, 0.481] | 0.283 | 0.123 [0.108, 0.141] | 0.023 | -0.123 [-0.146, -0.103] | 0.902 | 0.763 | 0.834 | 0.579 | 0.683 | 0.050 | 0.212 [0.192, 0.231] | 0.096 | 0.320 [0.295, 0.344] | 0.239 | 0.507 [0.489, 0.527] | +0.036 [+0.031, +0.041] | -0.213 [-0.235, -0.191] |
| U-Net|main|pixel_eps8_morph | 3 | 380 | 0.747 [0.731, 0.762] | 0.538 | 0.500 [0.468, 0.529] | 0.282 | 0.635 [0.611, 0.656] | 0.420 | 0.134 [0.117, 0.155] | 0.031 | -0.247 [-0.279, -0.218] | 0.942 | 0.762 | 0.858 | 0.554 | 0.681 | 0.045 | 0.211 [0.189, 0.233] | 0.150 | 0.367 [0.338, 0.396] | 0.406 | 0.706 [0.685, 0.725] | +0.026 [+0.020, +0.032] | -0.370 [-0.398, -0.342] |
| U-Net|marker-repaired+nm-GT|glyph_v2 | 1 | 380 | 0.510 [0.491, 0.530] | 0.321 | 0.510 [0.491, 0.531] | 0.314 | 0.345 [0.321, 0.369] | 0.186 | 0.119 [0.103, 0.138] | 0.029 | +0.001 [-0.008, +0.009] | 0.876 | 0.875 | 0.688 | 0.660 | 0.822 | 0.081 | 0.265 [0.244, 0.286] | 0.185 | 0.422 [0.399, 0.445] | 0.237 | 0.501 [0.482, 0.522] | +0.036 [+0.029, +0.043] | -0.185 [-0.209, -0.161] |
| U-Net|marker-repaired+nm-GT|pixel_eps8_morph | 1 | 380 | 0.648 [0.625, 0.670] | 0.444 | 0.686 [0.667, 0.705] | 0.503 | 0.256 [0.235, 0.281] | 0.108 | 0.131 [0.113, 0.152] | 0.033 | +0.038 [+0.021, +0.057] | 0.869 | 0.899 | 0.592 | 0.638 | 0.834 | 0.096 | 0.315 [0.288, 0.340] | 0.289 | 0.530 [0.506, 0.555] | 0.338 | 0.627 [0.606, 0.650] | +0.012 [+0.003, +0.021] | -0.225 [-0.248, -0.202] |
| U-Net|marker-repaired|bbox | 3 | 380 | 0.939 [0.921, 0.956] | 0.891 | 0.956 [0.942, 0.969] | 0.895 | 0.617 [0.579, 0.655] | 0.442 | 0.179 [0.156, 0.203] | 0.047 | +0.017 [+0.001, +0.033] | 0.957 | 0.963 | 0.720 | 0.724 | 0.886 | 0.345 | 0.548 [0.513, 0.582] | 0.583 | 0.761 [0.734, 0.788] | 0.823 | 0.902 [0.882, 0.922] | -0.013 [-0.021, -0.005] | -0.295 [-0.326, -0.265] |
| YOLO11s|main|bbox | 1 | 380 | 0.956 [0.951, 0.960] | 0.958 | 0.894 [0.881, 0.905] | 0.886 | 0.815 [0.791, 0.834] | 0.607 | 0.101 [0.085, 0.118] | 0.003 | -0.062 [-0.075, -0.051] | 0.996 | 0.988 | 0.848 | 0.865 | 0.858 | 0.446 | 0.696 [0.662, 0.728] | 0.383 | 0.678 [0.649, 0.704] | 0.864 | 0.947 [0.940, 0.954] | -0.025 [-0.036, -0.015] | -0.347 [-0.379, -0.313] |

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
| U-Net | marker-repaired | box | 3 | 0.914 [0.899, 0.927] | 627 | 0.711 [0.663, 0.758] | 185 | 0.762 [0.703, 0.821] | 102 |
| U-Net | marker-repaired+nm-GT | glyph | 1 | 0.379 [0.367, 0.392] | 627 | 0.355 [0.320, 0.391] | 185 | 0.371 [0.327, 0.417] | 102 |
| U-Net | marker-repaired+nm-GT | pix | 1 | 0.526 [0.507, 0.544] | 627 | 0.517 [0.473, 0.560] | 185 | 0.579 [0.523, 0.634] | 102 |
| U-Net+box-aux | main | glyph | 1 | 0.468 [0.455, 0.480] | 627 | 0.383 [0.346, 0.422] | 185 | 0.423 [0.375, 0.474] | 102 |
| U-Net+box-aux | main | pix | 3 | 0.687 [0.674, 0.699] | 627 | 0.544 [0.496, 0.593] | 185 | 0.600 [0.541, 0.662] | 102 |
| YOLO11m | main | box | 1 | 0.954 [0.951, 0.957] | 627 | 0.882 [0.860, 0.899] | 185 | 0.894 [0.869, 0.914] | 102 |
| YOLO11n | main | box | 1 | 0.946 [0.940, 0.950] | 627 | 0.812 [0.780, 0.841] | 185 | 0.794 [0.749, 0.839] | 102 |
| YOLO11s | authneg | box | 1 | 0.951 [0.948, 0.954] | 627 | 0.877 [0.851, 0.899] | 185 | 0.889 [0.863, 0.911] | 102 |
| YOLO11s | defaug | box | 1 | 0.953 [0.950, 0.957] | 627 | 0.873 [0.846, 0.893] | 185 | 0.863 [0.830, 0.894] | 102 |
| YOLO11s | full1280 | box | 1 | 0.930 [0.923, 0.937] | 627 | 0.756 [0.712, 0.795] | 185 | 0.850 [0.814, 0.883] | 102 |
| YOLO11s | main | box | 3 | 0.954 [0.951, 0.957] | 627 | 0.878 [0.857, 0.895] | 185 | 0.886 [0.861, 0.907] | 102 |

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
| U-Net | marker-repaired | box | False | 3 | box | 126 | 0.633 [0.573, 0.695] | 0.898 [0.881, 0.915] | 0.265 |
| U-Net | marker-repaired | box | False | 3 | pix | 41 | 0.385 [0.300, 0.470] | 0.377 [0.350, 0.405] | -0.008 |
| U-Net | marker-repaired+nm-GT | glyph | False | 1 | box | 126 | 0.338 [0.293, 0.383] | 0.361 [0.348, 0.376] | 0.023 |
| U-Net | marker-repaired+nm-GT | glyph | False | 1 | pix | 41 | 0.392 [0.301, 0.479] | 0.389 [0.363, 0.417] | -0.003 |
| U-Net | marker-repaired+nm-GT | pix | False | 1 | box | 126 | 0.430 [0.377, 0.484] | 0.492 [0.473, 0.510] | 0.062 |
| U-Net | marker-repaired+nm-GT | pix | False | 1 | pix | 41 | 0.457 [0.360, 0.552] | 0.505 [0.474, 0.539] | 0.048 |
| YOLO11s | geminionly | box | True | 1 | box | 126 | 0.752 [0.703, 0.806] | 0.938 [0.929, 0.946] | 0.186 |
| YOLO11s | geminionly | box | True | 1 | pix | 41 | 0.422 [0.357, 0.494] | 0.330 [0.308, 0.355] | -0.092 |
| YOLO11s | main | box | False | 3 | box | 126 | 0.872 [0.846, 0.895] | 0.945 [0.938, 0.951] | 0.073 |
| YOLO11s | main | box | False | 3 | pix | 41 | 0.443 [0.382, 0.507] | 0.336 [0.313, 0.361] | -0.107 |

## T5. YOLO ablasyonları ve maliyet

| run | epochs | best_epoch | train_min | params_M | val_tile_mAP50 | AP50_val | AP5095_val | AP50_val_lt1pct | AP50_test | AP5095_test | AP50_test_lt1pct | val_AUPRC_box | val_AUPRC_pix | ms_per_img |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aiforge-tiles-yolo11s-img1024-conserv-s42 | 30 | 26 | 57.0 | 9.43 | 0.9812 | 0.879 | 0.781 | 0.854 | 0.882 | 0.782 | 0.872 | 0.967 | 0.384 | 98 |
| aiforge-tiles-yolo11s-img1024-conserv-s43 | 30 | 26 | 59.3 | 9.43 | 0.9876 | 0.873 | 0.781 | 0.845 | 0.878 | 0.781 | 0.857 | 0.961 | 0.365 | — |
| aiforge-tiles-yolo11s-img1024-conserv-s44 | 30 | 23 | 53.4 | 9.43 | 0.9882 | 0.868 | 0.769 | 0.825 | 0.879 | 0.777 | 0.862 | 0.966 | 0.355 | — |
| aiforge-full-yolo11s-img1280-conserv-s42 | 30 | 27 | 33.1 | 9.43 | 0.9601 | 0.957 | 0.879 | 0.915 | 0.960 | 0.881 | 0.933 | 0.922 | 0.334 | 303 |
| aiforge-tiles-yolo11s-img1024-default-s42 | 30 | 28 | 43.4 | 9.43 | 0.9859 | 0.867 | 0.758 | 0.838 | 0.864 | 0.765 | 0.849 | 0.967 | 0.373 | — |
| aiforge-tiles_authneg-yolo11s-img1024-conserv-s42 | 30 | 22 | 53.5 | 9.43 | 0.9875 | 0.889 | 0.777 | 0.848 | 0.874 | 0.774 | 0.863 | 0.967 | 0.361 | — |
| aiforge-tiles-yolo11n-img1024-conserv-s42 | 30 | 28 | 31.6 | 2.6 | 0.9599 | 0.849 | 0.723 | 0.776 | 0.871 | 0.738 | 0.846 | 0.944 | 0.366 | 81 |
| aiforge-tiles-yolo11m-img1024-conserv-s42 | 30 | 23 | 87.1 | 20.05 | 0.982 | 0.897 | 0.800 | 0.868 | 0.897 | 0.800 | 0.882 | 0.960 | 0.356 | 76 |
| aiforge-tiles_geminionly-yolo11s-img1024-conserv-s42 | 30 | 30 | 58.9 | 9.43 | 0.9909 | 0.865 | 0.765 | 0.807 | 0.855 | 0.760 | 0.813 | 0.957 | 0.344 | — |

## T5b. Hız (GPU boş, 100 test görüntüsü, ms/görüntü)

| model | family | ms_per_img | params_M | mean_MP | gpu |
|---|---|---|---|---|---|
| unet_bbox_s42 | seg | 80.91 | 14.33 | 3.14 | NVIDIA GeForce RTX 5080 |
| dlv3p_bbox_s42 | seg | 122.02 | 12.33 | 3.14 | NVIDIA GeForce RTX 5080 |
| unetjoint_pixel_eps8_morph_s42 | seg | 82.87 | 14.33 | 3.14 | NVIDIA GeForce RTX 5080 |
| aiforge-tiles-yolo11s-img1024-conserv-s42 | yolo | 98.1 | 9.43 | 3.14 | NVIDIA GeForce RTX 5080 |
| aiforge-tiles-yolo11n-img1024-conserv-s42 | yolo | 80.53 | 2.6 | 3.14 | NVIDIA GeForce RTX 5080 |
| aiforge-tiles-yolo11m-img1024-conserv-s42 | yolo | 76.3 | 20.05 | 3.14 | NVIDIA GeForce RTX 5080 |
| aiforge-full-yolo11s-img1280-conserv-s42 | yolo | 303.01 | 9.43 | 3.14 | NVIDIA GeForce RTX 5080 |

## T5c. Tespit metriği (val, tüm aileler, alan/üretici kırılımı)

| model | family | group | n | AP50 | AP50_95 | Recall50_bestF1 | mean_boxes_per_img |
|---|---|---|---|---|---|---|---|
| aiforge-tiles-yolo11s-img1024-conserv-s42 | yolo | all | 487 | 0.8787 | 0.781 | 0.9671 | 5.82 |
| aiforge-tiles-yolo11s-img1024-conserv-s42 | yolo | area_<1% | 231 | 0.854 | 0.7247 | 0.9481 | 10.06 |
| aiforge-tiles-yolo11s-img1024-conserv-s42 | yolo | area_1-5% | 254 | 0.9287 | 0.8445 | 0.9921 | 2.0 |
| aiforge-tiles-yolo11s-img1024-conserv-s42 | yolo | area_5-20% | 2 | 1.0 | 1.0 | 1.0 | 1.0 |
| aiforge-tiles-yolo11s-img1024-conserv-s42 | yolo | tool_gemini | 446 | 0.8869 | 0.7947 | 0.9731 | 4.51 |
| aiforge-tiles-yolo11s-img1024-conserv-s42 | yolo | tool_ideogram | 41 | 0.8465 | 0.6303 | 0.9268 | 20.0 |
| aiforge-tiles-yolo11s-img1024-conserv-s43 | yolo | all | 487 | 0.873 | 0.7808 | 0.9692 | 5.21 |
| aiforge-tiles-yolo11s-img1024-conserv-s43 | yolo | area_<1% | 231 | 0.8447 | 0.7188 | 0.9654 | 8.57 |
| aiforge-tiles-yolo11s-img1024-conserv-s43 | yolo | area_1-5% | 254 | 0.9125 | 0.8453 | 0.9803 | 2.19 |
| aiforge-tiles-yolo11s-img1024-conserv-s43 | yolo | area_5-20% | 2 | 1.0 | 0.9505 | 1.0 | 1.0 |
| aiforge-tiles-yolo11s-img1024-conserv-s43 | yolo | tool_gemini | 446 | 0.8851 | 0.7996 | 0.9709 | 4.11 |
| aiforge-tiles-yolo11s-img1024-conserv-s43 | yolo | tool_ideogram | 41 | 0.8251 | 0.6003 | 0.9756 | 17.24 |
| aiforge-tiles-yolo11s-img1024-conserv-s44 | yolo | all | 487 | 0.8675 | 0.769 | 0.9774 | 5.24 |
| aiforge-tiles-yolo11s-img1024-conserv-s44 | yolo | area_<1% | 231 | 0.8251 | 0.6897 | 0.9654 | 8.86 |
| aiforge-tiles-yolo11s-img1024-conserv-s44 | yolo | area_1-5% | 254 | 0.9229 | 0.8538 | 0.9882 | 1.98 |
| aiforge-tiles-yolo11s-img1024-conserv-s44 | yolo | area_5-20% | 2 | 1.0 | 1.0 | 1.0 | 1.0 |
| aiforge-tiles-yolo11s-img1024-conserv-s44 | yolo | tool_gemini | 446 | 0.8808 | 0.789 | 0.9776 | 4.22 |
| aiforge-tiles-yolo11s-img1024-conserv-s44 | yolo | tool_ideogram | 41 | 0.8004 | 0.5678 | 0.9512 | 16.34 |
| aiforge-tiles-yolo11s-img1024-default-s42 | yolo | all | 487 | 0.8671 | 0.758 | 0.9671 | 5.88 |
| aiforge-tiles-yolo11s-img1024-default-s42 | yolo | area_<1% | 231 | 0.8377 | 0.6896 | 0.9437 | 9.45 |
| aiforge-tiles-yolo11s-img1024-default-s42 | yolo | area_1-5% | 254 | 0.9138 | 0.8322 | 0.9843 | 2.67 |
| aiforge-tiles-yolo11s-img1024-default-s42 | yolo | area_5-20% | 2 | 1.0 | 1.0 | 1.0 | 1.0 |
| aiforge-tiles-yolo11s-img1024-default-s42 | yolo | tool_gemini | 446 | 0.8771 | 0.7801 | 0.9709 | 4.7 |
| aiforge-tiles-yolo11s-img1024-default-s42 | yolo | tool_ideogram | 41 | 0.8224 | 0.5386 | 0.9268 | 18.71 |
| aiforge-tiles_authneg-yolo11s-img1024-conserv-s42 | yolo | all | 487 | 0.8891 | 0.7768 | 0.9733 | 2.81 |
| aiforge-tiles_authneg-yolo11s-img1024-conserv-s42 | yolo | area_<1% | 231 | 0.8484 | 0.7065 | 0.9524 | 3.92 |
| aiforge-tiles_authneg-yolo11s-img1024-conserv-s42 | yolo | area_1-5% | 254 | 0.9316 | 0.8499 | 0.9843 | 1.81 |
| aiforge-tiles_authneg-yolo11s-img1024-conserv-s42 | yolo | area_5-20% | 2 | 1.0 | 1.0 | 1.0 | 1.0 |
| aiforge-tiles_authneg-yolo11s-img1024-conserv-s42 | yolo | tool_gemini | 446 | 0.8952 | 0.7955 | 0.9641 | 2.44 |
| aiforge-tiles_authneg-yolo11s-img1024-conserv-s42 | yolo | tool_ideogram | 41 | 0.824 | 0.5756 | 0.9268 | 6.8 |
| aiforge-tiles-yolo11n-img1024-conserv-s42 | yolo | all | 487 | 0.8486 | 0.7226 | 0.9035 | 7.26 |
| aiforge-tiles-yolo11n-img1024-conserv-s42 | yolo | area_<1% | 231 | 0.7764 | 0.6288 | 0.8571 | 12.13 |
| aiforge-tiles-yolo11n-img1024-conserv-s42 | yolo | area_1-5% | 254 | 0.9217 | 0.8152 | 0.9528 | 2.89 |
| aiforge-tiles-yolo11n-img1024-conserv-s42 | yolo | area_5-20% | 2 | 1.0 | 0.901 | 1.0 | 1.5 |
| aiforge-tiles-yolo11n-img1024-conserv-s42 | yolo | tool_gemini | 446 | 0.8596 | 0.7462 | 0.917 | 5.57 |
| aiforge-tiles-yolo11n-img1024-conserv-s42 | yolo | tool_ideogram | 41 | 0.7202 | 0.4591 | 0.8537 | 25.68 |
| aiforge-tiles-yolo11m-img1024-conserv-s42 | yolo | all | 487 | 0.8973 | 0.7996 | 0.9713 | 5.58 |
| aiforge-tiles-yolo11m-img1024-conserv-s42 | yolo | area_<1% | 231 | 0.8678 | 0.7483 | 0.9567 | 8.94 |
| aiforge-tiles-yolo11m-img1024-conserv-s42 | yolo | area_1-5% | 254 | 0.9259 | 0.8468 | 0.9843 | 2.55 |
| aiforge-tiles-yolo11m-img1024-conserv-s42 | yolo | area_5-20% | 2 | 1.0 | 1.0 | 1.0 | 1.0 |
| aiforge-tiles-yolo11m-img1024-conserv-s42 | yolo | tool_gemini | 446 | 0.9058 | 0.8204 | 0.9641 | 4.47 |
| aiforge-tiles-yolo11m-img1024-conserv-s42 | yolo | tool_ideogram | 41 | 0.8081 | 0.5671 | 0.9268 | 17.61 |
| aiforge-tiles_geminionly-yolo11s-img1024-conserv-s42 | yolo | all | 487 | 0.8653 | 0.7655 | 0.9507 | 3.28 |
| aiforge-tiles_geminionly-yolo11s-img1024-conserv-s42 | yolo | area_<1% | 231 | 0.8075 | 0.6718 | 0.9091 | 4.56 |
| aiforge-tiles_geminionly-yolo11s-img1024-conserv-s42 | yolo | area_1-5% | 254 | 0.9393 | 0.8668 | 0.9764 | 2.14 |
| aiforge-tiles_geminionly-yolo11s-img1024-conserv-s42 | yolo | area_5-20% | 2 | 1.0 | 0.9505 | 1.0 | 1.0 |
| aiforge-tiles_geminionly-yolo11s-img1024-conserv-s42 | yolo | tool_gemini | 446 | 0.8792 | 0.788 | 0.9574 | 2.89 |
| aiforge-tiles_geminionly-yolo11s-img1024-conserv-s42 | yolo | tool_ideogram | 41 | 0.7298 | 0.4924 | 0.7561 | 7.56 |
| aiforge-full-yolo11s-img1280-conserv-s42 | yolo | all | 487 | 0.9574 | 0.8795 | 0.924 | 5.85 |
| aiforge-full-yolo11s-img1280-conserv-s42 | yolo | area_<1% | 231 | 0.9148 | 0.7887 | 0.8615 | 10.72 |
| aiforge-full-yolo11s-img1280-conserv-s42 | yolo | area_1-5% | 254 | 0.9896 | 0.9547 | 0.9843 | 1.45 |
| aiforge-full-yolo11s-img1280-conserv-s42 | yolo | area_5-20% | 2 | 1.0 | 1.0 | 1.0 | 1.5 |
| aiforge-full-yolo11s-img1280-conserv-s42 | yolo | tool_gemini | 446 | 0.974 | 0.9107 | 0.9574 | 3.01 |
| aiforge-full-yolo11s-img1280-conserv-s42 | yolo | tool_ideogram | 41 | 0.6975 | 0.4775 | 0.5854 | 36.76 |
| unet_bbox_s42 | unet | all | 487 | 0.8853 | 0.606 | 0.8912 | 2.15 |
| unet_bbox_s42 | unet | area_<1% | 231 | 0.8194 | 0.5538 | 0.8009 | 2.43 |
| unet_bbox_s42 | unet | area_1-5% | 254 | 0.9516 | 0.6627 | 0.9724 | 1.9 |
| unet_bbox_s42 | unet | area_5-20% | 2 | 1.0 | 0.5535 | 1.0 | 1.5 |
| unet_bbox_s42 | unet | tool_gemini | 446 | 0.9181 | 0.6381 | 0.9372 | 1.9 |
| unet_bbox_s42 | unet | tool_ideogram | 41 | 0.4968 | 0.2503 | 0.6341 | 4.9 |
| unet_bbox_s43 | unet | all | 487 | 0.8561 | 0.5827 | 0.8871 | 2.0 |
| unet_bbox_s43 | unet | area_<1% | 231 | 0.7935 | 0.5261 | 0.8182 | 2.11 |
| unet_bbox_s43 | unet | area_1-5% | 254 | 0.9279 | 0.6532 | 0.9528 | 1.9 |
| unet_bbox_s43 | unet | area_5-20% | 2 | 0.505 | 0.505 | 0.5 | 1.5 |
| unet_bbox_s43 | unet | tool_gemini | 446 | 0.8815 | 0.6178 | 0.9148 | 1.88 |
| unet_bbox_s43 | unet | tool_ideogram | 41 | 0.503 | 0.2173 | 0.5854 | 3.29 |
| unet_bbox_s44 | unet | all | 487 | 0.3373 | 0.1777 | 0.7413 | 4.0 |
| unet_bbox_s44 | unet | area_<1% | 231 | 0.3682 | 0.216 | 0.671 | 3.87 |
| unet_bbox_s44 | unet | area_1-5% | 254 | 0.3008 | 0.1367 | 0.8071 | 4.11 |
| unet_bbox_s44 | unet | area_5-20% | 2 | 0.505 | 0.505 | 0.5 | 4.0 |
| unet_bbox_s44 | unet | tool_gemini | 446 | 0.3531 | 0.1891 | 0.7646 | 3.69 |
| unet_bbox_s44 | unet | tool_ideogram | 41 | 0.1815 | 0.0671 | 0.4146 | 7.34 |
| unet_pixel_eps8_morph_s42 | unet | all | 487 | 0.1181 | 0.0627 | 0.3388 | 20.68 |
| unet_pixel_eps8_morph_s42 | unet | area_<1% | 231 | 0.2251 | 0.1208 | 0.4545 | 18.4 |
| unet_pixel_eps8_morph_s42 | unet | area_1-5% | 254 | 0.0418 | 0.0166 | 0.2283 | 22.81 |
| unet_pixel_eps8_morph_s42 | unet | area_5-20% | 2 | 0.0 | 0.0 | 0.0 | 12.5 |
| unet_pixel_eps8_morph_s42 | unet | tool_gemini | 446 | 0.1129 | 0.0596 | 0.3229 | 19.28 |
| unet_pixel_eps8_morph_s42 | unet | tool_ideogram | 41 | 0.3062 | 0.1648 | 0.4878 | 35.85 |
| unet_pixel_eps8_morph_s43 | unet | all | 487 | 0.0688 | 0.0313 | 0.2546 | 21.67 |
| unet_pixel_eps8_morph_s43 | unet | area_<1% | 231 | 0.136 | 0.065 | 0.3463 | 17.82 |
| unet_pixel_eps8_morph_s43 | unet | area_1-5% | 254 | 0.0226 | 0.0065 | 0.1654 | 25.25 |
| unet_pixel_eps8_morph_s43 | unet | area_5-20% | 2 | 0.0 | 0.0 | 0.0 | 13.0 |
| unet_pixel_eps8_morph_s43 | unet | tool_gemini | 446 | 0.0672 | 0.031 | 0.2466 | 21.39 |
| unet_pixel_eps8_morph_s43 | unet | tool_ideogram | 41 | 0.1456 | 0.058 | 0.3659 | 24.73 |
| unet_pixel_eps8_morph_s44 | unet | all | 487 | 0.0889 | 0.0417 | 0.1109 | 19.26 |
| unet_pixel_eps8_morph_s44 | unet | area_<1% | 231 | 0.1779 | 0.0865 | 0.4329 | 14.17 |
| unet_pixel_eps8_morph_s44 | unet | area_1-5% | 254 | 0.0215 | 0.0074 | 0.1811 | 23.93 |
| unet_pixel_eps8_morph_s44 | unet | area_5-20% | 2 | 0.0 | 0.0 | 0.0 | 13.0 |
| unet_pixel_eps8_morph_s44 | unet | tool_gemini | 446 | 0.0834 | 0.0405 | 0.1099 | 18.85 |
| unet_pixel_eps8_morph_s44 | unet | tool_ideogram | 41 | 0.2994 | 0.1458 | 0.3659 | 23.73 |
| unet_glyph_v2_s42 | unet | all | 487 | 0.0104 | 0.0043 | 0.0205 | 60.34 |
| unet_glyph_v2_s42 | unet | area_<1% | 231 | 0.0114 | 0.0046 | 0.0173 | 40.74 |
| unet_glyph_v2_s42 | unet | area_1-5% | 254 | 0.0002 | 0.0001 | 0.0433 | 78.43 |
| unet_glyph_v2_s42 | unet | area_5-20% | 2 | 0.0 | 0.0 | 0.0 | 27.5 |
| unet_glyph_v2_s42 | unet | tool_gemini | 446 | 0.0103 | 0.0042 | 0.009 | 60.26 |
| unet_glyph_v2_s42 | unet | tool_ideogram | 41 | 0.0061 | 0.0027 | 0.0488 | 61.24 |
| dlv3p_bbox_s42 | unet | all | 487 | 0.8512 | 0.6 | 0.8871 | 2.96 |
| dlv3p_bbox_s42 | unet | area_<1% | 231 | 0.7552 | 0.4927 | 0.7835 | 3.48 |
| dlv3p_bbox_s42 | unet | area_1-5% | 254 | 0.9594 | 0.7136 | 0.9803 | 2.51 |
| dlv3p_bbox_s42 | unet | area_5-20% | 2 | 1.0 | 0.702 | 1.0 | 1.5 |
| dlv3p_bbox_s42 | unet | tool_gemini | 446 | 0.8791 | 0.631 | 0.935 | 2.63 |
| dlv3p_bbox_s42 | unet | tool_ideogram | 41 | 0.4967 | 0.2408 | 0.5854 | 6.59 |
| unet_bbox_mr_s42 | unet | all | 487 | 0.777 | 0.4738 | 0.8378 | 2.83 |
| unet_bbox_mr_s42 | unet | area_<1% | 231 | 0.6936 | 0.4288 | 0.7316 | 3.17 |
| unet_bbox_mr_s42 | unet | area_1-5% | 254 | 0.8763 | 0.5322 | 0.937 | 2.52 |
| unet_bbox_mr_s42 | unet | area_5-20% | 2 | 1.0 | 0.5545 | 1.0 | 1.0 |
| unet_bbox_mr_s42 | unet | tool_gemini | 446 | 0.8098 | 0.5045 | 0.8722 | 2.58 |
| unet_bbox_mr_s42 | unet | tool_ideogram | 41 | 0.4234 | 0.1662 | 0.7073 | 5.51 |
| unetjoint_pixel_eps8_morph_s42 | unet | all | 487 | 0.1343 | 0.0685 | 0.4066 | 12.62 |
| unetjoint_pixel_eps8_morph_s42 | unet | area_<1% | 231 | 0.2333 | 0.1222 | 0.5152 | 11.29 |
| unetjoint_pixel_eps8_morph_s42 | unet | area_1-5% | 254 | 0.0524 | 0.0234 | 0.3071 | 13.82 |
| unetjoint_pixel_eps8_morph_s42 | unet | area_5-20% | 2 | 0.0 | 0.0 | 0.0 | 14.5 |
| unetjoint_pixel_eps8_morph_s42 | unet | tool_gemini | 446 | 0.1296 | 0.0677 | 0.3946 | 11.83 |
| unetjoint_pixel_eps8_morph_s42 | unet | tool_ideogram | 41 | 0.3743 | 0.1437 | 0.3902 | 21.24 |
| unet_bbox_mr_s43 | unet | all | 487 | 0.7714 | 0.4668 | 0.848 | 2.89 |
| unet_bbox_mr_s43 | unet | area_<1% | 231 | 0.7041 | 0.4283 | 0.7619 | 3.17 |
| unet_bbox_mr_s43 | unet | area_1-5% | 254 | 0.8354 | 0.5087 | 0.9252 | 2.65 |
| unet_bbox_mr_s43 | unet | area_5-20% | 2 | 1.0 | 0.802 | 1.0 | 1.0 |
| unet_bbox_mr_s43 | unet | tool_gemini | 446 | 0.7943 | 0.4892 | 0.8879 | 2.66 |
| unet_bbox_mr_s43 | unet | tool_ideogram | 41 | 0.4777 | 0.196 | 0.6585 | 5.39 |
| unet_bbox_mr_s44 | unet | all | 487 | 0.8023 | 0.5092 | 0.8234 | 2.8 |
| unet_bbox_mr_s44 | unet | area_<1% | 231 | 0.7256 | 0.451 | 0.7316 | 3.29 |
| unet_bbox_mr_s44 | unet | area_1-5% | 254 | 0.8718 | 0.5677 | 0.9055 | 2.35 |
| unet_bbox_mr_s44 | unet | area_5-20% | 2 | 0.6667 | 0.2596 | 1.0 | 2.0 |
| unet_bbox_mr_s44 | unet | tool_gemini | 446 | 0.8311 | 0.534 | 0.8655 | 2.51 |
| unet_bbox_mr_s44 | unet | tool_ideogram | 41 | 0.4209 | 0.2054 | 0.6098 | 5.9 |
| unetjoint_pixel_eps8_morph_s43 | unet | all | 487 | 0.0945 | 0.0479 | 0.3121 | 13.28 |
| unetjoint_pixel_eps8_morph_s43 | unet | area_<1% | 231 | 0.1995 | 0.1011 | 0.4459 | 9.72 |
| unetjoint_pixel_eps8_morph_s43 | unet | area_1-5% | 254 | 0.0183 | 0.0074 | 0.1929 | 16.55 |
| unetjoint_pixel_eps8_morph_s43 | unet | area_5-20% | 2 | 0.0 | 0.0 | 0.0 | 10.0 |
| unetjoint_pixel_eps8_morph_s43 | unet | tool_gemini | 446 | 0.0912 | 0.0464 | 0.1099 | 13.18 |
| unetjoint_pixel_eps8_morph_s43 | unet | tool_ideogram | 41 | 0.2613 | 0.1318 | 0.2927 | 14.39 |
| unetjoint_pixel_eps8_morph_s44 | unet | all | 487 | 0.1083 | 0.052 | 0.2546 | 12.11 |
| unetjoint_pixel_eps8_morph_s44 | unet | area_<1% | 231 | 0.227 | 0.1195 | 0.3463 | 8.12 |
| unetjoint_pixel_eps8_morph_s44 | unet | area_1-5% | 254 | 0.039 | 0.0125 | 0.1732 | 15.76 |
| unetjoint_pixel_eps8_morph_s44 | unet | area_5-20% | 2 | 0.0 | 0.0 | 0.0 | 9.0 |
| unetjoint_pixel_eps8_morph_s44 | unet | tool_gemini | 446 | 0.1081 | 0.0513 | 0.2646 | 11.98 |
| unetjoint_pixel_eps8_morph_s44 | unet | tool_ideogram | 41 | 0.2297 | 0.1327 | 0.439 | 13.49 |

## T6. Dayanıklılık (relΔ pooled AUPRC, kendi GT'sinde)

| family | variant | train_gt | blur1 | clahe | gray | jpeg75 | rescale075 |
|---|---|---|---|---|---|---|---|
| DeepLabV3+ | main | box | -13.6% | -1.3% | -46.7% | -11.9% | -7.1% |
| DeepLabV3+ | main | glyph | -34.2% | -0.2% | -60.9% | -16.8% | -22.7% |
| DeepLabV3+ | main | pix | -53.6% | +0.8% | -58.5% | -20.2% | -38.7% |
| U-Net | main | box | -20.7% ± 6.7 | — | — | -16.6% ± 1.4 | -12.2% ± 4.3 |
| U-Net | main | glyph | -59.0% ± 6.8 | — | — | -18.0% ± 0.8 | -42.7% ± 5.4 |
| U-Net | main | pix | -59.1% ± 4.5 | — | — | -22.5% ± 2.3 | -40.3% ± 1.5 |
| U-Net | marker-repaired | box | -25.9% ± 5.0 | -2.1% ± 0.8 | -59.6% ± 2.6 | -32.6% ± 1.8 | -14.0% ± 2.7 |
| U-Net | marker-repaired+nm-GT | glyph | -44.4% | -2.8% | -66.7% | -32.5% | -28.9% |
| U-Net | marker-repaired+nm-GT | pix | -75.7% | -0.7% | -53.8% | -50.3% | -51.5% |
| U-Net+box-aux | main | glyph | -31.4% | -1.2% | -62.3% | -18.3% | -20.1% |
| U-Net+box-aux | main | pix | -46.0% ± 2.9 | -0.8% ± 0.8 | -66.3% ± 3.3 | -20.7% ± 2.0 | -30.8% ± 2.0 |
| YOLO11s | main | box | -1.4% | -1.2% | -58.6% | -48.5% | -0.7% |

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
| U-Net | marker-repaired | box | authentic | 3 | 0.8923 | 0.0028 | 0.8959 | 0.2402 | 0.0024 |
| U-Net | marker-repaired | box | benign_pasteback | 3 | 0.8329 | 0.0076 | 0.8322 | 0.3771 | 0.004 |
| U-Net | marker-repaired | box | fullimage_resample | 3 | 0.8564 | 0.003 | 0.8578 | 0.3518 | 0.0034 |
| U-Net | marker-repaired+nm-GT | glyph | authentic | 1 | 0.7834 | nan | 0.7301 | 0.271 | 0.0044 |
| U-Net | marker-repaired+nm-GT | glyph | benign_pasteback | 1 | 0.7329 | nan | 0.6666 | 0.3676 | 0.005 |
| U-Net | marker-repaired+nm-GT | glyph | fullimage_resample | 1 | 0.7409 | nan | 0.6767 | 0.3306 | 0.0049 |
| U-Net | marker-repaired+nm-GT | pix | authentic | 1 | 0.8437 | nan | 0.8365 | 0.1848 | 0.0024 |
| U-Net | marker-repaired+nm-GT | pix | benign_pasteback | 1 | 0.779 | nan | 0.7772 | 0.3018 | 0.0032 |
| U-Net | marker-repaired+nm-GT | pix | fullimage_resample | 1 | 0.783 | nan | 0.7848 | 0.2936 | 0.0033 |
| U-Net+box-aux | main | glyph | authentic | 1 | 0.8013 | nan | 0.7338 | 0.2752 | 0.0043 |
| U-Net+box-aux | main | glyph | benign_pasteback | 1 | 0.7828 | nan | 0.7109 | 0.2998 | 0.0045 |
| U-Net+box-aux | main | glyph | fullimage_resample | 1 | 0.7697 | nan | 0.6997 | 0.3203 | 0.0047 |
| U-Net+box-aux | main | pix | authentic | 3 | 0.9052 | 0.0075 | 0.9297 | 0.102 | 0.0018 |
| U-Net+box-aux | main | pix | benign_pasteback | 3 | 0.8884 | 0.0089 | 0.9163 | 0.1335 | 0.002 |
| U-Net+box-aux | main | pix | fullimage_resample | 3 | 0.8813 | 0.0138 | 0.9091 | 0.141 | 0.0022 |

## T8. Faz 1 AUPRC düzeltmesi (W6)

| variant | reported_phase1_auprc | reproduced_subsampled_auprc | exact_auprc_all_pixels | exact_auroc | pos_rate | n_crops | n_pixels |
|---|---|---|---|---|---|---|---|
| rgb | 0.947 | 0.9423 | 0.8053 | 0.9344 | 0.03306 | 200 | 52428800 |
| rgbfreq | 0.939 | 0.9331 | 0.7922 | 0.9255 | 0.03306 | 200 | 52428800 |

# Faz 3 — sentez tabloları (otomatik; synthesize_p3.py)

## T1. Dual-GT (val 487, temiz) — tüm aileler

| family | variant | train_gt | seeds | AUPRC_box | F1tau_box | AUPRC_glyph | F1tau_glyph | AUPRC_pix | F1tau_pix |
|---|---|---|---|---|---|---|---|---|---|
| DeepLabV3+ | main | box | 3 | 0.929 ± 0.009 | 0.831 ± 0.007 | 0.260 ± 0.005 | 0.400 ± 0.002 | 0.382 ± 0.002 | 0.463 ± 0.001 |
| DeepLabV3+ | main | glyph | 3 | 0.357 ± 0.011 | 0.504 ± 0.006 | 0.558 ± 0.010 | 0.564 ± 0.004 | 0.409 ± 0.007 | 0.495 ± 0.001 |
| DeepLabV3+ | main | pix | 3 | 0.537 ± 0.027 | 0.572 ± 0.011 | 0.437 ± 0.011 | 0.481 ± 0.013 | 0.596 ± 0.022 | 0.544 ± 0.013 |
| U-Net | 20ep | box | 3 | 0.912 ± 0.031 | 0.845 ± 0.010 | 0.261 ± 0.004 | 0.391 ± 0.006 | 0.371 ± 0.003 | 0.452 ± 0.007 |
| U-Net | 20ep | glyph | 3 | 0.346 ± 0.063 | 0.381 ± 0.021 | 0.598 ± 0.082 | 0.637 ± 0.037 | 0.342 ± 0.061 | 0.432 ± 0.038 |
| U-Net | 20ep | pix | 3 | 0.597 ± 0.064 | 0.596 ± 0.026 | 0.485 ± 0.018 | 0.524 ± 0.013 | 0.591 ± 0.015 | 0.566 ± 0.009 |
| U-Net | main | box | 3 | 0.923 ± 0.015 | 0.842 ± 0.023 | 0.257 ± 0.007 | 0.387 ± 0.009 | 0.374 ± 0.008 | 0.447 ± 0.011 |
| U-Net | main | glyph | 3 | 0.351 ± 0.015 | 0.401 ± 0.007 | 0.676 ± 0.019 | 0.674 ± 0.011 | 0.379 ± 0.012 | 0.457 ± 0.006 |
| U-Net | main | pix | 3 | 0.579 ± 0.014 | 0.600 ± 0.014 | 0.511 ± 0.010 | 0.532 ± 0.008 | 0.610 ± 0.012 | 0.577 ± 0.007 |
| U-Net | marker-repaired | box | 3 | 0.912 ± 0.006 | 0.797 ± 0.006 | 0.265 ± 0.001 | 0.377 ± 0.004 | 0.373 ± 0.013 | 0.427 ± 0.006 |
| U-Net | marker-repaired+nm-GT | glyph | 3 | 0.336 ± 0.026 | 0.378 ± 0.020 | 0.644 ± 0.014 | 0.647 ± 0.014 | 0.353 ± 0.017 | 0.436 ± 0.017 |
| U-Net | marker-repaired+nm-GT | pix | 3 | 0.480 ± 0.053 | 0.498 ± 0.027 | 0.415 ± 0.044 | 0.458 ± 0.021 | 0.507 ± 0.045 | 0.485 ± 0.017 |
| U-Net+box-aux | main | glyph | 3 | 0.395 ± 0.073 | 0.429 ± 0.011 | 0.641 ± 0.063 | 0.639 ± 0.035 | 0.386 ± 0.051 | 0.458 ± 0.021 |
| U-Net+box-aux | main | pix | 3 | 0.668 ± 0.019 | 0.629 ± 0.013 | 0.523 ± 0.009 | 0.551 ± 0.006 | 0.633 ± 0.013 | 0.588 ± 0.007 |
| YOLO11m | main | box | 3 | 0.952 ± 0.008 | 0.941 ± 0.002 | 0.222 ± 0.002 | 0.418 ± 0.001 | 0.373 ± 0.014 | 0.492 ± 0.001 |
| YOLO11n | main | box | 3 | 0.955 ± 0.010 | 0.917 ± 0.009 | 0.223 ± 0.003 | 0.408 ± 0.004 | 0.370 ± 0.005 | 0.474 ± 0.005 |
| YOLO11s | authneg | box | 3 | 0.964 ± 0.004 | 0.944 ± 0.005 | 0.223 ± 0.000 | 0.421 ± 0.002 | 0.382 ± 0.019 | 0.495 ± 0.003 |
| YOLO11s | defaug | box | 3 | 0.968 ± 0.001 | 0.950 ± 0.005 | 0.218 ± 0.003 | 0.422 ± 0.002 | 0.366 ± 0.006 | 0.496 ± 0.002 |
| YOLO11s | full1280 | box | 3 | 0.923 ± 0.004 | 0.899 ± 0.003 | 0.207 ± 0.004 | 0.395 ± 0.001 | 0.341 ± 0.007 | 0.464 ± 0.002 |
| YOLO11s | main | box | 3 | 0.965 ± 0.003 | 0.947 ± 0.005 | 0.221 ± 0.003 | 0.420 ± 0.002 | 0.368 ± 0.015 | 0.494 ± 0.003 |

## T2. Resmî test (box GT; piksel GT 89 çift) + tespit AP

| family | variant | train_gt | seeds | AUPRC_box_812 | AUPRC_box_740 | F1tau_box | IoUtau_box | AUPRC_pix_89 | AP50 | AP50_95 |
|---|---|---|---|---|---|---|---|---|---|---|
| DeepLabV3+ | main | box | 3 | 0.951 ± 0.001 | 0.950 ± 0.002 | 0.830 ± 0.003 | 0.761 ± 0.005 | 0.408 ± 0.011 | 0.849 ± 0.024 | 0.589 ± 0.010 |
| DeepLabV3+ | main | glyph | 3 | 0.391 ± 0.020 | 0.389 ± 0.021 | 0.504 ± 0.008 | 0.362 ± 0.008 | 0.419 ± 0.030 | — | — |
| DeepLabV3+ | main | pix | 3 | 0.572 ± 0.030 | 0.572 ± 0.032 | 0.579 ± 0.016 | 0.444 ± 0.017 | 0.660 ± 0.007 | — | — |
| U-Net | 20ep | box | 3 | 0.946 ± 0.015 | 0.945 ± 0.016 | 0.849 ± 0.013 | 0.790 ± 0.016 | 0.428 ± 0.012 | — | — |
| U-Net | 20ep | glyph | 3 | 0.412 ± 0.082 | 0.412 ± 0.082 | 0.388 ± 0.021 | 0.253 ± 0.017 | 0.443 ± 0.061 | — | — |
| U-Net | 20ep | pix | 3 | 0.649 ± 0.079 | 0.648 ± 0.081 | 0.613 ± 0.034 | 0.478 ± 0.039 | 0.631 ± 0.025 | — | — |
| U-Net | main | box | 3 | 0.950 ± 0.011 | 0.949 ± 0.012 | 0.846 ± 0.015 | 0.789 ± 0.018 | 0.427 ± 0.002 | 0.712 ± 0.279 | 0.453 ± 0.224 |
| U-Net | main | glyph | 3 | 0.403 ± 0.011 | 0.403 ± 0.010 | 0.405 ± 0.007 | 0.266 ± 0.006 | 0.410 ± 0.017 | 0.011 | 0.002 |
| U-Net | main | pix | 3 | 0.638 ± 0.015 | 0.640 ± 0.014 | 0.616 ± 0.013 | 0.480 ± 0.015 | 0.654 ± 0.016 | 0.107 ± 0.027 | 0.053 ± 0.014 |
| U-Net | marker-repaired | box | 3 | 0.940 ± 0.003 | 0.940 ± 0.002 | 0.799 ± 0.016 | 0.727 ± 0.015 | 0.395 ± 0.027 | 0.804 ± 0.005 | 0.486 ± 0.007 |
| U-Net | marker-repaired+nm-GT | glyph | 3 | 0.393 ± 0.018 | 0.394 ± 0.019 | 0.382 ± 0.020 | 0.247 ± 0.016 | 0.342 ± 0.011 | — | — |
| U-Net | marker-repaired+nm-GT | pix | 3 | 0.536 ± 0.060 | 0.538 ± 0.057 | 0.512 ± 0.030 | 0.384 ± 0.026 | 0.569 ± 0.024 | — | — |
| U-Net+box-aux | main | glyph | 3 | 0.438 ± 0.069 | 0.438 ± 0.072 | 0.435 ± 0.013 | 0.294 ± 0.012 | 0.440 ± 0.043 | — | — |
| U-Net+box-aux | main | pix | 3 | 0.702 ± 0.020 | 0.702 ± 0.021 | 0.641 ± 0.012 | 0.506 ± 0.015 | 0.652 ± 0.003 | 0.125 ± 0.020 | 0.064 ± 0.010 |
| YOLO11m | main | box | 3 | 0.966 ± 0.008 | 0.966 ± 0.009 | 0.948 ± 0.002 | 0.920 ± 0.001 | 0.437 ± 0.024 | 0.882 ± 0.015 | 0.785 ± 0.014 |
| YOLO11n | main | box | 3 | 0.966 ± 0.003 | 0.967 ± 0.003 | 0.928 ± 0.003 | 0.894 ± 0.005 | 0.401 ± 0.012 | 0.871 ± 0.002 | 0.733 ± 0.004 |
| YOLO11s | authneg | box | 3 | 0.969 ± 0.002 | 0.970 ± 0.002 | 0.938 ± 0.006 | 0.907 ± 0.008 | 0.446 ± 0.036 | 0.874 ± 0.003 | 0.770 ± 0.005 |
| YOLO11s | defaug | box | 3 | 0.973 ± 0.002 | 0.973 ± 0.002 | 0.954 ± 0.003 | 0.926 ± 0.003 | 0.402 ± 0.013 | 0.867 ± 0.004 | 0.769 ± 0.005 |
| YOLO11s | full1280 | box | 3 | 0.936 ± 0.005 | 0.937 ± 0.005 | 0.906 ± 0.004 | 0.872 ± 0.005 | 0.441 ± 0.016 | 0.961 ± 0.003 | 0.879 ± 0.004 |
| YOLO11s | main | box | 3 | 0.969 ± 0.006 | 0.969 ± 0.007 | 0.946 ± 0.007 | 0.918 ± 0.006 | 0.404 ± 0.042 | 0.880 ± 0.002 | 0.780 ± 0.003 |

## T3. Sondalar (val çiftleri; işaretçili alt küme için AP; seed-ortalaması + grup bootstrap CI)

| model | seeds | n_res | APbox_forged | cover_forged | APbox_forged_repaired | cover_forged_repaired | APbox_authentic_plus_marker | cover_authentic_plus_marker | APbox_authentic | cover_authentic | dAP_repair | AUROC_forged | AUROC_forged_repaired | AUROC_authentic_plus_marker | AUROC_render_only | AUROC_content_only | inbox_render_only | APbox_all_render_only | inbox_content_only | APbox_all_content_only | inbox_forged | APbox_all_forged | dAPbox_clahe | dAPbox_gray |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DeepLabV3+|main|bbox | 3 | 380 | 0.982 [0.976, 0.988] | 0.921 | 0.810 [0.779, 0.836] | 0.649 | 0.931 [0.918, 0.943] | 0.785 | 0.186 [0.164, 0.210] | 0.043 | -0.173 [-0.203, -0.146] | 0.974 | 0.883 | 0.904 | 0.656 | 0.762 | 0.279 | 0.453 [0.419, 0.487] | 0.408 | 0.600 [0.569, 0.632] | 0.868 | 0.934 [0.918, 0.949] | -0.003 [-0.009, +0.004] | -0.338 [-0.370, -0.308] |
| DeepLabV3+|main|glyph_v2 | 3 | 380 | 0.640 [0.625, 0.654] | 0.495 | 0.521 [0.495, 0.545] | 0.373 | 0.576 [0.557, 0.594] | 0.440 | 0.147 [0.130, 0.167] | 0.055 | -0.119 [-0.144, -0.096] | 0.919 | 0.786 | 0.835 | 0.581 | 0.699 | 0.088 | 0.284 [0.262, 0.307] | 0.169 | 0.404 [0.380, 0.429] | 0.345 | 0.608 [0.590, 0.625] | +0.030 [+0.025, +0.035] | -0.201 [-0.229, -0.173] |
| DeepLabV3+|main|pixel_eps8_morph | 3 | 380 | 0.718 [0.701, 0.735] | 0.562 | 0.575 [0.547, 0.602] | 0.397 | 0.612 [0.589, 0.634] | 0.450 | 0.159 [0.141, 0.181] | 0.061 | -0.143 [-0.171, -0.118] | 0.948 | 0.797 | 0.845 | 0.561 | 0.715 | 0.073 | 0.268 [0.247, 0.291] | 0.207 | 0.456 [0.429, 0.483] | 0.415 | 0.692 [0.673, 0.710] | +0.034 [+0.028, +0.041] | -0.262 [-0.288, -0.235] |
| U-Net+box-aux|main|glyph_v2 | 3 | 380 | 0.659 [0.643, 0.676] | 0.372 | 0.499 [0.470, 0.525] | 0.227 | 0.582 [0.562, 0.602] | 0.329 | 0.131 [0.116, 0.149] | 0.027 | -0.161 [-0.188, -0.136] | 0.932 | 0.771 | 0.871 | 0.582 | 0.683 | 0.057 | 0.259 [0.234, 0.281] | 0.105 | 0.362 [0.336, 0.388] | 0.263 | 0.625 [0.607, 0.644] | +0.011 [+0.006, +0.016] | -0.259 [-0.288, -0.232] |
| U-Net+box-aux|main|pixel_eps8_morph | 3 | 380 | 0.795 [0.782, 0.807] | 0.558 | 0.600 [0.566, 0.631] | 0.343 | 0.700 [0.680, 0.719] | 0.457 | 0.148 [0.128, 0.171] | 0.033 | -0.194 [-0.226, -0.164] | 0.961 | 0.805 | 0.887 | 0.593 | 0.716 | 0.061 | 0.294 [0.266, 0.320] | 0.163 | 0.441 [0.409, 0.470] | 0.413 | 0.757 [0.739, 0.775] | +0.035 [+0.029, +0.041] | -0.338 [-0.367, -0.309] |
| U-Net|geminionly|bbox | 3 | 380 | 0.990 [0.986, 0.993] | 0.961 | 0.715 [0.676, 0.751] | 0.581 | 0.910 [0.892, 0.927] | 0.816 | 0.170 [0.150, 0.192] | 0.032 | -0.275 [-0.313, -0.238] | 0.957 | 0.827 | 0.891 | 0.641 | 0.714 | 0.195 | 0.380 [0.346, 0.412] | 0.271 | 0.488 [0.454, 0.522] | 0.834 | 0.908 [0.886, 0.929] | +0.003 [-0.003, +0.010] | -0.463 [-0.499, -0.426] |
| U-Net|main|bbox | 3 | 380 | 0.988 [0.982, 0.992] | 0.939 | 0.766 [0.731, 0.797] | 0.595 | 0.905 [0.886, 0.922] | 0.795 | 0.176 [0.155, 0.201] | 0.041 | -0.222 [-0.256, -0.190] | 0.971 | 0.837 | 0.903 | 0.637 | 0.733 | 0.237 | 0.415 [0.381, 0.447] | 0.348 | 0.557 [0.521, 0.589] | 0.864 | 0.933 [0.915, 0.950] | +0.000 [-0.004, +0.005] | -0.388 [-0.421, -0.355] |
| U-Net|main|glyph_v2 | 3 | 380 | 0.522 [0.505, 0.541] | 0.325 | 0.398 [0.372, 0.425] | 0.195 | 0.461 [0.442, 0.481] | 0.283 | 0.123 [0.108, 0.141] | 0.023 | -0.123 [-0.146, -0.103] | 0.902 | 0.763 | 0.834 | 0.579 | 0.683 | 0.050 | 0.212 [0.192, 0.231] | 0.096 | 0.320 [0.295, 0.344] | 0.239 | 0.507 [0.489, 0.527] | +0.036 [+0.031, +0.041] | -0.213 [-0.235, -0.191] |
| U-Net|main|pixel_eps8_morph | 3 | 380 | 0.747 [0.731, 0.762] | 0.538 | 0.500 [0.468, 0.529] | 0.282 | 0.635 [0.611, 0.656] | 0.420 | 0.134 [0.117, 0.155] | 0.031 | -0.247 [-0.279, -0.218] | 0.942 | 0.762 | 0.858 | 0.554 | 0.681 | 0.045 | 0.211 [0.189, 0.233] | 0.150 | 0.367 [0.338, 0.396] | 0.406 | 0.706 [0.685, 0.725] | +0.026 [+0.020, +0.032] | -0.370 [-0.398, -0.342] |
| U-Net|marker-repaired+geminionly|bbox | 3 | 380 | 0.941 [0.922, 0.957] | 0.874 | 0.950 [0.935, 0.963] | 0.860 | 0.636 [0.602, 0.670] | 0.434 | 0.201 [0.179, 0.226] | 0.041 | +0.009 [-0.004, +0.023] | 0.959 | 0.958 | 0.744 | 0.725 | 0.879 | 0.368 | 0.554 [0.519, 0.586] | 0.584 | 0.747 [0.719, 0.774] | 0.825 | 0.893 [0.872, 0.914] | -0.007 [-0.014, -0.000] | -0.325 [-0.356, -0.293] |
| U-Net|marker-repaired+nm-GT|glyph_v2 | 3 | 380 | 0.502 [0.483, 0.522] | 0.301 | 0.511 [0.492, 0.530] | 0.305 | 0.338 [0.316, 0.360] | 0.167 | 0.121 [0.106, 0.138] | 0.026 | +0.009 [+0.002, +0.016] | 0.891 | 0.891 | 0.695 | 0.662 | 0.830 | 0.081 | 0.280 [0.259, 0.301] | 0.180 | 0.425 [0.404, 0.446] | 0.224 | 0.494 [0.474, 0.515] | +0.031 [+0.025, +0.037] | -0.151 [-0.173, -0.129] |
| U-Net|marker-repaired+nm-GT|pixel_eps8_morph | 3 | 380 | 0.647 [0.624, 0.668] | 0.419 | 0.700 [0.680, 0.718] | 0.490 | 0.262 [0.244, 0.284] | 0.102 | 0.141 [0.123, 0.161] | 0.035 | +0.054 [+0.037, +0.072] | 0.894 | 0.920 | 0.604 | 0.655 | 0.851 | 0.091 | 0.323 [0.298, 0.346] | 0.281 | 0.541 [0.517, 0.566] | 0.316 | 0.627 [0.606, 0.649] | +0.010 [+0.002, +0.018] | -0.213 [-0.234, -0.193] |
| U-Net|marker-repaired|bbox | 3 | 380 | 0.939 [0.921, 0.956] | 0.891 | 0.956 [0.942, 0.969] | 0.895 | 0.617 [0.579, 0.655] | 0.442 | 0.179 [0.156, 0.203] | 0.047 | +0.017 [+0.001, +0.033] | 0.957 | 0.963 | 0.720 | 0.724 | 0.886 | 0.345 | 0.548 [0.513, 0.582] | 0.583 | 0.761 [0.734, 0.788] | 0.823 | 0.902 [0.882, 0.922] | -0.013 [-0.021, -0.005] | -0.295 [-0.326, -0.265] |
| YOLO11s|main|bbox | 3 | 380 | 0.955 [0.951, 0.959] | 0.957 | 0.897 [0.887, 0.907] | 0.876 | 0.808 [0.786, 0.828] | 0.553 | 0.104 [0.090, 0.120] | 0.002 | -0.058 [-0.068, -0.049] | 0.998 | 0.995 | 0.864 | 0.890 | 0.884 | 0.448 | 0.699 [0.669, 0.729] | 0.390 | 0.684 [0.657, 0.709] | 0.867 | 0.946 [0.939, 0.952] | -0.020 [-0.029, -0.013] | -0.357 [-0.388, -0.327] |

## T3b. Test: işaretçili vs işaretçisiz görüntüler (box GT, görüntü-AP)

| family | variant | train_gt | seeds | AP_with_marker | n_with_marker | AP_no_marker | n_no_marker | AP_no_marker_gemini | n_nm_gem |
|---|---|---|---|---|---|---|---|---|---|
| DeepLabV3+ | main | box | 3 | 0.962 [0.954, 0.968] | 627 | 0.702 [0.654, 0.746] | 185 | 0.751 [0.692, 0.807] | 102 |
| DeepLabV3+ | main | glyph | 3 | 0.522 [0.509, 0.536] | 627 | 0.415 [0.377, 0.450] | 185 | 0.480 [0.434, 0.529] | 102 |
| DeepLabV3+ | main | pix | 3 | 0.619 [0.603, 0.634] | 627 | 0.547 [0.501, 0.592] | 185 | 0.638 [0.587, 0.695] | 102 |
| U-Net | 20ep | box | 3 | 0.971 [0.964, 0.977] | 627 | 0.694 [0.644, 0.742] | 185 | 0.733 [0.670, 0.795] | 102 |
| U-Net | 20ep | glyph | 3 | 0.397 [0.386, 0.409] | 627 | 0.318 [0.283, 0.355] | 185 | 0.358 [0.311, 0.406] | 102 |
| U-Net | 20ep | pix | 3 | 0.657 [0.643, 0.669] | 627 | 0.509 [0.461, 0.557] | 185 | 0.589 [0.527, 0.651] | 102 |
| U-Net | main | box | 3 | 0.972 [0.966, 0.977] | 627 | 0.690 [0.640, 0.739] | 185 | 0.722 [0.657, 0.788] | 102 |
| U-Net | main | glyph | 3 | 0.388 [0.377, 0.399] | 627 | 0.339 [0.307, 0.374] | 185 | 0.361 [0.317, 0.405] | 102 |
| U-Net | main | pix | 3 | 0.639 [0.624, 0.652] | 627 | 0.503 [0.455, 0.550] | 185 | 0.559 [0.504, 0.622] | 102 |
| U-Net | marker-repaired | box | 3 | 0.914 [0.899, 0.927] | 627 | 0.711 [0.663, 0.758] | 185 | 0.762 [0.703, 0.821] | 102 |
| U-Net | marker-repaired+nm-GT | glyph | 3 | 0.361 [0.349, 0.373] | 627 | 0.334 [0.301, 0.370] | 185 | 0.355 [0.311, 0.400] | 102 |
| U-Net | marker-repaired+nm-GT | pix | 3 | 0.505 [0.488, 0.523] | 627 | 0.507 [0.463, 0.550] | 185 | 0.576 [0.520, 0.631] | 102 |
| U-Net+box-aux | main | glyph | 3 | 0.474 [0.461, 0.486] | 627 | 0.376 [0.340, 0.414] | 185 | 0.412 [0.362, 0.461] | 102 |
| U-Net+box-aux | main | pix | 3 | 0.687 [0.674, 0.699] | 627 | 0.544 [0.496, 0.593] | 185 | 0.600 [0.541, 0.662] | 102 |
| YOLO11m | main | box | 3 | 0.956 [0.953, 0.958] | 627 | 0.879 [0.857, 0.896] | 185 | 0.893 [0.866, 0.913] | 102 |
| YOLO11n | main | box | 3 | 0.946 [0.940, 0.950] | 627 | 0.825 [0.799, 0.848] | 185 | 0.813 [0.777, 0.845] | 102 |
| YOLO11s | authneg | box | 3 | 0.952 [0.949, 0.954] | 627 | 0.863 [0.838, 0.885] | 185 | 0.869 [0.842, 0.895] | 102 |
| YOLO11s | defaug | box | 3 | 0.956 [0.953, 0.958] | 627 | 0.878 [0.856, 0.895] | 185 | 0.869 [0.842, 0.894] | 102 |
| YOLO11s | full1280 | box | 3 | 0.933 [0.926, 0.938] | 627 | 0.750 [0.709, 0.786] | 185 | 0.847 [0.815, 0.877] | 102 |
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
| U-Net | marker-repaired+geminionly | box | True | 3 | box | 126 | 0.463 [0.400, 0.528] | 0.874 [0.856, 0.893] | 0.412 |
| U-Net | marker-repaired+geminionly | box | True | 3 | pix | 41 | 0.307 [0.218, 0.402] | 0.367 [0.339, 0.395] | 0.060 |
| U-Net | marker-repaired+nm-GT | glyph | False | 3 | box | 126 | 0.308 [0.266, 0.352] | 0.347 [0.334, 0.361] | 0.039 |
| U-Net | marker-repaired+nm-GT | glyph | False | 3 | pix | 41 | 0.354 [0.267, 0.433] | 0.373 [0.347, 0.399] | 0.019 |
| U-Net | marker-repaired+nm-GT | pix | False | 3 | box | 126 | 0.413 [0.359, 0.468] | 0.468 [0.450, 0.487] | 0.055 |
| U-Net | marker-repaired+nm-GT | pix | False | 3 | pix | 41 | 0.446 [0.350, 0.542] | 0.471 [0.438, 0.506] | 0.024 |
| U-Net | marker-repaired+nm-GT+geminionly | glyph | True | 3 | box | 126 | 0.203 [0.168, 0.240] | 0.315 [0.303, 0.327] | 0.112 |
| U-Net | marker-repaired+nm-GT+geminionly | glyph | True | 3 | pix | 41 | 0.239 [0.168, 0.314] | 0.352 [0.328, 0.377] | 0.113 |
| U-Net | marker-repaired+nm-GT+geminionly | pix | True | 3 | box | 126 | 0.317 [0.267, 0.366] | 0.472 [0.453, 0.491] | 0.155 |
| U-Net | marker-repaired+nm-GT+geminionly | pix | True | 3 | pix | 41 | 0.350 [0.260, 0.449] | 0.480 [0.448, 0.514] | 0.130 |
| YOLO11s | geminionly | box | True | 3 | box | 126 | 0.725 [0.675, 0.778] | 0.938 [0.930, 0.946] | 0.214 |
| YOLO11s | geminionly | box | True | 3 | pix | 41 | 0.402 [0.330, 0.475] | 0.334 [0.311, 0.358] | -0.068 |
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
| aiforge-tiles_geminionly-yolo11s-img1024-conserv-s43 | 30 | 30 | 31.2 | 9.43 | 0.9882 | 0.865 | 0.760 | 0.803 | 0.858 | 0.755 | 0.821 | 0.959 | 0.388 | — |
| aiforge-full-yolo11s-img1280-conserv-s43 | 30 | 30 | 27.0 | 9.43 | 0.9629 | 0.956 | 0.879 | 0.908 | 0.959 | 0.875 | 0.928 | 0.920 | 0.340 | — |
| aiforge-tiles-yolo11s-img1024-default-s43 | 30 | 27 | 34.0 | 9.43 | 0.9876 | 0.878 | 0.773 | 0.835 | 0.871 | 0.774 | 0.852 | 0.968 | 0.363 | — |
| aiforge-tiles_authneg-yolo11s-img1024-conserv-s43 | 30 | 28 | 48.8 | 9.43 | 0.9869 | 0.876 | 0.769 | 0.842 | 0.876 | 0.770 | 0.858 | 0.966 | 0.394 | — |
| aiforge-tiles-yolo11n-img1024-conserv-s43 | 30 | 30 | 29.2 | 2.6 | 0.9735 | 0.857 | 0.721 | 0.803 | 0.873 | 0.730 | 0.850 | 0.961 | 0.369 | — |
| aiforge-tiles-yolo11m-img1024-conserv-s43 | 30 | 21 | 66.7 | 20.05 | 0.9716 | 0.887 | 0.781 | 0.876 | 0.879 | 0.780 | 0.880 | 0.945 | 0.378 | — |
| aiforge-tiles_geminionly-yolo11s-img1024-conserv-s44 | 30 | 26 | 27.6 | 9.43 | 0.9909 | 0.870 | 0.767 | 0.803 | 0.857 | 0.761 | 0.808 | 0.959 | 0.344 | — |
| aiforge-full-yolo11s-img1280-conserv-s44 | 30 | 30 | 25.3 | 9.43 | 0.9648 | 0.961 | 0.885 | 0.924 | 0.965 | 0.880 | 0.939 | 0.927 | 0.349 | — |
| aiforge-tiles-yolo11s-img1024-default-s44 | 30 | 25 | 32.4 | 9.43 | 0.9862 | 0.861 | 0.761 | 0.818 | 0.866 | 0.769 | 0.838 | 0.968 | 0.364 | — |
| aiforge-tiles_authneg-yolo11s-img1024-conserv-s44 | 30 | 24 | 46.6 | 9.43 | 0.9799 | 0.873 | 0.767 | 0.829 | 0.871 | 0.764 | 0.855 | 0.960 | 0.392 | — |
| aiforge-tiles-yolo11n-img1024-conserv-s44 | 30 | 30 | 29.7 | 2.6 | 0.9651 | 0.847 | 0.712 | 0.796 | 0.869 | 0.732 | 0.837 | 0.960 | 0.377 | — |
| aiforge-tiles-yolo11m-img1024-conserv-s44 | 30 | 22 | 72.1 | 20.05 | 0.967 | 0.868 | 0.769 | 0.861 | 0.868 | 0.773 | 0.870 | 0.950 | 0.384 | — |

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
| aiforge-tiles_geminionly-yolo11s-img1024-conserv-s43 | yolo | all | 487 | 0.8647 | 0.7602 | 0.9384 | 3.03 |
| aiforge-tiles_geminionly-yolo11s-img1024-conserv-s43 | yolo | area_<1% | 231 | 0.8035 | 0.6682 | 0.9091 | 4.12 |
| aiforge-tiles_geminionly-yolo11s-img1024-conserv-s43 | yolo | area_1-5% | 254 | 0.9395 | 0.8533 | 0.9843 | 2.05 |
| aiforge-tiles_geminionly-yolo11s-img1024-conserv-s43 | yolo | area_5-20% | 2 | 1.0 | 1.0 | 1.0 | 1.5 |
| aiforge-tiles_geminionly-yolo11s-img1024-conserv-s43 | yolo | tool_gemini | 446 | 0.8838 | 0.789 | 0.9686 | 2.73 |
| aiforge-tiles_geminionly-yolo11s-img1024-conserv-s43 | yolo | tool_ideogram | 41 | 0.6777 | 0.4441 | 0.6098 | 6.27 |
| aiforge-full-yolo11s-img1280-conserv-s43 | yolo | all | 487 | 0.9559 | 0.8792 | 0.9261 | 4.01 |
| aiforge-full-yolo11s-img1280-conserv-s43 | yolo | area_<1% | 231 | 0.9084 | 0.7851 | 0.8528 | 6.77 |
| aiforge-full-yolo11s-img1280-conserv-s43 | yolo | area_1-5% | 254 | 0.9901 | 0.9581 | 0.9921 | 1.5 |
| aiforge-full-yolo11s-img1280-conserv-s43 | yolo | area_5-20% | 2 | 1.0 | 0.7772 | 1.0 | 3.0 |
| aiforge-full-yolo11s-img1280-conserv-s43 | yolo | tool_gemini | 446 | 0.9823 | 0.9163 | 0.9641 | 2.62 |
| aiforge-full-yolo11s-img1280-conserv-s43 | yolo | tool_ideogram | 41 | 0.6242 | 0.4305 | 0.5366 | 19.1 |
| aiforge-tiles-yolo11s-img1024-default-s43 | yolo | all | 487 | 0.8778 | 0.7726 | 0.9713 | 8.3 |
| aiforge-tiles-yolo11s-img1024-default-s43 | yolo | area_<1% | 231 | 0.8352 | 0.7064 | 0.9567 | 14.77 |
| aiforge-tiles-yolo11s-img1024-default-s43 | yolo | area_1-5% | 254 | 0.9201 | 0.8387 | 0.9961 | 2.48 |
| aiforge-tiles-yolo11s-img1024-default-s43 | yolo | area_5-20% | 2 | 1.0 | 0.9505 | 1.0 | 1.0 |
| aiforge-tiles-yolo11s-img1024-default-s43 | yolo | tool_gemini | 446 | 0.8823 | 0.7906 | 0.9843 | 6.43 |
| aiforge-tiles-yolo11s-img1024-default-s43 | yolo | tool_ideogram | 41 | 0.8287 | 0.579 | 0.9268 | 28.66 |
| aiforge-tiles_authneg-yolo11s-img1024-conserv-s43 | yolo | all | 487 | 0.8765 | 0.7693 | 0.9651 | 3.02 |
| aiforge-tiles_authneg-yolo11s-img1024-conserv-s43 | yolo | area_<1% | 231 | 0.8421 | 0.7003 | 0.9264 | 4.31 |
| aiforge-tiles_authneg-yolo11s-img1024-conserv-s43 | yolo | area_1-5% | 254 | 0.9136 | 0.8347 | 0.9764 | 1.86 |
| aiforge-tiles_authneg-yolo11s-img1024-conserv-s43 | yolo | area_5-20% | 2 | 1.0 | 0.9252 | 1.0 | 1.0 |
| aiforge-tiles_authneg-yolo11s-img1024-conserv-s43 | yolo | tool_gemini | 446 | 0.8877 | 0.7883 | 0.9641 | 2.56 |
| aiforge-tiles_authneg-yolo11s-img1024-conserv-s43 | yolo | tool_ideogram | 41 | 0.8212 | 0.5681 | 0.9268 | 8.0 |
| aiforge-tiles-yolo11n-img1024-conserv-s43 | yolo | all | 487 | 0.8572 | 0.7205 | 0.9528 | 8.16 |
| aiforge-tiles-yolo11n-img1024-conserv-s43 | yolo | area_<1% | 231 | 0.803 | 0.6377 | 0.9221 | 13.72 |
| aiforge-tiles-yolo11n-img1024-conserv-s43 | yolo | area_1-5% | 254 | 0.9173 | 0.8076 | 0.9803 | 3.15 |
| aiforge-tiles-yolo11n-img1024-conserv-s43 | yolo | area_5-20% | 2 | 1.0 | 0.9505 | 1.0 | 3.0 |
| aiforge-tiles-yolo11n-img1024-conserv-s43 | yolo | tool_gemini | 446 | 0.8632 | 0.7401 | 0.935 | 6.2 |
| aiforge-tiles-yolo11n-img1024-conserv-s43 | yolo | tool_ideogram | 41 | 0.7665 | 0.4986 | 0.9024 | 29.51 |
| aiforge-tiles-yolo11m-img1024-conserv-s43 | yolo | all | 487 | 0.8875 | 0.7807 | 0.9446 | 5.28 |
| aiforge-tiles-yolo11m-img1024-conserv-s43 | yolo | area_<1% | 231 | 0.8762 | 0.7451 | 0.9048 | 7.66 |
| aiforge-tiles-yolo11m-img1024-conserv-s43 | yolo | area_1-5% | 254 | 0.8992 | 0.8211 | 0.9764 | 3.15 |
| aiforge-tiles-yolo11m-img1024-conserv-s43 | yolo | area_5-20% | 2 | 1.0 | 1.0 | 1.0 | 1.5 |
| aiforge-tiles-yolo11m-img1024-conserv-s43 | yolo | tool_gemini | 446 | 0.8936 | 0.803 | 0.9529 | 4.3 |
| aiforge-tiles-yolo11m-img1024-conserv-s43 | yolo | tool_ideogram | 41 | 0.8333 | 0.5663 | 0.9268 | 16.0 |
| aiforge-tiles_geminionly-yolo11s-img1024-conserv-s44 | yolo | all | 487 | 0.8697 | 0.7671 | 0.9487 | 3.55 |
| aiforge-tiles_geminionly-yolo11s-img1024-conserv-s44 | yolo | area_<1% | 231 | 0.8026 | 0.6753 | 0.9177 | 4.89 |
| aiforge-tiles_geminionly-yolo11s-img1024-conserv-s44 | yolo | area_1-5% | 254 | 0.9446 | 0.8612 | 0.9764 | 2.36 |
| aiforge-tiles_geminionly-yolo11s-img1024-conserv-s44 | yolo | area_5-20% | 2 | 1.0 | 1.0 | 1.0 | 1.0 |
| aiforge-tiles_geminionly-yolo11s-img1024-conserv-s44 | yolo | tool_gemini | 446 | 0.8862 | 0.7962 | 0.9709 | 3.22 |
| aiforge-tiles_geminionly-yolo11s-img1024-conserv-s44 | yolo | tool_ideogram | 41 | 0.7345 | 0.4598 | 0.7073 | 7.24 |
| aiforge-full-yolo11s-img1280-conserv-s44 | yolo | all | 487 | 0.9611 | 0.8849 | 0.9281 | 3.92 |
| aiforge-full-yolo11s-img1280-conserv-s44 | yolo | area_<1% | 231 | 0.9239 | 0.7972 | 0.8874 | 6.49 |
| aiforge-full-yolo11s-img1280-conserv-s44 | yolo | area_1-5% | 254 | 0.9895 | 0.9592 | 0.9921 | 1.59 |
| aiforge-full-yolo11s-img1280-conserv-s44 | yolo | area_5-20% | 2 | 1.0 | 0.901 | 1.0 | 2.0 |
| aiforge-full-yolo11s-img1280-conserv-s44 | yolo | tool_gemini | 446 | 0.9761 | 0.9167 | 0.9619 | 2.54 |
| aiforge-full-yolo11s-img1280-conserv-s44 | yolo | tool_ideogram | 41 | 0.7159 | 0.469 | 0.6585 | 18.9 |
| aiforge-tiles-yolo11s-img1024-default-s44 | yolo | all | 487 | 0.861 | 0.7615 | 0.9815 | 7.48 |
| aiforge-tiles-yolo11s-img1024-default-s44 | yolo | area_<1% | 231 | 0.818 | 0.6901 | 0.974 | 13.01 |
| aiforge-tiles-yolo11s-img1024-default-s44 | yolo | area_1-5% | 254 | 0.9109 | 0.8367 | 0.9882 | 2.5 |
| aiforge-tiles-yolo11s-img1024-default-s44 | yolo | area_5-20% | 2 | 1.0 | 0.9505 | 1.0 | 1.0 |
| aiforge-tiles-yolo11s-img1024-default-s44 | yolo | tool_gemini | 446 | 0.87 | 0.7799 | 0.9821 | 6.16 |
| aiforge-tiles-yolo11s-img1024-default-s44 | yolo | tool_ideogram | 41 | 0.8157 | 0.5799 | 0.9756 | 21.8 |
| aiforge-tiles_authneg-yolo11s-img1024-conserv-s44 | yolo | all | 487 | 0.8726 | 0.7671 | 0.9651 | 3.09 |
| aiforge-tiles_authneg-yolo11s-img1024-conserv-s44 | yolo | area_<1% | 231 | 0.8292 | 0.698 | 0.9481 | 4.6 |
| aiforge-tiles_authneg-yolo11s-img1024-conserv-s44 | yolo | area_1-5% | 254 | 0.9129 | 0.8353 | 0.9843 | 1.74 |
| aiforge-tiles_authneg-yolo11s-img1024-conserv-s44 | yolo | area_5-20% | 2 | 1.0 | 0.9505 | 1.0 | 1.0 |
| aiforge-tiles_authneg-yolo11s-img1024-conserv-s44 | yolo | tool_gemini | 446 | 0.8783 | 0.785 | 0.9686 | 2.55 |
| aiforge-tiles_authneg-yolo11s-img1024-conserv-s44 | yolo | tool_ideogram | 41 | 0.7924 | 0.5696 | 0.9756 | 9.05 |
| aiforge-tiles-yolo11n-img1024-conserv-s44 | yolo | all | 487 | 0.8474 | 0.7124 | 0.9363 | 6.85 |
| aiforge-tiles-yolo11n-img1024-conserv-s44 | yolo | area_<1% | 231 | 0.7963 | 0.6277 | 0.8788 | 11.59 |
| aiforge-tiles-yolo11n-img1024-conserv-s44 | yolo | area_1-5% | 254 | 0.9011 | 0.8012 | 0.9843 | 2.56 |
| aiforge-tiles-yolo11n-img1024-conserv-s44 | yolo | area_5-20% | 2 | 1.0 | 0.9505 | 1.0 | 3.5 |
| aiforge-tiles-yolo11n-img1024-conserv-s44 | yolo | tool_gemini | 446 | 0.8533 | 0.7371 | 0.935 | 5.28 |
| aiforge-tiles-yolo11n-img1024-conserv-s44 | yolo | tool_ideogram | 41 | 0.7603 | 0.4516 | 0.8537 | 23.98 |
| aiforge-tiles-yolo11m-img1024-conserv-s44 | yolo | all | 487 | 0.8681 | 0.7687 | 0.9548 | 5.61 |
| aiforge-tiles-yolo11m-img1024-conserv-s44 | yolo | area_<1% | 231 | 0.8609 | 0.7301 | 0.9351 | 8.6 |
| aiforge-tiles-yolo11m-img1024-conserv-s44 | yolo | area_1-5% | 254 | 0.9061 | 0.8298 | 0.9724 | 2.93 |
| aiforge-tiles-yolo11m-img1024-conserv-s44 | yolo | area_5-20% | 2 | 1.0 | 1.0 | 1.0 | 1.0 |
| aiforge-tiles-yolo11m-img1024-conserv-s44 | yolo | tool_gemini | 446 | 0.8704 | 0.7837 | 0.9664 | 4.66 |
| aiforge-tiles-yolo11m-img1024-conserv-s44 | yolo | tool_ideogram | 41 | 0.8347 | 0.5809 | 0.8537 | 15.93 |
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
| dlv3p_bbox_s43 | unet | all | 487 | 0.8409 | 0.5976 | 0.8789 | 2.65 |
| dlv3p_bbox_s43 | unet | area_<1% | 231 | 0.7481 | 0.4934 | 0.7749 | 3.06 |
| dlv3p_bbox_s43 | unet | area_1-5% | 254 | 0.9334 | 0.7185 | 0.9724 | 2.3 |
| dlv3p_bbox_s43 | unet | area_5-20% | 2 | 1.0 | 0.504 | 1.0 | 1.0 |
| dlv3p_bbox_s43 | unet | tool_gemini | 446 | 0.8678 | 0.6306 | 0.926 | 2.35 |
| dlv3p_bbox_s43 | unet | tool_ideogram | 41 | 0.5149 | 0.2429 | 0.5854 | 5.95 |
| dlv3p_bbox_s44 | unet | all | 487 | 0.8817 | 0.6028 | 0.885 | 3.69 |
| dlv3p_bbox_s44 | unet | area_<1% | 231 | 0.7915 | 0.5255 | 0.7879 | 4.84 |
| dlv3p_bbox_s44 | unet | area_1-5% | 254 | 0.9738 | 0.6843 | 0.9764 | 2.65 |
| dlv3p_bbox_s44 | unet | area_5-20% | 2 | 0.505 | 0.4545 | 0.5 | 2.5 |
| dlv3p_bbox_s44 | unet | tool_gemini | 446 | 0.9097 | 0.6377 | 0.9283 | 2.92 |
| dlv3p_bbox_s44 | unet | tool_ideogram | 41 | 0.4644 | 0.2084 | 0.4146 | 12.07 |

## T6. Dayanıklılık (relΔ pooled AUPRC, kendi GT'sinde)

| family | variant | train_gt | blur1 | clahe | gray | jpeg75 | rescale075 |
|---|---|---|---|---|---|---|---|
| DeepLabV3+ | main | box | -14.4% ± 2.4 | -0.6% ± 1.3 | -52.7% ± 5.7 | -13.5% ± 2.0 | -8.3% ± 1.3 |
| DeepLabV3+ | main | glyph | -33.8% ± 1.0 | -0.1% ± 0.1 | -57.9% ± 4.6 | -17.7% ± 1.7 | -21.5% ± 2.0 |
| DeepLabV3+ | main | pix | -49.5% ± 10.3 | +1.5% ± 2.0 | -63.6% ± 4.7 | -20.6% ± 2.7 | -35.1% ± 8.6 |
| U-Net | main | box | -20.7% ± 6.7 | — | — | -16.6% ± 1.4 | -12.2% ± 4.3 |
| U-Net | main | glyph | -59.0% ± 6.8 | — | — | -18.0% ± 0.8 | -42.7% ± 5.4 |
| U-Net | main | pix | -59.1% ± 4.5 | — | — | -22.5% ± 2.3 | -40.3% ± 1.5 |
| U-Net | marker-repaired | box | -25.9% ± 5.0 | -2.1% ± 0.8 | -59.6% ± 2.6 | -32.6% ± 1.8 | -14.0% ± 2.7 |
| U-Net | marker-repaired+nm-GT | glyph | -51.4% ± 6.0 | -0.9% ± 2.1 | -66.5% ± 2.6 | -36.6% ± 4.2 | -36.9% ± 6.9 |
| U-Net | marker-repaired+nm-GT | pix | -73.6% ± 4.5 | -0.3% ± 0.4 | -50.3% ± 10.1 | -50.4% ± 1.8 | -50.2% ± 3.1 |
| U-Net+box-aux | main | glyph | -43.3% ± 17.3 | -2.2% ± 1.3 | -62.2% ± 0.7 | -17.8% ± 0.5 | -31.0% ± 15.5 |
| U-Net+box-aux | main | pix | -46.0% ± 2.9 | -0.8% ± 0.8 | -66.3% ± 3.3 | -20.7% ± 2.0 | -30.8% ± 2.0 |
| YOLO11s | main | box | -1.3% ± 0.1 | -1.0% ± 0.2 | -57.5% ± 3.0 | -45.2% ± 4.0 | -0.6% ± 0.2 |

## T7. Kontroller (val; image-level AUROC forged vs kontrol; bayrak oranı @0.5 (>0.1 % piksel); FP@τ*)

| family | variant | train_gt | source | seeds | auroc_top1 | auroc_top1_sd | auroc_max | flag05 | fp_tau |
|---|---|---|---|---|---|---|---|---|---|
| DeepLabV3+ | main | box | authentic | 3 | 0.8372 | 0.0294 | 0.921 | 0.4778 | 0.0027 |
| DeepLabV3+ | main | box | benign_pasteback | 3 | 0.8125 | 0.0291 | 0.8987 | 0.538 | 0.0033 |
| DeepLabV3+ | main | box | fullimage_resample | 3 | 0.816 | 0.0296 | 0.9069 | 0.5441 | 0.0032 |
| DeepLabV3+ | main | glyph | authentic | 3 | 0.7503 | 0.023 | 0.8401 | 0.3518 | 0.0099 |
| DeepLabV3+ | main | glyph | benign_pasteback | 3 | 0.7287 | 0.0237 | 0.8189 | 0.3943 | 0.0103 |
| DeepLabV3+ | main | glyph | fullimage_resample | 3 | 0.722 | 0.0228 | 0.8221 | 0.4018 | 0.0107 |
| DeepLabV3+ | main | pix | authentic | 3 | 0.8276 | 0.0007 | 0.9116 | 0.2553 | 0.005 |
| DeepLabV3+ | main | pix | benign_pasteback | 3 | 0.8062 | 0.002 | 0.8971 | 0.3046 | 0.0054 |
| DeepLabV3+ | main | pix | fullimage_resample | 3 | 0.7903 | 0.013 | 0.8906 | 0.3457 | 0.0058 |
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
| U-Net | marker-repaired+nm-GT | glyph | authentic | 3 | 0.7794 | 0.0117 | 0.7214 | 0.2772 | 0.0044 |
| U-Net | marker-repaired+nm-GT | glyph | benign_pasteback | 3 | 0.7299 | 0.0062 | 0.6732 | 0.3607 | 0.0049 |
| U-Net | marker-repaired+nm-GT | glyph | fullimage_resample | 3 | 0.7295 | 0.0134 | 0.6677 | 0.3634 | 0.005 |
| U-Net | marker-repaired+nm-GT | pix | authentic | 3 | 0.8345 | 0.0144 | 0.8519 | 0.1889 | 0.0026 |
| U-Net | marker-repaired+nm-GT | pix | benign_pasteback | 3 | 0.7697 | 0.0149 | 0.7923 | 0.2895 | 0.0033 |
| U-Net | marker-repaired+nm-GT | pix | fullimage_resample | 3 | 0.78 | 0.0162 | 0.7964 | 0.2731 | 0.0034 |
| U-Net+box-aux | main | glyph | authentic | 3 | 0.8015 | 0.0282 | 0.7689 | 0.2683 | 0.0044 |
| U-Net+box-aux | main | glyph | benign_pasteback | 3 | 0.7847 | 0.0231 | 0.7485 | 0.2964 | 0.0046 |
| U-Net+box-aux | main | glyph | fullimage_resample | 3 | 0.7751 | 0.0327 | 0.7426 | 0.3135 | 0.0048 |
| U-Net+box-aux | main | pix | authentic | 3 | 0.9052 | 0.0075 | 0.9297 | 0.102 | 0.0018 |
| U-Net+box-aux | main | pix | benign_pasteback | 3 | 0.8884 | 0.0089 | 0.9163 | 0.1335 | 0.002 |
| U-Net+box-aux | main | pix | fullimage_resample | 3 | 0.8813 | 0.0138 | 0.9091 | 0.141 | 0.0022 |

## T8. Faz 1 AUPRC düzeltmesi (W6)

| variant | reported_phase1_auprc | reproduced_subsampled_auprc | exact_auprc_all_pixels | exact_auroc | pos_rate | n_crops | n_pixels |
|---|---|---|---|---|---|---|---|
| rgb | 0.947 | 0.9423 | 0.8053 | 0.9344 | 0.03306 | 200 | 52428800 |
| rgbfreq | 0.939 | 0.9331 | 0.7922 | 0.9255 | 0.03306 | 200 | 52428800 |

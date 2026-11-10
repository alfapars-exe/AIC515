# Faz 4 — ResNet-18 görüntü düzeyi sonuçlar (otomatik: calibrate_p4.py)

Seçilen agregasyon: **max** · val AUROC ortalamaları (ham, 3 tohum): {'max': 0.7856, 'top2': 0.7856, 'top3': 0.783, 'top5': 0.7622}


## val — aile × karar kuralı (ortalama ± std, tohumlar üzerinden)

| aile                              | rule         |   n_model |   auroc |   auroc_std |   tpr |   tpr_std |   fpr |   fpr_std |   bal_acc |
|:----------------------------------|:-------------|----------:|--------:|------------:|------:|----------:|------:|----------:|----------:|
| Faz 1 (resnet18_ft_best)          | eski_max_0.5 |         1 |   0.768 |         nan | 0.708 |       nan | 0.375 |       nan |     0.667 |
| Faz 1 (resnet18_ft_best)          | kalibre      |         1 |   0.768 |         nan | 0.5   |       nan | 0     |       nan |     0.75  |
| Faz 4 tam veri (ham)              | eski_max_0.5 |         1 |   0.786 |         nan | 0.708 |       nan | 0.333 |       nan |     0.688 |
| Faz 4 tam veri (ham)              | kalibre      |         1 |   0.786 |         nan | 0.583 |       nan | 0.167 |       nan |     0.708 |
| Faz 4 tam veri + işaretçi onarımı | eski_max_0.5 |         1 |   0.604 |         nan | 0.917 |       nan | 0.792 |       nan |     0.562 |
| Faz 4 tam veri + işaretçi onarımı | kalibre      |         1 |   0.604 |         nan | 0.708 |       nan | 0.542 |       nan |     0.583 |

## test — aile × karar kuralı (ortalama ± std, tohumlar üzerinden)

| aile                              | rule         |   n_model |   auroc |   auroc_std |   tpr |   tpr_std |   fpr |   fpr_std |   bal_acc |
|:----------------------------------|:-------------|----------:|--------:|------------:|------:|----------:|------:|----------:|----------:|
| Faz 1 (resnet18_ft_best)          | eski_max_0.5 |         1 |   0.896 |         nan | 0.875 |       nan | 0.375 |       nan |     0.75  |
| Faz 1 (resnet18_ft_best)          | kalibre      |         1 |   0.896 |         nan | 0.5   |       nan | 0     |       nan |     0.75  |
| Faz 4 tam veri (ham)              | eski_max_0.5 |         1 |   0.802 |         nan | 0.917 |       nan | 0.708 |       nan |     0.604 |
| Faz 4 tam veri (ham)              | kalibre      |         1 |   0.802 |         nan | 0.917 |       nan | 0.458 |       nan |     0.729 |
| Faz 4 tam veri + işaretçi onarımı | eski_max_0.5 |         1 |   0.686 |         nan | 0.917 |       nan | 0.958 |       nan |     0.479 |
| Faz 4 tam veri + işaretçi onarımı | kalibre      |         1 |   0.686 |         nan | 0.875 |       nan | 0.875 |       nan |     0.5   |

## Model başına (kalibre kural)

| model                          | rule    | agg   |   thr | split   |   n_pos |   n_neg |   tpr |   fpr |   bal_acc |   auroc |   tpr_lo |   tpr_hi |   fpr_lo |   fpr_hi |   auroc_lo |   auroc_hi |   paired_n |   paired_tpr |   paired_auroc |
|:-------------------------------|:--------|:------|------:|:--------|--------:|--------:|------:|------:|----------:|--------:|---------:|---------:|---------:|---------:|-----------:|-----------:|-----------:|-------------:|---------------:|
| resnet18_ft_best               | kalibre | max   | 0.996 | val     |      24 |      24 | 0.5   | 0     |     0.75  |   0.768 |    0.314 |    0.686 |    0     |    0.138 |    nan     |    nan     |        nan |          nan |        nan     |
| resnet18_ft_best               | kalibre | max   | 0.996 | test    |      24 |      24 | 0.5   | 0     |     0.75  |   0.896 |    0.314 |    0.686 |    0     |    0.138 |      0.798 |      0.971 |          1 |            0 |          0.958 |
| resnet18_ft_full_mr_s42_dryrun | kalibre | max   | 0.59  | val     |      24 |      24 | 0.708 | 0.542 |     0.583 |   0.604 |    0.508 |    0.851 |    0.351 |    0.721 |    nan     |    nan     |        nan |          nan |        nan     |
| resnet18_ft_full_mr_s42_dryrun | kalibre | max   | 0.59  | test    |      24 |      24 | 0.875 | 0.875 |     0.5   |   0.686 |    0.69  |    0.957 |    0.69  |    0.957 |      0.522 |      0.843 |          1 |            1 |          0.646 |
| resnet18_ft_full_s42_dryrun    | kalibre | max   | 0.575 | val     |      24 |      24 | 0.583 | 0.167 |     0.708 |   0.786 |    0.388 |    0.755 |    0.067 |    0.359 |    nan     |    nan     |        nan |          nan |        nan     |
| resnet18_ft_full_s42_dryrun    | kalibre | max   | 0.575 | test    |      24 |      24 | 0.917 | 0.458 |     0.729 |   0.802 |    0.742 |    0.977 |    0.279 |    0.649 |      0.668 |      0.917 |          1 |            1 |          0.542 |

## İşaretçi sondaları (val, kalıntılı kimlikler)

| aile                              |   n |   thr |   tpr_forged |   tpr_forged_repaired |   fpr_auth |   fpr_auth_marker_only |   auroc_forged_vs_auth |   auroc_repaired_vs_auth |   auroc_markeronly_vs_auth |   median_score_drop_on_repair |
|:----------------------------------|----:|------:|-------------:|----------------------:|-----------:|-----------------------:|-----------------------:|-------------------------:|---------------------------:|------------------------------:|
| Faz 1 (resnet18_ft_best)          |  11 | 0.996 |        0.909 |                 0.182 |      0     |                  0.727 |                  0.992 |                    0.744 |                      0.959 |                         0.2   |
| Faz 4 tam veri (ham)              |  11 | 0.575 |        0.909 |                 0.455 |      0.273 |                  0.727 |                  0.884 |                    0.595 |                      0.806 |                         0.289 |
| Faz 4 tam veri + işaretçi onarımı |  11 | 0.59  |        0.727 |                 0.636 |      0.636 |                  0.727 |                  0.554 |                    0.488 |                      0.545 |                         0     |

### Model başına

| model                          |   n |   thr |   tpr_forged |   tpr_forged_repaired |   fpr_auth |   fpr_auth_marker_only |   auroc_forged_vs_auth |   auroc_repaired_vs_auth |   auroc_markeronly_vs_auth |   median_score_drop_on_repair |
|:-------------------------------|----:|------:|-------------:|----------------------:|-----------:|-----------------------:|-----------------------:|-------------------------:|---------------------------:|------------------------------:|
| resnet18_ft_best               |  11 | 0.996 |        0.909 |                 0.182 |      0     |                  0.727 |                  0.992 |                    0.744 |                      0.959 |                         0.2   |
| resnet18_ft_full_mr_s42_dryrun |  11 | 0.59  |        0.727 |                 0.636 |      0.636 |                  0.727 |                  0.554 |                    0.488 |                      0.545 |                         0     |
| resnet18_ft_full_s42_dryrun    |  11 | 0.575 |        0.909 |                 0.455 |      0.273 |                  0.727 |                  0.884 |                    0.595 |                      0.806 |                         0.289 |

## Test kırılımları (kalibre kural; TPR)

| aile                              | by     | value                    |   n |   tpr |   tpr_std |   auroc |
|:----------------------------------|:-------|:-------------------------|----:|------:|----------:|--------:|
| Faz 1 (resnet18_ft_best)          | area   | 1-5%                     |   1 | 0     |       nan | nan     |
| Faz 1 (resnet18_ft_best)          | area   | <1%                      |  23 | 0.522 |       nan |   0.909 |
| Faz 1 (resnet18_ft_best)          | clean  | eğitimle kaynak paylaşan |   3 | 0.667 |       nan |   0.986 |
| Faz 1 (resnet18_ft_best)          | clean  | temiz test               |  21 | 0.476 |       nan |   0.883 |
| Faz 1 (resnet18_ft_best)          | marker | işaretçili               |  11 | 0.818 |       nan |   0.985 |
| Faz 1 (resnet18_ft_best)          | marker | işaretçisiz              |  13 | 0.231 |       nan |   0.821 |
| Faz 1 (resnet18_ft_best)          | tool   | gemini                   |  24 | 0.5   |       nan |   0.896 |
| Faz 4 tam veri (ham)              | area   | 1-5%                     |   1 | 1     |       nan | nan     |
| Faz 4 tam veri (ham)              | area   | <1%                      |  23 | 0.913 |       nan |   0.808 |
| Faz 4 tam veri (ham)              | clean  | eğitimle kaynak paylaşan |   3 | 1     |       nan |   0.819 |
| Faz 4 tam veri (ham)              | clean  | temiz test               |  21 | 0.905 |       nan |   0.8   |
| Faz 4 tam veri (ham)              | marker | işaretçili               |  11 | 0.818 |       nan |   0.765 |
| Faz 4 tam veri (ham)              | marker | işaretçisiz              |  13 | 1     |       nan |   0.833 |
| Faz 4 tam veri (ham)              | tool   | gemini                   |  24 | 0.917 |       nan |   0.802 |
| Faz 4 tam veri + işaretçi onarımı | area   | 1-5%                     |   1 | 1     |       nan | nan     |
| Faz 4 tam veri + işaretçi onarımı | area   | <1%                      |  23 | 0.87  |       nan |   0.69  |
| Faz 4 tam veri + işaretçi onarımı | clean  | eğitimle kaynak paylaşan |   3 | 1     |       nan |   0.743 |
| Faz 4 tam veri + işaretçi onarımı | clean  | temiz test               |  21 | 0.857 |       nan |   0.678 |
| Faz 4 tam veri + işaretçi onarımı | marker | işaretçili               |  11 | 0.727 |       nan |   0.544 |
| Faz 4 tam veri + işaretçi onarımı | marker | işaretçisiz              |  13 | 1     |       nan |   0.806 |
| Faz 4 tam veri + işaretçi onarımı | tool   | gemini                   |  24 | 0.875 |       nan |   0.686 |

## Val — tüm agregasyonlar

| model                          | agg   |   val_auroc |    thr |   val_tpr |   val_fpr |   youden_j |   n_pos |   n_neg |
|:-------------------------------|:------|------------:|-------:|----------:|----------:|-----------:|--------:|--------:|
| resnet18_ft_best               | max   |      0.7682 | 0.9962 |    0.5    |    0      |     0.5    |      24 |      24 |
| resnet18_ft_best               | top2  |      0.7622 | 0.715  |    0.5833 |    0.0417 |     0.5417 |      24 |      24 |
| resnet18_ft_best               | top3  |      0.7682 | 0.6795 |    0.5833 |    0.0417 |     0.5417 |      24 |      24 |
| resnet18_ft_best               | top5  |      0.7543 | 0.3037 |    0.6667 |    0.125  |     0.5417 |      24 |      24 |
| resnet18_ft_full_mr_s42_dryrun | max   |      0.6042 | 0.5898 |    0.7083 |    0.5417 |     0.1667 |      24 |      24 |
| resnet18_ft_full_mr_s42_dryrun | top2  |      0.6311 | 0.5218 |    0.875  |    0.625  |     0.25   |      24 |      24 |
| resnet18_ft_full_mr_s42_dryrun | top3  |      0.6293 | 0.5622 |    0.7083 |    0.4583 |     0.25   |      24 |      24 |
| resnet18_ft_full_mr_s42_dryrun | top5  |      0.6068 | 0.539  |    0.6667 |    0.4583 |     0.2083 |      24 |      24 |
| resnet18_ft_full_s42_dryrun    | max   |      0.7856 | 0.5746 |    0.5833 |    0.1667 |     0.4167 |      24 |      24 |
| resnet18_ft_full_s42_dryrun    | top2  |      0.7856 | 0.6023 |    0.5    |    0.0833 |     0.4167 |      24 |      24 |
| resnet18_ft_full_s42_dryrun    | top3  |      0.783  | 0.5579 |    0.5    |    0.0833 |     0.4167 |      24 |      24 |
| resnet18_ft_full_s42_dryrun    | top5  |      0.7622 | 0.404  |    0.75   |    0.3333 |     0.4167 |      24 |      24 |
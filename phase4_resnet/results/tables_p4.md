# Faz 4 — ResNet-18 görüntü düzeyi sonuçlar (otomatik: calibrate_p4.py)

Seçilen agregasyon: **max** · val AUROC ortalamaları (ham, 3 tohum): {'max': 0.9655, 'top2': 0.962, 'top3': 0.9579, 'top5': 0.9508}


## val — aile × karar kuralı (ortalama ± std, tohumlar üzerinden)

| aile                              | rule         |   n_model |   auroc |   auroc_std |   tpr |   tpr_std |   fpr |   fpr_std |   bal_acc |
|:----------------------------------|:-------------|----------:|--------:|------------:|------:|----------:|------:|----------:|----------:|
| Faz 1 (resnet18_ft_best)          | eski_max_0.5 |         1 |   0.893 |     nan     | 0.916 |   nan     | 0.456 |   nan     |     0.73  |
| Faz 1 (resnet18_ft_best)          | kalibre      |         1 |   0.893 |     nan     | 0.791 |   nan     | 0.101 |   nan     |     0.845 |
| Faz 4 tam veri (ham)              | eski_max_0.5 |         3 |   0.965 |       0.002 | 0.951 |     0.005 | 0.242 |     0.048 |     0.855 |
| Faz 4 tam veri (ham)              | kalibre      |         3 |   0.965 |       0.002 | 0.886 |     0.003 | 0.021 |     0.011 |     0.933 |
| Faz 4 tam veri + işaretçi onarımı | eski_max_0.5 |         3 |   0.948 |       0.002 | 0.958 |     0.008 | 0.374 |     0.016 |     0.792 |
| Faz 4 tam veri + işaretçi onarımı | kalibre      |         3 |   0.948 |       0.002 | 0.84  |     0.018 | 0.061 |     0.017 |     0.889 |

## test — aile × karar kuralı (ortalama ± std, tohumlar üzerinden)

| aile                              | rule         |   n_model |   auroc |   auroc_std |   tpr |   tpr_std |   fpr |   fpr_std |   bal_acc |
|:----------------------------------|:-------------|----------:|--------:|------------:|------:|----------:|------:|----------:|----------:|
| Faz 1 (resnet18_ft_best)          | eski_max_0.5 |         1 |   0.919 |     nan     | 0.93  |   nan     | 0.404 |   nan     |     0.763 |
| Faz 1 (resnet18_ft_best)          | kalibre      |         1 |   0.919 |     nan     | 0.805 |   nan     | 0.079 |   nan     |     0.863 |
| Faz 4 tam veri (ham)              | eski_max_0.5 |         3 |   0.978 |       0.003 | 0.963 |     0.007 | 0.109 |     0.034 |     0.927 |
| Faz 4 tam veri (ham)              | kalibre      |         3 |   0.978 |       0.003 | 0.901 |     0.014 | 0.022 |     0.011 |     0.94  |
| Faz 4 tam veri + işaretçi onarımı | eski_max_0.5 |         3 |   0.958 |       0.004 | 0.967 |     0.004 | 0.345 |     0.034 |     0.811 |
| Faz 4 tam veri + işaretçi onarımı | kalibre      |         3 |   0.958 |       0.004 | 0.834 |     0.011 | 0.064 |     0.017 |     0.885 |

## Model başına (kalibre kural)

| model                   | rule    | agg   |   thr | split   |   n_pos |   n_neg |   tpr |   fpr |   bal_acc |   auroc |   tpr_lo |   tpr_hi |   fpr_lo |   fpr_hi |   auroc_lo |   auroc_hi |   paired_n |   paired_tpr |   paired_auroc |
|:------------------------|:--------|:------|------:|:--------|--------:|--------:|------:|------:|----------:|--------:|---------:|---------:|---------:|---------:|-----------:|-----------:|-----------:|-------------:|---------------:|
| resnet18_ft_best        | kalibre | max   | 0.938 | val     |     487 |     487 | 0.791 | 0.101 |     0.845 |   0.893 |    0.752 |    0.824 |    0.077 |    0.131 |    nan     |    nan     |        nan |      nan     |        nan     |
| resnet18_ft_best        | kalibre | max   | 0.938 | test    |     812 |      89 | 0.805 | 0.079 |     0.863 |   0.919 |    0.777 |    0.831 |    0.039 |    0.154 |      0.894 |      0.939 |         89 |        0.876 |          0.938 |
| resnet18_ft_full_mr_s42 | kalibre | max   | 0.965 | val     |     487 |     487 | 0.856 | 0.08  |     0.888 |   0.949 |    0.822 |    0.885 |    0.059 |    0.108 |    nan     |    nan     |        nan |      nan     |        nan     |
| resnet18_ft_full_mr_s42 | kalibre | max   | 0.965 | test    |     812 |      89 | 0.844 | 0.079 |     0.882 |   0.96  |    0.817 |    0.867 |    0.039 |    0.154 |      0.944 |      0.974 |         89 |        0.719 |          0.933 |
| resnet18_ft_full_mr_s43 | kalibre | max   | 0.981 | val     |     487 |     487 | 0.842 | 0.053 |     0.894 |   0.949 |    0.807 |    0.872 |    0.037 |    0.077 |    nan     |    nan     |        nan |      nan     |        nan     |
| resnet18_ft_full_mr_s43 | kalibre | max   | 0.981 | test    |     812 |      89 | 0.836 | 0.045 |     0.896 |   0.959 |    0.809 |    0.86  |    0.018 |    0.11  |      0.943 |      0.972 |         89 |        0.775 |          0.926 |
| resnet18_ft_full_mr_s44 | kalibre | max   | 0.982 | val     |     487 |     487 | 0.821 | 0.049 |     0.886 |   0.946 |    0.785 |    0.853 |    0.033 |    0.072 |    nan     |    nan     |        nan |      nan     |        nan     |
| resnet18_ft_full_mr_s44 | kalibre | max   | 0.982 | test    |     812 |      89 | 0.821 | 0.067 |     0.877 |   0.953 |    0.794 |    0.846 |    0.031 |    0.139 |      0.934 |      0.969 |         89 |        0.663 |          0.91  |
| resnet18_ft_full_s42    | kalibre | max   | 0.995 | val     |     487 |     487 | 0.883 | 0.01  |     0.936 |   0.965 |    0.851 |    0.909 |    0.004 |    0.024 |    nan     |    nan     |        nan |      nan     |        nan     |
| resnet18_ft_full_s42    | kalibre | max   | 0.995 | test    |     812 |      89 | 0.889 | 0.022 |     0.933 |   0.977 |    0.866 |    0.909 |    0.006 |    0.078 |      0.96  |      0.989 |         89 |        0.933 |          0.981 |
| resnet18_ft_full_s43    | kalibre | max   | 0.94  | val     |     487 |     487 | 0.889 | 0.033 |     0.928 |   0.968 |    0.858 |    0.914 |    0.02  |    0.053 |    nan     |    nan     |        nan |      nan     |        nan     |
| resnet18_ft_full_s43    | kalibre | max   | 0.94  | test    |     812 |      89 | 0.916 | 0.034 |     0.941 |   0.976 |    0.895 |    0.933 |    0.012 |    0.094 |      0.963 |      0.986 |         89 |        0.955 |          0.98  |
| resnet18_ft_full_s44    | kalibre | max   | 0.992 | val     |     487 |     487 | 0.887 | 0.018 |     0.934 |   0.964 |    0.856 |    0.912 |    0.01  |    0.035 |    nan     |    nan     |        nan |      nan     |        nan     |
| resnet18_ft_full_s44    | kalibre | max   | 0.992 | test    |     812 |      89 | 0.899 | 0.011 |     0.944 |   0.981 |    0.876 |    0.918 |    0.002 |    0.061 |      0.971 |      0.988 |         89 |        0.933 |          0.982 |

## İşaretçi sondaları (val, kalıntılı kimlikler)

| aile                              |   n |   thr |   tpr_forged |   tpr_forged_repaired |   fpr_auth |   fpr_auth_marker_only |   auroc_forged_vs_auth |   auroc_repaired_vs_auth |   auroc_markeronly_vs_auth |   median_score_drop_on_repair |
|:----------------------------------|----:|------:|-------------:|----------------------:|-----------:|-----------------------:|-----------------------:|-------------------------:|---------------------------:|------------------------------:|
| Faz 1 (resnet18_ft_best)          | 368 | 0.938 |        0.91  |                 0.207 |      0.073 |                  0.848 |                  0.964 |                    0.586 |                      0.945 |                         0.36  |
| Faz 4 tam veri (ham)              | 368 | 0.976 |        0.995 |                 0.471 |      0.016 |                  0.955 |                  0.999 |                    0.772 |                      0.995 |                         0.029 |
| Faz 4 tam veri + işaretçi onarımı | 368 | 0.976 |        0.88  |                 0.899 |      0.049 |                  0.458 |                  0.967 |                    0.974 |                      0.797 |                        -0     |

### Model başına

| model                   |   n |   thr |   tpr_forged |   tpr_forged_repaired |   fpr_auth |   fpr_auth_marker_only |   auroc_forged_vs_auth |   auroc_repaired_vs_auth |   auroc_markeronly_vs_auth |   median_score_drop_on_repair |
|:------------------------|----:|------:|-------------:|----------------------:|-----------:|-----------------------:|-----------------------:|-------------------------:|---------------------------:|------------------------------:|
| resnet18_ft_best        | 368 | 0.938 |        0.91  |                 0.207 |      0.073 |                  0.848 |                  0.964 |                    0.586 |                      0.945 |                         0.36  |
| resnet18_ft_full_mr_s42 | 368 | 0.965 |        0.886 |                 0.905 |      0.06  |                  0.454 |                  0.966 |                    0.974 |                      0.787 |                        -0     |
| resnet18_ft_full_mr_s43 | 368 | 0.981 |        0.883 |                 0.894 |      0.046 |                  0.459 |                  0.967 |                    0.973 |                      0.805 |                        -0     |
| resnet18_ft_full_mr_s44 | 368 | 0.982 |        0.872 |                 0.899 |      0.041 |                  0.462 |                  0.966 |                    0.975 |                      0.799 |                         0     |
| resnet18_ft_full_s42    | 368 | 0.995 |        0.989 |                 0.459 |      0.008 |                  0.935 |                  0.998 |                    0.78  |                      0.995 |                         0.009 |
| resnet18_ft_full_s43    | 368 | 0.94  |        1     |                 0.503 |      0.035 |                  0.973 |                  0.999 |                    0.773 |                      0.993 |                         0.056 |
| resnet18_ft_full_s44    | 368 | 0.992 |        0.995 |                 0.451 |      0.005 |                  0.957 |                  0.999 |                    0.764 |                      0.996 |                         0.023 |

## Test kırılımları (kalibre kural; TPR)

| aile                              | by     | value                    |   n |   tpr |   tpr_std |   auroc |
|:----------------------------------|:-------|:-------------------------|----:|------:|----------:|--------:|
| Faz 1 (resnet18_ft_best)          | area   | 1-5%                     | 381 | 0.843 |   nan     |   0.923 |
| Faz 1 (resnet18_ft_best)          | area   | 5-20%                    |   2 | 1     |   nan     |   0.989 |
| Faz 1 (resnet18_ft_best)          | area   | <1%                      | 429 | 0.772 |   nan     |   0.915 |
| Faz 1 (resnet18_ft_best)          | clean  | eğitimle kaynak paylaşan |  72 | 0.861 |   nan     |   0.943 |
| Faz 1 (resnet18_ft_best)          | clean  | temiz test               | 740 | 0.8   |   nan     |   0.916 |
| Faz 1 (resnet18_ft_best)          | marker | işaretçili               | 627 | 0.906 |   nan     |   0.954 |
| Faz 1 (resnet18_ft_best)          | marker | işaretçisiz              | 185 | 0.465 |   nan     |   0.8   |
| Faz 1 (resnet18_ft_best)          | tool   | gemini                   | 727 | 0.839 |   nan     |   0.926 |
| Faz 1 (resnet18_ft_best)          | tool   | ideogram                 |  85 | 0.518 |   nan     |   0.86  |
| Faz 4 tam veri (ham)              | area   | 1-5%                     | 381 | 0.96  |     0.005 |   0.987 |
| Faz 4 tam veri (ham)              | area   | 5-20%                    |   2 | 1     |     0     |   1     |
| Faz 4 tam veri (ham)              | area   | <1%                      | 429 | 0.849 |     0.021 |   0.969 |
| Faz 4 tam veri (ham)              | clean  | eğitimle kaynak paylaşan |  72 | 0.94  |     0.021 |   0.982 |
| Faz 4 tam veri (ham)              | clean  | temiz test               | 740 | 0.898 |     0.013 |   0.977 |
| Faz 4 tam veri (ham)              | marker | işaretçili               | 627 | 0.985 |     0.004 |   0.994 |
| Faz 4 tam veri (ham)              | marker | işaretçisiz              | 185 | 0.618 |     0.047 |   0.92  |
| Faz 4 tam veri (ham)              | tool   | gemini                   | 727 | 0.934 |     0.012 |   0.982 |
| Faz 4 tam veri (ham)              | tool   | ideogram                 |  85 | 0.624 |     0.042 |   0.942 |
| Faz 4 tam veri + işaretçi onarımı | area   | 1-5%                     | 381 | 0.847 |     0.02  |   0.961 |
| Faz 4 tam veri + işaretçi onarımı | area   | 5-20%                    |   2 | 0.833 |     0.289 |   0.981 |
| Faz 4 tam veri + işaretçi onarımı | area   | <1%                      | 429 | 0.822 |     0.009 |   0.954 |
| Faz 4 tam veri + işaretçi onarımı | clean  | eğitimle kaynak paylaşan |  72 | 0.907 |     0.008 |   0.969 |
| Faz 4 tam veri + işaretçi onarımı | clean  | temiz test               | 740 | 0.827 |     0.013 |   0.956 |
| Faz 4 tam veri + işaretçi onarımı | marker | işaretçili               | 627 | 0.863 |     0.008 |   0.966 |
| Faz 4 tam veri + işaretçi onarımı | marker | işaretçisiz              | 185 | 0.735 |     0.029 |   0.928 |
| Faz 4 tam veri + işaretçi onarımı | tool   | gemini                   | 727 | 0.845 |     0.011 |   0.961 |
| Faz 4 tam veri + işaretçi onarımı | tool   | ideogram                 |  85 | 0.737 |     0.024 |   0.929 |

## Val — tüm agregasyonlar

| model                   | agg   |   val_auroc |    thr |   val_tpr |   val_fpr |   youden_j |   n_pos |   n_neg |
|:------------------------|:------|------------:|-------:|----------:|----------:|-----------:|--------:|--------:|
| resnet18_ft_best        | max   |      0.8931 | 0.9381 |    0.7906 |    0.1006 |     0.6899 |     487 |     487 |
| resnet18_ft_best        | top2  |      0.8901 | 0.8683 |    0.7967 |    0.1294 |     0.6674 |     487 |     487 |
| resnet18_ft_best        | top3  |      0.8783 | 0.8857 |    0.7084 |    0.0903 |     0.6181 |     487 |     487 |
| resnet18_ft_best        | top5  |      0.8526 | 0.54   |    0.8111 |    0.2464 |     0.5647 |     487 |     487 |
| resnet18_ft_full_mr_s42 | max   |      0.9491 | 0.9654 |    0.8563 |    0.0801 |     0.7762 |     487 |     487 |
| resnet18_ft_full_mr_s42 | top2  |      0.9461 | 0.9041 |    0.848  |    0.0842 |     0.7639 |     487 |     487 |
| resnet18_ft_full_mr_s42 | top3  |      0.9377 | 0.7428 |    0.8583 |    0.1314 |     0.7269 |     487 |     487 |
| resnet18_ft_full_mr_s42 | top5  |      0.9206 | 0.5561 |    0.8419 |    0.1561 |     0.6858 |     487 |     487 |
| resnet18_ft_full_mr_s43 | max   |      0.9488 | 0.9812 |    0.8419 |    0.0534 |     0.7885 |     487 |     487 |
| resnet18_ft_full_mr_s43 | top2  |      0.9469 | 0.9595 |    0.8131 |    0.039  |     0.7741 |     487 |     487 |
| resnet18_ft_full_mr_s43 | top3  |      0.9408 | 0.9189 |    0.77   |    0.039  |     0.731  |     487 |     487 |
| resnet18_ft_full_mr_s43 | top5  |      0.9262 | 0.5464 |    0.8522 |    0.1437 |     0.7084 |     487 |     487 |
| resnet18_ft_full_mr_s44 | max   |      0.9461 | 0.982  |    0.8214 |    0.0493 |     0.7721 |     487 |     487 |
| resnet18_ft_full_mr_s44 | top2  |      0.9451 | 0.9214 |    0.8234 |    0.0637 |     0.7598 |     487 |     487 |
| resnet18_ft_full_mr_s44 | top3  |      0.9378 | 0.7626 |    0.846  |    0.117  |     0.729  |     487 |     487 |
| resnet18_ft_full_mr_s44 | top5  |      0.9247 | 0.4084 |    0.9158 |    0.2156 |     0.7002 |     487 |     487 |
| resnet18_ft_full_s42    | max   |      0.9647 | 0.9951 |    0.883  |    0.0103 |     0.8727 |     487 |     487 |
| resnet18_ft_full_s42    | top2  |      0.9613 | 0.9564 |    0.883  |    0.0246 |     0.8583 |     487 |     487 |
| resnet18_ft_full_s42    | top3  |      0.9571 | 0.6667 |    0.9158 |    0.0924 |     0.8234 |     487 |     487 |
| resnet18_ft_full_s42    | top5  |      0.949  | 0.5355 |    0.8809 |    0.0862 |     0.7947 |     487 |     487 |
| resnet18_ft_full_s43    | max   |      0.9679 | 0.9399 |    0.8891 |    0.0329 |     0.8563 |     487 |     487 |
| resnet18_ft_full_s43    | top2  |      0.9656 | 0.808  |    0.8768 |    0.0308 |     0.846  |     487 |     487 |
| resnet18_ft_full_s43    | top3  |      0.9627 | 0.6571 |    0.8747 |    0.0431 |     0.8316 |     487 |     487 |
| resnet18_ft_full_s43    | top5  |      0.9569 | 0.4001 |    0.8871 |    0.076  |     0.8111 |     487 |     487 |
| resnet18_ft_full_s44    | max   |      0.9638 | 0.9922 |    0.8871 |    0.0185 |     0.8686 |     487 |     487 |
| resnet18_ft_full_s44    | top2  |      0.9591 | 0.9415 |    0.8789 |    0.0287 |     0.8501 |     487 |     487 |
| resnet18_ft_full_s44    | top3  |      0.9537 | 0.7555 |    0.8706 |    0.0678 |     0.8029 |     487 |     487 |
| resnet18_ft_full_s44    | top5  |      0.9465 | 0.5155 |    0.883  |    0.1027 |     0.7803 |     487 |     487 |
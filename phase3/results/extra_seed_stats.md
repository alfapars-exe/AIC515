# Ek tohum istatistikleri (otomatik; seed_extra_stats.py)

## 1. Tohum bazında havuzlanmış AUPRC

| model     | split   | subset     | gt    |    s42 |    s43 |    s44 |   mean |    sd |
|:----------|:--------|:-----------|:------|-------:|-------:|-------:|-------:|------:|
| std_box   | val     | all        | box   | 0.9334 | 0.9296 | 0.9061 |  0.923 | 0.015 |
| std_box   | val     | all        | pix   | 0.3772 | 0.3811 | 0.3652 |  0.374 | 0.008 |
| std_box   | val     | all        | glyph | 0.2567 | 0.2644 | 0.2514 |  0.257 | 0.007 |
| std_box   | test    | all        | box   | 0.9571 | 0.9557 | 0.9368 |  0.95  | 0.011 |
| std_box   | test    | clean_test | box   | 0.9563 | 0.9558 | 0.9359 |  0.949 | 0.012 |
| std_box   | test    | paired_89  | pix   | 0.4292 | 0.4257 | 0.4264 |  0.427 | 0.002 |
| mr_box    | val     | all        | box   | 0.9163 | 0.916  | 0.9052 |  0.912 | 0.006 |
| mr_box    | val     | all        | pix   | 0.3886 | 0.3689 | 0.3629 |  0.373 | 0.013 |
| mr_box    | val     | all        | glyph | 0.2664 | 0.2639 | 0.2658 |  0.265 | 0.001 |
| mr_box    | test    | all        | box   | 0.9392 | 0.9431 | 0.9373 |  0.94  | 0.003 |
| mr_box    | test    | clean_test | box   | 0.939  | 0.9421 | 0.9376 |  0.94  | 0.002 |
| mr_box    | test    | paired_89  | pix   | 0.4253 | 0.3729 | 0.3882 |  0.395 | 0.027 |
| std_pix   | val     | all        | box   | 0.5897 | 0.5639 | 0.5838 |  0.579 | 0.014 |
| std_pix   | val     | all        | pix   | 0.5959 | 0.6158 | 0.6174 |  0.61  | 0.012 |
| std_pix   | val     | all        | glyph | 0.5002 | 0.5125 | 0.5203 |  0.511 | 0.01  |
| std_pix   | test    | all        | box   | 0.6551 | 0.6268 | 0.6336 |  0.638 | 0.015 |
| std_pix   | test    | clean_test | box   | 0.6555 | 0.6292 | 0.6338 |  0.64  | 0.014 |
| std_pix   | test    | paired_89  | pix   | 0.645  | 0.6725 | 0.6443 |  0.654 | 0.016 |
| joint_pix | val     | all        | box   | 0.6468 | 0.6754 | 0.6825 |  0.668 | 0.019 |
| joint_pix | val     | all        | pix   | 0.6218 | 0.6476 | 0.6304 |  0.633 | 0.013 |
| joint_pix | val     | all        | glyph | 0.5177 | 0.5333 | 0.5169 |  0.523 | 0.009 |
| joint_pix | test    | all        | box   | 0.6823 | 0.7024 | 0.7223 |  0.702 | 0.02  |
| joint_pix | test    | clean_test | box   | 0.6803 | 0.7042 | 0.7214 |  0.702 | 0.021 |
| joint_pix | test    | paired_89  | pix   | 0.6503 | 0.655  | 0.6501 |  0.652 | 0.003 |

## 2. Eşli görüntü başı AP farkı (tohum ortalamalı, grup-bootstrap %95 GA)

| karşılaştırma                          | bölme   | gt   |   n |   grup |   AP_a |   AP_b | fark                    |
|:---------------------------------------|:--------|:-----|----:|-------:|-------:|-------:|:------------------------|
| birleşik piksel − standart piksel      | val     | box  | 487 |    469 |  0.639 |  0.588 | +0.051 [+0.043, +0.058] |
| birleşik piksel − standart piksel      | val     | pix  | 485 |    467 |  0.59  |  0.579 | +0.011 [+0.005, +0.017] |
| birleşik piksel − standart piksel      | test    | box  | 812 |    795 |  0.654 |  0.608 | +0.046 [+0.040, +0.052] |
| birleşik piksel − standart piksel      | test    | pix  |  88 |     84 |  0.589 |  0.585 | +0.004 [-0.008, +0.015] |
| işaretçi-onarımlı kutu − standart kutu | val     | box  | 487 |    469 |  0.858 |  0.896 | -0.038 [-0.054, -0.021] |
| işaretçi-onarımlı kutu − standart kutu | val     | pix  | 485 |    467 |  0.372 |  0.377 | -0.005 [-0.015, +0.005] |
| işaretçi-onarımlı kutu − standart kutu | test    | box  | 812 |    795 |  0.868 |  0.908 | -0.040 [-0.052, -0.029] |
| işaretçi-onarımlı kutu − standart kutu | test    | pix  |  88 |     84 |  0.317 |  0.37  | -0.053 [-0.085, -0.029] |

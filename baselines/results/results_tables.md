### B) Klasik öznitelikler + ML
| Model | Acc | Bal.Acc | AUROC | AUPRC | F1@0.5 | AUROC tamp-vs-paired | AUROC tamp-vs-same | AUROC gemini | AUROC ideogram | AUROC <1% | AUROC 1-5% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SVM-RBF (C=0.5, γ=0.01) | 0.534 | 0.606 | 0.646 | 0.443 | 0.546 | 0.552 | 0.744 | 0.648 | 0.641 | 0.643 | 0.648 |
| RandomForest (500 ağaç) | 0.622 | 0.653 | 0.706 | 0.536 | 0.573 | 0.610 | 0.807 | 0.696 | 0.738 | 0.732 | 0.685 |

### C) ResNet18
| Ayar | best ep / çalışan | Acc | Bal.Acc | AUROC | AUPRC | F1@0.5 | AUROC tamp-vs-paired | AUROC tamp-vs-same | AUROC gemini | AUROC ideogram | AUROC <1% | AUROC 1-5% | TN/FP/FN/TP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ResNet18 probe (pretrained, dondurulmuş + yeni FC) | 10/12 | 0.639 | 0.693 | 0.726 | 0.554 | 0.616 | 0.635 | 0.820 | 0.735 | 0.698 | 0.680 | 0.767 | 103/93/14/86 |
| ResNet18 full fine-tune (pretrained) | 6/11 | 0.885 | 0.845 | 0.899 | 0.884 | 0.809 | 0.878 | 0.922 | 0.910 | 0.867 | 0.883 | 0.913 | 190/6/28/72 |
| ResNet18 scratch (rastgele init) | 4/9 | 0.503 | 0.610 | 0.679 | 0.497 | 0.561 | 0.573 | 0.790 | 0.670 | 0.708 | 0.656 | 0.707 | 55/141/6/94 |

### D) U-Net
| Varyant | best ep / çalışan | Pixel-F1@0.5 | IoU@0.5 | Oracle-F1 | Global-eşik F1 (eşik) | Pixel-AUPRC | Pixel-AUROC | F1@0.5 (yalnız bbox crop) | IoU (bbox crop) | Temiz crop FP piksel oranı | Temiz crop'ların >%1 FP'li payı |
|---|---|---|---|---|---|---|---|---|---|---|---|
| U-Net RGB (D1, 3 kanal) | 11/15 | 0.707 | 0.637 | 0.740 | 0.708 (0.45) | 0.947 | 0.948 | 0.724 | 0.653 | 0.26% | 8.5% |
| U-Net RGB+FFT-HP+Laplacian (D2, 5 kanal) | 12/15 | 0.674 | 0.613 | 0.711 | 0.674 (0.45) | 0.939 | 0.940 | 0.696 | 0.635 | 0.14% | 3.4% |

| Varyant | F1 gemini | F1 ideogram | F1 <1% | F1 1-5% | F1 5-20% | F1 cord | F1 sroie | F1 wildreceipt | F1 xfund |
|---|---|---|---|---|---|---|---|---|---|
| U-Net RGB (D1, 3 kanal) | 0.800 (n=75) | 0.497 (n=25) | 0.606 (n=47) | 0.830 (n=52) | 0.756 (n=1) | 0.478 | 0.869 | 0.765 | 0.601 |
| U-Net RGB+FFT-HP+Laplacian (D2, 5 kanal) | 0.779 (n=75) | 0.449 (n=25) | 0.589 (n=47) | 0.792 (n=52) | 0.788 (n=1) | 0.474 | 0.839 | 0.772 | 0.475 |

### E) Dayanıklılık
| Varyant | Koşul | Pixel-F1@0.5 | IoU@0.5 | Oracle-F1 | Pixel-AUPRC | Pixel-AUROC | Temiz crop FP | F1 gemini | F1 ideogram |
|---|---|---|---|---|---|---|---|---|---|
| U-Net RGB (D1, 3 kanal) | clean (PNG) | 0.708 | 0.637 | 0.740 | 0.947 | 0.948 | 0.26% | 0.800 | 0.497 |
| U-Net RGB (D1, 3 kanal) | JPEG Q75 | 0.636 | 0.565 | 0.680 | 0.929 | 0.932 | 0.18% | 0.728 | 0.444 |
| U-Net RGB (D1, 3 kanal) | rescale 0.75x→1x | 0.638 | 0.560 | 0.676 | 0.939 | 0.943 | 1.24% | 0.734 | 0.421 |
| U-Net RGB+FFT-HP+Laplacian (D2, 5 kanal) | clean (PNG) | 0.674 | 0.613 | 0.711 | 0.939 | 0.940 | 0.14% | 0.779 | 0.449 |
| U-Net RGB+FFT-HP+Laplacian (D2, 5 kanal) | JPEG Q75 | 0.604 | 0.540 | 0.640 | 0.912 | 0.917 | 0.13% | 0.715 | 0.328 |
| U-Net RGB+FFT-HP+Laplacian (D2, 5 kanal) | rescale 0.75x→1x | 0.633 | 0.566 | 0.674 | 0.924 | 0.929 | 0.42% | 0.755 | 0.353 |

### F) Kaynak
| Model | Eğitim süresi (s) | Peak GPU bellek (GiB) | Parametre | Çıkarım (ms/örnek) | Checkpoint (MB) | Not |
|---|---|---|---|---|---|---|
| svm_rbf | 7 | 0.00 | — | 27.9 | — | CPU; çıkarım süresi = öznitelik çıkarımı/patch |
| random_forest | 6 | 0.00 | — | 27.9 | — | CPU; çıkarım süresi = öznitelik çıkarımı/patch |
| resnet18_probe | 64 | 0.33 | 11.18 M | 8.2 | 42.7 | 12/12 epoch, 224 patch, batch 1 bf16 |
| resnet18_ft | 76 | 0.59 | 11.18 M | 7.3 | 42.7 | 11/12 epoch, 224 patch, batch 1 bf16 |
| resnet18_scratch | 65 | 0.59 | 11.18 M | 7.5 | 42.7 | 9/12 epoch, 224 patch, batch 1 bf16 |
| unet_rgb | 603 | 1.70 | 14.33 M | 15.8 | 54.8 | 15/15 epoch, 512 crop, batch 1 bf16 |
| unet_rgbfreq | 601 | 1.75 | 14.33 M | 18.8 | 54.8 | 15/15 epoch, 512 crop, batch 1 bf16 |

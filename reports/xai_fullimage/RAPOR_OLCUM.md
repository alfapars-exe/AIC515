# Tam görüntü XAI ölçümleri (2026-11-10 12:12, NVIDIA GeForce RTX 5080, max_evals=500)

## SHAP (tam görüntü karar hedefi + döşemeli mozaik)


| görüntü | boyut | model | hedef | harita = görüntü | f(x) | uygulama skoru | E[f] | ΣSHAP | toplamsallık hatası | süre (s) | tepe GPU (MB) | GT içi + kütle | not |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| testing:000000180 | 432x648 | PHASE2/unet_bbox_s42 | full_image_score | ✓ | 0.4803 | 0.4797 | 0.0007 | 0.4796 | 5.6e-17 | 17.6 | 3020 | 0.0 | ızgara 108x162 (4 px) |
| testing:000000180 | 432x648 | PHASE1/unet_rgbfreq_best | full_image_score | ✓ | 0.1023 | 0.1023 | 0.0499 | 0.0524 | 5.6e-17 | 28.8 | 3517 | 0.045 | ızgara 108x162 (4 px) |
| testing:000000180 | 432x648 | PHASE1/resnet18_ft_best | full_max_patch | ✓ | 0.8697 | 0.8697 | 0.0326 | 0.8371 | 0.0e+00 | 3.7 | 2277 | 0.0 | ızgara 108x162 (4 px) |
| testing:000000180 | 432x648 | PHASE2/dlv3p_bbox_s42 | full_image_score | ✓ | 0.4233 | 0.4235 | 0.0079 | 0.4153 | 5.6e-17 | 9.7 | 2955 | 0.006 | ızgara 108x162 (4 px) |
| testing:000000180 | 432x648 | YOLO/aiforge-full-yolo11s-img1280-conserv-s42 | yolo_box_conf | ✓ | 0.9028 | 0.903 | 0.0 | 0.9028 | 1.1e-16 | 10.6 | 2707 | 0.314 | ızgara 108x162 (4 px) |
| testing:000000180 | 432x648 | YOLO/aiforge-tiles-yolo11m-img1024-conserv-s42 | yolo_box_conf | ✓ | 0.3408 | 0.341 | 0.0 | 0.3408 | 0.0e+00 | 11.8 | 3426 | 0.094 | ızgara 108x162 (4 px) |
| testing:000000180 | 432x648 | PHASE2/unet_bbox_s42 | tile_mosaic | ✓ | 0.6047 | 0.4797 | 0.0011 | 0.661 | 4.4e-16 | 9.2 | 3020 | 0.0 | 2 pencere · kapsama tam · taban farkı 0.0008 |
| testing:000000038 | 864x1296 | PHASE2/unet_bbox_s42 | full_image_score | ✓ | 0.1175 | 0.1176 | 0.0007 | 0.1167 | 1.1e-04 | 18.6 | 3021 | 0.124 | ızgara 124x186 (7 px) |
| testing:000000038 | 864x1296 | PHASE1/unet_rgbfreq_best | full_image_score | ✓ | 0.044 | 0.044 | 0.0507 | -0.0067 | 4.9e-17 | 106.0 | 3516 | 0.002 | ızgara 124x186 (7 px) |
| testing:000000038 | 864x1296 | PHASE1/resnet18_ft_best | full_max_patch | ✓ | 0.8245 | 0.8233 | 0.0268 | 0.7977 | 3.3e-16 | 15.5 | 2308 | 0.146 | ızgara 124x186 (7 px) |
| testing:000000038 | 864x1296 | PHASE2/dlv3p_bbox_s42 | full_image_score | ✓ | 0.4167 | 0.4155 | 0.0056 | 0.4111 | 7.9e-05 | 19.9 | 2955 | 0.007 | ızgara 124x186 (7 px) |
| testing:000000038 | 864x1296 | YOLO/aiforge-full-yolo11s-img1280-conserv-s42 | yolo_box_conf | ✓ | 0.3293 | 0.329 | 0.0 | 0.3293 | 1.7e-16 | 12.3 | 2707 | 0.197 | ızgara 124x186 (7 px) |
| testing:000000038 | 864x1296 | YOLO/aiforge-tiles-yolo11m-img1024-conserv-s42 | yolo_box_conf | ✓ | 0.6602 | 0.66 | 0.0 | 0.6602 | 1.1e-15 | 22.3 | 3426 | 0.261 | ızgara 124x186 (7 px) |
| testing:000000038 | 864x1296 | PHASE2/unet_bbox_s42 | tile_mosaic | ✓ | 0.139 | 0.1176 | 0.0019 | 0.7406 | 1.1e-16 | 42.3 | 3020 | 0.147 | 8 pencere · kapsama tam · taban farkı 0.0009 |
| testing:000000273 | 928x2317 | PHASE2/unet_bbox_s42 | full_image_score | ✓ | 0.6936 | 0.6937 | 0.004 | 0.6897 | 6.9e-06 | 26.3 | 3020 | 0.202 | ızgara 72x179 (13 px) |
| testing:000000273 | 928x2317 | PHASE1/unet_rgbfreq_best | full_image_score | ✓ | 0.8192 | 0.8192 | 0.0864 | 0.7328 | 2.2e-16 | 263.5 | 3516 | 0.106 | ızgara 72x179 (13 px) |
| testing:000000273 | 928x2317 | PHASE1/resnet18_ft_best | full_max_patch | ✓ | 0.9999 | 0.9999 | 0.4905 | 0.5094 | 1.1e-16 | 24.2 | 2309 | 0.167 | ızgara 72x179 (13 px) |
| testing:000000273 | 928x2317 | PHASE2/dlv3p_bbox_s42 | full_image_score | ✓ | 0.7198 | 0.7197 | 0.0142 | 0.7055 | 2.8e-15 | 25.7 | 2956 | 0.168 | ızgara 72x179 (13 px) |
| testing:000000273 | 928x2317 | YOLO/aiforge-full-yolo11s-img1280-conserv-s42 | yolo_box_conf | ✓ | 0.8926 | 0.893 | 0.0 | 0.8926 | 5.2e-15 | 20.5 | 2369 | 0.327 | ızgara 72x179 (13 px) |
| testing:000000273 | 928x2317 | YOLO/aiforge-tiles-yolo11m-img1024-conserv-s42 | yolo_box_conf | ✓ | 0.9473 | 0.947 | 0.0 | 0.9473 | 3.3e-16 | 22.7 | 3426 | 0.326 | ızgara 72x179 (13 px) |
| testing:000000273 | 928x2317 | PHASE2/unet_bbox_s42 | tile_mosaic | ✓ | 0.2083 | 0.6937 | 0.0033 | 1.9318 | 1.1e-16 | 90.4 | 3020 | 0.337 | 18 pencere · kapsama tam · taban farkı 0.0016 |
| testing:000000038 (2× büyütülmüş) | 1728x2592 | PHASE2/unet_bbox_s42 | full_image_score | ✓ | 0.296 | 0.2962 | 0.0004 | 0.2956 | 3.3e-15 | 52.5 | 3021 | 0.194 | ızgara 124x186 (14 px) |
| testing:000000038 (2× büyütülmüş) | 1728x2592 | PHASE1/resnet18_ft_best | full_max_patch | ✓ | 0.9005 | 0.9005 | 0.0336 | 0.8669 | 1.7e-15 | 50.7 | 2309 | 0.158 | ızgara 124x186 (14 px) |
| testing:000000038 (2× büyütülmüş) | 1728x2592 | YOLO/aiforge-full-yolo11s-img1280-conserv-s42 | yolo_box_conf | ✓ | 0.4707 | 0.471 | 0.0 | 0.4707 | 1.1e-16 | 29.8 | 2707 | 0.19 | ızgara 124x186 (14 px) |
| testing:000000038 (2× büyütülmüş) | 1728x2592 | PHASE2/unet_bbox_s42 | tile_mosaic | ✓ | 0.1152 | 0.2962 | 0.0021 | 1.1203 | 1.1e-16 | 180.4 | 3020 | 0.342 | 35 pencere · kapsama tam · taban farkı 0.0017 |

Not: tepe GPU belleği uygulamanın önceden GPU'ya yüklediği tüm modelleri (34 checkpoint) içerir; açıklamanın kendi payı bunun küçük bir kısmıdır. 'uygulama skoru' tahmin tablosundaki değerdir; f(x) aynı hedefin SHAP değerlendiricisindeki değeridir (fark yalnız toplu/karışık duyarlıklı çıkarımdan kaynaklanan sayısal gürültü). Döşemeli mozaikte f(x)/E[f] pencere ortalamasıdır.

## Kernel — tam görüntü natif döşemeli aktivasyon

| görüntü | boyut | model | cihaz | döşeme | saklama ölçeği | kapsama | göreli Δ döşemeli ↔ tek geçiş (aynı cihaz) | göreli Δ GPU ↔ CPU (TF32) | süre (s) | tepe GPU (MB) | saklanan (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| testing:000000180 | 432x648 | PHASE2/unet_bbox_s42 | cuda | 1 | 1 | ✓ | 0.0e+00 | 6.5e-03 | 0.18 | 1927 | 94 |
| testing:000000180 | 432x648 | PHASE1/resnet18_ft_best | cuda | 1 | 1 | ✓ | 0.0e+00 | 2.4e-03 | 0.07 | 1914 | 94 |
| testing:000000180 | 432x648 | YOLO/aiforge-full-yolo11s-img1280-conserv-s42 | cuda | 1 | 1 | ✓ | 0.0e+00 | 1.2e-03 | 0.16 | 1890 | 86 |
| testing:000000038 | 864x1296 | PHASE2/unet_bbox_s42 | cuda | 2 | 1 | ✓ | 0.0e+00 | 5.1e-03 | 0.22 | 2091 | 376 |
| testing:000000038 | 864x1296 | PHASE1/resnet18_ft_best | cuda | 2 | 1 | ✓ | 0.0e+00 | 2.6e-03 | 0.18 | 2079 | 376 |
| testing:000000038 | 864x1296 | YOLO/aiforge-full-yolo11s-img1280-conserv-s42 | cuda | 2 | 1 | ✓ | 2.6e-04 | 2.1e-03 | 0.27 | 2015 | 346 |
| testing:000000273 | 928x2317 | PHASE2/unet_bbox_s42 | cuda | 3 | 1 | ✓ | 2.7e-02 | 2.7e-02 | 0.45 | 2125 | 722 |
| testing:000000273 | 928x2317 | PHASE1/resnet18_ft_best | cuda | 3 | 1 | ✓ | 2.3e-02 | 2.3e-02 | 0.33 | 2113 | 722 |
| testing:000000273 | 928x2317 | YOLO/aiforge-full-yolo11s-img1280-conserv-s42 | cuda | 3 | 1 | ✓ | 1.5e-04 | 1.1e-03 | 0.42 | 2039 | 665 |
| testing:000000038 (2× büyütülmüş) | 1728x2592 | PHASE2/unet_bbox_s42 | cuda | 6 | 1 | ✓ | 1.1e-02 | 1.0e-02 | 0.69 | 2169 | 752 |
| testing:000000038 (2× büyütülmüş) | 1728x2592 | PHASE1/resnet18_ft_best | cuda | 6 | 1 | ✓ | 2.8e-03 | 3.7e-03 | 0.81 | 2157 | 752 |
| testing:000000038 (2× büyütülmüş) | 1728x2592 | YOLO/aiforge-full-yolo11s-img1280-conserv-s42 | cuda | 6 | 1 | ✓ | 2.1e-03 | 1.8e-03 | 1.0 | 2075 | 692 |

Not (Kernel): GPU'da TF32 kapalıyken döşemeli ↔ tek geçiş farkı 0'dır (`checks.json`); tablodaki farklar TF32 açıkken cuDNN'in döşeme boyutuna göre farklı algoritma seçmesinden kaynaklanır. CPU'da döşemeli = tek geçiş (test T7).

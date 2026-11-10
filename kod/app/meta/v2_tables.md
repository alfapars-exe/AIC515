## AIForge-Doc v2 (GPT-Image-2) — harici test, yeniden eğitim yok

567 çiftlik birincil küme: v1 karşılığı test bölümünde, kaynak belgesi v1 eğitiminde hiç görülmemiş. v2'de yalnız düzenlenen kutu geri yapıştırılır; yeşil işaretçi kalıntısı yoktur (3066 çiftin hiçbirinde kutu dışında farklı piksel yok; renk kuralı yalnız yeşil tonlu tek bir fişte, kutunun içinde tetikleniyor). Eşikler v1 doğrulamadan, değişmeden. ΔAP: aynı spesifikasyonun v1 test görüntüsünden v2 görüntüsüne görüntü-başı AP değişimi; v1 görüntüsünde işaretçi kalıntısı olan (397) ve olmayan (170) çiftler ayrı.

| model (v1'de eğitildi)           |   box AUPRC |   piksel AUPRC |   glif AUPRC |   görüntü AUROC |   ΔAP v1 kalıntılı |   ΔAP v1 kalıntısız |
|:---------------------------------|------------:|---------------:|-------------:|----------------:|-------------------:|--------------------:|
| U-Net box (3 tohum)              |       0.52  |          0.272 |        0.209 |           0.717 |             -0.49  |              -0.169 |
| U-Net piksel (3)                 |       0.227 |          0.246 |        0.252 |           0.693 |             -0.398 |              -0.172 |
| U-Net glif (3)                   |       0.147 |          0.192 |        0.304 |           0.688 |             -0.171 |              -0.112 |
| DeepLabV3+ box                   |       0.539 |          0.28  |        0.218 |           0.724 |             -0.375 |              -0.141 |
| U-Net box, işaretçisiz (3 tohum) |       0.711 |          0.391 |        0.297 |           0.84  |             -0.188 |              -0.008 |
| U-Net piksel, işaretçisiz        |       0.353 |          0.43  |        0.416 |           0.826 |             -0.106 |              -0.078 |
| U-Net glif, işaretçisiz          |       0.214 |          0.287 |        0.446 |           0.764 |             -0.076 |              -0.065 |
| U-Net box, yalnız Gemini         |       0.449 |          0.248 |        0.175 |           0.742 |             -0.54  |              -0.189 |
| YOLO11s kare (3)                 |       0.887 |          0.354 |        0.244 |           0.919 |             -0.149 |              -0.033 |
| YOLO11s tam 1280                 |       0.697 |          0.328 |        0.215 |           0.8   |             -0.202 |              -0.083 |
| YOLO11s yalnız Gemini            |       0.843 |          0.338 |        0.229 |           0.865 |             -0.212 |              -0.055 |
| YOLO11s +authentic kare          |       0.863 |          0.345 |        0.241 |           0.937 |             -0.191 |              -0.055 |

**Okuma:** box U-Net v1'de 0.95 iken v2'de 0.52'ye düşüyor; kaybın çoğu v1'de işaretçi taşıyan spesifikasyonlarda. İşaretçisiz eğitilen box U-Net kalıntısız spesifikasyonlarda ölçülebilir kayıp göstermiyor (3 tohum: −0.008 [−0.069, +0.053]); kareli YOLO11s 0.89'da kalıyor.

**Tüm v2 (3066 çift) görüntü AUROC** — 'seen': kaynak belge v1 eğitiminde vardı; README zero-shot taban çizgileri: TruFor 0.599, DocTamper 0.585, insan 0.501.

| model                                     | subset   |   n_pairs |   auroc_top1pct |
|:------------------------------------------|:---------|----------:|----------------:|
| unet_bbox_s42                             | all      |      3066 |           0.742 |
| unet_bbox_s42                             | primary  |       567 |           0.736 |
| unet_bbox_s42                             | seen     |      2494 |           0.743 |
| unet_pixel_eps8_morph_s42                 | all      |      3066 |           0.712 |
| unet_pixel_eps8_morph_s42                 | primary  |       567 |           0.712 |
| unet_pixel_eps8_morph_s42                 | seen     |      2494 |           0.712 |
| unet_glyph_v2_s42                         | all      |      3066 |           0.703 |
| unet_glyph_v2_s42                         | primary  |       567 |           0.694 |
| unet_glyph_v2_s42                         | seen     |      2494 |           0.705 |
| aiforge-tiles-yolo11s-img1024-conserv-s42 | all      |      3066 |           0.912 |
| aiforge-tiles-yolo11s-img1024-conserv-s42 | primary  |       567 |           0.915 |
| aiforge-tiles-yolo11s-img1024-conserv-s42 | seen     |      2494 |           0.911 |
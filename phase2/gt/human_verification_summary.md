# İkinci GT denetimi — özet (değerlendirici türü doğrulanmadı)

> Değerlendiricinin türü (insan / model destekli) doğrulanmadı; bu özet doğrulanmış insan etiketlemesi sayılmaz. Dosya ve anahtar adlarındaki 'human' sözcüğü tarihsel addır.

Tam satır: 100 / 100

| karar | OK | PARTIAL | WRONG |
|---|---|---|---|
| bbox_verdict | 99 (99% [95, 100]) | 1 (1% [0, 5]) | 0 (0% [0, 4]) |
| eps8_verdict | 36 (36% [27, 46]) | 64 (64% [54, 73]) | 0 (0% [0, 4]) |
| glyph_v2_verdict | 42 (42% [33, 52]) | 54 (54% [44, 63]) | 4 (4% [2, 10]) |

| ikinci denetim–VLM | n | uyum | κ [%95 GA] |
|---|---|---|---|
| eps8_verdict | 96 | 0.47 | 0.07 [-0.08, 0.21] (ağırlıklı 0.06) |
| glyph_v2_verdict | 96 | 0.59 | 0.24 [0.07, 0.40] (ağırlıklı 0.21) |
| outside_change | 96 | 0.79 | -0.02 [-0.06, 0.00] |
| green_marker | 96 | 0.41 | 0.11 [-0.02, 0.23] |

| kaynak (eps8_verdict) | OK | PARTIAL | WRONG |
|---|---|---|---|
| cord | 17 | 9 | 0 |
| sroie | 8 | 16 | 0 |
| wildreceipt | 0 | 39 | 0 |
| xfund | 11 | 0 | 0 |

**Bildiri cümlesi taslağı:** Tabakalı 100 eğitim çiftinin ikinci denetiminde (değerlendirici türü doğrulanmamıştır) piksel-değişim GT'si 36/100 OK, 64 PARTIAL, 0 WRONG; glif GT'si 42/100 OK; resmî kutu 99/100 OK bulunmuştur. İkinci denetim ile VLM denetimi arasındaki uyum piksel-değişim GT'si için %47 (Cohen κ = 0.07 [-0.08, 0.21], n = 96) olmuştur.
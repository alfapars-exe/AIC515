# AIForge-Doc v1 — veri kürasyonu ve bias raporu (Lab Rat dataset-curation)

Toplam sahte görüntü: 4061; split: train 2762, test 812, val 487

## 1. Dağılımlar ve dengesizlik

- **tool**: gemini 3639 (89.6%), ideogram 422 (10.4%) — dengesizlik oranı max/min = 8.6
- **source_dataset**: wildreceipt 1696 (41.8%), cord 1000 (24.6%), sroie 946 (23.3%), xfund 419 (10.3%) — dengesizlik oranı max/min = 4.0
- **doc_type**: receipt 3642 (89.7%), form 419 (10.3%) — dengesizlik oranı max/min = 8.7
- **language**: en 2642 (65.1%), id 1000 (24.6%), it 81 (2.0%), de 78 (1.9%), es 67 (1.6%), pt 62 (1.5%), fr 56 (1.4%), zh 38 (0.9%), ja 37 (0.9%) — dengesizlik oranı max/min = 71.4 — <%5 sınıflar: it, de, es, pt, fr, zh, ja
- **area_bin**: <1% 2057 (50.7%), 1-5% 1989 (49.0%), 5-20% 15 (0.4%), >20% 0 (0.0%) — dengesizlik oranı max/min = 2057.0 — <%5 sınıflar: 5-20%, >20%
- **marker**: yes 3135 (77.2%), no 926 (22.8%) — dengesizlik oranı max/min = 3.4

## 2. Karıştırıcılar (Cramér's V, ki-kare)

| değişken çifti | V | p | yorum |
|---|---|---|---|
| tool × source_dataset | 0.429 | 2.0e-161 | orta |
| tool × area_bin | 0.177 | 3.1e-28 | zayıf |
| tool × marker | 0.608 | 0.0e+00 | güçlü |
| source_dataset × area_bin | 0.483 | 0.0e+00 | orta |
| marker × area_bin | 0.324 | 2.0e-93 | orta |
| tool × language | 0.391 | 1.5e-128 | orta |

**tool × source_dataset (n)**

| tool     |   cord |   sroie |   wildreceipt |   xfund |   All |
|:---------|-------:|--------:|--------------:|--------:|------:|
| gemini   |    900 |     812 |          1696 |     231 |  3639 |
| ideogram |    100 |     134 |             0 |     188 |   422 |
| All      |   1000 |     946 |          1696 |     419 |  4061 |

**tool × area_bin (satır %)**

| tool     |   <1% |   1-5% |   5-20% |
|:---------|------:|-------:|--------:|
| gemini   |  47.7 |   51.9 |     0.4 |
| ideogram |  76.5 |   23.5 |     0   |

**tool × marker (satır %)**

| tool     |   no |   yes |
|:---------|-----:|------:|
| gemini   | 14.1 |  85.9 |
| ideogram | 97.9 |   2.1 |

## 3. Split dengesi (ki-kare, train/val/test)

| değişken | χ² p | yorum |
|---|---|---|
| tool | 0.308 | dağılımlar benzer (p≥0.05) |
| source_dataset | 0.997 | dağılımlar benzer (p≥0.05) |
| area_bin | 0.422 | dağılımlar benzer (p≥0.05) |
| marker | 0.892 | dağılımlar benzer (p≥0.05) |

**split × tool (satır %)**

| p_split   |   gemini |   ideogram |
|:----------|---------:|-----------:|
| test      |     89.5 |       10.5 |
| train     |     89.3 |       10.7 |
| val       |     91.6 |        8.4 |

**split × source (satır %)**

| p_split   |   cord |   sroie |   wildreceipt |   xfund |
|:----------|-------:|--------:|--------------:|--------:|
| test      |   24.6 |    23.3 |          41.7 |    10.3 |
| train     |   24.6 |    23.1 |          42   |    10.4 |
| val       |   24.8 |    24.4 |          40.7 |    10.1 |

Ideogram örnek sayısı split başına: {'train': 296, 'test': 85, 'val': 41} — held-out üretici analizinin istatistiksel gücü sınırlı (val 41).

# İnsan doğrulama grid'i — otomatik tutarlılık özeti

Örneklem: 100 training çifti, tabaka = kaynak×tool×area_bin, seed=42. PDF: `verification_grid.pdf` (5 çift/sayfa). Form: `verification_sheet.csv` (human_verdict: OK/PARTIAL/WRONG, note).

| ölçüt | örneklem (n=100) | tüm train (n=3247) |
|---|---|---|
| n_outside_bbox_component(eps8_morph) | 3 | 150 (4.6%) |
| n_outside_px_gt_1pct(eps8) | 15 | 594 (18.3%) |
| n_any_outside_px(eps8) | 28 | 1109 (34.2%) |
| n_empty_glyph | 0 | 17 (0.5%) |
| n_empty_eps8 | 0 | 12 (0.4%) |
| n_eps8_bbox_coverage_lt_20pct | 7 | 173 (5.3%) |
| n_polarity_inverted | 8 | 193 (5.9%) |
| n_polarity_inverted_v2 | 1 | 21 (0.6%) |
| n_empty_glyph_v2 | 0 | 17 (0.5%) |

Tabaka dağılımı:

| tabaka (kaynak, tool, area_bin) | mevcut | örneklenen |
|---|---|---|
| ('cord', 'gemini-nano', '1-5%') | 26 | 2 |
| ('cord', 'gemini-nano', '<1%') | 694 | 20 |
| ('cord', 'qwen-inpaint', '1-5%') | 4 | 1 |
| ('cord', 'qwen-inpaint', '<1%') | 76 | 3 |
| ('sroie', 'gemini-nano', '1-5%') | 490 | 14 |
| ('sroie', 'gemini-nano', '<1%') | 160 | 5 |
| ('sroie', 'qwen-inpaint', '1-5%') | 76 | 3 |
| ('sroie', 'qwen-inpaint', '<1%') | 29 | 2 |
| ('wildreceipt', 'gemini-nano', '1-5%') | 1011 | 28 |
| ('wildreceipt', 'gemini-nano', '5-20%') | 13 | 1 |
| ('wildreceipt', 'gemini-nano', '<1%') | 333 | 10 |
| ('xfund', 'gemini-nano', '<1%') | 185 | 6 |
| ('xfund', 'qwen-inpaint', '<1%') | 150 | 5 |

Bayrak tanımları: `outside_bbox_component` = pixel_eps8_morph'ta bbox ile kesişmeyen ≥1 bağlı bileşen; `outside_px_gt_1pct` = eps8 pozitiflerinin >%1'i bbox dışında; `empty_glyph` = glif maskesi boş; `empty_eps8` = D>8 boş; `low_bbox_coverage_eps8` = D>8 bbox'ın <%20'sini kaplıyor; `polarity_inverted` = spec kuralı (bbox içi medyan gri <128) tetiklendi; `polarity_inverted_v2` = azınlık-sınıf kuralı tetiklendi; `empty_glyph_v2` = glyph_v2 boş.